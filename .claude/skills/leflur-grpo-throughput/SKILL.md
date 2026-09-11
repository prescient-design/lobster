# LeFlur GRPO throughput

One command turns a live GRPO run's policy log + `squeue` + queue dirs into a **throughput
snapshot**: per-step wall-clock, worker-pool census, per-pool backlog, and a **bottleneck verdict
with the lever that would help**. Use it to decide — as training runs — whether to add fold
workers, add shape workers, lighten the oracle, or scale the policy, instead of guessing.

## Invocation

```bash
cd /cv/home/lisanzas/lobster/.claude/worktrees/complex_infra
/cv/home/lisanzas/lobster/.venv/bin/python scripts/_grpo_throughput.py \
  --job <policy_jobid> \
  --outbase <run OUTBASE>          # enables per-pool backlog (the key bottleneck signal)
```

- `--job <id>` derives the log at `/cv/scratch/u/lisanzas/slurm_logs/leflur_grpo/policy_<id>.out`;
  or pass `--log <path>` directly.
- `--outbase <dir>` unlocks the **backlog** read (`<OUTBASE>/queue/new` = fold, `shape_queue/new` =
  shape). Without it you still get cadence + pool census, but no backlog → weaker verdict.
- `--last N` (default 10): how many recent step intervals to summarize. Right after adding/removing
  workers, use `--last 4` to read the *post-change* cadence (a stale slow step inflates the mean).
- `--csv <path>`: append one timestamped row per call — build a throughput history across the run
  (pair with `CronCreate` or repeated calls; see **Tracking over the run**).
- `--shape-floor <s>` (default 92): the shape-bound s/step reference (repack+ESMFold wave + gen +
  backward). The verdict compares cadence to this.

## What it reports

```
=== GRPO throughput :: policy_20978255.out (step 1354, last update 59s ago) ===
  params: group_size=64  accum_targets=10  ->  640 designs/step
  s/step (last 12): mean 137s  median 119s  [min 105 / max 277]
  trend: 149s -> 124s (improving)
  recent steps: 1349:154s  1350:140s  1351:108s  1352:122s  1353:105s  1354:117s
  throughput: 26.3 steps/hr  ~16,848 designs/hr
  pools: policy=1  fold=10 run (+0 pend)  shape=24 run (+0 pend)
  backlog (new/): fold=0  shape=8
  VERDICT: REWARD-LATENCY BOUND (s/step 137 >> shape floor 92, backlog ~0)
  LEVER:   ... fewer AF2 recycles / lighter fold oracle, or bigger accum_targets ...
```

- **designs/step** = `group_size × accum_targets` (the burst every step folds+repacks; NRANKS only
  parallelizes it, it does not change the count).
- **s/step** mean/median over the window + the raw recent deltas so you see the actual trajectory.
  `last update … ago` much larger than s/step ⇒ the run may be wedged (check the log for a
  Traceback / NCCL watchdog).
- **pools**: RUNNING (+PENDING) counts of `leflur_grpo_worker` (fold), `leflur_grpo_shape` (shape),
  `leflur_grpo_policy`.
- **backlog (new/)**: unclaimed jobs waiting in each pool's queue. This is the live starvation
  signal. `claimed/` is NOT a backlog — it accumulates historically (hundreds of thousands of files
  over a multi-day run) and is ignored.

## The verdict → lever map

The classifier reads cadence + per-pool `new/` backlog (the log carries no timing breadcrumbs and
the gen banners smear across the whole interval, so a gen/reward split is *not* recoverable — these
two signals are what's trustworthy):

| Verdict | Trigger | Lever |
|---|---|---|
| **FOLD-BOUND** | fold `new/` backlog > accum_targets | add fold workers (cooperative array on the same `$QUEUE`, see below) or lighten the fold (fewer AF2 recycles). Adding workers is the direct fix. |
| **SHAPE-BOUND** | shape `new/` backlog > accum_targets | raise `M_SHAPE` (more repack+ESMFold a10g workers); keep `shape.n_shards` matched. |
| **REWARD-LATENCY BOUND** | s/step ≫ floor **but** backlogs ~0 | pools drain each wave, so the *slowest single fold/repack* sets the floor — **adding workers won't help**. Cut per-fold latency (recycles / lighter oracle) or raise `accum_targets` to amortize the wave. Scale `NRANKS` only if generation is the heavy part. |
| **NEAR-FLOOR / AT-FLOOR** | s/step ≈ floor | shape-floor limited and healthy; marginal gains only. |

**Key distinction:** a standing backlog means a pool is *too small* → add workers. No backlog but
slow → the per-wave oracle *latency* is the floor → more workers do nothing; make each fold cheaper
or fold more per wave. This is the decision the tool exists to make.

## Adding workers cooperatively (fold OR shape — no relaunch)

**Both** pools claim jobs by identical atomic `os.rename` on their shared queue, so a **new array
pointing at the same queue just cooperates** — no policy change, no relaunch, for either pool. The
`claim_one()` sort-key `stat` is guarded ([[grpo-shape-pool-claimone-race]]) and `worker_id` only
NAMES the claimed file (`{job}.{worker}.json`), it does NOT partition work, so worker-id collisions
between arrays are harmless. (The earlier "shape is a fixed array, grow only via relaunch" note was
WRONG — verified 09-08 against `ligandmpnn_repack_server.py` `claim_one`; shape adds cooperatively
exactly like fold.)

Write an sbatch script byte-identical to the launch wrapper's block for that pool and submit it.
**Confirm with the user before submitting** (standing constraint).

- **Fold** (when FOLD-BOUND) — `/cv/scratch/u/lisanzas/slurm_logs/leflur_grpo/add_fold_workers.sh`:
  `--partition=llm-b200 --account=llm --qos=llm --gres=gpu:b200:1 --job-name=leflur_grpo_worker`,
  wrap = flock-staged `stage_af2_params.sh` then `bindcraft_reward_server.py --queue_dir
  <OUTBASE>/queue --worker_id $SLURM_ARRAY_TASK_ID --num_recycles 3 --max_idle_s 3600`.
- **Shape** (when SHAPE-BOUND) — `/cv/scratch/u/lisanzas/slurm_logs/leflur_grpo/add_shape_workers.sh`:
  `--partition=gpu --account=llm --qos=llm --gres=gpu:a10g:1 --job-name=leflur_grpo_shape`, wrap =
  `ligandmpnn_repack_server.py --queue_dir <OUTBASE>/shape_queue --worker_id $SLURM_ARRAY_TASK_ID
  --device cuda --max_idle_s 3600`. (Used 09-08 on the two-oracle run: +24 array 21106866 → shape
  backlog went to 0, floor then flipped to fold.)

To *rescale* a pool (shrink, or change GPU tier) rather than just add, use the relaunch recipe in
[[leflur-grpo-launch]] (Step 6). Note the useful ceiling per pool = shard-jobs/step (fold =
`reward.n_shards`×accum_targets; shape = `shape.n_shards`×accum_targets); adding past that leaves
workers idle.

## Tracking over the run

Two ways to build a history so you can see throughput drift and correlate it with worker
changes / checkpoints:

- **On demand**: rerun the command whenever you want a fresh snapshot (cheap, read-only).
- **Logged**: add `--csv /cv/scratch/u/lisanzas/slurm_logs/leflur_grpo/thru_<jobid>.csv`; each call
  appends `wall_time, step, mean_s_step, median_s_step, steps_hr, designs_hr, fold_run, fold_pend,
  shape_run, shape_pend, fold_backlog, shape_backlog, verdict`. Drive it periodically with
  `CronCreate` (e.g. every ~20 min) for an unattended throughput trace, or with a `Monitor` if you
  want to be pinged when cadence crosses a threshold.

## Standing constraints

- Read-only + `squeue` only; no SLURM submission, safe to run anytime.
- Blessed venv `/cv/home/lisanzas/lobster/.venv/bin/python`; work in the `complex_infra` worktree.
- Any worker add/rescale it recommends is a SLURM submission → **confirm with the user first**; GPU
  jobs `--account=llm --qos=llm`.
- Never cancel `20065898 / 20065897 / 20078516 / 20020442 / 19648543`.

## Live run registry

| Run | Policy job | OUTBASE |
|---|---|---|
| two-oracle TM (esmfold binder-TM + BindCraft complex-TM, z-scored) | `20978255` | `/cv/scratch/u/lisanzas/rl_leflur_binder_grpo_elj_sparse_clash_chainbreak_lc_jaccard_rog_solelec_esmfoldbinder_bindcraftcomplex_tm_zscore` |
