---
name: leflur-grpo-launch
description: >-
  Launch or relaunch a LeFlur GRPO binder training run (policy trainer + its reward worker
  pools) on SLURM the correct way. Use whenever the user asks to "launch a GRPO run", "start
  training", "relaunch this run", "put this run on a full node / multi-GPU", "scale the reward
  pool", or "warm-start a new arm from a checkpoint". Encodes the launch script's env knobs,
  the pool-sizing rules, the account/qos policy, and the confirm-before-submit discipline.
---

# LeFlur GRPO launch

One submit wrapper stands up an entire run: the **policy** trainer (b200) plus the **reward
worker pool(s)** it talks to over a shared filesystem queue. Everything is driven by env vars —
there is no per-run source edit. Get the env right and the run is correct.

```bash
cd /cv/home/lisanzas/lobster/.claude/worktrees/complex_infra
CONFIG=experiment/<exp_config_name> \
OUTBASE=/cv/scratch/u/lisanzas/<run_output_dir> \
N_WORKERS=<fold pool>  M_SHAPE=<shape pool>  NRANKS=<policy GPUs> \
POLICY_HRS=168 \
bash scripts/launch_leflur_grpo.slurm
```

The script (`scripts/launch_leflur_grpo.slurm`) submits up to three jobs and prints their IDs:
a **fold** worker array (`$QUEUE`), a **shape** worker array (`$SHAPE_QUEUE`), and the **policy**
job. It strips leaked `SLURM_*` itself (line 43) and passes `--account=llm --qos=llm` on every
job. **Never** hand-roll the `sbatch` — always go through this wrapper.

## Step 1 — which pools does this run need?

Read the config's `grpo:` weights and turn on exactly the pools they require. Turning on a pool
the run never calls just wastes a10g; leaving one off that it needs → the policy blocks forever
on an empty queue.

| Pool | Env | Turn ON when the config has… | Worker |
|---|---|---|---|
| **Fold** (confidence) | `N_WORKERS` | any of `w_iptm`, `w_ptm`, `w_abag_iptm`, `w_plddt`, `w_gpde`, `w_pae_global`, `w_pae_interface`, `w_ipsae` > 0 | Protenix **or** BindCraft (see `FOLD_BACKEND`) |
| **Shape** (repack) | `M_SHAPE` | any of `w_shape`, `w_sc_clash`, `w_aar`, `w_elj_scalar`, `w_solelec_scalar`, `w_fasol_scalar`, `w_sctm_esmfold(_scalar)` > 0 | `ligandmpnn_repack_server.py` |

- **Protenix-free / esmfold-only runs set `N_WORKERS=0`** (no fold pool). The `need_conf` gate in
  the trainer skips the fold call entirely. The esmfold sc-TM term is served by the **shape** pool,
  not the fold pool.
- A config without a `shape:` block must run `M_SHAPE=0` (the wrapper only passes
  `shape.queue_dir` when `M_SHAPE>0`; overriding it on a run with no `shape:` block is a Hydra error).

## Step 2 — size the pools

**Keep `N_WORKERS` == `reward.n_shards` and `M_SHAPE` == `shape.n_shards` × (groups you want
scored concurrently).** A group of `G` designs is split into `n_shards` shards; a pool of that many
workers scores a whole group in one wave (wall-clock = slowest shard, not the sum).

- Single-GPU policy (`NRANKS=1`): `N_WORKERS` / `M_SHAPE` = `n_shards` (default 8) is enough — the
  one rank submits `accum_targets` groups roughly serially.
- **Multi-GPU policy: scale the pool ~`NRANKS`× the shards.** All ranks generate in parallel and
  submit their groups near-simultaneously, so the pool must absorb the burst. For a full node
  (`NRANKS=8`, `n_shards=8`) use **`M_SHAPE=48`** (~6 groups/wave). This is the lever that turns
  policy parallelism into real speedup — see [[grpo-multigpu-speedup-pool-scaling]] (2-rank gave
  1.70x **only** with the pool scaled ~N×; same pool → reward-bound 1.17x).

## Step 3 — policy scale (NRANKS): full node vs single GPU

`NRANKS` = b200 GPUs on **one** node (single-node DDP; `accum_targets` are sharded across ranks,
encoder grads all-reduced). **b200 node = 8 GPUs / 92 CPU / ~1992G.**

- `NRANKS=1` → plain python, single GPU (byte-identical to the historical path).
- `NRANKS=8` → **full node**, `torchrun --standalone --nnodes=1 --nproc_per_node=8`. Resources
  auto-scale (`8*NRANKS` cpu, `64*NRANKS`G). Both 8-way footguns are already fixed in
  `rl_train.py`: `map_location="cpu"` on ckpt load (line ~287, avoids the cuda:0 leak OOM,
  [[grpo-8way-load-checkpoint-cuda0-oom]]) and proper `LOCAL_RANK`/`set_device`/NCCL-timeout
  (lines ~215–256, [[grpo-8way-nccl-hang-pool-starvation]]).
- **Multi-node (`>1` node) is NOT supported** by the launch script today (`--nnodes=1` is
  hard-coded). Extending it needs a cross-node `torchrun` rendezvous (`--nnodes=K --node_rank
  --rdzv_endpoint`) driven by an sbatch `--nodes=K --ntasks-per-node=1`.

**Is scaling the policy even worth it? Diagnose first.** From the policy log, time between
consecutive `step N [...]` lines = step wall-clock; the per-target generation banners
(`StructureBackboneTransform` + schedule setup, ~14s apart) sum to the **gen** fraction; the rest
is reward-wait + backward. If gen is a large fraction → `NRANKS` helps (gen collapses ~N×). If the
run is already reward-bound → scale the **pool**, not the policy. Realistic full-node speedup on a
~50/50 gen/reward-bound run is **~1.5–2x**, not 8x (reward+backward become the floor).

## Step 4 — FOLD_BACKEND (only when the fold pool is on)

`FOLD_BACKEND=protenix` (default) runs the Protenix worker array. `FOLD_BACKEND=bindcraft` runs the
BindCraft/ColabDesign AF2-multimer array (`scripts/bindcraft_reward_server.py`, its own venv) for
the docking-aware `pae_interface` reward. Both are byte-compatible drop-ins on the same
atomic-rename queue; the policy/client is unchanged. Use `bindcraft` for a run whose only conf
metric is `w_pae_interface>0`; stay on `protenix` for `w_iptm`/`w_ptm`/… AF2 params (~5 GB) live
**only on S3** and are streamed once per node to `/local_scratch` (flock-guarded). See
[[bindcraft-af2-pae-install]].

### Worker placement — a10g default vs b200 3-node pool

The fold-worker array defaults to the dynamic **a10g** cloud pool (`WORKER_PARTITION=gpu`,
`WORKER_GRES=gpu:a10g:1`). For a large, steadier pool — BindCraft AF2 folds are heavy and the a10g
autoscaler churns mid-run (exit 120:0) — place the workers on the **static b200** nodes. 24
single-GPU array tasks pack 8-per-node onto **3 b200 nodes** (the flock staging keys on each node's
`/local_scratch`, so the ~5 GB AF2 sync runs once per node, not once per GPU):

```bash
FOLD_BACKEND=bindcraft \
WORKER_PARTITION=llm-b200 WORKER_GRES=gpu:b200:1 \
N_WORKERS=24 REWARD_NSHARDS=24 \
CONFIG=experiment/rl_leflur_binder_grpo_..._bindcraftpae \
OUTBASE=/cv/scratch/u/lisanzas/<run> \
NRANKS=8 \
bash scripts/launch_leflur_grpo.slurm     # confirm with the user before submitting
```

**`REWARD_NSHARDS` must equal `N_WORKERS`** (here 24). It sets `reward.n_shards` (config default 8),
which is how many shards one rollout group's designs are split into for parallel folding. Leave it at
8 and 16 of the 24 workers sit idle every step — the pool-scaling trap ([[grpo-multigpu-speedup-pool-scaling]]:
scaling the pool without scaling the shards gives no speedup). `NRANKS=8` additionally puts the policy
on a full b200 node, so a full b200 3-node-worker run consumes **4 b200 nodes** (1 policy + 3
workers) — check `sinfo -p llm-b200` for free nodes first. Same overrides with
`WORKER_GRES=gpu:b200:1` work for a protenix pool if its a10g array is churning.

## Step 5 — warm-start from a checkpoint

A bare GRPO checkpoint (`{state_dict, grpo_step}`) must be **materialized** into a Lightning
envelope before `load_from_checkpoint` can read it:

```bash
/cv/home/lisanzas/lobster/.venv/bin/python scripts/_materialize_grpo_ckpt.py \
  --grpo /cv/scratch/u/lisanzas/<run>/ckpts/grpo_step_<N>.ckpt \
  --base <leflur_binder_3di.ckpt> \
  --out  /cv/scratch/u/lisanzas/<run>/ckpts/grpo_step_<N>_lightning.ckpt
```

Base envelope:
`/cv/home/lisanzas/.cache/lobster/leflur/checkpoints/Sidney-Lisanza__leflur/models--Sidney-Lisanza--leflur/snapshots/17b9c68c459d742c7b36b042e0ac40cc8b06f6c7/leflur_binder_3di.ckpt`.
Then point `model.ckpt_path` at the `_lightning.ckpt` in the config (or via `EXTRA`). See
[[grpo-ckpt-materialize-for-eval]]. **Relaunching a run mid-flight?** Warm-start from that run's
**latest** `grpo_step_*_lightning.ckpt` if one exists past the original warm-start; otherwise reuse
the original.

## Step 6 — relaunching a live run (cancel → clear → resubmit)

Two policy jobs must not share an `output_dir`/queue. To rescale a running arm:

1. `scancel <policy_jid> <shape_jid>` (and the fold array if present). **Verify the IDs first** and
   **never** cancel the protected set `20065898 / 20065897 / 20078516 / 20020442 / 19648543`.
2. Clear stale queue state so new workers don't claim dead jobs — **use literal paths, one `find`
   per command** (the worktree sandbox refuses a `find` whose path is a shell variable):
   `find /cv/scratch/.../shape_queue/new -type f -delete` then `.../claimed -type f -delete`.
3. Resubmit with the new `NRANKS`/`M_SHAPE`.

Relaunching costs only the current partial step (checkpoints land every `ckpt_every`, default 25),
so rescaling early is cheap.

## Step 7 — verify after launch (do not walk away on "submitted")

1. `squeue -j <policy_jid> -o "%.12i %.8T %.6D %b %R"` → confirm `NODES=1`, `gres/gpu:b200:<NRANKS>`.
   A full-node request may sit `PENDING (Resources)` until 8 GPUs free on one node (check queue
   pressure with `squeue -h -p llm-b200 -t PENDING | wc -l`).
2. When it starts, the log must show `Distributed GRPO: rank 0/<NRANKS>` (world size correct) then
   `Starting GRPO: … accum_targets=…`.
3. **Arm a `Monitor`** on `/cv/scratch/u/lisanzas/slurm_logs/leflur_grpo/policy_<jid>.out` filtering
   for `step [0-9]+ \[` (throughput) **and** the failure signatures `Traceback|CUDA out of
   memory|NCCL|watchdog|RuntimeError` — so a crashloop is never mistaken for silence. Compare the
   new step cadence to the pre-scale baseline to confirm the speedup landed.

## Standing constraints (always)

- **Confirm before submitting** any SLURM job — get explicit user go-ahead on config, scale, and
  warm-start point.
- GPU jobs → `--account=llm --qos=llm` (**NOT** ai4dd — its a10g GrpTRES is chronically saturated);
  CPU jobs → `--account=llm` on `defq` (**NOT** the default `basic`). The wrapper already does this.
- Work in the `complex_infra` worktree; blessed venv `/cv/home/lisanzas/lobster/.venv/bin/python`
  (BindCraft venv `/cv/scratch/u/lisanzas/bindcraft-venv`).
- Keep BindCraft/AF2 weights on S3, off shared disk.
- Never cancel `20065898 / 20065897 / 20078516 / 20020442 / 19648543` ([[never-cancel-hard-chord-run]]).
- Watch `/cv/scratch` quota — a `0:53` no-logs death = scratch full ([[scratch-quota-0-53-death]]).

## Full env-knob reference (`scripts/launch_leflur_grpo.slurm`)

| Env | Default | Meaning |
|---|---|---|
| `CONFIG` | `experiment/rl_leflur_binder_grpo_overfit` | Hydra experiment config name |
| `OUTBASE` | `.../rl_leflur_binder_grpo_overfit` | run output dir; `QUEUE`/`SHAPE_QUEUE` derive from it |
| `NRANKS` | 1 | policy b200 GPUs on one node (single-node DDP) |
| `N_WORKERS` | 8 | fold (Protenix/BindCraft) worker array size; 0 = none |
| `REWARD_NSHARDS` | "" (config value, 8) | fold-pool fan-out (`reward.n_shards`); **set == `N_WORKERS`** on a big pool (b200 3-node → 24) or extra workers idle |
| `M_SHAPE` | 0 | shape (LigandMPNN-repack) worker array size; 0 = none |
| `FOLD_BACKEND` | `protenix` | `protenix` \| `bindcraft` — which oracle drains `$QUEUE` |
| `POLICY_HRS` | 12 | policy walltime hours (use 168 for a real multi-day run) |
| `MAX_IDLE_S` | 3600 | worker idle-exit seconds (must exceed inter-step gap) |
| `REWARD_SEEDS` | "" (seed 101) | comma-sep Protenix seeds averaged per design |
| `WORKER_ACCOUNT`/`WORKER_QOS` | `llm`/`llm` | fold pool account/qos |
| `SHAPE_ACCOUNT`/`SHAPE_QOS` | `llm`/`llm` | shape pool account/qos |
| `WORKER_PARTITION`/`WORKER_GRES` | `gpu`/`gpu:a10g:1` | fold pool placement (→ `llm-b200`/`gpu:b200:1` when a10g autoscaler churns) |
| `SHAPE_PARTITION`/`SHAPE_GRES` | `gpu`/`gpu:a10g:1` | shape pool placement |
| `BINDCRAFT_VENV`/`_RECYCLES`/`_TARGET_MANIFEST`/`_STAGE_ROOT` | see script | BindCraft-only knobs |
| `DISTILL_DIR` | `.../grpo_distill_complexes` | passing predicted complexes saved here ("" disables) |
| `EXTRA` | "" | extra hydra overrides appended to the policy command |

## Worked example — full-node relaunch of the esmfold-zscore arm

The run started single-GPU; rescaled to a full node for throughput:
```bash
CONFIG=experiment/rl_leflur_binder_grpo_elj_sparse_clash_chainbreak_lc_jaccard_rog_solelec_esmfoldsctm_zscore \
OUTBASE=/cv/scratch/u/lisanzas/rl_leflur_binder_grpo_elj_sparse_clash_chainbreak_lc_jaccard_rog_solelec_esmfoldsctm_zscore \
N_WORKERS=0 M_SHAPE=48 NRANKS=8 POLICY_HRS=168 \
bash scripts/launch_leflur_grpo.slurm
```
`N_WORKERS=0` (esmfold sc-TM rides the **shape** pool, no Protenix); `M_SHAPE=48` for the 8-rank
burst; `NRANKS=8` full node. Diagnosis had shown ~50/50 gen/reward split at 1 GPU (~4.5 min/step),
so a full node was expected to ~halve step time.
