# Plan: BindCraft AF2-multimer `pae_interaction` GRPO reward (weights on S3)

## Goal

Add an **AF2-multimer interface-PAE** reward to the LeFlur binder GRPO loop, using **BindCraft's
ColabDesign AF2 scoring** to produce `pae_interaction` (mean cross-chain PAE, Å; lower = better).
This is the field-standard in-silico binder discriminator (BindCraft / RFdiffusion filter) and,
unlike the ESMFold sc-TM arm, it is **docking-aware** (folds the *complex*, not the binder monomer).

Big model weights (AF2 multimer params, ~5 GB) live **on S3**, not on the shared cluster disks
(`/cv/home` 23 G free, `/cv/scratch` 93 % used) — each worker syncs them to **node-local
`/local_scratch`** (495 G free, ephemeral) at startup. This is the explicit user constraint: *"keep
it on s3 so we don't run out of space here as we do with the other folding models."*

User decisions (locked): (1) **BindCraft/ColabDesign AF2** backend, not the leaner af2_initial_guess.
(2) **Benchmark 1 vs 3 recycles** during install, then pick.

## Speed context (why this is worth it)

Per-fold latency on a10g (measured unless noted):
- ESMFold monomer (~100 res binder): **~1.8 s** — but docking-blind (sc-TM gated NEGATIVE twice).
- **AF2-multimer complex** (BindCraft, model_1_v3, single-seq binder): **~10–25 s (1 recycle) est.**
  — fastest *docking-aware* option; ~1.5–3× faster than Protenix-v2. Confirm by benchmark.
- Protenix-v2 complex (`cycle=10, step=200`, MSA cached, 1 seed): **~25–35 s median**, 16–81 s.

## The reward side is already wired (no trainer/client change)

The Protenix reward client is metric-agnostic. `grpo.w_pae_interface` is plumbed end-to-end
(`_leflur_grpo_trainer.py:206` → `conf_weights` at :2088 → `reward_from_confidence`) and oriented
`1 − pae/32` (`_protenix_reward.py`, `_PAE_CEIL=32.0`), defaulting to 0. BindCraft's
`pae_interaction` **is** the mean cross-chain PAE, i.e. the same `pae_interface` quantity. So a
BindCraft worker only needs to write `done/<job>.json` with keys `{pae_interface, iptm, ptm}`
matching the existing contract; then set `w_pae_interface > 0` in a config. The queue protocol,
LRU cache, sharding, timeout-flooring, and wandb logging are all reused.

## S3 layout (write-confirmed: `s3://prescient-data-dev/sandbox/lisanzas/`)

```
s3://prescient-data-dev/sandbox/lisanzas/bindcraft/
  params/            # AF2 multimer params (params_model_*_multimer_v3.npz, ~5 GB) — THE big item
  repo_snapshot/     # optional: pinned BindCraft + ColabDesign tarball for reproducible workers
```
Venv + repo checkout go on `/cv/scratch` (like the protenix venv); only the **params** must be
S3-resident and streamed, since they are the space hog that recurs across worker nodes.

## Phases

### Phase 1 — install (login node; NO SLURM) — ✅ DONE (09-04)
1. ✅ BindCraft cloned → `/cv/scratch/u/lisanzas/denovo_dataset/repos/BindCraft`.
2. ✅ uv venv `/cv/scratch/u/lisanzas/bindcraft-venv` (py3.10): `jax[cuda12]==0.6.0` + jaxlib cuda12,
   ColabDesign `1.1.3` (`--no-deps`, from sokrypton github), numpy<2, scipy, pandas, biopython, chex,
   dm-haiku, flax<0.10, dm-tree, ml-collections, immutabledict, optax, joblib, matplotlib. NO
   PyRosetta / pdbfixer / py3dmol — scoring path (`i_pae`) does not need them (PyRosetta is only for
   BindCraft's relax/filter). Imports verified on CPU.
3. ✅ AF2 params (`alphafold_params_2022-12-06.tar`, 5.59 GB → 15 npz) downloaded once, `aws s3 sync`
   to `s3://prescient-data-dev/sandbox/lisanzas/bindcraft/params/` (15 npz incl. the 5
   `params_model_*_multimer_v3.npz`), tar + extracted copies deleted from shared disk. **Weights live
   ONLY on S3.**
4. ✅ `scripts/stage_af2_params.sh` — `aws s3 sync s3://.../bindcraft/params "$STAGE/af2_params/params"`
   (idempotent; prints the ColabDesign `data_dir`). Validated: S3 → `/local_scratch` pulls all 15 npz.
5. ✅ `scripts/_bindcraft_af2_score.py` — shared scorer `BindcraftAF2Scorer` (resident
   `mk_afdesign_model(protocol="binder", use_multimer=True)`; `score()` does
   `prep_inputs(pdb_filename, chain="A", binder_len)` [target template only — honest de-novo dock] +
   `predict(seq, num_recycles)` → `{pae_interface=i_pae, iptm=i_ptm, ptm, plddt, latency_s}`) +
   `binder_seq_from_pdb`. Gate complexes = chain A antigen / chain B binder. CPU import + seq
   extraction verified; the fold itself needs a GPU (deferred to Phase 1b).

**Scoring recipe (confirmed from BindCraft `bindcraft.py:203-209`, `colabdesign_utils.py:257-267`):**
`prediction_model.predict(seq=binder_seq, num_recycles=N)` → `aux["log"]` = `{plddt, ptm, i_ptm, pae,
i_pae}`. `i_pae` IS the mean cross-chain PAE = the reward's `pae_interface`. `num_recycles` = the 1-vs-3
knob.

### Phase 1b — benchmark 1 vs 3 recycles — ✅ DONE (09-04, job 20837955) — GATE PASSED

**Result** (350 folds, stratified Complexa gate pool, a10g):

| recycles | n | lat_med | lat_p90 | pae AUROC | iptm AUROC | pae PASS/FAIL med |
|---|---|---|---|---|---|---|
| 1 | 175 | 39.7 s | 42.4 s | **0.806** | 0.800 | 0.656 / 0.854 |
| 3 | 175 | 43.9 s | 48.7 s | **0.836** | 0.833 | 0.622 / 0.855 |

- **GATE PASS**: `pae_interface` AUROC 0.806 (1rec) / **0.836 (3rec)** both clear e_lj 0.720 and beat
  dsasa 0.757 / n_hb 0.742 — the **first docking-aware fold reward to pass** (ESMFold sc-TM was
  docking-blind). PASS designs fold to a tighter interface (pae 0.62) than FAIL (0.85).
- **Recycle choice: 3** — +4 s median (+10%) over 1rec buys +0.03 AUROC and a sharper PASS/FAIL gap.
  Latency ~44 s/fold ≈ Protenix (25–35 s), acceptable.
- **⚠️ UNITS (Phase 2 must fix):** ColabDesign `i_pae` is **normalized to [0,1]** (÷31 internally),
  NOT raw Å — CSV values 0.15–0.98. The reward contract orients `1 − pae/32` (`_PAE_CEIL=32`, Å), so
  the Phase 2 worker MUST emit `pae_interface = i_pae × 31.0` (convert to Å); otherwise
  `1 − 0.98/32 ≈ 0.97` for every design → flat/dead reward. Gate AUROC is scale-invariant so unaffected.
- Result CSV: `/cv/scratch/u/lisanzas/packed_potentials/bindcraft_bench.csv`;
  log: `/cv/scratch/u/lisanzas/slurm_logs/bindcraft/bench_20837955.out`.

### Phase 1b — benchmark 1 vs 3 recycles (GPU SLURM — CONFIRM before firing) — ✅ SCRIPTS READY
- `scripts/_bindcraft_bench.py` — scores a stratified subset of the Complexa gate pool
  (`complexa_fasol_gate_labeled.csv`, 3439 designs, 28.7% pass) at recycles {1,3}; writes per-design
  CSV; prints median/p90 **latency/fold**, `pae_interface` **AUROC vs pass** (oriented, lower=better),
  and AF2-`iptm` AUROC. Reference bar: e_lj 0.720, dsasa 0.757, n_hb 0.742; fa_sol 0.594 (neg).
- `scripts/_bindcraft_bench.slurm` — a10g (`--partition gpu --gres=gpu:a10g:1`), `--account=llm
  --qos=llm`, stages params from S3 to `/local_scratch/$SLURM_JOB_ID`, runs the bench. `N` (default
  200) + `OUT` env overrides.
- **Launch (needs user go-ahead — GPU SLURM):**
  `for v in $(env|grep -o '^SLURM[^=]*'); do unset "$v"; done;
   N=200 OUT=/cv/scratch/u/lisanzas/packed_potentials/bindcraft_bench.csv
   sbatch scripts/_bindcraft_bench.slurm`
- Decide recycle count from latency-vs-signal; a gate PASS (AUROC comfortably > 0.720) earns the full
  reward run, per fa_sol/e_elec discipline.

### Phase 2 — reward worker (mirror `scripts/protenix_reward_server.py`) — ✅ DONE (09-04)
- `scripts/bindcraft_reward_server.py` written + lint/format clean; imports in the blessed venv,
  manifest loader resolves all 38 templates. Queue-driven (`new/claimed/done/failed`, atomic-rename
  `claim_one` identical to the protenix server so the existing client drains it unchanged), resident
  AF2 model (`BindcraftAF2Scorer` loads params ONCE from staged `--data_dir`), per job: fold every
  design's binder seq against the target template → `done/<job>.json` with
  `{ptm, iptm, abag_iptm(=iptm), plddt, pae_interface, i_pae_raw, latency_s}` per design.
- **Antigen template resolution (no client change):** the queue job carries only `antigen_seq`, not a
  structure. BindCraft folds against the antigen *structure*, so the worker maps `job["target_id"]` →
  `(pdb_path, target_chain)` via `--target_manifest` (default `complexa_gen_targets.csv`, the same 38
  targets w/ `pdb_path`/`target_chain` cols the eval uses). The worker supplies the structural context
  the queue doesn't — client/trainer stay byte-identical.
- **✅ UNITS FIX applied:** emits `pae_interface = i_pae × 31.0` (`_IPAE_TO_ANGSTROM`), converting the
  ColabDesign normalized [0,1] `i_pae` back to Å for the `1 − pae/32` contract; raw kept as `i_pae_raw`.
- **Recycles = 3** (`DEFAULT_RECYCLES`, Phase 1b decision), `--num_recycles` overridable. Designs
  folded shortest-first to minimize ColabDesign retraces (scorer caches `(pdb, chain, len)`).
- Per-design errors floor to `None` (mirrors protenix); per-job errors → `failed/`. No distillation
  (kept lean). Runs standalone in the BindCraft venv (stdlib + `_bindcraft_af2_score`; colabdesign lazy).

### Phase 3 — config arm + launch
- New `experiment/rl_leflur_binder_grpo_..._bindcraftpae.yaml` with `grpo.w_pae_interface: 1.0`
  (± companions), `group_size 64`, Protenix off.
- Extend `scripts/launch_leflur_grpo.slurm` with a `FOLD_BACKEND=bindcraft` switch that submits the
  BindCraft worker array (its venv) instead of the protenix array, pointing at the same `QUEUE`.
- GPU jobs `--account=llm --qos=llm`; strip leaked `SLURM_*`.

## Verification
1. `stage_af2_params.sh` pulls params to `/local_scratch` and BindCraft AF2 loads from there (nothing
   large written to `/cv/home` or `/cv/scratch`).
2. Benchmark prints per-fold latency + `pae_interaction` AUROC vs pass (offline gate).
3. Worker parity: a fold's `pae_interface` from the worker matches a standalone BindCraft score on the
   same complex.
4. Short GRPO smoke run: reward drains from the bindcraft queue, `w_pae_interface` term logged in
   wandb, no shared-disk growth during the run.

## Out of scope
- No change to the Protenix or ESMFold rewards; this is an additive third fold backend.
- No BindCraft *hallucination/design* loop — we use only its AF2 **scoring** of a given complex.
- Offline-gate result governs whether this earns a full run (same discipline as fa_sol/e_elec),
  though `pae_interaction` is expected to pass where those failed.
