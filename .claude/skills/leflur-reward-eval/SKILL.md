# LeFlur reward eval (offline gate + per-arm reward check)

Two disciplined checks for GRPO reward terms, both built on **all-atom interface potentials** computed
on side-chain-packed complexes. Use them to answer the two questions that keep coming up:

- **(A) Offline gate — *before* wiring:** is a candidate potential term actually predictive of the
  pass label? Score it on the **Complexa design-model's own generations** (a labeled pool that passes
  at **>20%**, ~28.8%) and report AUROC vs pass + orthogonality to the incumbent `e_lj`. Only terms
  that survive the gate earn a GRPO run.
- **(B) Per-arm reward check — *after* training:** when an arm adds a new reward term, **measure that
  exact term on the arm's own generations** next to a reference arm. This separates "the gradient
  never flowed" from "the gradient flowed but the reward was hackable" — the two look identical if you
  only watch pass%.

This is the standing discipline: **compute a term → offline-gate it (A) → only then wire it as a
reward → after the run, per-arm reward check (B).** See the ledger of gate verdicts at the bottom.

## Why this dataset

The gate/scoring dataset is the **Complexa model's own generations**, not LeFlur's. Complexa passes at
~28.8% ([[complexa-benchmark-3way]]) vs LeFlur's ~5%, so its pool has enough positives (987 pass /
2452 fail = 28.7%) for a stable AUROC and a meaningful pass-rate-by-quartile trend. A term that can't
separate pass from fail *even on high-quality designs* has no business being a reward.

Gate pool (3439 packed designs, 35 targets, 28.7% pass):
```
/cv/scratch/u/lisanzas/denovo_dataset/binder/denovo/complexa_bench/minibinder/complexa_complexabench/gen
```
Layout `<root>/<target_id>/<design_id>_complex_packed.pdb` (e.g. `01_PD1/01_PD1_design049_complex_packed.pdb`).
Pass labels (design_id → passes): `/cv/scratch/u/lisanzas/packed_potentials/complexa_fasol_gate_labeled.csv`.

## Mode A — offline gate

Two stages: **compute** the potentials on the labeled pool, then **analyze** predictiveness.

```bash
# 1. compute all fast interface potentials + join pass labels (no SASA => ~240x faster; add --sc for
#    all-atom 3DZD shape-complementarity, or drop --no-sasa for dsasa/n_buns)
/cv/home/lisanzas/lobster/.venv/bin/python scripts/_packed_potentials_compute.py \
  --gen-root /cv/scratch/u/lisanzas/denovo_dataset/binder/denovo/complexa_bench/minibinder/complexa_complexabench/gen \
  --labels  /cv/scratch/u/lisanzas/packed_potentials/complexa_fasol_gate_labeled.csv \
  --no-sasa --nproc 32 \
  --out /cv/scratch/u/lisanzas/packed_potentials/<GATE>.csv

# 2. analyze: AUROC vs pass (auto-oriented), standardized logit β controlling for interface size +
#    binder_len, and pass-rate by quartile. iptm AUROC printed as the yardstick.
/cv/home/lisanzas/lobster/.venv/bin/python scripts/_packed_potentials_analyze.py \
  --csv /cv/scratch/u/lisanzas/packed_potentials/<GATE>.csv
```

Convenience wrapper for the standard byte-faithful recompute on the canonical pool:
`bash scripts/_run_faelec_gate.sh` → writes `.../packed_potentials/complexa_gate_faelec.csv`.

**Reading the result.** A term earns a run only if it clears the reference bar AND adds signal beyond
interface size:
- **AUROC** must beat the field. Reference AUROCs on this pool: `e_lj` / `e_lj_atr` **0.720**,
  `dsasa` 0.757 / `n_hb` 0.742 (full-SASA compute), and the negatives `fa_sol` **0.594**,
  MaSIF, `C_mpnn` **0.292** (anti-predictive). ~0.5 = noise.
- **β(·|len,ifsz)** — the standardized logit coefficient controlling for `n_iface_res` + `binder_len`.
  A term whose AUROC is just "bigger interface → more passes" collapses here. Correct physical sign +
  non-trivial magnitude = real orthogonal signal.
- **quartile pass%** should trend monotonically in the oriented direction.
- **Orthogonality to `e_lj`** is the real test — many terms are redundant with LJ burial. A pooled
  lift that vanishes under GroupKFold-by-target (within-target AUROC) is target-confounding, not
  signal (the trap that killed fa_sol and MaSIF).

**iptm AUROC ~1.0 is expected and is NOT the yardstick to beat** — pass is *defined* from the Protenix
fold (pTM/ipTM), so iptm is nearly perfectly separable by construction. It's printed only as a sanity
check that the labels loaded.

## Mode B — per-arm reward check

After a reward-arm eval (via [[table6-eval-skill]]) packs the arm's 380 designs, measure the new
reward term on those packed gens and compare to a reference arm (same sampler, without the term).

```bash
# 1. compute potentials on each arm's OWN packed gens (from the table6 eval's gen dir)
/cv/home/lisanzas/lobster/.venv/bin/python scripts/_packed_potentials_compute.py \
  --gen-root /cv/home/lisanzas/binder_viz/<ARM_TAG>/gen --no-sasa --nproc 16 \
  --out /cv/scratch/u/lisanzas/packed_potentials/arm_<ARM_TAG>.csv
# ...repeat for the reference arm...

# 2. side-by-side mean +/- SEM of the reward term (+ companions)
/cv/home/lisanzas/lobster/.venv/bin/python scripts/_arm_reward_report.py \
  --label <ARM>=arm_<ARM_TAG>.csv --label <REF>=arm_<REF_TAG>.csv \
  [--cols fa_sol e_lj n_hb n_saltbridge e_elec ...]
```
(`--cols` defaults to `fa_sol fa_sol_per_atom e_lj e_lj_atr e_lj_rep e_elec n_hb n_saltbridge n_clash`;
bare filenames resolve under `/cv/scratch/u/lisanzas/packed_potentials/`.)

**Reading the result.** Look at the reward term first: if it moved decisively in the intended
direction (many SEM), **the gradient worked** — any pass% loss is a reward-QUALITY failure, not a
plumbing bug. Then look at the companions to see *how* it was optimized: a reward hacked by interface
retraction shows the term improving while `n_hb` / `n_saltbridge` / `e_lj_atr` all degrade together
(the design shrinks its interface instead of improving it).

Worked example — the **fa_sol arm** (`elj_sparse_rog_fasol_step300`) vs the rog arm
(`elj_sparse_rog_step300`): fa_sol **49.4 → 26.2** (~9 SEM — the reward was optimized) but n_hb
8.9→4.6, n_saltbridge 1.41→0.25, e_lj_atr −65.8→−48.7 — it minimized desolvation by *disengaging the
interface*, so pass% collapsed 3.95→1.32. Gradient fine; reward hackable. Matches the gate
(fa_sol AUROC 0.594). See [[fasol-offline-gate-negative]].

## Scripts

| Script | Role |
|---|---|
| `scripts/_packed_potentials_compute.py` | compute all-atom interface potentials on `*_complex_packed.pdb`; `--labels` joins pass; `--no-sasa` (fast), `--sc` (3DZD SC), `--shard/--nshards` |
| `scripts/_packed_potentials_analyze.py` | Mode A: AUROC vs pass + logit β(·|len,ifsz) + quartile pass%, ranked |
| `scripts/_arm_reward_report.py` | Mode B: mean±SEM of reward term(s) across arms |
| `scripts/_run_faelec_gate.sh` | wrapper: byte-faithful e_elec recompute on the canonical gate pool |

Packed PDBs are produced upstream by `scripts/_pack_gen_sidechains.py` (LigandMPNN whole-complex
repack) — the same packing the [[table6-eval-skill]] SC stage runs, so a table6-eval'd arm already has
its `*_complex_packed.pdb` under `binder_viz/<TAG>/gen/`.

## Standing constraints

- Both modes are **CPU-only, no SLURM** — the fast (`--no-sasa`) compute on 3439 designs runs locally
  in a few minutes (`--nproc 32`). Shard onto `defq --account=llm` only for the SASA-heavy path.
- Blessed venv: `/cv/home/lisanzas/lobster/.venv/bin/python`.
- Work in the `complex_infra` worktree (`/cv/home/lisanzas/lobster/.claude/worktrees/complex_infra`).
- A new potential term must be **byte-validated vs compiled tmol** (like `_fasol`/`_faelec`) before it
  is gated — the gate assumes the term is physically correct, not just plausible.

## Gate verdict ledger (this pool)

| Term | AUROC | verdict |
|---|---|---|
| e_lj / e_lj_atr | 0.720 | incumbent primary reward |
| dsasa | 0.757 | positive (full-SASA) |
| n_hb | 0.742 / 0.598* | positive |
| fa_sol | 0.594 | NEGATIVE — non-orthogonal to e_lj; arm hacked it ([[fasol-offline-gate-negative]]) |
| e_elec (byte-faithful) | 0.502 | NEGATIVE — chance; below raw-Coulomb diag 0.549; DDD screening does NOT rescue it; dielectric is distance-only (no burial term) ([[faelec-byte-faithful-reimpl]]) |
| MaSIF surface-comp | — | NEGATIVE ([[masif-offline-gate-negative]]) |
| C_mpnn (seq↔struct) | 0.292 | NEGATIVE, anti-predictive ([[mpnn-consistency-reward-gate-negative]]) |
| ESMFold sc-TM (binder monomer) | 0.431† | NEGATIVE, anti-predictive — pass mean sc_tm 0.809 < fail 0.862; within-tgt 0.487 (15/32 wrong-dir); lift +0.004; NOT degeneracy-driven (r=−0.138); monomer fold is docking-blind ([[esmfold-selfconsist-gate-negative]]) |

\*AUROC depends on whether the compute included SASA-dependent normalizers; compare within one CSV.
†intended-direction AUROC (higher sc_tm→pass); analyzer prints 0.569 with dir `-` (flipped).
