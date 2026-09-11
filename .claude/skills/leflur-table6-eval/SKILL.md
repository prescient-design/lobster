---
name: leflur-table6-eval
description: >-
  Generate binder designs (cold-A8 sampler, N/target x 38 Complexa targets) from a GRPO
  checkpoint and compute the full 10-column Table-6 metric panel on them. Use whenever the
  user asks to "run the table 6 metrics", "table 6 eval", "generate designs + table 6
  metrics", or "eval the latest checkpoint" for a LeFlur/GRPO binder run.
---

# LeFlur Table-6 eval

End-to-end reproduction of `tab:seqablation` (Table 6) for **any** GRPO binder checkpoint:
cold-A8 generation followed by all 14 Table-6 metrics (10 columns + LC-AA/LC-3Di/ddG/ipSAE). One command submits the whole SLURM
DAG with `afterok` dependencies, so it completes without the session alive.

## Invocation

```
/leflur-table6-eval <run-name-or-outbase> [--ndes 10] [--ckpt latest] [--no-protenix]
```

`<run-name-or-outbase>` is either a friendly name from the **run registry** below or a full
run `OUTBASE` (the dir containing `ckpts/grpo_step_*.ckpt`).

Run it with the driver (from the `complex_infra` worktree):

```bash
bash scripts/_run_table6_eval.sh --outbase <OUTBASE> --ndes 10 --tag <TAG> --protenix
```

- `--ckpt latest` (default): picks the numerically highest `grpo_step_*.ckpt` in `<OUTBASE>/ckpts`.
  Pass `--ckpt grpo_step_75.ckpt` to pin one.
- `--ndes 10` (default): designs per target (10 x 38 = 380).
- `--tag <MODEL>`: output label; defaults to `<outbase-basename minus rl_leflur_binder_grpo_>_step<N>`.
- `--protenix` (default) / `--no-protenix`: include or skip the Protenix fold pass (pass%).
- `--gpu a10g` (default) `| b200`: GPU tier for the two GPU jobs (gen + Protenix). `b200` overrides
  the sbatch line with `--partition=llm-b200 --gres=gpu:b200:1` (account/qos stay `llm`). **Gen is
  not the bottleneck** (~24 it/s, ~8 s/design on a10g, whole array done in minutes); the wall-clock
  long poles are the **Protenix** fold and the serial **pack→SC** CPU chain — so `b200` mainly helps
  Protenix. Default to `a10g` (plentiful, uncontended); reach for `b200` only when it's idle and
  Protenix latency matters.

## Run registry (friendly name -> OUTBASE)

| Friendly name | OUTBASE | Default tag |
|---|---|---|
| six-term (elj+chainbreak+chord+lc+jaccard+contactband, G64) | `/cv/scratch/u/lisanzas/rl_leflur_binder_grpo_elj_chainbreak_chord_lc_jaccard_contactband` | `sixterm_step<N>` |
| elj-dense (per-token e_lj; A/B dense arm, G64, 8-GPU) | `/cv/scratch/u/lisanzas/rl_leflur_binder_grpo_elj_dense_clash_chainbreak_lc_jaccard` | `eljdense_step<N>` |
| elj-sparse (per-design scalar e_lj; A/B sparse arm, G64, 8-GPU) | `/cv/scratch/u/lisanzas/rl_leflur_binder_grpo_elj_sparse_clash_chainbreak_lc_jaccard` | `eljsparse_step<N>` |
| rog (eight-term: six-term + Rg + 3Di-LC, G64) | `/cv/scratch/u/lisanzas/rl_leflur_binder_grpo_elj_chainbreak_chord_lc_jaccard_contactband_rog_lc3di` | `rog_step<N>` |
| rog+fa_sol (nine-term: rog + fa_sol_per_res companion w=0.5, G64) | `/cv/scratch/u/lisanzas/rl_leflur_binder_grpo_elj_chainbreak_chord_lc_jaccard_contactband_rog_lc3di_fasol` | `rogfasol_step<N>` |

Add new runs here as they are evaluated. Base ckpt for materialization (leflur_binder_3di
envelope):
`/cv/home/lisanzas/.cache/lobster/leflur/checkpoints/Sidney-Lisanza__leflur/models--Sidney-Lisanza--leflur/snapshots/17b9c68c459d742c7b36b042e0ac40cc8b06f6c7/leflur_binder_3di.ckpt`

## The 18 metrics (the 10 Table-6 columns + 8 diagnostics)

| # | Column | Producer | Direction |
|---|--------|----------|-----------|
| 1 | pass% (Protenix pTM>0.80 & ipTM>0.70) | `score_complexa_minibinders_home.sh` | ↑ |
| 2 | AAR (ProteinMPNN) | `aar_compute.slurm` (`GEN_ROOT`) | ↑ |
| 3 | AAR iface | `aar_compute.slurm` | ↑ |
| 4 | C_mpnn | `aar_compute.slurm` | ↑ |
| 5 | H (bits, binder AA-comp Shannon entropy) | inline in `_summ_table6.py` | ↑ |
| 6 | 3Di TV (whole-binder vs Complexa ref) | `_grpo_dist_metrics.py --gen-root` | ↓ |
| 7 | E_clash | `_grpo_bench_geom.py --gen-root` | ↓ |
| 8 | iface f (native band 0.14-0.16) | `_grpo_bench_geom.py --gen-root` | →nat |
| 9 | SC (all-atom LigandMPNN-packed 3DZD) | `pack_gen_home.sh` -> `packed_sc_gen_home.sh` (two-stage, see below) | ↑ |
| 10 | break% (≥1 backbone C-N bond >2Å) | inline in `_summ_table6.py` | ↓ |
| 11 | LC-AA (linguistic complexity of binder AA seq) | `_grpo_lc_metrics.py --gen-root` | ↑ |
| 12 | LC-3Di (linguistic complexity of binder 3Di string) | `_grpo_lc_metrics.py --gen-root` | ↑ |
| 13 | ddG (tmol interface ΔΔG, relaxed, 10%-trimmed) | `_tmol_ddg_compute.py --relax` (needs packed complexes) | ↓ |
| 14 | ipSAE (interface pTM-from-PAE, d0res, max A↔B) | `_grpo_ipsae_metrics.py --prot-root` (needs PAE-dumped Protenix) | ↑ |
| 15 | e_lj (all-atom LigandMPNN-packed interface Lennard-Jones) | `packed_sc_gen_home.sh` (same `sc_shard_*.csv` as SC) | ↓ |
| 16 | TMbind (gen-vs-Protenix-refold binder-monomer self-consistency TM) | `_grpo_sctm_metrics.py --gen-root --prot-root` (needs Protenix) | ↑ |
| 17 | TMcplx (gen-vs-Protenix-refold whole-complex self-consistency TM, docking-aware) | `_grpo_sctm_metrics.py --gen-root --prot-root` (needs Protenix) | ↑ |
| 18 | pae_interaction (mean cross-chain interface PAE, Å; AF2-multimer/BindCraft) | `_grpo_ipsae_metrics.py --prot-root` (same PAE-dumped Protenix job as ipSAE) | ↓ |

`_summ_table6.py` reads all of the above and prints/writes the single Table-6 row.
**Base A8 reference row** (sanity anchor): `pass 5.61 / AAR 0.255 / AARif 0.222 / Cmpnn 0.108 /
H 3.61 / 3DiTV 0.437 / Eclash 24.7 / ifacef 0.227 / SC 0.906 / e_lj <pending base-A8 pack> /
break 28.1 / LC-AA 0.519 / LC-3Di 0.267 / ddG -37.9 / ipSAE <pending PAE re-fold>` (LC + ddG
measured on the full 3800-design base A8 pool; ipSAE requires re-folding with the per-token PAE
dump on, see metric 14 below; e_lj needs the base-A8 pool side-chain-packed — no base pack pool
exists yet, so it is `TBD` in the ref string).

### Metric 15 (e_lj) — how it is computed

**e_lj** is the all-atom interface Lennard-Jones energy (`e_lj_atr + e_lj_rep`, kcal/mol) computed by
`scripts/_packed_potentials_compute.py --sc --no-sasa` — the **same run and the same `sc_shard_*.csv`
files** that produce SC (metric 9). It is already a column in every SC shard, so **no extra job, pack,
or driver change is needed**: `_summ_table6.py` macro-averages the `e_lj` column from
`<fulleval>/sc/sc_shard_*.csv` alongside `sc_allatom`. Target-balanced mean; **lower (more negative)
is better** — a favorable, well-packed interface has attractive net LJ, while a clashing/eroded
interface is repulsive (positive). Complements SC (shape) with an energetic view of the same packed
interface.

### Metrics 11-14 + 18 (LC-AA / LC-3Di / ddG / ipSAE / pae_interaction) — how they are computed

- **LC-AA / LC-3Di** (`scripts/_grpo_lc_metrics.py --gen-root <gen> --tag <TAG> --out <json>`, single
  defq CPU, `afterok:gen`): within-sequence anti-degeneracy score `LC = prod_{k=1..3} coverage(k)` in
  `[0,1]` (the trainer's own `plm_design_rl.rewards.linguistic_complexity` — the `w_seq_complexity` /
  `w_struct_complexity` reward engine). LC-AA scores the binder AA 1-letter string; LC-3Di derives the
  binder 3Di token string from the backbone via `derive_3di_tokens` then scores it. Higher = less
  degenerate; **no external reference** (each seq vs itself). Healthy ~0.52 (LC-AA), collapse < 0.35.
  Emits `per_arm[<tag>]["overall"]["lc_aa"|"lc_3di"] = {mean,sem,n}` (target-balanced macro).
- **ddG** (`scripts/_tmol_ddg_compute.py --gen-root <gen> --out <csv> --relax --nproc 30 --threads 1`,
  single defq CPU 32-core, **`afterok:pack`** — it scores the LigandMPNN-**packed** `*_complex_packed.pdb`,
  so it must chain off the pack job, not gen): Rosetta `beta_nov2016` interface ΔΔG. `_summ_table6.py`
  globs the CSV (`err==""`), sums the six cross-chain terms `ddg_{atr,rep,sol,lkball,elec,hb}` to get
  `ddg_iface`, and reports the **10%-trimmed pooled mean** (clash-tail robust). Lower (more negative) =
  better; Complexa reference ≈ −39. The blessed venv carries tmol (0.1.48+cu129torch2.8) — no separate
  interpreter needed.
- **ipSAE** (`scripts/_grpo_ipsae_metrics.py --prot-root <protenix_out> --tag <TAG> --out <json>`,
  single defq CPU, **`afterok:<Protenix score>`** — it reads the fold's per-token PAE, so it chains off
  the Protenix job): the Dunbrack-lab **ipSAE** interface confidence, the `d0res` headline variant
  (github.com/DunbrackLab/IPSAE, MIT). For every cross-chain residue pair with `PAE < 10 Å`, apply the
  pTM transform `1/(1+(pae/d0)^2)` with a per-aligned-residue `d0` set from that residue's count of
  good-PAE partners (`calc_d0_array`: `L=max(26,n); d0=max(1, 1.24·(L-15)^⅓−1.8)`); average over each
  aligned residue's valid partners, take the max over aligned residues, and report the **max of the
  A→B and B→A directions** (chain A = antigen, B = binder). Higher = more confident interface (0..1).
  **Requires the Protenix fold to have dumped per-token PAE** — the scorer must run with env
  `PROTENIX_NEED_PAE=1` (→ `protenix pred --need_atom_confidence true`), which writes each design's
  `*_full_data_sample_0.json` carrying `token_pair_pae` + `token_asym_id` (no CIF/coords needed). The
  driver sets `PROTENIX_NEED_PAE=1` on the Protenix job by default. `--prot-root` is the Protenix
  output base (`$HOME/binder_viz/<TAG>/protenix`); the producer globs
  `<prot-root>/<tag>/task_*/**/*_full_data_sample_0.json`. Emits
  `per_arm[<tag>]["overall"]["ipsae"] = {mean,sem,n,ntgt}` (target-balanced macro).
- **pae_interaction** (metric 18) is emitted by the **same** `_grpo_ipsae_metrics.py` job — **no extra
  job, driver, or DAG change**: the producer reads the identical `*_full_data_sample_0.json`
  (`token_pair_pae` + `token_asym_id`) and additionally computes the **AF2-multimer / BindCraft
  `pae_interaction`**: the mean cross-chain interface PAE
  `0.5·(mean(PAE[A_rows,B_cols]) + mean(PAE[B_rows,A_cols]))` over **all** antigen↔binder token pairs
  (no PAE/distance cutoff — unlike ipSAE's thresholded, d0res-weighted pTM transform). Chain A =
  antigen, B = binder (from `token_asym_id` first-appearance order). **Lower (Å) = a more confident
  interface.** Emits `per_arm[<tag>]["overall"]["pae_interaction"] = {mean,sem,n,ntgt}` (target-balanced
  macro); `_summ_table6.py` reads it into the `paeI` column. Same `PROTENIX_NEED_PAE=1` precondition as
  ipSAE (both are read off the per-token PAE dump).

### Metrics 16-17 (TMbind / TMcplx) — how they are computed

- **TMbind / TMcplx** (`scripts/_grpo_sctm_metrics.py --gen-root <gen> --prot-root <protenix_out>
  --tag <TAG> --out <json>`, single defq CPU, **`afterok:<Protenix score>`** — it pairs each generated
  PDB with its Protenix-refolded structure, so it chains off the Protenix job): the
  **generated-vs-Protenix-predicted self-consistency TM-scores** (length-tolerant `tmtools` TM-align).
  **TMbind** aligns the generated **binder monomer** to its Protenix refold (a clean *fold*
  self-consistency, docking-blind — does the binder fold to the shape it was generated as). **TMcplx**
  aligns the generated **whole antigen+binder complex** to the Protenix-refolded complex, antigen-
  anchored, so it also measures whether the binder re-docks to the generated pose (**docking-aware**).
  These are the exact same two signals the two-oracle TM reward arm trains on (ESMFold binder TM /
  BindCraft complex TM), measured here offline via Protenix for eval comparability. Both higher = more
  self-consistent (0..1). `--gen-root` is the gen dir (`$HOME/binder_viz/<TAG>/gen`) and `--prot-root`
  the Protenix output base (`$HOME/binder_viz/<TAG>/protenix`). Emits
  `per_arm[<tag>]["overall"]["tm_binder"|"tm_complex"] = {mean,sem,n,ntgt}` (target-balanced macro).
  `_summ_table6.py` reads these via `--sctm-json` and appends them as the last two Table-6 columns.

### SC (metric 9) — how it is really computed

SC is **all-atom**, so the gens (backbone + CB only) must be side-chain-packed first. This is a
**two-stage packed-potentials pipeline**, chained `afterok`:

1. **Pack** — `slurm/scripts/pack_gen_home.sh` (defq, array `0-39%40`, `NSHARDS=40`, env `GEN=$GENDIR`)
   runs `scripts/_pack_gen_sidechains.py --gen_root "$GEN" --shard --nshards`. For every
   `*_complex.pdb` it splits antigen (`chains[0]`) + binder (`B`/last), repacks the **whole complex**
   with the LigandMPNN side-chain packer (`Repacker.pack_complex`, `num_denoising_steps=3`,
   `num_samples=8` — the reward's own config), and writes a full-atom `*_complex_packed.pdb` next to
   each design. Shards are disjoint index slices; the **whole pool must be packed** before scoring.
2. **Score SC** — `slurm/scripts/packed_sc_gen_home.sh` (defq, array `0-39%40`, env `GEN=$GENDIR`,
   `OUTDIR=$FULLEVAL/sc`, `--dependency=afterok:<pack>`) runs
   `scripts/_packed_potentials_compute.py --gen-root "$GEN" --sc --no-sasa --shard --nshards --out
   "$OUTDIR/sc_shard_<task>.csv"`. It reads the `*_complex_packed.pdb`, computes **only** the
   all-atom 3DZD interface shape-complementarity (`sc_allatom`), and **skips SASA** (`--no-sasa`,
   which is the slow term). One CSV shard per array task.

`_summ_table6.py` consolidates the SC column by globbing `<fulleval>/sc/sc_shard_*.csv` and averaging
the `sc_allatom` column.

**Do NOT use `scripts/_grpo_sidechain_sc.py` for this.** That script computes the same all-atom SC
quantity inline (repack + score in one pass) and *does* have a `--gen-root` flag, but it is **not
wired into this pipeline** — the driver uses the two-stage packed-potentials path above (pack once,
reuse the packed PDBs, score SC-only). `_grpo_sidechain_sc.py` is kept only for one-off/back-compat
scoring.

## Sampler (cold A8 — matches Table 6's own gens)

`nsteps=200, T_seq=0.273, T_struc=0.316, stochasticity 20/60, use_esmfold=false`, native
length from the manifest, plus the schedule/penalty EXTRA (baked into the driver as the
default, same string as `slurm/scripts/eval_chord_s150_aar.sh`):
`inference_schedule_seq=Power^2, struc=Linear, tri=Log, sequence_diversity_penalty=2,
tri_diversity_penalty=8`. **Keep diversity penalty = 2** (used in both Table 1 and Tables 3–6;
only the sampler differs). Do NOT set it to 0.

Manifest (38 targets):
`/cv/scratch/u/lisanzas/denovo_dataset/binder/denovo/complexa_bench/targets/complexa_gen_targets.csv`.

## Outputs

- Designs: `/cv/home/lisanzas/binder_viz/<TAG>/gen/<target_id>/*_{binder,complex}.pdb`
- Protenix + filtered: `/cv/home/lisanzas/binder_viz/<TAG>/{protenix,filtered}`
- Metric shards + JSONs: `/cv/scratch/u/lisanzas/fulleval_<TAG>/{aar,sc,ddg}`,
  `scripts/_grpo_dist_metrics_<TAG>.json`, `scripts/_grpo_bench_geom_<TAG>.json`,
  `scripts/_grpo_lc_metrics_<TAG>.json` (LC), `fulleval_<TAG>/ddg/<TAG>_ddg_relaxed.csv` (ddG),
  `scripts/_grpo_ipsae_metrics_<TAG>.json` (ipSAE **and** pae_interaction),
  `scripts/_grpo_sctm_metrics_<TAG>.json` (TMbind/TMcplx)
- **Final Table-6 row**: `/cv/scratch/u/lisanzas/fulleval_<TAG>/table6_row_<TAG>.txt`

To report the result live, arm a `Monitor` on the consolidation log
(`/cv/scratch/u/lisanzas/slurm_logs/table6/summ_<jobid>.out`) or the row file, then read the
row and compare each column to the base-A8 reference.

## Results ledger — "what were our results for the <arm> eval"

**Do not dig through `fulleval_*` dirs by hand.** Every completed Table-6 eval is one command:

```bash
/cv/home/lisanzas/lobster/.venv/bin/python scripts/_table6_ledger.py            # all arms, ranked by pass%
/cv/home/lisanzas/lobster/.venv/bin/python scripts/_table6_ledger.py --grep solelec   # filter by tag substring
```

It scans `/cv/scratch/u/lisanzas/fulleval_*/table6_row_*.txt`, prints each arm's final TABLE-6 row
aligned and sorted by pass% next to the base-A8 reference, and appends three **interface-energy
columns** — `e_lj`, `e_elec` (Rosetta REF2015 `fa_elec`), `fa_sol` (LK desolvation) — recomputed live
from that arm's own packed side-chain shards `fulleval_<TAG>/sc/sc_shard_*.csv` as target-macro means
(per-target mean, then averaged across targets, matching `_summ_table6.py`'s `e_lj`). It never goes
stale (it reflects whatever rows/shards exist on disk right now). This is the standing answer whenever
the user asks for a past eval — run it, don't reconstruct numbers from memory.

> **`fa_elec` / `fa_sol` here mean the actual interface ENERGIES (kcal/mol), not any training arm.**
> They are the `e_elec` and `fa_sol` columns above — the physical REF2015 interface terms measured on
> the packed complex, reported next to `e_lj`. A `-` means that arm's sc shards predate the fa_sol/elec
> compute. (e_lj appears twice: the row's value and the shard-recomputed target-macro; they match when
> the row used the same averaging, and can differ for older arms whose row used a different shard set.)

**Curated snapshot (as of 2026-09-04; run the script for the live list + all columns).** Interface
energies are target-macro means, kcal/mol. **The bar is base-A8 pass% = 5.61; nothing beats it yet.**

| arm (step) | pass% | e_lj | e_elec (fa_elec) | fa_sol | ipSAE | note |
|---|---|---|---|---|---|---|
| **base A8 (ref)** | **5.61** | — | — | — | — | the bar |
| eightterm s75 | 5.26 | 128.7 | −5.99 | — | — | ties base early, then erodes |
| sixterm s275 | 5.26 | 22.1 | −11.99 | — | — | best AAR (0.332); erodes by s450 |
| elj_sparse s175 | 4.74 | 16.1 | −8.44 | — | 0.068 | |
| elj_sparse_rog s300 | 3.95 | 9.0 | −10.62 | — | 0.054 | best Rg among arms |
| **solelec s300** | 1.84 | 15.04 | **−2.46** | **50.00** | 0.053 | fa_sol+fa_elec reward; combined dilutes |
| **solelec s525** | 1.58 | 13.66 | **−1.26** | **42.80** | 0.045 | later step erodes further |
| **faelec s750** | 1.84 | 22.87 | **−1.37** | **56.28** | 0.044 | fa_elec reward arm |
| **rog_fasol s800** | 1.84 | 1.13 | −1.30 | 33.56 | 0.056 | fa_sol reward arm — hacks e_lj, pass collapses |
| faelec s1100 | 1.32 | 15.80 | −2.08 | 51.86 | 0.047 | |
| fasol standalone s1450 | 0.79 | −8.01 | −0.68 | 26.19 | 0.034 | |

**Reading it.** The solvation/electrostatics reward family (fa_sol, fa_elec, solelec) all sit at
≤1.84% with ipSAE ≈ 0 — they optimize their surrogate while dockability decays. Note the interface
energies confirm it: `e_elec` stays small (−1 to −2.5 kcal/mol — near the noise floor that killed the
term offline) and `fa_sol` rises (worse desolvation) as those arms disengage the interface. Matches
the offline gates ([[fasol-offline-gate-negative]] 0.594, [[faelec-byte-faithful-reimpl]] 0.502,
[[grpo-solelec-combined-reward]] combined dilutes). This is why the next reward is the docking-aware
BindCraft/AF2 `pae_interface` (see the `leflur-grpo-launch` skill's FOLD_BACKEND path).

## Standing constraints

- **Confirm before submitting SLURM**: get explicit user go-ahead before running the driver
  (it fires GPU + CPU array jobs).
- GPU jobs → `--account=llm --qos=llm` (NOT ai4dd); CPU jobs → `--account=llm` on `defq`
  (NOT the default `basic`). The driver already does this and strips leaked `SLURM_*` env.
- Blessed venv: `/cv/home/lisanzas/lobster/.venv/bin/python`.
- Work in the `complex_infra` worktree
  (`/cv/home/lisanzas/lobster/.claude/worktrees/complex_infra`).
- Never cancel jobs `20065898 / 20065897 / 20078516 / 20020442 / 19648543`. This eval reads a
  checkpoint copy and does not interact with any live training run.
- `binder_viz` lives under `$HOME` (233G quota) — 380 designs + folds are small, but watch it.

## How the driver wires the DAG

`materialize(CPU)` → `gen(GPU a10g array, 38)` → all `afterok:gen`:
- `AAR array` (`aar_compute.slurm`, defq `0-31%32`)
- `pack array` (`pack_gen_home.sh`, defq `0-39%40`) → **`afterok:pack`** → `packed-SC array`
  (`packed_sc_gen_home.sh`, defq `0-39%40`) — SC is the only two-stage metric
- `3Di-TV` (single defq CPU, `--wrap`)
- `geom` (single defq CPU, `--wrap`)
- `LC` (`_grpo_lc_metrics.py`, single defq CPU, `--wrap`) — LC-AA + LC-3Di
- `ddG` (`_tmol_ddg_compute.py --relax`, single defq CPU 32-core) — chained **`afterok:pack`** (not
  gen), since it scores the packed `*_complex_packed.pdb`; the second `afterok:pack` child alongside SC
- `[Protenix array]` (GPU a10g→llm) if `--protenix` — launched with `PROTENIX_NEED_PAE=1` so the fold
  dumps per-token PAE → **`afterok:<Protenix>`** → two children:
  - `ipSAE` (`_grpo_ipsae_metrics.py`, single defq CPU, `--wrap`), reading
    `<TAG>/protenix/**/*_full_data_sample_0.json`
  - `[sctm]` (`_grpo_sctm_metrics.py`, single defq CPU, `--wrap`) — TMbind + TMcplx (metrics 16-17),
    reading both `--gen-root <TAG>/gen` and `--prot-root <TAG>/protenix`, writing
    `scripts/_grpo_sctm_metrics_<TAG>.json`

→ `consolidate` (`_summ_table6.py`, with `--sctm-json`) `afterok:<AAR:SC:DIST:GEOM:LC:DDG[:SCORE:IPSAE:SCTM]>`
→ writes `table6_row_<TAG>.txt`. Note the consolidation depends on the **SC** and **ddG** jobs, not the
pack job (pack is an intermediate; SC and ddG are its `afterok` children); TMbind/TMcplx are only
present when `--protenix` is on (they need the Protenix refold).

The `--gen-root`/`--tag` (and `GEN_ROOT` env for `aar_compute.slurm`) overrides let every
metric script target an arbitrary gen dir by path, bypassing the hardcoded per-arm registries,
so no per-run source edits are needed. These overrides are additive and byte-identical to
registry mode (verified: geom E_clash/iface_frac match exactly on a shared pool).
