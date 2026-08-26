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
cold-A8 generation followed by all 10 Table-6 metrics. One command submits the whole SLURM
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

Add new runs here as they are evaluated. Base ckpt for materialization (leflur_binder_3di
envelope):
`/cv/home/lisanzas/.cache/lobster/leflur/checkpoints/Sidney-Lisanza__leflur/models--Sidney-Lisanza--leflur/snapshots/17b9c68c459d742c7b36b042e0ac40cc8b06f6c7/leflur_binder_3di.ckpt`

## The 10 metrics (== Table 6)

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

`_summ_table6.py` reads all of the above and prints/writes the single Table-6 row.
**Base A8 reference row** (sanity anchor): `5.61 0.255 0.222 0.108 3.61 0.437 24.7 0.227 0.906 28.1`.

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
- Metric shards + JSONs: `/cv/scratch/u/lisanzas/fulleval_<TAG>/{aar,sc}`,
  `scripts/_grpo_dist_metrics_<TAG>.json`, `scripts/_grpo_bench_geom_<TAG>.json`
- **Final Table-6 row**: `/cv/scratch/u/lisanzas/fulleval_<TAG>/table6_row_<TAG>.txt`

To report the result live, arm a `Monitor` on the consolidation log
(`/cv/scratch/u/lisanzas/slurm_logs/table6/summ_<jobid>.out`) or the row file, then read the
row and compare each column to the base-A8 reference.

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
- `[Protenix array]` (GPU a10g→llm) if `--protenix`

→ `consolidate` (`_summ_table6.py`) `afterok:<AAR:SC:DIST:GEOM[:SCORE]>` → writes
`table6_row_<TAG>.txt`. Note the consolidation depends on the **SC** job, not the pack job (pack is
an intermediate; SC is its `afterok` child).

The `--gen-root`/`--tag` (and `GEN_ROOT` env for `aar_compute.slurm`) overrides let every
metric script target an arbitrary gen dir by path, bypassing the hardcoded per-arm registries,
so no per-run source edits are needed. These overrides are additive and byte-identical to
registry mode (verified: geom E_clash/iface_frac match exactly on a shared pool).
