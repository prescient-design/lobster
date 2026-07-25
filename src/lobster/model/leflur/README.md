# LeFlur

LeFlur is a discrete-flow-matching model for protein and protein-ligand design.
The core checkpoints support five inference modes; two additional
complex-trained checkpoints add de-novo binder design:

| Mode | Input | Output | Checkpoint |
|---|---|---|---|
| **Unconditional generation** | length(s) | novel sequences + structures | `leflur-base`, `leflur-ted` |
| **Forward folding** | sequence (or PDB to extract sequence) | predicted structure | `leflur-ted` |
| **Inverse folding** | PDB / CIF | designed sequence | `leflur-ted` |
| **Ligand-conditioned generation** | ligand (atoms + bonds) | binding protein | `leflur-pl` |
| **Ligand-conditioned forward/inverse folding** | protein + ligand | structure or sequence with pocket awareness | `leflur-pl` |
| **De-novo binder design** | target PDB + epitope residues | binder sequence + structure | `leflur-binder-3di`, `leflur-binder-disto` |

The three core checkpoints (`leflur-base`, `leflur-ted`, `leflur-pl`;
~17 GiB total) live on HuggingFace at
[`Sidney-Lisanza/leflur`](https://huggingface.co/Sidney-Lisanza/leflur) and
download automatically on first use. The two binder checkpoints
(`leflur-binder-3di`, `leflur-binder-disto`; ~5.7 GiB each) live in the same
repo — see [**De-novo binder design**](#7-de-novo-binder-design--complexa)
below and [`docs/leflur/binder_design.md`](../../../../docs/leflur/binder_design.md).

## Quickstart

```bash
# 1. Install (CPU)
uv sync --extra mgm --extra struct-cpu

# 2. Authenticate with HuggingFace (one-time, public repo so any token works)
export HF_TOKEN=hf_xxx

# 3. Generate 10 unconditional proteins (downloads leflur-ted on first run, ~6 GiB)
uv run lobster_generate --config-name experiment/generate_unconditional \
    paths=public generation.num_samples=10 output_dir=./out/uncond
```

Results land under `./out/uncond/`, with one CSV of per-sample metrics
(TM-score, percent identity, structural diversity) plus decoded PDB files.

## Documentation

End-user docs live under [`docs/leflur/`](../../../../docs/leflur/):

- **[`installation.md`](../../../../docs/leflur/installation.md)** — Python
  extras, environment variables, HuggingFace auth, optional Foldseek
  binary for diversity metrics.
- **[`quickstart.md`](../../../../docs/leflur/quickstart.md)** — five-minute
  walkthroughs for each of the five inference modes, using the bundled
  test PDBs and ligand files.
- **[`binder_design.md`](../../../../docs/leflur/binder_design.md)** —
  de-novo binder design end-to-end: the two binder checkpoints, running one
  target + the full Complexa 38-target benchmark, and Protenix scoring.
- **[`checkpoints.md`](../../../../docs/leflur/checkpoints.md)** — the three
  canonical checkpoints, how to list / inspect / fetch them, and how the
  paired Latent Generator codecs are pulled in automatically.
- **[`benchmarks.md`](../../../../docs/leflur/benchmarks.md)** — the four
  canonical evaluation benchmarks (CAMEO 2022, MultiFlow test, PoseBusters
  paired and no-overlap), how to fetch them from HuggingFace via
  `lobster_leflur_benchmarks`, and how each one maps to a publication
  table.
- **[`cli.md`](../../../../docs/leflur/cli.md)** — full CLI reference for
  the four entry points: `lobster_generate`, `lobster_autoencode`,
  `lobster_leflur_checkpoints`, and `lobster_leflur_benchmarks`.

## How the pieces fit together

```
                            ┌──────────────────────────────┐
                            │  HuggingFace                 │
                            │  Sidney-Lisanza/leflur       │
                            │    leflur-base               │
                            │    leflur-ted                │
                            │    leflur-pl                 │
                            └──────────────┬───────────────┘
                                           │  resolve_checkpoint()
                                           ▼
   ┌────────────────────────────────────────────────────────────────────┐
   │  lobster.model.leflur                                              │
   │    LeFlurSequenceStructureEncoderLightningModule  (protein-only)   │
   │    LeFlurProteinLigandLightningModule              (protein+ligand)│
   └─────────────────────────────┬──────────────────────────────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        ▼                        ▼                        ▼
  lobster_generate         lobster_autoencode    lobster_leflur_checkpoints
  (5 inference modes)      (encode/decode PDBs)  (list / fetch / inspect)
```

Path resolution and checkpoint locations are controlled through Hydra's
`paths={public,internal}` overlay. External users keep the default
(`paths=public`); internal collaborators reading checkpoints from a shared
filesystem override with `paths=internal`. See
[`docs/leflur/installation.md`](../../../../docs/leflur/installation.md)
for the full list of environment variables.

## Benchmarks

The numbers below are reproduced from the LeFlur paper. Each subsection lists
the canonical Hydra config + the single command needed to regenerate the
LeFlur rows. Best-of-N rows (`N30 NLL`, `N30 SR8/SR9`, `N30 oracle`) all build
on the same base generate command — see
[**Best-of-N ranking with pseudo-NLL**](#best-of-n-ranking-with-pseudo-nll)
below for the ranking step.

> All commands assume `paths=public` (HuggingFace checkpoints + HuggingFace
> benchmark inputs). The seed matches the canonical Hydra config; override
> with `seed=<int>` to regenerate with different randomness.

**Pre-fetch the inputs.** Each table consumes one benchmark dataset
mirrored to the dataset side of
[`Sidney-Lisanza/leflur`](https://huggingface.co/datasets/Sidney-Lisanza/leflur).
Pull the four canonical benchmarks once with the dedicated CLI (anonymous
download — no HF token required):

```bash
lobster_leflur_benchmarks fetch cameo                                # tables 1, 3
lobster_leflur_benchmarks fetch multiflow_test                       # tables 1, 3 (MultiFlow rows)
lobster_leflur_benchmarks fetch posebusters_benchmark_no_overlap     # tables 2, 4
```

Files land at `${LOBSTER_CACHE}/benchmarks/<short-name>/`, which is what
`${paths.benchmarks.<name>}` interpolates to under `paths=public` — the
generate commands below work unchanged after the fetch. See
[`docs/leflur/benchmarks.md`](../../../../docs/leflur/benchmarks.md) for
the per-benchmark schema, citations, and licenses.

### 1. Inverse folding — CAMEO 2022

| Model | Tokens | AAR (%) | TM | RMSD (Å) | Pass (%) | pLDDT |
|---|---:|---:|---:|---:|---:|---:|
| ProteinMPNN | — | 42.93 | 0.85 | 4.18 | **55.9** | **0.75** |
| DPLM-2 650M | 8192 | **49.22** | 0.81 | 4.84 | 41.7 | 0.72 |
| LeFlur-P 470M SLQ | 256 | 34.00 | 0.76 | 4.60 | 37.0 | 0.66 |
| LeFlur-P **N30 NLL** 470M SLQ | 256 | 34.90 | 0.80 | 4.73 | 39.4 | 0.70 |
| LeFlur-P N30 SR8 470M SLQ | 256 | 35.10 | 0.81 | 4.49 | 39.4 | 0.70 |
| LeFlur-P N30 SR9 470M SLQ | 256 | 34.68 | 0.81 | 4.44 | 40.2 | 0.70 |
| LeFlur-P N30 oracle 470M SLQ | 256 | 35.11 | **0.86** | **3.18** | 52.0 | 0.71 |

**N1 base.** Reproduces the `LeFlur-P 470M SLQ` row:

```bash
lobster_generate --config-name experiment/generate_inverse_folding \
    paths=public output_dir=./out/inverse_folding_cameo
```

**N30 NLL.** Pseudo-NLL best-of-30 — see
[Best-of-N ranking](#best-of-n-ranking-with-pseudo-nll) for the second step.

### 2. Inverse folding with ligand — PoseBusters

| Model | Tokens | AAR (%) | AAR P (%) | TMscore | GF+IP (%) |
|---|---:|---:|---:|---:|---:|
| LigandMPNN | — | 52.87 | 59.40 | **0.647** | 41.5 |
| LeFlur-pl 470M | 4375 | **68.20** | 74.80 | 0.603 | 41.6 |
| LeFlur-pl **N30 NLL** 470M | 4375 | 67.47 | **75.37** | 0.595 | 41.6 |
| LeFlur-pl N30 oracle 470M | 4375 | 66.60 | 75.18 | 0.640 | **70.4** |

`AAR P` = pocket AAR (residues within 5 Å of any ligand atom). `GF+IP`
(good-fold + in-pocket) requires Boltz-2 TM ≥ 0.5 AND ≥ 1 ligand atom
within 6 Å of a pocket residue. N1 base:

```bash
lobster_generate --config-name experiment/generate_ligand_conditioned_inverse_folding \
    paths=public output_dir=./out/inverse_folding_posebusters
```

### 3. Forward folding — CAMEO 2022

| Model | Tokens | TM | RMSD (Å) | Pass (%) |
|---|---:|---:|---:|---:|
| ESMFold 3B | — | **0.85** | **4.34** | **49.6** |
| DPLM-2 650M | 8192 | 0.70 | 7.40 | 11.8 |
| LeFlur-P 470M SLQ | 256 | 0.67 | 11.97 | 17.3 |
| LeFlur-P **N30 NLL** 470M SLQ | 256 | 0.69 | 12.34 | 17.3 |
| LeFlur-P N30 Oracle 470M SLQ | 256 | 0.75 | 6.73 | 26.8 |

N1 base:

```bash
lobster_generate --config-name experiment/generate_forward_folding \
    paths=public output_dir=./out/forward_folding_cameo
```

### 4. Forward folding with ligand — PoseBusters

| Model | Tokens | TM | RMSD (Å) | GF+IP (%) |
|---|---:|---:|---:|---:|
| RF3 | — | 0.437 | 17.14 | 20.8 |
| Boltz-2 (single seq) | — | 0.651 | 11.73 | **49.6** |
| LeFlur-pl 470M | 4375 | 0.703 | 12.19 | 26.4 |
| LeFlur-pl **N30 NLL** 470M | 4375 | 0.755 | 10.00 | 28.5 |
| LeFlur-pl N30 oracle-TM 470M | 4375 | **0.793** | **7.59** | 30.1 |
| LeFlur-pl N30 oracle-GF+IP 470M | 4375 | 0.696 | 11.22 | **56.9** |

N1 base:

```bash
lobster_generate --config-name experiment/generate_ligand_conditioned_forward_folding \
    paths=public output_dir=./out/forward_folding_posebusters
```

### 5. Unconditional generation (avg across lengths 100–500)

| Method | Tokens | Pass (%) | Avg TM | H/S/C (%) | Clusters |
|---|---:|---:|---:|---:|---:|
| Genie2 + ProteinMPNN | — | 52.0 | 0.808 | 67.6/5.5/26.9 | 223 |
| Proteina + ProteinMPNN | — | 66.4 | 0.909 | 55.6/11.7/32.8 | 230 |
| La Proteina 650M | — | 79.6 | 0.922 | 70.7/6.2/23.1 | 169 |
| DPLM-2 650M | 8192 | 60.4 | 0.85 | 38.2/**17.2**/**44.6** | 141 |
| LeFlur-P 470M SLQ | 256 | 84.8 | 0.937 | **84.3**/0.2/15.5 | 301 |
| LeFlur-P-val 470M SLQ | 256 | 75.8 | 0.895 | 69.1/7.0/24.0 | 288 |
| LeFlur-P-val **NLL** 470M SLQ | 256 | 84.8 | 0.887 | 76.8/3.6/19.6 | 310 |
| LeFlur-P-val-sr8 470M SLQ | 256 | 81.8 | 0.937 | 74.2/6.1/19.7 | 310 |
| LeFlur-P-val-sr9 470M SLQ | 256 | **85.2** | **0.938** | 74.8/5.9/19.3 | **312** |

`H/S/C` are DSSP helix/strand/coil percentages. `-val` is the valine
logit-bias variant (`+1` on the valine logit for the first 25 sampling
steps). `sr8`/`sr9` are self-reflection refinement with TM cutoffs
τ = 0.833 / 0.9 respectively. N1 base:

```bash
lobster_generate --config-name experiment/generate_unconditional \
    paths=public output_dir=./out/unconditional
```

### 6. Ligand-conditioned protein generation (Boltz-design protocol)

| Method | Tokens | Pass (%) | ipTM | iPDE | unique ligands |
|---|---:|---:|---:|---:|---:|
| Proteina Complexa 650M | — | **7.2** | **0.750** | **1.669** | **4** |
| LeFlur-pl 470M | 4375 | 2.0 | 0.678 | 2.431 | 2 |

Length 100, 100 designs per ligand (IAI / FAD / SAM / OQO); a design passes
if Boltz-2 predicts ipTM ≥ 0.9 AND iPDE ≤ 1. N1 base:

```bash
lobster_generate --config-name experiment/generate_ligand_conditioned \
    paths=public output_dir=./out/ligand_conditioned
```

### 7. De-novo binder design — Complexa

Design novel protein binders against a target antigen + epitope. Two dedicated
complex-trained checkpoints: `leflur-binder-3di` (adds a 3Di structural track)
and `leflur-binder-disto` (two-track). The **Complexa 38-target benchmark** is
the canonical evaluation; a design PASSES when Protenix co-folding gives
**pTM > 0.80 AND ipTM > 0.70**.

| Model | Config | Pass (%) | Coverage | Folds/covered |
|---|---|---:|---:|---:|
| Complexa (reference) | — | **28.80** | **35 / 35** † | **11.51** |
| LeFlur `leflur-binder-3di` (best) | `experiment/generate_binder_3di_best` | **7.18** | 36 / 38 | 4.11 |
| LeFlur `leflur-binder-3di` (default) | `experiment/generate_binder_3di` | 6.05 | 36 / 38 | 4.03 |
| LeFlur `leflur-binder-disto` | `experiment/generate_binder_disto` | 6.37 | 33 / 38 | 4.30 |

Both 3Di rows use the **same** `leflur-binder-3di` checkpoint — they differ only
in the sampler schedule. `generate_binder_3di_best` (seq=Log, 3Di=Power,
`stochasticity_seq=60`) is the strongest arm we measured (7.18% at 36/38);
`generate_binder_3di` (a8: seq=Power, 3Di=Log, `stochasticity_tri=80`) is the
established/documented recipe at matching coverage. `Coverage` = targets with
≥ 1 passing design; `Folds/covered` = distinct Foldseek clusters (TM > 0.5)
among a covered target's passing binders. 100 designs/target. † 3 of 38 targets
OOM'd for the Complexa reference model and are excluded from its denominator.
Fetch + run:

```bash
# One-time: fetch the 38-target benchmark (target PDBs + MSAs + manifests)
lobster_leflur_benchmarks fetch complexa-binder

# Loop all 38 targets, 100 designs each (default = leflur-binder-3di)
uv run python examples/run_complexa_binder.py --n-designs 100 \
    --out-dir ./out/complexa_3di
```

Scoring with Protenix runs in a separate environment — see
[`docs/leflur/binder_design.md`](../../../../docs/leflur/binder_design.md) for
the end-to-end fetch → design → score walkthrough, the sampler recipe, and how
to design against your own target.

## Best-of-N ranking with pseudo-NLL

Every `N30 NLL` row in the tables above uses LeFlur's own pseudo-NLL
estimator as a confidence ranker over `N=30` candidates per target. The
ranker lives in [`_pll_scoring.py`](_pll_scoring.py) and is wired into the
CLI as `generation.mode=score_pll`. Workflow:

1. Generate `N` candidates per target with any of the generate modes
   (`num_samples=N` for unconditional, `n_designs_per_structure=N` for
   inverse folding, etc.). This produces a `sequences_*.csv` under the
   output dir.
2. Score every row with `lobster_generate --config-name experiment/score_pll`.
   Pass `generation.rank_within=<group_column>` to compute per-target
   ranks; pass `generation.variants=[...]` to restrict to the variants of
   interest (`joint_protein` is the conference-default ranker).
3. The scored CSV is written next to your output dir, with one
   `pll_<variant>` column per scored variant and one `rank_<variant>`
   integer-rank column per group.

Example (N30 NLL inverse folding on CAMEO — row 3 of Table 1):

```bash
# Step 1: 30 candidates per structure
lobster_generate --config-name experiment/generate_inverse_folding \
    paths=public \
    generation.n_designs_per_structure=30 \
    output_dir=./out/if_cameo_n30

# Step 2: rank each target's 30 candidates by joint_protein NLL
lobster_generate --config-name experiment/score_pll \
    paths=public \
    generation.candidates_csv=$(ls ./out/if_cameo_n30/sequences_inverse_folding_*.csv) \
    generation.rank_within=input_structure \
    generation.variants='[joint_protein]' \
    output_dir=./out/if_cameo_n30

# The argmin candidate per input_structure (rank_joint_protein == 1) is the
# N30 NLL pick — filter the resulting CSV however you like:
#   awk -F, '$NF==1' ./out/if_cameo_n30/pll_scores_*.csv
```

For the protein-ligand checkpoint (tables 2, 4, 6):

```bash
lobster_generate --config-name experiment/score_pll \
    paths=public \
    model.ckpt_path=leflur-pl \
    model._target_=lobster.model.leflur.LeFlurProteinLigandLightningModule \
    generation.candidates_csv=./out/pl_run/sequences_*.csv \
    generation.rank_within=target_id \
    generation.variants='[joint_protein, joint_all, joint_true_4]'
```

Available variants:

- **Protein-only** (4): `seq`, `struc`, `joint_protein` (additive seq + struc;
  the paper's default), `joint_true_2` (true AO-ARM over the unified
  2L-token stream — slightly higher variance but matches the
  Section 4.2 derivation).
- **Protein-ligand** (8): `seq`, `struc`, `lig_atom`, `lig_struc`,
  `joint_protein`, `joint_ligand`, `joint_all`, `joint_true_4`.

See the docstrings in
[`_pll_scoring.py`](_pll_scoring.py) for the Monte-Carlo stratified-`t`
estimator details and the recommended `K` (default 32) / `eps` (default
0.02) hyperparameters.

## API entry points

For Python integration outside the CLIs:

```python
from lobster.model.leflur import (
    LeFlurSequenceStructureEncoderLightningModule,
    LeFlurProteinLigandLightningModule,
    resolve_checkpoint,
)

# Auto-downloads to ~/.cache/lobster/leflur/checkpoints/ on first call
ckpt_path = resolve_checkpoint("leflur-ted")
model = LeFlurSequenceStructureEncoderLightningModule.load_from_checkpoint(
    ckpt_path,
    map_location="cuda",
).eval()
```

The protein-ligand ablation, baseline, and Pareto-front evaluator classes
live under [`lobster.metrics.protein_ligand`](../../metrics/protein_ligand/).

## Citation

If you use LeFlur (or its LatentGenerator structure tokenizer) in your work,
please cite:

> Sidney L. Lisanza, Karina Zadorozhny, Frederic A. Dreyer, and Kyunghyun Cho. **LeFlur: A Biomolecular Design Model with Latent Structure Tokens.** *The 2026 Workshop on Generative and Agentic AI for Biology*, 2026. <https://openreview.net/forum?id=z5EwGneX36>

```bibtex
@inproceedings{lisanza2026leflur,
  title     = {{LeFlur}: A Biomolecular Design Model with Latent Structure Tokens},
  author    = {Lisanza, Sidney L. and Zadorozhny, Karina and Dreyer, Frederic A. and Cho, Kyunghyun},
  booktitle = {The 2026 Workshop on Generative and Agentic AI for Biology},
  year      = {2026},
  url       = {https://openreview.net/forum?id=z5EwGneX36}
}
```

You may also want to cite the LBSTER codebase that LeFlur ships through;
see [`README.md#citations`](../../../../README.md#citations) at the repo
root.
