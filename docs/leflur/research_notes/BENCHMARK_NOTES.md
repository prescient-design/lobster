# Inverse Folding Benchmark: Gen-UME vs LigandMPNN vs ProteinMPNN

## Overview

This document describes all changes and additions made to benchmark Gen-UME inverse folding against LigandMPNN and ProteinMPNN on the PoseBusters benchmark dataset, including pocket-specific amino acid recovery (AAR) tracking.

---

## 1. Pocket AAR for Gen-UME Protein-Only Inverse Folding

### Problem

The `generate.py` inverse folding mode only tracked overall percent identity (AAR). There was no way to measure how well the model recovers amino acids specifically within the ligand-binding pocket, which is critical for comparing protein-only vs protein-ligand models.

### Changes

#### `src/lobster/hydra_config/experiment/generate_inverse_folding_90M_slq_posebuster_ponly.yaml`

Added two config parameters under `generation:`:

```yaml
# Ligand structures for pocket AAR computation (optional)
ligand_structures: "/cv/home/lisanzas/lobster/data/posebusters/processed/posebusters_benchmark/*ligand.pt"
pocket_distance_threshold: 5.0  # Angstrom distance cutoff for pocket definition
```

#### `src/lobster/cmdline/generate.py` — `_generate_inverse_folding` function

**Setup phase (after structure path resolution):**
- Reads optional `ligand_structures` config and `pocket_distance_threshold` (default 5.0)
- Resolves ligand file glob and builds a mapping from protein paths to ligand paths using the `_protein.pt` / `_ligand.pt` naming convention
- Defines two inline helper functions:
  - `_compute_pocket_mask(protein_coords, ligand_coords, protein_mask, threshold)` — returns boolean mask of residues with CA within threshold of any ligand atom (reuses logic from `protein_ligand_inverse_folding.py`)
  - `_compute_aar(predicted_seq, ground_truth_seq, aar_mask)` — computes amino acid recovery rate with optional mask support (reuses logic from `protein_ligand_inverse_folding.py`)
- Initializes aggregate lists: `all_aar_overall`, `all_aar_pocket`, `all_aar_nonpocket`, `all_n_pocket_residues`

**Per-batch (after filtering):**
- For each protein sample, loads the matching `*_ligand.pt` file from the mapping
- Extracts ligand coordinates with fallback keys (`atom_coords` -> `coords` -> `ligand_coords`)

**Per-design (after percent identity computation):**
- Computes pocket mask from protein CA coords and ligand coords
- Computes overall/pocket/non-pocket AAR
- Logs per-sample AAR breakdown

**Aggregate statistics:**
- Reports pocket AAR summary (overall, pocket, non-pocket, delta, avg pocket size) in log output
- Includes `aar_overall`, `aar_pocket`, `aar_nonpocket`, `n_pocket_residues` in CSV via `calculate_aggregate_stats`

**Backward compatibility:** If `ligand_structures` is not set (null), pocket AAR is simply skipped.

---

## 2. LigandMPNN / ProteinMPNN Benchmarking

### Setup

**Repository:** Cloned [LigandMPNN](https://github.com/dauparas/LigandMPNN) to `/cv/home/lisanzas/LigandMPNN/`

**Model parameters:** Downloaded via `bash get_model_params.sh "./model_params"` — includes:
- `proteinmpnn_v_48_020.pt` (ProteinMPNN, 0.20A noise)
- `ligandmpnn_v_32_010_25.pt` (LigandMPNN, 0.10A noise)

**Numpy compatibility fix:** Patched `LigandMPNN/openfold/np/residue_constants.py` to replace deprecated `np.int` with `np.int64` (3 occurrences on lines 1124, 1127, 1283).

**Prody dependency:** Installed via `uv pip install prody` in the lobster environment.

### Input Preparation

**PDB element column fix:** The GT complex PDB files from `writepdb_ligand_complex` were missing the element column (positions 77-78 in PDB format), causing LigandMPNN/prody to fail to determine atom types for ligand HETATM records. Created a fix script that adds proper element symbols inferred from atom names.

**Input files used:**
- **Protein-only PDBs:** `posebusters_benchmark_inverse_folding_450M_ep188/{id}_protein.pdb` (280 files)
- **Complex PDBs (fixed):** `/cv/home/lisanzas/LigandMPNN/inputs/fixed_complex_pdbs/{id}_complex.pdb` (280 files, with element columns added)

**Input JSONs:** Generated for LigandMPNN's `--pdb_path_multi` flag:
- `/cv/home/lisanzas/LigandMPNN/inputs/protein_only_paths.json` (280 entries)
- `/cv/home/lisanzas/LigandMPNN/inputs/complex_paths.json` (280 entries, pointing to fixed PDBs)

### Baseline Runs

All runs used `--seed 111` and `--number_of_batches 1` (1 design per structure).

```bash
# ProteinMPNN (protein-only baseline)
cd /cv/home/lisanzas/LigandMPNN
uv run --project /cv/home/lisanzas/lobster python run.py \
    --model_type "protein_mpnn" --seed 111 \
    --pdb_path_multi "./inputs/protein_only_paths.json" \
    --out_folder "./outputs/proteinmpnn_posebusters" \
    --number_of_batches 1

# LigandMPNN WITH ligand context
uv run --project /cv/home/lisanzas/lobster python run.py \
    --model_type "ligand_mpnn" --seed 111 \
    --pdb_path_multi "./inputs/complex_paths.json" \
    --out_folder "./outputs/ligandmpnn_posebusters" \
    --number_of_batches 1

# LigandMPNN WITHOUT ligand context (ablation)
uv run --project /cv/home/lisanzas/lobster python run.py \
    --model_type "ligand_mpnn" --seed 111 \
    --pdb_path_multi "./inputs/complex_paths.json" \
    --out_folder "./outputs/ligandmpnn_no_context_posebusters" \
    --ligand_mpnn_use_atom_context 0 \
    --number_of_batches 1
```

**Output locations:**
- `/cv/home/lisanzas/LigandMPNN/outputs/proteinmpnn_posebusters/seqs/*.fa`
- `/cv/home/lisanzas/LigandMPNN/outputs/ligandmpnn_posebusters/seqs/*.fa`
- `/cv/home/lisanzas/LigandMPNN/outputs/ligandmpnn_no_context_posebusters/seqs/*.fa`

---

## 3. New Scripts

### `lobster/scripts/prepare_ligandmpnn_inputs.py`

Generates input JSON files for LigandMPNN's `--pdb_path_multi` flag by scanning existing GT PDB files. Filters out reconstructed/decoded files — only includes `{id}_complex.pdb` and `{id}_protein.pdb`.

```bash
uv run python scripts/prepare_ligandmpnn_inputs.py \
    --pdb_dir ./posebusters_benchmark_inverse_folding_450M_ep188 \
    --output_dir /cv/home/lisanzas/LigandMPNN/inputs
```

### `lobster/scripts/fix_pdb_elements.py`

Fixes PDB files by adding element column (positions 77-78) to HETATM records. Infers elements from atom names. Required because `writepdb_ligand_complex` doesn't write element columns, and LigandMPNN/prody needs them.

```bash
uv run python scripts/fix_pdb_elements.py \
    --input_dir ./posebusters_benchmark_inverse_folding_450M_ep188 \
    --output_dir /cv/home/lisanzas/LigandMPNN/inputs/fixed_complex_pdbs
```

### `lobster/scripts/compare_inverse_folding_baselines.py`

Unified comparison script that:
1. Loads GT sequences and ligand coordinates from `.pt` files
2. Parses FASTA outputs from LigandMPNN/ProteinMPNN and Gen-UME
3. Computes pocket mask (CA within 5A of ligand atoms)
4. Computes overall/pocket/non-pocket AAR for each model
5. Outputs per-sample CSV and a summary comparison table

**Token mapping:** Uses the lobster standard residue order: `ARNDCQEGHILKMFPSTWYV` (indices 0-19, sorted by 3-letter AA code alphabetically: ALA, ARG, ASN, ASP, CYS, GLN, GLU, GLY, HIS, ILE, LEU, LYS, MET, PHE, PRO, SER, THR, TRP, TYR, VAL).

```bash
uv run python scripts/compare_inverse_folding_baselines.py \
    --gt_data_dir ./data/posebusters/processed/posebusters_benchmark \
    --genume_fasta_dir ./posebusters_benchmark_inverse_folding_450M_ep188 \
    --proteinmpnn_dir /cv/home/lisanzas/LigandMPNN/outputs/proteinmpnn_posebusters/seqs \
    --ligandmpnn_dir /cv/home/lisanzas/LigandMPNN/outputs/ligandmpnn_posebusters/seqs \
    --ligandmpnn_nocontext_dir /cv/home/lisanzas/LigandMPNN/outputs/ligandmpnn_no_context_posebusters/seqs \
    --output inverse_folding_baseline_comparison.csv
```

---

## 4. Results

**Dataset:** PoseBusters Benchmark (424 total samples, 277 with all model outputs)
**Pocket threshold:** 5.0 Angstroms
**Average pocket size:** 7.7 residues

```
Model                                Overall AAR   Pocket AAR  Non-pocket AAR    Delta
-----------------------------------------------------------------------------------------------
ProteinMPNN (n=277)                       46.64%       42.58%          46.69%   -4.11%
LigandMPNN no context (n=277)             50.92%       43.96%          51.05%   -7.09%
LigandMPNN w/ ligand (n=277)              52.02%       58.68%          51.72%   +6.95%
Gen-UME 450M no ligand (n=277)            44.84%       50.53%          44.62%   +5.91%
Gen-UME 450M w/ ligand (n=277)            52.05%       57.46%          51.86%   +5.60%
```

### Key Findings

- **Overall AAR**: Gen-UME 450M w/ ligand (52.05%) matches LigandMPNN w/ ligand (52.02%); both outperform ProteinMPNN (46.64%).
- **Pocket AAR**: LigandMPNN w/ ligand leads at 58.68%, Gen-UME 450M w/ ligand close at 57.46%.
- **Without ligand context**: ProteinMPNN (-4.11%) and LigandMPNN no-context (-7.09%) show worse pocket recovery, confirming pocket residues are harder without ligand information.
- **Gen-UME structural advantage**: Gen-UME without ligand still shows a positive delta (+5.91%), suggesting it captures pocket structure signals even without explicit ligand context.
- **Ligand context swing**: LigandMPNN goes from -7.09% to +6.95% delta (14% swing) when ligand is provided.

### Per-sample results

Saved to `lobster/inverse_folding_baseline_comparison.csv` with columns:
- `complex_id`, `length`, `n_pocket_residues`
- `proteinmpnn_aar_overall`, `proteinmpnn_aar_pocket`, `proteinmpnn_aar_nonpocket`
- `ligandmpnn_aar_overall`, `ligandmpnn_aar_pocket`, `ligandmpnn_aar_nonpocket`
- `ligandmpnn_nc_aar_overall`, `ligandmpnn_nc_aar_pocket`, `ligandmpnn_nc_aar_nonpocket`
- `genume_no_ligand_aar_overall`, `genume_no_ligand_aar_pocket`, `genume_no_ligand_aar_nonpocket`
- `genume_with_ligand_aar_overall`, `genume_with_ligand_aar_pocket`, `genume_with_ligand_aar_nonpocket`

---

## 5. File Summary

### New files created

| File | Purpose |
|------|---------|
| `lobster/scripts/prepare_ligandmpnn_inputs.py` | Generate input JSONs for LigandMPNN batch processing |
| `lobster/scripts/fix_pdb_elements.py` | Fix missing element columns in complex PDB files |
| `lobster/scripts/compare_inverse_folding_baselines.py` | Unified comparison script for all models |
| `lobster/inverse_folding_baseline_comparison.csv` | Per-sample benchmark results |
| `lobster/BENCHMARK_NOTES.md` | This document |
| `lobster/scripts/compare_dplm2_cameo.py` | DPLM-2 vs baselines AAR comparison on CAMEO |
| `lobster/cameo_baseline_comparison_with_dplm2.csv` | CAMEO per-sample results including DPLM-2 |

### Modified files

| File | Change |
|------|--------|
| `src/lobster/hydra_config/experiment/generate_inverse_folding_90M_slq_posebuster_ponly.yaml` | Added `ligand_structures` and `pocket_distance_threshold` config |
| `src/lobster/cmdline/generate.py` (`_generate_inverse_folding`) | Added ligand loading, pocket mask computation, pocket AAR tracking |
| `/cv/home/lisanzas/LigandMPNN/openfold/np/residue_constants.py` | Fixed `np.int` -> `np.int64` for numpy compatibility |

### External repos

| Path | Purpose |
|------|---------|
| `/cv/home/lisanzas/LigandMPNN/` | Cloned LigandMPNN repo with model params |
| `/cv/home/lisanzas/LigandMPNN/inputs/` | Input JSONs and fixed complex PDB files |
| `/cv/home/lisanzas/LigandMPNN/outputs/` | ProteinMPNN and LigandMPNN FASTA outputs |
| `/cv/home/lisanzas/dplm/` | Cloned DPLM repo with venv and model outputs |
| `/cv/home/lisanzas/dplm/generation-results/dplm2_650m_cameo/` | DPLM-2 inverse folding predictions on CAMEO |
| `/cv/home/lisanzas/dplm/data-bin/cameo_lobster/` | Tokenized CAMEO structures for DPLM-2 |

---

## 6. DPLM-2 Benchmarking (CAMEO Dataset)

### Setup

**Repository:** Cloned [DPLM](https://github.com/bytedance/dplm) to `/cv/home/lisanzas/dplm/`

**Model:** `dplm2_650m` (650M parameters) from [HuggingFace](https://huggingface.co/airkingbd/dplm2_650m)

**Environment:** Python 3.12 venv at `/cv/home/lisanzas/dplm/.venv/` with PyTorch nightly (cu128) for B200 GPU support.

**Python 3.12 compatibility fixes applied:**
- Replaced `import imp` with `import importlib` in `src/byprot/datamodules/dataset/tokenized_protein.py`
- Fixed mutable dataclass defaults (`field(default=X())` -> `field(default_factory=X)`) across multiple files in `src/byprot/models/` and `esm` site-packages
- Upgraded hydra-core from 1.2.0 to 1.3.2 for Python 3.12 dataclass compatibility
- Downgraded biotite to <1.0 for ESM `filter_backbone` API compatibility
- Pinned numpy<2 and setuptools<70

### Data Preparation

**CAMEO dataset:** 127 structures from `/cv/data/ai4dd/data2/lisanzas/AFDB/valid_cameo/` (subset of the DPLM CAMEO 2022 set of 183 structures from EigenFold).

**Structure tokenization:** Converted PDB files to DPLM-2 structure tokens using the built-in structure tokenizer (`airkingbd/struct_tokenizer`). Output saved to `/cv/home/lisanzas/dplm/data-bin/cameo_lobster/struct_seq.fasta`.

### Inverse Folding Run

```bash
cd /cv/home/lisanzas/dplm && source .venv/bin/activate
python generate_dplm2.py \
    --model_name airkingbd/dplm2_650m \
    --task inverse_folding \
    --input_fasta_path data-bin/cameo_lobster/struct_seq.fasta \
    --max_iter 100 \
    --unmasking_strategy deterministic \
    --sampling_strategy argmax \
    --batch_size 4 \
    --save_pdb False \
    --saveto generation-results/dplm2_650m_cameo
```

**Output:** `/cv/home/lisanzas/dplm/generation-results/dplm2_650m_cameo/inverse_folding/aatype.fasta` (127 predicted sequences)

### CAMEO Inverse Folding Results

**Dataset:** CAMEO 2022 (127 structures, protein-only, no ligand context)

```
Model                    Overall AAR (n=127)
-------------------------------------------------
ProteinMPNN                          42.93%
LigandMPNN                           46.72%
DPLM-2 650M                          49.22%
```

### Key Findings

- **DPLM-2 outperforms both baselines** on CAMEO protein-only inverse folding: 49.22% AAR vs LigandMPNN 46.72% (+2.50%) and ProteinMPNN 42.93% (+6.29%).
- This is a protein-only comparison (no ligand context for any model). DPLM-2's multimodal training on structure+sequence data provides an advantage for structure-conditioned sequence recovery.
- Note: DPLM-2 uses tokenized structure representations rather than raw coordinates, which is a fundamentally different conditioning approach from ProteinMPNN/LigandMPNN's graph-based backbone encoding.

### Per-sample results

Saved to `lobster/cameo_baseline_comparison_with_dplm2.csv` with columns:
- `sample_id`, `length`, `proteinmpnn_aar`, `ligandmpnn_aar`, `dplm2_aar`

### Comparison script

```bash
cd /cv/home/lisanzas/lobster
uv run python scripts/compare_dplm2_cameo.py
```

---

## 7. Pending

- **Gen-UME 90M protein-only**: Run with the modified `generate_inverse_folding_90M_slq_posebuster_ponly.yaml` config on a GPU node to get baseline pocket AAR for the smaller model. Command:
  ```bash
  uv run python -m lobster.cmdline.generate \
      --config-path "../hydra_config/experiment" \
      --config-name generate_inverse_folding_90M_slq_posebuster_ponly
  ```
- **writepdb_ligand_complex fix**: Consider patching `writepdb_ligand_complex` in `src/lobster/model/latent_generator/io/_write_pdb.py` to include element columns in HETATM records, avoiding the need for `fix_pdb_elements.py` in future runs.
