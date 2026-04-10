# Gen-UME: Generative Unified Molecular Encoder

Gen-UME is a generative model for protein and protein-ligand design based on discrete flow matching. It supports **unconditional generation**, **inverse folding**, **forward folding**, **ligand-conditioned protein generation**, and co-folding validation via RF3 and Boltz2.


## Protein-Only Generation

All protein-only examples use Hydra configs in `src/lobster/hydra_config/experiment/`. The configs already contain checkpoint paths and default data paths so they can be run directly.

### Unconditional Generation

Generate novel protein structures and sequences from scratch. Self-reflection is enabled by default to improve structure-sequence consistency. Uses the 750M model with length-100 sequences.

```bash
uv run python -m lobster.cmdline.generate \
    --config-path "../hydra_config/experiment" \
    --config-name generate_unconditional_750M_L100_self_reflection \
    output_dir=./output/unconditional_750M_L100/ \
    generation.num_samples=10
```

Config: `src/lobster/hydra_config/experiment/generate_unconditional_750M_L100_self_reflection.yaml`

### Inverse Folding

Design sequences for given protein structures using the 750M model.

```bash
uv run python -m lobster.cmdline.generate \
    --config-path "../hydra_config/experiment" \
    --config-name generate_inverse_folding_750M \
    output_dir=./output/inverse_folding_750M/ \
    'generation.input_structures=[test_data/inv_folding/9jl9.pdb,test_data/inv_folding/9XZO.pdb]'
```

Supported input formats: single PDB/CIF file, directory, glob pattern, or list of files.

Config: `src/lobster/hydra_config/experiment/generate_inverse_folding_750M.yaml`

### Forward Folding

Predict structures from sequences (sequences are extracted from input structure files) using the 750M model.

```bash
uv run python -m lobster.cmdline.generate \
    --config-path "../hydra_config/experiment" \
    --config-name generate_forward_folding_750M \
    output_dir=./output/forward_folding_750M/ \
    'generation.input_structures=[test_data/inv_folding/9jl9.pdb,test_data/inv_folding/9XZO.pdb]'
```

Config: `src/lobster/hydra_config/experiment/generate_forward_folding_750M.yaml`

## Protein-Ligand Generation

Gen-UME Protein-Ligand extends the model to handle protein-ligand complexes. Providing ligand context improves sequence design and structure prediction for binding pocket residues.

The protein-ligand model uses `ProteinLigandEncoderLightningModule` and expects paired `*_protein.pt` / `*_ligand.pt` files as input data (see [Data Format](#data-format)).

### Protein-Ligand Inverse Folding

Design protein sequences conditioned on both protein structure and ligand context.

```bash
uv run python -m lobster.cmdline.evaluate_protein_ligand_inverse_folding \
    --checkpoint path/to/checkpoint.ckpt \
    --data_dir path/to/data/ \
    --structure_path ./output/pl_inverse_folding/ \
    --output ./output/pl_inverse_folding/results.csv \
    --nsteps 100 \
    --num_samples -1 \
    --decode_structure \
    --save_gt_structure \
    --minimize_ligand
```

Key options:
- `--decode_structure`: Save predicted protein structures as PDB files (includes decoded ligand)
- `--save_gt_structure`: Save ground truth protein and complex structures
- `--minimize_ligand`: Apply geometry correction to decoded ligands
- `--pocket_threshold`: Distance in angstroms to define binding pocket residues (default: 5.0)

### Protein-Ligand Forward Folding

Predict protein structure from sequence with ligand context. Supports best-of-N evaluation to select the best prediction from multiple stochastic samples.

```bash
uv run python -m lobster.cmdline.evaluate_protein_ligand_forward_folding \
    --checkpoint path/to/checkpoint.ckpt \
    --data_dir path/to/data/ \
    --structure_path ./output/pl_forward_folding/ \
    --output ./output/pl_forward_folding/results.csv \
    --nsteps 100 \
    --num_samples -1 \
    --num_predictions 5 \
    --best_of_n_metric tm_score \
    --save_all_predictions \
    --save_structures \
    --save_gt_structure \
    --minimize_ligand
```

Key options:
- `--num_predictions N`: Generate N predictions per sample, select best (default: 1)
- `--best_of_n_metric`: Selection metric: `tm_score` (higher is better) or `rmsd` (lower is better)
- `--save_all_predictions`: Save all N predictions, not just the best
- `--ligand_context_mode`: How ligand is provided: `structure_tokens` (GT 3D encoded via FSQ) or `atom_bond_only` (atom types + bonds, model generates 3D)

### Ligand-Conditioned Protein Generation

Generate novel protein sequences and structures conditioned on a target ligand. The model receives ligand atom types and bond connectivity, then generates a protein binder.

```bash
uv run python -m lobster.cmdline.evaluate_ligand_conditioned_protein_generation \
    --checkpoint path/to/checkpoint.ckpt \
    --data_dir path/to/data/ \
    --raw_data_dir path/to/posebusters_benchmark_set/ \
    --structure_path ./output/conditioned_gen/ \
    --output ./output/conditioned_gen/results.csv \
    --nsteps 100 \
    --num_samples -1 \
    --save_structures \
    --minimize_ligand
```

ESMFold self-consistency validation can be skipped by passing `plm_fold=None` in the evaluator (useful when only RF3/Boltz2 co-folding validation is needed).

### LigandMPNN Baseline

Run the LigandMPNN inverse folding baseline for comparison.

```bash
uv run python -m lobster.cmdline.evaluate_ligandmpnn_baseline \
    --data_dir path/to/data/ \
    --raw_data_dir path/to/posebusters_benchmark_set/ \
    --output results.csv \
    --structure_path ./output/ \
    --num_samples -1
```

## Evaluation Pipeline

The full evaluation pipeline runs all tasks and co-folding validation via SLURM.

### Phase 1: Model Evaluation

Submits 4 parallel GPU jobs (inverse folding, forward folding, conditioned generation, LigandMPNN baseline):

```bash
CKPT=/path/to/checkpoint.ckpt EVAL_TAG=my_experiment \
    bash slurm/scripts/run_full_eval.sh
```

### Phase 2: Co-folding Validation

After Phase 1 completes, run RF3 and/or Boltz2 co-folding on the designed sequences:

```bash
SKIP_PHASE1=1 EVAL_TAG=my_experiment CKPT=/path/to/checkpoint.ckpt \
    COFOLD_BACKEND=rf3 \
    COFOLD_TASKS=if,cg \
    bash slurm/scripts/run_full_eval.sh
```

Options:
- `COFOLD_BACKEND`: `rf3`, `boltz`, or `both`
- `COFOLD_TASKS`: comma-separated list of `if`, `ff`, `cg`, `lmpnn`
- `RF3_N_CHUNKS`: number of parallel RF3 GPU jobs (default: 4)

### Phase 3: Merge Results

After Phase 2 completes, merge co-fold results into evaluation CSVs:

```bash
SKIP_PHASE1=1 SKIP_PHASE2=1 EVAL_TAG=my_experiment CKPT=/path/to/checkpoint.ckpt \
    bash slurm/scripts/run_full_eval.sh
```

## Ligand Placement Metrics

Protein-ligand contact metrics are computed using `compute_protein_ligand_contacts()` from `lobster.metrics._generation_utils`:

- **`ligand_contacts_protein`**: Whether any protein CA atom is within 6A of any predicted ligand atom
- **`ligand_in_pocket`**: Whether the predicted ligand contacts at least one GT binding pocket residue (pocket defined as GT protein residues with CA within 5A of GT ligand)
- **`good_fold_and_in_pocket`**: `ligand_in_pocket` AND TM-score > 0.5 (correct fold + correct pocket)
- **`n_protein_ligand_contacts`**: Number of protein residues within 6A of ligand
- **`n_pocket_contacts`**: Number of GT pocket residues contacted by predicted ligand

These metrics are consistent across the forward folding evaluator, conditioned generation evaluator, and `merge_cofold_results.py` (for Boltz2/RF3 validation).

### Aligned Ligand RMSD

`compute_aligned_ligand_rmsd()` aligns the predicted protein to the GT protein via Kabsch on CA atoms, applies the same rigid-body transform to the predicted ligand, then computes RMSD against the GT ligand in the aligned frame. This measures ligand positioning quality relative to the protein fold.

## Training

### Data Configs

Training data configs are in `src/lobster/hydra_config/data/`:

- `structure_ligand_all.yaml`: Full combined training (PDB + AFDB + PDBBind + SAIR + PLINDER + Distillation + Redesign)
- `structure_ligand_pdb_afdb_sair_plinder.yaml`: PLINDER baseline (5 datasets)
- `structure_ligand_distillation.yaml`: Distillation + redesign datasets only

All configs support `balance_datasets: true` with `max_cluster_replicates` to cap upsampling of small datasets (e.g., set to 5 to prevent distillation/redesign from dominating).

### Training Commands

```bash
# Protein-ligand training with ALL config
lobster_train data=structure_ligand_all model=gen_ume_protein_ligand \
    trainer.devices=8 data.batch_size=28

# PLINDER baseline
lobster_train data=structure_ligand_pdb_afdb_sair_plinder model=gen_ume_protein_ligand \
    trainer.devices=8 data.batch_size=28
```

## Data Format

The protein-ligand evaluators expect paired `.pt` files:

```
data_dir/
├── pdb_id_protein.pt  # Contains: coords_res, sequence, mask, indices
└── pdb_id_ligand.pt   # Contains: atom_coords, atom_names/element_indices, mask, bond_matrix
```

Bundled test data (4 complexes from PoseBusters): `test_data/protein_ligand/`

### PoseBusters Benchmark

The evaluation benchmark uses the PoseBusters Benchmark set (Buttenschoen et al., 2023), filtered to 206 samples to remove overlap with PDBBind training data. Of these, 125 are evaluated (81 skipped for protein length > 512).

## Benchmark: Gen-UME vs Proteina-Complexa

Head-to-head comparison on ligand-conditioned protein generation using `scripts/benchmark_conditioned_gen.py`:

```bash
# Submit benchmark (30 PoseBusters ligands, 16 Gen-UME + 5 Proteina designs)
python scripts/benchmark_conditioned_gen.py submit \
    --checkpoint path/to/checkpoint.ckpt \
    --num_designs 5 --genume_designs 16 \
    --num_ligands 30 --output_dir /scratch/benchmark/

# ESMFold-filtered benchmark (111 designs -> ESMFold -> RF3 top 5)
python scripts/benchmark_conditioned_gen.py submit_filtered \
    --checkpoint path/to/checkpoint.ckpt \
    --num_designs 111 --rf3_top_k 5 --proteina_designs 5 \
    --num_ligands 30 --output_dir /scratch/benchmark_filtered/

# Merge and compare
python scripts/benchmark_conditioned_gen.py merge --output_dir /scratch/benchmark/
```

## Installation

```bash
cd /path/to/lobster
uv pip install -e .
```

---

**Last Updated**: April 2026
