# Gen-UME: Generative Unified Molecular Encoder

Gen-UME is a generative model for protein and protein-ligand design based on discrete flow matching. It supports **unconditional generation**, **inverse folding**, **forward folding**, and **protein-ligand** variants of inverse/forward folding where ligand context improves binding pocket design and structure prediction.


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
    --checkpoint /cv/scratch/u/lisanzas/gen_ume_protein_ligand_medium_afdb/runs/2026-02-25T14-46-35/epoch=138-step=24802-val_loss=1.6219.ckpt \
    --data_dir test_data/protein_ligand/ \
    --structure_path ./output/pl_inverse_folding/ \
    --output ./output/pl_inverse_folding/results.csv \
    --nsteps 50 \
    --device cuda \
    --num_samples -1 \
    --temperature_seq 0.2946377400416276 \
    --temperature_struc 0.5872683450058442 \
    --temperature_ligand 0.818357063066881 \
    --stochasticity_seq 10 \
    --stochasticity_struc 40 \
    --stochasticity_ligand 40 \
    --inference_schedule_seq LogInferenceSchedule \
    --inference_schedule_struc LinearInferenceSchedule \
    --inference_schedule_ligand_atom LinearInferenceSchedule \
    --inference_schedule_ligand_struc LogInferenceSchedule \
    --pocket_threshold 5 \
    --save_gt_structure \
    --decode_structure \
    --save_reconstructed_input \
    --minimize_ligand \
    --use_esmfold
```

Structures and results are written to `./output/pl_inverse_folding/`.

Key options:
- `--decode_structure`: Save predicted protein structures as PDB files (includes decoded ligand)
- `--save_gt_structure`: Save ground truth protein and complex structures
- `--save_reconstructed_input`: Save reconstructed input structures for comparison
- `--minimize_ligand`: Apply Open Babel geometry correction to decoded ligands
- `--use_esmfold`: Run ESMFold validation on designed sequences
- `--pocket_threshold`: Distance in angstroms to define binding pocket residues (default: 5.0)

### Protein-Ligand Forward Folding

Predict protein structure from sequence with ligand context.

```bash
uv run python -m lobster.cmdline.evaluate_protein_ligand_forward_folding \
    --checkpoint /cv/scratch/u/lisanzas/gen_ume_protein_ligand_medium_afdb/runs/2026-02-25T14-46-35/epoch=138-step=24802-val_loss=1.6219.ckpt \
    --data_dir test_data/protein_ligand/ \
    --structure_path ./output/pl_forward_folding/ \
    --output ./output/pl_forward_folding/results.csv \
    --nsteps 200 \
    --device cuda \
    --num_samples -1 \
    --temperature_seq 0.15279667854390633 \
    --temperature_struc 0.18605909386731256 \
    --temperature_ligand 0.5819150856331732 \
    --stochasticity_seq 10 \
    --stochasticity_struc 10 \
    --stochasticity_ligand 20 \
    --inference_schedule_seq LinearInferenceSchedule \
    --inference_schedule_struc PowerInferenceSchedule \
    --inference_schedule_ligand_atom PowerInferenceSchedule \
    --inference_schedule_ligand_struc LinearInferenceSchedule \
    --ligand_context_mode atom_bond_only \
    --pocket_threshold 5 \
    --save_gt_structure \
    --minimize_ligand \
    --save_structures
```

Structures and results are written to `./output/pl_forward_folding/`.

Key options:
- `--save_structures`: Save predicted protein structures as PDB files
- `--ligand_context_mode`: How ligand context is provided (`atom_bond_only`, etc.)
- `--inference_schedule_*`: Inference schedule per modality (`LinearInferenceSchedule`, `PowerInferenceSchedule`, `LogInferenceSchedule`)

### Data Format

The protein-ligand evaluators expect paired `.pt` files:

```
data_dir/
├── pdb_id_protein.pt  # Contains: coords_res, sequence, mask, indices
└── pdb_id_ligand.pt   # Contains: atom_coords, atom_names/element_indices, mask, bond_matrix
```

Bundled test data (4 complexes from PoseBusters): `test_data/protein_ligand/`

## Installation

```bash
cd /path/to/lobster
uv pip install -e .
```

---

**Last Updated**: March 2026
