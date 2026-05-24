# LeFlur CLI Reference

LeFlur exposes three console scripts. They're registered as entry points
in `pyproject.toml` and become available as PATH commands once you've
done `uv sync` (or `pip install -e .`).

| Command | Purpose | Driven by |
|---|---|---|
| `lobster_generate` | Run any of the five inference modes | Hydra config |
| `lobster_autoencode` | Encode/decode PDB or paired PL `.pt` files through the latent space | Hydra config |
| `lobster_leflur_checkpoints` | List / inspect / fetch / cache checkpoints | argparse subcommands |

## `lobster_generate`

The main entry point. Selects an inference mode through Hydra and dispatches
internally to the appropriate runner.

```bash
lobster_generate --config-name experiment/<CONFIG_NAME> [overrides...]
```

### The five Tier-1 modes and their configs

| Mode (`generation.mode`) | Config name | Default checkpoint |
|---|---|---|
| `unconditional` | `experiment/generate_unconditional` | `leflur-ted` |
| `forward_folding` | `experiment/generate_forward_folding` | `leflur-ted` |
| `inverse_folding` | `experiment/generate_inverse_folding` | `leflur-ted` |
| `inpainting` | `experiment/generate_inpainting` | `leflur-ted` |
| `ligand_conditioned` | `experiment/generate_ligand_conditioned` | `leflur-pl` |
| `protein_ligand_forward_folding` | `experiment/generate_ligand_conditioned_forward_folding` | `leflur-pl` |
| `protein_ligand_inverse_folding` | `experiment/generate_ligand_conditioned_inverse_folding` | `leflur-pl` |

### Common overrides

All of these work on every config (some are no-ops where they don't apply):

| Override | Description |
|---|---|
| `paths=public` / `paths=internal` | Switch checkpoint and benchmark resolution. External users want `public` (HuggingFace + `$LOBSTER_CACHE`). |
| `model.ckpt_path=<short_name|hf:// uri|local path>` | Override the checkpoint without touching the YAML. |
| `output_dir=<path>` | Where artifacts and CSVs are written. |
| `seed=<int>` | Per-run RNG seed. |
| `generation.num_samples=<N>` | Cap the number of inputs / outputs (smoke tests). |
| `generation.batch_size=<N>` | GPU batch size. |
| `generation.nsteps=<N>` | Flow-matching integration steps. |
| `generation.use_esmfold=<bool>` | Toggle ESMFold validation. |

### Mode-specific overrides

**Unconditional**:

- `generation.length=[100,200,300]` — sequence lengths to sample
- `generation.enable_self_reflection=<bool>` — refinement pass (default `true`)
- `generation.calculate_foldseek_diversity=<bool>` — diversity clustering

**Forward / inverse folding**:

- `generation.input_structures=<path|glob|list>` — input PDB/CIF files
- `generation.use_se3_augmentation=<bool>` — applicable to inverse folding

**Ligand-conditioned modes**:

- `generation.data_dir=<path>` — directory of `*_protein.pt` / `*_ligand.pt` pairs
- `generation.raw_data_dir=<path>` — directory of original SDF / PDB files (for SMILES extraction)
- `generation.num_predictions=<N>` — best-of-N sampling
- `generation.best_of_n_metric=<plddt|tm_score|rmsd>` — best-of-N selection criterion
- `generation.minimize_ligand=<bool>` — geometry correction on decoded ligands
- `generation.use_protenix=<bool>`, `generation.use_boltz=<bool>` — co-folding validators
- `generation.ligand_context_mode=<structure_tokens|atom_bond_only>` — how the ligand is provided

### Tip: dry-run a config

To see exactly what a config resolves to without running anything:

```bash
lobster_generate --config-name experiment/generate_unconditional --cfg job paths=public
```

## `lobster_autoencode`

Round-trip protein or protein-ligand structures through the latent
generator codec. Useful for measuring reconstruction quality or
preparing latent representations for downstream tasks.

```bash
lobster_autoencode --config-name experiment/<CONFIG_NAME> [overrides...]
```

Two configs:

| Config | Inputs | Outputs |
|---|---|---|
| `experiment/autoencode` | PDB / CIF | reconstructed PDB, latent tokens |
| `experiment/autoencode_protein_ligand` | paired `*_protein.pt` / `*_ligand.pt` | reconstructed complex, latent tokens |

Common overrides:

| Override | Description |
|---|---|
| `autoencode.input_structures=<path|glob|list>` | (protein-only) input files |
| `autoencode.data_dir=<path>` | (PL) directory of paired `.pt` files |
| `autoencode.save_decoded_structure=<bool>` | Save reconstructed PDBs |
| `autoencode.save_latent_tokens=<bool>` | Save discrete latent tokens as `.pt` |

The autoencode CLI auto-detects which Lightning module to instantiate
based on the config name and the presence of ligand keys.

## `lobster_leflur_checkpoints`

argparse-driven checkpoint management. Five subcommands:

```bash
lobster_leflur_checkpoints list [--family protein|protein_ligand] [--tag canonical|publication|...]
lobster_leflur_checkpoints inspect <short_name>
lobster_leflur_checkpoints fetch <short_name>
lobster_leflur_checkpoints cache
lobster_leflur_checkpoints cache --clear [--dry-run]
```

By design this CLI **does not** add / update / delete registry entries —
the publication scope freezes the registry in source. To add a new
canonical checkpoint, edit
`src/lobster/model/leflur/checkpoints.py` and re-install.

### `list`

```text
$ lobster_leflur_checkpoints list
short_name   family          tags                              description
-----------  --------------  --------------------------------  --------------------------------------------------------------
leflur-base  protein         canonical,protein-only            Canonical protein-only base checkpoint (de-novo, last; ...).
leflur-ted   protein         canonical,protein-only,publication Canonical protein-only TED-CATH SS-balanced checkpoint ...
leflur-pl    protein_ligand  canonical,protein-ligand          Production protein-ligand checkpoint (2026-02-11).
```

### `inspect`

Prints full `CheckpointInfo` metadata. See [`checkpoints.md`](checkpoints.md)
for an example.

### `fetch`

Pre-downloads a checkpoint into `${LOBSTER_CACHE}/checkpoints/`. Subsequent
calls are no-ops. Useful for warming a cache on shared infra before a
batch job.

```bash
lobster_leflur_checkpoints fetch leflur-ted
```

### `cache`

Without arguments: prints what's currently cached.

With `--clear`: removes all cached checkpoint files. Add `--dry-run` to
preview the deletion without acting.

```bash
$ lobster_leflur_checkpoints cache
Cache root: /home/sid/.cache/lobster/leflur/checkpoints
  6.2 GiB  leflur_denovo_ted_cath_ss_balanced_ckpt_2026-03-18T12-20-59.ckpt
  5.1 GiB  leflur_protein_ligand.ckpt
Total: 11.3 GiB
```

## Data formats

### PDB / CIF input

Standard PDB or mmCIF files. LeFlur reads ATOM records and ignores HETATM
(unless you're using a ligand-conditioned mode, in which case ligands
come from the paired `.pt` files instead).

### Paired `.pt` files (protein-ligand modes)

The protein-ligand modes consume a directory of paired files:

```
data_dir/
├── 1abc_protein.pt   # { coords_res, sequence, mask, indices }
├── 1abc_ligand.pt    # { atom_coords, element_indices, mask, bond_matrix }
├── 2xyz_protein.pt
└── 2xyz_ligand.pt
```

These files are produced by `lobster.transforms._structure_transforms.StructureLigandTransform`.
You can generate them from raw PDB+SDF pairs via the prep scripts in
`scripts/` (see the PoseBusters benchmark preparation pipeline).

The bundled test set lives at `test_data/protein_ligand/` (4 PoseBusters
complexes).

## Output schema

Every `lobster_generate` invocation writes one CSV plus a directory of
artifacts. The CSV columns are stable across modes:

| Column | Description |
|---|---|
| `idx` | Sample index |
| `pdb_id` | Source PDB id (folding / PL modes) or generation id |
| `sequence` | Designed sequence |
| `tm_score`, `rmsd`, `lddt` | Structural quality (vs GT or ESMFold) |
| `percent_identity` | Sequence vs GT |
| `esmfold_tm_score` | ESMFold self-consistency (when `use_esmfold=true`) |
| `ligand_in_pocket`, `n_pocket_contacts`, `good_fold_and_in_pocket` | PL contact metrics (PL modes only) |
| `runtime_s` | Per-sample wall clock |

Headers are written eagerly so you can `tail -f` the CSV during long runs.
The MetricsCSVWriter handles resume cleanly: if you re-run with the same
`output_dir`, existing rows are kept and only new samples are appended.
