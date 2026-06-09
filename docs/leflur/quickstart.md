# LeFlur Quickstart

This walkthrough takes you from a fresh install to running each of the five
LeFlur inference modes. Each section is self-contained: you can jump
straight to whichever task you need.

> **Prerequisites.** You've completed [`installation.md`](installation.md)
> and `lobster_leflur_checkpoints list` shows the three canonical entries.
> Every command below runs from the repo root with the `uv` environment
> activated (or prefixed by `uv run`).

> **Where outputs go.** All quickstart commands write under `./out/<mode>`
> in the current directory. Override with `output_dir=/your/path`. CSV
> metrics are written incrementally so you can monitor progress with
> `tail -f out/<mode>/*results*.csv`.

## 1. Unconditional generation

Sample novel sequence+structure pairs from scratch. The default config
uses the publication checkpoint (`leflur-ted`) with self-reflection
refinement and ESMFold validation.

```bash
uv run lobster_generate \
    --config-name experiment/generate_unconditional \
    paths=public \
    generation.num_samples=10 \
    generation.length=[100] \
    output_dir=./out/uncond
```

**First run downloads ~6 GiB** (`leflur-ted` + paired LG codec). Subsequent
runs are cached. Expect ~5 minutes for 10 samples on an A100; longer with
self-reflection iterations.

Outputs:

```
out/uncond/
├── results.csv                                # per-sample metrics
├── designs/                                   # decoded PDB structures
├── esmfold/                                   # ESMFold validation structures
└── foldseek_clusters.tsv                      # diversity clustering
```

Disable Foldseek (skip the binary install) with `generation.calculate_foldseek_diversity=false`.

## 2. Forward folding (sequence → structure)

Predict a structure for one or more sequences. Inputs are PDB / CIF files
that LeFlur reads sequences from; the structure portion is ignored.

```bash
uv run lobster_generate \
    --config-name experiment/generate_forward_folding \
    paths=public \
    'generation.input_structures=[test_data/inv_folding/9jl9.pdb,test_data/inv_folding/9XZO.pdb]' \
    output_dir=./out/ff
```

Accepted input forms:

- Single PDB / CIF file: `generation.input_structures=test_data/inv_folding/9jl9.pdb`
- Glob: `'generation.input_structures="test_data/inv_folding/*.pdb"'`
- List: `'generation.input_structures=[a.pdb,b.pdb,c.cif]'`
- Directory: `generation.input_structures=test_data/inv_folding/`

## 3. Inverse folding (structure → sequence)

Design a sequence for a given backbone:

```bash
uv run lobster_generate \
    --config-name experiment/generate_inverse_folding \
    paths=public \
    'generation.input_structures=[test_data/inv_folding/9jl9.pdb]' \
    output_dir=./out/if
```

Per-sample outputs include the designed sequence, the structure decoded
from the design's token, and the ESMFold-validated structure (allowing
direct TM-score comparison against the input).

## 4. Ligand-conditioned protein generation

Design a novel binder for a ligand. The model receives ligand atom types
and bond connectivity, then emits a protein sequence + structure that
should fit the ligand's pocket.

```bash
uv run lobster_generate \
    --config-name experiment/generate_ligand_conditioned \
    paths=public \
    generation.data_dir=test_data/protein_ligand \
    output_dir=./out/lig_cond
```

The bundled `test_data/protein_ligand/` contains 4 PoseBusters complexes
in the expected paired-`.pt` format (see [`cli.md`](cli.md) for the data
schema).

By default the evaluator runs ESMFold self-consistency. Disable with
`generation.use_esmfold=false`; enable RF3 / Boltz2 co-folding validation
with `generation.use_protenix=true` or `generation.use_boltz=true`
respectively (each requires the corresponding extra installed).

## 5. Ligand-conditioned forward / inverse folding

Same protein backbones as modes 2–3, but conditioned on a ligand. Useful
for evaluating how pocket awareness changes structure or sequence
prediction.

**Forward folding (sequence + ligand → structure)**

```bash
uv run lobster_generate \
    --config-name experiment/generate_ligand_conditioned_forward_folding \
    paths=public \
    generation.data_dir=test_data/protein_ligand \
    output_dir=./out/pl_ff
```

**Inverse folding (structure + ligand → sequence)**

```bash
uv run lobster_generate \
    --config-name experiment/generate_ligand_conditioned_inverse_folding \
    paths=public \
    generation.data_dir=test_data/protein_ligand \
    output_dir=./out/pl_if
```

Both modes report TM-score / RMSD against the ground-truth complex plus
pocket-aware contact metrics (`ligand_in_pocket`,
`good_fold_and_in_pocket`, `n_pocket_contacts`).

## 6. Autoencode (round-trip a structure through the latent space)

Useful for understanding reconstruction quality and as a debugging tool
when configuring new benchmarks.

```bash
uv run lobster_autoencode \
    --config-name experiment/autoencode \
    paths=public \
    'autoencode.input_structures=[test_data/inv_folding/9jl9.pdb]' \
    output_dir=./out/autoenc
```

The protein-ligand variant uses `experiment/autoencode_protein_ligand`
and takes paired `*_protein.pt` / `*_ligand.pt` files.

## Common overrides

These work on every `lobster_generate` config:

| Override | Default | Effect |
|---|---|---|
| `generation.num_samples=N` | varies | Cap the number of inputs processed. Useful for smoke tests. |
| `generation.nsteps=200` | 200 (PL) / 1000 (uncond) | Flow-matching integration steps. Lower = faster, slightly noisier. |
| `generation.batch_size=4` | 1 | Larger batches need more GPU memory; useful on A100 80GB. |
| `seed=12345` | per-config | Make runs deterministic. |
| `output_dir=/path` | `${paths.evaluations.out_root}/<mode>_canonical` | Where artifacts land. |
| `model.ckpt_path=leflur-base` | per-mode | Use a different canonical checkpoint. |

For all knobs, run `lobster_generate --help` or peek at
`src/lobster/hydra_config/experiment/<config_name>.yaml`.

## Next steps

- Switch checkpoints or use your own: [`checkpoints.md`](checkpoints.md)
- Full CLI surface (`generate` / `autoencode` / `leflur_checkpoints`):
  [`cli.md`](cli.md)
