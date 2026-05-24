# LeFlur

LeFlur is a discrete-flow-matching model for protein and protein-ligand design.
A single set of checkpoints supports five inference modes:

| Mode | Input | Output | Checkpoint |
|---|---|---|---|
| **Unconditional generation** | length(s) | novel sequences + structures | `leflur-base`, `leflur-ted` |
| **Forward folding** | sequence (or PDB to extract sequence) | predicted structure | `leflur-ted` |
| **Inverse folding** | PDB / CIF | designed sequence | `leflur-ted` |
| **Ligand-conditioned generation** | ligand (atoms + bonds) | binding protein | `leflur-pl` |
| **Ligand-conditioned forward/inverse folding** | protein + ligand | structure or sequence with pocket awareness | `leflur-pl` |

The three canonical checkpoints (~17 GiB total) live on HuggingFace at
[`Sidney-Lisanza/leflur`](https://huggingface.co/Sidney-Lisanza/leflur) and
download automatically on first use.

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
- **[`checkpoints.md`](../../../../docs/leflur/checkpoints.md)** — the three
  canonical checkpoints, how to list / inspect / fetch them, and how the
  paired Latent Generator codecs are pulled in automatically.
- **[`cli.md`](../../../../docs/leflur/cli.md)** — full CLI reference for
  the three entry points: `lobster_generate`, `lobster_autoencode`, and
  `lobster_leflur_checkpoints`.

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

If you use LeFlur in your research, please cite the LBSTER codebase (see the
[top-level README](../../../../README.md#citations)). A LeFlur-specific paper
citation will be added here on publication.
