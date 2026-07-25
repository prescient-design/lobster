# LeFlur Checkpoints

LeFlur ships five publicly distributed checkpoints on HuggingFace: three core
checkpoints (~17 GiB) covering every published folding/generation evaluation,
plus two complex-trained binder-design checkpoints (~5.7 GiB each).

## The three core checkpoints

| Short name | Family | Size | Purpose |
|---|---|---|---|
| **`leflur-base`** | protein-only | ~6 GiB | De-novo unconditional generation, default temperatures. |
| **`leflur-ted`** | protein-only | ~6 GiB | TED-CATH SS-balanced fine-tune. Best designability / quality trade-off; recommended for publication tables. Default for unconditional, forward, and inverse folding. |
| **`leflur-pl`** | protein-ligand | ~5 GiB | Production protein-ligand checkpoint. Drives all ligand-conditioned generation + protein-ligand forward / inverse folding. |

## The two binder-design checkpoints

| Short name | Family | Size | Purpose |
|---|---|---|---|
| **`leflur-binder-3di`** | protein (complex) | ~5.7 GiB | **Default 3Di binder framework.** Sequence + latent-structure + 3Di generative tracks, complex/epitope-trained. Drives de-novo binder design against a target + epitope. Configs `experiment/generate_binder_3di` (a8) and `experiment/generate_binder_3di_best` (best; 7.18% vs 6.05%). |
| **`leflur-binder-disto`** | protein (complex) | ~5.7 GiB | **Default non-3Di binder framework.** Two-track (sequence + latent structure) complex checkpoint with a distogram auxiliary head. Config `experiment/generate_binder_disto`. |

See [`binder_design.md`](binder_design.md) for the end-to-end binder workflow
(fetch → design → score on the Complexa 38-target benchmark).

All five live on [`Sidney-Lisanza/leflur`](https://huggingface.co/Sidney-Lisanza/leflur).
The paired Latent Generator codecs that LeFlur uses internally
(`LG full attention`, `LG Protein Ligand fsq 4375`,
`LG Protein Ligand cont`) live on
[`Sidney-Lisanza/latent_generator`](https://huggingface.co/Sidney-Lisanza/latent_generator)
and are pulled in automatically — you do not need to fetch them
yourself.

## Three ways to reference a checkpoint

Anywhere LeFlur accepts a `ckpt_path` (CLI override, Python API, Hydra
config) you can provide any of these four forms:

1. **Short name** (recommended) — `leflur-ted`. Resolved through the
   `KNOWN_CHECKPOINTS` registry, downloaded into `${LOBSTER_CACHE}` on
   first use.
2. **`hf://` URI** — `hf://Sidney-Lisanza/leflur/leflur_protein_ligand.ckpt`.
   Same resolution path; bypasses the registry. Useful for one-off
   experiments with a checkpoint we haven't promoted to canonical.
3. **HTTPS URL** — `https://huggingface.co/Sidney-Lisanza/leflur/resolve/main/leflur_protein_ligand.ckpt`.
   Works without an HF token for public repos.
4. **Local path** — `/abs/or/relative/path/to.ckpt`. No download; returned
   verbatim if the file exists.

`s3://` URIs are **rejected** with a clear message — we don't ship S3
credentials in the public package.

## Listing, inspecting, and pre-fetching

LeFlur ships a dedicated CLI for checkpoint management:

```bash
# List everything in the registry
lobster_leflur_checkpoints list

# Filter
lobster_leflur_checkpoints list --family protein_ligand --tag canonical

# Full metadata for one checkpoint
lobster_leflur_checkpoints inspect leflur-ted

# Pre-download (no-op if already cached)
lobster_leflur_checkpoints fetch leflur-ted

# Show what's in the cache
lobster_leflur_checkpoints cache

# Clear the cache (dry-run first, then for real)
lobster_leflur_checkpoints cache --clear --dry-run
lobster_leflur_checkpoints cache --clear
```

`inspect` shows the `hf://` URI, the direct HTTPS URL (for downloading
without the CLI), the paired LG codec, and the recommended Hydra config:

```
$ lobster_leflur_checkpoints inspect leflur-ted
short_name      : leflur-ted
family          : protein
tags            : canonical, protein-only, publication
hf_uri          : hf://Sidney-Lisanza/leflur/leflur_denovo_ted_cath_ss_balanced_ckpt_2026-03-18T12-20-59.ckpt
https_url       : https://huggingface.co/Sidney-Lisanza/leflur/resolve/main/leflur_denovo_ted_cath_ss_balanced_ckpt_2026-03-18T12-20-59.ckpt
paired_lg_codec : LG full attention
recommended_cfg : experiment/generate_unconditional_denovo
description     : Canonical protein-only TED-CATH SS-balanced checkpoint ...
```

## Where checkpoints live on disk

Downloaded files land under `${LOBSTER_CACHE}/checkpoints/`. The cache
layout is content-addressed by the HuggingFace filename:

```
$LOBSTER_CACHE/                                          # default ~/.cache/lobster/leflur
├── checkpoints/
│   ├── leflur_denovo_last_ckpt_2026-03-11T12-11-53.ckpt          # leflur-base
│   ├── leflur_denovo_ted_cath_ss_balanced_ckpt_2026-03-18*.ckpt  # leflur-ted
│   ├── leflur_protein_ligand.ckpt                                # leflur-pl
│   └── checkpoints_for_lg/                                       # paired LG codecs
│       └── LG_Protein_Ligand_fsq_4375_2026-01-05.ckpt
└── benchmarks/                                          # per-dataset eval fixtures
```

If you've configured a non-default `LOBSTER_CACHE`, the layout is the
same — just rooted at your custom location.

## Using checkpoints from Python

For programmatic use outside the CLIs:

```python
from lobster.model.leflur import (
    LeFlurSequenceStructureEncoderLightningModule,
    LeFlurProteinLigandLightningModule,
    resolve_checkpoint,
)

# Protein-only
ckpt = resolve_checkpoint("leflur-ted")
model = LeFlurSequenceStructureEncoderLightningModule.load_from_checkpoint(
    ckpt, map_location="cuda"
).eval()

# Protein-ligand
ckpt = resolve_checkpoint("leflur-pl")
pl_model = LeFlurProteinLigandLightningModule.load_from_checkpoint(
    ckpt, map_location="cuda"
).eval()
```

`resolve_checkpoint` is idempotent and thread-safe: concurrent calls
re-use the same cached file.

## Bring your own checkpoint

If you have a custom-trained checkpoint (e.g. fine-tuned on private data),
just pass its local path or `hf://` URI directly:

```bash
lobster_generate --config-name experiment/generate_unconditional \
    paths=public \
    model.ckpt_path=/scratch/my_finetune/last.ckpt \
    output_dir=./out/uncond_custom
```

The checkpoint must be a Lightning `.ckpt` produced by
`LeFlurSequenceStructureEncoderLightningModule` (protein-only) or
`LeFlurProteinLigandLightningModule` (protein-ligand). The Lightning
module auto-detects which Latent Generator codec to load from the
checkpoint's hparams.

## Reproducing the publication

The four most relevant configs:

| Result | Config | Checkpoint | Benchmark |
|---|---|---|---|
| Unconditional generation (Table 5) | `experiment/generate_unconditional` | `leflur-ted` | — (no benchmark inputs) |
| Forward folding (Table 3) | `experiment/generate_forward_folding` | `leflur-ted` | `cameo` |
| Inverse folding (Table 1) | `experiment/generate_inverse_folding` | `leflur-ted` | `cameo` |
| Ligand-conditioned forward (Table 4) | `experiment/generate_ligand_conditioned_forward_folding` | `leflur-pl` | `posebusters_benchmark_no_overlap` |
| Ligand-conditioned inverse (Table 2) | `experiment/generate_ligand_conditioned_inverse_folding` | `leflur-pl` | `posebusters_benchmark_no_overlap` |
| De-novo binder design (3Di, a8) | `experiment/generate_binder_3di` | `leflur-binder-3di` | `complexa-binder` |
| De-novo binder design (3Di, best) | `experiment/generate_binder_3di_best` | `leflur-binder-3di` | `complexa-binder` |
| De-novo binder design (disto) | `experiment/generate_binder_disto` | `leflur-binder-disto` | `complexa-binder` |

Each Tier-1 config is enforced by automated tests to use only canonical
checkpoint references — see the tests under
`tests/lobster/hydra_config/test_paths_overlay.py`.

Benchmark inputs for the four folding tables are mirrored to the
**dataset** side of `Sidney-Lisanza/leflur` and fetched via the
`lobster_leflur_benchmarks` CLI — see
[`benchmarks.md`](benchmarks.md) for the full registry, the per-dataset
schema, and end-to-end reproduction commands from a clean machine.
