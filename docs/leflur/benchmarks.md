# LeFlur Benchmarks

The benchmark inputs LeFlur is evaluated on in the paper are mirrored to a
single HuggingFace **dataset** repo, side-by-side with the model
checkpoints under
[`Sidney-Lisanza/leflur`](https://huggingface.co/datasets/Sidney-Lisanza/leflur).
Each benchmark is pre-tokenized into one `.pt` per target (or per
protein-ligand pair), so the publication reproduction commands run
end-to-end on a clean machine with no further data wrangling.

## The four canonical benchmarks

| Short name | HF subdir | Files | Schema | Drives |
|---|---|---:|---|---|
| **`cameo`** | `cameo-2022/` | 127 | protein-only | LeFlur Table 1 inverse folding + Table 3 forward folding (CAMEO 2022 rows). |
| **`multiflow_test`** | `multiflow-test/` | 449 | protein-only | LeFlur Table 1 inverse folding + Table 3 forward folding (MultiFlow rows). |
| **`posebusters_benchmark_no_overlap`** | `posebusters-benchmark-no-overlap/` | 412 (206 pairs) | protein + ligand | LeFlur Table 2 protein-ligand inverse folding + Table 4 protein-ligand forward folding. This is the canonical `leflur-pl` evaluation set. |
| **`posebusters_benchmark`** | `posebusters-benchmark/` | 856 (428 pairs) | protein + ligand | Supplementary PoseBusters tables (no overlap filtering). |

All four live on the dataset side of
[`Sidney-Lisanza/leflur`](https://huggingface.co/datasets/Sidney-Lisanza/leflur)
and are anonymously downloadable — no HF token is required.

### What a record contains

- **CAMEO** (`cameo`) — one `.pt` per target with keys
  `pdb_path`, `sequence`, `sequence_str`, `coords_res`, `chains_ids`,
  `indices`, `mask`, `real_chains`. `pdb_path` is the **basename** of the
  source PDB (e.g. `7dz2.C.pdb`) — the absolute path inside the original
  Genentech file tree is stripped on upload (see
  [Path sanitisation](#path-sanitisation-on-upload) below).
- **MultiFlow** (`multiflow_test`) — one `.pt` per target with keys
  `sequence`, `coords_res`, `mask`, `indices`, `chains`. No `pdb_path`
  field; the upstream MultiFlow release pre-tokenizes everything.
- **PoseBusters** (`posebusters_benchmark*`) — paired files per target:
  `{id}_{ligand}_protein.pt` carries the protein (CAMEO-style schema)
  and `{id}_{ligand}_ligand.pt` carries the ligand
  (`atom_names`, `atom_coords`, `atom_indices`, `element_indices`,
  `bond_matrix`). `pdb_path` on the protein record is the basename.

## Three ways to reference a benchmark

Anywhere LeFlur accepts a benchmark directory (CLI override, Python API,
Hydra interpolation) you can provide any of these three forms:

1. **Short name** (recommended) — `cameo`. Resolved through the
   `KNOWN_BENCHMARKS` registry; downloads into `${LOBSTER_CACHE}` on
   first use, then re-uses the cache forever.
2. **`hf-dataset://` URI** —
   `hf-dataset://Sidney-Lisanza/leflur/cameo-2022`. Same resolution path;
   bypasses the registry. Useful for one-off subdirs we haven't promoted
   to canonical.
3. **Local directory** — `/abs/or/relative/path/to/benchmark_dir`. No
   download; returned verbatim if the directory exists.

`s3://` URIs are **rejected** with a clear message — the public flow
ships no S3 credentials.

## Listing, inspecting, and pre-fetching

LeFlur ships a dedicated CLI for benchmark management, mirroring the
`lobster_leflur_checkpoints` UX:

```bash
# List everything in the registry
lobster_leflur_benchmarks list

# Filter by tag
lobster_leflur_benchmarks list --tag canonical
lobster_leflur_benchmarks list --tag protein-ligand

# Full metadata for one benchmark
lobster_leflur_benchmarks inspect cameo

# Pre-download (no-op if already cached). After this, every
# `lobster_generate --config-name experiment/generate_*` config that
# interpolates `${paths.benchmarks.<name>}` will find the data.
lobster_leflur_benchmarks fetch cameo
lobster_leflur_benchmarks fetch multiflow_test
lobster_leflur_benchmarks fetch posebusters_benchmark_no_overlap

# Show what's in the cache
lobster_leflur_benchmarks cache

# Clear the cache (dry-run first, then for real)
lobster_leflur_benchmarks cache --clear --dry-run
lobster_leflur_benchmarks cache --clear

# Re-build the dataset card without uploading
lobster_leflur_benchmarks dataset-card --print
```

`inspect` shows the HF URI, the direct browse URL, the per-record schema,
the file glob, and the license / citation:

```
$ lobster_leflur_benchmarks inspect posebusters_benchmark_no_overlap
short_name      : posebusters_benchmark_no_overlap
tags            : canonical, protein-ligand, publication
hf_uri          : hf-dataset://Sidney-Lisanza/leflur/posebusters-benchmark-no-overlap
https_url       : https://huggingface.co/datasets/Sidney-Lisanza/leflur/tree/main/posebusters-benchmark-no-overlap
cache_subdir    : posebusters_benchmark_no_overlap
pattern         : *.pt
schema_keys     : pdb_path, sequence, sequence_str, coords_res, chains_ids, indices, mask, real_chains, atom_names, atom_coords, atom_indices, element_indices, bond_matrix
license         : CC-BY-4.0 (matches upstream PoseBusters release)
citation        : Buttenschoen et al., 'PoseBusters: AI-based docking methods fail to generate physically valid poses or generalise to novel sequences', Chem. Sci. (2024). ...
description     : PoseBusters benchmark, deduplicated against the LeFlur training set ...
```

## Where benchmarks live on disk

Downloaded files land under `${LOBSTER_CACHE}/benchmarks/<short_name>/`,
flattened so each `.pt` sits directly under the benchmark's cache subdir
(no HF-side directory nesting leaks through). This is exactly the layout
that `paths/public.yaml` interpolates as `${paths.benchmarks.<name>}`,
so the generate configs work unchanged after `fetch`:

```
$LOBSTER_CACHE/                                          # default ~/.cache/lobster/leflur
├── checkpoints/                                         # see docs/leflur/checkpoints.md
└── benchmarks/
    ├── cameo/                                           # ${paths.benchmarks.cameo}
    │   ├── 7dz2.C.pt
    │   └── ...
    ├── multiflow_test/                                  # ${paths.benchmarks.multiflow_test}
    │   ├── 5S9R_processed.pt
    │   └── ...
    └── posebusters_benchmark_no_overlap/                # ${paths.benchmarks.posebusters_benchmark_no_overlap}
        ├── 5S8I_2LY_protein.pt
        ├── 5S8I_2LY_ligand.pt
        └── ...
```

If you've configured a non-default `LOBSTER_CACHE`, the layout is the
same — just rooted at your custom location.

## Reproducing the publication

Each table maps to a single benchmark short name + canonical Hydra
config. Once you've run the matching `lobster_leflur_benchmarks fetch`,
every command below runs end-to-end with `paths=public`.

| Result | Benchmark | Config | Checkpoint |
|---|---|---|---|
| Inverse folding (Table 1) | `cameo` | `experiment/generate_inverse_folding` | `leflur-ted` |
| PL inverse folding (Table 2) | `posebusters_benchmark_no_overlap` | `experiment/generate_ligand_conditioned_inverse_folding` | `leflur-pl` |
| Forward folding (Table 3) | `cameo` | `experiment/generate_forward_folding` | `leflur-ted` |
| PL forward folding (Table 4) | `posebusters_benchmark_no_overlap` | `experiment/generate_ligand_conditioned_forward_folding` | `leflur-pl` |

End-to-end Table 2 reproduction from a clean machine:

```bash
# 1. Install LeFlur with the protein-ligand extra
uv sync --extra mgm --extra struct-cpu

# 2. (No HF auth required — both repos are public.)

# 3. Pre-fetch the benchmark (~10 MiB; downloads from HF into $LOBSTER_CACHE)
lobster_leflur_benchmarks fetch posebusters_benchmark_no_overlap

# 4. Run Table 2 — protein-ligand inverse folding on PoseBusters NO. The
#    `leflur-pl` checkpoint (~5 GiB) is fetched on first call.
lobster_generate --config-name experiment/generate_ligand_conditioned_inverse_folding \
    paths=public \
    output_dir=./out/inverse_folding_posebusters
```

The corresponding Best-of-N (N30 NLL) row uses the two-step pseudo-NLL
ranker — see the
[**Best-of-N ranking with pseudo-NLL**](../../src/lobster/model/leflur/README.md#best-of-n-ranking-with-pseudo-nll)
section in the LeFlur README for the second `lobster_generate
--config-name experiment/score_pll` step.

## Using benchmarks from Python

For programmatic use outside the CLIs:

```python
from glob import glob
from pathlib import Path

import torch

from lobster.model.leflur import fetch_benchmark, resolve_benchmark

# fetch_benchmark() is the friendlier wrapper for short names.
data_dir = fetch_benchmark("posebusters_benchmark_no_overlap")

# Both halves of each PoseBusters pair are glob-paired by basename — the
# same convention every InverseFoldingEvaluator class uses.
for pf in sorted(Path(data_dir).glob("*_protein.pt")):
    lf = pf.with_name(pf.name.replace("_protein.pt", "_ligand.pt"))
    protein = torch.load(pf, weights_only=False, map_location="cpu")
    ligand  = torch.load(lf, weights_only=False, map_location="cpu")
    ...

# resolve_benchmark() also accepts ``hf-dataset://`` URIs and local dirs.
custom_dir = resolve_benchmark("hf-dataset://Sidney-Lisanza/leflur/cameo-2022")
```

`resolve_benchmark` is idempotent: subsequent calls re-use the same
cached snapshot rather than re-downloading.

## Path sanitisation on upload

LeFlur's internal benchmark `.pt` files carry a `pdb_path` field that
points at the original Genentech file tree (e.g.
`/cv/data/ai4dd/data2/lisanzas/.../7dz2.C.pdb`). The
`lobster_leflur_benchmarks upload` CLI rewrites this to its **basename**
before pushing to HuggingFace, so the published records only carry
`pdb_path = "7dz2.C.pdb"`. Downstream code that wants to locate the raw
PDB should resolve it against an RCSB / CAMEO / PoseBusters mirror by
basename.

The same sanitiser handles every benchmark uniformly — CAMEO (single
record), PoseBusters (paired protein + ligand records), and any future
benchmark with a `pdb_path` field. MultiFlow lacks the field and is
passed through bit-identically.

Pass `--no-sanitize` to `upload` only when you explicitly want the
internal paths preserved (e.g. for an internal reproducibility diff).

## Bring your own benchmark

You don't have to register a custom benchmark to evaluate against it —
the generate configs accept any local directory via Hydra:

```bash
lobster_generate --config-name experiment/generate_inverse_folding \
    paths=public \
    generation.input_structures=/abs/path/to/my_benchmark/*.pt \
    output_dir=./out/my_benchmark_run
```

The directory must contain `.pt` files in the appropriate schema (see
the [What a record contains](#what-a-record-contains) section above).
For protein-only inverse / forward folding, follow the CAMEO schema. For
protein-ligand, follow the PoseBusters paired schema.

If you want to register a new benchmark formally — for example to
distribute it on HF alongside the canonical four — add an entry to
`KNOWN_BENCHMARKS` in
[`src/lobster/model/leflur/benchmarks.py`](../../src/lobster/model/leflur/benchmarks.py),
add a matching `${paths.benchmarks.<name>}` line to
[`src/lobster/hydra_config/paths/public.yaml`](../../src/lobster/hydra_config/paths/public.yaml),
then run `lobster_leflur_benchmarks upload <name>
--token "$HF_TOKEN" --with-card` to push the data and refresh the dataset
card in lockstep.
