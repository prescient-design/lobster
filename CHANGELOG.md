# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — LeFlur publication release

- **LeFlur model** at `lobster.model.leflur` — a discrete-flow-matching model
  for protein and protein-ligand design. Replaces the internal `gen_ume`
  module (a deprecation shim at `lobster.model.gen_ume` preserves the old
  import paths). Supports five inference modes:
  - Unconditional generation
  - Forward folding (sequence → structure)
  - Inverse folding (structure → sequence)
  - Ligand-conditioned protein generation
  - Ligand-conditioned forward / inverse folding
- **Three canonical checkpoints** on
  [`Sidney-Lisanza/leflur`](https://huggingface.co/Sidney-Lisanza/leflur):
  `leflur-base`, `leflur-ted`, `leflur-pl` (~17 GiB total).
- **`lobster_generate`** console entry point — Hydra-driven CLI dispatching to
  all five inference modes via `--config-name experiment/<mode>`.
- **`lobster_autoencode`** console entry point — round-trip PDB or paired
  protein-ligand `.pt` files through the Latent Generator codec; auto-detects
  protein-only vs protein-ligand inputs.
- **`lobster_leflur_checkpoints`** console entry point — `list / inspect /
  fetch / cache` subcommands for managing the canonical checkpoint registry.
- **`resolve_checkpoint()`** API — accepts short names, `hf://` URIs, HTTPS
  URLs, or local paths. Rejects `s3://` with a clear message. Idempotent
  cache under `$LOBSTER_CACHE` (default `~/.cache/lobster/leflur`).
- **Hydra path overlay** at `lobster/hydra_config/paths/{internal,public}.yaml` —
  external users get the `public` overlay (HuggingFace + local cache),
  internal collaborators get `internal` (shared filesystem). All Tier-1
  experiment configs are enforced (via tests) to interpolate through this
  overlay rather than hard-coding paths.
- **Experiment config tiering** at `lobster/hydra_config/experiment/`:
  - 9 Tier-1 canonical configs (flat at `experiment/`)
  - 48 Tier-2 research configs (under `experiment/research/`)
  - 1 Tier-3 legacy config (under `experiment/legacy/`)
  - 32 per-PDB R5 inpainting configs (`experiment/denovo_r5/`) are kept
    locally on internal machines but gitignored on the publication branch
    since they are research-only sweeps.
- **`lobster.metrics.protein_ligand`** subpackage — groups three
  `Evaluator` classes (forward folding, inverse folding, ligand-conditioned
  generation), two ablation scripts, and the LigandMPNN baseline.
- **Public user docs** at `docs/leflur/`:
  - `installation.md` — extras, env vars, HF auth, Foldseek
  - `quickstart.md` — five-minute walkthroughs of all inference modes
  - `checkpoints.md` — registry + CLI + paired LG codec auto-resolution
  - `cli.md` — full reference for the three entry points
- **Test suites** at `tests/lobster/`:
  - `model/leflur/` — checkpoint resolver, registry invariants, Lightning
    module load smoke (CPU + GPU variants)
  - `cmdline/` — Tier-1 dispatch smoke, CLI surface, ligand-conditioned
    runner config defaults
  - `metrics/` — RMSD-sqrt(3) regression, MetricsCSVWriter contract
  - `hydra_config/` — path overlay regression (no internal literals in
    public-facing configs), tier invariants

### Changed

- **`leflur-pl` repointed** to the `gen_ume_protein_ligand_no_geom_medium`
  2026-03-11 checkpoint (replacing the earlier 2026-02-11 medium snapshot).
  The 2/11 weights underperformed by ~25 pp absolute AAR on the PoseBusters
  no-overlap inverse-folding benchmark (overall 0.45 → 0.70, pocket
  0.51 → 0.78). The 3/11 no-geom-loss variant is the checkpoint that
  produced the conference benchmark numbers. The HuggingFace blob at
  `Sidney-Lisanza/leflur/leflur_protein_ligand.ckpt` and the registered
  `local_source_path` in `KNOWN_CHECKPOINTS` were both updated in lockstep;
  internal users running `paths=internal` see the new path via
  `paths.checkpoints.pl`, external users via `paths=public` get the new
  weights on next `lobster_generate` invocation (the old cached blob is
  invalidated when `huggingface_hub` resolves the new commit SHA, or
  immediately via `lobster_leflur_checkpoints cache --clear`).
- `lobster/__init__.py` imports `ensure_package` before the subpackages
  (fixes a latent circular-import in modules that call `ensure_package` at
  module-import time).
- `cmdline/__init__.py` now re-exports `generate`, `autoencode`, and
  `manage_leflur_checkpoints` alongside the existing `train`, `embed`,
  etc.
- Lightning module checkpoint loading auto-resolves the paired Latent
  Generator codec via `install_paired_lg_codec_overrides()` — checkpoints
  trained against an internal `/cv/...` LG codec are transparently
  redirected to the public HuggingFace mirror at inference time, with no
  upstream changes to the `latent_generator` library.

### Removed

- Orphaned duplicate evaluators
  `lobster/metrics/evaluate_protein_ligand_{forward,inverse}_folding.py`
  (superseded by the cmdline entry-point versions).
- Stale internal-only docs `src/lobster/model/leflur/CHECKPOINTS.md` and
  `src/lobster/model/leflur/BOND_MATRIX_LATENT_GENERATOR_PLAN.md`
  (superseded by `docs/leflur/checkpoints.md` and the code itself; the
  bond-matrix planning doc was archived under
  `docs/leflur/research_notes/`).
- **Standalone argparse evaluation CLIs**
  `cmdline/evaluation/{evaluate_inverse_folding, evaluate_ligand_conditioned_protein_generation, evaluate_protein_ligand_forward_folding, evaluate_protein_ligand_inverse_folding}.py`.
  All four were thin wrappers around runners that are already exposed
  via Hydra modes. The single canonical evaluation entry point is now
  `lobster_generate generation.mode=<X>` with the matching
  `experiment/generate_*.yaml` config. Migration:

  | Old standalone | New invocation |
  |---|---|
  | `python -m lobster.cmdline.evaluation.evaluate_inverse_folding ...` | `lobster_generate --config-name experiment/generate_inverse_folding ...` |
  | `python -m lobster.cmdline.evaluation.evaluate_ligand_conditioned_protein_generation ...` | `lobster_generate --config-name experiment/generate_ligand_conditioned ...` |
  | `python -m lobster.cmdline.evaluation.evaluate_protein_ligand_forward_folding ...` | `lobster_generate --config-name experiment/generate_ligand_conditioned_forward_folding ...` |
  | `python -m lobster.cmdline.evaluation.evaluate_protein_ligand_inverse_folding ...` | `lobster_generate --config-name experiment/generate_ligand_conditioned_inverse_folding ...` |

  The two competitor baselines `esmfold_baseline.py` (ESMFold forward
  fold) and `evaluate_ligandmpnn_baseline.py` (LigandMPNN inverse fold)
  remain available locally but are no longer tracked on the publication
  branch — they evaluate external tools, not LeFlur.
- **Training-only Hydra configs added on this branch** — 62 YAMLs the
  publication branch added on top of `origin/main` for LeFlur / Latent
  Generator training, training-time callbacks (LeFlur evaluators,
  protein-ligand decode/forward/inverse callbacks, LG defaults, S3
  backup), training data configs (`structure_ligand_*`,
  `structure_pdb_afdb_*`, swissprot variants, transform_fn variants),
  model defaults (`model/leflur*.yaml`, `model/latent_generator*.yaml`),
  generation parameter sub-configs (`generation/`), and 10 dev/training
  experiment entries (`experiment/train_*`, `experiment/test_gen_ume`).
  They live on disk for local research workflows and are reachable via
  `paths=internal`, but are gitignored on the publication branch so the
  shipped Hydra surface is inference-only:
  `paths/`, `experiment/{generate_*, autoencode*, esmfold_baseline}.yaml`,
  `experiment/{research,legacy}/`. The 11 Hydra configs the branch had
  *modified* on top of main (training callbacks + training data configs +
  `experiment/train_gen_ume.yaml`) were reverted to their `origin/main`
  state in lockstep so the publication PR no longer touches them.
  Hydra configs that existed in `origin/main` unchanged — including
  every other `model/*.yaml`, `data/*.yaml`, `callbacks/*.yaml`,
  `evaluation/`, `experiment/ume-2/`, top-level `train.yaml` /
  `finetune.yaml` / `intervene*.yaml` / `evaluate.yaml` / etc. — remain
  tracked exactly as they are in main.
- **Tier-2 research Hydra configs** at
  `src/lobster/hydra_config/experiment/research/` (48 YAMLs). These are
  internal-only research sweeps (binder design, denovo CAMEO / RGN2 /
  multiflow / PoseBusters variants, 90M / 450M / 750M model size
  ablations, FSQ vs SLQ codec studies, etc.) that exercise
  `${paths.checkpoints.research_*}` interpolations only available under
  the internal overlay. They live on disk for local research workflows
  and are gitignored on the publication branch. The
  `test_research_tier_composes` smoke (3 representative compose checks)
  was dropped alongside — the publication CI has no signal to recover
  from configs it doesn't ship. The Tier-1 inference compose tests
  (`test_paths_internal_resolves` / `test_paths_public_resolves`) and
  tier invariants (`test_tier1_uses_canonical_checkpoint`,
  `test_no_internal_paths_in_publication_configs`) all remain green.
- **Optional research-only `_diffusion_loss.py`** at
  `src/lobster/model/losses/_diffusion_loss.py`. The
  `DiffusionLoss` / `SimpleMLPAdaLN` / `create_diffusion_loss` exports
  were only reachable via the (research-only)
  `use_diffusion_loss_structure=True` Lightning-module flag. No
  canonical LeFlur checkpoint (`leflur-base`, `leflur-ted`,
  `leflur-pl`) sets this flag — the protein-ligand checkpoint that
  drives the published PoseBusters numbers is the `no_geom_medium`
  variant, which uses the standard discrete flow-matching loss for
  structure tokens. `lobster/model/losses/__init__.py` is reverted to
  its `origin/main` state (no diffusion imports); the conditional
  import inside `LeFlurProteinLigandLightningModule` now targets the
  underscore module directly and raises a clear `ImportError` with
  recovery instructions when the optional file is absent. The file
  itself stays on disk for internal experiments.
