import logging
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from loguru import logger

from lobster.metrics import (
    MetricsPlotter,
    MetricsCSVWriter,
)
from lobster.model import LobsterPLMFold
from lobster.model.leflur import resolve_checkpoint
from lobster.cmdline._ligand_conditioned_runner import (
    LigandConditionedRunConfig,
    ProteinLigandForwardFoldingRunConfig,
    ProteinLigandInverseFoldingRunConfig,
    run_ligand_conditioned_generation,
    run_protein_ligand_forward_folding,
    run_protein_ligand_inverse_folding,
)

# Per-mode generation pipelines. See lobster.cmdline.generate_modes.
from lobster.cmdline.generate_modes._binders import _generate_binders
from lobster.cmdline.generate_modes._forward_folding import _generate_forward_folding
from lobster.cmdline.generate_modes._inpainting import _generate_inpainting
from lobster.cmdline.generate_modes._inverse_folding import _generate_inverse_folding
from lobster.cmdline.generate_modes._unconditional import _generate_unconditional

# Set up logging
logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="../hydra_config", config_name="experiment/generate_unconditional")
def generate(cfg: DictConfig) -> None:
    """Hydra entry point for the ``lobster_generate`` console script.

    Dispatches on ``cfg.generation.mode`` to one of the per-mode helpers in
    this module. Seven modes are wired in:

    * ``unconditional`` — sample novel sequences + structures.
    * ``forward_folding`` — sequence → structure.
    * ``inverse_folding`` — structure → sequence.
    * ``inpainting`` — complete a partial protein.
    * ``ligand_conditioned`` — de-novo design conditioned on a ligand pocket.
    * ``protein_ligand_forward_folding`` — sequence + ligand → structure.
    * ``protein_ligand_inverse_folding`` — structure + ligand → sequence.

    Checkpoint paths in ``cfg.model.ckpt_path`` may be a short name from
    :data:`lobster.model.leflur.KNOWN_CHECKPOINTS` (e.g. ``leflur-ted``), an
    ``hf://`` URI, an ``https://huggingface.co/...`` URL, or a local path —
    :func:`lobster.model.leflur.resolve_checkpoint` materialises them to a
    concrete file before loading.

    Parameters
    ----------
    cfg
        Hydra-composed config. Tier-1 entry-point configs live under
        ``src/lobster/hydra_config/experiment/`` (use
        ``--config-name experiment/<config>``).

    See Also
    --------
    docs/leflur/cli.md
        User-facing reference for the CLI surface.
    docs/leflur/quickstart.md
        Five-minute walkthroughs of each inference mode.
    """
    logger.info("Starting LeFlur structure generation")
    logger.info("Config:\n %s", OmegaConf.to_yaml(cfg))

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Set seed for reproducibility
    if cfg.get("seed"):
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cfg.seed)
        logger.info(f"Set random seed to {cfg.seed}")

    # Load model
    logger.info("Loading LeFlur model...")
    if hasattr(cfg.model, "ckpt_path") and cfg.model.ckpt_path is not None:
        # Accept short names ("leflur-ted"), hf:// URIs, https://huggingface.co/...
        # URLs, and local paths. See `lobster.model.leflur.checkpoints`.
        resolved_ckpt = resolve_checkpoint(cfg.model.ckpt_path)
        logger.info(
            f"Loading model from checkpoint: {cfg.model.ckpt_path!r} "
            f"-> {resolved_ckpt}"
        )
        model_cls = hydra.utils.get_class(cfg.model._target_)
        model = model_cls.load_from_checkpoint(str(resolved_ckpt))
    else:
        logger.info("Instantiating fresh model (no checkpoint provided)")
        model = hydra.utils.instantiate(cfg.model)

    model.to(device)
    model.eval()
    logger.info("✓ Model loaded successfully")

    # Initialize ESMFold if requested
    plm_fold = None
    if cfg.generation.get("use_esmfold", False):
        logger.info("Loading ESMFold for structure validation...")

        plm_fold = LobsterPLMFold(model_name="esmfold_v1", max_length=cfg.generation.get("max_length", 512))
        plm_fold.to(device)
        logger.info("✓ ESMFold loaded successfully")

    # Create output directory
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Initialize CSV logging and plotting if enabled
    csv_writer = None
    plotter = None
    if cfg.generation.get("save_csv_metrics", True):
        generation_mode = cfg.generation.mode
        csv_writer = MetricsCSVWriter(output_dir, generation_mode, resume=cfg.generation.get("resume", False))
        logger.info(f"CSV metrics logging enabled for {generation_mode} mode")

        # Initialize plotter if plotting is enabled
        if cfg.generation.get("create_plots", True):
            plotter = MetricsPlotter(output_dir, generation_mode)
            logger.info(f"Plotting enabled for {generation_mode} mode")

    # Generate structures
    generation_mode = cfg.generation.mode
    logger.info(f"Generation mode: {generation_mode}")

    if generation_mode == "unconditional":
        _generate_unconditional(model, cfg, device, output_dir, plm_fold, csv_writer, plotter)
    elif generation_mode == "inverse_folding":
        _generate_inverse_folding(model, cfg, device, output_dir, plm_fold, csv_writer, plotter)
    elif generation_mode == "forward_folding":
        _generate_forward_folding(model, cfg, device, output_dir, plm_fold, csv_writer, plotter)
    elif generation_mode == "inpainting":
        _generate_inpainting(model, cfg, device, output_dir, plm_fold, csv_writer, plotter)
    elif generation_mode == "binder_design":
        _generate_binders(model, cfg, device, output_dir, plm_fold, csv_writer, plotter)
    elif generation_mode == "ligand_conditioned":
        _generate_ligand_conditioned(model, cfg, device, output_dir, plm_fold)
    elif generation_mode == "protein_ligand_forward_folding":
        _generate_protein_ligand_forward_folding(model, cfg, device, output_dir, plm_fold)
    elif generation_mode == "protein_ligand_inverse_folding":
        _generate_protein_ligand_inverse_folding(model, cfg, device, output_dir, plm_fold)
    else:
        raise ValueError(f"Unknown generation mode: {generation_mode}")

    logger.info("Generation completed successfully!")


def _generate_ligand_conditioned(
    model,
    cfg: DictConfig,
    device,
    output_dir: Path,
    plm_fold=None,
) -> None:
    """Hydra-driven dispatch into the shared ligand-conditioned runner.

    Reads ``cfg.generation.ligand_conditioned`` (a free-form sub-tree that
    maps 1:1 to :class:`LigandConditionedRunConfig`) and forwards. Falls
    back to ``cfg.generation`` keys when no nested ``ligand_conditioned``
    sub-tree is provided so the same flat config used by the smoke tests
    keeps working.
    """
    from lobster.model.leflur import LeFlurProteinLigandLightningModule

    if not isinstance(model, LeFlurProteinLigandLightningModule):
        raise TypeError(
            f"generation.mode=ligand_conditioned requires a "
            f"LeFlurProteinLigandLightningModule, got {type(model).__name__}. "
            f"Pass `model.ckpt_path=leflur-pl` (or another PL checkpoint)."
        )

    raw: dict = {}
    if cfg.generation.get("ligand_conditioned") is not None:
        raw.update(OmegaConf.to_container(cfg.generation.ligand_conditioned, resolve=True))

    # Top-level shortcuts so common knobs work without nesting.
    for key in (
        "length",
        "num_designs",
        "nsteps",
        "num_samples",
        "seed",
        "save_structures",
        "minimize_ligand",
        "minimize_mode",
        "force_field",
        "minimize_steps",
        "ligand_context_mode",
        "skip_esmfold",
        "use_protenix",
        "use_boltz",
        "raw_data_dir",
    ):
        if key in cfg.generation and key not in raw:
            raw[key] = cfg.generation.get(key)

    if "data_dir" not in raw:
        raw["data_dir"] = cfg.generation.get(
            "data_dir"
        ) or cfg.generation.get("input_structures")
    if raw.get("data_dir") is None:
        raise ValueError(
            "ligand_conditioned mode requires generation.data_dir (or "
            "generation.ligand_conditioned.data_dir) pointing at a "
            "directory of *_ligand.pt fixtures."
        )

    raw.setdefault("output_dir", str(output_dir))
    raw.setdefault("device", str(device))
    raw.setdefault("max_length", cfg.generation.get("max_length", 512))

    config = LigandConditionedRunConfig(**raw)
    results = run_ligand_conditioned_generation(model, config, plm_fold=plm_fold)
    logger.info(
        "Ligand-conditioned summary: scTM=%.3f scRMSD=%.2f over %d ligands",
        results["summary"].get("mean_scTM", float("nan")),
        results["summary"].get("mean_scRMSD", float("nan")),
        results["summary"].get("n_ligands", 0),
    )


def _collect_pl_task_kwargs(cfg: DictConfig, sub_key: str, top_level_keys: tuple[str, ...]) -> dict:
    """Common knob-collection logic for the three PL Hydra dispatch helpers.

    Reads ``cfg.generation.<sub_key>`` (if present, as a nested OmegaConf
    block) and merges with top-level ``cfg.generation`` shortcuts so a
    flat config without nesting still works.
    """
    raw: dict = {}
    sub = cfg.generation.get(sub_key)
    if sub is not None:
        raw.update(OmegaConf.to_container(sub, resolve=True))
    for key in top_level_keys:
        if key in cfg.generation and key not in raw:
            raw[key] = cfg.generation.get(key)
    if "data_dir" not in raw:
        raw["data_dir"] = cfg.generation.get(
            "data_dir"
        ) or cfg.generation.get("input_structures")
    return raw


def _generate_protein_ligand_forward_folding(
    model,
    cfg: DictConfig,
    device,
    output_dir: Path,
    plm_fold=None,
) -> None:
    """Hydra-driven dispatch into the protein-ligand forward folding runner."""
    from lobster.model.leflur import LeFlurProteinLigandLightningModule

    if not isinstance(model, LeFlurProteinLigandLightningModule):
        raise TypeError(
            f"generation.mode=protein_ligand_forward_folding requires a "
            f"LeFlurProteinLigandLightningModule, got {type(model).__name__}."
        )

    raw = _collect_pl_task_kwargs(
        cfg,
        sub_key="protein_ligand_forward_folding",
        top_level_keys=(
            "nsteps",
            "num_samples",
            "seed",
            "max_protein_length",
            "save_structures",
            "save_gt_structure",
            "minimize_ligand",
            "minimize_mode",
            "force_field",
            "minimize_steps",
            "ligand_context_mode",
            "skip_esmfold",
            "use_protenix",
            "use_boltz",
            "raw_data_dir",
            "num_predictions",
            "best_of_n_metric",
            "save_all_predictions",
            "try_reflection",
            "temperature_seq",
            "temperature_struc",
            "stochasticity_seq",
            "stochasticity_struc",
            "temperature_ligand",
            "stochasticity_ligand",
            "inference_schedule_seq",
            "inference_schedule_struc",
            "inference_schedule_ligand_atom",
            "inference_schedule_ligand_struc",
            "pocket_distance_threshold",
        ),
    )
    if raw.get("data_dir") is None:
        raise ValueError(
            "protein_ligand_forward_folding mode requires generation.data_dir "
            "pointing at the test fixtures."
        )
    raw.setdefault("output_dir", str(output_dir))
    raw.setdefault("device", str(device))
    raw.setdefault("max_length", cfg.generation.get("max_length", 512))

    config = ProteinLigandForwardFoldingRunConfig(**raw)
    results = run_protein_ligand_forward_folding(model, config, plm_fold=plm_fold)
    logger.info(
        "PL forward folding summary: TM=%.3f over %d samples",
        results["summary"].get("mean_tm", float("nan")),
        results["summary"].get("n_samples", 0),
    )


def _generate_protein_ligand_inverse_folding(
    model,
    cfg: DictConfig,
    device,
    output_dir: Path,
    plm_fold=None,
) -> None:
    """Hydra-driven dispatch into the protein-ligand inverse folding runner."""
    from lobster.model.leflur import LeFlurProteinLigandLightningModule

    if not isinstance(model, LeFlurProteinLigandLightningModule):
        raise TypeError(
            f"generation.mode=protein_ligand_inverse_folding requires a "
            f"LeFlurProteinLigandLightningModule, got {type(model).__name__}."
        )

    raw = _collect_pl_task_kwargs(
        cfg,
        sub_key="protein_ligand_inverse_folding",
        top_level_keys=(
            "nsteps",
            "num_samples",
            "seed",
            "max_protein_length",
            "decode_structure",
            "save_gt_structure",
            "save_reconstructed_input",
            "minimize_ligand",
            "minimize_mode",
            "force_field",
            "minimize_steps",
            "use_se3_augmentation",
            "se3_translation_scale",
            "use_esmfold",
            "use_protenix",
            "use_boltz",
            "raw_data_dir",
            "temperature_seq",
            "temperature_struc",
            "stochasticity_seq",
            "stochasticity_struc",
            "temperature_ligand",
            "stochasticity_ligand",
            "inference_schedule_seq",
            "inference_schedule_struc",
            "inference_schedule_ligand_atom",
            "inference_schedule_ligand_struc",
            "pocket_distance_threshold",
        ),
    )
    if raw.get("data_dir") is None:
        raise ValueError(
            "protein_ligand_inverse_folding mode requires generation.data_dir "
            "pointing at the test fixtures."
        )
    raw.setdefault("output_dir", str(output_dir))
    raw.setdefault("device", str(device))
    raw.setdefault("max_length", cfg.generation.get("max_length", 512))

    config = ProteinLigandInverseFoldingRunConfig(**raw)
    results = run_protein_ligand_inverse_folding(model, config, plm_fold=plm_fold)
    logger.info(
        "PL inverse folding summary: %%id=%.3f over %d samples",
        results["summary"].get("mean_percent_identity", float("nan")),
        results["summary"].get("n_samples", 0),
    )


if __name__ == "__main__":
    generate()
