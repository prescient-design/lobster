"""Shared runners for the three ligand-conditioned LeFlur tasks.

Used exclusively by the Hydra entry point :mod:`lobster.cmdline.generate`
(modes ``ligand_conditioned``, ``protein_ligand_forward_folding``,
``protein_ligand_inverse_folding``) -- there is no separate argparse CLI
for these tasks on the publication branch. Invoke via::

    lobster_generate --config-name experiment/generate_ligand_conditioned
    lobster_generate --config-name experiment/generate_ligand_conditioned_forward_folding
    lobster_generate --config-name experiment/generate_ligand_conditioned_inverse_folding

Three runners are exposed, each pairing a frozen-shape knob bundle
(``*RunConfig``) with a thin wrapper around the corresponding evaluator
under :mod:`lobster.metrics`:

- :func:`run_ligand_conditioned_generation` →
  :class:`lobster.metrics.protein_ligand.ligand_conditioned_generation.LigandConditionedProteinGenerationEvaluator`.
- :func:`run_protein_ligand_forward_folding` →
  :class:`lobster.metrics.protein_ligand.forward_folding.ProteinLigandForwardFoldingEvaluator`.
- :func:`run_protein_ligand_inverse_folding` →
  :class:`lobster.metrics.protein_ligand.inverse_folding.ProteinLigandInverseFoldingEvaluator`.

Each runner is intentionally **pure orchestration**: no argument parsing,
no model loading, no ESMFold loading — those live in the calling entry
points so the Hydra and argparse paths can stay independent in how they
collect their parameters while agreeing on a single source of truth for
the evaluation pipeline.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# --- Knob bundle ----------------------------------------------------------


@dataclass
class LigandConditionedRunConfig:
    """Every parameter the underlying evaluator needs (plus output knobs).

    Matches the public surface of
    :class:`LigandConditionedProteinGenerationEvaluator` so the runner is a
    1:1 forwarder. Defaults mirror the values committed in the conference
    benchmark + the existing argparse CLI so behaviour stays unchanged.
    """

    # Data
    data_dir: str
    output_dir: str
    output_csv_name: str = "ligand_cond_protein_gen_results.csv"
    structure_path: str | None = None

    # Sampling
    length: int = 100
    pocket_distance_threshold: float = 5.0
    num_samples: int | None = None
    num_designs: int = 10
    nsteps: int = 200
    device: str = "cuda"
    max_length: int = 512

    # Temperatures / stochasticities
    temperature_seq: float = 0.153
    temperature_struc: float = 0.05
    stochasticity_seq: int = 20
    stochasticity_struc: int = 20
    temperature_ligand: float = 0.1
    stochasticity_ligand: int = 5

    # Schedules + context
    ligand_context_mode: str = "atom_bond_only"
    inference_schedule_seq: str = "LinearInferenceSchedule"
    inference_schedule_struc: str = "PowerInferenceSchedule"
    inference_schedule_ligand_atom: str = "PowerInferenceSchedule"
    inference_schedule_ligand_struc: str = "LinearInferenceSchedule"

    # Post-processing
    save_structures: bool = False
    minimize_ligand: bool = False
    minimize_mode: str = "bonds_and_angles"
    force_field: str = "MMFF94"
    minimize_steps: int = 500

    # External validators
    use_protenix: bool = False
    use_boltz: bool = False
    raw_data_dir: str | None = None

    # Misc
    seed: int = 1234
    skip_esmfold: bool = False

    # Free-form extras forwarded verbatim to the evaluator (for forward
    # compatibility — Hydra configs can drop new keys here without bumping
    # this dataclass).
    extras: dict[str, Any] = field(default_factory=dict)


# --- Runner ---------------------------------------------------------------


def run_ligand_conditioned_generation(
    model,
    config: LigandConditionedRunConfig,
    *,
    plm_fold=None,
) -> dict[str, Any]:
    """Run the ligand-conditioned self-consistency evaluation.

    Parameters
    ----------
    model
        A ``LeFlurProteinLigandLightningModule`` already moved to
        ``config.device``.
    config
        Knob bundle. See :class:`LigandConditionedRunConfig`.
    plm_fold
        Optional pre-loaded :class:`LobsterPLMFold` for ESMFold validation.
        Ignored when ``config.skip_esmfold`` is true.

    Returns
    -------
    dict
        Evaluator output (``results_df``, ``summary``, ...). Also writes
        a CSV to ``config.output_dir / config.output_csv_name`` and (when
        ``config.structure_path`` is set) per-design PDB files.
    """
    from lobster.metrics.protein_ligand.ligand_conditioned_generation import (
        LigandConditionedProteinGenerationEvaluator,
    )

    data_dir = Path(config.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Ligand-conditioned data_dir does not exist: {data_dir}")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.structure_path:
        Path(config.structure_path).mkdir(parents=True, exist_ok=True)

    logger.info(
        "Ligand-conditioned generation: data_dir=%s, length=%d, num_designs=%d, nsteps=%d, num_samples=%s",
        data_dir,
        config.length,
        config.num_designs,
        config.nsteps,
        config.num_samples,
    )

    evaluator = LigandConditionedProteinGenerationEvaluator(
        data_dir=str(data_dir),
        length=config.length,
        pocket_distance_threshold=config.pocket_distance_threshold,
        num_samples=config.num_samples,
        num_designs=config.num_designs,
        nsteps=config.nsteps,
        device=config.device,
        max_length=config.max_length,
        temperature_seq=config.temperature_seq,
        temperature_struc=config.temperature_struc,
        stochasticity_seq=config.stochasticity_seq,
        stochasticity_struc=config.stochasticity_struc,
        temperature_ligand=config.temperature_ligand,
        stochasticity_ligand=config.stochasticity_ligand,
        ligand_context_mode=config.ligand_context_mode,
        inference_schedule_seq=config.inference_schedule_seq,
        inference_schedule_struc=config.inference_schedule_struc,
        inference_schedule_ligand_atom=config.inference_schedule_ligand_atom,
        inference_schedule_ligand_struc=config.inference_schedule_ligand_struc,
        save_structures=config.save_structures,
        minimize_ligand=config.minimize_ligand,
        minimize_mode=config.minimize_mode,
        force_field=config.force_field,
        minimize_steps=config.minimize_steps,
        plm_fold=plm_fold if not config.skip_esmfold else None,
        use_protenix=config.use_protenix,
        use_boltz=config.use_boltz,
        raw_data_dir=config.raw_data_dir,
        **config.extras,
    )

    samples = evaluator.load_test_set()
    logger.info("Loaded %d ligand-conditioned samples", len(samples))

    results = evaluator.evaluate(model, samples, structure_path=config.structure_path)

    # Write CSV. When structure_path is set we co-locate so the artefacts
    # stay in a single directory (matches old evaluator behaviour).
    csv_target_dir = Path(config.structure_path) if config.structure_path else output_dir
    csv_path = csv_target_dir / os.path.basename(config.output_csv_name)
    results["results_df"].to_csv(csv_path, index=False)
    logger.info("Wrote ligand-conditioned results to %s", csv_path)
    return results


# --- Protein-ligand forward folding ---------------------------------------


@dataclass
class ProteinLigandForwardFoldingRunConfig:
    """Knob bundle for protein-ligand forward folding.

    Mirrors :class:`ProteinLigandForwardFoldingEvaluator` so the runner is a
    1:1 forwarder. Defaults match the conference benchmark sweep.
    """

    data_dir: str
    output_dir: str
    output_csv_name: str = "protein_ligand_forward_folding_results.csv"
    structure_path: str | None = None

    pocket_distance_threshold: float = 5.0
    num_samples: int | None = None
    nsteps: int = 200
    device: str = "cuda"
    max_length: int = 512
    max_protein_length: int = 512

    temperature_seq: float = 0.5
    temperature_struc: float = 0.5
    stochasticity_seq: int = 20
    stochasticity_struc: int = 20
    temperature_ligand: float = 0.5
    stochasticity_ligand: int = 20

    ligand_context_mode: str = "atom_bond_only"
    inference_schedule_seq: str = "LinearInferenceSchedule"
    inference_schedule_struc: str = "PowerInferenceSchedule"
    inference_schedule_ligand_atom: str | None = None
    inference_schedule_ligand_struc: str | None = None

    save_structures: bool = False
    save_gt_structure: bool = False
    minimize_ligand: bool = False
    minimize_mode: str = "bonds_and_angles"
    force_field: str = "MMFF94"
    minimize_steps: int = 500

    # Best-of-N
    num_predictions: int = 1
    best_of_n_metric: str = "plddt"
    save_all_predictions: bool = False

    try_reflection: bool = False

    use_protenix: bool = False
    use_boltz: bool = False
    raw_data_dir: str | None = None

    seed: int = 1234
    skip_esmfold: bool = False

    extras: dict[str, Any] = field(default_factory=dict)


def run_protein_ligand_forward_folding(
    model,
    config: ProteinLigandForwardFoldingRunConfig,
    *,
    plm_fold=None,
) -> dict[str, Any]:
    """Run protein-ligand forward folding (sequence -> structure with ligand context)."""
    from lobster.metrics.protein_ligand.forward_folding import (
        ProteinLigandForwardFoldingEvaluator,
    )

    data_dir = Path(config.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Protein-ligand forward folding data_dir does not exist: {data_dir}")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if config.structure_path:
        Path(config.structure_path).mkdir(parents=True, exist_ok=True)

    logger.info(
        "Protein-ligand forward folding: data_dir=%s, nsteps=%d, num_samples=%s",
        data_dir,
        config.nsteps,
        config.num_samples,
    )

    evaluator = ProteinLigandForwardFoldingEvaluator(
        data_dir=str(data_dir),
        pocket_distance_threshold=config.pocket_distance_threshold,
        num_samples=config.num_samples,
        nsteps=config.nsteps,
        device=config.device,
        max_length=config.max_length,
        max_protein_length=config.max_protein_length,
        temperature_seq=config.temperature_seq,
        temperature_struc=config.temperature_struc,
        save_structures=config.save_structures,
        save_gt_structure=config.save_gt_structure,
        minimize_ligand=config.minimize_ligand,
        minimize_mode=config.minimize_mode,
        force_field=config.force_field,
        minimize_steps=config.minimize_steps,
        stochasticity_seq=config.stochasticity_seq,
        stochasticity_struc=config.stochasticity_struc,
        temperature_ligand=config.temperature_ligand,
        stochasticity_ligand=config.stochasticity_ligand,
        ligand_context_mode=config.ligand_context_mode,
        inference_schedule_seq=config.inference_schedule_seq,
        inference_schedule_struc=config.inference_schedule_struc,
        inference_schedule_ligand_atom=config.inference_schedule_ligand_atom,
        inference_schedule_ligand_struc=config.inference_schedule_ligand_struc,
        num_predictions=config.num_predictions,
        best_of_n_metric=config.best_of_n_metric,
        save_all_predictions=config.save_all_predictions,
        try_reflection=config.try_reflection,
        use_protenix=config.use_protenix,
        use_boltz=config.use_boltz,
        raw_data_dir=config.raw_data_dir,
        **config.extras,
    )

    samples = evaluator.load_test_set()
    logger.info("Loaded %d forward-folding samples", len(samples))

    results = evaluator.evaluate(model, samples, structure_path=config.structure_path)

    csv_target_dir = Path(config.structure_path) if config.structure_path else output_dir
    csv_path = csv_target_dir / os.path.basename(config.output_csv_name)
    results["results_df"].to_csv(csv_path, index=False)
    logger.info("Wrote forward-folding results to %s", csv_path)
    return results


# --- Protein-ligand inverse folding ---------------------------------------


@dataclass
class ProteinLigandInverseFoldingRunConfig:
    """Knob bundle for protein-ligand inverse folding."""

    data_dir: str
    output_dir: str
    output_csv_name: str = "protein_ligand_inverse_folding_results.csv"
    structure_path: str | None = None

    pocket_distance_threshold: float = 5.0
    num_samples: int | None = None
    nsteps: int = 100
    device: str = "cuda"
    max_length: int = 512
    max_protein_length: int = 512

    temperature_seq: float = 0.5
    temperature_struc: float = 0.5
    stochasticity_seq: int = 20
    stochasticity_struc: int = 20
    temperature_ligand: float = 0.5
    stochasticity_ligand: int = 20

    inference_schedule_seq: str = "LogInferenceSchedule"
    inference_schedule_struc: str = "LinearInferenceSchedule"
    inference_schedule_ligand_atom: str | None = None
    inference_schedule_ligand_struc: str | None = None

    decode_structure: bool = False
    save_gt_structure: bool = False
    save_reconstructed_input: bool = False

    minimize_ligand: bool = False
    minimize_mode: str = "bonds_and_angles"
    force_field: str = "MMFF94"
    minimize_steps: int = 500

    use_se3_augmentation: bool = False
    se3_translation_scale: float = 1.0

    use_esmfold: bool = False
    use_protenix: bool = False
    use_boltz: bool = False
    raw_data_dir: str | None = None

    seed: int = 1234

    extras: dict[str, Any] = field(default_factory=dict)


def run_protein_ligand_inverse_folding(
    model,
    config: ProteinLigandInverseFoldingRunConfig,
    *,
    plm_fold=None,
) -> dict[str, Any]:
    """Run protein-ligand inverse folding (structure -> sequence with ligand context)."""
    from lobster.metrics.protein_ligand.inverse_folding import (
        ProteinLigandInverseFoldingEvaluator,
    )

    data_dir = Path(config.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Protein-ligand inverse folding data_dir does not exist: {data_dir}")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if config.structure_path:
        Path(config.structure_path).mkdir(parents=True, exist_ok=True)

    logger.info(
        "Protein-ligand inverse folding: data_dir=%s, nsteps=%d, num_samples=%s",
        data_dir,
        config.nsteps,
        config.num_samples,
    )

    evaluator = ProteinLigandInverseFoldingEvaluator(
        data_dir=str(data_dir),
        pocket_distance_threshold=config.pocket_distance_threshold,
        num_samples=config.num_samples,
        nsteps=config.nsteps,
        device=config.device,
        max_length=config.max_length,
        decode_structure=config.decode_structure,
        save_gt_structure=config.save_gt_structure,
        minimize_ligand=config.minimize_ligand,
        minimize_mode=config.minimize_mode,
        force_field=config.force_field,
        minimize_steps=config.minimize_steps,
        save_reconstructed_input=config.save_reconstructed_input,
        use_se3_augmentation=config.use_se3_augmentation,
        se3_translation_scale=config.se3_translation_scale,
        temperature_seq=config.temperature_seq,
        temperature_struc=config.temperature_struc,
        stochasticity_seq=config.stochasticity_seq,
        stochasticity_struc=config.stochasticity_struc,
        temperature_ligand=config.temperature_ligand,
        stochasticity_ligand=config.stochasticity_ligand,
        inference_schedule_seq=config.inference_schedule_seq,
        inference_schedule_struc=config.inference_schedule_struc,
        inference_schedule_ligand_atom=config.inference_schedule_ligand_atom,
        inference_schedule_ligand_struc=config.inference_schedule_ligand_struc,
        use_esmfold=config.use_esmfold,
        plm_fold=plm_fold if config.use_esmfold else None,
        max_protein_length=config.max_protein_length,
        use_protenix=config.use_protenix,
        use_boltz=config.use_boltz,
        raw_data_dir=config.raw_data_dir,
        **config.extras,
    )

    samples = evaluator.load_test_set()
    logger.info("Loaded %d inverse-folding samples", len(samples))

    results = evaluator.evaluate(model, samples, structure_path=config.structure_path)

    csv_target_dir = Path(config.structure_path) if config.structure_path else output_dir
    csv_path = csv_target_dir / os.path.basename(config.output_csv_name)
    results["results_df"].to_csv(csv_path, index=False)
    logger.info("Wrote inverse-folding results to %s", csv_path)
    return results


__all__ = [
    "LigandConditionedRunConfig",
    "ProteinLigandForwardFoldingRunConfig",
    "ProteinLigandInverseFoldingRunConfig",
    "run_ligand_conditioned_generation",
    "run_protein_ligand_forward_folding",
    "run_protein_ligand_inverse_folding",
]
