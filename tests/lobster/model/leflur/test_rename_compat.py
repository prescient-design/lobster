"""Verify the gen_ume → leflur rename did not break checkpoint loading.

The historical checkpoints referenced in the conference benchmark and
pseudo-likelihood scoring plans were trained when the Lightning module lived at
``lobster.model.gen_ume._gen_ume_sequence_structure_encoder_lightning_module``
under the class name ``UMESequenceStructureEncoderLightningModule`` (and PL
analogue ``ProteinLigandEncoderLightningModule``). After the rename the new
canonical home is ``lobster.model.leflur`` with the ``LeFlur*`` class names.

Pickle resolution risk: PyTorch Lightning checkpoints store class references in
``callbacks``/``hyper_parameters`` via fully qualified module paths. The
``lobster.model.gen_ume`` package therefore must remain importable and must
re-export the renamed classes under their historical names. The deprecation
shim in :mod:`lobster.model.gen_ume.__init__` provides that.

The checkpoints exercised here are the three canonical ones referenced from
``.cursor/plans/conference_benchmark_comparison_9b71ca71.plan.md`` and
``.cursor/plans/gen-ume_pseudo-likelihood_scoring_face7fdf.plan.md``:

- ``BASE`` — ``gen_ume_denovo_last_ckpt_2026-03-11T12-11-53`` (LEFLUR-P)
- ``TED``  — ``gen_ume_denovo_ted_cath_ss_balanced_ckpt_2026-03-18T12-20-59``
  (LEFLUR-T)
- ``PL``   — ``gen_ume_protein_ligand_medium`` (canonical protein-ligand)

Each test is skipped automatically when the corresponding local checkpoint file
is not present so the suite still runs in CI / HuggingFace-only environments.
Phase 4 of the LeFlur release plan replaces these local paths with
``hf://Sidney-Lisanza/leflur/...`` URIs resolved via
``lobster.model.leflur.checkpoints.resolve_checkpoint`` and the smoke tests
will move with them.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
import torch

# Canonical local paths for the historical conference-benchmark checkpoints.
# These are the actual files the eval dirs referenced (the eval-dir suffix
# ``..._ckpt_<YYYY-MM-DDTHH-MM-SS>`` is the eval-launch timestamp, not the
# checkpoint timestamp). Resolved by inspecting the symlink targets and the
# ``ckpt_path`` hyperparameter on each checkpoint.
BASE_CKPT = Path(
    "/cv/scratch/u/lisanzas/gen_ume_denovo/runs/2026-03-08T17-09-23/last.ckpt"
)
TED_CKPT = Path(
    "/cv/scratch/u/lisanzas/gen_ume_denovo_ted_cath/runs/2026-03-14T15-41-36/last.ckpt"
)
PL_CKPT = Path(
    "/cv/scratch/u/lisanzas/gen_ume_protein_ligand_medium/runs"
    "/2026-02-11T19-45-30/epoch=278-step=40057-val_loss=1.6365.ckpt"
)


def _skip_if_missing(path: Path) -> None:
    if not path.exists():
        pytest.skip(f"local checkpoint not present: {path}")


def _load_raw(path: Path) -> dict:
    return torch.load(str(path), map_location="cpu", weights_only=False)


def test_leflur_module_exports_new_class_names() -> None:
    """The new package exposes the LeFlur* names."""
    from lobster.model import leflur

    for name in (
        "LeFlurSequenceStructureEncoderModule",
        "LeFlurSequenceStructureEncoderLightningModule",
        "LeFlurProteinLigandEncoderModule",
        "LeFlurProteinLigandLightningModule",
    ):
        assert hasattr(leflur, name), f"lobster.model.leflur missing {name}"


def test_gen_ume_deprecation_shim_aliases_old_names() -> None:
    """Old historical names still resolve via the deprecation shim, and emit
    a DeprecationWarning at import time."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # Force a fresh import so the shim's warnings.warn() actually fires.
        import importlib
        import sys

        sys.modules.pop("lobster.model.gen_ume", None)
        gen_ume = importlib.import_module("lobster.model.gen_ume")

    from lobster.model.leflur import (
        LeFlurProteinLigandEncoderModule,
        LeFlurProteinLigandLightningModule,
        LeFlurSequenceStructureEncoderLightningModule,
        LeFlurSequenceStructureEncoderModule,
    )

    assert gen_ume.UMESequenceStructureEncoderModule is LeFlurSequenceStructureEncoderModule
    assert (
        gen_ume.UMESequenceStructureEncoderLightningModule
        is LeFlurSequenceStructureEncoderLightningModule
    )
    assert gen_ume.ProteinLigandEncoderModule is LeFlurProteinLigandEncoderModule
    assert gen_ume.ProteinLigandEncoderLightningModule is LeFlurProteinLigandLightningModule

    assert any(
        issubclass(w.category, DeprecationWarning)
        and "gen_ume is deprecated" in str(w.message)
        for w in caught
    ), "expected DeprecationWarning from lobster.model.gen_ume"


@pytest.mark.parametrize("ckpt_path", [BASE_CKPT, TED_CKPT], ids=["base", "ted"])
def test_protein_only_checkpoint_state_dict_matches_renamed_module(
    ckpt_path: Path,
) -> None:
    """Loading the raw checkpoint and asserting the saved ``state_dict`` keys
    match the module instantiated under its new ``LeFlur*`` class.

    This avoids depending on the Latent Generator and warm-start file paths
    that the Lightning module's ``__init__`` would normally consume, while
    still proving the checkpoint's binary on-disk layout is compatible with
    the renamed Python code.
    """
    _skip_if_missing(ckpt_path)
    ck = _load_raw(ckpt_path)

    assert "state_dict" in ck
    assert "hyper_parameters" in ck
    hp = ck["hyper_parameters"]
    assert isinstance(hp, dict)
    # Hyperparameter dict is plain (no Hydra _target_) — load_from_checkpoint
    # will instantiate by passing **hp into LeFlurSequenceStructureEncoderLightningModule.
    assert "_target_" not in hp
    assert "latent_generator_model_name" in hp
    assert "interpolant" in hp

    # The state_dict must use the same key prefixes the renamed
    # LeFlurSequenceStructureEncoderLightningModule exposes (we did NOT rename
    # any attribute, only the class itself, so its keys are stable):
    # ``encoder.*`` (LeFlurSequenceStructureEncoderModule), ``quantizer.*``,
    # ``strucure_encoder.*`` [sic — historical typo, retained for ckpt compat],
    # ``decoder_factory.*``, ``loss_factory.*``.
    keys = set(ck["state_dict"].keys())
    assert any(k.startswith("encoder.") for k in keys), "missing encoder.* state"
    assert any(k.startswith("quantizer.") for k in keys), "missing quantizer.* state"
    assert any(
        k.startswith("strucure_encoder.") for k in keys
    ), "missing strucure_encoder.* state (sic — historical typo)"


def test_protein_ligand_checkpoint_state_dict_matches_renamed_module() -> None:
    """Same check for the protein-ligand canonical checkpoint."""
    _skip_if_missing(PL_CKPT)
    ck = _load_raw(PL_CKPT)

    assert "state_dict" in ck
    assert "hyper_parameters" in ck
    hp = ck["hyper_parameters"]
    assert isinstance(hp, dict)
    assert "_target_" not in hp
    assert "latent_generator_model_name" in hp
    # PL-specific kwargs that the renamed LeFlurProteinLigandLightningModule
    # must still accept.
    assert "ligand_atom_vocab_size" in hp
    assert "ligand_mask_token_id" in hp
    assert "bond_loss_weight" in hp

    # The protein-ligand checkpoint stores the LG-side structure encoder
    # under the ``structure_encoder.*`` prefix (no double-``s`` typo here)
    # and the ligand-aware encoder under ``encoder.*``. ``quantizer.*`` and
    # ``decoder_factory.*`` belong to the LG. Critically, the
    # ``encoder.ligand_atom_output``/``encoder.ligand_structure_output``
    # heads must be present — they are the ligand-specific outputs the
    # renamed ``LeFlurProteinLigandLightningModule`` defines.
    keys = set(ck["state_dict"].keys())
    prefixes = {k.split(".", 1)[0] for k in keys}
    assert "encoder" in prefixes, f"missing encoder.* state (got {sorted(prefixes)})"
    assert (
        "structure_encoder" in prefixes
    ), f"missing structure_encoder.* state (got {sorted(prefixes)})"
    assert any(
        k.startswith("encoder.ligand_atom_output") for k in keys
    ), "missing ligand_atom_output head — PL ckpt is incompatible with renamed module"
    assert any(
        k.startswith("encoder.ligand_structure_output") for k in keys
    ), "missing ligand_structure_output head — PL ckpt is incompatible with renamed module"


@pytest.mark.slow
@pytest.mark.parametrize(
    "ckpt_path,classname",
    [
        (TED_CKPT, "LeFlurSequenceStructureEncoderLightningModule"),
    ],
    ids=["ted"],
)
def test_protein_only_checkpoint_loads_via_lightning(
    ckpt_path: Path, classname: str
) -> None:
    """End-to-end ``load_from_checkpoint`` test on the TED checkpoint.

    BASE is intentionally excluded from this end-to-end variant because its
    ``hyper_parameters['ckpt_path']`` points at a sibling warm-start
    checkpoint and the constructor will try to read that file too; the
    state-dict-keys test above already covers BASE structurally without
    pulling in the warm-start dependency. TED's ``ckpt_path`` is ``None``
    so the constructor only needs the Latent Generator (HF-hosted
    ``LG full attention``) and the local file under test.

    Marked ``slow`` because it downloads the LG checkpoint from HuggingFace
    and instantiates the full module; CI can opt in via ``pytest -m slow``.
    """
    _skip_if_missing(ckpt_path)

    from lobster.model.leflur import LeFlurSequenceStructureEncoderLightningModule

    module = LeFlurSequenceStructureEncoderLightningModule.load_from_checkpoint(
        str(ckpt_path),
        map_location="cpu",
        strict=True,
    )
    assert isinstance(module, LeFlurSequenceStructureEncoderLightningModule)
    assert type(module).__name__ == classname

    # Sanity: the renamed sub-modules expose the historical names so
    # downstream scripts (score_leflur_pll.py, generate.py, …) keep working.
    assert hasattr(module, "encoder")
    assert hasattr(module, "quantizer")
    assert hasattr(module, "interpolant_seq")
    assert hasattr(module, "interpolant_struc")
