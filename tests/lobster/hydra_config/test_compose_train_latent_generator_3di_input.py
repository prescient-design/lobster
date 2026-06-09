"""Hydra compose smoke for the new 3Di-input training experiment.

Mirrors the dispatch pattern used by ``test_paths_overlay.py``: clears
the global Hydra state, initialises against the canonical
``src/lobster/hydra_config/`` directory, composes the experiment, and
asserts the top-level keys land.

This is a *training* config (not a publication-tier inference config), so
it lives in its own file rather than being added to the parametrised
``_COMPOSE_TARGETS`` in ``test_paths_overlay.py`` — that list is
explicitly Tier-1 / inference scope.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

REPO_ROOT = Path(__file__).resolve().parents[3]
HYDRA_ROOT = REPO_ROOT / "src" / "lobster" / "hydra_config"


_OPTIONAL_DEPS = ("torch_geometric", "rotary_embedding_torch")
_MISSING = [m for m in _OPTIONAL_DEPS if importlib.util.find_spec(m) is None]
if _MISSING:
    pytest.skip(
        f"Skipping 3Di-input train compose smoke: missing {_MISSING}",
        allow_module_level=True,
    )


@pytest.fixture(autouse=True)
def _clear_hydra():
    GlobalHydra.instance().clear()
    yield
    GlobalHydra.instance().clear()


def test_train_latent_generator_3di_input_composes() -> None:
    """The new experiment YAML composes under the default path overlay."""
    with initialize_config_dir(
        version_base=None,
        config_dir=str(HYDRA_ROOT),
        job_name="train_3di_input_smoke",
    ):
        cfg = compose(
            config_name="train",
            overrides=["experiment=train_latent_generator_3di_input"],
        )

    assert cfg.get("model") is not None, "experiment did not register a model"
    assert cfg.get("data") is not None, "experiment did not register data"
    assert cfg.get("trainer") is not None, "experiment did not register trainer"

    assert cfg.model._target_ == "lobster.model.latent_generator.tokenizer.Tokenizer3diInput", (
        f"unexpected model target: {cfg.model._target_}"
    )
    assert cfg.data._target_ == "lobster.data._coord_structure_datamodule.StructureLightningDataModule", (
        f"unexpected data target: {cfg.data._target_}"
    )

    decoder_target = cfg.model.decoder_factory.decoder_mapping.vit_decoder._target_
    assert decoder_target == "lobster.model.latent_generator.structure_decoder.ViTDecoder", (
        f"unexpected decoder target: {decoder_target}"
    )
    loss_names = set(cfg.model.loss_factory.loss_mapping.keys())
    assert {"l2_loss", "pairwise_l2_loss"}.issubset(loss_names), f"missing canonical losses: {loss_names}"

    transform_names = set(cfg.data.transform_fn.keys())
    assert {"structure_backbone_transform", "structure_3di_transform"}.issubset(transform_names), (
        f"data overlay must compose Structure3diTransform after StructureBackboneTransform; got {transform_names}"
    )

    collate_target = cfg.data.collate_fn._target_
    assert collate_target.endswith(".collate_fn_backbone") or (
        "_partial_" in cfg.data.collate_fn
        and cfg.data.collate_fn._partial_
        and collate_target.endswith(".collate_fn_backbone")
    ), (
        f"data overlay must use plain `collate_fn_backbone` (not `..._binder_target`, "
        f"which drops `3di_states`); got {collate_target}"
    )
