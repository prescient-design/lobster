"""Hydra compose smoke for the flow-matching 3Di-input training experiment.

Mirrors `test_compose_train_latent_generator_3di_input.py`. Verifies that
the new experiment composes, the new decoder/interpolant slots resolve to
the flow-matching classes, and the five Proteina-style deltas are wired
through to the right yaml keys.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

REPO_ROOT = Path(__file__).resolve().parents[3]
HYDRA_ROOT = REPO_ROOT / "src" / "lobster" / "hydra_config"


_OPTIONAL_DEPS = ("torch_geometric", "rotary_embedding_torch", "bionemo.moco")
_MISSING = [m for m in _OPTIONAL_DEPS if importlib.util.find_spec(m) is None]
if _MISSING:
    pytest.skip(
        f"Skipping 3Di-input flow train compose smoke: missing {_MISSING}",
        allow_module_level=True,
    )


@pytest.fixture(autouse=True)
def _clear_hydra():
    GlobalHydra.instance().clear()
    yield
    GlobalHydra.instance().clear()


def test_train_latent_generator_3di_input_flow_composes() -> None:
    with initialize_config_dir(
        version_base=None,
        config_dir=str(HYDRA_ROOT),
        job_name="train_3di_input_flow_smoke",
    ):
        cfg = compose(
            config_name="train",
            overrides=["experiment=train_latent_generator_3di_input_flow"],
        )

    assert cfg.get("model") is not None
    assert cfg.get("data") is not None
    assert cfg.get("trainer") is not None

    assert cfg.model._target_ == "lobster.model.latent_generator.tokenizer.Tokenizer3diInputFlow", (
        f"unexpected model target: {cfg.model._target_}"
    )

    decoder_target = cfg.model.decoder_factory.decoder_mapping.vit_decoder._target_
    assert decoder_target == "lobster.model.latent_generator.structure_decoder.ViTDecoderConditional", (
        f"unexpected decoder target: {decoder_target}"
    )

    interp_target = cfg.model.interpolant._target_
    assert interp_target.endswith("ContinuousFlowMatcher"), f"unexpected interpolant target: {interp_target}"
    assert cfg.model.interpolant.prediction_type == "data"
    assert cfg.model.interpolant.augmentation_type == "kabsch"

    # delta #2
    assert cfg.model.center_x1 is True
    assert cfg.model.center_every_step is True

    # delta #3
    assert cfg.model.decoder_factory.decoder_mapping.vit_decoder.num_registers == 4
    assert cfg.model.decoder_factory.decoder_mapping.vit_decoder.time_cond_dim == 128

    # delta #4
    assert cfg.model.sampling_mode == "ode"
    assert cfg.model.sc_scale_noise == 0.0
    assert cfg.model.sc_scale_score == 0.0

    # delta #5
    assert cfg.model.autoguidance_model is None
    assert cfg.model.autoguidance_ratio == 0.0

    # CFG
    assert cfg.model.p_uncond == 0.15
    assert cfg.model.guidance_scale == 1.0

    # Step H: random SO(3) augmentation on GT (train-only).
    assert cfg.model.random_so3_aug is True

    # Step J: Proteina-style coordinate scaling (Angstrom -> nm).
    assert cfg.model.coord_scale == 0.1

    transform_names = set(cfg.data.transform_fn.keys())
    assert {"structure_backbone_transform", "structure_3di_transform"}.issubset(transform_names), (
        f"missing 3Di transform: {transform_names}"
    )

    collate_target = cfg.data.collate_fn._target_
    assert collate_target.endswith(".collate_fn_backbone"), (
        f"data overlay must use plain collate_fn_backbone; got {collate_target}"
    )
