"""Hydra compose smoke for the self-conditioned 3Di-input flow experiment (Step F).

Mirrors ``test_compose_train_latent_generator_3di_input_flow.py`` and adds
assertions for the self-conditioning knobs (decoder + tokenizer) and the
new wandb project name.
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
        f"Skipping 3Di-input flow self-cond train compose smoke: missing {_MISSING}",
        allow_module_level=True,
    )


@pytest.fixture(autouse=True)
def _clear_hydra():
    GlobalHydra.instance().clear()
    yield
    GlobalHydra.instance().clear()


def test_train_latent_generator_3di_input_flow_selfcond_composes() -> None:
    with initialize_config_dir(
        version_base=None,
        config_dir=str(HYDRA_ROOT),
        job_name="train_3di_input_flow_selfcond_smoke",
    ):
        cfg = compose(
            config_name="train",
            overrides=["experiment=train_latent_generator_3di_input_flow_selfcond"],
        )

    assert cfg.get("model") is not None
    assert cfg.get("data") is not None
    assert cfg.get("trainer") is not None

    assert cfg.model._target_ == "lobster.model.latent_generator.tokenizer.Tokenizer3diInputFlow"

    decoder_cfg = cfg.model.decoder_factory.decoder_mapping.vit_decoder
    assert decoder_cfg._target_ == "lobster.model.latent_generator.structure_decoder.ViTDecoderConditional"

    # Step F: self-conditioning enabled on both sides of the contract.
    assert cfg.model.use_self_conditioning is True, (
        "tokenizer must have use_self_conditioning=True for the selfcond variant"
    )
    assert cfg.model.selfcond_train_prob == 0.5
    assert decoder_cfg.use_self_conditioning is True, (
        "decoder must have use_self_conditioning=True so coord_in_proj_selfcond "
        "is actually built; otherwise the runtime guard in Tokenizer3diInputFlow "
        "__init__ raises"
    )

    # The rest of the Step E deltas must still be wired through unchanged.
    interp_target = cfg.model.interpolant._target_
    assert interp_target.endswith("ContinuousFlowMatcher")
    assert cfg.model.interpolant.prediction_type == "data"
    assert cfg.model.interpolant.augmentation_type == "kabsch"

    assert cfg.model.center_x1 is True
    assert cfg.model.center_every_step is True
    assert decoder_cfg.num_registers == 4
    assert decoder_cfg.time_cond_dim == 128
    assert cfg.model.sampling_mode == "ode"
    assert cfg.model.p_uncond == 0.15
    assert cfg.model.guidance_scale == 1.0

    # Step H: random SO(3) augmentation on GT (train-only). Compatible with
    # self-conditioning -- the warm no-grad forward and the grad forward
    # both see the same rotated x_t for that step.
    assert cfg.model.random_so3_aug is True

    # Step J: Proteina-style coordinate scaling (Angstrom -> nm).
    assert cfg.model.coord_scale == 0.1

    # Distinct wandb project so the selfcond run A/Bs cleanly against the
    # plain-flow baseline (and the deterministic regression).
    assert cfg.logger.project == "lobster_latent_generator_3di_input_flow_selfcond"

    # Sanity: same data overlay as the plain-flow run.
    transform_names = set(cfg.data.transform_fn.keys())
    assert {"structure_backbone_transform", "structure_3di_transform"}.issubset(transform_names)
    collate_target = cfg.data.collate_fn._target_
    assert collate_target.endswith(".collate_fn_backbone")
