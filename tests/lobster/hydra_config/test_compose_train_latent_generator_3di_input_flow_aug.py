"""Hydra compose smoke for the Step H aug variants of the 3 flow experiments.

Each `_aug` experiment must:
  - load its underlying model yaml (unchanged from the non-aug companion);
  - inherit `random_so3_aug: true` from the model yaml default (no need to
    explicitly set it in the experiment yaml);
  - use a distinct wandb project so the augmented runs A/B cleanly against
    the pinned no-aug baselines.

These tests parallel the existing `test_compose_train_latent_generator_3di_input_flow{,_selfcond,_base}.py`
files but assert the aug-specific knobs.
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
        f"Skipping 3Di-input flow aug train compose smoke: missing {_MISSING}",
        allow_module_level=True,
    )


@pytest.fixture(autouse=True)
def _clear_hydra():
    GlobalHydra.instance().clear()
    yield
    GlobalHydra.instance().clear()


def _compose(experiment_name: str):
    """Compose `cfg.train` with the given experiment override."""
    with initialize_config_dir(
        version_base=None,
        config_dir=str(HYDRA_ROOT),
        job_name=f"smoke_{experiment_name}",
    ):
        return compose(config_name="train", overrides=[f"experiment={experiment_name}"])


def _assert_common_flow_invariants(cfg) -> None:
    """Knobs that every flow experiment must wire through (Steps E + H)."""
    assert cfg.model._target_ == "lobster.model.latent_generator.tokenizer.Tokenizer3diInputFlow"
    decoder_cfg = cfg.model.decoder_factory.decoder_mapping.vit_decoder
    assert decoder_cfg._target_ == "lobster.model.latent_generator.structure_decoder.ViTDecoderConditional"
    assert cfg.model.interpolant._target_.endswith("ContinuousFlowMatcher")
    assert cfg.model.interpolant.prediction_type == "data"
    assert cfg.model.interpolant.augmentation_type == "kabsch"
    assert cfg.model.center_x1 is True
    assert cfg.model.center_every_step is True
    assert decoder_cfg.num_registers == 4
    assert cfg.model.sampling_mode == "ode"
    assert cfg.model.p_uncond == 0.15
    assert cfg.model.guidance_scale == 1.0

    # Step H: this is what every aug variant must have ON. The
    # corresponding pinned baselines override to false in their launchers.
    assert cfg.model.random_so3_aug is True, (
        "the _aug experiment must inherit random_so3_aug=true from the "
        "model yaml -- otherwise it provides no A/B against the pinned no-aug "
        "baseline that lives in the same wandb workspace."
    )

    # Step J: Proteina-style coordinate scaling (Angstrom -> nm).
    assert cfg.model.coord_scale == 0.1

    # Same data overlay across all flow runs.
    transform_names = set(cfg.data.transform_fn.keys())
    assert {"structure_backbone_transform", "structure_3di_transform"}.issubset(transform_names)
    assert cfg.data.collate_fn._target_.endswith(".collate_fn_backbone")


def test_train_latent_generator_3di_input_flow_aug_composes() -> None:
    cfg = _compose("train_latent_generator_3di_input_flow_aug")
    _assert_common_flow_invariants(cfg)
    # Small + plain dims (parity with the non-aug small flow).
    decoder_cfg = cfg.model.decoder_factory.decoder_mapping.vit_decoder
    assert decoder_cfg.struc_token_dim == 512
    assert decoder_cfg.uvit_n_layers == 6
    assert decoder_cfg.time_cond_dim == 128
    # NOTE: `use_self_conditioning` is intentionally NOT asserted here -- the
    # small + plain model yaml relies on the Python default (False) and does
    # not set the key explicitly. The non-aug compose test makes the same
    # omission. The selfcond_aug + base_aug tests below DO assert it because
    # their model yamls set the key explicitly.

    # Distinct wandb project so this run A/Bs against the pinned no-aug
    # baseline (job 12478043 / project `lobster_latent_generator_3di_input_flow`).
    assert cfg.logger.project == "lobster_latent_generator_3di_input_flow_aug"


def test_train_latent_generator_3di_input_flow_selfcond_aug_composes() -> None:
    cfg = _compose("train_latent_generator_3di_input_flow_selfcond_aug")
    _assert_common_flow_invariants(cfg)
    # Small + self-cond dims + flags (parity with the non-aug self-cond flow).
    decoder_cfg = cfg.model.decoder_factory.decoder_mapping.vit_decoder
    assert decoder_cfg.struc_token_dim == 512
    assert decoder_cfg.uvit_n_layers == 6
    assert decoder_cfg.time_cond_dim == 128
    assert cfg.model.use_self_conditioning is True
    assert cfg.model.selfcond_train_prob == 0.5
    assert decoder_cfg.use_self_conditioning is True
    # Distinct wandb project (A/Bs against job 12531194).
    assert cfg.logger.project == "lobster_latent_generator_3di_input_flow_selfcond_aug"


def test_train_latent_generator_3di_input_flow_base_aug_composes() -> None:
    cfg = _compose("train_latent_generator_3di_input_flow_base_aug")
    _assert_common_flow_invariants(cfg)
    # BASE dims (parity with the non-aug BASE flow).
    decoder_cfg = cfg.model.decoder_factory.decoder_mapping.vit_decoder
    assert decoder_cfg.struc_token_dim == 768
    assert decoder_cfg.uvit_n_layers == 12
    assert decoder_cfg.uvit_n_heads == 12
    assert decoder_cfg.uvit_dim_head == 64
    assert decoder_cfg.time_cond_dim == 256
    assert cfg.model.use_self_conditioning is False
    assert decoder_cfg.use_self_conditioning is False
    assert abs(cfg.model.optim.lr - 1e-4) < 1e-9
    # Distinct wandb project (A/Bs against job 12531195).
    assert cfg.logger.project == "lobster_latent_generator_3di_input_flow_base_aug"
