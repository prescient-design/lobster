"""Hydra compose smoke for the BASE-size 3Di-input flow experiment (Step G)."""

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
        f"Skipping 3Di-input flow BASE train compose smoke: missing {_MISSING}",
        allow_module_level=True,
    )


@pytest.fixture(autouse=True)
def _clear_hydra():
    GlobalHydra.instance().clear()
    yield
    GlobalHydra.instance().clear()


def test_train_latent_generator_3di_input_flow_base_composes() -> None:
    with initialize_config_dir(
        version_base=None,
        config_dir=str(HYDRA_ROOT),
        job_name="train_3di_input_flow_base_smoke",
    ):
        cfg = compose(
            config_name="train",
            overrides=["experiment=train_latent_generator_3di_input_flow_base"],
        )

    assert cfg.get("model") is not None
    assert cfg.get("data") is not None
    assert cfg.get("trainer") is not None

    assert cfg.model._target_ == "lobster.model.latent_generator.tokenizer.Tokenizer3diInputFlow"

    decoder_cfg = cfg.model.decoder_factory.decoder_mapping.vit_decoder
    assert decoder_cfg._target_ == "lobster.model.latent_generator.structure_decoder.ViTDecoderConditional"

    # Step G BASE knobs: width + depth + heads + dim_head all bumped, with
    # time_cond_dim matched to the wider residual stream. These four
    # asserts collectively define what "BASE" means -- if anyone tweaks
    # the yaml the wrong way, this test catches it before the SLURM run.
    assert decoder_cfg.struc_token_dim == 768
    assert decoder_cfg.uvit_n_layers == 12
    assert decoder_cfg.uvit_n_heads == 12
    assert decoder_cfg.uvit_dim_head == 64
    assert decoder_cfg.time_cond_dim == 256, (
        "time_cond_dim must be 256 for the BASE width (matches ESMFold2 fourier_dim "
        "at c_token=768); leaving it at 128 starves the FiLM time gating."
    )

    # Step G explicitly disables self-conditioning so the scale-axis A/B
    # against the small plain-flow baseline is clean. The "BASE + self-cond"
    # combination is a deliberate follow-up, not this run.
    assert cfg.model.use_self_conditioning is False
    assert decoder_cfg.use_self_conditioning is False

    # The rest of the Step E deltas must still be wired through unchanged.
    assert cfg.model.interpolant._target_.endswith("ContinuousFlowMatcher")
    assert cfg.model.interpolant.prediction_type == "data"
    assert cfg.model.interpolant.augmentation_type == "kabsch"
    assert cfg.model.center_x1 is True
    assert cfg.model.center_every_step is True
    assert decoder_cfg.num_registers == 4
    assert cfg.model.sampling_mode == "ode"
    assert cfg.model.p_uncond == 0.15
    assert cfg.model.guidance_scale == 1.0

    # Step H: random SO(3) augmentation on GT (train-only). Especially
    # relevant for BASE -- the larger parameter count would otherwise
    # let the model memorise canonical PDB orientations.
    assert cfg.model.random_so3_aug is True

    # Step J: Proteina-style coordinate scaling (Angstrom -> nm).
    assert cfg.model.coord_scale == 0.1

    # Constant 1e-4 across S+BASE (Step Q in the plan: flat schedule for clean A/B).
    assert abs(cfg.model.optim.lr - 1e-4) < 1e-9, (
        f"BASE lr should be 1e-4 (Step Q flat-LR schedule for cross-size A/B); got {cfg.model.optim.lr}"
    )

    # Distinct wandb project so the BASE run A/Bs cleanly against the
    # small plain-flow baseline AND the Step F self-cond run.
    assert cfg.logger.project == "lobster_latent_generator_3di_input_flow_base"

    # Sanity: same data overlay as the other flow variants.
    transform_names = set(cfg.data.transform_fn.keys())
    assert {"structure_backbone_transform", "structure_3di_transform"}.issubset(transform_names)
    collate_target = cfg.data.collate_fn._target_
    assert collate_target.endswith(".collate_fn_backbone")
