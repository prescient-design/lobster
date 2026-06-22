"""Hydra-compose smoke for the active 3Di flow-matching training family.

Verifies that the four production training experiments (base + the
three coord-CE variants) compose cleanly through ``train.yaml``, and
that the resulting cfg points at the right model class and data
overlay. Replaces the older ``test_compose_train_latent_generator_3di_input_flow{,_aug,_base,_selfcond}.py``
suite (those covered abandoned variants whose yamls were removed in
``fd7d5fd``).
"""

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


# (experiment_name, expected_aux_3di_coord_ce_weight, expected_track_aux_3di_coord_ce)
ACTIVE_EXPERIMENTS = [
    ("train_latent_generator_3di_input_flow_nokabsch_velocity_base", None, None),
    ("train_latent_generator_3di_input_flow_nokabsch_velocity_base_3di_coord_ce", 0.10, None),
    ("train_latent_generator_3di_input_flow_nokabsch_velocity_base_3di_coord_ce_correct_cb", 0.10, None),
    ("train_latent_generator_3di_input_flow_nokabsch_velocity_base_3di_coord_ce_w0", 0.0, True),
]


@pytest.mark.parametrize("experiment, weight, track", ACTIVE_EXPERIMENTS)
def test_active_3di_flow_experiment_composes(experiment, weight, track) -> None:
    with initialize_config_dir(
        version_base=None,
        config_dir=str(HYDRA_ROOT),
        job_name=f"compose_smoke_{experiment}",
    ):
        cfg = compose(config_name="train", overrides=[f"experiment={experiment}"])

    # Top-level cfg shape -- model / data / trainer all present.
    assert cfg.get("model") is not None
    assert cfg.get("data") is not None
    assert cfg.get("trainer") is not None

    # Every active experiment instantiates the flow-matching tokenizer.
    assert cfg.model._target_ == "lobster.model.latent_generator.tokenizer.Tokenizer3diInputFlow", (
        f"unexpected model target: {cfg.model._target_}"
    )

    # Decoder is the conditional U-ViT used by every 3Di-flow variant.
    decoder_target = cfg.model.decoder_factory.decoder_mapping.vit_decoder._target_
    assert decoder_target == "lobster.model.latent_generator.structure_decoder.ViTDecoderConditional"

    # Velocity-prediction continuous flow matching, no Kabsch coupling
    # (data augmentation handles SE(3) equivariance instead).
    interp_target = cfg.model.interpolant._target_
    assert interp_target.endswith("ContinuousFlowMatcher")
    assert cfg.model.interpolant.prediction_type == "velocity"
    assert cfg.model.interpolant.augmentation_type is None

    # Coord-scaling and centering knobs are stable across the family.
    assert cfg.model.coord_scale == 0.1
    assert cfg.model.center_x1 is True
    assert cfg.model.center_every_step is True
    assert cfg.model.random_so3_aug is True

    # CFG knobs.
    assert cfg.model.p_uncond == 0.15
    assert cfg.model.guidance_scale == 1.0

    # Variant-specific aux-loss config.
    if weight is not None:
        assert cfg.model.aux_3di_coord_ce_weight == pytest.approx(weight)
    if track is not None:
        assert bool(cfg.model.track_aux_3di_coord_ce) is bool(track)

    # Data overlay is the shared `pdb_with_3di` mix.
    transform_names = set(cfg.data.transform_fn.keys())
    assert {"structure_backbone_transform", "structure_3di_transform"}.issubset(transform_names), (
        f"missing 3Di transform: {transform_names}"
    )
    collate_target = cfg.data.collate_fn._target_
    assert collate_target.endswith(".collate_fn_backbone"), (
        f"data overlay must use plain collate_fn_backbone; got {collate_target}"
    )
