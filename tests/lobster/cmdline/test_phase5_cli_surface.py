"""Smoke tests for the Phase 5 unified CLI surface.

Validates without running any heavy inference:

- ``lobster_generate`` is importable and `@hydra.main`-decorated against the
  config_path/config_name we registered in pyproject.toml.
- ``lobster_autoencode`` is importable and `@hydra.main`-decorated.
- ``generation.mode=ligand_conditioned`` dispatches into the shared
  :mod:`lobster.cmdline._ligand_conditioned_runner` (verified via the
  mode-dispatch table without invoking the model).
- The shared runner accepts a :class:`LigandConditionedRunConfig` and
  surfaces a clear error when ``data_dir`` is missing on disk.
- Both new Hydra configs compose under ``paths=internal`` AND
  ``paths=public`` (Phase 3 overlay invariants).
"""

from __future__ import annotations

import importlib

import pytest
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra


REPO_HYDRA_MODULE = "lobster.hydra_config"


# --- Entry-point importability --------------------------------------------


def test_lobster_generate_importable() -> None:
    """`generate.py` exports a callable Hydra entry point named ``generate``."""
    mod = importlib.import_module("lobster.cmdline.generate")
    assert hasattr(mod, "generate"), "lobster.cmdline.generate.generate missing"
    assert callable(mod.generate)


def test_lobster_autoencode_importable() -> None:
    """`autoencode.py` exports a callable Hydra entry point named ``autoencode``."""
    mod = importlib.import_module("lobster.cmdline.autoencode")
    assert hasattr(mod, "autoencode"), "lobster.cmdline.autoencode.autoencode missing"
    assert callable(mod.autoencode)


def test_console_scripts_registered() -> None:
    """``pyproject.toml`` registers both new console scripts."""
    import tomllib
    from pathlib import Path

    # Walk up until we find the pyproject.toml that defines the lobster project.
    pyproject_file = None
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "pyproject.toml"
        if candidate.exists():
            pyproject_file = candidate
            break
    assert pyproject_file is not None, "could not locate pyproject.toml above this test"

    with open(pyproject_file, "rb") as fh:
        data = tomllib.load(fh)
    scripts = data["project"]["scripts"]
    assert scripts.get("lobster_generate") == "lobster.cmdline.generate:generate"
    assert scripts.get("lobster_autoencode") == "lobster.cmdline.autoencode:autoencode"
    # Phase 4 entry stays put.
    assert scripts.get("lobster_leflur_checkpoints") == "lobster.cmdline.manage_leflur_checkpoints:main"


# --- Hydra compose checks --------------------------------------------------


@pytest.fixture
def hydra_clean():
    GlobalHydra.instance().clear()
    yield
    GlobalHydra.instance().clear()


PHASE5_CONFIGS_AND_EXPECTED_MODE = {
    "experiment/generate_ligand_conditioned": "ligand_conditioned",
    "experiment/generate_ligand_conditioned_forward_folding": ("protein_ligand_forward_folding"),
    "experiment/generate_ligand_conditioned_inverse_folding": ("protein_ligand_inverse_folding"),
    "experiment/autoencode": None,
    "experiment/autoencode_protein_ligand": None,
}


@pytest.mark.parametrize("config_name", list(PHASE5_CONFIGS_AND_EXPECTED_MODE))
@pytest.mark.parametrize("overlay", ["internal", "public"])
def test_phase5_configs_compose(hydra_clean, config_name, overlay) -> None:
    """Phase 5 configs compose under both internal and public path overlays."""
    with initialize_config_module(config_module=REPO_HYDRA_MODULE, version_base=None):
        cfg = compose(
            config_name=config_name,
            overrides=[
                f"paths={overlay}",
                "paths.root_dir=/tmp/leflur_compose_smoke",
            ],
        )
    assert "model" in cfg, f"{config_name} missing model block"
    expected_mode = PHASE5_CONFIGS_AND_EXPECTED_MODE[config_name]
    if config_name.startswith("experiment/autoencode"):
        assert "autoencode" in cfg
        assert cfg.model.ckpt_path.startswith("leflur-")
    else:
        assert cfg.generation.mode == expected_mode, f"{config_name} should set generation.mode={expected_mode}"
        assert cfg.model.ckpt_path == "leflur-pl"


# --- Mode dispatcher -------------------------------------------------------


def test_generate_dispatcher_has_all_ligand_conditioned_branches() -> None:
    """All three ligand-conditioned branches are wired into the dispatcher."""
    import importlib
    import inspect

    generate_mod = importlib.import_module("lobster.cmdline.generate")

    src = inspect.getsource(generate_mod.generate)
    for mode_str, helper in (
        ('"ligand_conditioned"', "_generate_ligand_conditioned"),
        (
            '"protein_ligand_forward_folding"',
            "_generate_protein_ligand_forward_folding",
        ),
        (
            '"protein_ligand_inverse_folding"',
            "_generate_protein_ligand_inverse_folding",
        ),
    ):
        assert f"generation_mode == {mode_str}" in src, f"generate.py is missing the {mode_str} dispatch branch"
        assert helper in src, f"generate.py is missing {helper} dispatch"


# --- Shared runner contract ------------------------------------------------


@pytest.mark.parametrize(
    "config_cls_name, runner_name",
    [
        ("LigandConditionedRunConfig", "run_ligand_conditioned_generation"),
        (
            "ProteinLigandForwardFoldingRunConfig",
            "run_protein_ligand_forward_folding",
        ),
        (
            "ProteinLigandInverseFoldingRunConfig",
            "run_protein_ligand_inverse_folding",
        ),
    ],
)
def test_ligand_conditioned_runners_reject_missing_data_dir(tmp_path, config_cls_name: str, runner_name: str) -> None:
    """Every PL runner surfaces a clear ``FileNotFoundError`` for bad data_dir."""
    runner_mod = importlib.import_module("lobster.cmdline._ligand_conditioned_runner")
    config_cls = getattr(runner_mod, config_cls_name)
    runner = getattr(runner_mod, runner_name)

    config = config_cls(
        data_dir=str(tmp_path / "does-not-exist"),
        output_dir=str(tmp_path / "out"),
    )
    sentinel_model = object()
    with pytest.raises(FileNotFoundError):
        runner(sentinel_model, config)


def test_ligand_conditioned_run_config_defaults() -> None:
    """:class:`LigandConditionedRunConfig` defaults stay aligned with the legacy CLI."""
    from lobster.cmdline._ligand_conditioned_runner import (
        LigandConditionedRunConfig,
    )

    cfg = LigandConditionedRunConfig(data_dir="/tmp/x", output_dir="/tmp/y")
    assert cfg.length == 100
    assert cfg.num_designs == 10
    assert cfg.nsteps == 200
    assert cfg.ligand_context_mode == "atom_bond_only"
    assert cfg.skip_esmfold is False


def test_protein_ligand_forward_folding_config_defaults() -> None:
    from lobster.cmdline._ligand_conditioned_runner import (
        ProteinLigandForwardFoldingRunConfig,
    )

    cfg = ProteinLigandForwardFoldingRunConfig(data_dir="/tmp/x", output_dir="/tmp/y")
    assert cfg.nsteps == 200
    assert cfg.num_predictions == 1
    assert cfg.best_of_n_metric == "plddt"
    assert cfg.try_reflection is False


def test_protein_ligand_inverse_folding_config_defaults() -> None:
    from lobster.cmdline._ligand_conditioned_runner import (
        ProteinLigandInverseFoldingRunConfig,
    )

    cfg = ProteinLigandInverseFoldingRunConfig(data_dir="/tmp/x", output_dir="/tmp/y")
    assert cfg.nsteps == 100
    assert cfg.inference_schedule_seq == "LogInferenceSchedule"
    assert cfg.use_se3_augmentation is False
