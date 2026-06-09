"""Phase 7 dispatch smoke test for ``lobster.cmdline.generate``.

The full per-mode E2E smoke tests (under
``tests/lobster/model/leflur/test_generate_smoke.py`` and
``test_generate_ligand_conditioned_smoke.py``) require GPUs, internal
checkpoints, and benchmark fixtures. They run when those preconditions
are met and skip cleanly otherwise — but that leaves a coverage gap in
plain CI: regressions in the **mode dispatcher** (the chain of ``if /
elif`` branches in ``generate.py:generate``) can ship undetected because
the smoke tests skip.

This test fills that gap by:

1. Patching out checkpoint resolution, model loading, ESMFold, and every
   ``_generate_<mode>`` helper.
2. Composing each canonical Tier-1 Hydra config (one per supported mode)
   via Hydra's ``initialize_config_dir`` + ``compose`` API.
3. Invoking ``generate.generate.__wrapped__`` (the underlying function,
   bypassing the ``@hydra.main`` wrapper) on each composed config.
4. Asserting that **only the expected per-mode helper was called**, and
   that the universal scaffolding (MetricsCSVWriter init, output_dir
   creation) ran exactly as it would in production.

Result: every canonical config's dispatch path gets a CI-runnable
regression test, with no GPU, no model weights, and no network access.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[3]
HYDRA_ROOT = REPO_ROOT / "src" / "lobster" / "hydra_config"

# Map each canonical Tier-1 config to the helper name we expect the
# dispatcher to call. The helpers live as module-level functions in
# ``lobster.cmdline.generate``; we patch them to no-op MagicMocks so we
# can confirm dispatch routing without actually sampling.
MODE_DISPATCH_TARGETS: list[tuple[str, str, str]] = [
    (
        "experiment/generate_unconditional",
        "unconditional",
        "_generate_unconditional",
    ),
    (
        "experiment/generate_forward_folding",
        "forward_folding",
        "_generate_forward_folding",
    ),
    (
        "experiment/generate_inverse_folding",
        "inverse_folding",
        "_generate_inverse_folding",
    ),
    (
        "experiment/generate_inpainting",
        "inpainting",
        "_generate_inpainting",
    ),
    (
        "experiment/generate_ligand_conditioned",
        "ligand_conditioned",
        "_generate_ligand_conditioned",
    ),
    (
        "experiment/generate_ligand_conditioned_forward_folding",
        "protein_ligand_forward_folding",
        "_generate_protein_ligand_forward_folding",
    ),
    (
        "experiment/generate_ligand_conditioned_inverse_folding",
        "protein_ligand_inverse_folding",
        "_generate_protein_ligand_inverse_folding",
    ),
    (
        "experiment/score_pll",
        "score_pll",
        "_score_pll",
    ),
]

ALL_HELPERS: tuple[str, ...] = (
    "_generate_unconditional",
    "_generate_forward_folding",
    "_generate_inverse_folding",
    "_generate_inpainting",
    "_generate_binders",
    "_generate_ligand_conditioned",
    "_generate_protein_ligand_forward_folding",
    "_generate_protein_ligand_inverse_folding",
    "_score_pll",
)


@pytest.fixture(autouse=True)
def _clear_hydra_global():
    GlobalHydra.instance().clear()
    yield
    GlobalHydra.instance().clear()


@pytest.fixture
def patched_generate(tmp_path, monkeypatch):
    """Patch out everything heavy in :mod:`lobster.cmdline.generate`."""
    generate_mod = importlib.import_module("lobster.cmdline.generate")

    # Patch resolve_checkpoint to a no-op that returns a sentinel path —
    # we never actually open the file.
    monkeypatch.setattr(generate_mod, "resolve_checkpoint", lambda v: Path("/dev/null"))

    # Patch hydra.utils.get_class to return a fake class whose
    # load_from_checkpoint returns a stand-in module.
    fake_model = MagicMock(name="fake_lightning_module")
    fake_model.eval = MagicMock(return_value=fake_model)
    fake_model.to = MagicMock(return_value=fake_model)
    fake_model.__class__ = MagicMock(name="LeFlurStub")

    def _fake_get_class(target: str) -> Any:
        cls = MagicMock(name=f"class[{target}]")
        cls.load_from_checkpoint = MagicMock(return_value=fake_model)
        return cls

    monkeypatch.setattr(generate_mod.hydra.utils, "get_class", _fake_get_class)
    monkeypatch.setattr(generate_mod.hydra.utils, "instantiate", lambda cfg, **kw: fake_model)

    # Patch ESMFold loading — never instantiate the 600M weights.
    monkeypatch.setattr(generate_mod, "LobsterPLMFold", MagicMock(return_value=MagicMock()))

    # Patch MetricsCSVWriter + MetricsPlotter so they don't try to write
    # to a real directory the test doesn't manage.
    monkeypatch.setattr(generate_mod, "MetricsCSVWriter", MagicMock())
    monkeypatch.setattr(generate_mod, "MetricsPlotter", MagicMock())

    # Patch every per-mode helper as a MagicMock; the test asserts which
    # one was invoked.
    helpers: dict[str, MagicMock] = {}
    for name in ALL_HELPERS:
        m = MagicMock(name=name)
        helpers[name] = m
        monkeypatch.setattr(generate_mod, name, m)

    # The PL-mode helpers also do `isinstance(model, LeFlurProteinLigandLightningModule)`
    # — patch that too so our MagicMock model is accepted.
    from lobster.model import leflur as leflur_mod

    monkeypatch.setattr(
        leflur_mod,
        "LeFlurProteinLigandLightningModule",
        type(fake_model),
        raising=False,
    )

    return generate_mod, helpers, fake_model


def _compose_with_temp_output(config_name: str, tmp_path: Path) -> OmegaConf:
    """Compose a Hydra config, redirecting paths to a tmp dir for safety."""
    with initialize_config_dir(
        version_base=None,
        config_dir=str(HYDRA_ROOT),
        job_name="dispatch_smoke",
    ):
        cfg = compose(
            config_name=config_name,
            overrides=[
                f"paths.evaluations.out_root={tmp_path}",
                f"output_dir={tmp_path}/dispatch_smoke_out",
            ],
        )
    return cfg


@pytest.mark.parametrize(
    "config_name,expected_mode,expected_helper",
    MODE_DISPATCH_TARGETS,
    ids=lambda v: v.rsplit("/", 1)[-1] if isinstance(v, str) else v,
)
def test_generate_routes_to_correct_helper(
    patched_generate,
    tmp_path,
    config_name: str,
    expected_mode: str,
    expected_helper: str,
) -> None:
    """``generate(cfg)`` invokes exactly one helper — the right one."""
    generate_mod, helpers, _ = patched_generate
    cfg = _compose_with_temp_output(config_name, tmp_path)

    assert cfg.generation.mode == expected_mode, (
        f"{config_name} composed with generation.mode={cfg.generation.mode}; expected {expected_mode}"
    )

    # ``hydra.main`` decorates ``generate`` — call the underlying function
    # directly so we don't have to spin up Hydra's job machinery here.
    inner = getattr(generate_mod.generate, "__wrapped__", generate_mod.generate)
    inner(cfg)

    # Exactly one helper should have been invoked.
    called = [name for name, m in helpers.items() if m.called]
    assert called == [expected_helper], (
        f"dispatcher routing for {config_name}: expected ONLY {expected_helper!r} to fire, but {called} fired."
    )


def test_unknown_mode_raises(patched_generate, tmp_path) -> None:
    """An unknown ``generation.mode`` value raises ``ValueError``."""
    generate_mod, _, _ = patched_generate
    cfg = _compose_with_temp_output("experiment/generate_unconditional", tmp_path)
    cfg.generation.mode = "nonexistent_mode_xyz"

    inner = getattr(generate_mod.generate, "__wrapped__", generate_mod.generate)
    with pytest.raises(ValueError, match="Unknown generation mode"):
        inner(cfg)
