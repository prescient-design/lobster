"""Phase 3 path-overlay regression tests.

Two complementary checks:

1. ``test_no_internal_paths_in_publication_configs`` — the inference-facing
   ``experiment/generate_*.yaml`` and ``experiment/autoencode*.yaml`` configs
   must NOT embed Genentech-internal paths (``/cv/data``, ``/cv/scratch``,
   ``/cv/home``). They must instead interpolate ``${paths.*}`` against
   ``paths/internal.yaml`` or ``paths/public.yaml``.
2. ``test_paths_internal_resolves`` / ``test_paths_public_resolves`` — each
   publication-tier config composes cleanly under both overlays and resolves
   the documented ``${paths.checkpoints.*}`` / ``${paths.evaluations.*}``
   interpolations.

These run on every CI invocation; if anyone re-introduces a ``/cv/...``
literal, the build breaks and the gen_ume → LeFlur publication scope stays
intact.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

REPO_ROOT = Path(__file__).resolve().parents[3]
HYDRA_ROOT = REPO_ROOT / "src" / "lobster" / "hydra_config"
EXPERIMENT_DIR = HYDRA_ROOT / "experiment"

# Patterns we forbid in publication configs.
FORBIDDEN = re.compile(r"/cv/(data|scratch|home)\b")

# Globs that are "publication scope" — the path overlay must cover them.
# Phase 6 narrowed this to Tier 1 (top-level `experiment/*.yaml`). The
# Tier-2 `experiment/research/` configs and the single Tier-3
# `experiment/legacy/generate_unconditional_old.yaml` are kept on disk
# for internal research workflows but gitignored on the publication
# branch — they reference research/legacy `${paths.checkpoints.*}` keys
# that only resolve under the internal overlay.
PUBLICATION_GLOBS = ("generate_*.yaml", "autoencode*.yaml")


def _publication_configs() -> list[Path]:
    paths: list[Path] = []
    for pattern in PUBLICATION_GLOBS:
        # Tier 1 lives flat under experiment/ — explicitly excluded subtiers.
        for p in sorted(EXPERIMENT_DIR.glob(pattern)):
            paths.append(p)
    return paths


@pytest.mark.parametrize("config_path", _publication_configs(), ids=lambda p: p.name)
def test_no_internal_paths_in_publication_configs(config_path: Path) -> None:
    """No raw /cv/data or /cv/scratch (or /cv/home) literals — must use ${paths.*}."""
    text = config_path.read_text()
    offending = [
        f"  L{lineno}: {line.rstrip()}"
        for lineno, line in enumerate(text.splitlines(), start=1)
        if FORBIDDEN.search(line)
    ]
    assert not offending, (
        f"{config_path.relative_to(REPO_ROOT)} embeds hard-coded "
        f"Genentech-internal paths. Replace each with a "
        f"`${{paths.<group>.<name>}}` interpolation against "
        f"`paths/internal.yaml` / `paths/public.yaml`. Offending lines:\n" + "\n".join(offending)
    )


# A representative subset spanning the canonical task modes; full sweep
# would slow CI without adding signal. If a config breaks composition we
# want one fixture-style assertion per task type — and after Phase 6 those
# targets live at the top-level (Tier 1).
_COMPOSE_TARGETS = [
    "experiment/generate_unconditional",
    "experiment/generate_forward_folding",
    "experiment/generate_inverse_folding",
    "experiment/generate_inpainting",
    "experiment/generate_ligand_conditioned",
    "experiment/generate_ligand_conditioned_forward_folding",
    "experiment/generate_ligand_conditioned_inverse_folding",
    "experiment/autoencode",
    "experiment/autoencode_protein_ligand",
]

# NOTE: experiment/research/ configs are kept on disk for internal research
# workflows but are gitignored on the publication branch. The dedicated
# `test_research_tier_composes` smoke that previously parametrized over a
# representative subset was dropped alongside the untracking — there is no
# publication-CI signal to recover (the files simply aren't shipped).


@pytest.fixture(autouse=True)
def _clear_hydra():
    GlobalHydra.instance().clear()
    yield
    GlobalHydra.instance().clear()


@pytest.mark.parametrize("config_name", _COMPOSE_TARGETS)
def test_paths_internal_resolves(config_name: str) -> None:
    """Every Tier-1 canonical config composes under `paths=internal`."""
    with initialize_config_dir(
        version_base=None,
        config_dir=str(HYDRA_ROOT),
        job_name="paths_internal_smoke",
    ):
        cfg = compose(config_name=config_name)
        assert cfg.get("model") is not None or cfg.get("autoencode") is not None
        out_dir = cfg.get("output_dir")
        assert out_dir, f"{config_name} missing output_dir"


@pytest.mark.parametrize("config_name", _COMPOSE_TARGETS)
def test_paths_public_resolves(config_name: str, monkeypatch) -> None:
    """`paths=public` swap composes cleanly for every Tier-1 config."""
    monkeypatch.setenv("HOME", "/tmp/leflur_public_smoke")
    monkeypatch.delenv("LOBSTER_OUT", raising=False)
    monkeypatch.delenv("LOBSTER_CACHE", raising=False)
    monkeypatch.delenv("FOLDSEEK_BIN", raising=False)
    with initialize_config_dir(
        version_base=None,
        config_dir=str(HYDRA_ROOT),
        job_name="paths_public_smoke",
    ):
        cfg = compose(config_name=config_name, overrides=["paths=public"])
        assert cfg.get("model") is not None or cfg.get("autoencode") is not None


# --- Tier invariants (Phase 6) -------------------------------------------

TIER_1_SHORT_NAMES = {"leflur-base", "leflur-ted", "leflur-pl"}
TIER_1_ALLOWED_INTERPOLATIONS = (
    "${paths.checkpoints.base}",
    "${paths.checkpoints.ted}",
    "${paths.checkpoints.pl}",
)


@pytest.mark.parametrize("config_path", _publication_configs(), ids=lambda p: p.name)
def test_tier1_uses_canonical_checkpoint(config_path: Path) -> None:
    """Tier-1 configs must use canonical short names (or canonical path interpolations).

    Forbidding `${paths.checkpoints.research_*}` / `${paths.checkpoints.legacy_*}`
    here is how we keep the "happy path" set buildable for external users —
    those interpolations only resolve under the internal overlay.
    """
    text = config_path.read_text()
    has_ckpt = re.search(r"^\s*ckpt_path\s*:\s*(?P<val>.+)$", text, re.MULTILINE)
    if has_ckpt is None:
        # Some autoencode configs nest ckpt_path differently; skip.
        return
    val = has_ckpt.group("val").strip().strip("\"'")
    if val in TIER_1_SHORT_NAMES:
        return
    if any(allow in val for allow in TIER_1_ALLOWED_INTERPOLATIONS):
        return
    raise AssertionError(
        f"{config_path.relative_to(REPO_ROOT)} uses ckpt_path={val!r}; "
        f"Tier 1 must be one of {TIER_1_SHORT_NAMES} or "
        f"{TIER_1_ALLOWED_INTERPOLATIONS}. Move this config under the "
        f"(gitignored) experiment/research/ or experiment/legacy/ subdir "
        f"if it needs a research/legacy checkpoint."
    )
