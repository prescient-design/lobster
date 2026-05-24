"""LeFlur inference-checkpoint registry + resolver.

Public surface
--------------

- :data:`KNOWN_CHECKPOINTS` — short-name → :class:`CheckpointInfo` registry.
- :func:`resolve_checkpoint` — turns any of (short name, ``hf://`` URI,
  ``https://huggingface.co/`` URL, local path) into a concrete local
  :class:`pathlib.Path` that the LeFlur Lightning modules can pass straight
  into :meth:`LeFlurSequenceStructureEncoderLightningModule.load_from_checkpoint`
  or :meth:`LeFlurProteinLigandLightningModule.load_from_checkpoint`.
- :func:`list_checkpoints` — pretty-printable copy of the registry.
- :func:`clear_cache` — drop the cached HF downloads.

Cache layout
~~~~~~~~~~~~

Downloads land under ``$LOBSTER_CACHE`` (default ``~/.cache/lobster/leflur``)
in a ``checkpoints/`` subdir. Files are content-addressed by their filename
on HF, so re-runs are O(1) cache hits.

Scope
~~~~~

The registry is intentionally *minimal*: only the three checkpoints needed
to reproduce the publication results live here.

- ``leflur-base`` — protein-only base (default unconditional generation).
- ``leflur-ted`` — protein-only TED-CATH SS-balanced (publication tables).
- ``leflur-pl``  — production protein-ligand (all ligand-conditioned modes).

Research / legacy snapshots are **not** distributed on HuggingFace —
internal collaborators still load them from their original ``/cv/...``
locations via Hydra's ``paths/internal.yaml`` overlay. Adding a new
checkpoint later is non-breaking: just append a ``CheckpointInfo``.

Notes
~~~~~

- The HF repo ``Sidney-Lisanza/leflur`` is populated in Phase 4 of the LeFlur
  publication cleanup (see ``.cursor/plans/genume_publication_cleanup_*``).
  Until then the ``hf://`` URIs below resolve to 404; the resolver still
  works for local paths and ``s3://`` is rejected with a clear message.
- Internal (``/cv/...``) paths are intentionally *not* hard-coded here —
  Hydra's ``paths/internal.yaml`` overlay owns those literals so the public
  / internal split stays in one place (see ``src/lobster/hydra_config/paths/``).
"""

from __future__ import annotations

import logging
import os
import shutil
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Iterable

logger = logging.getLogger(__name__)

# --- Constants -------------------------------------------------------------

HF_REPO_ID = "Sidney-Lisanza/leflur"
HF_REVISION = "main"

# Where downloaded files land. Honours ``$LOBSTER_CACHE`` and ``$HOME`` so the
# same code works on a developer laptop, on /home/$USER on the cluster, and
# under a CI runner with an isolated cache dir.
_DEFAULT_CACHE = Path.home() / ".cache" / "lobster" / "leflur"


def _cache_root() -> Path:
    return Path(os.environ.get("LOBSTER_CACHE", _DEFAULT_CACHE)) / "checkpoints"


# --- Registry --------------------------------------------------------------


@dataclass(frozen=True)
class CheckpointInfo:
    """Metadata for a publicly-distributed LeFlur checkpoint.

    Attributes
    ----------
    short_name
        Stable identifier (``leflur-base``, ``leflur-ted``, ...). Use this on
        the CLI: ``lobster_generate ... generation.checkpoint=leflur-ted``.
    hf_path
        Path within the HF repo (defaults to ``Sidney-Lisanza/leflur``). Used
        by :func:`resolve_checkpoint` to build the download call.
    description
        One-line human-readable summary. Surfaced by ``lobster_leflur_checkpoints list``.
    family
        ``"protein"`` or ``"protein_ligand"`` — drives which Lightning module
        class loads it.
    recommended_generation_config
        Hydra experiment config that matches the params used in the
        conference benchmark for this checkpoint (Phase 6 Tier 1 / Tier 2).
    paired_lg_codec
        Short name of the ``Sidney-Lisanza/latent_generator`` HF codec the
        LeFlur sample loop loads at inference time. Optional metadata,
        consumed by ``lobster_leflur_checkpoints inspect``.
    tags
        Free-form labels (``"canonical"``, ``"research"``, ``"legacy"``).
    hf_repo_id
        HuggingFace repo this checkpoint lives in. Defaults to
        :data:`HF_REPO_ID`; the paired LG codecs override this to
        ``Sidney-Lisanza/latent_generator``.
    local_source_path
        Internal ``/cv/...`` source the upload CLI reads from. Optional —
        absent for non-uploadable / external entries.
    """

    short_name: str
    hf_path: str
    description: str
    family: str = "protein"
    recommended_generation_config: str | None = None
    paired_lg_codec: str | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)
    hf_repo_id: str = ""  # filled in __post_init__ when empty (see below)
    local_source_path: str | None = None

    def __post_init__(self) -> None:
        # Frozen dataclasses need ``object.__setattr__`` for default backfill.
        if not self.hf_repo_id:
            object.__setattr__(self, "hf_repo_id", HF_REPO_ID)

    @property
    def hf_uri(self) -> str:
        """Canonical ``hf://`` URI for documentation / debugging."""
        return f"hf://{self.hf_repo_id}/{self.hf_path}"

    @property
    def https_url(self) -> str:
        """Direct download URL (works without an HF token for public repos)."""
        return f"https://huggingface.co/{self.hf_repo_id}/resolve/{HF_REVISION}/{self.hf_path}"


# Registry. Short names use kebab-case to match the plan
# (``leflur-base`` / ``leflur-ted`` / ``leflur-pl``).
#
# The two canonical protein-only checkpoints come straight from the
# conference benchmark plan (see
# ``.cursor/plans/conference_benchmark_comparison_9b71ca71.plan.md``); the
# various "TED-stoch / TED-val / etc." benchmark variants share these two
# parent ckpts with different generation hyperparameters — that lives in
# the Phase 6 config tiers, not here.
#
# Research / legacy snapshots are intentionally **not** distributed on
# HuggingFace: they're stale, large, and only useful for reproducing
# in-flight experiments that internal collaborators run from the original
# ``/cv/...`` paths. Adding them later is non-breaking — just append new
# ``CheckpointInfo`` entries.
KNOWN_CHECKPOINTS: dict[str, CheckpointInfo] = {
    "leflur-base": CheckpointInfo(
        short_name="leflur-base",
        hf_path="leflur_denovo_last_ckpt_2026-03-11T12-11-53.ckpt",
        description=(
            "Canonical protein-only base checkpoint (de-novo, last; "
            "2026-03-11). Recommended for unconditional generation under "
            "default temperatures/stochasticities."
        ),
        family="protein",
        recommended_generation_config="experiment/generate_unconditional_denovo",
        paired_lg_codec="LG full attention",
        tags=("canonical", "protein-only"),
        local_source_path=(
            "/cv/scratch/u/lisanzas/evaluations/checkpoint_snapshots/"
            "gen_ume_denovo_last_2026-03-08T17-09-23_2026-03-11T12-11-53.ckpt"
        ),
    ),
    "leflur-ted": CheckpointInfo(
        short_name="leflur-ted",
        hf_path="leflur_denovo_ted_cath_ss_balanced_ckpt_2026-03-18T12-20-59.ckpt",
        description=(
            "Canonical protein-only TED-CATH SS-balanced checkpoint "
            "(2026-03-18). Best designability/quality trade-off in the "
            "conference benchmark; recommended for the publication tables."
        ),
        family="protein",
        recommended_generation_config="experiment/generate_unconditional_denovo",
        paired_lg_codec="LG full attention",
        tags=("canonical", "protein-only", "publication"),
        local_source_path=(
            "/cv/scratch/u/lisanzas/gen_ume_denovo/runs/2026-03-06T15-30-31/epoch=17-step=6937-val_loss=0.8192.ckpt"
        ),
    ),
    "leflur-pl": CheckpointInfo(
        short_name="leflur-pl",
        hf_path="leflur_protein_ligand.ckpt",
        description=(
            "Production protein-ligand checkpoint (2026-03-11; no-geom-loss "
            "medium variant). Drives all ligand-conditioned generation + "
            "protein-ligand forward/inverse folding evaluations and reproduces "
            "the PoseBusters NO inverse-folding numbers reported in the "
            "conference benchmark (AAR overall no-ligand ~0.70, pocket ~0.78)."
        ),
        family="protein_ligand",
        recommended_generation_config=("experiment/generate_forward_folding_protein_ligand_cameo"),
        paired_lg_codec="LG Protein Ligand fsq 4375",
        tags=("canonical", "protein-ligand"),
        local_source_path=(
            "/cv/scratch/u/lisanzas/gen_ume_protein_ligand_no_geom_medium/runs/2026-03-11T13-22-20/last.ckpt"
        ),
    ),
}


# Paired LG codecs use a separate HF repo (`Sidney-Lisanza/latent_generator`)
# under the existing `checkpoints_for_lg/` prefix that
# `latent_generator/cmdline/inference.py` already speaks. These entries are
# kept out of ``KNOWN_CHECKPOINTS`` because they're not consumed by
# ``resolve_checkpoint`` — they ride the LG library's own download path,
# which ``install_paired_lg_codec_overrides()`` has rewired to HF URLs.
# We use ``CheckpointInfo`` here only so the upload CLI can share code.
LG_HF_CHECKPOINT_PREFIX = "checkpoints_for_lg"

PAIRED_LG_CHECKPOINTS: dict[str, CheckpointInfo] = {
    "LG Protein Ligand fsq 4375": CheckpointInfo(
        short_name="LG Protein Ligand fsq 4375",
        hf_path=f"{LG_HF_CHECKPOINT_PREFIX}/LG_Protein_Ligand_fsq_4375_2026-01-05.ckpt",
        description=(
            "Frozen protein-ligand LG codec paired with leflur-pl "
            "(FSQ quantizer, 4375 tokens). Auto-downloaded at sample time."
        ),
        family="protein_ligand",
        tags=("canonical", "lg-codec"),
        hf_repo_id="Sidney-Lisanza/latent_generator",
        local_source_path=("/cv/data/ai4dd/data2/ume/latent_generator_/runs/2026-01-05T16-13-19/last.ckpt"),
    ),
    "LG Protein Ligand cont": CheckpointInfo(
        short_name="LG Protein Ligand cont",
        hf_path=f"{LG_HF_CHECKPOINT_PREFIX}/LG_Protein_Ligand_continuous_2026-01-24.ckpt",
        description=(
            "Continuous protein-ligand LG codec (no quantization, bond "
            "matrix). Used by leflur-pl variants when configured."
        ),
        family="protein_ligand",
        tags=("canonical", "lg-codec"),
        hf_repo_id="Sidney-Lisanza/latent_generator",
        local_source_path=(
            "/cv/scratch/u/lisanzas/latent_generator_continuous_bond_element/runs/2026-01-24T21-03-23/last.ckpt"
        ),
    ),
}


def list_checkpoints(family: str | None = None) -> list[CheckpointInfo]:
    """Return registry entries, optionally filtered by ``family``."""
    entries = list(KNOWN_CHECKPOINTS.values())
    if family is not None:
        entries = [e for e in entries if e.family == family]
    return entries


# --- Paired LatentGenerator codecs ----------------------------------------
#
# LeFlur instantiates a frozen LatentGenerator (LG) "codec" at sample time
# via :data:`lobster.model.latent_generator.cmdline.methods`. A few of those
# codec entries currently point at internal ``s3://`` or ``/cv/...`` paths
# (see :mod:`lobster.model.latent_generator.cmdline.inference`), which would
# break ``pip install lobster`` users.
#
# ``PAIRED_LG_CODECS`` records the HF URL each LeFlur-required codec
# *should* resolve to. ``install_paired_lg_codec_overrides()`` rewrites the
# in-memory ``methods`` registry to use those URLs; the existing LG download
# path in ``inference.load_model`` already handles ``https://huggingface.co/``
# URLs natively (see ``urllib.request.urlretrieve`` block there), so no
# further wiring is needed.
LG_HF_REPO_ID = "Sidney-Lisanza/latent_generator"
LG_HF_REVISION = "main"
LG_HF_PREFIX = f"https://huggingface.co/{LG_HF_REPO_ID}/resolve/{LG_HF_REVISION}/checkpoints_for_lg"

PAIRED_LG_CODECS: dict[str, str] = {
    # Protein-only LeFlur ("LG full attention") is already HF-backed
    # natively in inference.py — listed here for documentation only.
    "LG full attention": f"{LG_HF_PREFIX}/LG_full_attention.ckpt",
    # Production protein-ligand LeFlur — currently s3:// in inference.py.
    "LG Protein Ligand fsq 4375": (f"{LG_HF_PREFIX}/LG_Protein_Ligand_fsq_4375_2026-01-05.ckpt"),
    # Continuous protein-ligand variant — currently /cv/scratch/ in inference.py.
    "LG Protein Ligand cont": (f"{LG_HF_PREFIX}/LG_Protein_Ligand_continuous_2026-01-24.ckpt"),
}

_paired_lg_overrides_installed = False


def install_paired_lg_codec_overrides() -> dict[str, str]:
    """Rewrite the LG ``methods`` registry to use HF URLs for LeFlur codecs.

    Safe to call multiple times: the second and subsequent calls are
    no-ops. Returns a mapping ``codec_name -> previous_checkpoint`` so
    callers can audit what was overridden (useful for tests).

    Notes
    -----
    This is import-time wiring used by the LeFlur Lightning modules. It
    intentionally does **not** touch codecs LeFlur doesn't depend on — the
    LG library remains usable standalone for any researcher who wants the
    original internal paths.
    """
    global _paired_lg_overrides_installed
    if _paired_lg_overrides_installed:
        return {}

    # Lazy import — keeps the LeFlur public surface from pulling LG in
    # at module-load time for users who don't need it.
    from lobster.model.latent_generator.cmdline import methods as lg_methods

    previous: dict[str, str] = {}
    for codec_name, hf_url in PAIRED_LG_CODECS.items():
        entry = lg_methods.get(codec_name)
        if entry is None:
            logger.debug(
                "install_paired_lg_codec_overrides: codec %r not present in "
                "latent_generator methods registry, skipping.",
                codec_name,
            )
            continue
        current = entry.model_config.checkpoint
        if current == hf_url:
            continue
        previous[codec_name] = current
        entry.model_config.checkpoint = hf_url
        logger.info(
            "Patched LG codec %r checkpoint: %s -> %s",
            codec_name,
            current,
            hf_url,
        )

    _paired_lg_overrides_installed = True
    return previous


# --- Resolver --------------------------------------------------------------


def _is_hf_uri(value: str) -> bool:
    return value.startswith("hf://") or value.startswith("https://huggingface.co/")


def _is_local_path(value: str) -> bool:
    # Anything that starts with `/`, `./`, `~/`, or is an existing file.
    return (
        value.startswith("/")
        or value.startswith("./")
        or value.startswith("../")
        or value.startswith("~")
        or Path(value).exists()
    )


def _parse_hf_uri(uri: str) -> tuple[str, str, str]:
    """Return ``(repo_id, revision, filename)`` for an HF URI.

    Accepts both ``hf://owner/repo/path/to/file`` and
    ``https://huggingface.co/owner/repo/resolve/<rev>/path/to/file``.
    Defaults revision to :data:`HF_REVISION`.
    """
    if uri.startswith("hf://"):
        body = uri[len("hf://") :]
        parts = body.split("/", 2)
        if len(parts) < 3:
            raise ValueError(f"hf:// URI must be hf://<owner>/<repo>/<path>, got {uri!r}")
        owner, repo, filename = parts
        return f"{owner}/{repo}", HF_REVISION, filename

    if uri.startswith("https://huggingface.co/"):
        parsed = urllib.parse.urlparse(uri)
        # /<owner>/<repo>/{resolve,blob}/<rev>/<filename>
        segments = parsed.path.strip("/").split("/")
        if len(segments) < 5 or segments[2] not in {"resolve", "blob"}:
            raise ValueError(
                f"https://huggingface.co/... URL must be "
                f"https://huggingface.co/<owner>/<repo>/resolve/<rev>/<path>, "
                f"got {uri!r}"
            )
        owner, repo = segments[0], segments[1]
        revision = segments[3]
        filename = "/".join(segments[4:])
        return f"{owner}/{repo}", revision, filename

    raise ValueError(f"Not an HF URI: {uri!r}")


def _hf_hub_download(repo_id: str, filename: str, revision: str) -> Path:
    """Wrap :func:`huggingface_hub.hf_hub_download` with our cache layout."""
    from huggingface_hub import hf_hub_download  # imported lazily for speed

    cache_dir = _cache_root() / repo_id.replace("/", "__")
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        cache_dir=str(cache_dir),
    )
    return Path(local_path)


def resolve_checkpoint(uri_or_name: str | Path) -> Path:
    """Materialise *uri_or_name* into a concrete local checkpoint path.

    Accepted inputs:

    - **Short name** (e.g. ``"leflur-ted"``) — looked up in
      :data:`KNOWN_CHECKPOINTS` and downloaded from HuggingFace if not
      already cached.
    - **HF URI** (``hf://owner/repo/path/to/file.ckpt`` or
      ``https://huggingface.co/owner/repo/resolve/<rev>/path``) —
      downloaded into the LeFlur cache.
    - **Local path** (``str`` or :class:`pathlib.Path`) — returned verbatim
      after an existence check.

    Parameters
    ----------
    uri_or_name
        See the variants above.

    Returns
    -------
    pathlib.Path
        Absolute path to the local checkpoint file.

    Raises
    ------
    FileNotFoundError
        If a local path was given but the file does not exist.
    ValueError
        If the URI cannot be parsed or the short name is unknown.
    """
    if isinstance(uri_or_name, Path):
        uri_or_name = str(uri_or_name)

    if not isinstance(uri_or_name, str):
        raise TypeError(f"resolve_checkpoint() expects str or Path, got {type(uri_or_name).__name__}")

    value = uri_or_name.strip()

    # 1. Short name — look up the registered HF URI.
    if value in KNOWN_CHECKPOINTS:
        info = KNOWN_CHECKPOINTS[value]
        logger.info("resolve_checkpoint: short name %r -> %s", value, info.hf_uri)
        return _hf_hub_download(HF_REPO_ID, info.hf_path, HF_REVISION)

    # 2. HF URI — download via huggingface_hub.
    if _is_hf_uri(value):
        repo_id, revision, filename = _parse_hf_uri(value)
        logger.info(
            "resolve_checkpoint: HF URI %r -> repo=%s rev=%s filename=%s",
            value,
            repo_id,
            revision,
            filename,
        )
        return _hf_hub_download(repo_id, filename, revision)

    # 3. S3 — out of scope for the public flow. Tell the user how to
    # downgrade.
    if value.startswith("s3://"):
        raise ValueError(
            f"resolve_checkpoint() does not download from S3 in the public "
            f"flow. Either pre-download the file or pass a local path; "
            f"alternatively bring the LeFlur checkpoint into the registry "
            f"via Phase 4 ``KNOWN_CHECKPOINTS``. Offending URI: {value!r}"
        )

    # 4. Local path. Expanduser + existence check.
    path = Path(value).expanduser()
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {path!s}. Pass a short name (one of "
            f"{sorted(KNOWN_CHECKPOINTS)}), an hf:// URI, or an existing "
            f"local path."
        )
    return path.resolve()


# --- Cache management ------------------------------------------------------


def cache_dir() -> Path:
    """Where LeFlur stores downloaded checkpoints."""
    return _cache_root()


def cached_files() -> list[Path]:
    """List every file currently in the LeFlur checkpoint cache."""
    root = _cache_root()
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file())


# --- Upload helpers (for populating Sidney-Lisanza/leflur) --------------


def _entry_for_upload(
    short_name: str,
) -> CheckpointInfo:
    """Return a registry entry that knows where to upload itself."""
    if short_name in KNOWN_CHECKPOINTS:
        return KNOWN_CHECKPOINTS[short_name]
    if short_name in PAIRED_LG_CHECKPOINTS:
        return PAIRED_LG_CHECKPOINTS[short_name]
    raise ValueError(
        f"Unknown checkpoint {short_name!r}. Choose from "
        f"{sorted(list(KNOWN_CHECKPOINTS) + list(PAIRED_LG_CHECKPOINTS))}"
    )


def upload_checkpoint(
    short_name: str,
    *,
    source_path: str | Path | None = None,
    repo_id: str | None = None,
    revision: str = HF_REVISION,
    token: str | None = None,
    dry_run: bool = False,
    create_repo_if_missing: bool = True,
    commit_message: str | None = None,
) -> dict[str, str]:
    """Upload a registered checkpoint to HuggingFace.

    Wraps :meth:`huggingface_hub.HfApi.upload_file` so the heavy lifting
    (multipart, LFS, retries) stays in HF land.

    Parameters
    ----------
    short_name
        Either a :data:`KNOWN_CHECKPOINTS` key or a
        :data:`PAIRED_LG_CHECKPOINTS` key. The entry's ``hf_repo_id``,
        ``hf_path``, and ``local_source_path`` drive the upload.
    source_path
        Override the registered ``local_source_path``. Useful for re-uploading
        a fresh copy without editing the registry.
    repo_id, revision
        Override the registered ``hf_repo_id`` / default revision.
    token
        HF token. Falls back to ``$HF_TOKEN`` /
        :func:`huggingface_hub.HfFolder.get_token`.
    dry_run
        If ``True``, log the planned upload but make no network calls.
    create_repo_if_missing
        Create the target HF repo before upload if it doesn't exist yet.
    commit_message
        Override the auto-generated commit message.

    Returns
    -------
    dict
        Summary with ``short_name``, ``repo_id``, ``hf_path``,
        ``source_path``, ``dry_run``, ``commit_url`` (or empty on dry-run).
    """
    from huggingface_hub import HfApi
    from huggingface_hub.utils import RepositoryNotFoundError

    info = _entry_for_upload(short_name)

    src = Path(source_path) if source_path else (Path(info.local_source_path) if info.local_source_path else None)
    if src is None:
        raise ValueError(f"{short_name!r} has no local_source_path registered; pass --source explicitly.")
    src = src.expanduser()
    if not src.exists():
        raise FileNotFoundError(
            f"Source checkpoint not found at {src}. Pass a different --source or update the registry."
        )

    target_repo = repo_id or info.hf_repo_id
    summary = {
        "short_name": short_name,
        "repo_id": target_repo,
        "revision": revision,
        "hf_path": info.hf_path,
        "source_path": str(src),
        "size_bytes": str(src.stat().st_size),
        "dry_run": str(dry_run),
        "commit_url": "",
    }
    logger.info(
        "upload_checkpoint%s %s (%s) -> %s/%s@%s",
        " [dry-run]" if dry_run else "",
        short_name,
        src,
        target_repo,
        info.hf_path,
        revision,
    )
    if dry_run:
        return summary

    api = HfApi(token=token)
    if create_repo_if_missing:
        try:
            api.repo_info(repo_id=target_repo, repo_type="model")
        except RepositoryNotFoundError:
            logger.info("Creating HF repo %s (model)", target_repo)
            api.create_repo(repo_id=target_repo, repo_type="model", exist_ok=True)

    commit_info = api.upload_file(
        path_or_fileobj=str(src),
        path_in_repo=info.hf_path,
        repo_id=target_repo,
        repo_type="model",
        revision=revision,
        commit_message=(commit_message or f"Upload {short_name} ({src.name})"),
    )
    summary["commit_url"] = getattr(commit_info, "commit_url", "") or str(commit_info)
    return summary


def clear_cache(*, dry_run: bool = False) -> Iterable[Path]:
    """Delete the LeFlur checkpoint cache. Yields the deleted paths.

    Parameters
    ----------
    dry_run
        If ``True``, yield what *would* be removed without touching disk.
    """
    root = _cache_root()
    if not root.exists():
        return []

    deleted: list[Path] = []
    for child in sorted(root.iterdir()):
        if dry_run:
            deleted.append(child)
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
        deleted.append(child)
    return deleted
