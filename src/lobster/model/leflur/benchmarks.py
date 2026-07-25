"""LeFlur benchmark dataset registry + resolver.

Companion to :mod:`lobster.model.leflur.checkpoints`. Where ``checkpoints``
hosts publication model weights, this module hosts the small pre-tokenized
``.pt`` benchmark inputs that the LeFlur evaluation pipelines consume —
CAMEO 2022, MultiFlow test, PoseBusters splits, and so on.

Public surface
--------------

- :data:`KNOWN_BENCHMARKS` — short-name → :class:`BenchmarkInfo` registry.
- :func:`resolve_benchmark` — turns any of (short name, ``hf-dataset://`` URI,
  local directory) into a concrete local directory containing the ``.pt``
  fixtures the generate-mode helpers glob.
- :func:`list_benchmarks` — pretty-printable copy of the registry.
- :func:`fetch_benchmark` — eagerly snapshot-download a benchmark into the
  LeFlur cache; surfaced by the ``lobster_leflur_benchmarks fetch`` CLI.
- :func:`upload_benchmark` — populate the LeFlur dataset repo from a local
  source dir (with optional ``.pt`` rewriting to strip internal paths).
- :func:`clear_benchmark_cache` — drop the cached HF snapshots.

Cache layout
~~~~~~~~~~~~

Downloads land under ``$LOBSTER_CACHE`` (default ``~/.cache/lobster/leflur``)
in a ``benchmarks/<short-name>/`` subdir. Each benchmark short-name maps
to a subdirectory under the canonical HF dataset repo
(``datasets/Sidney-Lisanza/leflur``). For example,
``KNOWN_BENCHMARKS['cameo']`` lives on HF as
``datasets/Sidney-Lisanza/leflur/cameo-2022/*.pt`` and lands locally at
``${paths.cache_dir}/benchmarks/cameo/*.pt`` — which is exactly the path
``hydra_config/paths/public.yaml`` already interpolates from
``${paths.benchmarks.cameo}``.

Scope
~~~~~

Initial publication scope covers the two protein-only forward/inverse
folding benchmarks referenced by the LeFlur paper's CAMEO and MultiFlow
tables. PoseBusters splits and any future benchmarks slot in as new
``BenchmarkInfo`` entries; no source changes to the generate-mode helpers
are required because the registry's ``cache_subdir`` matches
``paths.benchmarks.<name>`` by construction.
"""

from __future__ import annotations

import logging
import os
import shutil
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# --- Constants -------------------------------------------------------------

HF_DATASET_REPO_ID = "Sidney-Lisanza/leflur"
HF_DATASET_REVISION = "main"
HF_DATASET_REPO_TYPE = "dataset"

# Mirrors :mod:`lobster.model.leflur.checkpoints` — same root, sibling
# ``benchmarks/`` subdir to keep one cache for everything LeFlur-related.
_DEFAULT_CACHE = Path.home() / ".cache" / "lobster" / "leflur"


def _cache_root() -> Path:
    """Root of the LeFlur benchmark cache (``$LOBSTER_CACHE/benchmarks``)."""
    return Path(os.environ.get("LOBSTER_CACHE", _DEFAULT_CACHE)) / "benchmarks"


# --- Registry --------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkInfo:
    """Metadata for a publicly-distributed LeFlur benchmark dataset.

    Attributes
    ----------
    short_name
        Stable identifier (``cameo``, ``multiflow_test``, ...). Mirrors the
        key under ``${paths.benchmarks}`` in
        ``src/lobster/hydra_config/paths/{internal,public}.yaml``.
    hf_subdir
        Path *within* the HF dataset repo (``cameo-2022``,
        ``multiflow-test``, ...). ``snapshot_download`` pulls every file
        under this prefix into :attr:`cache_subdir`.
    cache_subdir
        Local cache subdirectory under :func:`_cache_root` to materialise
        into. Defaults to :attr:`short_name`. The combination of
        ``$LOBSTER_CACHE/benchmarks/<cache_subdir>`` matches the path
        ``paths/public.yaml`` already maps to ``${paths.benchmarks.<short_name>}``.
    description
        One-line human-readable summary. Surfaced by
        ``lobster_leflur_benchmarks list``.
    citation
        Source citation (paper / website) shown by ``inspect``.
    license
        License string for the underlying dataset (e.g.
        ``"CC-BY-4.0 (per-target PDB licenses apply)"``).
    schema_keys
        Tuple of keys present in each ``.pt`` file's top-level ``dict``.
        Useful for callers that want to validate before tokenizing.
    pattern
        Glob (relative to the resolved benchmark dir) that the generate-mode
        helpers consume. Defaults to ``"*.pt"``.
    tags
        Free-form labels (``"canonical"``, ``"protein"``, ``"protein-ligand"``).
    hf_repo_id
        HF dataset repo this benchmark lives in. Defaults to
        :data:`HF_DATASET_REPO_ID`.
    local_source_path
        Internal ``/cv/...`` source the upload CLI reads from. Optional —
        absent for non-uploadable / external-only entries.
    """

    short_name: str
    hf_subdir: str
    description: str
    citation: str = ""
    license: str = ""
    schema_keys: tuple[str, ...] = field(default_factory=tuple)
    pattern: str = "*.pt"
    tags: tuple[str, ...] = field(default_factory=tuple)
    cache_subdir: str = ""
    hf_repo_id: str = ""
    local_source_path: str | None = None

    def __post_init__(self) -> None:
        if not self.hf_repo_id:
            object.__setattr__(self, "hf_repo_id", HF_DATASET_REPO_ID)
        if not self.cache_subdir:
            object.__setattr__(self, "cache_subdir", self.short_name)

    @property
    def hf_uri(self) -> str:
        """Canonical ``hf-dataset://`` URI (documentation / debugging)."""
        return f"hf-dataset://{self.hf_repo_id}/{self.hf_subdir}"

    @property
    def https_url(self) -> str:
        """Direct browse URL for the subfolder on huggingface.co."""
        return f"https://huggingface.co/datasets/{self.hf_repo_id}/tree/{HF_DATASET_REVISION}/{self.hf_subdir}"


# Registry. Keys deliberately match ``paths.benchmarks.<name>`` so the
# Hydra path overlay continues to interpolate directly:
#
#     ${paths.cache_dir}/benchmarks/cameo  ==  cache_subdir of "cameo"
#
# Adding a new benchmark is a non-breaking, two-line change here +
# (optionally) a new ``cache_subdir`` rule under ``paths/public.yaml``.
KNOWN_BENCHMARKS: dict[str, BenchmarkInfo] = {
    "cameo": BenchmarkInfo(
        short_name="cameo",
        hf_subdir="cameo-2022",
        description=(
            "CAMEO 2022 monomer benchmark, pre-tokenized into one .pt per "
            "target. Drives the CAMEO 2022 rows of LeFlur Table 1 (inverse "
            "folding) and Table 3 (forward folding)."
        ),
        citation=(
            "Haas et al., 'Continuous Automated Model EvaluatiOn (CAMEO) "
            "complementing the critical assessment of structure prediction "
            "in CASP12', Proteins (2018). Per-target PDB structures are "
            "released under their original PDB / RCSB licenses."
        ),
        license="academic; per-target PDB licenses apply",
        schema_keys=(
            "pdb_path",
            "sequence",
            "sequence_str",
            "coords_res",
            "chains_ids",
            "indices",
            "mask",
            "real_chains",
        ),
        tags=("canonical", "protein", "publication"),
        local_source_path="/cv/data/ai4dd/data2/lisanzas/AFDB/valid_cameo_processed",
    ),
    "multiflow_test": BenchmarkInfo(
        short_name="multiflow_test",
        hf_subdir="multiflow-test",
        description=(
            "Filtered MultiFlow test set, pre-tokenized into one .pt per "
            "target. Drives the MultiFlow rows of LeFlur Table 1 (inverse "
            "folding) and Table 3 (forward folding)."
        ),
        citation=(
            "Campbell et al., 'Generative Flows on Discrete State-Spaces: "
            "Enabling Multimodal Flows with Applications to Protein Co-Design', "
            "ICML 2024."
        ),
        license="MIT (matches upstream MultiFlow data release)",
        schema_keys=("sequence", "coords_res", "mask", "indices", "chains"),
        tags=("canonical", "protein", "publication"),
        local_source_path="/cv/data/ai4dd/data2/lisanzas/multi_flow_data/test_set_filtered_pt",
    ),
    "posebusters_benchmark_no_overlap": BenchmarkInfo(
        short_name="posebusters_benchmark_no_overlap",
        hf_subdir="posebusters-benchmark-no-overlap",
        description=(
            "PoseBusters benchmark, deduplicated against the LeFlur "
            "training set (the 'no-overlap' subset). Pre-tokenized as "
            "paired <id>_<ligand>_{protein,ligand}.pt files. Drives the "
            "publication PoseBusters rows of LeFlur Table 2 (PL inverse "
            "folding) and Table 4 (PL forward folding) — this is the "
            "canonical leflur-pl evaluation set."
        ),
        citation=(
            "Buttenschoen et al., 'PoseBusters: AI-based docking methods "
            "fail to generate physically valid poses or generalise to novel "
            "sequences', Chem. Sci. (2024). 'no-overlap' filtering removes "
            "targets that share clusters with the LeFlur protein-ligand "
            "training set, isolating true held-out generalisation."
        ),
        license="CC-BY-4.0 (matches upstream PoseBusters release)",
        schema_keys=(
            # union of protein + ligand record keys (each file holds one)
            "pdb_path",
            "sequence",
            "sequence_str",
            "coords_res",
            "chains_ids",
            "indices",
            "mask",
            "real_chains",
            "atom_names",
            "atom_coords",
            "atom_indices",
            "element_indices",
            "bond_matrix",
        ),
        tags=("canonical", "protein-ligand", "publication"),
        local_source_path="/cv/home/lisanzas/lobster/data/posebusters/processed/posebusters_benchmark_no_overlap",
    ),
    "posebusters_benchmark": BenchmarkInfo(
        short_name="posebusters_benchmark",
        hf_subdir="posebusters-benchmark",
        description=(
            "Full PoseBusters benchmark (no overlap filtering applied), "
            "pre-tokenized as paired <id>_<ligand>_{protein,ligand}.pt "
            "files. Reported in supplementary tables alongside the "
            "deduplicated 'no_overlap' subset — most publication numbers "
            "use the 'no_overlap' variant."
        ),
        citation=(
            "Buttenschoen et al., 'PoseBusters: AI-based docking methods "
            "fail to generate physically valid poses or generalise to novel "
            "sequences', Chem. Sci. (2024)."
        ),
        license="CC-BY-4.0 (matches upstream PoseBusters release)",
        schema_keys=(
            "pdb_path",
            "sequence",
            "sequence_str",
            "coords_res",
            "chains_ids",
            "indices",
            "mask",
            "real_chains",
            "atom_names",
            "atom_coords",
            "atom_indices",
            "element_indices",
            "bond_matrix",
        ),
        tags=("protein-ligand",),
        local_source_path="/cv/home/lisanzas/lobster/data/posebusters/processed/posebusters_benchmark",
    ),
    "complexa-binder": BenchmarkInfo(
        short_name="complexa-binder",
        hf_subdir="complexa-binder",
        description=(
            "Complexa de-novo binder-design benchmark: 38 therapeutic target "
            "antigens (target PDB + epitope residues + deep MSA per antigen). "
            "Drives the LeFlur binder-design evaluation (PASS = pTM>0.80 AND "
            "ipTM>0.70 by Protenix co-folding). Ships two manifests: "
            "complexa_gen_targets.csv (generation: pdb/chain/epitope/binder-len "
            "range) and complexa_score_targets.csv (scoring: antigen seq + a3m). "
            "Paths in both manifests are relative to the benchmark dir. Run with "
            "examples/run_complexa_binder.py."
        ),
        citation=(
            "Complexa de-novo binder benchmark (this work). Per-target antigen "
            "structures derive from their original PDB / RCSB entries and are "
            "subject to those licenses."
        ),
        license="academic; per-target PDB licenses apply",
        schema_keys=(
            "target_id",
            "pdb_path",
            "target_chain",
            "epitope_indices",
            "binder_len_min",
            "binder_len_max",
        ),
        pattern="complexa_gen_targets.csv",
        tags=("binder", "complex", "publication"),
        local_source_path=("/cv/scratch/u/lisanzas/denovo_dataset/binder/denovo/complexa_bench/targets/hf_export"),
    ),
}


def list_benchmarks(tag: str | None = None) -> list[BenchmarkInfo]:
    """Return registry entries, optionally filtered by ``tag``."""
    entries = list(KNOWN_BENCHMARKS.values())
    if tag is not None:
        entries = [e for e in entries if tag in e.tags]
    return entries


# --- Resolver --------------------------------------------------------------


def _parse_hf_dataset_uri(uri: str) -> tuple[str, str]:
    """Return ``(repo_id, subdir)`` for an ``hf-dataset://`` URI.

    Accepts ``hf-dataset://owner/repo`` (no subdir, whole-repo snapshot)
    or ``hf-dataset://owner/repo/path/to/subdir``.
    """
    if not uri.startswith("hf-dataset://"):
        raise ValueError(f"Not an hf-dataset:// URI: {uri!r}")
    body = uri[len("hf-dataset://") :]
    parts = body.split("/", 2)
    if len(parts) < 2:
        raise ValueError(f"hf-dataset:// URI must be hf-dataset://<owner>/<repo>[/<subdir>], got {uri!r}")
    owner, repo = parts[0], parts[1]
    subdir = parts[2] if len(parts) >= 3 else ""
    return f"{owner}/{repo}", subdir


def _snapshot_download(
    repo_id: str,
    subdir: str,
    *,
    local_dir: Path,
    revision: str = HF_DATASET_REVISION,
) -> Path:
    """Snapshot *subdir* from *repo_id* into *local_dir* as a flat directory.

    Wraps :func:`huggingface_hub.snapshot_download` (with ``allow_patterns``
    scoped to ``subdir/*``) and then lifts the downloaded files **up** out
    of the HF-managed ``subdir/`` nesting so the returned directory matches
    the flat layout ``paths/public.yaml`` expects (one ``*.pt`` per target
    directly under ``${paths.cache_dir}/benchmarks/<short_name>/``).

    Returns the local directory containing the lifted files (== ``local_dir``).
    """
    from huggingface_hub import snapshot_download  # imported lazily

    local_dir.mkdir(parents=True, exist_ok=True)
    allow = f"{subdir.rstrip('/')}/*" if subdir else None
    snapshot_download(
        repo_id=repo_id,
        repo_type=HF_DATASET_REPO_TYPE,
        revision=revision,
        local_dir=str(local_dir),
        allow_patterns=allow,
    )

    if subdir:
        # snapshot_download mirrors the repo layout, so files land at
        # ``local_dir/<subdir>/*``. Lift them up one level so callers can
        # glob ``local_dir/*.pt`` (matching ``paths.benchmarks.<name>``).
        nested = local_dir / subdir
        if nested.is_dir():
            for f in nested.iterdir():
                target = local_dir / f.name
                if target.exists():
                    if f.is_dir():
                        shutil.rmtree(f)
                    else:
                        f.unlink()
                    continue
                f.rename(target)
            # Drop empty HF-managed subdirs left behind by the lift.
            try:
                nested.rmdir()
            except OSError:
                pass
    return local_dir


def resolve_benchmark(uri_or_name: str | Path) -> Path:
    """Materialise *uri_or_name* into a concrete local benchmark directory.

    Accepted inputs:

    - **Short name** (e.g. ``"cameo"``) — looked up in
      :data:`KNOWN_BENCHMARKS` and snapshot-downloaded from HuggingFace
      into ``${LOBSTER_CACHE}/benchmarks/<short-name>/`` if not already
      cached.
    - **hf-dataset URI** (``hf-dataset://owner/repo[/subdir]``) — downloaded
      into the LeFlur benchmark cache.
    - **Local path** (``str`` or :class:`pathlib.Path`) — returned verbatim
      after an existence check.

    Returns
    -------
    pathlib.Path
        Absolute path to the local directory containing the benchmark
        ``.pt`` files (or whatever pattern the registry entry advertises).

    Raises
    ------
    FileNotFoundError
        If a local path was given but the directory does not exist.
    ValueError
        If the URI cannot be parsed or the short name is unknown.
    """
    if isinstance(uri_or_name, Path):
        uri_or_name = str(uri_or_name)

    if not isinstance(uri_or_name, str):
        raise TypeError(f"resolve_benchmark() expects str or Path, got {type(uri_or_name).__name__}")

    value = uri_or_name.strip()

    if value in KNOWN_BENCHMARKS:
        info = KNOWN_BENCHMARKS[value]
        target = _cache_root() / info.cache_subdir
        if target.exists() and any(target.glob(info.pattern)):
            logger.info("resolve_benchmark: cache hit for %r -> %s", value, target)
            return target.resolve()
        logger.info(
            "resolve_benchmark: short name %r -> %s (downloading)",
            value,
            info.hf_uri,
        )
        return _snapshot_download(
            info.hf_repo_id,
            info.hf_subdir,
            local_dir=_cache_root() / info.cache_subdir,
        ).resolve()

    if value.startswith("hf-dataset://"):
        repo_id, subdir = _parse_hf_dataset_uri(value)
        target = _cache_root() / repo_id.replace("/", "__") / (subdir or "_root")
        logger.info(
            "resolve_benchmark: HF URI %r -> repo=%s subdir=%s -> %s",
            value,
            repo_id,
            subdir or "<root>",
            target,
        )
        return _snapshot_download(repo_id, subdir, local_dir=target).resolve()

    if value.startswith("s3://"):
        raise ValueError(
            f"resolve_benchmark() does not download from S3 in the public "
            f"flow. Either pre-download the directory or pass a local path. "
            f"Offending URI: {value!r}"
        )

    path = Path(value).expanduser()
    if not path.exists():
        raise FileNotFoundError(
            f"Benchmark directory not found at {path!s}. Pass a short name "
            f"(one of {sorted(KNOWN_BENCHMARKS)}), an hf-dataset:// URI, "
            f"or an existing local directory."
        )
    return path.resolve()


# --- Convenience entry points ---------------------------------------------


def fetch_benchmark(short_name: str) -> Path:
    """Eagerly resolve *short_name* and return the local dir.

    Thin wrapper around :func:`resolve_benchmark` whose only job is to
    raise a friendlier error for unknown short names. Used by
    ``lobster_leflur_benchmarks fetch``.
    """
    if short_name not in KNOWN_BENCHMARKS:
        raise ValueError(f"Unknown benchmark {short_name!r}. Known names: {sorted(KNOWN_BENCHMARKS)}")
    return resolve_benchmark(short_name)


def cache_dir() -> Path:
    """Where LeFlur stores downloaded benchmark snapshots."""
    return _cache_root()


def cached_files() -> list[Path]:
    """List every file currently in the LeFlur benchmark cache."""
    root = _cache_root()
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file())


def clear_benchmark_cache(*, dry_run: bool = False) -> Iterable[Path]:
    """Delete the LeFlur benchmark cache. Yields the deleted paths.

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


# --- Upload helpers --------------------------------------------------------


def _sanitize_record(record: dict) -> dict:
    """Strip internal-only path fields from a benchmark ``.pt`` payload.

    Rewrites the ``pdb_path`` absolute path (which hard-codes the
    Genentech-internal ``/cv/data/...`` or ``/cv/home/...`` location)
    down to its basename so downstream code that looks up the
    corresponding raw PDB by filename still works. All other keys are
    passed through untouched.

    Used uniformly for CAMEO (protein), PoseBusters (paired protein +
    ligand), and any future benchmark with a ``pdb_path`` field.
    MultiFlow records have a leaner schema and are returned unchanged.
    """
    if not isinstance(record, dict):
        return record
    if "pdb_path" not in record:
        return record
    rewritten = dict(record)
    pdb_path = rewritten["pdb_path"]
    if isinstance(pdb_path, str):
        rewritten["pdb_path"] = Path(pdb_path).name
    return rewritten


# Alias preserved for back-compat with any external code (and tests).
_sanitize_cameo_record = _sanitize_record


def generate_dataset_card_md(repo_id: str = HF_DATASET_REPO_ID) -> str:
    """Build the dataset-card README.md for the LeFlur HF dataset repo.

    Pure function over :data:`KNOWN_BENCHMARKS` — running it from a clean
    checkout always gives the same output. The front-matter follows the
    HuggingFace `dataset card spec
    <https://huggingface.co/docs/hub/datasets-cards>`_; the body links
    to the model repo of the same name and documents the per-subdir
    schema each benchmark advertises.
    """
    licenses = sorted({info.license for info in KNOWN_BENCHMARKS.values() if info.license})
    license_block = licenses[0] if len(licenses) == 1 else "other"

    front_matter_lines = [
        "---",
        "pretty_name: LeFlur Benchmarks",
        f"license: {license_block}",
        "task_categories:",
        "  - other",
        "tags:",
        "  - protein",
        "  - protein-structure",
        "  - inverse-folding",
        "  - forward-folding",
        "configs:",
    ]
    for info in KNOWN_BENCHMARKS.values():
        front_matter_lines.extend(
            [
                f"  - config_name: {info.short_name}",
                "    data_files:",
                "      - split: test",
                f'        path: "{info.hf_subdir}/*.pt"',
            ]
        )
    front_matter_lines.append("---")

    body_lines = [
        "",
        "# LeFlur Benchmarks",
        "",
        "Pre-tokenized protein structure benchmarks used to evaluate the",
        "[LeFlur](https://huggingface.co/" + repo_id + ") protein and",
        "protein-ligand generative model. Each subdirectory contains one",
        "`.pt` file per target, ready to be consumed by the `lobster_generate`",
        "forward/inverse folding pipelines via the `${paths.benchmarks.<name>}`",
        "Hydra interpolation in `paths/public.yaml`.",
        "",
        "## Quickstart",
        "",
        "```bash",
        "pip install lbster",
        "lobster_leflur_benchmarks fetch cameo",
        "lobster_leflur_benchmarks fetch multiflow_test",
        "",
        "# Inverse folding on the CAMEO 2022 monomer benchmark",
        "lobster_generate --config-name experiment/generate_inverse_folding \\",
        "    paths=public model.ckpt_path=leflur-ted \\",
        "    'generation.input_structures=${paths.benchmarks.cameo}/*.pt'",
        "```",
        "",
        "## Benchmarks",
        "",
    ]
    for info in KNOWN_BENCHMARKS.values():
        body_lines.extend(
            [
                f"### `{info.short_name}` (`{info.hf_subdir}/`)",
                "",
                info.description,
                "",
                f"- **License**: {info.license or 'see citation'}",
                f"- **Citation**: {info.citation or '-'}",
                f"- **File glob**: `{info.hf_subdir}/{info.pattern}`",
                "- **Per-record schema** (top-level keys of each `.pt`):",
            ]
        )
        for key in info.schema_keys:
            body_lines.append(f"  - `{key}`")
        body_lines.append("")

    body_lines.extend(
        [
            "## Notes on the CAMEO `pdb_path` field",
            "",
            "The CAMEO records expose a `pdb_path` string that names the",
            "source PDB file (e.g. `7dz2.C.pdb`). Only the basename is",
            "published here — the absolute path inside the original",
            "Genentech file tree has been stripped on upload. Downstream code",
            "that wants to load the raw PDB should look it up against an",
            "RCSB / CAMEO mirror by basename.",
            "",
            "## See also",
            "",
            f"- [`{repo_id}`](https://huggingface.co/{repo_id}) — the LeFlur",
            "  model repo (3 canonical checkpoints).",
            "- [LeFlur README](https://github.com/prescient-design/lobster/blob/main/src/lobster/model/leflur/README.md)",
            "  — paper tables + one-line reproduction commands.",
            "",
        ]
    )

    return "\n".join(front_matter_lines + body_lines) + "\n"


def upload_dataset_card(
    *,
    repo_id: str | None = None,
    revision: str = HF_DATASET_REVISION,
    token: str | None = None,
    dry_run: bool = False,
    create_repo_if_missing: bool = True,
    commit_message: str | None = None,
) -> dict[str, str]:
    """Upload (or refresh) the LeFlur benchmarks dataset card README.md.

    Builds the body via :func:`generate_dataset_card_md` so the published
    card always tracks the registry. Returns a summary identical in shape
    to :func:`upload_benchmark` so the CLI can format both uniformly.
    """
    from huggingface_hub import HfApi
    from huggingface_hub.utils import RepositoryNotFoundError

    target_repo = repo_id or HF_DATASET_REPO_ID
    body = generate_dataset_card_md(target_repo)
    summary = {
        "short_name": "<dataset-card>",
        "repo_id": target_repo,
        "repo_type": HF_DATASET_REPO_TYPE,
        "revision": revision,
        "hf_subdir": "README.md",
        "source_dir": "<generated>",
        "num_files": "1",
        "total_bytes": str(len(body.encode("utf-8"))),
        "dry_run": str(dry_run),
        "sanitize": "n/a",
        "commit_url": "",
    }
    logger.info(
        "upload_dataset_card%s (%d bytes) -> %s/README.md@%s",
        " [dry-run]" if dry_run else "",
        len(body.encode("utf-8")),
        target_repo,
        revision,
    )
    if dry_run:
        return summary

    api = HfApi(token=token)
    if create_repo_if_missing:
        try:
            api.repo_info(repo_id=target_repo, repo_type=HF_DATASET_REPO_TYPE)
        except RepositoryNotFoundError:
            logger.info("Creating HF repo %s (dataset)", target_repo)
            api.create_repo(
                repo_id=target_repo,
                repo_type=HF_DATASET_REPO_TYPE,
                exist_ok=True,
            )

    commit_info = api.upload_file(
        path_or_fileobj=body.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=target_repo,
        repo_type=HF_DATASET_REPO_TYPE,
        revision=revision,
        commit_message=(commit_message or "Update LeFlur benchmarks dataset card"),
    )
    summary["commit_url"] = getattr(commit_info, "commit_url", "") or str(commit_info)
    return summary


# Chunk size for `upload_benchmark`. HF's `upload_folder` reliably commits
# batches under ~50 files but the LFS preupload endpoint flakes (401s) on
# larger one-shot uploads of small files — so we always go through chunks.
_UPLOAD_CHUNK_SIZE = 50
_UPLOAD_MAX_RETRIES = 4
_UPLOAD_BACKOFF_SECONDS = (2, 5, 15, 30)  # cumulative wait per retry


def _ensure_dataset_repo(
    api,
    repo_id: str,
) -> None:
    """Make sure *repo_id* exists as a dataset repo. Idempotent."""
    from huggingface_hub.utils import RepositoryNotFoundError

    try:
        api.repo_info(repo_id=repo_id, repo_type=HF_DATASET_REPO_TYPE)
        return  # already exists, nothing to do
    except RepositoryNotFoundError:
        pass

    last_exc: Exception | None = None
    for attempt, sleep_s in enumerate((0, *_UPLOAD_BACKOFF_SECONDS)):
        if sleep_s:
            import time as _time

            _time.sleep(sleep_s)
        try:
            api.create_repo(repo_id=repo_id, repo_type=HF_DATASET_REPO_TYPE, exist_ok=True)
            return
        except Exception as exc:  # transient 401/5xx
            last_exc = exc
            logger.warning(
                "create_repo(%s, dataset) attempt %d failed: %s. Retrying.",
                repo_id,
                attempt + 1,
                str(exc).splitlines()[0],
            )
    raise RuntimeError(
        f"Failed to create dataset repo {repo_id!r} after {_UPLOAD_MAX_RETRIES + 1} attempts. Last error: {last_exc}"
    ) from last_exc


def _upload_chunk_with_retry(
    api,
    *,
    chunk_dir: Path,
    path_in_repo: str,
    repo_id: str,
    revision: str,
    commit_message: str,
) -> str:
    """Upload one chunk-directory via `upload_folder`, retrying on flake.

    HF's preupload endpoint occasionally 401s (mis-formatted as
    ``RepositoryNotFoundError``) on otherwise-valid requests. The
    request body and signature don't change between attempts, so a
    small linear backoff is enough.
    """
    last_exc: Exception | None = None
    for attempt, sleep_s in enumerate((0, *_UPLOAD_BACKOFF_SECONDS)):
        if sleep_s:
            import time as _time

            _time.sleep(sleep_s)
        try:
            commit_info = api.upload_folder(
                folder_path=str(chunk_dir),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type=HF_DATASET_REPO_TYPE,
                revision=revision,
                commit_message=commit_message,
            )
            return getattr(commit_info, "commit_url", "") or str(commit_info)
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "upload_folder(%s/%s) attempt %d failed: %s. Retrying.",
                repo_id,
                path_in_repo,
                attempt + 1,
                str(exc).splitlines()[0],
            )
    raise RuntimeError(
        f"Failed to upload chunk to {repo_id}/{path_in_repo} after "
        f"{_UPLOAD_MAX_RETRIES + 1} attempts. Last error: {last_exc}"
    ) from last_exc


def upload_benchmark(
    short_name: str,
    *,
    source_dir: str | Path | None = None,
    repo_id: str | None = None,
    revision: str = HF_DATASET_REVISION,
    token: str | None = None,
    dry_run: bool = False,
    create_repo_if_missing: bool = True,
    commit_message: str | None = None,
    sanitize: bool = True,
    chunk_size: int = _UPLOAD_CHUNK_SIZE,
) -> dict[str, str]:
    """Upload every ``.pt`` under *source_dir* to ``hf_subdir`` on HF.

    Wraps :meth:`huggingface_hub.HfApi.upload_folder`.

    Parameters
    ----------
    short_name
        Key into :data:`KNOWN_BENCHMARKS`.
    source_dir
        Local source. Defaults to the registry's ``local_source_path``.
    repo_id, revision
        Override the registered ``hf_repo_id`` / default revision.
    token
        HF token. Falls back to ``$HF_TOKEN`` /
        :func:`huggingface_hub.HfFolder.get_token`.
    dry_run
        If ``True``, log the planned upload but make no network calls.
    create_repo_if_missing
        Create the target HF dataset repo before upload if it doesn't
        exist yet.
    commit_message
        Override the auto-generated commit message.
    sanitize
        If ``True`` (default), rewrite each ``.pt`` through
        :func:`_sanitize_cameo_record` before upload, into a temp dir, so
        the published files don't leak internal paths. Set ``False`` to
        upload the source dir bit-identically.

    Returns
    -------
    dict
        Summary with ``short_name``, ``repo_id``, ``hf_subdir``,
        ``source_dir``, ``num_files``, ``total_bytes``, ``dry_run``,
        ``commit_url`` (empty on dry-run).
    """
    import tempfile

    import torch
    from huggingface_hub import HfApi

    if short_name not in KNOWN_BENCHMARKS:
        raise ValueError(f"Unknown benchmark {short_name!r}. Known names: {sorted(KNOWN_BENCHMARKS)}")
    info = KNOWN_BENCHMARKS[short_name]

    src = Path(source_dir) if source_dir else (Path(info.local_source_path) if info.local_source_path else None)
    if src is None:
        raise ValueError(f"{short_name!r} has no local_source_path registered; pass --source explicitly.")
    src = src.expanduser()
    if not src.is_dir():
        raise FileNotFoundError(f"Source directory not found: {src}")

    pt_files = sorted(src.glob(info.pattern))
    if not pt_files:
        raise FileNotFoundError(f"No files matching {info.pattern!r} under {src}. Nothing to upload.")
    total_bytes = sum(p.stat().st_size for p in pt_files)

    target_repo = repo_id or info.hf_repo_id
    summary = {
        "short_name": short_name,
        "repo_id": target_repo,
        "repo_type": HF_DATASET_REPO_TYPE,
        "revision": revision,
        "hf_subdir": info.hf_subdir,
        "source_dir": str(src),
        "num_files": str(len(pt_files)),
        "total_bytes": str(total_bytes),
        "dry_run": str(dry_run),
        "sanitize": str(sanitize),
        "commit_url": "",
    }
    logger.info(
        "upload_benchmark%s %s (%d files, %.1f MiB) -> %s/%s@%s",
        " [dry-run]" if dry_run else "",
        short_name,
        len(pt_files),
        total_bytes / (1024 * 1024),
        target_repo,
        info.hf_subdir,
        revision,
    )
    if dry_run:
        return summary

    api = HfApi(token=token)
    if create_repo_if_missing:
        _ensure_dataset_repo(api, target_repo)

    # Chunk the upload so each `upload_folder` call sees a small,
    # well-formed batch. HF's preupload endpoint occasionally 401s on
    # one-shot uploads of hundreds of small files; chunking + per-chunk
    # retry eliminates that failure mode entirely.
    chunks = [pt_files[i : i + chunk_size] for i in range(0, len(pt_files), chunk_size)]
    summary["num_chunks"] = str(len(chunks))
    summary["chunk_size"] = str(chunk_size)
    last_commit_url = ""

    for chunk_idx, chunk in enumerate(chunks, start=1):
        with tempfile.TemporaryDirectory(prefix=f"leflur-upload-{short_name}-c{chunk_idx}-") as tmpdir:
            staging = Path(tmpdir) / info.hf_subdir
            staging.mkdir(parents=True, exist_ok=True)
            if sanitize:
                for src_pt in chunk:
                    record = torch.load(src_pt, weights_only=False, map_location="cpu")
                    rewritten = _sanitize_record(record)
                    torch.save(rewritten, staging / src_pt.name)
            else:
                for src_pt in chunk:
                    shutil.copy2(src_pt, staging / src_pt.name)

            chunk_msg = (
                f"{commit_message} (chunk {chunk_idx}/{len(chunks)})"
                if commit_message
                else (f"Upload {short_name} benchmark chunk {chunk_idx}/{len(chunks)} ({len(chunk)} files)")
            )
            last_commit_url = _upload_chunk_with_retry(
                api,
                chunk_dir=staging,
                path_in_repo=info.hf_subdir,
                repo_id=target_repo,
                revision=revision,
                commit_message=chunk_msg,
            )
        logger.info(
            "upload_benchmark: %s chunk %d/%d (%d files) -> %s",
            short_name,
            chunk_idx,
            len(chunks),
            len(chunk),
            last_commit_url,
        )

    summary["commit_url"] = last_commit_url
    return summary
