"""Tests for ``lobster.model.leflur.benchmarks``.

Mocks :func:`huggingface_hub.snapshot_download` and :class:`HfApi` so the
suite runs offline. Covers:

* Short-name → HF snapshot download with HF-managed subdir flattening.
* ``hf-dataset://`` URI parsing + download.
* Local-path passthrough (existing dir) + ``FileNotFoundError`` (missing).
* S3 URI rejection with actionable message.
* Cache helpers (``cache_benchmark_dir``, ``cached_benchmark_files``,
  ``clear_benchmark_cache``) without hitting the network.
* Registry sanity: the four canonical short names exist, each
  ``BenchmarkInfo`` has consistent ``hf_uri`` / ``https_url`` derivations,
  and the keys match the keys advertised by ``paths/{internal,public}.yaml``.
* Sanitiser strips absolute ``pdb_path`` down to basename for both the
  protein (CAMEO) and protein-ligand (PoseBusters) schemas, and is a
  no-op for the MultiFlow schema which lacks ``pdb_path``.
* Dataset-card builder covers every registered benchmark.
* Upload helpers (``upload_benchmark``, ``upload_dataset_card``) call
  ``HfApi`` exactly as expected and chunk large uploads correctly.

The CLI is exercised lightly via ``manage_leflur_benchmarks.build_parser``
and ``main`` to confirm subcommand dispatch + exit codes.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest
import torch
import yaml

from lobster.model.leflur import (
    KNOWN_BENCHMARKS,
    BenchmarkInfo,
    cache_benchmark_dir,
    cached_benchmark_files,
    clear_benchmark_cache,
    fetch_benchmark,
    generate_dataset_card_md,
    list_benchmarks,
    resolve_benchmark,
    upload_benchmark,
    upload_dataset_card,
)
from lobster.model.leflur.benchmarks import (
    HF_DATASET_REPO_ID,
    HF_DATASET_REPO_TYPE,
    HF_DATASET_REVISION,
    _sanitize_record,
)


# --- Fixtures --------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    """Point ``$LOBSTER_CACHE`` at a per-test tmpdir so the suite never
    touches the developer's real cache."""
    monkeypatch.setenv("LOBSTER_CACHE", str(tmp_path))
    yield tmp_path


@pytest.fixture
def fake_snapshot_download(monkeypatch):
    """Replace :func:`huggingface_hub.snapshot_download` with a recorder.

    The fake creates a small ``.pt``-shaped fixture under the requested
    ``local_dir`` so the resolver's "lift-files-out-of-nested-subdir"
    branch is exercised end-to-end.
    """
    calls: list[dict] = []

    def _fake(
        *,
        repo_id: str,
        repo_type: str,
        revision: str,
        local_dir: str,
        allow_patterns: str | None = None,
    ) -> str:
        calls.append(
            {
                "repo_id": repo_id,
                "repo_type": repo_type,
                "revision": revision,
                "local_dir": local_dir,
                "allow_patterns": allow_patterns,
            }
        )
        root = Path(local_dir)
        root.mkdir(parents=True, exist_ok=True)
        # Mirror HF's snapshot layout: files land under ``<local_dir>/<subdir>/``.
        if allow_patterns and allow_patterns.endswith("/*"):
            subdir = root / allow_patterns[:-2]
        else:
            subdir = root
        subdir.mkdir(parents=True, exist_ok=True)
        for name in ("a.pt", "b.pt"):
            (subdir / name).write_bytes(b"\x00")
        return str(root)

    monkeypatch.setattr("huggingface_hub.snapshot_download", _fake, raising=True)
    return calls


# --- Registry sanity ------------------------------------------------------


def test_canonical_short_names_registered() -> None:
    """The four publication benchmarks live in the registry."""
    expected = {
        "cameo",
        "multiflow_test",
        "posebusters_benchmark",
        "posebusters_benchmark_no_overlap",
    }
    assert expected.issubset(KNOWN_BENCHMARKS.keys()), f"Registry missing one of {expected - set(KNOWN_BENCHMARKS)}"


def test_registry_keys_match_public_paths_yaml() -> None:
    """Every short name must have a matching ``${paths.benchmarks.X}`` entry.

    Otherwise ``lobster_leflur_benchmarks fetch`` would land files where
    the generate-mode Hydra configs can't see them.
    """
    paths_yaml = Path(__file__).resolve().parents[4] / "src" / "lobster" / "hydra_config" / "paths" / "public.yaml"
    overlay = yaml.safe_load(paths_yaml.read_text())
    benchmark_keys = set(overlay["benchmarks"].keys())
    for short_name in KNOWN_BENCHMARKS:
        assert short_name in benchmark_keys, (
            f"{short_name!r} in KNOWN_BENCHMARKS but missing from paths/public.yaml ${{paths.benchmarks.{short_name}}}"
        )


def test_hf_uri_and_https_url_match_subdir() -> None:
    """``hf_uri`` and ``https_url`` derive consistently from registry fields."""
    for info in KNOWN_BENCHMARKS.values():
        assert info.hf_uri == (f"hf-dataset://{info.hf_repo_id}/{info.hf_subdir}")
        assert info.https_url == (
            f"https://huggingface.co/datasets/{info.hf_repo_id}/tree/{HF_DATASET_REVISION}/{info.hf_subdir}"
        )


def test_benchmark_info_is_frozen() -> None:
    """`BenchmarkInfo` is a frozen dataclass — regression for mutation bugs."""
    from dataclasses import FrozenInstanceError

    info = BenchmarkInfo(
        short_name="x",
        hf_subdir="x-subdir",
        description="d",
    )
    with pytest.raises(FrozenInstanceError):
        info.short_name = "y"  # type: ignore[misc]


def test_benchmark_info_defaults_post_init() -> None:
    """`cache_subdir` and `hf_repo_id` default sensibly when omitted."""
    info = BenchmarkInfo(short_name="x", hf_subdir="y", description="d")
    assert info.cache_subdir == "x"
    assert info.hf_repo_id == HF_DATASET_REPO_ID


def test_list_benchmarks_tag_filter() -> None:
    canonical = list_benchmarks(tag="canonical")
    pl = list_benchmarks(tag="protein-ligand")
    assert all("canonical" in info.tags for info in canonical)
    assert all("protein-ligand" in info.tags for info in pl)
    # ``cameo`` is canonical-protein, never protein-ligand; sanity-check
    assert any(i.short_name == "cameo" for i in canonical)
    assert all(i.short_name != "cameo" for i in pl)


# --- Resolver: short names -----------------------------------------------


def test_resolve_short_name_triggers_snapshot_download(fake_snapshot_download, tmp_path) -> None:
    info = KNOWN_BENCHMARKS["cameo"]
    path = resolve_benchmark("cameo")
    assert path.is_dir()
    assert fake_snapshot_download[0]["repo_id"] == HF_DATASET_REPO_ID
    assert fake_snapshot_download[0]["repo_type"] == HF_DATASET_REPO_TYPE
    assert fake_snapshot_download[0]["allow_patterns"] == f"{info.hf_subdir}/*"
    # Files were lifted out of the HF-nested subdir.
    files = sorted(p.name for p in path.glob("*.pt"))
    assert files == ["a.pt", "b.pt"]
    assert not (path / info.hf_subdir).exists()


def test_resolve_short_name_cache_hit_skips_download(fake_snapshot_download, tmp_path) -> None:
    """Cache hit avoids a second snapshot download."""
    info = KNOWN_BENCHMARKS["multiflow_test"]
    cache_root = cache_benchmark_dir() / info.cache_subdir
    cache_root.mkdir(parents=True, exist_ok=True)
    (cache_root / "already.pt").write_bytes(b"\x00")

    path = resolve_benchmark("multiflow_test")
    assert path == cache_root.resolve()
    assert fake_snapshot_download == [], "cache hit should not download"


def test_resolve_short_name_handles_whitespace(
    fake_snapshot_download,
) -> None:
    resolve_benchmark("  cameo ")
    assert fake_snapshot_download, "whitespace-padded short name should resolve"


def test_fetch_unknown_short_name_raises() -> None:
    with pytest.raises(ValueError, match="Unknown benchmark"):
        fetch_benchmark("not-a-thing")


# --- Resolver: HF-dataset URIs --------------------------------------------


def test_resolve_hf_dataset_uri(fake_snapshot_download, tmp_path) -> None:
    resolve_benchmark("hf-dataset://Sidney-Lisanza/leflur/cameo-2022")
    assert fake_snapshot_download[0]["repo_id"] == "Sidney-Lisanza/leflur"
    assert fake_snapshot_download[0]["allow_patterns"] == "cameo-2022/*"


def test_resolve_hf_dataset_uri_repo_root(fake_snapshot_download) -> None:
    """An owner/repo URI with no subdir downloads the whole repo."""
    resolve_benchmark("hf-dataset://Sidney-Lisanza/leflur")
    assert fake_snapshot_download[0]["repo_id"] == "Sidney-Lisanza/leflur"
    assert fake_snapshot_download[0]["allow_patterns"] is None


def test_invalid_hf_dataset_uri_raises(monkeypatch) -> None:
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        mock.Mock(side_effect=AssertionError("should not be called")),
    )
    with pytest.raises(ValueError, match="hf-dataset://"):
        resolve_benchmark("hf-dataset://only-owner")


# --- Resolver: local paths ------------------------------------------------


def test_resolve_local_dir_existing(tmp_path) -> None:
    out = resolve_benchmark(str(tmp_path))
    assert out == tmp_path.resolve()


def test_resolve_local_dir_pathlib(tmp_path) -> None:
    out = resolve_benchmark(tmp_path)
    assert out == tmp_path.resolve()


def test_resolve_missing_local_dir_raises(tmp_path) -> None:
    missing = tmp_path / "absent"
    with pytest.raises(FileNotFoundError, match="absent"):
        resolve_benchmark(str(missing))


# --- Resolver: S3 + wrong type --------------------------------------------


def test_resolve_s3_rejected_with_hint() -> None:
    with pytest.raises(ValueError, match="does not download from S3"):
        resolve_benchmark("s3://prescient-lobster/cameo/")


def test_resolve_wrong_type_raises() -> None:
    with pytest.raises(TypeError, match="expects str or Path"):
        resolve_benchmark(42)  # type: ignore[arg-type]


# --- Cache helpers --------------------------------------------------------


def test_cache_dir_honours_lobster_cache_env(tmp_path, monkeypatch) -> None:
    target = tmp_path / "custom"
    monkeypatch.setenv("LOBSTER_CACHE", str(target))
    assert cache_benchmark_dir() == target / "benchmarks"


def test_cached_benchmark_files_empty_when_cache_missing() -> None:
    assert cached_benchmark_files() == []


def test_clear_benchmark_cache_removes_subdirs(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LOBSTER_CACHE", str(tmp_path))
    root = cache_benchmark_dir()
    (root / "cameo").mkdir(parents=True)
    (root / "cameo" / "a.pt").write_bytes(b"")
    (root / "cameo" / "b.pt").write_bytes(b"")
    assert len(cached_benchmark_files()) == 2

    preview = list(clear_benchmark_cache(dry_run=True))
    assert preview, "dry-run should still report what would be removed"
    assert len(cached_benchmark_files()) == 2, "dry-run must not touch disk"

    deleted = list(clear_benchmark_cache())
    assert deleted
    assert cached_benchmark_files() == []


# --- Sanitiser ------------------------------------------------------------


def test_sanitize_record_rewrites_absolute_pdb_path_to_basename() -> None:
    record = {
        "pdb_path": "/cv/data/ai4dd/data2/lisanzas/AFDB/valid_cameo_processed/7dz2.C.pdb",
        "sequence": [1, 2, 3],
        "coords_res": [[0.0, 0.0, 0.0]],
    }
    out = _sanitize_record(record)
    assert out["pdb_path"] == "7dz2.C.pdb"
    # Other keys untouched + original record left intact (defensive copy).
    assert out["sequence"] == [1, 2, 3]
    assert record["pdb_path"].startswith("/cv/")


def test_sanitize_record_handles_protein_ligand_pair() -> None:
    """Both halves of a PoseBusters pair sanitize identically."""
    protein = {"pdb_path": "/cv/home/lisanzas/.../5S8I_2LY_protein.pdb"}
    ligand = {"pdb_path": "/cv/home/lisanzas/.../5S8I_2LY_ligand.pdb"}
    assert _sanitize_record(protein)["pdb_path"] == "5S8I_2LY_protein.pdb"
    assert _sanitize_record(ligand)["pdb_path"] == "5S8I_2LY_ligand.pdb"


def test_sanitize_record_no_pdb_path_is_noop() -> None:
    """MultiFlow records (no ``pdb_path``) are returned untouched."""
    record = {"sequence": [1, 2, 3], "coords_res": [[0.0, 0.0, 0.0]]}
    out = _sanitize_record(record)
    assert out is record


def test_sanitize_record_non_dict_passthrough() -> None:
    assert _sanitize_record("not-a-dict") == "not-a-dict"  # type: ignore[arg-type]


# --- Dataset card builder -------------------------------------------------


def test_generate_dataset_card_md_lists_every_registered_benchmark() -> None:
    md = generate_dataset_card_md()
    for short_name in KNOWN_BENCHMARKS:
        assert f"`{short_name}`" in md, f"card omits {short_name!r}"
    # Frontmatter must be a parseable YAML block.
    assert md.startswith("---")
    front_matter = md.split("---\n", 2)[1]
    yaml.safe_load(front_matter)
    # Body mentions the canonical repo + the reproduction CLI verb.
    assert HF_DATASET_REPO_ID in md
    assert "lobster_leflur_benchmarks fetch cameo" in md


# --- Upload helpers --------------------------------------------------------


@pytest.fixture
def fake_hf_api(monkeypatch):
    """Replace :class:`huggingface_hub.HfApi` with a recording stub.

    Backs both ``upload_benchmark`` (which calls ``upload_folder``) and
    ``upload_dataset_card`` (which calls ``upload_file``).
    """
    calls = {
        "uploaded_folders": [],
        "uploaded_files": [],
        "created": [],
        "repo_info": [],
    }

    class _FakeCommit:
        commit_url = "https://huggingface.co/fake/commit/abc"

    class _FakeApi:
        def __init__(self, token: str | None = None):
            self.token = token

        def repo_info(self, repo_id: str, repo_type: str):
            calls["repo_info"].append({"repo_id": repo_id, "repo_type": repo_type})

        def create_repo(self, repo_id: str, repo_type: str, exist_ok: bool = False):
            calls["created"].append(
                {
                    "repo_id": repo_id,
                    "repo_type": repo_type,
                    "exist_ok": exist_ok,
                }
            )

        def upload_folder(
            self,
            *,
            folder_path: str,
            path_in_repo: str,
            repo_id: str,
            repo_type: str,
            revision: str,
            commit_message: str,
        ):
            staged = sorted(p.name for p in Path(folder_path).iterdir())
            calls["uploaded_folders"].append(
                {
                    "folder_path": folder_path,
                    "path_in_repo": path_in_repo,
                    "repo_id": repo_id,
                    "repo_type": repo_type,
                    "revision": revision,
                    "commit_message": commit_message,
                    "staged_files": staged,
                }
            )
            return _FakeCommit()

        def upload_file(
            self,
            *,
            path_or_fileobj,
            path_in_repo: str,
            repo_id: str,
            repo_type: str,
            revision: str,
            commit_message: str,
        ):
            payload = path_or_fileobj if isinstance(path_or_fileobj, bytes) else Path(path_or_fileobj).read_bytes()
            calls["uploaded_files"].append(
                {
                    "size_bytes": len(payload),
                    "path_in_repo": path_in_repo,
                    "repo_id": repo_id,
                    "repo_type": repo_type,
                    "revision": revision,
                    "commit_message": commit_message,
                }
            )
            return _FakeCommit()

    monkeypatch.setattr("huggingface_hub.HfApi", _FakeApi, raising=True)
    return calls


def _make_source_with_pt_files(src: Path, n: int = 3, *, pdb_path_prefix: str = "/cv/internal/") -> None:
    """Populate *src* with *n* sanitisable ``.pt`` records."""
    src.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        torch.save(
            {
                "pdb_path": f"{pdb_path_prefix}target{i}.pdb",
                "sequence": torch.zeros(10, dtype=torch.long),
                "coords_res": torch.zeros(10, 3),
            },
            src / f"target{i}.pt",
        )


def test_upload_benchmark_dry_run_skips_network(tmp_path, monkeypatch) -> None:
    src = tmp_path / "fake_cameo"
    _make_source_with_pt_files(src, n=2)

    monkeypatch.setattr(
        "huggingface_hub.HfApi",
        mock.Mock(side_effect=AssertionError("HfApi should not be used")),
        raising=True,
    )

    summary = upload_benchmark("cameo", source_dir=src, dry_run=True)
    assert summary["dry_run"] == "True"
    assert summary["repo_id"] == HF_DATASET_REPO_ID
    assert summary["hf_subdir"] == "cameo-2022"
    assert summary["num_files"] == "2"
    assert summary["commit_url"] == ""


def test_upload_benchmark_sanitises_and_chunks(tmp_path, fake_hf_api) -> None:
    """End-to-end: sanitise pdb_path + chunk into ≤chunk_size batches."""
    src = tmp_path / "fake_cameo"
    _make_source_with_pt_files(src, n=5, pdb_path_prefix="/cv/data/secret/")

    summary = upload_benchmark(
        "cameo",
        source_dir=src,
        token="fake-token",
        chunk_size=2,
    )
    # 5 files, chunk_size=2 -> 3 chunks (2 + 2 + 1)
    assert summary["num_chunks"] == "3"
    assert summary["num_files"] == "5"
    assert summary["commit_url"].startswith("https://")
    assert len(fake_hf_api["uploaded_folders"]) == 3

    # Verify the staged files were sanitised (pdb_path is now basename).
    for upload in fake_hf_api["uploaded_folders"]:
        staged = Path(upload["folder_path"])
        for staged_pt in staged.glob("*.pt"):
            payload = torch.load(staged_pt, weights_only=False, map_location="cpu")
            assert payload["pdb_path"].startswith("target"), (
                f"sanitise leaked an internal path: {payload['pdb_path']!r}"
            )
            assert "/" not in payload["pdb_path"]


def test_upload_benchmark_no_sanitize_copies_bit_identically(tmp_path, fake_hf_api) -> None:
    src = tmp_path / "fake_cameo"
    _make_source_with_pt_files(src, n=2, pdb_path_prefix="/cv/data/secret/")

    upload_benchmark(
        "cameo",
        source_dir=src,
        token="fake-token",
        chunk_size=10,
        sanitize=False,
    )
    staged = Path(fake_hf_api["uploaded_folders"][0]["folder_path"])
    for staged_pt in staged.glob("*.pt"):
        payload = torch.load(staged_pt, weights_only=False, map_location="cpu")
        # ``sanitize=False`` must leave the internal path intact.
        assert payload["pdb_path"].startswith("/cv/data/secret/")


def test_upload_benchmark_unknown_short_name_raises(tmp_path) -> None:
    with pytest.raises(ValueError, match="Unknown benchmark"):
        upload_benchmark("not-a-thing", source_dir=tmp_path, dry_run=True)


def test_upload_benchmark_missing_source_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        upload_benchmark("cameo", source_dir=tmp_path / "absent", dry_run=True)


def test_upload_benchmark_no_files_raises(tmp_path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="No files matching"):
        upload_benchmark("cameo", source_dir=tmp_path / "empty", dry_run=True)


def test_upload_benchmark_with_no_local_source_path(tmp_path) -> None:
    """A registry entry without ``local_source_path`` must require --source."""
    info = BenchmarkInfo(short_name="custom", hf_subdir="custom-sub", description="d")
    with mock.patch.dict(
        "lobster.model.leflur.benchmarks.KNOWN_BENCHMARKS",
        {"custom": info},
        clear=False,
    ):
        with pytest.raises(ValueError, match="has no local_source_path"):
            upload_benchmark("custom", dry_run=True)


def test_upload_dataset_card_dry_run_skips_network(monkeypatch) -> None:
    monkeypatch.setattr(
        "huggingface_hub.HfApi",
        mock.Mock(side_effect=AssertionError("HfApi should not be used")),
        raising=True,
    )
    summary = upload_dataset_card(dry_run=True)
    assert summary["dry_run"] == "True"
    assert summary["repo_id"] == HF_DATASET_REPO_ID
    assert summary["hf_subdir"] == "README.md"
    assert int(summary["total_bytes"]) > 0
    assert summary["commit_url"] == ""


def test_upload_dataset_card_calls_hf_api(fake_hf_api) -> None:
    summary = upload_dataset_card(token="fake-token")
    assert summary["commit_url"].startswith("https://")
    assert len(fake_hf_api["uploaded_files"]) == 1
    uploaded = fake_hf_api["uploaded_files"][0]
    assert uploaded["repo_id"] == HF_DATASET_REPO_ID
    assert uploaded["path_in_repo"] == "README.md"
    assert uploaded["repo_type"] == HF_DATASET_REPO_TYPE


# --- Upload retry / repo-create branches ----------------------------------


def test_upload_chunk_retries_then_succeeds(tmp_path, monkeypatch) -> None:
    """Per-chunk upload retries on flake, then commits."""
    from lobster.model.leflur import benchmarks as bm

    src = tmp_path / "fake_cameo"
    _make_source_with_pt_files(src, n=1)

    attempts = {"n": 0}

    class _FlakyApi:
        def __init__(self, token: str | None = None):
            self.token = token

        def repo_info(self, repo_id: str, repo_type: str):
            return None

        def create_repo(self, *args, **kwargs):
            raise AssertionError("create_repo should not be called")

        def upload_folder(self, **kwargs):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise RuntimeError("transient 401")

            class _C:
                commit_url = "https://hf/ok"

            return _C()

    monkeypatch.setattr("huggingface_hub.HfApi", _FlakyApi, raising=True)
    # Eliminate the real backoff sleep so the test stays fast.
    monkeypatch.setattr(bm, "_UPLOAD_BACKOFF_SECONDS", (0, 0, 0, 0))

    summary = upload_benchmark("cameo", source_dir=src, token="fake-token", chunk_size=10)
    assert attempts["n"] == 2
    assert summary["commit_url"] == "https://hf/ok"


def test_ensure_dataset_repo_creates_when_missing(tmp_path, monkeypatch) -> None:
    """If ``repo_info`` 404s we create the repo (and only on the missing branch)."""
    from huggingface_hub.utils import RepositoryNotFoundError

    from lobster.model.leflur import benchmarks as bm

    src = tmp_path / "fake_cameo"
    _make_source_with_pt_files(src, n=1)

    created: list[dict] = []

    class _Api:
        def __init__(self, token: str | None = None):
            self.token = token

        def repo_info(self, repo_id: str, repo_type: str):
            raise RepositoryNotFoundError("404")

        def create_repo(self, repo_id: str, repo_type: str, exist_ok: bool = False):
            created.append({"repo_id": repo_id, "repo_type": repo_type})

        def upload_folder(self, **kwargs):
            class _C:
                commit_url = "https://hf/ok"

            return _C()

    monkeypatch.setattr("huggingface_hub.HfApi", _Api, raising=True)
    monkeypatch.setattr(bm, "_UPLOAD_BACKOFF_SECONDS", (0,))

    upload_benchmark("cameo", source_dir=src, token="fake-token", chunk_size=10)
    assert created == [{"repo_id": HF_DATASET_REPO_ID, "repo_type": HF_DATASET_REPO_TYPE}]


# --- CLI argparse wiring (lightweight, no HF calls) ----------------------


def test_cli_list_runs(capsys) -> None:
    from lobster.cmdline.manage_leflur_benchmarks import main

    exit_code = main(["list"])
    out = capsys.readouterr().out
    assert exit_code == 0
    for short_name in (
        "cameo",
        "multiflow_test",
        "posebusters_benchmark_no_overlap",
    ):
        assert short_name in out


def test_cli_list_with_tag_filter(capsys) -> None:
    from lobster.cmdline.manage_leflur_benchmarks import main

    exit_code = main(["list", "--tag", "protein-ligand"])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "posebusters_benchmark" in out
    assert "cameo" not in out


def test_cli_list_with_no_matches(capsys) -> None:
    from lobster.cmdline.manage_leflur_benchmarks import main

    exit_code = main(["list", "--tag", "nonexistent-tag"])
    assert exit_code == 1
    out = capsys.readouterr().out
    assert "No benchmarks match" in out


def test_cli_inspect_canonical(capsys) -> None:
    from lobster.cmdline.manage_leflur_benchmarks import main

    exit_code = main(["inspect", "cameo"])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "Sidney-Lisanza/leflur" in out
    assert "publication" in out  # tag


def test_cli_inspect_unknown_short_name(capsys) -> None:
    from lobster.cmdline.manage_leflur_benchmarks import main

    exit_code = main(["inspect", "not-a-thing"])
    err = capsys.readouterr().err
    assert exit_code == 2
    assert "Unknown short name" in err


def test_cli_fetch_short_name(capsys, fake_snapshot_download) -> None:
    from lobster.cmdline.manage_leflur_benchmarks import main

    exit_code = main(["fetch", "cameo"])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "cameo" in out
    assert "files matching" in out


def test_cli_fetch_unknown_short_name(capsys) -> None:
    from lobster.cmdline.manage_leflur_benchmarks import main

    exit_code = main(["fetch", "not-a-thing"])
    err = capsys.readouterr().err
    assert exit_code == 2
    assert "Unknown benchmark" in err


def test_cli_cache_empty(capsys) -> None:
    from lobster.cmdline.manage_leflur_benchmarks import main

    exit_code = main(["cache"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "(cache is empty)" in out


def test_cli_cache_clear_dry_run(capsys, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LOBSTER_CACHE", str(tmp_path))
    root = cache_benchmark_dir()
    (root / "cameo").mkdir(parents=True)
    (root / "cameo" / "a.pt").write_bytes(b"")

    from lobster.cmdline.manage_leflur_benchmarks import main

    exit_code = main(["cache", "--clear", "--dry-run"])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "would remove" in out
    assert (root / "cameo" / "a.pt").exists()


def test_cli_upload_dry_run(capsys, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "huggingface_hub.HfApi",
        mock.Mock(side_effect=AssertionError("HfApi should not be used")),
        raising=True,
    )
    src = tmp_path / "fake_cameo"
    _make_source_with_pt_files(src, n=2)

    from lobster.cmdline.manage_leflur_benchmarks import main

    exit_code = main(["upload", "cameo", "--dry-run", "--source", str(src)])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "cameo" in out
    assert "would upload" in out


def test_cli_upload_no_targets(capsys) -> None:
    from lobster.cmdline.manage_leflur_benchmarks import main

    exit_code = main(["upload"])
    err = capsys.readouterr().err
    assert exit_code == 2
    assert "pass at least one short name" in err


def test_cli_dataset_card_print(capsys) -> None:
    from lobster.cmdline.manage_leflur_benchmarks import main

    exit_code = main(["dataset-card", "--print"])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert out.startswith("---")
    for short_name in KNOWN_BENCHMARKS:
        assert f"`{short_name}`" in out


def test_cli_dataset_card_dry_run(capsys, monkeypatch) -> None:
    monkeypatch.setattr(
        "huggingface_hub.HfApi",
        mock.Mock(side_effect=AssertionError("HfApi should not be used")),
        raising=True,
    )
    from lobster.cmdline.manage_leflur_benchmarks import main

    exit_code = main(["dataset-card", "--dry-run"])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "would upload dataset card" in out


# --- End-to-end snapshot smoke (mocked HF) --------------------------------


def test_resolve_then_load_matches_publication_loader(fake_snapshot_download, tmp_path, monkeypatch) -> None:
    """The snapshot-flattening output matches what the publication loader globs.

    Locks the contract between ``_snapshot_download`` and
    ``ProteinLigandInverseFoldingEvaluator._load_dataset``, which calls
    ``sorted(glob(os.path.join(data_dir, "*_protein.pt")))`` directly
    under the resolved benchmark directory.
    """

    # Replace the generic fake with one that stages PoseBusters-shaped pairs.
    def _fake(
        *,
        repo_id: str,
        repo_type: str,
        revision: str,
        local_dir: str,
        allow_patterns: str | None = None,
    ) -> str:
        root = Path(local_dir)
        root.mkdir(parents=True, exist_ok=True)
        subdir = root / allow_patterns[:-2] if allow_patterns else root
        subdir.mkdir(parents=True, exist_ok=True)
        for tag in ("5SIS_JSM", "5SB2_1K2"):
            for suffix in ("protein", "ligand"):
                (subdir / f"{tag}_{suffix}.pt").write_bytes(b"\x00")
        return str(root)

    monkeypatch.setattr("huggingface_hub.snapshot_download", _fake, raising=True)

    path = resolve_benchmark("posebusters_benchmark_no_overlap")
    # Files are at the flat layout the loader expects.
    proteins = sorted(p.name for p in path.glob("*_protein.pt"))
    ligands = sorted(p.name for p in path.glob("*_ligand.pt"))
    assert proteins == ["5SB2_1K2_protein.pt", "5SIS_JSM_protein.pt"]
    assert ligands == ["5SB2_1K2_ligand.pt", "5SIS_JSM_ligand.pt"]


# --- Module re-export surface --------------------------------------------


def test_public_reexports_match_dunder_all() -> None:
    """`lobster.model.leflur.__all__` exposes the benchmark surface."""
    import lobster.model.leflur as leflur

    for symbol in (
        "KNOWN_BENCHMARKS",
        "BenchmarkInfo",
        "cache_benchmark_dir",
        "cached_benchmark_files",
        "clear_benchmark_cache",
        "fetch_benchmark",
        "generate_dataset_card_md",
        "list_benchmarks",
        "resolve_benchmark",
        "upload_benchmark",
        "upload_dataset_card",
    ):
        assert symbol in leflur.__all__, f"{symbol!r} missing from lobster.model.leflur.__all__"
        assert hasattr(leflur, symbol), f"{symbol!r} listed in __all__ but not bound on the module"
