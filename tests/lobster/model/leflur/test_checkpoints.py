"""Tests for ``lobster.model.leflur.checkpoints``.

Mocks :func:`huggingface_hub.hf_hub_download` so the suite runs offline.
Covers:

* Short-name → HF download (registered name lookup).
* ``hf://`` URI parsing + download.
* ``https://huggingface.co/.../resolve/...`` URL parsing + download.
* Local-path passthrough (existing file) + ``FileNotFoundError`` (missing).
* S3 URI rejection with actionable message.
* Cache helpers (``cache_dir``, ``cached_files``, ``clear_cache``) without
  hitting the network.
* Registry sanity: the three canonical short names exist, and each
  ``CheckpointInfo`` has consistent ``hf_uri`` / ``https_url`` derivations.

The CLI is exercised lightly via ``manage_leflur_checkpoints.build_parser``.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from lobster.model.leflur import (
    KNOWN_CHECKPOINTS,
    CheckpointInfo,
    cached_files,
    cache_dir,
    clear_cache,
    list_checkpoints,
    resolve_checkpoint,
)
from lobster.model.leflur.checkpoints import HF_REPO_ID, HF_REVISION


# --- Fixtures --------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    """Point ``$LOBSTER_CACHE`` at a per-test tmpdir."""
    monkeypatch.setenv("LOBSTER_CACHE", str(tmp_path))
    yield tmp_path


@pytest.fixture
def fake_hf_download(monkeypatch):
    """Replace :func:`huggingface_hub.hf_hub_download` with a recorder.

    The fake creates an empty file under the cache dir so downstream code
    can ``Path(...).exists()`` cleanly.
    """
    calls: list[dict] = []

    def _fake(*, repo_id: str, filename: str, revision: str, cache_dir: str):
        calls.append(
            {"repo_id": repo_id, "filename": filename, "revision": revision}
        )
        out = Path(cache_dir) / Path(filename).name
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"")
        return str(out)

    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download", _fake, raising=True
    )
    return calls


# --- Registry sanity ------------------------------------------------------


def test_canonical_short_names_registered() -> None:
    """Plan-mandated canonical short names live in the registry."""
    for short_name in ("leflur-base", "leflur-ted", "leflur-pl"):
        assert short_name in KNOWN_CHECKPOINTS, (
            f"{short_name!r} missing from KNOWN_CHECKPOINTS — required by "
            f"Phase 4 of the LeFlur publication cleanup."
        )


def test_hf_uri_and_https_url_match_repo_id() -> None:
    """``hf_uri`` and ``https_url`` derive consistently from ``hf_path``."""
    for info in KNOWN_CHECKPOINTS.values():
        assert info.hf_uri == f"hf://{HF_REPO_ID}/{info.hf_path}"
        assert info.https_url == (
            f"https://huggingface.co/{HF_REPO_ID}/resolve/{HF_REVISION}/"
            f"{info.hf_path}"
        )


def test_list_checkpoints_family_filter() -> None:
    protein_only = list_checkpoints(family="protein")
    pl = list_checkpoints(family="protein_ligand")
    assert all(info.family == "protein" for info in protein_only)
    assert all(info.family == "protein_ligand" for info in pl)
    assert any(info.short_name == "leflur-pl" for info in pl)
    assert all(info.short_name != "leflur-pl" for info in protein_only)


# --- Resolver: short names -----------------------------------------------


def test_resolve_short_name_triggers_hf_download(fake_hf_download) -> None:
    info = KNOWN_CHECKPOINTS["leflur-ted"]
    path = resolve_checkpoint("leflur-ted")
    assert isinstance(path, Path) and path.exists()
    assert fake_hf_download == [
        {
            "repo_id": HF_REPO_ID,
            "filename": info.hf_path,
            "revision": HF_REVISION,
        }
    ]


def test_resolve_short_name_handles_whitespace(fake_hf_download) -> None:
    resolve_checkpoint("  leflur-base ")
    assert fake_hf_download[0]["filename"] == (
        KNOWN_CHECKPOINTS["leflur-base"].hf_path
    )


# --- Resolver: HF URIs ----------------------------------------------------


def test_resolve_hf_uri(fake_hf_download) -> None:
    resolve_checkpoint("hf://Sidney-Lisanza/leflur/custom/path.ckpt")
    assert fake_hf_download == [
        {
            "repo_id": "Sidney-Lisanza/leflur",
            "filename": "custom/path.ckpt",
            "revision": HF_REVISION,
        }
    ]


def test_resolve_https_huggingface_url(fake_hf_download) -> None:
    resolve_checkpoint(
        "https://huggingface.co/Sidney-Lisanza/leflur/resolve/main/x/y.ckpt"
    )
    assert fake_hf_download == [
        {
            "repo_id": "Sidney-Lisanza/leflur",
            "filename": "x/y.ckpt",
            "revision": "main",
        }
    ]


def test_resolve_https_blob_url_normalises(fake_hf_download) -> None:
    """``/blob/`` URLs are tolerated and parsed the same way as ``/resolve/``."""
    resolve_checkpoint(
        "https://huggingface.co/Sidney-Lisanza/leflur/blob/main/x.ckpt"
    )
    assert fake_hf_download[0]["filename"] == "x.ckpt"


def test_invalid_hf_uri_raises(monkeypatch) -> None:
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        mock.Mock(side_effect=AssertionError("should not be called")),
    )
    with pytest.raises(ValueError, match="hf://"):
        resolve_checkpoint("hf://only-owner")


# --- Resolver: local paths ------------------------------------------------


def test_resolve_local_path_existing(tmp_path) -> None:
    f = tmp_path / "model.ckpt"
    f.write_bytes(b"\x00")
    out = resolve_checkpoint(str(f))
    assert out == f.resolve()


def test_resolve_local_path_pathlib(tmp_path) -> None:
    f = tmp_path / "model.ckpt"
    f.write_bytes(b"\x00")
    out = resolve_checkpoint(f)
    assert out == f.resolve()


def test_resolve_missing_local_path_raises(tmp_path) -> None:
    missing = tmp_path / "absent.ckpt"
    with pytest.raises(FileNotFoundError, match="absent.ckpt"):
        resolve_checkpoint(str(missing))


# --- Resolver: S3 ---------------------------------------------------------


def test_resolve_s3_rejected_with_hint() -> None:
    with pytest.raises(ValueError, match="does not download from S3"):
        resolve_checkpoint("s3://prescient-lobster/some.ckpt")


# --- Resolver: wrong type --------------------------------------------------


def test_resolve_wrong_type_raises() -> None:
    with pytest.raises(TypeError, match="expects str or Path"):
        resolve_checkpoint(42)  # type: ignore[arg-type]


# --- Cache helpers --------------------------------------------------------


def test_cache_dir_honours_lobster_cache_env(tmp_path, monkeypatch) -> None:
    target = tmp_path / "custom"
    monkeypatch.setenv("LOBSTER_CACHE", str(target))
    assert cache_dir() == target / "checkpoints"


def test_cached_files_returns_empty_when_cache_missing(tmp_path) -> None:
    assert cached_files() == []


def test_clear_cache_removes_subdirs(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LOBSTER_CACHE", str(tmp_path))
    root = cache_dir()
    (root / "Sidney-Lisanza__leflur").mkdir(parents=True)
    (root / "Sidney-Lisanza__leflur" / "a.ckpt").write_bytes(b"")
    (root / "Sidney-Lisanza__leflur" / "b.ckpt").write_bytes(b"")
    assert len(cached_files()) == 2

    preview = list(clear_cache(dry_run=True))
    assert preview, "dry-run should still report what would be removed"
    assert len(cached_files()) == 2, "dry-run must not touch disk"

    deleted = list(clear_cache())
    assert deleted
    assert cached_files() == []


# --- CLI argparse wiring (lightweight, no HF calls) ----------------------


def test_cli_list_runs(capsys, fake_hf_download) -> None:
    """The CLI ``list`` subcommand prints registered short names."""
    from lobster.cmdline.manage_leflur_checkpoints import main

    exit_code = main(["list"])
    out = capsys.readouterr().out
    assert exit_code == 0
    for short_name in ("leflur-base", "leflur-ted", "leflur-pl"):
        assert short_name in out


def test_cli_inspect_canonical(capsys, fake_hf_download) -> None:
    from lobster.cmdline.manage_leflur_checkpoints import main

    exit_code = main(["inspect", "leflur-ted"])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "Sidney-Lisanza/leflur" in out
    assert "publication" in out  # tag


def test_cli_inspect_unknown_short_name(capsys, fake_hf_download) -> None:
    from lobster.cmdline.manage_leflur_checkpoints import main

    exit_code = main(["inspect", "leflur-not-a-thing"])
    err = capsys.readouterr().err
    assert exit_code == 2
    assert "Unknown short name" in err


def test_cli_fetch_short_name(capsys, fake_hf_download) -> None:
    from lobster.cmdline.manage_leflur_checkpoints import main

    exit_code = main(["fetch", "leflur-base"])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "leflur-base" in out
    assert "size" in out


def test_cli_cache_empty(capsys) -> None:
    from lobster.cmdline.manage_leflur_checkpoints import main

    exit_code = main(["cache"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "(cache is empty)" in out


def test_cli_cache_clear_dry_run(capsys, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LOBSTER_CACHE", str(tmp_path))
    root = cache_dir()
    (root / "sub").mkdir(parents=True)

    from lobster.cmdline.manage_leflur_checkpoints import main

    exit_code = main(["cache", "--clear", "--dry-run"])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "would remove" in out
    # dry-run must not delete
    assert (root / "sub").exists()


def test_cheap_typecheck_on_checkpoint_info() -> None:
    """`CheckpointInfo` is frozen + dataclass-shaped (regression for mutation bugs)."""
    info = CheckpointInfo(
        short_name="x", hf_path="x.ckpt", description="d"
    )
    with pytest.raises(Exception):
        info.short_name = "y"  # type: ignore[misc]


# --- Paired LG codec override -------------------------------------------


def _reset_paired_lg_override_flag() -> None:
    """Tests need a clean ``_paired_lg_overrides_installed=False`` to verify."""
    import lobster.model.leflur.checkpoints as ckpt_mod

    ckpt_mod._paired_lg_overrides_installed = False


def test_install_paired_lg_codec_overrides_rewrites_internal_paths(
    monkeypatch,
) -> None:
    """LG codecs LeFlur depends on get flipped to HF URLs in the registry."""
    from lobster.model.latent_generator.cmdline import methods as lg_methods
    from lobster.model.leflur.checkpoints import (
        PAIRED_LG_CODECS,
        install_paired_lg_codec_overrides,
    )

    # Save & restore original checkpoints for affected codec entries.
    originals = {
        name: lg_methods[name].model_config.checkpoint
        for name in PAIRED_LG_CODECS
        if name in lg_methods
    }
    _reset_paired_lg_override_flag()
    try:
        previous = install_paired_lg_codec_overrides()

        for name, hf_url in PAIRED_LG_CODECS.items():
            if name not in lg_methods:
                continue
            assert lg_methods[name].model_config.checkpoint == hf_url, (
                f"codec {name!r} should now point at {hf_url}"
            )

        # Any codec whose checkpoint changed must show up in the ``previous`` map.
        for name in PAIRED_LG_CODECS:
            if name not in lg_methods:
                continue
            if originals[name] != PAIRED_LG_CODECS[name]:
                assert previous.get(name) == originals[name]
    finally:
        for name, ckpt in originals.items():
            lg_methods[name].model_config.checkpoint = ckpt
        _reset_paired_lg_override_flag()


# --- Upload helpers --------------------------------------------------------


@pytest.fixture
def fake_hf_api(monkeypatch):
    """Replace :class:`huggingface_hub.HfApi` with a recording stub."""
    calls = {"uploaded": [], "created": [], "repo_info": []}

    class _FakeCommit:
        commit_url = "https://huggingface.co/fake/commit/abc"

    class _FakeApi:
        def __init__(self, token: str | None = None):
            self.token = token

        def repo_info(self, repo_id: str, repo_type: str):
            calls["repo_info"].append({"repo_id": repo_id, "repo_type": repo_type})

        def create_repo(self, repo_id: str, repo_type: str, exist_ok: bool = False):
            calls["created"].append(
                {"repo_id": repo_id, "repo_type": repo_type, "exist_ok": exist_ok}
            )

        def upload_file(
            self,
            *,
            path_or_fileobj: str,
            path_in_repo: str,
            repo_id: str,
            repo_type: str,
            revision: str,
            commit_message: str,
        ):
            calls["uploaded"].append(
                {
                    "path_or_fileobj": path_or_fileobj,
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


def test_upload_checkpoint_dry_run_skips_network(tmp_path, monkeypatch) -> None:
    """``dry_run=True`` populates the summary without touching HfApi."""
    from lobster.model.leflur import upload_checkpoint

    src = tmp_path / "fake.ckpt"
    src.write_bytes(b"x" * 1024)

    sentinel = mock.Mock(side_effect=AssertionError("HfApi should not be used"))
    monkeypatch.setattr("huggingface_hub.HfApi", sentinel, raising=True)

    summary = upload_checkpoint(
        "leflur-ted", source_path=src, dry_run=True
    )
    assert summary["dry_run"] == "True"
    assert summary["repo_id"] == "Sidney-Lisanza/leflur"
    assert summary["hf_path"].startswith("leflur_denovo_ted")
    assert summary["source_path"] == str(src)
    assert summary["commit_url"] == ""


def test_upload_checkpoint_calls_hf_api(tmp_path, fake_hf_api) -> None:
    from lobster.model.leflur import upload_checkpoint

    src = tmp_path / "fake.ckpt"
    src.write_bytes(b"y" * 4096)

    summary = upload_checkpoint(
        "leflur-base", source_path=src, token="fake-token"
    )

    assert summary["commit_url"].startswith("https://")
    assert len(fake_hf_api["uploaded"]) == 1
    uploaded = fake_hf_api["uploaded"][0]
    assert uploaded["repo_id"] == "Sidney-Lisanza/leflur"
    assert uploaded["path_in_repo"].endswith(".ckpt")
    assert uploaded["repo_type"] == "model"
    # Source path passed through verbatim
    assert uploaded["path_or_fileobj"] == str(src)


def test_upload_checkpoint_missing_source_raises(tmp_path) -> None:
    from lobster.model.leflur import upload_checkpoint

    with pytest.raises(FileNotFoundError):
        upload_checkpoint(
            "leflur-ted",
            source_path=tmp_path / "does-not-exist.ckpt",
            dry_run=True,
        )


def test_upload_checkpoint_unknown_short_name_raises(tmp_path) -> None:
    from lobster.model.leflur import upload_checkpoint

    with pytest.raises(ValueError, match="Unknown checkpoint"):
        upload_checkpoint("leflur-mystery", source_path=tmp_path, dry_run=True)


def test_upload_checkpoint_lg_codec_target_repo(tmp_path, fake_hf_api) -> None:
    """Paired LG codecs land in Sidney-Lisanza/latent_generator, not /leflur."""
    from lobster.model.leflur import upload_checkpoint

    src = tmp_path / "lg.ckpt"
    src.write_bytes(b"z" * 2048)

    summary = upload_checkpoint(
        "LG Protein Ligand fsq 4375", source_path=src, token="fake-token"
    )
    assert summary["repo_id"] == "Sidney-Lisanza/latent_generator"
    assert fake_hf_api["uploaded"][0]["repo_id"] == "Sidney-Lisanza/latent_generator"
    assert fake_hf_api["uploaded"][0]["path_in_repo"].startswith(
        "checkpoints_for_lg/"
    )


def test_cli_upload_dry_run_all(capsys, tmp_path, monkeypatch) -> None:
    """``upload --all --dry-run`` prints a plan without touching HfApi."""
    monkeypatch.setattr(
        "huggingface_hub.HfApi",
        mock.Mock(side_effect=AssertionError("HfApi should not be used")),
        raising=True,
    )
    # Make the registered local source paths exist by pointing them at a fake.
    fake_src = tmp_path / "fake.ckpt"
    fake_src.write_bytes(b"q" * 16)

    from lobster.cmdline.manage_leflur_checkpoints import main

    exit_code = main(
        ["upload", "--all", "--dry-run", "--source", str(fake_src)]
    )
    out = capsys.readouterr().out
    assert exit_code == 0
    for short_name in ("leflur-base", "leflur-ted", "leflur-pl"):
        assert short_name in out


def test_install_paired_lg_codec_overrides_is_idempotent() -> None:
    """Subsequent calls are no-ops (return ``{}``)."""
    from lobster.model.latent_generator.cmdline import methods as lg_methods
    from lobster.model.leflur.checkpoints import (
        PAIRED_LG_CODECS,
        install_paired_lg_codec_overrides,
    )

    originals = {
        name: lg_methods[name].model_config.checkpoint
        for name in PAIRED_LG_CODECS
        if name in lg_methods
    }
    _reset_paired_lg_override_flag()
    try:
        install_paired_lg_codec_overrides()
        second_call = install_paired_lg_codec_overrides()
        assert second_call == {}
    finally:
        for name, ckpt in originals.items():
            lg_methods[name].model_config.checkpoint = ckpt
        _reset_paired_lg_override_flag()
