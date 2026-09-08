import json
from datetime import datetime, UTC
from unittest.mock import MagicMock, call, mock_open, patch

import pytest
from botocore.exceptions import ClientError, CredentialRetrievalError, NoCredentialsError

from lobster.model._utils_checkpoint import (
    S3ObjectMetadata,
    _download_checkpoint,
    _normalize_etag,
    checkpoint_checksum_path,
    download_checkpoint,
    get_s3_last_modified_timestamp,
    get_s3_object_metadata,
    is_local_checkpoint_valid,
    load_checkpoint_with_retry,
)


def _metadata(
    *,
    etag: str = "abc123def456",
    content_length: int = 100,
    s3_uri: str = "s3://bucket/checkpoint.ckpt",
) -> S3ObjectMetadata:
    return S3ObjectMetadata(
        etag=etag,
        content_length=content_length,
        last_modified=datetime(2025, 7, 11, 6, 17, 18, tzinfo=UTC),
        s3_uri=s3_uri,
    )


def test_normalize_etag_strips_quotes():
    assert _normalize_etag('"abc123"') == "abc123"
    assert _normalize_etag("abc123") == "abc123"


@patch("lobster.model._utils_checkpoint.boto3.client")
def test_get_s3_object_metadata(mock_boto_client):
    mock_s3 = MagicMock()
    mock_boto_client.return_value = mock_s3
    mock_s3.head_object.return_value = {
        "ETag": '"abc123def456"',
        "ContentLength": 42,
        "LastModified": datetime(2025, 7, 11, 6, 17, 18, tzinfo=UTC),
    }

    metadata = get_s3_object_metadata("s3://bucket/path/model.ckpt")

    mock_s3.head_object.assert_called_once_with(Bucket="bucket", Key="path/model.ckpt")
    assert metadata.etag == "abc123def456"
    assert metadata.content_length == 42
    assert metadata.s3_uri == "s3://bucket/path/model.ckpt"


@patch("lobster.model._utils_checkpoint.get_s3_object_metadata")
def test_get_s3_last_modified_timestamp(mock_get_metadata):
    mock_get_metadata.return_value = _metadata()
    assert get_s3_last_modified_timestamp("s3://bucket/checkpoint.ckpt") == "20250711-061718"


def test_checkpoint_checksum_path():
    assert checkpoint_checksum_path("/tmp/model.ckpt") == "/tmp/model.ckpt.checksum.json"


@patch("lobster.model._utils_checkpoint.os.path.getsize", return_value=100)
@patch("lobster.model._utils_checkpoint.os.path.exists")
def test_is_local_checkpoint_valid_with_matching_sidecar(mock_exists, mock_getsize):
    metadata = _metadata()

    def exists(path):
        return path in {"/tmp/model.ckpt", "/tmp/model.ckpt.checksum.json"}

    mock_exists.side_effect = exists
    sidecar = {"etag": metadata.etag, "content_length": metadata.content_length, "s3_uri": metadata.s3_uri}

    with (
        patch("builtins.open", mock_open(read_data=json.dumps(sidecar))),
        patch("lobster.model._utils_checkpoint._local_md5_hex", return_value=metadata.etag),
    ):
        assert is_local_checkpoint_valid("/tmp/model.ckpt", metadata) is True


@patch("lobster.model._utils_checkpoint.os.path.getsize", return_value=100)
@patch("lobster.model._utils_checkpoint.os.path.exists")
def test_is_local_checkpoint_valid_sidecar_etag_mismatch(mock_exists, mock_getsize):
    metadata = _metadata(etag="newetag")

    def exists(path):
        return path in {"/tmp/model.ckpt", "/tmp/model.ckpt.checksum.json"}

    mock_exists.side_effect = exists
    sidecar = {"etag": "oldetag", "content_length": 100, "s3_uri": metadata.s3_uri}

    with patch("builtins.open", mock_open(read_data=json.dumps(sidecar))):
        assert is_local_checkpoint_valid("/tmp/model.ckpt", metadata) is False


@patch("lobster.model._utils_checkpoint.os.path.getsize", return_value=50)
@patch("lobster.model._utils_checkpoint.os.path.exists", return_value=True)
def test_is_local_checkpoint_valid_size_mismatch(mock_exists, mock_getsize):
    assert is_local_checkpoint_valid("/tmp/model.ckpt", _metadata(content_length=100)) is False


@patch("lobster.model._utils_checkpoint._write_checksum_sidecar")
@patch("lobster.model._utils_checkpoint._local_md5_hex", return_value="abc123def456")
@patch("lobster.model._utils_checkpoint.os.path.getsize", return_value=100)
@patch("lobster.model._utils_checkpoint.os.path.exists")
def test_is_local_checkpoint_valid_legacy_cache_md5_match(mock_exists, mock_getsize, mock_md5, mock_write_sidecar):
    """Legacy caches without sidecars are accepted when single-part MD5 matches."""

    def exists(path):
        return path == "/tmp/model.ckpt"

    mock_exists.side_effect = exists
    metadata = _metadata(etag="abc123def456")

    assert is_local_checkpoint_valid("/tmp/model.ckpt", metadata) is True
    mock_write_sidecar.assert_called_once_with("/tmp/model.ckpt", metadata)


@patch("lobster.model._utils_checkpoint.os.path.getsize", return_value=100)
@patch("lobster.model._utils_checkpoint.os.path.exists")
def test_is_local_checkpoint_valid_legacy_multipart_missing_sidecar(mock_exists, mock_getsize):
    def exists(path):
        return path == "/tmp/model.ckpt"

    mock_exists.side_effect = exists
    assert is_local_checkpoint_valid("/tmp/model.ckpt", _metadata(etag="abc123-3")) is False


# Test download_checkpoint function
@patch("lobster.model._utils_checkpoint.get_s3_object_metadata")
@patch("lobster.model._utils_checkpoint.is_local_checkpoint_valid", return_value=False)
@patch("lobster.model._utils_checkpoint._remove_local_checkpoint")
@patch("lobster.model._utils_checkpoint._write_checksum_sidecar")
@patch("lobster.model._utils_checkpoint.os.path.getsize", return_value=100)
@patch("lobster.model._utils_checkpoint._local_md5_hex", return_value="abc123def456")
@patch("lobster.model._utils_checkpoint._download_checkpoint")
def test_download_checkpoint_file_not_valid(
    mock_download, mock_md5, mock_getsize, mock_write_sidecar, mock_remove, mock_valid, mock_get_metadata
):
    """Test download_checkpoint when local cache is missing/invalid."""
    metadata = _metadata()
    mock_get_metadata.return_value = metadata

    download_checkpoint("s3://bucket/checkpoint.ckpt", "/tmp", "model.ckpt")

    mock_download.assert_called_once_with("s3://bucket/checkpoint.ckpt", "/tmp/model.ckpt", "model.ckpt")
    mock_write_sidecar.assert_called_once_with("/tmp/model.ckpt", metadata)


@patch("lobster.model._utils_checkpoint.get_s3_object_metadata")
@patch("lobster.model._utils_checkpoint.is_local_checkpoint_valid", return_value=True)
@patch("lobster.model._utils_checkpoint._download_checkpoint")
def test_download_checkpoint_file_valid_no_force(mock_download, mock_valid, mock_get_metadata):
    """Test download_checkpoint when checksum-valid cache exists."""
    mock_get_metadata.return_value = _metadata()

    download_checkpoint("s3://bucket/checkpoint.ckpt", "/tmp", "model.ckpt")

    mock_download.assert_not_called()


@patch("lobster.model._utils_checkpoint.is_local_checkpoint_valid", return_value=True)
@patch("lobster.model._utils_checkpoint._remove_local_checkpoint")
@patch("lobster.model._utils_checkpoint._write_checksum_sidecar")
@patch("lobster.model._utils_checkpoint.os.path.getsize", return_value=100)
@patch("lobster.model._utils_checkpoint._local_md5_hex", return_value="abc123def456")
@patch("lobster.model._utils_checkpoint._download_checkpoint")
def test_download_checkpoint_force_redownload(
    mock_download, mock_md5, mock_getsize, mock_write_sidecar, mock_remove, mock_valid
):
    """Test download_checkpoint with force_redownload=True."""
    metadata = _metadata()

    download_checkpoint(
        "s3://bucket/checkpoint.ckpt",
        "/tmp",
        "model.ckpt",
        force_redownload=True,
        metadata=metadata,
    )

    mock_download.assert_called_once_with("s3://bucket/checkpoint.ckpt", "/tmp/model.ckpt", "model.ckpt")
    mock_write_sidecar.assert_called_once_with("/tmp/model.ckpt", metadata)


@patch("lobster.model._utils_checkpoint.is_local_checkpoint_valid", return_value=False)
@patch("lobster.model._utils_checkpoint._remove_local_checkpoint")
@patch("lobster.model._utils_checkpoint.os.path.getsize", return_value=50)
@patch("lobster.model._utils_checkpoint._download_checkpoint")
def test_download_checkpoint_rejects_size_mismatch_after_download(mock_download, mock_getsize, mock_remove, mock_valid):
    metadata = _metadata(content_length=100)

    with pytest.raises(RuntimeError, match="size mismatch"):
        download_checkpoint(
            "s3://bucket/checkpoint.ckpt",
            "/tmp",
            "model.ckpt",
            metadata=metadata,
        )


@patch("lobster.model._utils_checkpoint.is_local_checkpoint_valid", return_value=False)
@patch("lobster.model._utils_checkpoint._remove_local_checkpoint")
@patch("lobster.model._utils_checkpoint.os.path.getsize", return_value=100)
@patch("lobster.model._utils_checkpoint._local_md5_hex", return_value="deadbeef")
@patch("lobster.model._utils_checkpoint._download_checkpoint")
def test_download_checkpoint_rejects_md5_mismatch_after_download(
    mock_download, mock_md5, mock_getsize, mock_remove, mock_valid
):
    metadata = _metadata(etag="abc123def456")

    with pytest.raises(RuntimeError, match="checksum mismatch"):
        download_checkpoint(
            "s3://bucket/checkpoint.ckpt",
            "/tmp",
            "model.ckpt",
            metadata=metadata,
        )


# Test _download_checkpoint function
@patch("lobster.model._utils_checkpoint.download_from_s3")
@patch("lobster.model._utils_checkpoint.os.makedirs")
def test_download_checkpoint_success(mock_makedirs, mock_download_s3):
    """Test successful checkpoint download."""
    _download_checkpoint("s3://bucket/checkpoint.ckpt", "/tmp/model.ckpt", "model.ckpt")

    mock_makedirs.assert_called_once_with("/tmp", exist_ok=True)
    mock_download_s3.assert_called_once_with("s3://bucket/checkpoint.ckpt", "/tmp/model.ckpt", force_redownload=True)


@patch("lobster.model._utils_checkpoint.download_from_s3")
@patch("lobster.model._utils_checkpoint.os.makedirs")
def test_download_checkpoint_client_error(mock_makedirs, mock_download_s3):
    """Test checkpoint download with ClientError."""
    mock_download_s3.side_effect = ClientError(
        {"Error": {"Code": "NoSuchBucket", "Message": "Bucket does not exist"}}, "GetObject"
    )

    with pytest.raises(NotImplementedError) as exc_info:
        _download_checkpoint("s3://bucket/checkpoint.ckpt", "/tmp/model.ckpt", "model.ckpt")

    assert "We haven't yet released these checkpoints" in str(exc_info.value)


@patch("lobster.model._utils_checkpoint.download_from_s3")
@patch("lobster.model._utils_checkpoint.os.makedirs")
def test_download_checkpoint_no_credentials(mock_makedirs, mock_download_s3):
    """Test checkpoint download with NoCredentialsError."""
    mock_download_s3.side_effect = NoCredentialsError()

    with pytest.raises(NotImplementedError) as exc_info:
        _download_checkpoint("s3://bucket/checkpoint.ckpt", "/tmp/model.ckpt", "model.ckpt")

    assert "We haven't yet released these checkpoints" in str(exc_info.value)


@patch("lobster.model._utils_checkpoint.download_from_s3")
@patch("lobster.model._utils_checkpoint.os.makedirs")
def test_download_checkpoint_credential_retrieval_error(mock_makedirs, mock_download_s3):
    """Test checkpoint download with CredentialRetrievalError."""
    mock_download_s3.side_effect = CredentialRetrievalError(provider="env", error_msg="No credentials")

    with pytest.raises(NotImplementedError) as exc_info:
        _download_checkpoint("s3://bucket/checkpoint.ckpt", "/tmp/model.ckpt", "model.ckpt")

    assert "We haven't yet released these checkpoints" in str(exc_info.value)


# Test load_checkpoint_with_retry function
@patch("lobster.model._utils_checkpoint.download_checkpoint")
@patch("lobster.model._utils_checkpoint.os.path.exists")
@patch("lobster.model._utils_checkpoint.os.remove")
def test_load_checkpoint_with_retry_success(mock_remove, mock_exists, mock_download):
    """Test successful checkpoint loading."""
    mock_load_func = MagicMock(return_value="loaded_model")
    metadata = _metadata()

    result = load_checkpoint_with_retry(
        "s3://bucket/checkpoint.ckpt",
        "/tmp",
        "model.ckpt",
        mock_load_func,
        metadata=metadata,
    )

    mock_download.assert_called_once_with("s3://bucket/checkpoint.ckpt", "/tmp", "model.ckpt", metadata=metadata)
    mock_load_func.assert_called_once_with("/tmp/model.ckpt")
    assert result == "loaded_model"


@patch("lobster.model._utils_checkpoint.download_checkpoint")
@patch("lobster.model._utils_checkpoint._remove_local_checkpoint")
def test_load_checkpoint_with_retry_corruption_recovery(mock_remove, mock_download):
    """Test checkpoint loading with corruption recovery."""
    mock_load_func = MagicMock()
    mock_load_func.side_effect = [RuntimeError("PytorchStreamReader failed reading zip archive"), "loaded_model"]
    metadata = _metadata()

    result = load_checkpoint_with_retry(
        "s3://bucket/checkpoint.ckpt",
        "/tmp",
        "model.ckpt",
        mock_load_func,
        metadata=metadata,
    )

    # Should call download_checkpoint twice (initial + force redownload)
    assert mock_download.call_count == 2
    mock_download.assert_has_calls(
        [
            call("s3://bucket/checkpoint.ckpt", "/tmp", "model.ckpt", metadata=metadata),
            call(
                "s3://bucket/checkpoint.ckpt",
                "/tmp",
                "model.ckpt",
                force_redownload=True,
                metadata=metadata,
            ),
        ]
    )

    # Should call load_func twice
    assert mock_load_func.call_count == 2
    mock_load_func.assert_has_calls([call("/tmp/model.ckpt"), call("/tmp/model.ckpt")])

    # Should remove corrupted file (+ sidecar)
    mock_remove.assert_called_once_with("/tmp/model.ckpt")

    assert result == "loaded_model"


@patch("lobster.model._utils_checkpoint.download_checkpoint")
@patch("lobster.model._utils_checkpoint._remove_local_checkpoint")
def test_load_checkpoint_with_retry_other_runtime_error(mock_remove, mock_download):
    """Test checkpoint loading with non-corruption RuntimeError."""
    mock_load_func = MagicMock()
    mock_load_func.side_effect = RuntimeError("Some other error")

    with pytest.raises(RuntimeError) as exc_info:
        load_checkpoint_with_retry("s3://bucket/checkpoint.ckpt", "/tmp", "model.ckpt", mock_load_func)

    assert "Some other error" in str(exc_info.value)
    mock_remove.assert_not_called()


@patch("lobster.model._utils_checkpoint.download_checkpoint")
def test_load_checkpoint_with_retry_with_args_kwargs(mock_download):
    """Test checkpoint loading with additional args and kwargs."""
    mock_load_func = MagicMock(return_value="loaded_model")

    result = load_checkpoint_with_retry(
        "s3://bucket/checkpoint.ckpt",
        "/tmp",
        "model.ckpt",
        mock_load_func,
        "arg1",
        "arg2",
        kwarg1="value1",
        kwarg2="value2",
    )

    mock_load_func.assert_called_once_with("/tmp/model.ckpt", "arg1", "arg2", kwarg1="value1", kwarg2="value2")
    assert result == "loaded_model"
