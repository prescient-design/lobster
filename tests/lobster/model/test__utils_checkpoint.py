from unittest.mock import MagicMock, call, patch

import pytest
from botocore.exceptions import ClientError, CredentialRetrievalError, NoCredentialsError

from lobster.model._utils_checkpoint import _download_checkpoint, download_checkpoint, load_checkpoint_with_retry


# Test download_checkpoint function
@patch("lobster.model._utils_checkpoint.os.path.exists")
@patch("lobster.model._utils_checkpoint._download_checkpoint")
def test_download_checkpoint_file_not_exists(mock_download, mock_exists):
    """Test download_checkpoint when file doesn't exist."""
    mock_exists.return_value = False

    download_checkpoint("s3://bucket/checkpoint.ckpt", "/tmp", "model.ckpt")

    mock_download.assert_called_once_with("s3://bucket/checkpoint.ckpt", "/tmp/model.ckpt", "model.ckpt")


@patch("lobster.model._utils_checkpoint.os.path.exists")
@patch("lobster.model._utils_checkpoint._download_checkpoint")
def test_download_checkpoint_file_exists_no_force(mock_download, mock_exists):
    """Test download_checkpoint when file exists and force_redownload=False."""
    mock_exists.return_value = True

    download_checkpoint("s3://bucket/checkpoint.ckpt", "/tmp", "model.ckpt")

    mock_download.assert_not_called()


@patch("lobster.model._utils_checkpoint.os.path.exists")
@patch("lobster.model._utils_checkpoint._download_checkpoint")
def test_download_checkpoint_force_redownload(mock_download, mock_exists):
    """Test download_checkpoint with force_redownload=True."""
    mock_exists.return_value = True

    download_checkpoint("s3://bucket/checkpoint.ckpt", "/tmp", "model.ckpt", force_redownload=True)

    mock_download.assert_called_once_with("s3://bucket/checkpoint.ckpt", "/tmp/model.ckpt", "model.ckpt")


# Test _download_checkpoint function
@patch("lobster.model._utils_checkpoint.download_from_s3")
@patch("lobster.model._utils_checkpoint.os.makedirs")
def test_download_checkpoint_success(mock_makedirs, mock_download_s3):
    """Test successful checkpoint download."""
    _download_checkpoint("s3://bucket/checkpoint.ckpt", "/tmp/model.ckpt", "model.ckpt")

    mock_makedirs.assert_called_once_with("/tmp", exist_ok=True)
    mock_download_s3.assert_called_once_with("s3://bucket/checkpoint.ckpt", "/tmp/model.ckpt")


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

    result = load_checkpoint_with_retry("s3://bucket/checkpoint.ckpt", "/tmp", "model.ckpt", mock_load_func)

    mock_download.assert_called_once_with("s3://bucket/checkpoint.ckpt", "/tmp", "model.ckpt")
    mock_load_func.assert_called_once_with("/tmp/model.ckpt")
    assert result == "loaded_model"


@patch("lobster.model._utils_checkpoint.download_checkpoint")
@patch("lobster.model._utils_checkpoint.os.path.exists")
@patch("lobster.model._utils_checkpoint.os.remove")
def test_load_checkpoint_with_retry_corruption_recovery(mock_remove, mock_exists, mock_download):
    """Test checkpoint loading with corruption recovery."""
    mock_load_func = MagicMock()
    mock_load_func.side_effect = [RuntimeError("PytorchStreamReader failed reading zip archive"), "loaded_model"]
    mock_exists.return_value = True

    result = load_checkpoint_with_retry("s3://bucket/checkpoint.ckpt", "/tmp", "model.ckpt", mock_load_func)

    # Should call download_checkpoint twice (initial + force redownload)
    assert mock_download.call_count == 2
    mock_download.assert_has_calls(
        [
            call("s3://bucket/checkpoint.ckpt", "/tmp", "model.ckpt"),
            call("s3://bucket/checkpoint.ckpt", "/tmp", "model.ckpt", force_redownload=True),
        ]
    )

    # Should call load_func twice
    assert mock_load_func.call_count == 2
    mock_load_func.assert_has_calls([call("/tmp/model.ckpt"), call("/tmp/model.ckpt")])

    # Should remove corrupted file
    mock_remove.assert_called_once_with("/tmp/model.ckpt")

    assert result == "loaded_model"


@patch("lobster.model._utils_checkpoint.download_checkpoint")
@patch("lobster.model._utils_checkpoint.os.path.exists")
@patch("lobster.model._utils_checkpoint.os.remove")
def test_load_checkpoint_with_retry_other_runtime_error(mock_remove, mock_exists, mock_download):
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
