import hashlib
import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError, CredentialRetrievalError, NoCredentialsError

from lobster.constants import UME_CHECKPOINT_DICT_S3_BUCKET, UME_CHECKPOINT_DICT_S3_KEY
from lobster.data._utils import download_from_s3

logger = logging.getLogger(__name__)

CHECKSUM_SIDECAR_SUFFIX = ".checksum.json"


@dataclass(frozen=True)
class S3ObjectMetadata:
    """Metadata from an S3 HEAD request used for cache invalidation."""

    etag: str
    content_length: int
    last_modified: datetime
    s3_uri: str


def _normalize_etag(etag: str) -> str:
    """Strip surrounding quotes from an S3 ETag value."""
    return etag.strip('"')


def get_s3_object_metadata(s3_url: str) -> S3ObjectMetadata:
    """Fetch ETag, ContentLength, and LastModified for an S3 object.

    The ETag is used as a practical content checksum for cache validation.
    For single-part uploads it is the MD5 of the object bytes; for multipart
    uploads it is a composite value (contains ``-``) that still changes when
    the object bytes change. Published SHA256 checksums are not currently
    available for UME checkpoints, so ETag + size is the verification signal.

    Parameters
    ----------
    s3_url : str
        S3 URL in format ``s3://bucket/key``.

    Returns
    -------
    S3ObjectMetadata
        Remote object metadata used for local cache verification.
    """
    parsed = urlparse(s3_url)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")

    s3 = boto3.client("s3")
    response = s3.head_object(Bucket=bucket, Key=key)

    return S3ObjectMetadata(
        etag=_normalize_etag(response["ETag"]),
        content_length=int(response["ContentLength"]),
        last_modified=response["LastModified"],
        s3_uri=s3_url,
    )


def get_s3_last_modified_timestamp(s3_url: str) -> str:
    """Get the LastModified timestamp from an S3 object and format it for filename use.

    Parameters
    ----------
    s3_url : str
        S3 URL in format s3://bucket/key

    Returns
    -------
    str
        Formatted timestamp string (YYYYMMDD-HHMMSS)

    Examples
    --------
    >>> timestamp = get_s3_last_modified_timestamp("s3://prescient-lobster/ume/runs/2025-06-29T00-03-44/epoch=0-step=24500-val_loss=0.7878.ckpt")
    >>> print(timestamp)
    '20250629_000344'
    """
    metadata = get_s3_object_metadata(s3_url)
    return metadata.last_modified.strftime("%Y%m%d-%H%M%S")


def get_ume_checkpoints() -> dict[str, str]:
    """Get the UME checkpoints from S3."""
    client = boto3.client("s3")
    response = client.get_object(Bucket=UME_CHECKPOINT_DICT_S3_BUCKET, Key=UME_CHECKPOINT_DICT_S3_KEY)
    decoded_body = response["Body"].read().decode("utf-8")

    return json.loads(decoded_body)


def checkpoint_checksum_path(local_path: str) -> str:
    """Return the sidecar path that stores cache checksum metadata."""
    return f"{local_path}{CHECKSUM_SIDECAR_SUFFIX}"


def _write_checksum_sidecar(local_path: str, metadata: S3ObjectMetadata) -> None:
    sidecar = {
        "etag": metadata.etag,
        "content_length": metadata.content_length,
        "s3_uri": metadata.s3_uri,
    }
    with open(checkpoint_checksum_path(local_path), "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2, sort_keys=True)
        f.write("\n")


def _read_checksum_sidecar(local_path: str) -> dict | None:
    sidecar_path = checkpoint_checksum_path(local_path)
    if not os.path.exists(sidecar_path):
        return None
    try:
        with open(sidecar_path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        logger.warning("Failed to read checkpoint checksum sidecar at %s", sidecar_path)
        return None
    if not isinstance(data, dict) or "etag" not in data or "content_length" not in data:
        return None
    return data


def _local_md5_hex(local_path: str) -> str:
    digest = hashlib.md5()
    with open(local_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_local_checkpoint_valid(local_path: str, metadata: S3ObjectMetadata) -> bool:
    """Return True if the local checkpoint matches remote ETag/size metadata.

    Validation rules:
    - Local file must exist and its size must match ``ContentLength``.
    - A checksum sidecar must exist and its stored ETag/size must match remote.
    - For single-part ETags (no ``-``), also verify the local MD5 equals the ETag.
      Multipart ETags are not plain MD5s, so only the sidecar + size check applies.
    """
    if not os.path.exists(local_path):
        return False

    try:
        local_size = os.path.getsize(local_path)
    except OSError:
        return False

    if local_size != metadata.content_length:
        logger.info(
            "Local checkpoint size mismatch for %s (local=%s, remote=%s)",
            local_path,
            local_size,
            metadata.content_length,
        )
        return False

    sidecar = _read_checksum_sidecar(local_path)
    if sidecar is None:
        # Legacy caches (pre-checksum) without a sidecar: accept only when we can
        # cryptographically confirm a single-part ETag via MD5; otherwise redownload.
        if "-" in metadata.etag:
            logger.info("Missing checksum sidecar for %s; re-downloading to refresh cache metadata", local_path)
            return False
        local_md5 = _local_md5_hex(local_path)
        if local_md5 != metadata.etag:
            logger.info("Local checkpoint MD5 does not match remote ETag for %s; re-downloading", local_path)
            return False
        _write_checksum_sidecar(local_path, metadata)
        return True

    if sidecar.get("etag") != metadata.etag or int(sidecar["content_length"]) != metadata.content_length:
        logger.info("Checksum sidecar mismatch for %s; remote checkpoint changed", local_path)
        return False

    if "-" not in metadata.etag:
        local_md5 = _local_md5_hex(local_path)
        if local_md5 != metadata.etag:
            logger.info("Local checkpoint MD5 does not match stored/remote ETag for %s; re-downloading", local_path)
            return False

    return True


def _remove_local_checkpoint(local_path: str) -> None:
    for path in (local_path, checkpoint_checksum_path(local_path)):
        if os.path.exists(path):
            os.remove(path)


def download_checkpoint(
    checkpoint_path: str,
    local_directory: str,
    local_filename: str,
    force_redownload: bool = False,
    metadata: S3ObjectMetadata | None = None,
) -> None:
    """Download checkpoint from S3 to local path with checksum-aware caching.

    When a local file already exists, its size and checksum sidecar (S3 ETag)
    are compared to the remote object. A mismatch or missing/invalid sidecar
    triggers a re-download so stale or corrupt caches are not reused.

    Parameters
    ----------
    checkpoint_path : str
        S3 path to the checkpoint file
    local_directory : str
        Local directory where to save the checkpoint
    local_filename : str
        Filename to save the checkpoint as
    force_redownload : bool, default=False
        Whether to force redownload even if file exists
    metadata : S3ObjectMetadata | None, optional
        Pre-fetched remote metadata. When None, a HEAD request is issued.
    """
    local_path = os.path.join(local_directory, local_filename)
    remote_metadata = metadata or get_s3_object_metadata(checkpoint_path)

    if not force_redownload and is_local_checkpoint_valid(local_path, remote_metadata):
        logger.debug("Using cached checkpoint with matching checksum: %s", local_path)
        return

    if force_redownload:
        logger.info("Force redownloading %s checkpoint...", local_filename)
    elif os.path.exists(local_path):
        logger.info("Redownloading %s; local cache failed checksum verification", local_filename)

    _remove_local_checkpoint(local_path)
    _download_checkpoint(checkpoint_path, local_path, local_filename)

    actual_size = os.path.getsize(local_path)
    if actual_size != remote_metadata.content_length:
        _remove_local_checkpoint(local_path)
        raise RuntimeError(
            f"Downloaded checkpoint size mismatch for {local_filename}: "
            f"got {actual_size} bytes, expected {remote_metadata.content_length}"
        )

    if "-" not in remote_metadata.etag:
        local_md5 = _local_md5_hex(local_path)
        if local_md5 != remote_metadata.etag:
            _remove_local_checkpoint(local_path)
            raise RuntimeError(
                f"Downloaded checkpoint checksum mismatch for {local_filename}: "
                f"MD5 {local_md5} != ETag {remote_metadata.etag}"
            )

    _write_checksum_sidecar(local_path, remote_metadata)


def _download_checkpoint(checkpoint_path: str, local_path: str, local_filename: str) -> None:
    """Internal function to download checkpoint from S3 to local path."""
    try:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        logger.info(f"Downloading checkpoint to {local_path}")

        download_from_s3(checkpoint_path, local_path, force_redownload=True)

        logger.debug("Successfully downloaded model checkpoint.")

    except (ClientError, NoCredentialsError, CredentialRetrievalError) as e:
        raise NotImplementedError(
            "We haven't yet released these checkpoints and it's only available to members of Prescient Design for now. Stay tuned!"
        ) from e


def load_checkpoint_with_retry(
    checkpoint_path: str,
    local_directory: str,
    local_filename: str,
    load_func: Callable,
    *args,
    metadata: S3ObjectMetadata | None = None,
    **kwargs,
):
    """Load checkpoint with automatic retry on corruption.
    If previous download was stopped, the checkpoint will be corrupted
    so we need to redownload it.

     Parameters
     ----------
     checkpoint_path : str
         S3 path to the checkpoint file
     local_directory : str
         Local directory where the checkpoint is saved
     local_filename : str
         Filename of the checkpoint
     load_func : callable
         Function to load the checkpoint (e.g., cls.load_from_checkpoint)
     *args, **kwargs
         Arguments to pass to load_func
     metadata : S3ObjectMetadata | None, optional
         Pre-fetched remote object metadata for checksum verification.

     Returns
     -------
     The loaded model/checkpoint
    """
    local_path = os.path.join(local_directory, local_filename)

    # First, ensure checkpoint is downloaded and checksum-validated
    download_checkpoint(checkpoint_path, local_directory, local_filename, metadata=metadata)

    # Try to load the model
    try:
        model = load_func(local_path, *args, **kwargs)
        return model

    except RuntimeError as e:
        if "PytorchStreamReader failed reading zip archive" in str(e):
            logger.warning(f"Downloaded checkpoint {local_filename} appears corrupted. Redownloading...")

            # Remove corrupted file and redownload
            _remove_local_checkpoint(local_path)

            download_checkpoint(
                checkpoint_path,
                local_directory,
                local_filename,
                force_redownload=True,
                metadata=metadata,
            )

            # Try loading again
            try:
                model = load_func(local_path, *args, **kwargs)
                return model
            except Exception as e2:
                logger.error(f"Failed to load checkpoint even after redownload: {e2}")
                raise
        else:
            logger.error(f"Failed to load checkpoint (non-corruption error): {e}")
            raise
