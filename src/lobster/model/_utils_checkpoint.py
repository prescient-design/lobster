import json
import logging
import os
from collections.abc import Callable
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError, CredentialRetrievalError, NoCredentialsError

from lobster.constants import UME_CHECKPOINT_DICT_S3_BUCKET, UME_CHECKPOINT_DICT_S3_KEY
from lobster.data._utils import download_from_s3

logger = logging.getLogger(__name__)


def get_s3_last_modified_timestamp(s3_url: str) -> str:
    """Get the LastModified timestamp from an S3 object and format it for filename use.

    Parameters
    ----------
    s3_url : str
        S3 URL in format s3://bucket/key

    Returns
    -------
    str
        Formatted timestamp string (YYYYMMDD_HHMMSS)

    Examples
    --------
    >>> timestamp = get_s3_last_modified_timestamp("s3://prescient-lobster/ume/runs/2025-06-29T00-03-44/epoch=0-step=24500-val_loss=0.7878.ckpt")
    >>> print(timestamp)
    '20250629_000344'
    """
    # Parse S3 URL
    parsed = urlparse(s3_url)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")

    # Get object metadata
    s3 = boto3.client("s3")
    response = s3.head_object(Bucket=bucket, Key=key)

    # Format timestamp for filename use
    timestamp = response["LastModified"]
    formatted_timestamp = timestamp.strftime("%Y%m%d-%H%M%S")

    return formatted_timestamp


def get_ume_checkpoints() -> dict[str, str]:
    """Get the UME checkpoints from S3."""
    client = boto3.client("s3")
    response = client.get_object(Bucket=UME_CHECKPOINT_DICT_S3_BUCKET, Key=UME_CHECKPOINT_DICT_S3_KEY)
    decoded_body = response["Body"].read().decode("utf-8")

    return json.loads(decoded_body)


def download_checkpoint(
    checkpoint_path: str, local_directory: str, local_filename: str, force_redownload: bool = False
) -> None:
    """Download checkpoint from S3 to local path with proper error handling.

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
    """
    local_path = os.path.join(local_directory, local_filename)

    # Download if not already cached or if force_redownload is True
    if not os.path.exists(local_path) or force_redownload:
        if force_redownload and os.path.exists(local_path):
            logger.info(f"Force redownloading {local_filename} checkpoint...")

        _download_checkpoint(checkpoint_path, local_path, local_filename)


def _download_checkpoint(checkpoint_path: str, local_path: str, local_filename: str) -> None:
    """Internal function to download checkpoint from S3 to local path."""
    try:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        logger.info(f"Downloading checkpoint to {local_path}")

        download_from_s3(checkpoint_path, local_path)

        logger.info("Successfully downloaded model checkpoint.")

    except (ClientError, NoCredentialsError, CredentialRetrievalError) as e:
        raise NotImplementedError(
            "We haven't yet released these checkpoints and it's only available to members of Prescient Design for now. Stay tuned!"
        ) from e


def load_checkpoint_with_retry(
    checkpoint_path: str, local_directory: str, local_filename: str, load_func: Callable, *args, **kwargs
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

     Returns
     -------
     The loaded model/checkpoint
    """
    local_path = os.path.join(local_directory, local_filename)

    # Get function name safely, handling mock objects
    func_name = getattr(load_func, "__name__", str(load_func))

    logger.debug("Loading checkpoint with retry:")
    logger.debug(f"  - S3 path: {checkpoint_path}")
    logger.debug(f"  - Local path: {local_path}")
    logger.debug(f"  - Load function: {func_name}")

    # First, ensure checkpoint is downloaded
    logger.debug("Ensuring checkpoint is downloaded...")
    download_checkpoint(checkpoint_path, local_directory, local_filename)

    # Try to load the model
    logger.debug("Attempting to load checkpoint...")
    try:
        model = load_func(local_path, *args, **kwargs)
        logger.info("✅ Successfully loaded checkpoint.")
        return model
    except RuntimeError as e:
        if "PytorchStreamReader failed reading zip archive" in str(e):
            logger.warning(f"❌ Downloaded checkpoint {local_filename} appears corrupted. Redownloading...")
            logger.warning(f"   Error: {e}")

            # Remove corrupted file and redownload
            if os.path.exists(local_path):
                logger.info(f"Removing corrupted file: {local_path}")
                os.remove(local_path)

            # Force redownload
            logger.info("Forcing redownload of checkpoint...")
            download_checkpoint(checkpoint_path, local_directory, local_filename, force_redownload=True)

            # Try loading again
            logger.info("Attempting to load checkpoint after redownload...")
            try:
                model = load_func(local_path, *args, **kwargs)
                logger.info("✅ Successfully loaded checkpoint after redownload")
                return model
            except Exception as e2:
                logger.error(f"❌ Failed to load checkpoint even after redownload: {e2}")
                raise
        else:
            logger.error(f"❌ Failed to load checkpoint (non-corruption error): {e}")
            raise
