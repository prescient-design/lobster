import os
import logging

import torch

from lobster.data import download_from_s3

logger = logging.getLogger(__name__)


def load_checkpoint_from_s3_uri_or_local_path(
    model_ckpt: str, device: str | None = None, cache_dir: str | None = None, force_redownload: bool = False
) -> dict[str, torch.Tensor]:
    """Load a checkpoint from a local path or S3 URI."""

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_ckpt.startswith("s3://"):
        if cache_dir is None:
            raise ValueError("cache_dir must be provided if model_ckpt is an S3 path")

        model_name = model_ckpt.split("/")[-1]
        model_dirname = model_ckpt.split("/")[-2]
        os.makedirs(os.path.join(cache_dir, model_dirname), exist_ok=True)

        local_filepath = os.path.join(cache_dir, model_dirname, model_name)

        if os.path.exists(local_filepath) and not force_redownload:
            logger.info(f"Checkpoint already exists at {local_filepath}")

        else:
            logger.info(f"Downloading checkpoint from {model_ckpt} to {local_filepath}")
            download_from_s3(model_ckpt, local_filepath)

        model_ckpt = local_filepath

    logger.info(f"Loading checkpoint from {model_ckpt} with torch.load, map_location={device}, weights_only=False")
    checkpoint = torch.load(model_ckpt, map_location=device, weights_only=False)

    return checkpoint


def map_checkpoint_keys(
    state_dict: dict[str, torch.Tensor], original_prefix: str, new_prefix: str
) -> dict[str, torch.Tensor]:
    """Map checkpoint keys to match current model structure."""
    mapped_state_dict = {}
    for key, value in state_dict.items():
        mapped_state_dict[key.replace(original_prefix, new_prefix, 1)] = value

    return mapped_state_dict


# TODO: Add config.json support for checkpoint parameters
# Once config.json files are added alongside checkpoints in S3, implement a
# load_config_json() function to load non-critical params (model_size, pad_token_id, max_length)


def infer_architecture_from_state_dict(state_dict: dict[str, torch.Tensor], prefix: str = "") -> dict:
    """
    Infer critical architecture parameters from checkpoint state_dict.
    
    Infers vocab_size and hidden_size from encoder weight shapes.
    These parameters MUST match checkpoint weights and cannot be overridden.
    """
    inferred_params = {}
    
    # Try different possible key formats
    encoder_key_variants = [
        f"{prefix}model.encoder.weight",
        "model.encoder.weight",
        "encoder.neobert.model.encoder.weight",
    ]
    
    for encoder_key in encoder_key_variants:
        if encoder_key in state_dict:
            vocab_size, hidden_size = state_dict[encoder_key].shape
            inferred_params["vocab_size"] = vocab_size
            inferred_params["hidden_size"] = hidden_size
            logger.info(f"Inferred from checkpoint key '{encoder_key}': vocab_size={vocab_size}, hidden_size={hidden_size}")
            break
    
    if not inferred_params:
        logger.warning(f"Could not find encoder.weight in state_dict to infer vocab_size/hidden_size. Keys: {list(state_dict.keys())[:10]}")
    
    return inferred_params
