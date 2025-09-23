import os
import logging

import torch
import torch.nn as nn

from lobster.data import download_from_s3

logger = logging.getLogger(__name__)


def _load_checkpoint_from_path(model: nn.Module, model_ckpt: str, cache_dir: str | None = None) -> None:
    """
    Load checkpoint from local path or S3 URL into the model.

    Parameters
    ----------
    model : nn.Module
        The model to load the checkpoint into
    model_ckpt : str
        Path to checkpoint file (local path or S3 URL)
    cache_dir : str | None, optional
        Directory to cache downloaded checkpoints (required for S3 URLs)

    Raises
    ------
    ValueError
        If cache_dir is None when model_ckpt is an S3 path
    """
    if model_ckpt.startswith("s3://"):
        if cache_dir is None:
            raise ValueError("cache_dir must be provided if model_ckpt is an S3 path")

        model_name = model_ckpt.split("/")[-1]
        local_filepath = os.path.join(cache_dir, model_name)

        logger.info(f"Downloading checkpoint from {model_ckpt} to {local_filepath}")
        download_from_s3(model_ckpt, local_filepath)

        model_ckpt = local_filepath

    device = next(iter(model.parameters())).device
    checkpoint = torch.load(model_ckpt, map_location=device, weights_only=False)

    logger.info(f"Loading checkpoint from {model_ckpt}")

    # Map checkpoint keys to match current model structure
    state_dict = checkpoint["state_dict"]
    mapped_state_dict = _map_checkpoint_keys(state_dict)
    model.load_state_dict(mapped_state_dict)


def _map_checkpoint_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Map checkpoint keys of NeoBERTLightningModuleto match current model structure.

    The checkpoint contains keys with an extra 'model.' prefix that needs to be removed.
    For example: 'model.model.encoder.weight' -> 'model.encoder.weight'
    """
    mapped_state_dict = {}

    for key, value in state_dict.items():
        # Remove the extra 'model.' prefix if it exists
        if key.startswith("model.model."):
            new_key = key.replace("model.model.", "model.", 1)
            mapped_state_dict[new_key] = value
            logger.debug(f"Mapped key: {key} -> {new_key}")

        elif key.startswith("model.decoder."):
            new_key = key.replace("model.decoder.", "decoder.", 1)
            mapped_state_dict[new_key] = value
            logger.debug(f"Mapped key: {key} -> {new_key}")

        else:
            mapped_state_dict[key] = value

    return mapped_state_dict
