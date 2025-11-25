"""Utilities for loading models and generating embeddings."""

import logging
import os
import tempfile
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import nn

from lobster.model._utils_checkpoint import download_checkpoint
from lobster.tokenization import UMETokenizerTransform

logger = logging.getLogger(__name__)


def load_model_for_embedding(
    checkpoint_path: str,
    model_type: Literal["neobert", "ume", "ume2"] = "neobert",
    device: str = "cuda",
    cache_dir: str | None = None,
) -> nn.Module:
    """Load a pretrained model for embedding generation.
    
    Unified interface for loading different model types. Handles S3 downloads automatically.
    For UME models, consider using UME.from_pretrained() for named checkpoints.
    """
    # Handle S3 downloads
    if checkpoint_path.startswith("s3://"):
        if cache_dir is None:
            cache_dir = os.path.join(tempfile.gettempdir(), "lobster_checkpoints")
        os.makedirs(cache_dir, exist_ok=True)
        
        local_filename = Path(checkpoint_path).name
        local_path = os.path.join(cache_dir, local_filename)
        
        if not os.path.exists(local_path):
            logger.info(f"Downloading checkpoint from {checkpoint_path}")
            download_checkpoint(checkpoint_path, cache_dir, local_filename)
        else:
            logger.info(f"Using cached checkpoint at {local_path}")
        
        checkpoint_path = local_path
    
    # Load model
    logger.info(f"Loading {model_type} model from {checkpoint_path}")
    
    if model_type == "neobert":
        from lobster.model.neobert import NeoBERTLightningModule
        model = NeoBERTLightningModule.load_from_checkpoint(checkpoint_path)
    elif model_type == "ume":
        from lobster.model import UME
        model = UME.load_from_checkpoint(checkpoint_path)
    elif model_type == "ume2":
        from lobster.model.ume2 import UMESequenceEncoderLightningModule
        model = UMESequenceEncoderLightningModule.load_from_checkpoint(checkpoint_path)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Choose from: neobert, ume, ume2")
    
    model = model.to(device)
    model.eval()
    logger.info(f"Model loaded successfully on {device}")
    return model


def embed_sequences_batch(
    sequences: list[str],
    model: nn.Module,
    tokenizer: UMETokenizerTransform,
    device: str = "cuda",
    aggregate: bool = True,
    ignore_padding: bool = True,
) -> np.ndarray:
    """Generate embeddings for a batch of sequences.
    
    Convenience wrapper that tokenizes, embeds via model.embed(), and converts to numpy.
    """
    encoded = tokenizer(sequences)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    
    with torch.inference_mode():
        embeddings = model.embed(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            aggregate=aggregate,
            ignore_padding=ignore_padding,
        )
    
    return embeddings.cpu().numpy()
