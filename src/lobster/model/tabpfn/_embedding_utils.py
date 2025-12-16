import logging
from typing import Literal

import lightning.pytorch as pl
import torch

from .._utils import model_typer

logger = logging.getLogger(__name__)


def initialize_embedding_model(
    embedding_model_type: str,
    embedding_model_name: str | None,
    embedding_checkpoint: str | None,
    max_length: int,
) -> pl.LightningModule:
    """Initialize protein embedding model.

    Parameters
    ----------
    embedding_model_type : str
        Type of embedding model
    embedding_model_name : str | None
        Pretrained model name
    embedding_checkpoint : str | None
        Path to checkpoint
    max_length : int
        Maximum sequence length

    Returns
    -------
    pl.LightningModule
        Initialized embedding model

    Raises
    ------
    ValueError
        If neither model name nor checkpoint provided
    """
    if (
        embedding_model_name is not None
        and embedding_checkpoint is not None
        or embedding_model_name is None
        and embedding_checkpoint is None
    ):
        raise ValueError("Must provide either embedding_model_name or embedding_checkpoint")

    model_cls = model_typer[embedding_model_type]
    logger.info(f"Loading embedding model: {model_cls}")

    if embedding_model_name is not None:
        return model_cls(model_name=embedding_model_name, max_length=max_length)

    else:
        return model_cls.load_from_checkpoint(
            embedding_checkpoint,
            max_length=max_length,
        )


def extract_embeddings(
    embedding_model: pl.LightningModule,
    batch: dict,
    pooling: Literal["mean", "max", "cls"],
    freeze_embeddings: bool,
    additional_features: list[str],
) -> torch.Tensor:
    """Extract embeddings from batch using the embedding model.

    Parameters
    ----------
    embedding_model : pl.LightningModule
        Protein embedding model
    batch : dict
        Batch dictionary containing input_ids, attention_mask, etc.
    pooling : Literal["mean", "max", "cls"]
        How to pool per-residue embeddings to sequence-level
    freeze_embeddings : bool
        Whether embeddings are frozen (affects gradient computation)
    additional_features : list[str]
        Names of additional features to concatenate

    Returns
    -------
    torch.Tensor
        Pooled embeddings of shape (batch_size, embedding_dim)
    """
    if freeze_embeddings:
        with torch.no_grad():
            outputs = _get_model_outputs(embedding_model, batch)
    else:
        outputs = _get_model_outputs(embedding_model, batch)

    hidden_states = outputs["hidden_states"][-1]
    embeddings = _pool_embeddings(hidden_states, batch["attention_mask"], pooling)

    if additional_features:
        embeddings = _concatenate_additional_features(embeddings, batch, additional_features)

    return embeddings


def _get_model_outputs(embedding_model: pl.LightningModule, batch: dict) -> dict:
    """Get outputs from embedding model, handling different model structures."""
    if hasattr(embedding_model, "model"):
        return embedding_model.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            output_hidden_states=True,
        )
    else:
        return embedding_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            output_hidden_states=True,
        )


def _pool_embeddings(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    pooling: Literal["mean", "max", "cls"],
) -> torch.Tensor:
    """Pool per-residue embeddings to sequence-level."""
    if pooling == "mean":
        attention_mask = attention_mask.unsqueeze(-1)
        return (hidden_states * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
    elif pooling == "max":
        return hidden_states.max(dim=1)[0]
    elif pooling == "cls":
        return hidden_states[:, 0, :]
    else:
        raise ValueError(f"Unknown pooling method: {pooling}")


def _concatenate_additional_features(
    embeddings: torch.Tensor,
    batch: dict,
    additional_features: list[str],
) -> torch.Tensor:
    """Concatenate additional features to embeddings."""
    additional_feats = []
    for feat_name in additional_features:
        if feat_name in batch:
            feat = batch[feat_name]
            if feat.dim() == 1:
                feat = feat.unsqueeze(-1)
            additional_feats.append(feat)

    if additional_feats:
        return torch.cat([embeddings] + additional_feats, dim=-1)

    return embeddings


def get_embedding_dim(embedding_model: pl.LightningModule, num_chains: int) -> int:
    """Get embedding dimension from model.

    Parameters
    ----------
    embedding_model : pl.LightningModule
        Protein embedding model
    num_chains : int
        Number of protein chains

    Returns
    -------
    int
        Total embedding dimension (hidden_size * num_chains)

    Raises
    ------
    ValueError
        If cannot determine hidden size from model
    """
    if hasattr(embedding_model, "config"):
        hidden_size = embedding_model.config.hidden_size
    elif hasattr(embedding_model, "embedding_dim"):
        hidden_size = embedding_model.embedding_dim
    else:
        raise ValueError("Cannot determine hidden size from embedding model")

    return hidden_size * num_chains
