"""Utility functions for PEER evaluation callback.

This module contains extracted utility functions to reduce the complexity of
the main PEEREvaluationCallback class.
"""

import logging

import lightning as L
import numpy as np
import torch
from torch import Tensor

from lobster.constants import PEER_TASK_METRICS, PEER_TASKS, PEERTask

logger = logging.getLogger(__name__)


def convert_numpy_to_python(obj):
    """Recursively convert NumPy scalars to Python types for clean YAML formatting."""
    match obj:
        case dict():
            return {key: convert_numpy_to_python(value) for key, value in obj.items()}
        case list() | tuple():
            return [convert_numpy_to_python(item) for item in obj]
        case np.integer():
            return int(obj)
        case np.floating():
            return float(obj)
        case np.bool_():
            return bool(obj)
        case _:
            return obj


def get_peer_task_metric(task: PEERTask) -> str:
    """Get the relevant evaluation metric for a given PEER task.

    Parameters
    ----------
    task : PEERTask
        The PEER task to get the metric for

    Returns
    -------
    str
        The metric name ('accuracy', 'spearman', 'rmse', or 'l5_precision')
    """
    # Return task-specific metric if available
    if task in PEER_TASK_METRICS:
        return PEER_TASK_METRICS[task]

    # Fallback to default based on task type
    task_type = PEER_TASKS[task][0]
    if task_type in {"binary", "multiclass", "multilabel"}:
        return "accuracy"
    else:  # regression
        return "rmse"


def peer_structure_collate_fn(batch):
    """Custom collation function for PEER structure prediction tasks.

    Handles variable-length tensors that can't be stacked by default collation.
    """
    inputs, targets = zip(*batch)
    return list(inputs), list(targets)


def peer_default_collate_fn(batch):
    """Default collation function for PEER tasks that can be batched normally."""
    try:
        return torch.utils.data.default_collate(batch)
    except Exception:
        # If default collation fails, fall back to list format
        return peer_structure_collate_fn(batch)


def flatten_and_filter_token_embeddings(
    batch_embeddings: Tensor,
    targets: Tensor,
    input_ids: Tensor | None = None,
    attention_mask: Tensor | None = None,
    ignore_target_value: int = -100,
    tokenizer=None,
    requires_tokenization: bool = True,
) -> tuple[Tensor, Tensor]:
    """Helper function to flatten embeddings and filter special tokens for token-level tasks.

    Parameters
    ----------
    batch_embeddings : Tensor
        Embeddings with shape [batch_size, seq_len, hidden_size] or [seq_len, hidden_size]
    targets : Tensor
        Target labels with shape [batch_size, seq_len] or [batch_size*seq_len] or [orig_seq_len]
    input_ids : Tensor | None, default=None
        Token IDs from tokenizer, used to identify special tokens
    attention_mask : Tensor | None, default=None
        Attention mask from tokenizer
    ignore_target_value : int, default=-100
        Value in targets to ignore (typically padding value)
    tokenizer : tokenizer object, default=None
        Tokenizer object to get special token IDs
    requires_tokenization : bool, default=True
        Whether the model requires tokenization

    Returns
    -------
    Tuple[Tensor, Tensor]
        Filtered embeddings and targets
    """
    # Handle both batched and single sequence cases
    if batch_embeddings.dim() == 3:
        _batch_size, _seq_len, hidden_size = batch_embeddings.shape
        # Flatten embeddings to (batch_size*seq_len, hidden_size)
        batch_embeddings_flat = batch_embeddings.reshape(-1, hidden_size)
    else:
        # Single sequence case [seq_len, hidden_size]
        _seq_len, hidden_size = batch_embeddings.shape
        batch_embeddings_flat = batch_embeddings

    # Handle target length mismatch with tokenized sequence
    orig_target_len = targets.numel()
    expected_len = batch_embeddings_flat.shape[0]

    # Flatten targets if not already flattened
    if targets.dim() > 1:
        targets_flat = targets.reshape(-1)
    else:
        targets_flat = targets

    # Handle length mismatch between targets and embeddings
    if orig_target_len != expected_len:
        if orig_target_len < expected_len:
            # Pad targets with ignore_target_value for padded positions
            padded_targets = torch.full(
                (expected_len,), ignore_target_value, dtype=targets_flat.dtype, device=targets_flat.device
            )
            padded_targets[:orig_target_len] = targets_flat
            targets_flat = padded_targets
        else:
            # Truncate targets if they're longer than embeddings
            targets_flat = targets_flat[:expected_len]

    # If we have tokenized input and we're using a tokenizer-based model, use it to filter special tokens
    if input_ids is not None and attention_mask is not None and requires_tokenization and tokenizer is not None:
        special_token_ids = {
            tokenizer.cls_token_id,
            tokenizer.eos_token_id,
            tokenizer.pad_token_id,
            tokenizer.sep_token_id,
            tokenizer.mask_token_id,
            tokenizer.unk_token_id,
        }

        # Create a mask for real tokens (not special tokens and not padding)
        valid_token_mask = torch.ones_like(input_ids, dtype=torch.bool)
        for special_id in special_token_ids:
            if special_id is not None:  # In case some IDs are None
                valid_token_mask &= input_ids != special_id
        valid_token_mask &= attention_mask.bool()

        # Also filter based on target values to ignore
        target_mask = targets_flat != ignore_target_value

        # Flatten mask & filter embeddings based on token mask
        valid_token_mask_flat = valid_token_mask.reshape(-1)

        # Ensure mask length matches
        if len(valid_token_mask_flat) != len(targets_flat):
            min_len = min(len(valid_token_mask_flat), len(targets_flat))
            valid_token_mask_flat = valid_token_mask_flat[:min_len]
            targets_flat = targets_flat[:min_len]
            batch_embeddings_flat = batch_embeddings_flat[:min_len]

        # Combine both masks
        combined_mask = valid_token_mask_flat & target_mask

        filtered_embeddings = batch_embeddings_flat[combined_mask]
        filtered_targets = targets_flat[combined_mask]
    else:
        # If no tokenizer info provided or using sequence-based model, just filter based on ignore value
        valid_mask = targets_flat != ignore_target_value
        filtered_embeddings = batch_embeddings_flat[valid_mask]
        filtered_targets = targets_flat[valid_mask]

    return filtered_embeddings, filtered_targets


def process_secondary_structure_item(
    pl_module: L.LightningModule,
    x,
    y,
    input_ids: Tensor | None = None,
    attention_mask: Tensor | None = None,
    process_and_embed_fn=None,
    tokenizer=None,
    requires_tokenization: bool = True,
) -> tuple[Tensor, Tensor]:
    """Process a single secondary structure prediction item.

    Parameters
    ----------
    pl_module : L.LightningModule
        The model to use for embedding
    x : input data
        Input sequence or tokenized input
    y : target data
        Target labels
    input_ids : Tensor | None
        Input token IDs if available
    attention_mask : Tensor | None
        Attention mask if available
    process_and_embed_fn : callable
        Function to process inputs and get embeddings
    tokenizer : tokenizer object
        Tokenizer for filtering special tokens
    requires_tokenization : bool
        Whether the model requires tokenization

    Returns
    -------
    Tuple[Tensor, Tensor]
        Filtered embeddings and targets, or empty tensors if no valid data
    """
    # Get per-residue embeddings
    batch_embeddings = process_and_embed_fn(pl_module, x, modality="amino_acid", aggregate=False)

    # Handle single item case (batch_size=1 due to collation)
    if batch_embeddings.dim() == 3:
        batch_embeddings = batch_embeddings.squeeze(0)  # Remove batch dimension

    # Use helper method to flatten and filter token-level embeddings
    filtered_embeddings, filtered_targets = flatten_and_filter_token_embeddings(
        batch_embeddings=batch_embeddings,
        targets=y,
        input_ids=input_ids,
        attention_mask=attention_mask,
        ignore_target_value=-100,  # Use -100 as the padding value for targets
        tokenizer=tokenizer,
        requires_tokenization=requires_tokenization,
    )

    # Only return if we have valid embeddings/targets
    if filtered_embeddings.numel() > 0 and filtered_targets.numel() > 0:
        return filtered_embeddings.cpu(), filtered_targets.cpu()
    else:
        return torch.tensor([]), torch.tensor([])


def process_proteinnet_item(
    pl_module: L.LightningModule,
    x,
    y,
    process_and_embed_fn=None,
    max_length: int = 512,
    manage_memory: bool = True,
    clear_memory_cache_fn=None,
) -> tuple[Tensor, Tensor]:
    """Process a single PROTEINNET contact prediction item.

    Parameters
    ----------
    pl_module : L.LightningModule
        The model to use for embedding
    x : input data
        Input sequence or tokenized input
    y : target data
        Tuple of (tertiary_coords, valid_mask)
    process_and_embed_fn : callable
        Function to process inputs and get embeddings
    max_length : int
        Maximum sequence length to prevent memory explosion
    manage_memory : bool
        Whether to manage memory aggressively
    clear_memory_cache_fn : callable
        Function to clear memory cache

    Returns
    -------
    Tuple[Tensor, Tensor]
        Contact embeddings and targets, or empty tensors if no valid data
    """
    # Handle contact map prediction with tertiary structure coordinates
    tertiary_coords, valid_mask = y

    # Get per-residue embeddings
    batch_embeddings = process_and_embed_fn(pl_module, x, modality="amino_acid", aggregate=False)

    # Handle single item case (batch_size=1 due to collation)
    if batch_embeddings.dim() == 3:
        batch_embeddings = batch_embeddings.squeeze(0)  # Remove batch dimension

    # Ensure mask length matches embedding sequence length
    seq_len = batch_embeddings.shape[0]  # Should be 512 (padded length)
    orig_len = len(valid_mask)  # Original sequence length

    # Truncate or pad the mask to match embedding length
    if orig_len > seq_len:
        # Truncate mask if original sequence is longer than max_length
        residue_mask = valid_mask[:seq_len].bool()
        tertiary_coords = tertiary_coords[:seq_len]
    elif orig_len < seq_len:
        # Pad mask with False for padded positions
        padded_mask = torch.zeros(seq_len, dtype=torch.bool)
        padded_mask[:orig_len] = valid_mask.bool()
        residue_mask = padded_mask

        # Pad tertiary coords with zeros
        padded_coords = torch.zeros((seq_len, tertiary_coords.shape[1]), dtype=tertiary_coords.dtype)
        padded_coords[:orig_len] = tertiary_coords
        tertiary_coords = padded_coords
    else:
        residue_mask = valid_mask.bool()

    # Only use valid (non-padded) positions for contact prediction
    true_residue_mask = residue_mask[:orig_len]  # Only original sequence positions

    if true_residue_mask.sum() == 0:
        return torch.tensor([]), torch.tensor([])

    valid_embeddings = batch_embeddings[:orig_len][true_residue_mask]
    valid_coords = tertiary_coords[:orig_len][true_residue_mask]

    # Limit sequence length for PROTEINNET to prevent memory explosion
    n_residues = valid_coords.shape[0]
    if n_residues > max_length:
        # Truncate to prevent quadratic memory explosion
        valid_embeddings = valid_embeddings[:max_length]
        valid_coords = valid_coords[:max_length]
        n_residues = max_length
        logger.debug(f"Truncated PROTEINNET sequence to {n_residues} residues to prevent OOM")

    # Calculate pairwise distances between 3D coordinates
    distances = torch.cdist(valid_coords, valid_coords)

    # Define contacts as residues closer than 8 Angstroms
    contacts = (distances < 8.0).float()

    # Process residue pairs in chunks to avoid memory accumulation
    chunk_size = 5000  # Process pairs in chunks of 5000
    contact_embeddings = []
    contact_targets = []

    pair_count = 0
    current_chunk_embeddings = []
    current_chunk_targets = []

    # For each residue pair, concatenate their embeddings
    for i in range(n_residues):
        for j in range(i + 4, n_residues):  # Skip local contacts (i to i+3)
            current_chunk_embeddings.append(torch.cat([valid_embeddings[i], valid_embeddings[j]]))
            current_chunk_targets.append(contacts[i, j])
            pair_count += 1

            # Process chunk when it's full
            if len(current_chunk_embeddings) >= chunk_size:
                if current_chunk_embeddings:
                    chunk_embeddings = torch.stack(current_chunk_embeddings)
                    chunk_targets = torch.tensor(current_chunk_targets)

                    contact_embeddings.append(chunk_embeddings.cpu())
                    contact_targets.append(chunk_targets.cpu())

                    # Clear current chunk and force memory cleanup
                    current_chunk_embeddings.clear()
                    current_chunk_targets.clear()
                    del chunk_embeddings, chunk_targets

                    # Clear memory cache for PROTEINNET
                    if manage_memory and clear_memory_cache_fn:
                        clear_memory_cache_fn()

    # Process remaining pairs in the last chunk
    if current_chunk_embeddings:
        chunk_embeddings = torch.stack(current_chunk_embeddings)
        chunk_targets = torch.tensor(current_chunk_targets)

        contact_embeddings.append(chunk_embeddings.cpu())
        contact_targets.append(chunk_targets.cpu())

        # Clear memory
        del chunk_embeddings, chunk_targets

    # Concatenate all chunks
    if contact_embeddings:
        batch_embeddings = torch.cat(contact_embeddings, dim=0)
        batch_targets = torch.cat(contact_targets, dim=0)

        # Only return if we have valid embeddings/targets
        if batch_embeddings.numel() > 0 and batch_targets.numel() > 0:
            final_embeddings = batch_embeddings.cpu()
            final_targets = batch_targets.cpu()

            # Clear intermediate results
            del batch_embeddings, batch_targets
            contact_embeddings.clear()
            contact_targets.clear()

            logger.debug(f"Processed {pair_count} residue pairs for PROTEINNET")
            return final_embeddings, final_targets

    return torch.tensor([]), torch.tensor([])


def process_fold_item(
    pl_module: L.LightningModule,
    x,
    y,
    process_and_embed_fn=None,
) -> tuple[Tensor, Tensor]:
    """Process a single fold classification item.

    Parameters
    ----------
    pl_module : L.LightningModule
        The model to use for embedding
    x : input data
        Input sequence or tokenized input
    y : target data
        Target fold class
    process_and_embed_fn : callable
        Function to process inputs and get embeddings

    Returns
    -------
    Tuple[Tensor, Tensor]
        Sequence embeddings and targets, or empty tensors if no valid data
    """
    # Standard fold classification - sequence-level task
    batch_embeddings = process_and_embed_fn(pl_module, x, modality="amino_acid", aggregate=True)

    # Ensure target has at least one dimension for concatenation
    target = y.cpu()
    if target.dim() == 0:
        target = target.unsqueeze(0)

    # Only return if we have valid embeddings/targets
    if batch_embeddings.numel() > 0 and target.numel() > 0:
        return batch_embeddings.cpu(), target
    else:
        return torch.tensor([]), torch.tensor([])


def aggregate_task_metrics(split_metrics: dict, relevant_metric: str) -> dict:
    """Aggregate metrics across multiple splits for a task.

    Parameters
    ----------
    split_metrics : dict
        Dictionary mapping split names to their metrics
    relevant_metric : str
        The main metric to aggregate

    Returns
    -------
    dict
        Dictionary with averaged metrics if multiple splits exist
    """
    if len(split_metrics) <= 1:
        return {}

    # Only average the relevant metric
    if any(relevant_metric in m for m in split_metrics.values()):
        values = [m.get(relevant_metric) for m in split_metrics.values() if relevant_metric in m]
        if values:
            # Convert to Python type to avoid NumPy scalars in output
            avg_value = sum(values) / len(values)
            return {relevant_metric: float(avg_value) if hasattr(avg_value, "item") else avg_value}

    return {}


def calculate_category_averages(category_metrics: dict) -> dict:
    """Calculate averages for each category.

    Parameters
    ----------
    category_metrics : dict
        Dictionary mapping categories to their metric collections

    Returns
    -------
    dict
        Dictionary with category averages
    """
    category_averages = {}
    for category, metrics_dict in category_metrics.items():
        category_averages[str(category)] = {}
        for metric_name, values in metrics_dict.items():
            if values:
                avg_value = sum(values) / len(values)
                # Convert to Python type to avoid NumPy scalars in output
                category_averages[str(category)][metric_name] = (
                    float(avg_value) if hasattr(avg_value, "item") else avg_value
                )
    return category_averages


def calculate_mean_metrics(all_metrics: dict) -> dict:
    """Calculate mean metrics across all tasks.

    Parameters
    ----------
    all_metrics : dict
        Dictionary mapping metric names to lists of values

    Returns
    -------
    dict
        Dictionary with mean values for each metric
    """
    mean_metrics = {}
    for metric_name, values in all_metrics.items():
        if values:
            avg_value = sum(values) / len(values)
            # Convert to Python type to avoid NumPy scalars in output
            mean_metrics[metric_name] = float(avg_value) if hasattr(avg_value, "item") else avg_value
    return mean_metrics
