"""Shared helpers used by both :mod:`lobster.cmdline.generate` and the
per-mode pipelines under :mod:`lobster.cmdline.generate_modes`.

This module exists to avoid circular imports — ``generate.py`` imports each
``_generate_<mode>`` function, and the mode files in turn need utilities like
the inference-schedule lookup, sequence-token sanity check, and ESMFold
validation pass. Keeping these helpers in their own leaf module breaks the cycle.
"""

from __future__ import annotations

from pathlib import Path

import torch
from loguru import logger

from bionemo.moco.schedules.inference_time_schedules import (
    LinearInferenceSchedule,
    LogInferenceSchedule,
    PowerInferenceSchedule,
)
from lobster.metrics import get_folded_structure_metrics
from lobster.model.latent_generator.io import writepdb
from lobster.model.latent_generator.utils.residue_constants import restype_order_with_x_inv


def _get_inference_schedule_class(schedule_name: str):
    """Convert schedule name string to schedule class.

    Args:
        schedule_name: String name of schedule ("LinearInferenceSchedule", "LogInferenceSchedule", "PowerInferenceSchedule")

    Returns:
        Schedule class (callable)
    """
    schedule_map = {
        "LinearInferenceSchedule": LinearInferenceSchedule,
        "LogInferenceSchedule": LogInferenceSchedule,
        "PowerInferenceSchedule": PowerInferenceSchedule,
    }

    if schedule_name not in schedule_map:
        raise ValueError(f"Unknown schedule name: {schedule_name}. Available options: {list(schedule_map.keys())}")

    return schedule_map[schedule_name]


def _check_sequence_tokens(
    sequences: torch.Tensor, mask: torch.Tensor, stage_name: str = "generation"
) -> tuple[bool, str]:
    """Check if generated sequences contain only valid amino acid tokens.

    Valid tokens are 0-19 (standard 20 amino acids).
    Invalid tokens include:
    - Token 20 (X = unknown amino acid)
    - Tokens > 20 (mask/special tokens)
    - Negative values

    Also checks for low-complexity sequences:
    - Sequences where one amino acid accounts for > 50% of the total

    Args:
        sequences: Sequence tensor (B, L) with amino acid token indices
        mask: Validity mask (B, L) indicating which positions are valid
        stage_name: Name of the generation stage for logging

    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if all sequences contain only valid tokens
        - error_message: Description of the issue if invalid, empty string if valid
    """
    batch_size = sequences.shape[0]

    for i in range(batch_size):
        # Get valid positions for this sample
        valid_positions = mask[i] == 1
        seq_i = sequences[i, valid_positions]

        # Check for unknown tokens (token 20 = X)
        num_unknown = (seq_i == 20).sum().item()
        if num_unknown > 0:
            return False, f"Sample {i} in {stage_name} contains {num_unknown} unknown 'X' tokens (token 20)"

        # Check for mask/special tokens (> 20)
        num_mask = (seq_i > 20).sum().item()
        if num_mask > 0:
            return False, f"Sample {i} in {stage_name} contains {num_mask} mask/special tokens (> 20)"

        # Check for negative values (should never happen, but check anyway)
        num_negative = (seq_i < 0).sum().item()
        if num_negative > 0:
            return False, f"Sample {i} in {stage_name} contains negative token values"

        # Check for low-complexity sequences (one amino acid > 50%)
        # Only check valid amino acids (0-19)
        valid_aa_mask = (seq_i >= 0) & (seq_i < 20)
        valid_aa_seq = seq_i[valid_aa_mask]

        if len(valid_aa_seq) > 0:
            # Count frequency of each amino acid
            aa_counts = torch.bincount(valid_aa_seq, minlength=20)
            max_count = aa_counts.max().item()
            max_percentage = (max_count / len(valid_aa_seq)) * 100

            if max_percentage > 50.0:
                max_aa_idx = aa_counts.argmax().item()
                return False, (
                    f"Sample {i} in {stage_name} is low-complexity: "
                    f"amino acid {max_aa_idx} accounts for {max_percentage:.1f}% "
                    f"({max_count}/{len(valid_aa_seq)}) of the sequence"
                )

    return True, ""


def _validate_with_esmfold(
    seq: torch.Tensor,
    x_recon_xyz: torch.Tensor,
    plm_fold,
    device: torch.device,
    output_dir: Path,
    prefix: str,
    original_paths: list[str] | None = None,
    mask: torch.Tensor | None = None,
    max_length: int | None = 512,
) -> dict[str, float] | None:
    """Validate generated structures using ESMFold."""
    # Convert sequences to strings
    sequence_str = []
    for i in range(seq.shape[0]):
        if mask is not None:
            # do not include the padded positions in the sequence
            seq_i = seq[i, mask[i] == 1]
            sequence_str.append("".join([restype_order_with_x_inv[j.item()] for j in seq_i]))
        else:
            sequence_str.append("".join([restype_order_with_x_inv[j.item()] for j in seq[i]]))

    # Tokenize sequences
    tokenized_input = plm_fold.tokenizer.batch_encode_plus(
        sequence_str,
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"].to(device)

    # Fold with ESMFold
    with torch.no_grad():
        outputs = plm_fold.model(tokenized_input)

    # Get folding metrics
    folded_structure_metrics, pred_coords = get_folded_structure_metrics(
        outputs, x_recon_xyz, sequence_str, mask=mask, device=device
    )

    # Log metrics
    logger.info("ESMFold validation metrics:")
    for key, value in folded_structure_metrics.items():
        # Skip internal fields that store chain group results
        if key.startswith("_all_") or key.startswith("_primary_"):
            continue
        # Format numeric values
        if isinstance(value, (int, float)):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")

    # Save folded structures
    for i in range(len(sequence_str)):
        if original_paths and i < len(original_paths):
            # Use original filename for inverse folding
            original_name = Path(original_paths[i]).stem
            filename = output_dir / f"{prefix}_{original_name}_esmfold.pdb"
        else:
            # Use generic naming for unconditional generation
            filename = output_dir / f"{prefix}_esmfold_{i:03d}.pdb"
        if mask is not None:
            pred_coords_i_masked = pred_coords[i, mask[i] == 1]
            seq_i_masked = seq[i, mask[i] == 1]
        else:
            pred_coords_i_masked = pred_coords[i]
            seq_i_masked = seq[i]
        writepdb(str(filename), pred_coords_i_masked, seq_i_masked)
        logger.info(f"Saved ESMFold structure: {filename}")

    # Return the metrics for aggregate statistics
    return folded_structure_metrics
