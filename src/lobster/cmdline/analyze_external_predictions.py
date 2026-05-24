#!/usr/bin/env python3
"""
Analyze external predicted structures against ground truth.

Compares predicted PDB structures from an external directory to ground truth
structures, using the same metrics (TM-score, RMSD) as the lobster generation
pipeline.
"""

import torch
import pandas as pd
from pathlib import Path
from loguru import logger
import argparse
from tmtools import tm_align
import numpy as np

from lobster.metrics._generation_utils import align_and_compute_rmsd
from lobster.model.latent_generator.io import load_pdb


def load_pdb_structure(pdb_path: Path) -> tuple[np.ndarray, str]:
    """
    Load structure from PDB file using lobster's load_pdb function.

    Args:
        pdb_path: Path to PDB file

    Returns:
        coords: Coordinates array of shape (L, 3, 3) for N, CA, C atoms
        sequence: Amino acid sequence string
    """
    try:
        # Use lobster's load_pdb function
        # Returns a dictionary with keys: 'sequence', 'sequence_str', 'coords_res', 'mask', etc.
        structure_data = load_pdb(str(pdb_path), add_batch_dim=False)

        if structure_data is None:
            logger.error(f"load_pdb returned None for {pdb_path}")
            return None, None

        # Extract sequence string (already in 1-letter code)
        seq_str = structure_data["sequence_str"]

        # Extract coordinates (N, 3, 3) - backbone atoms [N, CA, C]
        coords = structure_data["coords_res"]

        # Convert coords to numpy if it's a tensor
        if isinstance(coords, torch.Tensor):
            coords = coords.numpy()

        return coords, seq_str

    except Exception as e:
        logger.error(f"Error loading PDB {pdb_path}: {e}")
        return None, None


def load_pt_structure(pt_path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load structure from .pt file.

    Args:
        pt_path: Path to .pt file

    Returns:
        coords: Coordinates tensor of shape (L, 3, 3) for N, CA, C atoms
        sequence: Sequence tensor of shape (L,)
        mask: Mask tensor of shape (L,)
    """
    try:
        data = torch.load(pt_path, map_location="cpu")

        # Extract coordinates (assuming backbone atoms in order N, CA, C)
        if "coords_res" in data:
            coords = data["coords_res"]  # Should be (L, 3, 3)
        elif "bb_positions" in data:
            # If only CA positions, need to handle differently
            logger.warning(f"Only CA positions available in {pt_path}")
            coords = data["bb_positions"].unsqueeze(1).repeat(1, 3, 1)
        else:
            logger.error(f"No coordinate data found in {pt_path}")
            return None, None, None

        # Extract sequence - try both 'sequence' and 'seq' keys
        if "sequence" in data:
            sequence = data["sequence"]
        elif "seq" in data:
            sequence = data["seq"]
        else:
            logger.error(f"No sequence data found in {pt_path}. Available keys: {list(data.keys())}")
            return None, None, None

        # Extract mask
        if "mask" in data:
            mask = data["mask"]
        else:
            mask = torch.ones(sequence.shape[0])

        return coords, sequence, mask

    except Exception as e:
        logger.error(f"Error loading .pt file {pt_path}: {e}")
        return None, None, None


def extract_structure_id(filename: str) -> str:
    """
    Extract structure ID from filename.

    Args:
        filename: Filename (e.g., "5S9R_pred.pdb" or "5S9R.pt")

    Returns:
        Structure ID (e.g., "5S9R")
    """
    # Remove extension
    name = Path(filename).stem

    # Remove common suffixes
    for suffix in ["_pred", "_predicted", "_folded", "_structure"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]

    return name


def compare_sequences(seq1: str, seq2: str, struct_id: str) -> tuple[bool, float]:
    """
    Compare two sequences and return whether they match.

    Args:
        seq1: First sequence
        seq2: Second sequence
        struct_id: Structure ID for logging

    Returns:
        (match, identity): Whether sequences match exactly, and percent identity
    """
    if len(seq1) != len(seq2):
        logger.warning(f"{struct_id}: Sequence length mismatch - predicted={len(seq1)}, ground_truth={len(seq2)}")
        return False, 0.0

    # Calculate percent identity
    matches = sum(1 for a, b in zip(seq1, seq2) if a == b)
    percent_identity = (matches / len(seq1)) * 100.0

    if percent_identity < 100.0:
        logger.warning(
            f"{struct_id}: Sequence mismatch - {percent_identity:.1f}% identity ({matches}/{len(seq1)} residues match)"
        )
        # Show first difference
        for i, (a, b) in enumerate(zip(seq1, seq2)):
            if a != b:
                context_start = max(0, i - 5)
                context_end = min(len(seq1), i + 6)
                logger.warning(
                    f"  First difference at position {i}: "
                    f"predicted='{seq1[context_start:context_end]}' "
                    f"ground_truth='{seq2[context_start:context_end]}'"
                )
                break

    return percent_identity == 100.0, percent_identity


def calculate_metrics(
    pred_coords: torch.Tensor,
    gt_coords: torch.Tensor,
    gt_sequence: str,
    mask: torch.Tensor | None = None,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """
    Calculate TM-score and RMSD between predicted and ground truth structures.

    Args:
        pred_coords: Predicted coordinates, shape (L, 3, 3)
        gt_coords: Ground truth coordinates, shape (L, 3, 3)
        gt_sequence: Ground truth sequence string
        mask: Optional mask, shape (L,)
        device: torch device

    Returns:
        Dictionary with 'tm_score' and 'rmsd' keys
    """
    # Ensure tensors are on the correct device
    pred_coords = pred_coords.to(device)
    gt_coords = gt_coords.to(device)

    if mask is not None:
        mask = mask.to(device)
        # Apply mask
        pred_coords_masked = pred_coords[mask.bool()]
        gt_coords_masked = gt_coords[mask.bool()]
        # Filter sequence
        gt_sequence_masked = "".join([gt_sequence[i] for i in range(len(gt_sequence)) if mask[i] == 1])
    else:
        pred_coords_masked = pred_coords
        gt_coords_masked = gt_coords
        gt_sequence_masked = gt_sequence

    # Calculate TM-score using tm_align
    try:
        tm_out = tm_align(
            pred_coords_masked[:, 1, :].cpu().numpy(),  # CA atoms
            gt_coords_masked[:, 1, :].cpu().numpy(),  # CA atoms
            gt_sequence_masked,
            gt_sequence_masked,
        )
        tm_score = tm_out.tm_norm_chain1
    except Exception as e:
        logger.error(f"Error calculating TM-score: {e}")
        tm_score = 0.0

    # Calculate RMSD using Kabsch alignment
    try:
        rmsd = align_and_compute_rmsd(
            coords1=pred_coords_masked,
            coords2=gt_coords_masked,
            mask=None,  # Already masked
            return_aligned=False,
            device=device,
        )
    except Exception as e:
        logger.error(f"Error calculating RMSD: {e}")
        rmsd = 0.0

    return {
        "tm_score": float(tm_score),
        "rmsd": float(rmsd),
    }


def analyze_predictions(
    pred_dir: str,
    gt_dir: str,
    output_csv: str = None,
    device_str: str = "cpu",
    rmsd_threshold: float = 2.0,
    skip_sequence_mismatch: bool = False,
):
    """
    Analyze predicted structures against ground truth.

    Args:
        pred_dir: Directory containing predicted PDB files
        gt_dir: Directory containing ground truth .pt files
        output_csv: Optional path to save results CSV
        device_str: Device to use ('cpu' or 'cuda')
        rmsd_threshold: RMSD threshold for reporting pass rate
        skip_sequence_mismatch: If True, skip structures with sequence mismatches
    """
    pred_path = Path(pred_dir)
    gt_path = Path(gt_dir)

    # Set up device
    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    logger.info(f"Using device: {device}")

    # Find all predicted PDB files
    pred_files = sorted(list(pred_path.glob("*.pdb")))
    logger.info(f"Found {len(pred_files)} predicted PDB files in {pred_dir}")

    if len(pred_files) == 0:
        logger.error("No PDB files found in prediction directory")
        return

    # Build mapping from structure IDs to files
    pred_map = {}
    for pdb_file in pred_files:
        struct_id = extract_structure_id(pdb_file.name)
        pred_map[struct_id] = pdb_file

    # Find matching ground truth files
    results = []
    matched_count = 0
    missing_gt = []
    sequence_mismatches = []

    for struct_id, pred_file in pred_map.items():
        # Try to find matching .pt file
        gt_file = gt_path / f"{struct_id}.pt"

        if not gt_file.exists():
            # Try with different extensions
            possible_gt = list(gt_path.glob(f"{struct_id}*.pt"))
            if possible_gt:
                gt_file = possible_gt[0]
            else:
                logger.warning(f"No ground truth found for {struct_id}")
                missing_gt.append(struct_id)
                continue

        # Load predicted structure
        logger.info(f"Processing {struct_id}...")
        pred_coords, pred_seq = load_pdb_structure(pred_file)

        if pred_coords is None:
            logger.error(f"Failed to load predicted structure: {pred_file}")
            continue

        # Convert to tensor
        pred_coords = torch.from_numpy(pred_coords).float()

        # Load ground truth structure
        gt_coords, gt_seq, gt_mask = load_pt_structure(gt_file)

        if gt_coords is None:
            logger.error(f"Failed to load ground truth structure: {gt_file}")
            continue

        # Check length match
        if pred_coords.shape[0] != gt_coords.shape[0]:
            logger.warning(
                f"Length mismatch for {struct_id}: predicted={pred_coords.shape[0]}, ground_truth={gt_coords.shape[0]}"
            )
            # Try to truncate to shorter length
            min_len = min(pred_coords.shape[0], gt_coords.shape[0])
            pred_coords = pred_coords[:min_len]
            gt_coords = gt_coords[:min_len]
            if gt_mask is not None:
                gt_mask = gt_mask[:min_len]

        # Convert sequence tensor to string if needed
        if isinstance(gt_seq, torch.Tensor):
            # Assuming aatype indices (0-19 standard AAs, 20 = X)
            restypes = [
                "A",
                "R",
                "N",
                "D",
                "C",
                "Q",
                "E",
                "G",
                "H",
                "I",
                "L",
                "K",
                "M",
                "F",
                "P",
                "S",
                "T",
                "W",
                "Y",
                "V",
                "X",
            ]
            gt_seq_str = "".join([restypes[i] if i < len(restypes) else "X" for i in gt_seq])
        else:
            gt_seq_str = gt_seq

        # If pred_seq is already 1-letter codes, use as is
        # Otherwise try to extract from PDB
        if len(pred_seq) == pred_coords.shape[0] and all(c in "ACDEFGHIKLMNPQRSTVWYX" for c in pred_seq):
            pred_seq_str = pred_seq
        else:
            # Fallback: use ground truth sequence length
            pred_seq_str = gt_seq_str[: pred_coords.shape[0]]

        # Check sequence match
        seq_match, seq_identity = compare_sequences(pred_seq_str, gt_seq_str, struct_id)

        if not seq_match:
            sequence_mismatches.append((struct_id, seq_identity))
            if skip_sequence_mismatch:
                logger.warning(f"  Skipping {struct_id} due to sequence mismatch")
                continue

        # Calculate metrics
        metrics = calculate_metrics(
            pred_coords=pred_coords,
            gt_coords=gt_coords,
            gt_sequence=gt_seq_str,
            mask=gt_mask,
            device=device,
        )

        # Store results
        results.append(
            {
                "Structure_ID": struct_id,
                "Length": pred_coords.shape[0],
                "Seq_Identity": seq_identity,
                "TM_Score": metrics["tm_score"],
                "RMSD": metrics["rmsd"],
                "Pred_File": pred_file.name,
                "GT_File": gt_file.name,
            }
        )

        matched_count += 1

        # Log individual result
        logger.info(f"  {struct_id}: TM-score={metrics['tm_score']:.4f}, RMSD={metrics['rmsd']:.4f} Å")

    # Create DataFrame
    if not results:
        logger.error("No structures were successfully analyzed")
        return

    df = pd.DataFrame(results)

    # Sort by TM-score (descending)
    df = df.sort_values("TM_Score", ascending=False)

    # Calculate summary statistics
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY STATISTICS")
    logger.info("=" * 80)
    logger.info(f"Total structures analyzed: {len(df)}")
    logger.info(f"Structures with ground truth: {matched_count}/{len(pred_map)}")

    if missing_gt:
        logger.info(
            f"Missing ground truth for: {', '.join(missing_gt[:10])}"
            + (f" ... and {len(missing_gt) - 10} more" if len(missing_gt) > 10 else "")
        )

    if sequence_mismatches:
        logger.warning(f"\nSequence mismatches found: {len(sequence_mismatches)}")
        # Show worst mismatches
        worst_mismatches = sorted(sequence_mismatches, key=lambda x: x[1])[:5]
        for struct_id, identity in worst_mismatches:
            logger.warning(f"  {struct_id}: {identity:.1f}% identity")

    # Report sequence identity statistics
    logger.info("\nSequence Identity:")
    logger.info(f"  Mean: {df['Seq_Identity'].mean():.2f}%")
    logger.info(f"  Min:  {df['Seq_Identity'].min():.2f}%")
    logger.info(f"  Max:  {df['Seq_Identity'].max():.2f}%")
    exact_matches = len(df[df["Seq_Identity"] == 100.0])
    logger.info(f"  Exact matches: {exact_matches}/{len(df)} ({exact_matches / len(df) * 100:.1f}%)")

    logger.info("\nTM-Score:")
    logger.info(f"  Mean: {df['TM_Score'].mean():.4f}")
    logger.info(f"  Std:  {df['TM_Score'].std():.4f}")
    logger.info(f"  Min:  {df['TM_Score'].min():.4f}")
    logger.info(f"  Max:  {df['TM_Score'].max():.4f}")
    logger.info(f"  Median: {df['TM_Score'].median():.4f}")

    logger.info("\nRMSD:")
    logger.info(f"  Mean: {df['RMSD'].mean():.4f} Å")
    logger.info(f"  Std:  {df['RMSD'].std():.4f} Å")
    logger.info(f"  Min:  {df['RMSD'].min():.4f} Å")
    logger.info(f"  Max:  {df['RMSD'].max():.4f} Å")
    logger.info(f"  Median: {df['RMSD'].median():.4f} Å")

    # Calculate pass rate
    passing = len(df[df["RMSD"] < rmsd_threshold])
    pass_rate = (passing / len(df)) * 100
    logger.info(f"\nStructures with RMSD < {rmsd_threshold} Å: {passing}/{len(df)} ({pass_rate:.1f}%)")

    # Save to CSV
    if output_csv:
        output_path = Path(output_csv)
        df.to_csv(output_path, index=False)
        logger.info(f"\n✓ Results saved to: {output_path}")
    else:
        # Save to default location
        output_path = Path("external_predictions_analysis.csv")
        df.to_csv(output_path, index=False)
        logger.info(f"\n✓ Results saved to: {output_path}")

    # Create aggregate summary table (similar to forward folding summary)
    logger.info("\n" + "=" * 80)
    logger.info("AGGREGATE SUMMARY")
    logger.info("=" * 80)

    summary_table = pd.DataFrame(
        [
            {
                "Total_Structures": len(df),
                "Avg_TM_Score": round(df["TM_Score"].mean(), 4),
                "Std_TM_Score": round(df["TM_Score"].std(), 4),
                "Min_TM_Score": round(df["TM_Score"].min(), 4),
                "Max_TM_Score": round(df["TM_Score"].max(), 4),
                "Avg_RMSD": round(df["RMSD"].mean(), 4),
                "Std_RMSD": round(df["RMSD"].std(), 4),
                "Min_RMSD": round(df["RMSD"].min(), 4),
                "Max_RMSD": round(df["RMSD"].max(), 4),
                f"Structures_RMSD<{rmsd_threshold}": passing,
                f"Pct_RMSD<{rmsd_threshold}": round(pass_rate, 2),
            }
        ]
    )

    logger.info(f"\n{summary_table.to_string(index=False)}")

    # Save summary table
    summary_csv = output_path.parent / f"{output_path.stem}_summary.csv"
    summary_table.to_csv(summary_csv, index=False)
    logger.info(f"\n✓ Summary saved to: {summary_csv}")

    # Display top and bottom performers
    logger.info("\n" + "=" * 80)
    logger.info("TOP 10 STRUCTURES (by TM-score)")
    logger.info("=" * 80)
    logger.info(f"\n{df[['Structure_ID', 'Seq_Identity', 'TM_Score', 'RMSD']].head(10).to_string(index=False)}")

    logger.info("\n" + "=" * 80)
    logger.info("BOTTOM 10 STRUCTURES (by TM-score)")
    logger.info("=" * 80)
    logger.info(f"\n{df[['Structure_ID', 'Seq_Identity', 'TM_Score', 'RMSD']].tail(10).to_string(index=False)}")

    logger.info("\n" + "=" * 80)
    logger.info("Analysis complete!")
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze external predicted structures against ground truth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python analyze_external_predictions.py \\
      --pred-dir /path/to/predicted/pdbs \\
      --gt-dir /path/to/ground_truth/pt_files \\
      --output analysis_results.csv \\
      --device cuda

  # For DPLM2 predictions:
  python analyze_external_predictions.py \
      --pred-dir /homefs/home/lisanzas/scratch/Develop/dplm/generation-results/dplm2_650m/folding/pdb/ \
      --gt-dir /data2/lisanzas/multi_flow_data/test_set_filtered_pt/ \
      --output dplm2_folding_analysis.csv
        """,
    )

    parser.add_argument("--pred-dir", type=str, required=True, help="Directory containing predicted PDB files")

    parser.add_argument("--gt-dir", type=str, required=True, help="Directory containing ground truth .pt files")

    parser.add_argument(
        "--output",
        type=str,
        default="external_predictions_analysis.csv",
        help="Output CSV file path (default: external_predictions_analysis.csv)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for computation (default: cpu)",
    )

    parser.add_argument(
        "--rmsd-threshold", type=float, default=2.0, help="RMSD threshold for pass rate calculation (default: 2.0 Å)"
    )

    parser.add_argument(
        "--skip-sequence-mismatch",
        action="store_true",
        help="Skip structures with sequence mismatches (default: analyze anyway with warning)",
    )

    args = parser.parse_args()

    analyze_predictions(
        pred_dir=args.pred_dir,
        gt_dir=args.gt_dir,
        output_csv=args.output,
        device_str=args.device,
        rmsd_threshold=args.rmsd_threshold,
        skip_sequence_mismatch=args.skip_sequence_mismatch,
    )


if __name__ == "__main__":
    main()
