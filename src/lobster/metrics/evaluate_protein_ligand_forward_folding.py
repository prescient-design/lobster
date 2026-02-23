#!/usr/bin/env python
"""Standalone evaluation script for protein-ligand forward folding (structure prediction).

Evaluates structure prediction (TM-score, RMSD) on protein-ligand complexes.
Compares forward folding with and without ligand context.

Usage:
    # Evaluate a Gen-UME protein-ligand checkpoint
    uv run python -m lobster.metrics.evaluate_protein_ligand_forward_folding \
        --checkpoint /path/to/checkpoint.ckpt \
        --data_dir /path/to/pdbind/test/ \
        --output results.csv

    # With structure saving
    uv run python -m lobster.metrics.evaluate_protein_ligand_forward_folding \
        --checkpoint /path/to/checkpoint.ckpt \
        --data_dir /path/to/pdbind/test/ \
        --output results.csv \
        --structure_path ./structures/ \
        --save_structures \
        --save_gt_structure

    # Customize pocket threshold and number of samples
    uv run python -m lobster.metrics.evaluate_protein_ligand_forward_folding \
        --checkpoint /path/to/checkpoint.ckpt \
        --data_dir /path/to/pdbind/test/ \
        --output results.csv \
        --pocket_threshold 6.0 \
        --num_samples 500
"""

import argparse
import json
import os
import sys

import torch
from loguru import logger

from lobster.metrics.protein_ligand_forward_folding import ProteinLigandForwardFoldingEvaluator


def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load a Gen-UME protein-ligand model from checkpoint.

    Parameters
    ----------
    checkpoint_path : str
        Path to the model checkpoint (.ckpt file)
    device : str
        Device to load model on

    Returns
    -------
    model : LightningModule
        The loaded model
    """
    from lobster.model.gen_ume import ProteinLigandEncoderLightningModule

    logger.info(f"Loading model from {checkpoint_path}")

    # Load checkpoint
    model = ProteinLigandEncoderLightningModule.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
        strict=False,
    )
    model.eval()
    model.to(device)

    # Get max_length from encoder config
    max_length = None
    if hasattr(model, "encoder") and hasattr(model.encoder, "neobert"):
        if hasattr(model.encoder.neobert, "config") and hasattr(model.encoder.neobert.config, "max_length"):
            max_length = model.encoder.neobert.config.max_length
    model.max_length = max_length  # Store for later use

    logger.info(f"Model loaded successfully. Max length: {max_length}")
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate protein-ligand forward folding (structure prediction quality)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.ckpt file)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/cv/data/ai4dd/data2/lisanzas/pdb_bind_12_15_25/test/",
        help="Path to PDBBind test directory with *_protein.pt and *_ligand.pt pairs",
    )

    # Output options
    parser.add_argument(
        "--output",
        type=str,
        default="protein_ligand_forward_folding_results.csv",
        help="Output CSV file for per-structure results",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Output JSON file for summary statistics (optional)",
    )
    parser.add_argument(
        "--structure_path",
        type=str,
        default=None,
        help="Directory to save predicted structures (PDB)",
    )

    # Evaluation parameters
    parser.add_argument(
        "--pocket_threshold",
        type=float,
        default=5.0,
        help="Distance threshold (Å) for defining binding pocket residues",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (None = all available)",
    )
    parser.add_argument(
        "--nsteps",
        type=int,
        default=100,
        help="Number of diffusion steps for generation",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=768,
        help="Maximum combined sequence length (protein + ligand) to process",
    )

    # Temperature parameters
    parser.add_argument(
        "--temperature_seq",
        type=float,
        default=0.5,
        help="Temperature for sequence sampling",
    )
    parser.add_argument(
        "--temperature_struc",
        type=float,
        default=0.5,
        help="Temperature for structure sampling",
    )

    # Structure saving options
    parser.add_argument(
        "--save_structures",
        action="store_true",
        help="Save predicted structures as PDB files",
    )
    parser.add_argument(
        "--save_gt_structure",
        action="store_true",
        help="Save ground truth structures as PDB files",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for computation (default: cuda if available)",
    )

    args = parser.parse_args()

    # Validate inputs
    # Skip existence check for S3 paths (they're handled by the model loader)
    if not args.checkpoint.startswith("s3://") and not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    if not os.path.exists(args.data_dir):
        logger.error(f"Data directory not found: {args.data_dir}")
        sys.exit(1)

    # Load model
    model = load_model(args.checkpoint, args.device)

    # Use model's max_length if not overridden
    max_length = args.max_length
    if hasattr(model, "max_length") and model.max_length is not None:
        max_length = min(max_length, model.max_length)
        logger.info(f"Using max_length: {max_length}")

    # Create evaluator
    evaluator = ProteinLigandForwardFoldingEvaluator(
        data_dir=args.data_dir,
        pocket_distance_threshold=args.pocket_threshold,
        num_samples=args.num_samples,
        nsteps=args.nsteps,
        device=args.device,
        max_length=max_length,
        temperature_seq=args.temperature_seq,
        temperature_struc=args.temperature_struc,
        save_structures=args.save_structures,
        save_gt_structure=args.save_gt_structure,
    )

    # Load test set
    logger.info(f"Loading test set from {args.data_dir}")
    samples = evaluator.load_test_set()
    logger.info(f"Loaded {len(samples)} samples")

    # Run evaluation
    logger.info("Starting evaluation...")
    results = evaluator.evaluate(
        model=model,
        samples=samples,
        structure_path=args.structure_path,
    )

    # Save results
    results_df = results["results_df"]
    summary = results["summary"]

    # Save per-structure results to CSV
    results_df.to_csv(args.output, index=False)
    logger.info(f"Saved per-structure results to {args.output}")

    # Print summary
    print("\n" + "=" * 80)
    print("PROTEIN-LIGAND FORWARD FOLDING RESULTS")
    print("=" * 80)
    print(f"\nDataset: {args.data_dir}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Pocket threshold: {args.pocket_threshold} Å")
    print(f"Samples evaluated: {summary['n_samples']}")
    print(f"Mean pocket size: {summary['mean_pocket_size']:.1f} residues")

    print("\n--- TM-Score (higher is better) ---")
    print(f"\n{'Metric':<25} {'No Ligand':<15} {'With Ligand':<15} {'Delta':<15}")
    print("-" * 70)
    print(
        f"{'TM-Score':<25} "
        f"{summary['mean_tm_score_no_ligand']:<15.3f} "
        f"{summary['mean_tm_score_with_ligand']:<15.3f} "
        f"{summary['mean_tm_score_delta']:+.3f} ± {summary['std_tm_score_delta']:.3f}"
    )

    print("\n--- RMSD (Å, lower is better) ---")
    print(f"\n{'Region':<25} {'No Ligand':<15} {'With Ligand':<15} {'Delta':<15}")
    print("-" * 70)
    print(
        f"{'Overall':<25} "
        f"{summary['mean_rmsd_overall_no_ligand']:<15.2f} "
        f"{summary['mean_rmsd_overall_with_ligand']:<15.2f} "
        f"{summary['mean_rmsd_overall_delta']:+.2f} ± {summary['std_rmsd_overall_delta']:.2f}"
    )
    print(
        f"{'Pocket':<25} "
        f"{summary['mean_rmsd_pocket_no_ligand']:<15.2f} "
        f"{summary['mean_rmsd_pocket_with_ligand']:<15.2f} "
        f"{summary['mean_rmsd_pocket_delta']:+.2f} ± {summary['std_rmsd_pocket_delta']:.2f}"
    )
    print(
        f"{'Non-pocket':<25} "
        f"{summary['mean_rmsd_nonpocket_no_ligand']:<15.2f} "
        f"{summary['mean_rmsd_nonpocket_with_ligand']:<15.2f} "
        f"{summary['mean_rmsd_nonpocket_delta']:+.2f} ± {summary['std_rmsd_nonpocket_delta']:.2f}"
    )
    print("=" * 80)

    # Interpretation
    tm_delta = summary["mean_tm_score_delta"]
    rmsd_pocket_delta = summary["mean_rmsd_pocket_delta"]

    if tm_delta > 0.01:
        print(f"\n✓ Ligand context IMPROVES TM-score by {tm_delta:+.3f}")
    elif tm_delta < -0.01:
        print(f"\n✗ Ligand context HURTS TM-score by {tm_delta:+.3f}")
    else:
        print(f"\n○ Ligand context has minimal effect on TM-score ({tm_delta:+.3f})")

    if rmsd_pocket_delta < -0.1:
        print(f"✓ Ligand context IMPROVES pocket RMSD by {-rmsd_pocket_delta:.2f} Å")
    elif rmsd_pocket_delta > 0.1:
        print(f"✗ Ligand context HURTS pocket RMSD by {rmsd_pocket_delta:.2f} Å")
    else:
        print(f"○ Ligand context has minimal effect on pocket RMSD ({rmsd_pocket_delta:+.2f} Å)")

    # Save summary to JSON if requested
    if args.output_json:
        # Convert numpy/torch types to Python types for JSON serialization
        summary_json = {k: float(v) if hasattr(v, "item") else v for k, v in summary.items()}
        summary_json["checkpoint"] = args.checkpoint
        summary_json["data_dir"] = args.data_dir
        summary_json["pocket_threshold"] = args.pocket_threshold
        summary_json["nsteps"] = args.nsteps
        summary_json["max_length"] = max_length
        summary_json["temperature_seq"] = args.temperature_seq
        summary_json["temperature_struc"] = args.temperature_struc

        with open(args.output_json, "w") as f:
            json.dump(summary_json, f, indent=2)
        logger.info(f"Saved summary to {args.output_json}")

    logger.info("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
