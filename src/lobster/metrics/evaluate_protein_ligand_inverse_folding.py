#!/usr/bin/env python
"""Standalone evaluation script for protein-ligand inverse folding (sequence recovery).

Evaluates sequence recovery around ligand binding pockets on protein-ligand complexes.
Compares inverse folding with and without ligand context.

Usage:
    # Evaluate a Gen-UME protein-ligand checkpoint
    uv run python -m lobster.metrics.evaluate_protein_ligand_inverse_folding \
        --checkpoint /path/to/checkpoint.ckpt \
        --data_dir /path/to/pdbind/test/ \
        --output results.csv

    # With structure decoding and ground truth saving
    uv run python -m lobster.metrics.evaluate_protein_ligand_inverse_folding \
        --checkpoint /path/to/checkpoint.ckpt \
        --data_dir /path/to/pdbind/test/ \
        --output results.csv \
        --structure_path ./structures/ \
        --decode_structure \
        --save_gt_structure

    # Customize pocket threshold and number of samples
    uv run python -m lobster.metrics.evaluate_protein_ligand_inverse_folding \
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

from lobster.metrics.protein_ligand_inverse_folding import ProteinLigandInverseFoldingEvaluator


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
        description="Evaluate protein-ligand inverse folding (sequence recovery around binding pocket)",
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
        default="protein_ligand_inverse_folding_results.csv",
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
        help="Directory to save sequences (FASTA) and decoded structures (PDB)",
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

    # Structure decoding options
    parser.add_argument(
        "--decode_structure",
        action="store_true",
        help="Decode and save predicted structures as PDB files",
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

    # ESMFold validation
    parser.add_argument(
        "--use_esmfold",
        action="store_true",
        help="Validate designed sequences with ESMFold (fold and compare to GT structure)",
    )
    parser.add_argument(
        "--max_protein_length",
        type=int,
        default=512,
        help="Maximum protein-only length. Samples exceeding this are skipped. Also used as ESMFold max length (default: 512)",
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

    # Initialize ESMFold if requested
    plm_fold = None
    if args.use_esmfold:
        from lobster.model import LobsterPLMFold

        logger.info("Loading ESMFold for structure validation...")
        plm_fold = LobsterPLMFold(model_name="esmfold_v1", max_length=512)
        plm_fold.to(args.device)
        logger.info("ESMFold loaded successfully")

    # Use model's max_length if not overridden
    max_length = args.max_length
    if hasattr(model, "max_length") and model.max_length is not None:
        max_length = min(max_length, model.max_length)
        logger.info(f"Using max_length: {max_length}")

    # Create evaluator
    evaluator = ProteinLigandInverseFoldingEvaluator(
        data_dir=args.data_dir,
        pocket_distance_threshold=args.pocket_threshold,
        num_samples=args.num_samples,
        nsteps=args.nsteps,
        device=args.device,
        max_length=max_length,
        decode_structure=args.decode_structure,
        save_gt_structure=args.save_gt_structure,
        use_esmfold=args.use_esmfold,
        plm_fold=plm_fold,
        max_protein_length=args.max_protein_length,
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
    print("\n" + "=" * 70)
    print("PROTEIN-LIGAND INVERSE FOLDING RESULTS")
    print("=" * 70)
    print(f"\nDataset: {args.data_dir}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Pocket threshold: {args.pocket_threshold} Å")
    print(f"Samples evaluated: {summary['n_samples']}")
    print(f"Mean pocket size: {summary['mean_pocket_size']:.1f} residues")

    print("\n--- Sequence Recovery (AAR) ---")
    print(f"\n{'Region':<20} {'No Ligand':<15} {'With Ligand':<15} {'Delta':<15}")
    print("-" * 65)
    print(
        f"{'Overall':<20} "
        f"{summary['mean_aar_overall_no_ligand']:<15.2%} "
        f"{summary['mean_aar_overall_with_ligand']:<15.2%} "
        f"{summary['mean_aar_overall_delta']:+.2%}"
    )
    print(
        f"{'Pocket':<20} "
        f"{summary['mean_aar_pocket_no_ligand']:<15.2%} "
        f"{summary['mean_aar_pocket_with_ligand']:<15.2%} "
        f"{summary['mean_aar_pocket_delta']:+.2%} ± {summary['std_aar_pocket_delta']:.2%}"
    )
    print(
        f"{'Non-pocket':<20} "
        f"{summary['mean_aar_nonpocket_no_ligand']:<15.2%} "
        f"{summary['mean_aar_nonpocket_with_ligand']:<15.2%} "
        f"{summary['mean_aar_nonpocket_delta']:+.2%} ± {summary['std_aar_nonpocket_delta']:.2%}"
    )
    # ESMFold validation results
    if args.use_esmfold and "mean_esmfold_tm_no_ligand" in summary:
        print("\n--- ESMFold Designability Validation ---")
        print(f"  {'Condition':<20} {'TM-score':<12} {'RMSD (Å)':<12} {'Pocket RMSD':<14} {'pLDDT':<12} {'PAE':<12}")
        print("  " + "-" * 82)
        print(
            f"  {'GT sequence':<20} "
            f"{summary['mean_esmfold_tm_gt']:<12.3f} "
            f"{summary['mean_esmfold_rmsd_gt']:<12.2f} "
            f"{summary['mean_esmfold_rmsd_pocket_gt']:<14.2f} "
            f"{summary['mean_esmfold_plddt_gt']:<12.2f} "
            f"{summary['mean_esmfold_pae_gt']:<12.2f}"
        )
        print(
            f"  {'No ligand':<20} "
            f"{summary['mean_esmfold_tm_no_ligand']:<12.3f} "
            f"{summary['mean_esmfold_rmsd_no_ligand']:<12.2f} "
            f"{summary['mean_esmfold_rmsd_pocket_no_ligand']:<14.2f} "
            f"{summary['mean_esmfold_plddt_no_ligand']:<12.2f} "
            f"{summary['mean_esmfold_pae_no_ligand']:<12.2f}"
        )
        print(
            f"  {'With ligand':<20} "
            f"{summary['mean_esmfold_tm_with_ligand']:<12.3f} "
            f"{summary['mean_esmfold_rmsd_with_ligand']:<12.2f} "
            f"{summary['mean_esmfold_rmsd_pocket_with_ligand']:<14.2f} "
            f"{summary['mean_esmfold_plddt_with_ligand']:<12.2f} "
            f"{summary['mean_esmfold_pae_with_ligand']:<12.2f}"
        )
        print(
            f"  {'Delta (ligand)':<20} "
            f"{summary['mean_esmfold_tm_delta']:+<12.3f} "
            f"{summary['mean_esmfold_rmsd_delta']:+<12.2f} "
            f"{summary['mean_esmfold_rmsd_pocket_delta']:+<14.2f} "
            f"{summary['mean_esmfold_plddt_delta']:+<12.2f}"
        )

    print("=" * 70)

    # Interpretation
    pocket_delta = summary["mean_aar_pocket_delta"]
    if pocket_delta > 0.01:
        print(f"\nLigand context IMPROVES pocket sequence recovery by {pocket_delta:+.2%}")
    elif pocket_delta < -0.01:
        print(f"\nLigand context HURTS pocket sequence recovery by {pocket_delta:+.2%}")
    else:
        print(f"\nLigand context has minimal effect on pocket sequence recovery ({pocket_delta:+.2%})")

    # Save summary to JSON if requested
    if args.output_json:
        # Convert numpy/torch types to Python types for JSON serialization
        summary_json = {k: float(v) if hasattr(v, "item") else v for k, v in summary.items()}
        summary_json["checkpoint"] = args.checkpoint
        summary_json["data_dir"] = args.data_dir
        summary_json["pocket_threshold"] = args.pocket_threshold
        summary_json["nsteps"] = args.nsteps
        summary_json["max_length"] = max_length

        with open(args.output_json, "w") as f:
            json.dump(summary_json, f, indent=2)
        logger.info(f"Saved summary to {args.output_json}")

    logger.info("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
