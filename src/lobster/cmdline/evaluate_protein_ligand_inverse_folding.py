"""Standalone evaluation of inverse folding on protein-ligand complexes.

Evaluates whether ligand context improves inverse folding performance,
particularly for binding pocket residues.

Usage:
    uv run python -m lobster.cmdline.evaluate_protein_ligand_inverse_folding \
        --checkpoint path/to/model.ckpt \
        --data_dir /data2/lisanzas/pdb_bind_12_15_25/test/ \
        --output results.csv \
        --pocket_threshold 5.0 \
        --num_samples 100

Example (full test set):
    uv run python -m lobster.cmdline.evaluate_protein_ligand_inverse_folding \
        --checkpoint /data2/ume/gen_ume_protein_ligand/best.ckpt \
        --data_dir /data2/lisanzas/pdb_bind_12_15_25/test/ \
        --output protein_ligand_inverse_folding_results.csv \
        --num_samples -1
"""

import argparse
import sys
from pathlib import Path

import torch
from loguru import logger

from lobster.metrics.protein_ligand_inverse_folding import ProteinLigandInverseFoldingEvaluator
from lobster.model.gen_ume import ProteinLigandEncoderLightningModule


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate inverse folding on protein-ligand complexes with/without ligand context"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.ckpt file)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/data2/lisanzas/pdb_bind_12_15_25/test/",
        help="Path to protein-ligand test directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="protein_ligand_inverse_folding_results.csv",
        help="Output CSV file for results",
    )
    parser.add_argument(
        "--pocket_threshold",
        type=float,
        default=5.0,
        help="Distance threshold (Å) for defining binding pocket",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples to evaluate (-1 for all)",
    )
    parser.add_argument(
        "--nsteps",
        type=int,
        default=100,
        help="Number of diffusion steps for generation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for computation (cuda/cpu)",
    )

    args = parser.parse_args()

    # Validate checkpoint exists
    if not Path(args.checkpoint).exists():
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    # Validate data directory exists
    if not Path(args.data_dir).exists():
        logger.error(f"Data directory not found: {args.data_dir}")
        sys.exit(1)

    # Set device
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    try:
        model = ProteinLigandEncoderLightningModule.load_from_checkpoint(
            args.checkpoint,
            map_location=args.device,
        )
        model.eval()
        model.to(args.device)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)

    # Create evaluator
    num_samples = None if args.num_samples == -1 else args.num_samples

    # Get max_length from model if available
    max_length = 512  # default
    if hasattr(model, "encoder") and hasattr(model.encoder, "neobert"):
        if hasattr(model.encoder.neobert, "config") and hasattr(model.encoder.neobert.config, "max_length"):
            max_length = model.encoder.neobert.config.max_length
            logger.info(f"Using model's max_length: {max_length}")

    evaluator = ProteinLigandInverseFoldingEvaluator(
        data_dir=args.data_dir,
        pocket_distance_threshold=args.pocket_threshold,
        num_samples=num_samples,
        nsteps=args.nsteps,
        device=args.device,
        max_length=max_length,
    )

    # Load samples
    logger.info("Loading test samples...")
    samples = evaluator.load_test_set()
    logger.info(f"Loaded {len(samples)} samples")

    # Run evaluation
    logger.info("Running evaluation...")
    results = evaluator.evaluate(model, samples)

    # Save results
    results["results_df"].to_csv(args.output, index=False)
    logger.info(f"Results saved to {args.output}")

    # Print summary
    summary = results["summary"]
    print("\n" + "=" * 70)
    print("Protein-Ligand Inverse Folding Evaluation Results")
    print("=" * 70)

    print(f"\nSamples evaluated: {summary['n_samples']}")
    print(f"Average pocket size: {summary['mean_pocket_size']:.1f} residues")
    print(f"Pocket distance threshold: {args.pocket_threshold} Å")

    print("\n--- Overall Amino Acid Recovery ---")
    print(f"  Without ligand: {summary['mean_aar_overall_no_ligand']:.2%}")
    print(f"  With ligand:    {summary['mean_aar_overall_with_ligand']:.2%}")
    print(f"  Delta:          {summary['mean_aar_overall_delta']:+.2%}")

    print("\n--- Binding Pocket Amino Acid Recovery ---")
    print(f"  Without ligand: {summary['mean_aar_pocket_no_ligand']:.2%}")
    print(f"  With ligand:    {summary['mean_aar_pocket_with_ligand']:.2%}")
    print(f"  Delta:          {summary['mean_aar_pocket_delta']:+.2%} (±{summary['std_aar_pocket_delta']:.2%})")

    print("\n--- Non-Pocket Amino Acid Recovery ---")
    print(f"  Without ligand: {summary['mean_aar_nonpocket_no_ligand']:.2%}")
    print(f"  With ligand:    {summary['mean_aar_nonpocket_with_ligand']:.2%}")
    print(f"  Delta:          {summary['mean_aar_nonpocket_delta']:+.2%} (±{summary['std_aar_nonpocket_delta']:.2%})")

    print("\n" + "=" * 70)

    # Key insight
    pocket_delta = summary["mean_aar_pocket_delta"]
    if pocket_delta > 0.01:
        print(f"🎯 Ligand context IMPROVES pocket recovery by {pocket_delta * 100:.1f}%!")
    elif pocket_delta < -0.01:
        print(f"⚠️  Ligand context DECREASES pocket recovery by {abs(pocket_delta) * 100:.1f}%")
    else:
        print("📊 Ligand context has minimal effect on pocket recovery")

    print("=" * 70)


if __name__ == "__main__":
    main()
