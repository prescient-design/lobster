"""Standalone evaluation of LigandMPNN inverse folding baseline.

Runs LigandMPNN locally (default) or via Pylon endpoint for sequence design
on protein-ligand complexes. Co-folding validation is handled separately via
SLURM batch jobs (see submit_cofold_batch.py).

Usage:
    uv run python -m lobster.cmdline.evaluation.evaluate_ligandmpnn_baseline \
        --data_dir /path/to/posebusters_benchmark_no_overlap \
        --raw_data_dir /path/to/posebusters_benchmark_set \
        --output results.csv \
        --structure_path ./output/
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from lobster.metrics.protein_ligand.baseline_ligandmpnn import (
    LigandMPNNInverseFoldingBaselineEvaluator,
)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LigandMPNN inverse folding baseline on protein-ligand complexes"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to processed test data directory with *_protein.pt and *_ligand.pt pairs",
    )
    parser.add_argument(
        "--raw_data_dir",
        type=str,
        required=True,
        help="Path to raw benchmark data with SDF files for SMILES extraction",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ligandmpnn_baseline_results.csv",
        help="Output CSV path for per-sample results",
    )
    parser.add_argument(
        "--structure_path",
        type=str,
        default=None,
        help="Directory to save output structures",
    )
    parser.add_argument(
        "--pocket_threshold",
        type=float,
        default=5.0,
        help="Distance threshold (angstrom) for binding pocket definition",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=-1,
        help="Number of samples to evaluate (-1 = all)",
    )
    parser.add_argument(
        "--num_designs",
        type=int,
        default=10,
        help="Number of LigandMPNN designs per structure",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="LigandMPNN sampling temperature",
    )
    parser.add_argument(
        "--use_local_ligandmpnn",
        action="store_true",
        default=True,
        help="Run LigandMPNN locally via subprocess (default: True)",
    )
    parser.add_argument(
        "--use_pylon_ligandmpnn",
        action="store_true",
        default=False,
        help="Run LigandMPNN via Pylon endpoint instead of locally",
    )
    parser.add_argument(
        "--ligandmpnn_path",
        type=str,
        default="/cv/home/lisanzas/LigandMPNN",
        help="Path to local LigandMPNN repo (used with --use_local_ligandmpnn)",
    )
    parser.add_argument(
        "--max_protein_length",
        type=int,
        default=512,
        help="Maximum protein length to evaluate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    logger.info(f"Setting random seed: {args.seed}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not Path(args.data_dir).exists():
        logger.error(f"Data directory not found: {args.data_dir}")
        sys.exit(1)

    if not Path(args.raw_data_dir).exists():
        logger.error(f"Raw data directory not found: {args.raw_data_dir}")
        sys.exit(1)

    if args.structure_path:
        os.makedirs(args.structure_path, exist_ok=True)

    num_samples = None if args.num_samples == -1 else args.num_samples

    use_local = args.use_local_ligandmpnn and not args.use_pylon_ligandmpnn

    evaluator = LigandMPNNInverseFoldingBaselineEvaluator(
        data_dir=args.data_dir,
        raw_data_dir=args.raw_data_dir,
        pocket_distance_threshold=args.pocket_threshold,
        num_samples=num_samples,
        num_designs=args.num_designs,
        temperature=args.temperature,
        device="cpu",
        use_local_ligandmpnn=use_local,
        ligandmpnn_path=args.ligandmpnn_path,
        max_protein_length=args.max_protein_length,
    )

    logger.info(f"LigandMPNN mode: {'local' if use_local else 'Pylon'}")

    logger.info("Loading test samples...")
    samples = evaluator.load_test_set()
    logger.info(f"Loaded {len(samples)} samples")

    logger.info("Running LigandMPNN baseline evaluation...")
    results = evaluator.evaluate(samples=samples, structure_path=args.structure_path)

    results_df = results["results_df"]
    summary = results["summary"]

    output_path = args.output
    if args.structure_path:
        output_path = os.path.join(args.structure_path, os.path.basename(args.output))

    results_df.to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path}")

    summary_path = output_path.replace(".csv", "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(
            {k: float(v) if isinstance(v, (float, int, np.floating, np.integer)) else v for k, v in summary.items()},
            f,
            indent=2,
        )
    logger.info(f"Summary saved to {summary_path}")

    print("\n" + "=" * 70)
    print("LigandMPNN Inverse Folding Baseline Results")
    print("=" * 70)
    print(f"Samples evaluated: {summary['n_samples']}")

    if summary["n_samples"] > 0:
        print("\n--- Amino Acid Recovery ---")
        print(f"  Overall AAR: {summary['mean_aar_overall']:.2%}")
        print(f"  Pocket AAR:  {summary['mean_aar_pocket']:.2%}")
        print(f"  Mean pocket size: {summary['mean_pocket_size']:.1f} residues")

    print("=" * 70)
    print("Note: Run submit_cofold_batch.py --eval_csv on the output CSV for co-folding validation.")


if __name__ == "__main__":
    main()
