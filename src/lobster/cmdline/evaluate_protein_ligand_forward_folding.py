"""Standalone evaluation of forward folding on protein-ligand complexes.

Evaluates whether ligand context improves forward folding (structure prediction)
performance, particularly for binding pocket residues.

Usage:
    uv run python -m lobster.cmdline.evaluate_protein_ligand_forward_folding \
        --checkpoint path/to/model.ckpt \
        --data_dir /data2/lisanzas/pdb_bind_12_15_25/test/ \
        --output results.csv \
        --structure_path ./output/ \
        --pocket_threshold 5.0 \
        --num_samples 100

Example (full test set):
    uv run python -m lobster.cmdline.evaluate_protein_ligand_forward_folding \
        --checkpoint /data2/ume/gen_ume_protein_ligand/best.ckpt \
        --data_dir /data2/lisanzas/pdb_bind_12_15_25/test/ \
        --output protein_ligand_forward_folding_results.csv \
        --structure_path ./protein_ligand_eval/ \
        --save_structures \
        --num_samples -1
"""

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from lobster.metrics.protein_ligand_forward_folding import ProteinLigandForwardFoldingEvaluator
from lobster.model.gen_ume import ProteinLigandEncoderLightningModule


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate forward folding on protein-ligand complexes with/without ligand context"
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
        default="protein_ligand_forward_folding_results.csv",
        help="Output CSV file for results",
    )
    parser.add_argument(
        "--structure_path",
        type=str,
        default=None,
        help="Output directory for predicted structures (PDB files)",
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
    parser.add_argument(
        "--minimize_ligand",
        action="store_true",
        help="Apply geometry correction to decoded ligand structures",
    )
    parser.add_argument(
        "--minimize_mode",
        type=str,
        default="bonds_and_angles",
        choices=["bonds_only", "bonds_and_angles", "local", "full"],
        help="Minimization mode",
    )
    parser.add_argument(
        "--force_field",
        type=str,
        default="MMFF94",
        help="Force field for minimization (MMFF94, UFF, etc.)",
    )
    parser.add_argument(
        "--minimize_steps",
        type=int,
        default=500,
        help="Maximum number of minimization steps",
    )
    # Additional generation hyperparameters
    parser.add_argument(
        "--stochasticity_seq",
        type=int,
        default=20,
        help="Stochasticity parameter for sequence sampling",
    )
    parser.add_argument(
        "--stochasticity_struc",
        type=int,
        default=20,
        help="Stochasticity parameter for structure sampling",
    )
    parser.add_argument(
        "--temperature_ligand",
        type=float,
        default=0.5,
        help="Temperature for ligand structure sampling",
    )
    parser.add_argument(
        "--stochasticity_ligand",
        type=int,
        default=20,
        help="Stochasticity parameter for ligand structure sampling",
    )
    parser.add_argument(
        "--ligand_context_mode",
        type=str,
        default="structure_tokens",
        choices=["structure_tokens", "atom_bond_only"],
        help="How to provide ligand context: 'structure_tokens' or 'atom_bond_only'",
    )
    parser.add_argument(
        "--inference_schedule_seq",
        type=str,
        default="LogInferenceSchedule",
        choices=["LinearInferenceSchedule", "LogInferenceSchedule", "PowerInferenceSchedule"],
        help="Inference schedule for sequence generation",
    )
    parser.add_argument(
        "--inference_schedule_struc",
        type=str,
        default="LinearInferenceSchedule",
        choices=["LinearInferenceSchedule", "LogInferenceSchedule", "PowerInferenceSchedule"],
        help="Inference schedule for structure generation",
    )
    parser.add_argument(
        "--inference_schedule_ligand_atom",
        type=str,
        default=None,
        choices=["LinearInferenceSchedule", "LogInferenceSchedule", "PowerInferenceSchedule"],
        help="Inference schedule for ligand atom token generation (default: use sequence schedule)",
    )
    parser.add_argument(
        "--inference_schedule_ligand_struc",
        type=str,
        default=None,
        choices=["LinearInferenceSchedule", "LogInferenceSchedule", "PowerInferenceSchedule"],
        help="Inference schedule for ligand structure token generation (default: use structure schedule)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for reproducibility (sets torch, numpy, and python random seeds)",
    )
    parser.add_argument(
        "--num_predictions",
        type=int,
        default=1,
        help="Number of predictions per sample for best-of-N evaluation (default: 1)",
    )
    parser.add_argument(
        "--best_of_n_metric",
        type=str,
        default="rmsd",
        choices=["rmsd", "tm_score"],
        help="Metric to use for best-of-N selection: 'rmsd' (lower is better) or 'tm_score' (higher is better)",
    )
    parser.add_argument(
        "--save_all_predictions",
        action="store_true",
        help="Save all N predicted structures (not just the best). Requires --save_structures and --num_predictions > 1",
    )
    parser.add_argument(
        "--try_reflection",
        action="store_true",
        help="Try both original and reflected (mirror image) coordinates, selecting the one with higher TM-score. "
        "Useful if the model outputs mirror images of structures.",
    )
    parser.add_argument(
        "--max_protein_length",
        type=int,
        default=512,
        help="Maximum protein-only length. Samples exceeding this are skipped (default: 512)",
    )
    parser.add_argument(
        "--use_protenix",
        action="store_true",
        help="Additionally validate with Protenix co-folding via Pylon endpoint",
    )
    parser.add_argument(
        "--use_boltz",
        action="store_true",
        help="Additionally validate with Boltz-2 co-folding via Pylon (alternative to --use_protenix)",
    )
    parser.add_argument(
        "--raw_data_dir",
        type=str,
        default=None,
        help="Path to raw benchmark data with SDF files for SMILES extraction (required for --use_protenix/--use_boltz)",
    )

    args = parser.parse_args()

    # Set random seeds for reproducibility
    logger.info(f"Setting random seed: {args.seed}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Validate checkpoint exists
    if not Path(args.checkpoint).exists():
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    # Validate data directory exists
    if not Path(args.data_dir).exists():
        logger.error(f"Data directory not found: {args.data_dir}")
        sys.exit(1)

    # Create structure_path directory if specified
    if args.structure_path:
        os.makedirs(args.structure_path, exist_ok=True)
        logger.info(f"Output directory: {args.structure_path}")

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

    evaluator = ProteinLigandForwardFoldingEvaluator(
        data_dir=args.data_dir,
        pocket_distance_threshold=args.pocket_threshold,
        num_samples=num_samples,
        nsteps=args.nsteps,
        device=args.device,
        max_length=max_length,
        max_protein_length=args.max_protein_length,
        temperature_seq=args.temperature_seq,
        temperature_struc=args.temperature_struc,
        save_structures=args.save_structures,
        save_gt_structure=args.save_gt_structure,
        minimize_ligand=args.minimize_ligand,
        minimize_mode=args.minimize_mode,
        force_field=args.force_field,
        minimize_steps=args.minimize_steps,
        # Additional generation hyperparameters
        stochasticity_seq=args.stochasticity_seq,
        stochasticity_struc=args.stochasticity_struc,
        temperature_ligand=args.temperature_ligand,
        stochasticity_ligand=args.stochasticity_ligand,
        ligand_context_mode=args.ligand_context_mode,
        inference_schedule_seq=args.inference_schedule_seq,
        inference_schedule_struc=args.inference_schedule_struc,
        inference_schedule_ligand_atom=args.inference_schedule_ligand_atom,
        inference_schedule_ligand_struc=args.inference_schedule_ligand_struc,
        # Best-of-N parameters
        num_predictions=args.num_predictions,
        best_of_n_metric=args.best_of_n_metric,
        save_all_predictions=args.save_all_predictions,
        # Mirror image handling
        try_reflection=args.try_reflection,
        # Co-folding validation
        use_protenix=args.use_protenix,
        use_boltz=args.use_boltz,
        raw_data_dir=args.raw_data_dir,
    )

    # Load samples
    logger.info("Loading test samples...")
    samples = evaluator.load_test_set()
    logger.info(f"Loaded {len(samples)} samples")

    # Run evaluation
    logger.info("Running evaluation...")
    results = evaluator.evaluate(model, samples, structure_path=args.structure_path)

    # Save results CSV
    output_path = args.output
    if args.structure_path:
        output_path = os.path.join(args.structure_path, os.path.basename(args.output))
    results["results_df"].to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path}")

    # Print summary
    summary = results["summary"]
    print("\n" + "=" * 70)
    print("Protein-Ligand Forward Folding Evaluation Results")
    print("=" * 70)

    print(f"\nSamples evaluated: {summary['n_samples']}")
    print(f"Average pocket size: {summary['mean_pocket_size']:.1f} residues")
    print(f"Pocket distance threshold: {args.pocket_threshold} Å")
    if args.num_predictions > 1:
        print(f"Best-of-N: {args.num_predictions} predictions (selecting by {args.best_of_n_metric})")
    if args.try_reflection:
        print("Mirror image handling: enabled")
        if "reflection_rate_no_ligand" in summary:
            print(
                f"  Reflected (no ligand):   {summary['n_reflected_no_ligand']}/{summary['n_samples']} "
                f"({summary['reflection_rate_no_ligand']:.1%})"
            )
            print(
                f"  Reflected (with ligand): {summary['n_reflected_with_ligand']}/{summary['n_samples']} "
                f"({summary['reflection_rate_with_ligand']:.1%})"
            )

    print("\n--- TM-Score (Overall Structure Quality) ---")
    print(f"  Without ligand: {summary['mean_tm_score_no_ligand']:.4f}")
    print(f"  With ligand:    {summary['mean_tm_score_with_ligand']:.4f}")
    print(f"  Delta:          {summary['mean_tm_score_delta']:+.4f} (±{summary['std_tm_score_delta']:.4f})")

    print("\n--- Overall RMSD (Å) ---")
    print(f"  Without ligand: {summary['mean_rmsd_overall_no_ligand']:.2f}")
    print(f"  With ligand:    {summary['mean_rmsd_overall_with_ligand']:.2f}")
    print(f"  Delta:          {summary['mean_rmsd_overall_delta']:+.2f} (±{summary['std_rmsd_overall_delta']:.2f})")

    print("\n--- Binding Pocket RMSD (Å) ---")
    print(f"  Without ligand: {summary['mean_rmsd_pocket_no_ligand']:.2f}")
    print(f"  With ligand:    {summary['mean_rmsd_pocket_with_ligand']:.2f}")
    print(f"  Delta:          {summary['mean_rmsd_pocket_delta']:+.2f} (±{summary['std_rmsd_pocket_delta']:.2f})")

    print("\n--- Non-Pocket RMSD (Å) ---")
    print(f"  Without ligand: {summary['mean_rmsd_nonpocket_no_ligand']:.2f}")
    print(f"  With ligand:    {summary['mean_rmsd_nonpocket_with_ligand']:.2f}")
    print(f"  Delta:          {summary['mean_rmsd_nonpocket_delta']:+.2f} (±{summary['std_rmsd_nonpocket_delta']:.2f})")

    if "mean_ligand_rmsd_aligned" in summary:
        print("\n--- Ligand Placement ---")
        print(f"  Ligand RMSD (raw):     {summary['mean_ligand_rmsd']:.2f} Å")
        print(f"  Ligand RMSD (aligned): {summary['mean_ligand_rmsd_aligned']:.2f} Å")
        print(f"  Ligand centroid dist (aligned): {summary['mean_ligand_centroid_distance_aligned']:.2f} Å")
        print(f"  Protein-ligand contacts (6Å): {summary['mean_protein_ligand_contacts']:.1f}")
        print(f"  Frac ligand atoms contacted:  {summary['mean_frac_ligand_atoms_contacted']:.3f}")
        print(f"  Ligand contacts protein:       {summary.get('ligand_contacts_protein_fraction', 0):.1%}")
        print(f"  Ligand in correct pocket:      {summary['ligand_in_pocket_fraction']:.1%}")
        if "good_fold_and_in_pocket_fraction" in summary:
            print(f"  Good fold + in pocket (TM>0.5): {summary['good_fold_and_in_pocket_fraction']:.1%}")
        print(f"  Mean pocket contacts:          {summary.get('mean_pocket_contacts', 0):.1f}")

    print("\n" + "=" * 70)

    # Key insights
    tm_delta = summary["mean_tm_score_delta"]
    pocket_rmsd_delta = summary["mean_rmsd_pocket_delta"]

    if tm_delta > 0.01:
        print(f"🎯 Ligand context IMPROVES TM-score by {tm_delta:.4f}!")
    elif tm_delta < -0.01:
        print(f"⚠️  Ligand context DECREASES TM-score by {abs(tm_delta):.4f}")
    else:
        print("📊 Ligand context has minimal effect on TM-score")

    # For RMSD, negative delta means improvement (lower RMSD is better)
    if pocket_rmsd_delta < -0.1:
        print(f"🎯 Ligand context IMPROVES pocket RMSD by {abs(pocket_rmsd_delta):.2f} Å!")
    elif pocket_rmsd_delta > 0.1:
        print(f"⚠️  Ligand context INCREASES pocket RMSD by {pocket_rmsd_delta:.2f} Å")
    else:
        print("📊 Ligand context has minimal effect on pocket RMSD")

    print("=" * 70)


if __name__ == "__main__":
    main()
