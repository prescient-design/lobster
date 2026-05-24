"""Standalone evaluation of inverse folding on protein-ligand complexes.

Evaluates whether ligand context improves inverse folding performance,
particularly for binding pocket residues.

Usage:
    uv run python -m lobster.cmdline.evaluate_protein_ligand_inverse_folding \
        --checkpoint path/to/model.ckpt \
        --data_dir /data2/lisanzas/pdb_bind_12_15_25/test/ \
        --output results.csv \
        --structure_path ./output/ \
        --pocket_threshold 5.0 \
        --num_samples 100

Example (full test set):
    uv run python -m lobster.cmdline.evaluate_protein_ligand_inverse_folding \
        --checkpoint /data2/ume/gen_ume_protein_ligand/best.ckpt \
        --data_dir /data2/lisanzas/pdb_bind_12_15_25/test/ \
        --output protein_ligand_inverse_folding_results.csv \
        --structure_path ./protein_ligand_eval/ \
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

from lobster.cmdline._ligand_conditioned_runner import (
    ProteinLigandInverseFoldingRunConfig,
    run_protein_ligand_inverse_folding,
)
from lobster.model.leflur import (
    LeFlurProteinLigandLightningModule,
    resolve_checkpoint,
)


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
        "--structure_path",
        type=str,
        default=None,
        help="Output directory for designed sequences (FASTA files)",
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
        "--decode_structure",
        action="store_true",
        help="Decode and save predicted structures as PDB files",
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
        "--save_reconstructed_input",
        action="store_true",
        help="Save reconstructed input structures (encode then decode) to verify token fidelity",
    )
    parser.add_argument(
        "--use_se3_augmentation",
        action="store_true",
        help="Apply random SE3 augmentation (rotation + translation) to input structures before encoding",
    )
    parser.add_argument(
        "--se3_translation_scale",
        type=float,
        default=1.0,
        help="Scale factor for random translation when SE3 augmentation is enabled",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for reproducibility (sets torch, numpy, and python random seeds)",
    )
    parser.add_argument(
        "--use_esmfold",
        action="store_true",
        help="Validate designed sequences with ESMFold (fold and compare to GT structure)",
    )
    parser.add_argument(
        "--use_protenix",
        action="store_true",
        help="Validate designed sequences with Protenix co-folding via Pylon (protein + ligand SMILES)",
    )
    parser.add_argument(
        "--use_boltz",
        action="store_true",
        help="Validate designed sequences with Boltz-2 co-folding via Pylon (alternative to --use_protenix)",
    )
    parser.add_argument(
        "--raw_data_dir",
        type=str,
        default=None,
        help="Path to raw benchmark data with SDF files for SMILES extraction (required for --use_protenix/--use_boltz)",
    )
    parser.add_argument(
        "--max_protein_length",
        type=int,
        default=512,
        help="Maximum protein-only length. Samples exceeding this are skipped. Also used as ESMFold max length (default: 512)",
    )

    args = parser.parse_args()

    # Set random seeds for reproducibility
    logger.info(f"Setting random seed: {args.seed}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    try:
        ckpt_path = resolve_checkpoint(args.checkpoint)
    except (FileNotFoundError, ValueError) as exc:
        logger.error(f"Checkpoint resolution failed: {exc}")
        sys.exit(1)

    if not Path(args.data_dir).exists():
        logger.error(f"Data directory not found: {args.data_dir}")
        sys.exit(1)

    if args.structure_path:
        os.makedirs(args.structure_path, exist_ok=True)
        logger.info(f"Output directory: {args.structure_path}")

    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = "cpu"

    logger.info(f"Loading model from {ckpt_path}")
    try:
        model = LeFlurProteinLigandLightningModule.load_from_checkpoint(
            str(ckpt_path),
            map_location=args.device,
        )
        model.eval()
        model.to(args.device)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)

    plm_fold = None
    if args.use_esmfold:
        from lobster.model import LobsterPLMFold

        logger.info("Loading ESMFold for structure validation...")
        plm_fold = LobsterPLMFold(model_name="esmfold_v1", max_length=512)
        plm_fold.to(args.device)
        logger.info("ESMFold loaded successfully")

    num_samples = None if args.num_samples == -1 else args.num_samples

    max_length = 512
    if hasattr(model, "encoder") and hasattr(model.encoder, "neobert"):
        if hasattr(model.encoder.neobert, "config") and hasattr(model.encoder.neobert.config, "max_length"):
            max_length = model.encoder.neobert.config.max_length
            logger.info(f"Using model's max_length: {max_length}")

    csv_output_dir = (
        args.structure_path if args.structure_path else os.getcwd()
    )

    config = ProteinLigandInverseFoldingRunConfig(
        data_dir=args.data_dir,
        output_dir=csv_output_dir,
        output_csv_name=os.path.basename(args.output),
        structure_path=args.structure_path,
        pocket_distance_threshold=args.pocket_threshold,
        num_samples=num_samples,
        nsteps=args.nsteps,
        device=args.device,
        max_length=max_length,
        max_protein_length=args.max_protein_length,
        temperature_seq=args.temperature_seq,
        temperature_struc=args.temperature_struc,
        stochasticity_seq=args.stochasticity_seq,
        stochasticity_struc=args.stochasticity_struc,
        temperature_ligand=args.temperature_ligand,
        stochasticity_ligand=args.stochasticity_ligand,
        inference_schedule_seq=args.inference_schedule_seq,
        inference_schedule_struc=args.inference_schedule_struc,
        inference_schedule_ligand_atom=args.inference_schedule_ligand_atom,
        inference_schedule_ligand_struc=args.inference_schedule_ligand_struc,
        decode_structure=args.decode_structure,
        save_gt_structure=args.save_gt_structure,
        save_reconstructed_input=args.save_reconstructed_input,
        minimize_ligand=args.minimize_ligand,
        minimize_mode=args.minimize_mode,
        force_field=args.force_field,
        minimize_steps=args.minimize_steps,
        use_se3_augmentation=args.use_se3_augmentation,
        se3_translation_scale=args.se3_translation_scale,
        use_esmfold=args.use_esmfold,
        use_protenix=args.use_protenix,
        use_boltz=args.use_boltz,
        raw_data_dir=args.raw_data_dir,
        seed=args.seed,
    )

    if args.use_se3_augmentation:
        logger.info(f"SE3 augmentation ENABLED (translation_scale={args.se3_translation_scale})")
    else:
        logger.info("SE3 augmentation DISABLED (deterministic encoding)")

    results = run_protein_ligand_inverse_folding(model, config, plm_fold=plm_fold)

    # CSV already written by the shared runner.
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

    print("\n" + "=" * 70)

    # Key insight
    pocket_delta = summary["mean_aar_pocket_delta"]
    if pocket_delta > 0.01:
        print(f"Ligand context IMPROVES pocket recovery by {pocket_delta * 100:.1f}%!")
    elif pocket_delta < -0.01:
        print(f"Ligand context DECREASES pocket recovery by {abs(pocket_delta) * 100:.1f}%")
    else:
        print("Ligand context has minimal effect on pocket recovery")

    print("=" * 70)


if __name__ == "__main__":
    main()
