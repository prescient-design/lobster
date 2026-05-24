"""Standalone evaluation of ligand-conditioned protein generation.

Evaluates whether the model can generate self-consistent proteins conditioned
on a ligand. The model generates both sequence and structure from scratch;
the sequence is then folded with ESMFold, and the self-consistency between
the model-decoded structure and the ESMFold prediction is measured.

Usage:
    uv run python -m lobster.cmdline.evaluation.evaluate_ligand_conditioned_protein_generation \
        --output results.csv \
        --structure_path ./output/ \
        --length 100 \
        --num_samples 10

Example (full test set):
    uv run python -m lobster.cmdline.evaluation.evaluate_ligand_conditioned_protein_generation \
        --output ligand_cond_protein_gen_results.csv \
        --structure_path ./ligand_cond_eval/ \
        --length 100 \
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

from lobster.cmdline._ligand_conditioned_runner import (
    LigandConditionedRunConfig,
    run_ligand_conditioned_generation,
)
from lobster.model.leflur import (
    LeFlurProteinLigandLightningModule,
    resolve_checkpoint,
)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate ligand-conditioned protein generation via "
            "self-consistency (decoded structure vs ESMFold prediction)"
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/cv/scratch/u/lisanzas/gen_ume_protein_ligand_medium/runs/2026-02-11T19-45-30/epoch=278-step=40057-val_loss=1.6365.ckpt",
        help="Path to model checkpoint (.ckpt file)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/cv/home/lisanzas/lobster/data/posebusters/processed/posebusters_benchmark_no_overlap/",
        help="Path to directory with *_ligand.pt files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ligand_cond_protein_gen_results.csv",
        help="Output CSV file for results",
    )
    parser.add_argument(
        "--structure_path",
        type=str,
        default=None,
        help="Output directory for generated structures (PDB/FASTA files)",
    )
    parser.add_argument(
        "--length",
        type=int,
        default=100,
        help="Length of protein to generate (number of residues, default: 100)",
    )
    parser.add_argument(
        "--pocket_threshold",
        type=float,
        default=5.0,
        help="Distance threshold (angstrom) for defining binding pocket on decoded structure",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of ligands to evaluate (-1 for all)",
    )
    parser.add_argument(
        "--num_designs",
        type=int,
        default=10,
        help="Number of designs to generate per ligand (best by scTM is reported)",
    )
    parser.add_argument(
        "--nsteps",
        type=int,
        default=200,
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
        default=0.153,
        help="Temperature for sequence sampling",
    )
    parser.add_argument(
        "--temperature_struc",
        type=float,
        default=0.05,
        help="Temperature for structure sampling",
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
        default=0.1,
        help="Temperature for ligand structure sampling",
    )
    parser.add_argument(
        "--stochasticity_ligand",
        type=int,
        default=5,
        help="Stochasticity parameter for ligand structure sampling",
    )
    parser.add_argument(
        "--ligand_context_mode",
        type=str,
        default="atom_bond_only",
        choices=["structure_tokens", "atom_bond_only"],
        help="How to provide ligand context: 'atom_bond_only' or 'structure_tokens'",
    )
    parser.add_argument(
        "--inference_schedule_seq",
        type=str,
        default="LinearInferenceSchedule",
        choices=[
            "LinearInferenceSchedule",
            "LogInferenceSchedule",
            "PowerInferenceSchedule",
        ],
        help="Inference schedule for sequence generation",
    )
    parser.add_argument(
        "--inference_schedule_struc",
        type=str,
        default="PowerInferenceSchedule",
        choices=[
            "LinearInferenceSchedule",
            "LogInferenceSchedule",
            "PowerInferenceSchedule",
        ],
        help="Inference schedule for structure generation",
    )
    parser.add_argument(
        "--inference_schedule_ligand_atom",
        type=str,
        default="PowerInferenceSchedule",
        choices=[
            "LinearInferenceSchedule",
            "LogInferenceSchedule",
            "PowerInferenceSchedule",
        ],
        help="Inference schedule for ligand atom token generation",
    )
    parser.add_argument(
        "--inference_schedule_ligand_struc",
        type=str,
        default="LinearInferenceSchedule",
        choices=[
            "LinearInferenceSchedule",
            "LogInferenceSchedule",
            "PowerInferenceSchedule",
        ],
        help="Inference schedule for ligand structure token generation",
    )
    parser.add_argument(
        "--save_structures",
        action="store_true",
        help="Save decoded and ESMFold structures as PDB files",
    )
    parser.add_argument(
        "--minimize_ligand",
        action="store_true",
        help="Apply force-field minimization to decoded ligand geometry",
    )
    parser.add_argument(
        "--minimize_mode",
        type=str,
        default="bonds_and_angles",
        choices=["bonds_only", "bonds_and_angles", "local", "full"],
        help="Minimization mode for ligand geometry correction",
    )
    parser.add_argument(
        "--force_field",
        type=str,
        default="MMFF94",
        choices=["MMFF94", "MMFF94s", "UFF"],
        help="Force field for ligand minimization",
    )
    parser.add_argument(
        "--minimize_steps",
        type=int,
        default=500,
        help="Maximum number of ligand minimization steps",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--use_protenix",
        action="store_true",
        help="Validate with Protenix co-folding via Pylon endpoint (iptm, chain_pair_iptm, scTM)",
    )
    parser.add_argument(
        "--use_boltz",
        action="store_true",
        help="Validate with Boltz-2 co-folding via Pylon (alternative to --use_protenix)",
    )
    parser.add_argument(
        "--raw_data_dir",
        type=str,
        default=None,
        help="Path to raw benchmark data with SDF files for SMILES extraction (required for --use_protenix/--use_boltz)",
    )
    parser.add_argument(
        "--skip_esmfold",
        action="store_true",
        help="Skip ESMFold validation (use only Protenix for validation)",
    )

    args = parser.parse_args()

    # Set random seeds
    logger.info(f"Setting random seed: {args.seed}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Resolve checkpoint via the shared Phase 4 resolver so HF short names /
    # hf:// URIs / local paths all work uniformly.
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

    # Set device
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
    if not args.skip_esmfold:
        from lobster.model import LobsterPLMFold

        logger.info("Loading ESMFold for self-consistency evaluation...")
        plm_fold = LobsterPLMFold(model_name="esmfold_v1", max_length=512)
        plm_fold.to(args.device)
        logger.info("ESMFold loaded successfully")
    else:
        logger.info("ESMFold skipped (--skip_esmfold)")

    num_samples = None if args.num_samples == -1 else args.num_samples
    max_length = 512
    if hasattr(model, "encoder") and hasattr(model.encoder, "neobert"):
        if hasattr(model.encoder.neobert, "config") and hasattr(model.encoder.neobert.config, "max_length"):
            max_length = model.encoder.neobert.config.max_length
            logger.info(f"Using model's max_length: {max_length}")

    # Output dir for the CSV: when structure_path is given the runner
    # writes alongside the structures; otherwise we use the current
    # directory (matches pre-Phase-5 evaluator behaviour).
    csv_output_dir = args.structure_path if args.structure_path else os.getcwd()

    config = LigandConditionedRunConfig(
        data_dir=args.data_dir,
        output_dir=csv_output_dir,
        output_csv_name=os.path.basename(args.output),
        structure_path=args.structure_path,
        length=args.length,
        pocket_distance_threshold=args.pocket_threshold,
        num_samples=num_samples,
        num_designs=args.num_designs,
        nsteps=args.nsteps,
        device=args.device,
        max_length=max_length,
        temperature_seq=args.temperature_seq,
        temperature_struc=args.temperature_struc,
        stochasticity_seq=args.stochasticity_seq,
        stochasticity_struc=args.stochasticity_struc,
        temperature_ligand=args.temperature_ligand,
        stochasticity_ligand=args.stochasticity_ligand,
        ligand_context_mode=args.ligand_context_mode,
        inference_schedule_seq=args.inference_schedule_seq,
        inference_schedule_struc=args.inference_schedule_struc,
        inference_schedule_ligand_atom=args.inference_schedule_ligand_atom,
        inference_schedule_ligand_struc=args.inference_schedule_ligand_struc,
        save_structures=args.save_structures,
        minimize_ligand=args.minimize_ligand,
        minimize_mode=args.minimize_mode,
        force_field=args.force_field,
        minimize_steps=args.minimize_steps,
        seed=args.seed,
        skip_esmfold=args.skip_esmfold,
        use_protenix=args.use_protenix,
        use_boltz=args.use_boltz,
        raw_data_dir=args.raw_data_dir,
    )

    if args.minimize_ligand:
        logger.info(f"  Ligand minimization: {args.minimize_mode} ({args.force_field}, {args.minimize_steps} steps)")

    results = run_ligand_conditioned_generation(model, config, plm_fold=plm_fold)
    _print_summary(args, results["summary"])


def _print_summary(args, summary):
    """Print evaluation summary to stdout."""
    print("\n" + "=" * 70)
    print("Ligand-Conditioned Protein Generation: Self-Consistency Results")
    print("=" * 70)

    print(f"\nLigands evaluated:     {summary['n_ligands']}")
    print(f"Designs per ligand:    {summary['num_designs']}")
    print(f"Total designs:         {summary.get('n_total_designs', 'N/A')}")
    print(f"Generated protein len: {summary['protein_length']}")
    print(f"Ligand context mode:   {args.ligand_context_mode}")
    print(f"Pocket threshold:      {args.pocket_threshold} A")
    print(f"Avg pocket size:       {summary['mean_pocket_size']:.1f} residues")
    print("(Metrics below are over the best design per ligand)")

    print("\n--- Protein-Ligand Contacts ---")
    print(f"  Contacts (CA<4.5A):  {summary['mean_n_contacts']:.1f} (+/-{summary['std_n_contacts']:.1f})")
    print(
        f"  Residues in contact: {summary['mean_n_residues_in_contact']:.1f} "
        f"({summary['mean_frac_residues_in_contact']:.1%})"
    )
    print(f"  Ligand atoms contacted: {summary['mean_frac_ligand_atoms_in_contact']:.1%}")
    print(
        f"  Min distance (A):    {summary['mean_min_protein_ligand_dist']:.2f} "
        f"(+/-{summary['std_min_protein_ligand_dist']:.2f})"
    )

    print("\n--- Self-Consistency (Decoded vs ESMFold) ---")
    print(
        f"  scTM:           {summary['mean_scTM']:.4f} "
        f"(+/-{summary['std_scTM']:.4f}, "
        f"median {summary['median_scTM']:.4f})"
    )
    print(
        f"  scRMSD (A):     {summary['mean_scRMSD']:.2f} "
        f"(+/-{summary['std_scRMSD']:.2f}, "
        f"median {summary['median_scRMSD']:.2f})"
    )

    print("\n--- Pocket Self-Consistency ---")
    print(f"  pocket scTM:    {summary['mean_pocket_scTM']:.4f} (+/-{summary['std_pocket_scTM']:.4f})")
    print(f"  pocket scRMSD:  {summary['mean_pocket_scRMSD']:.2f} (+/-{summary['std_pocket_scRMSD']:.2f})")

    print("\n--- ESMFold Confidence ---")
    print(f"  pLDDT:          {summary['mean_plddt']:.2f} (+/-{summary['std_plddt']:.2f})")
    print(f"  PAE:            {summary['mean_pae']:.2f} (+/-{summary['std_pae']:.2f})")

    print("\n" + "=" * 70)

    # Key insights
    sc_tm = summary["mean_scTM"]
    if sc_tm > 0.5:
        print(f"High self-consistency (scTM={sc_tm:.3f} > 0.5): generated sequences fold into the predicted structure")
    elif sc_tm > 0.3:
        print(f"Moderate self-consistency (scTM={sc_tm:.3f}): partial agreement between decoded and folded structures")
    else:
        print(f"Low self-consistency (scTM={sc_tm:.3f} < 0.3): decoded and folded structures diverge significantly")

    plddt = summary["mean_plddt"]
    if plddt > 70:
        print(f"Good ESMFold confidence (pLDDT={plddt:.1f} > 70)")
    elif plddt > 50:
        print(f"Moderate ESMFold confidence (pLDDT={plddt:.1f})")
    else:
        print(f"Low ESMFold confidence (pLDDT={plddt:.1f} < 50)")

    mean_contacts = summary["mean_n_contacts"]
    min_dist = summary["mean_min_protein_ligand_dist"]
    if mean_contacts < 1:
        print(
            f"WARNING: No protein-ligand contacts (min dist={min_dist:.1f}A). Protein and ligand are not interacting."
        )
    elif mean_contacts < 5:
        print(f"Few protein-ligand contacts ({mean_contacts:.0f}). Weak interaction.")
    else:
        print(f"Protein-ligand contacts: {mean_contacts:.0f} (min dist={min_dist:.1f}A)")

    print("=" * 70)


if __name__ == "__main__":
    main()
