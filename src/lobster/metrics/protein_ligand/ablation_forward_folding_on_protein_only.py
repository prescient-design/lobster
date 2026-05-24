#!/usr/bin/env python
"""Evaluate protein-ligand model on protein-only data (e.g., CAMEO).

Forward folding: sequence → structure. Protein metrics only (TM-score, RMSD).
No ligand context; useful for assessing protein structure prediction on protein-only benchmarks.

Usage:
    uv run python -m lobster.metrics.protein_ligand.ablation_forward_folding_on_protein_only \
        --checkpoint /path/to/checkpoint.ckpt \
        --output results.csv

    # With structure saving
    uv run python -m lobster.metrics.protein_ligand.ablation_forward_folding_on_protein_only \
        --checkpoint /path/to/checkpoint.ckpt \
        --output results.csv \
        --structure_path ./structures/ \
        --save_structures \
        --save_gt_structure

    # Custom data path
    uv run python -m lobster.metrics.protein_ligand.ablation_forward_folding_on_protein_only \
        --checkpoint /path/to/checkpoint.ckpt \
        --data_dir "/path/to/protein_only/*.pt" \
        --output results.csv
"""

import argparse
import json
import os
import sys
from glob import glob

import pandas as pd
import torch
from loguru import logger
from tmtools import tm_align
from torch import Tensor
from tqdm import tqdm

from bionemo.moco.schedules.inference_time_schedules import (
    LinearInferenceSchedule,
    LogInferenceSchedule,
    PowerInferenceSchedule,
)

from lobster.metrics import align_and_compute_rmsd
from lobster.model.latent_generator.io import writepdb
from lobster.model.latent_generator.utils.residue_constants import restype_order_with_x_inv
from lobster.transforms._structure_transforms import AminoAcidTokenizerTransform, StructureBackboneTransform

INFERENCE_SCHEDULE_MAP = {
    "LinearInferenceSchedule": LinearInferenceSchedule,
    "LogInferenceSchedule": LogInferenceSchedule,
    "PowerInferenceSchedule": PowerInferenceSchedule,
}


def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load a LeFlur protein-ligand model from checkpoint.

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
    from lobster.model.leflur import LeFlurProteinLigandLightningModule

    logger.info(f"Loading protein-ligand model from {checkpoint_path} (protein-only evaluation mode)")

    model = LeFlurProteinLigandLightningModule.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
        strict=False,
    )
    model.eval()
    model.to(device)

    max_length = None
    if hasattr(model, "encoder") and hasattr(model.encoder, "neobert"):
        if hasattr(model.encoder.neobert, "config") and hasattr(model.encoder.neobert.config, "max_length"):
            max_length = model.encoder.neobert.config.max_length
    model.max_length = max_length

    logger.info(f"Model loaded successfully. Max length: {max_length}")
    return model


def load_protein_only_structures(
    data_dir: str,
    num_samples: int | None,
    max_length: int,
    device: str,
) -> list[dict]:
    """Load protein-only .pt files (e.g., CAMEO format)."""
    if "*" in data_dir:
        pt_files = sorted(glob(data_dir))
    else:
        pt_files = sorted(glob(os.path.join(data_dir, "*.pt")))

    if not pt_files:
        raise ValueError(f"No .pt files found at {data_dir}")

    transform = StructureBackboneTransform(max_length=max_length)
    max_files = (num_samples or len(pt_files)) * 3
    samples = []

    for pt_path in tqdm(pt_files[:max_files], desc="Loading structures"):
        try:
            data = torch.load(pt_path, weights_only=False, map_location=device)
            data = transform(data)

            if data["coords_res"].shape[0] < 30:
                continue
            percent_unknown = (data["sequence"] == 20).sum().float() / data["sequence"].shape[0]
            if percent_unknown > 0.1:
                continue

            pdb_id = os.path.splitext(os.path.basename(pt_path))[0]
            samples.append(
                {
                    "pdb_id": pdb_id,
                    "protein_coords": data["coords_res"],
                    "protein_sequence": data["sequence"],
                    "protein_mask": data.get("mask", torch.ones(data["coords_res"].shape[0], device=device)),
                    "protein_indices": data.get("indices", torch.arange(data["coords_res"].shape[0], device=device)),
                }
            )

            if num_samples and len(samples) >= num_samples:
                break
        except Exception as e:
            logger.warning(f"Failed to load {pt_path}: {e}")

    logger.info(f"Loaded {len(samples)} structures")
    return samples


def forward_fold(
    model,
    sample: dict,
    tokenizer_transform,
    device: str,
    nsteps: int,
    temperature_seq: float,
    temperature_struc: float,
    temperature_ligand: float,
    stochasticity_seq: int,
    stochasticity_struc: int,
    stochasticity_ligand: int,
    inference_schedule_seq,
    inference_schedule_struc,
    inference_schedule_ligand_atom,
    inference_schedule_ligand_struc,
) -> Tensor:
    """Run forward folding. Returns predicted coords [L, 3, 3]."""
    protein_mask = sample["protein_mask"].unsqueeze(0).float()
    protein_indices = sample["protein_indices"].unsqueeze(0).long()
    length = int(protein_mask.sum().item())

    gt_seq = sample["protein_sequence"]
    tokenized_data = tokenizer_transform({"sequence": gt_seq.cpu()})
    tokenized_seq = tokenized_data["sequence"].to(device).unsqueeze(0)

    with torch.no_grad():
        result = model.generate_sample(
            length=length,
            num_samples=1,
            forward_folding=True,
            nsteps=nsteps,
            temperature_seq=temperature_seq,
            temperature_struc=temperature_struc,
            temperature_ligand=temperature_ligand,
            stochasticity_seq=stochasticity_seq,
            stochasticity_struc=stochasticity_struc,
            stochasticity_ligand=stochasticity_ligand,
            inference_schedule_seq=inference_schedule_seq,
            inference_schedule_struc=inference_schedule_struc,
            inference_schedule_ligand_atom=inference_schedule_ligand_atom,
            inference_schedule_ligand_struc=inference_schedule_ligand_struc,
            input_sequence_tokens=tokenized_seq,
            input_mask=protein_mask,
            input_indices=protein_indices,
        )

    decoded_x = model.decode_structure(result, protein_mask)
    predicted_coords = None
    for decoder_name in decoded_x:
        if decoder_name == "vit_decoder":
            vit_output = decoded_x[decoder_name]
            predicted_coords = (
                vit_output.get("protein_coords", vit_output.get("coords"))
                if isinstance(vit_output, dict)
                else vit_output
            )
            break

    if predicted_coords is None:
        raise RuntimeError("No vit_decoder found in decoded structures")

    return predicted_coords.squeeze(0)


def compute_tm_score(pred_coords: Tensor, gt_coords: Tensor, sequence: Tensor, mask: Tensor) -> float:
    """Compute TM-score."""
    if mask is not None:
        mask = mask.bool()
        pred_coords = pred_coords[mask]
        gt_coords = gt_coords[mask]
        sequence = sequence[mask]
    if len(pred_coords) == 0:
        return float("nan")
    sequence_str = "".join([restype_order_with_x_inv.get(int(s), "X") for s in sequence.cpu().tolist()])
    pred_ca = pred_coords[:, 1, :].detach().cpu().numpy()
    gt_ca = gt_coords[:, 1, :].detach().cpu().numpy()
    tm_out = tm_align(pred_ca, gt_ca, sequence_str, sequence_str)
    return tm_out.tm_norm_chain1


def compute_rmsd(pred_coords: Tensor, gt_coords: Tensor, mask: Tensor) -> float:
    """Compute RMSD."""
    if mask is not None:
        mask = mask.bool()
        pred_coords = pred_coords[mask]
        gt_coords = gt_coords[mask]
    if len(pred_coords) == 0:
        return float("nan")
    return float(
        align_and_compute_rmsd(
            pred_coords.detach(),
            gt_coords.detach(),
            mask=None,
            return_aligned=False,
            device=pred_coords.device,
        )
    )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate protein-ligand model on protein-only data (forward folding, protein metrics only)",
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
        default="/cv/data/ai4dd/data2/lisanzas/AFDB/valid_cameo_processed/*.pt",
        help="Path to protein-only data (glob pattern for .pt files, e.g., CAMEO)",
    )

    # Output options
    parser.add_argument(
        "--output",
        type=str,
        default="protein_ligand_forward_folding_on_protein_only_results.csv",
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
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (None = all available)",
    )
    parser.add_argument(
        "--nsteps",
        type=int,
        default=200,
        help="Number of diffusion steps for generation",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum protein sequence length to process",
    )
    parser.add_argument(
        "--temperature_seq",
        type=float,
        default=0.15279667854390633,
        help="Temperature for sequence sampling",
    )
    parser.add_argument(
        "--temperature_struc",
        type=float,
        default=0.18605909386731256,
        help="Temperature for structure sampling",
    )
    parser.add_argument(
        "--temperature_ligand",
        type=float,
        default=0.5819150856331732,
        help="Temperature for ligand sampling (unused in protein-only mode)",
    )
    parser.add_argument(
        "--stochasticity_seq",
        type=int,
        default=10,
        help="Stochasticity for sequence sampling",
    )
    parser.add_argument(
        "--stochasticity_struc",
        type=int,
        default=10,
        help="Stochasticity for structure sampling",
    )
    parser.add_argument(
        "--stochasticity_ligand",
        type=int,
        default=20,
        help="Stochasticity for ligand sampling (unused in protein-only mode)",
    )
    parser.add_argument(
        "--inference_schedule_seq",
        type=str,
        default="LinearInferenceSchedule",
        choices=list(INFERENCE_SCHEDULE_MAP.keys()),
        help="Inference schedule for sequence",
    )
    parser.add_argument(
        "--inference_schedule_struc",
        type=str,
        default="PowerInferenceSchedule",
        choices=list(INFERENCE_SCHEDULE_MAP.keys()),
        help="Inference schedule for structure",
    )
    parser.add_argument(
        "--inference_schedule_ligand_atom",
        type=str,
        default="PowerInferenceSchedule",
        choices=list(INFERENCE_SCHEDULE_MAP.keys()),
        help="Inference schedule for ligand atoms (unused in protein-only mode)",
    )
    parser.add_argument(
        "--inference_schedule_ligand_struc",
        type=str,
        default="LinearInferenceSchedule",
        choices=list(INFERENCE_SCHEDULE_MAP.keys()),
        help="Inference schedule for ligand structure (unused in protein-only mode)",
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
    if not args.checkpoint.startswith("s3://") and not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    pt_count = (
        len(glob(args.data_dir))
        if "*" in args.data_dir
        else len(glob(os.path.join(args.data_dir, "*.pt")))
    )
    if pt_count == 0:
        logger.error(f"No .pt files found at {args.data_dir}")
        sys.exit(1)

    # Load model
    model = load_model(args.checkpoint, args.device)

    max_length = args.max_length
    if hasattr(model, "max_length") and model.max_length is not None:
        max_length = min(max_length, model.max_length)
        logger.info(f"Using max_length: {max_length}")

    tokenizer_transform = AminoAcidTokenizerTransform(max_length=max_length)
    samples = load_protein_only_structures(
        args.data_dir, args.num_samples, max_length, args.device
    )

    schedule_seq = INFERENCE_SCHEDULE_MAP[args.inference_schedule_seq]
    schedule_struc = INFERENCE_SCHEDULE_MAP[args.inference_schedule_struc]
    schedule_lig_atom = INFERENCE_SCHEDULE_MAP[args.inference_schedule_ligand_atom]
    schedule_lig_struc = INFERENCE_SCHEDULE_MAP[args.inference_schedule_ligand_struc]

    if args.structure_path:
        os.makedirs(args.structure_path, exist_ok=True)

    results = []
    for sample in tqdm(samples, desc="Evaluating forward folding"):
        pdb_id = sample["pdb_id"]
        gt_seq = sample["protein_sequence"]
        gt_coords = sample["protein_coords"]
        protein_mask = sample["protein_mask"]

        if len(gt_seq) > max_length:
            logger.warning(f"Skipping {pdb_id}: length {len(gt_seq)} > {max_length}")
            continue

        try:
            pred_coords = forward_fold(
                model,
                sample,
                tokenizer_transform,
                args.device,
                args.nsteps,
                args.temperature_seq,
                args.temperature_struc,
                args.temperature_ligand,
                args.stochasticity_seq,
                args.stochasticity_struc,
                args.stochasticity_ligand,
                schedule_seq,
                schedule_struc,
                schedule_lig_atom,
                schedule_lig_struc,
            )
        except Exception as e:
            logger.warning(f"Failed {pdb_id}: {e}")
            continue

        if args.structure_path:
            if args.save_gt_structure:
                writepdb(os.path.join(args.structure_path, f"{pdb_id}_gt.pdb"), gt_coords, gt_seq)
            if args.save_structures:
                writepdb(
                    os.path.join(args.structure_path, f"{pdb_id}_pred.pdb"),
                    pred_coords.detach(),
                    gt_seq,
                )

        results.append(
            {
                "pdb_id": pdb_id,
                "length": len(gt_seq),
                "tm_score": compute_tm_score(pred_coords, gt_coords, gt_seq, protein_mask),
                "rmsd": compute_rmsd(pred_coords, gt_coords, protein_mask),
            }
        )

    df = pd.DataFrame(results)
    if len(df) == 0:
        logger.warning("No samples were successfully evaluated")
        sys.exit(1)

    df.to_csv(args.output, index=False)
    logger.info(f"Saved per-structure results to {args.output}")

    summary = {
        "mean_tm_score": df["tm_score"].mean(),
        "std_tm_score": df["tm_score"].std(),
        "mean_rmsd": df["rmsd"].mean(),
        "std_rmsd": df["rmsd"].std(),
        "n_samples": len(df),
    }

    # Print summary
    print("\n" + "=" * 80)
    print("PROTEIN-LIGAND FORWARD FOLDING ON PROTEIN-ONLY DATA")
    print("=" * 80)
    print(f"\nDataset: {args.data_dir}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Samples evaluated: {summary['n_samples']}")

    print("\n--- Protein Metrics (No Ligand Context) ---")
    print(f"\n{'Metric':<25} {'Value':<20}")
    print("-" * 45)
    print(f"{'TM-Score':<25} {summary['mean_tm_score']:.3f} ± {summary['std_tm_score']:.3f}")
    print(f"{'RMSD (Å)':<25} {summary['mean_rmsd']:.2f} ± {summary['std_rmsd']:.2f}")
    print("=" * 80)

    if args.output_json:
        summary_json = {k: float(v) if hasattr(v, "item") else v for k, v in summary.items()}
        summary_json["checkpoint"] = args.checkpoint
        summary_json["data_dir"] = args.data_dir
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
