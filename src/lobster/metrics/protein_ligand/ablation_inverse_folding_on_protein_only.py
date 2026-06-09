#!/usr/bin/env python
"""Evaluate protein-ligand model on protein-only data (e.g., CAMEO).

Inverse folding: structure → sequence. Protein metrics only (amino acid recovery).
No ligand context; useful for assessing sequence recovery on protein-only benchmarks.

Usage:
    uv run python -m lobster.metrics.protein_ligand.ablation_inverse_folding_on_protein_only \
        --checkpoint /path/to/checkpoint.ckpt \
        --output results.csv

    # With structure decoding
    uv run python -m lobster.metrics.protein_ligand.ablation_inverse_folding_on_protein_only \
        --checkpoint /path/to/checkpoint.ckpt \
        --output results.csv \
        --structure_path ./structures/ \
        --decode_structure \
        --save_gt_structure

    # Custom data path
    uv run python -m lobster.metrics.protein_ligand.ablation_inverse_folding_on_protein_only \
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
from torch import Tensor
from tqdm import tqdm

from bionemo.moco.schedules.inference_time_schedules import (
    LinearInferenceSchedule,
    LogInferenceSchedule,
    PowerInferenceSchedule,
)

from lobster.metrics import get_folded_structure_metrics
from lobster.model.latent_generator.io import writepdb
from lobster.model.latent_generator.utils.residue_constants import (
    convert_lobster_aa_tokenization_to_standard_aa,
)
from lobster.transforms._structure_transforms import StructureBackboneTransform

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


# Standard AA codes (alphabetical order: A,R,N,D,C,Q,E,G,H,I,L,K,M,F,P,S,T,W,Y,V,X)
STANDARD_AA = "ARNDCQEGHILKMFPSTWYVX"

# Lobster to standard AA mapping (21-token vocab)
LOBSTER_TO_STANDARD = torch.tensor(
    [10, 0, 7, 19, 15, 6, 1, 16, 9, 3, 14, 11, 5, 13, 2, 18, 12, 8, 17, 4, 20],
    dtype=torch.long,
)


def inverse_fold(
    model,
    sample: dict,
    device: str,
    nsteps: int,
    decode_structure: bool = False,
    temperature_seq: float = 0.5,
    temperature_struc: float = 0.5,
    temperature_ligand: float = 0.5,
    stochasticity_seq: int = 20,
    stochasticity_struc: int = 20,
    stochasticity_ligand: int = 20,
    inference_schedule_seq=None,
    inference_schedule_struc=None,
    inference_schedule_ligand_atom=None,
    inference_schedule_ligand_struc=None,
) -> dict:
    """Run inverse folding. Returns dict with predicted_sequence, decoded_coords (optional)."""
    protein_coords = sample["protein_coords"].unsqueeze(0).float()
    protein_mask = sample["protein_mask"].unsqueeze(0).float()
    protein_indices = sample["protein_indices"].unsqueeze(0).long()

    with torch.no_grad():
        result = model.generate_sample(
            length=protein_coords.shape[1],
            num_samples=1,
            inverse_folding=True,
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
            input_structure_coords=protein_coords,
            input_mask=protein_mask,
            input_indices=protein_indices,
        )

        decoded_coords = None
        if decode_structure:
            decoded_x = model.decode_structure(result, protein_mask)
            for decoder_name in decoded_x:
                if decoder_name == "vit_decoder":
                    vit_output = decoded_x[decoder_name]
                    decoded_coords = (
                        vit_output.get("protein_coords", vit_output.get("coords"))
                        if isinstance(vit_output, dict)
                        else vit_output
                    )
                    break

    sequence_logits = result["sequence_logits"]
    uses_33_token_vocab = sequence_logits.shape[-1] == 33

    if uses_33_token_vocab:
        predicted_sequence = convert_lobster_aa_tokenization_to_standard_aa(sequence_logits, device=device).squeeze(0)
    else:
        predicted_sequence = sequence_logits.argmax(dim=-1).squeeze(0)
        predicted_sequence[predicted_sequence > 20] = 20
        predicted_sequence = LOBSTER_TO_STANDARD[predicted_sequence.long()].to(device)

    return {
        "predicted_sequence": predicted_sequence,
        "decoded_coords": decoded_coords.squeeze(0) if decoded_coords is not None else None,
    }


def compute_aar(pred_seq: Tensor, gt_seq: Tensor, mask: Tensor) -> float:
    """Compute amino acid recovery."""
    mask = mask.bool()
    if mask.sum() == 0:
        return float("nan")
    pred_seq = pred_seq[mask]
    gt_seq = gt_seq[mask]
    return (pred_seq == gt_seq).float().mean().item()


def fold_with_esmfold(
    plm_fold,
    pred_seq_str: str,
    gt_coords: Tensor,
    protein_mask: Tensor,
    device: str,
    max_length: int,
) -> dict | None:
    """Fold predicted sequence with ESMFold and compute TM-score, RMSD, pLDDT, PAE vs GT.

    Returns dict with esmfold_tm_score, esmfold_rmsd, esmfold_plddt, esmfold_pae, or None on failure.
    """
    if plm_fold is None or len(pred_seq_str) > max_length:
        return None

    try:
        tokenized_input = plm_fold.tokenizer.encode_plus(
            pred_seq_str,
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].to(device)

        with torch.no_grad():
            outputs = plm_fold.model(tokenized_input)

        mask_bool = protein_mask.bool()
        ref_coords = gt_coords[mask_bool].unsqueeze(0)

        folded_metrics, _ = get_folded_structure_metrics(outputs, ref_coords, [pred_seq_str], mask=None, device=device)

        def _f(v):
            return v.item() if hasattr(v, "item") else float(v)

        return {
            "esmfold_tm_score": _f(folded_metrics["_tm_score"]),
            "esmfold_rmsd": _f(folded_metrics["_rmsd"]),
            "esmfold_plddt": _f(folded_metrics["_plddt"]),
            "esmfold_pae": _f(folded_metrics["_predicted_aligned_error"]),
        }
    except Exception as e:
        logger.warning(f"ESMFold folding failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate protein-ligand model on protein-only data (inverse folding, protein metrics only)",
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
        default="protein_ligand_inverse_folding_on_protein_only_results.csv",
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
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (None = all available)",
    )
    parser.add_argument(
        "--nsteps",
        type=int,
        default=50,
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
        default=0.2946377400416276,
        help="Temperature for sequence sampling",
    )
    parser.add_argument(
        "--temperature_struc",
        type=float,
        default=0.5872683450058442,
        help="Temperature for structure sampling",
    )
    parser.add_argument(
        "--temperature_ligand",
        type=float,
        default=0.818357063066881,
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
        default=40,
        help="Stochasticity for structure sampling",
    )
    parser.add_argument(
        "--stochasticity_ligand",
        type=int,
        default=40,
        help="Stochasticity for ligand sampling (unused in protein-only mode)",
    )
    parser.add_argument(
        "--inference_schedule_seq",
        type=str,
        default="LogInferenceSchedule",
        choices=list(INFERENCE_SCHEDULE_MAP.keys()),
        help="Inference schedule for sequence",
    )
    parser.add_argument(
        "--inference_schedule_struc",
        type=str,
        default="LinearInferenceSchedule",
        choices=list(INFERENCE_SCHEDULE_MAP.keys()),
        help="Inference schedule for structure",
    )
    parser.add_argument(
        "--inference_schedule_ligand_atom",
        type=str,
        default="LinearInferenceSchedule",
        choices=list(INFERENCE_SCHEDULE_MAP.keys()),
        help="Inference schedule for ligand atoms (unused in protein-only mode)",
    )
    parser.add_argument(
        "--inference_schedule_ligand_struc",
        type=str,
        default="LogInferenceSchedule",
        choices=list(INFERENCE_SCHEDULE_MAP.keys()),
        help="Inference schedule for ligand structure (unused in protein-only mode)",
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
        "--use_esmfold",
        action="store_true",
        help="Fold predicted sequences with ESMFold and compute TM-score, RMSD, pLDDT, PAE vs ground truth",
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

    pt_count = len(glob(args.data_dir)) if "*" in args.data_dir else len(glob(os.path.join(args.data_dir, "*.pt")))
    if pt_count == 0:
        logger.error(f"No .pt files found at {args.data_dir}")
        sys.exit(1)

    # Load model
    model = load_model(args.checkpoint, args.device)

    max_length = args.max_length
    if hasattr(model, "max_length") and model.max_length is not None:
        max_length = min(max_length, model.max_length)
        logger.info(f"Using max_length: {max_length}")

    plm_fold = None
    if args.use_esmfold:
        from lobster.model import LobsterPLMFold

        logger.info("Loading ESMFold for structure validation...")
        plm_fold = LobsterPLMFold(model_name="esmfold_v1", max_length=max_length)
        plm_fold.model.to(args.device)
        plm_fold.model.eval()

    samples = load_protein_only_structures(args.data_dir, args.num_samples, max_length, args.device)

    schedule_seq = INFERENCE_SCHEDULE_MAP[args.inference_schedule_seq]
    schedule_struc = INFERENCE_SCHEDULE_MAP[args.inference_schedule_struc]
    schedule_lig_atom = INFERENCE_SCHEDULE_MAP[args.inference_schedule_ligand_atom]
    schedule_lig_struc = INFERENCE_SCHEDULE_MAP[args.inference_schedule_ligand_struc]

    if args.structure_path:
        os.makedirs(args.structure_path, exist_ok=True)

    results = []
    for sample in tqdm(samples, desc="Evaluating inverse folding"):
        pdb_id = sample["pdb_id"]
        gt_seq = sample["protein_sequence"]
        gt_coords = sample["protein_coords"]
        protein_mask = sample["protein_mask"]

        if len(gt_seq) > max_length:
            logger.warning(f"Skipping {pdb_id}: length {len(gt_seq)} > {max_length}")
            continue

        try:
            pred_result = inverse_fold(
                model,
                sample,
                args.device,
                args.nsteps,
                decode_structure=args.decode_structure,
                temperature_seq=args.temperature_seq,
                temperature_struc=args.temperature_struc,
                temperature_ligand=args.temperature_ligand,
                stochasticity_seq=args.stochasticity_seq,
                stochasticity_struc=args.stochasticity_struc,
                stochasticity_ligand=args.stochasticity_ligand,
                inference_schedule_seq=schedule_seq,
                inference_schedule_struc=schedule_struc,
                inference_schedule_ligand_atom=schedule_lig_atom,
                inference_schedule_ligand_struc=schedule_lig_struc,
            )
        except Exception as e:
            logger.warning(f"Failed {pdb_id}: {e}")
            continue

        pred_seq = pred_result["predicted_sequence"]
        aar = compute_aar(pred_seq, gt_seq, protein_mask)
        pred_seq_str = "".join([STANDARD_AA[int(s)] if int(s) < 21 else "X" for s in pred_seq.cpu().tolist()])

        esmfold_metrics = {}
        if args.use_esmfold and plm_fold is not None:
            esmfold_result = fold_with_esmfold(plm_fold, pred_seq_str, gt_coords, protein_mask, args.device, max_length)
            if esmfold_result:
                esmfold_metrics = esmfold_result

        if args.structure_path:
            gt_seq_str = "".join([STANDARD_AA[int(s)] if int(s) < 21 else "X" for s in gt_seq.cpu().tolist()])
            with open(os.path.join(args.structure_path, f"{pdb_id}_sequences.fasta"), "w") as f:
                f.write(f">{pdb_id}_gt\n{gt_seq_str}\n>{pdb_id}_pred\n{pred_seq_str}\n")
            if args.save_gt_structure:
                writepdb(
                    os.path.join(args.structure_path, f"{pdb_id}_gt.pdb"),
                    gt_coords,
                    gt_seq,
                )
            if args.decode_structure and pred_result["decoded_coords"] is not None:
                writepdb(
                    os.path.join(args.structure_path, f"{pdb_id}_decoded.pdb"),
                    pred_result["decoded_coords"],
                    pred_seq,
                )

        results.append(
            {
                "pdb_id": pdb_id,
                "length": len(gt_seq),
                "aar": aar,
                **esmfold_metrics,
            }
        )

    df = pd.DataFrame(results)
    if len(df) == 0:
        logger.warning("No samples were successfully evaluated")
        sys.exit(1)

    df.to_csv(args.output, index=False)
    logger.info(f"Saved per-structure results to {args.output}")

    summary = {
        "mean_aar": df["aar"].mean(),
        "std_aar": df["aar"].std(),
        "n_samples": len(df),
    }
    if args.use_esmfold and "esmfold_tm_score" in df.columns:
        summary.update(
            {
                "mean_esmfold_tm_score": df["esmfold_tm_score"].mean(),
                "std_esmfold_tm_score": df["esmfold_tm_score"].std(),
                "mean_esmfold_rmsd": df["esmfold_rmsd"].mean(),
                "std_esmfold_rmsd": df["esmfold_rmsd"].std(),
                "mean_esmfold_plddt": df["esmfold_plddt"].mean(),
                "std_esmfold_plddt": df["esmfold_plddt"].std(),
                "mean_esmfold_pae": df["esmfold_pae"].mean(),
                "std_esmfold_pae": df["esmfold_pae"].std(),
            }
        )

    # Print summary
    print("\n" + "=" * 80)
    print("PROTEIN-LIGAND INVERSE FOLDING ON PROTEIN-ONLY DATA")
    print("=" * 80)
    print(f"\nDataset: {args.data_dir}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Samples evaluated: {summary['n_samples']}")

    print("\n--- Protein Metrics (No Ligand Context) ---")
    print(f"\n{'Metric':<30} {'Value':<20}")
    print("-" * 50)
    print(f"{'AAR (Amino Acid Recovery)':<30} {summary['mean_aar']:.2%} ± {summary['std_aar']:.2%}")
    if args.use_esmfold and "mean_esmfold_tm_score" in summary:
        print("\n--- ESMFold Designability (folded designed seq vs GT structure) ---")
        print(f"{'TM-Score':<30} {summary['mean_esmfold_tm_score']:.3f} ± {summary['std_esmfold_tm_score']:.3f}")
        print(f"{'RMSD (Å)':<30} {summary['mean_esmfold_rmsd']:.2f} ± {summary['std_esmfold_rmsd']:.2f}")
        print(f"{'pLDDT':<30} {summary['mean_esmfold_plddt']:.2f} ± {summary['std_esmfold_plddt']:.2f}")
        print(f"{'PAE (Å)':<30} {summary['mean_esmfold_pae']:.2f} ± {summary['std_esmfold_pae']:.2f}")
    print("=" * 80)

    if args.output_json:
        summary_json = {k: float(v) if hasattr(v, "item") else v for k, v in summary.items()}
        summary_json["checkpoint"] = args.checkpoint
        summary_json["data_dir"] = args.data_dir
        summary_json["nsteps"] = args.nsteps
        summary_json["max_length"] = max_length

        with open(args.output_json, "w") as f:
            json.dump(summary_json, f, indent=2)
        logger.info(f"Saved summary to {args.output_json}")

    logger.info("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
