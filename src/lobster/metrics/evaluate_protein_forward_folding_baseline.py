#!/usr/bin/env python
"""Standalone baseline evaluation script for protein-only forward folding (structure prediction).

Evaluates structure prediction (TM-score, RMSD) on proteins using a protein-only Gen-UME model.
This serves as a baseline comparison for protein-ligand models.

Usage:
    # Evaluate a Gen-UME protein-only checkpoint
    uv run python -m lobster.metrics.evaluate_protein_forward_folding_baseline \
        --checkpoint /path/to/checkpoint.ckpt \
        --data_dir /path/to/pdbind/test/ \
        --output results.csv

    # With structure saving
    uv run python -m lobster.metrics.evaluate_protein_forward_folding_baseline \
        --checkpoint /path/to/checkpoint.ckpt \
        --data_dir /path/to/pdbind/test/ \
        --output results.csv \
        --structure_path ./structures/ \
        --save_structures \
        --save_gt_structure
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

from lobster.metrics import align_and_compute_rmsd
from lobster.model.latent_generator.io import writepdb
from lobster.model.latent_generator.utils.residue_constants import restype_order_with_x_inv
from lobster.transforms._structure_transforms import AminoAcidTokenizerTransform


def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load a Gen-UME protein-only model from checkpoint.

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
    from lobster.model.gen_ume import UMESequenceStructureEncoderLightningModule

    logger.info(f"Loading protein-only model from {checkpoint_path}")

    # Load checkpoint
    model = UMESequenceStructureEncoderLightningModule.load_from_checkpoint(
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
    model.max_length = max_length

    logger.info(f"Model loaded successfully. Max length: {max_length}")
    return model


class ProteinForwardFoldingBaselineEvaluator:
    """Evaluates forward folding on proteins using a protein-only model.

    This evaluator serves as a baseline for protein-ligand forward folding evaluation.
    It measures structure prediction quality without any ligand context.

    Parameters
    ----------
    data_dir : str
        Path to PDBBind test directory containing *_protein.pt and *_ligand.pt pairs
    pocket_distance_threshold : float
        Distance threshold (Å) for defining binding pocket residues
    num_samples : int, optional
        Limit number of samples to evaluate (None = all)
    nsteps : int
        Number of diffusion steps for generation
    device : str
        Device for computation
    max_length : int
        Maximum protein sequence length to process (default: 512).
    temperature_seq : float
        Temperature for sequence sampling
    temperature_struc : float
        Temperature for structure sampling
    save_structures : bool
        Whether to save predicted structures as PDB files (default: False).
    save_gt_structure : bool
        Whether to save ground truth structures as PDB files (default: False).
    """

    def __init__(
        self,
        data_dir: str,
        pocket_distance_threshold: float = 5.0,
        num_samples: int | None = None,
        nsteps: int = 100,
        device: str = "cuda",
        max_length: int = 512,
        temperature_seq: float = 0.5,
        temperature_struc: float = 0.5,
        save_structures: bool = False,
        save_gt_structure: bool = False,
    ):
        self.data_dir = data_dir
        self.pocket_distance_threshold = pocket_distance_threshold
        self.num_samples = num_samples
        self.nsteps = nsteps
        self.device = device
        self.max_length = max_length
        self.temperature_seq = temperature_seq
        self.temperature_struc = temperature_struc
        self.save_structures = save_structures
        self.save_gt_structure = save_gt_structure

        # Initialize tokenizer transform for sequence conversion
        self.tokenizer_transform = AminoAcidTokenizerTransform(max_length=max_length)

        # Element vocabulary for ligand (used for pocket computation)
        self.element_to_idx = {
            "PAD": 0,
            "MASK": 1,
            "UNK": 2,
            "C": 3,
            "N": 4,
            "O": 5,
            "S": 6,
            "P": 7,
            "H": 8,
            "F": 9,
            "Cl": 10,
            "Br": 11,
            "I": 12,
            "Fe": 13,
            "Zn": 14,
            "Mg": 15,
            "Ca": 16,
            "Mn": 17,
            "Cu": 18,
            "B": 19,
            "Si": 20,
            "Se": 21,
            "Co": 22,
            "Ni": 23,
            "Bi": 24,
        }

    def load_test_set(self) -> list[dict]:
        """Load PDBBind test protein-ligand pairs.

        Returns list of dicts with protein and ligand data (ligand used only for pocket definition).
        """
        protein_files = sorted(glob(os.path.join(self.data_dir, "*_protein.pt")))

        if not protein_files:
            raise ValueError(f"No protein files found in {self.data_dir}")

        if self.num_samples is not None:
            protein_files = protein_files[: self.num_samples]

        logger.info(f"Loading {len(protein_files)} protein-ligand pairs from {self.data_dir}")

        samples = []
        for pf in tqdm(protein_files, desc="Loading samples"):
            pdb_id = os.path.basename(pf).replace("_protein.pt", "")
            ligand_file = pf.replace("_protein.pt", "_ligand.pt")

            if not os.path.exists(ligand_file):
                logger.warning(f"Missing ligand file for {pdb_id}, skipping")
                continue

            protein_data = torch.load(pf, weights_only=False, map_location=self.device)
            ligand_data = torch.load(ligand_file, weights_only=False, map_location=self.device)

            protein_coords = protein_data.get("coords_res", protein_data.get("coords"))
            protein_sequence = protein_data.get("sequence")

            if protein_coords is None or protein_sequence is None:
                logger.warning(f"Missing protein data for {pdb_id}, skipping")
                continue

            protein_mask = protein_data.get("mask", torch.ones(protein_coords.shape[0], device=self.device))
            protein_indices = protein_data.get("indices", torch.arange(protein_coords.shape[0], device=self.device))

            # Load ligand coords for pocket computation
            ligand_coords = ligand_data.get("atom_coords", ligand_data.get("coords", ligand_data.get("ligand_coords")))
            if ligand_coords is None:
                logger.warning(f"Missing ligand coordinates for {pdb_id}, skipping")
                continue

            samples.append(
                {
                    "pdb_id": pdb_id,
                    "protein_coords": protein_coords,
                    "protein_sequence": protein_sequence,
                    "protein_mask": protein_mask,
                    "protein_indices": protein_indices,
                    "ligand_coords": ligand_coords,  # Only for pocket computation
                }
            )

        logger.info(f"Loaded {len(samples)} valid samples")
        return samples

    def compute_binding_pocket(
        self,
        protein_coords: Tensor,
        ligand_coords: Tensor,
        protein_mask: Tensor | None = None,
    ) -> Tensor:
        """Compute pocket mask based on distance to ligand."""
        if protein_coords.dim() == 3:
            ca_coords = protein_coords[:, 1, :]
        else:
            ca_coords = protein_coords

        distances = torch.cdist(ca_coords.unsqueeze(0), ligand_coords.unsqueeze(0)).squeeze(0)
        min_distances = distances.min(dim=1).values
        pocket_mask = min_distances < self.pocket_distance_threshold

        if protein_mask is not None:
            pocket_mask = pocket_mask & protein_mask.bool()

        return pocket_mask

    def forward_fold(self, model, sample: dict) -> dict:
        """Run forward folding on a protein sample.

        Parameters
        ----------
        model : LightningModule
            The Gen-UME protein-only model
        sample : dict
            Sample dictionary from load_test_set()

        Returns
        -------
        dict with:
            - predicted_coords: Tensor [L, 3, 3] (N, CA, C backbone)
            - structure_tokens: Tensor [L]
        """
        protein_mask = sample["protein_mask"].unsqueeze(0).float()
        protein_indices = sample["protein_indices"].unsqueeze(0).long()
        length = int(protein_mask.sum().item())

        # Tokenize sequence for forward folding
        gt_seq = sample["protein_sequence"]
        tokenized_data = self.tokenizer_transform({"sequence": gt_seq.cpu()})
        tokenized_seq = tokenized_data["sequence"].to(self.device).unsqueeze(0)

        # Generate sample (forward folding mode)
        with torch.no_grad():
            result = model.generate_sample(
                length=length,
                num_samples=1,
                forward_folding=True,
                nsteps=self.nsteps,
                temperature_seq=self.temperature_seq,
                temperature_struc=self.temperature_struc,
                input_sequence_tokens=tokenized_seq,
                input_mask=protein_mask,
                input_indices=protein_indices,
            )

        # Decode structure
        decoded_x = model.decode_structure(result, protein_mask)

        # Extract coordinates
        predicted_coords = None
        for decoder_name in decoded_x:
            if "vit_decoder" == decoder_name:
                vit_output = decoded_x[decoder_name]
                if isinstance(vit_output, dict):
                    predicted_coords = vit_output.get("protein_coords", vit_output.get("coords"))
                else:
                    predicted_coords = vit_output
                break

        if predicted_coords is None:
            raise RuntimeError("No vit_decoder found in decoded structures")

        structure_tokens = result.get("generated_struc_tokens")

        return {
            "predicted_coords": predicted_coords.squeeze(0),
            "structure_tokens": structure_tokens.squeeze(0) if structure_tokens is not None else None,
        }

    def compute_tm_score(
        self,
        pred_coords: Tensor,
        gt_coords: Tensor,
        sequence: Tensor,
        mask: Tensor | None = None,
    ) -> float:
        """Compute TM-score between predicted and ground truth structures."""
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

    def compute_rmsd(
        self,
        pred_coords: Tensor,
        gt_coords: Tensor,
        mask: Tensor | None = None,
    ) -> float:
        """Compute RMSD between predicted and ground truth structures."""
        if mask is not None:
            mask = mask.bool()
            pred_coords = pred_coords[mask]
            gt_coords = gt_coords[mask]

        if len(pred_coords) == 0:
            return float("nan")

        rmsd = align_and_compute_rmsd(
            coords1=pred_coords.detach(),
            coords2=gt_coords.detach(),
            mask=None,
            return_aligned=False,
            device=pred_coords.device,
        )
        return float(rmsd)

    def evaluate(self, model, samples: list[dict] | None = None, structure_path: str | None = None) -> dict:
        """Run full evaluation on PDBBind test set."""
        model.eval()
        model.to(self.device)

        if samples is None:
            samples = self.load_test_set()

        if structure_path:
            os.makedirs(structure_path, exist_ok=True)

        results = []
        skipped_samples = []

        for sample in tqdm(samples, desc="Evaluating forward folding (baseline)"):
            pdb_id = sample["pdb_id"]
            gt_seq = sample["protein_sequence"]
            gt_coords = sample["protein_coords"]
            protein_mask = sample["protein_mask"]

            protein_length = len(gt_seq)

            if protein_length > self.max_length:
                logger.warning(
                    f"Skipping {pdb_id}: protein length {protein_length} exceeds max_length {self.max_length}"
                )
                skipped_samples.append({"pdb_id": pdb_id, "protein_length": protein_length})
                continue

            # Compute binding pocket (using ligand coords)
            pocket_mask = self.compute_binding_pocket(gt_coords, sample["ligand_coords"], protein_mask)
            non_pocket_mask = protein_mask.bool() & ~pocket_mask

            # Run forward folding
            pred_result = self.forward_fold(model, sample)
            pred_coords = pred_result["predicted_coords"]

            # Save structures if requested
            if structure_path:
                if self.save_gt_structure:
                    gt_pdb_path = os.path.join(structure_path, f"{pdb_id}_gt_protein.pdb")
                    writepdb(gt_pdb_path, gt_coords, gt_seq)

                if self.save_structures:
                    pred_pdb_path = os.path.join(structure_path, f"{pdb_id}_pred_baseline.pdb")
                    writepdb(pred_pdb_path, pred_coords.detach(), gt_seq)

            # Compute metrics
            result = {
                "pdb_id": pdb_id,
                "length": len(gt_seq),
                "n_pocket_residues": int(pocket_mask.sum().item()),
                "n_nonpocket_residues": int(non_pocket_mask.sum().item()),
                "tm_score": self.compute_tm_score(pred_coords, gt_coords, gt_seq, protein_mask),
                "rmsd_overall": self.compute_rmsd(pred_coords, gt_coords, protein_mask),
                "rmsd_pocket": self.compute_rmsd(pred_coords, gt_coords, pocket_mask),
                "rmsd_nonpocket": self.compute_rmsd(pred_coords, gt_coords, non_pocket_mask),
            }
            results.append(result)

        if skipped_samples:
            logger.info(f"Skipped {len(skipped_samples)} samples due to length > {self.max_length}")

        results_df = pd.DataFrame(results)

        if len(results_df) == 0:
            logger.warning("No samples were successfully evaluated")
            return {"results_df": results_df, "summary": {}}

        summary = {
            "mean_tm_score": results_df["tm_score"].mean(),
            "std_tm_score": results_df["tm_score"].std(),
            "mean_rmsd_overall": results_df["rmsd_overall"].mean(),
            "std_rmsd_overall": results_df["rmsd_overall"].std(),
            "mean_rmsd_pocket": results_df["rmsd_pocket"].mean(),
            "std_rmsd_pocket": results_df["rmsd_pocket"].std(),
            "mean_rmsd_nonpocket": results_df["rmsd_nonpocket"].mean(),
            "std_rmsd_nonpocket": results_df["rmsd_nonpocket"].std(),
            "n_samples": len(results_df),
            "mean_pocket_size": results_df["n_pocket_residues"].mean(),
        }

        return {"results_df": results_df, "summary": summary}


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate protein-only forward folding (baseline for protein-ligand comparison)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.ckpt file)")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/cv/data/ai4dd/data2/lisanzas/pdb_bind_12_15_25/test/",
        help="Path to PDBBind test directory",
    )
    parser.add_argument(
        "--output", type=str, default="protein_forward_folding_baseline_results.csv", help="Output CSV file"
    )
    parser.add_argument("--output_json", type=str, default=None, help="Output JSON file for summary statistics")
    parser.add_argument("--structure_path", type=str, default=None, help="Directory to save structures (PDB)")
    parser.add_argument("--pocket_threshold", type=float, default=5.0, help="Distance threshold (Å) for binding pocket")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to evaluate")
    parser.add_argument("--nsteps", type=int, default=100, help="Number of diffusion steps")
    parser.add_argument("--max_length", type=int, default=768, help="Maximum protein sequence length")
    parser.add_argument("--temperature_seq", type=float, default=0.5, help="Temperature for sequence sampling")
    parser.add_argument("--temperature_struc", type=float, default=0.5, help="Temperature for structure sampling")
    parser.add_argument("--save_structures", action="store_true", help="Save predicted structures as PDB files")
    parser.add_argument("--save_gt_structure", action="store_true", help="Save ground truth structures as PDB files")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")

    args = parser.parse_args()

    # Skip existence check for S3 paths (they're handled by the model loader)
    if not args.checkpoint.startswith("s3://") and not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    if not os.path.exists(args.data_dir):
        logger.error(f"Data directory not found: {args.data_dir}")
        sys.exit(1)

    # Load model
    model = load_model(args.checkpoint, args.device)

    max_length = args.max_length
    if hasattr(model, "max_length") and model.max_length is not None:
        max_length = min(max_length, model.max_length)
        logger.info(f"Using max_length: {max_length}")

    # Create evaluator
    evaluator = ProteinForwardFoldingBaselineEvaluator(
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
    results = evaluator.evaluate(model=model, samples=samples, structure_path=args.structure_path)

    results_df = results["results_df"]
    summary = results["summary"]

    results_df.to_csv(args.output, index=False)
    logger.info(f"Saved per-structure results to {args.output}")

    # Print summary
    print("\n" + "=" * 70)
    print("PROTEIN FORWARD FOLDING BASELINE RESULTS")
    print("=" * 70)
    print(f"\nDataset: {args.data_dir}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Pocket threshold: {args.pocket_threshold} Å")
    print(f"Samples evaluated: {summary['n_samples']}")
    print(f"Mean pocket size: {summary['mean_pocket_size']:.1f} residues")

    print("\n--- Results (Protein-Only Model, No Ligand Context) ---")
    print(f"\n{'Metric':<25} {'Value':<20}")
    print("-" * 45)
    print(f"{'TM-Score':<25} {summary['mean_tm_score']:.3f} ± {summary['std_tm_score']:.3f}")
    print(f"{'RMSD Overall (Å)':<25} {summary['mean_rmsd_overall']:.2f} ± {summary['std_rmsd_overall']:.2f}")
    print(f"{'RMSD Pocket (Å)':<25} {summary['mean_rmsd_pocket']:.2f} ± {summary['std_rmsd_pocket']:.2f}")
    print(f"{'RMSD Non-pocket (Å)':<25} {summary['mean_rmsd_nonpocket']:.2f} ± {summary['std_rmsd_nonpocket']:.2f}")
    print("=" * 70)

    print("\nNote: This is a protein-only baseline. Compare with protein-ligand model")
    print("      to see if ligand context improves structure prediction.")

    if args.output_json:
        summary_json = {k: float(v) if hasattr(v, "item") else v for k, v in summary.items()}
        summary_json["checkpoint"] = args.checkpoint
        summary_json["data_dir"] = args.data_dir
        summary_json["pocket_threshold"] = args.pocket_threshold
        summary_json["nsteps"] = args.nsteps
        summary_json["max_length"] = max_length
        summary_json["temperature_seq"] = args.temperature_seq
        summary_json["temperature_struc"] = args.temperature_struc
        summary_json["model_type"] = "protein_only_baseline"

        with open(args.output_json, "w") as f:
            json.dump(summary_json, f, indent=2)
        logger.info(f"Saved summary to {args.output_json}")

    logger.info("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
