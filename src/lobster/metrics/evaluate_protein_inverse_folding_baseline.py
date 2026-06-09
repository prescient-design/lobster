#!/usr/bin/env python
"""Standalone baseline evaluation script for protein-only inverse folding (sequence recovery).

Evaluates sequence recovery on proteins using a protein-only Gen-UME model.
This serves as a baseline comparison for protein-ligand models.

Usage:
    # Evaluate a Gen-UME protein-only checkpoint
    uv run python -m lobster.metrics.evaluate_protein_inverse_folding_baseline \
        --checkpoint /path/to/checkpoint.ckpt \
        --data_dir /path/to/pdbind/test/ \
        --output results.csv

    # With structure decoding
    uv run python -m lobster.metrics.evaluate_protein_inverse_folding_baseline \
        --checkpoint /path/to/checkpoint.ckpt \
        --data_dir /path/to/pdbind/test/ \
        --output results.csv \
        --structure_path ./structures/ \
        --decode_structure \
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
from torch import Tensor
from tqdm import tqdm

from lobster.model.latent_generator.io import writepdb
from lobster.model.latent_generator.utils.residue_constants import (
    convert_lobster_aa_tokenization_to_standard_aa,
)


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
    from lobster.model.leflur import LeFlurSequenceStructureEncoderLightningModule

    logger.info(f"Loading protein-only model from {checkpoint_path}")

    # Load checkpoint
    model = LeFlurSequenceStructureEncoderLightningModule.load_from_checkpoint(
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


class ProteinInverseFoldingBaselineEvaluator:
    """Evaluates inverse folding on proteins using a protein-only model.

    This evaluator serves as a baseline for protein-ligand inverse folding evaluation.
    It measures sequence recovery without any ligand context.

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
    decode_structure : bool
        Whether to decode and save predicted structures as PDB files (default: False).
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
        decode_structure: bool = False,
        save_gt_structure: bool = False,
    ):
        self.data_dir = data_dir
        self.pocket_distance_threshold = pocket_distance_threshold
        self.num_samples = num_samples
        self.nsteps = nsteps
        self.device = device
        self.max_length = max_length
        self.decode_structure = decode_structure
        self.save_gt_structure = save_gt_structure

        # Standard amino acid mapping (alphabetical order)
        self.standard_aa_map = {
            0: "A",
            1: "R",
            2: "N",
            3: "D",
            4: "C",
            5: "Q",
            6: "E",
            7: "G",
            8: "H",
            9: "I",
            10: "L",
            11: "K",
            12: "M",
            13: "F",
            14: "P",
            15: "S",
            16: "T",
            17: "W",
            18: "Y",
            19: "V",
            20: "X",
        }

        # Lobster amino acid mapping (for 21-token vocab model outputs)
        self.lobster_aa_map = {
            0: "L",
            1: "A",
            2: "G",
            3: "V",
            4: "S",
            5: "E",
            6: "R",
            7: "T",
            8: "I",
            9: "D",
            10: "P",
            11: "K",
            12: "Q",
            13: "F",
            14: "N",
            15: "Y",
            16: "M",
            17: "H",
            18: "W",
            19: "C",
            20: "X",
        }

        # Mapping from lobster tokenization to standard (alphabetical) tokenization
        self.lobster_to_standard = torch.tensor(
            [
                10,
                0,
                7,
                19,
                15,
                6,
                1,
                16,
                9,
                3,
                14,
                11,
                5,
                13,
                2,
                18,
                12,
                8,
                17,
                4,
                20,
            ],
            dtype=torch.long,
            device=device,
        )

        # Element vocabulary for pocket computation
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

    def inverse_fold(self, model, sample: dict) -> dict:
        """Run inverse folding on a protein sample.

        Parameters
        ----------
        model : LightningModule
            The Gen-UME protein-only model
        sample : dict
            Sample dictionary from load_test_set()

        Returns
        -------
        dict with:
            - predicted_sequence: Tensor [L]
            - sequence_logits: Tensor [L, vocab_size]
            - decoded_coords: Tensor [L, 3, 3] (if decode_structure=True)
        """
        protein_coords = sample["protein_coords"].unsqueeze(0).float()
        protein_mask = sample["protein_mask"].unsqueeze(0).float()
        protein_indices = sample["protein_indices"].unsqueeze(0).long()
        length = protein_coords.shape[1]

        # Generate sample (inverse folding mode)
        with torch.no_grad():
            result = model.generate_sample(
                length=length,
                num_samples=1,
                inverse_folding=True,
                nsteps=self.nsteps,
                input_structure_coords=protein_coords,
                input_mask=protein_mask,
                input_indices=protein_indices,
            )

            # Decode structure to coordinates (optional)
            decoded_coords = None
            if self.decode_structure:
                decoded_x = model.decode_structure(result, protein_mask)
                for decoder_name in decoded_x:
                    if "vit_decoder" == decoder_name:
                        vit_output = decoded_x[decoder_name]
                        if isinstance(vit_output, dict):
                            decoded_coords = vit_output.get("protein_coords", vit_output.get("coords"))
                        else:
                            decoded_coords = vit_output
                        break

        # Get predicted sequence
        sequence_logits = result["sequence_logits"]  # [1, L, vocab_size]
        uses_33_token_vocab = sequence_logits.shape[-1] == 33

        # Handle both 33-token and 21-token vocab formats
        if uses_33_token_vocab:
            predicted_sequence = convert_lobster_aa_tokenization_to_standard_aa(
                sequence_logits, device=sequence_logits.device
            ).squeeze(0)
        else:
            predicted_sequence = sequence_logits.argmax(dim=-1).squeeze(0)
            predicted_sequence[predicted_sequence > 20] = 20
            predicted_sequence = self.lobster_to_standard[predicted_sequence.long()]

        return {
            "predicted_sequence": predicted_sequence,
            "sequence_logits": sequence_logits.squeeze(0),
            "decoded_coords": decoded_coords.squeeze(0) if decoded_coords is not None else None,
        }

    def compute_aar(
        self,
        predicted_seq: Tensor,
        ground_truth_seq: Tensor,
        mask: Tensor | None = None,
    ) -> float:
        """Compute amino acid recovery rate."""
        if mask is not None:
            mask = mask.bool()
            if mask.sum() == 0:
                return float("nan")
            predicted_seq = predicted_seq[mask]
            ground_truth_seq = ground_truth_seq[mask]

        if len(predicted_seq) == 0:
            return float("nan")

        return (predicted_seq == ground_truth_seq).float().mean().item()

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

        for sample in tqdm(samples, desc="Evaluating inverse folding (baseline)"):
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

            # Run inverse folding
            pred_result = self.inverse_fold(model, sample)
            pred_seq = pred_result["predicted_sequence"]

            # Save structures and sequences if requested
            if structure_path:
                # Save sequences as FASTA
                gt_seq_str = self.sequence_to_string(gt_seq)
                pred_seq_str = self.sequence_to_string(pred_seq)

                fasta_path = os.path.join(structure_path, f"{pdb_id}_sequences.fasta")
                with open(fasta_path, "w") as f:
                    f.write(f">{pdb_id}_ground_truth\n{gt_seq_str}\n")
                    f.write(f">{pdb_id}_predicted_baseline\n{pred_seq_str}\n")

                if self.save_gt_structure:
                    gt_pdb_path = os.path.join(structure_path, f"{pdb_id}_gt_protein.pdb")
                    writepdb(gt_pdb_path, gt_coords, gt_seq)

                if self.decode_structure and pred_result["decoded_coords"] is not None:
                    pred_pdb_path = os.path.join(structure_path, f"{pdb_id}_decoded_baseline.pdb")
                    writepdb(pred_pdb_path, pred_result["decoded_coords"], pred_seq)

            # Compute metrics
            result = {
                "pdb_id": pdb_id,
                "length": len(gt_seq),
                "n_pocket_residues": int(pocket_mask.sum().item()),
                "n_nonpocket_residues": int(non_pocket_mask.sum().item()),
                "aar_overall": self.compute_aar(pred_seq, gt_seq, protein_mask),
                "aar_pocket": self.compute_aar(pred_seq, gt_seq, pocket_mask),
                "aar_nonpocket": self.compute_aar(pred_seq, gt_seq, non_pocket_mask),
            }
            results.append(result)

        if skipped_samples:
            logger.info(f"Skipped {len(skipped_samples)} samples due to length > {self.max_length}")

        results_df = pd.DataFrame(results)

        if len(results_df) == 0:
            logger.warning("No samples were successfully evaluated")
            return {"results_df": results_df, "summary": {}}

        summary = {
            "mean_aar_overall": results_df["aar_overall"].mean(),
            "std_aar_overall": results_df["aar_overall"].std(),
            "mean_aar_pocket": results_df["aar_pocket"].mean(),
            "std_aar_pocket": results_df["aar_pocket"].std(),
            "mean_aar_nonpocket": results_df["aar_nonpocket"].mean(),
            "std_aar_nonpocket": results_df["aar_nonpocket"].std(),
            "n_samples": len(results_df),
            "mean_pocket_size": results_df["n_pocket_residues"].mean(),
        }

        return {"results_df": results_df, "summary": summary}

    def sequence_to_string(self, seq_tensor: Tensor) -> str:
        """Convert sequence tensor (in standard format) to string."""
        return "".join([self.standard_aa_map.get(int(s), "X") for s in seq_tensor.cpu().tolist()])


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate protein-only inverse folding (baseline for protein-ligand comparison)",
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
        "--output", type=str, default="protein_inverse_folding_baseline_results.csv", help="Output CSV file"
    )
    parser.add_argument("--output_json", type=str, default=None, help="Output JSON file for summary statistics")
    parser.add_argument("--structure_path", type=str, default=None, help="Directory to save sequences and structures")
    parser.add_argument("--pocket_threshold", type=float, default=5.0, help="Distance threshold (Å) for binding pocket")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to evaluate")
    parser.add_argument("--nsteps", type=int, default=100, help="Number of diffusion steps")
    parser.add_argument("--max_length", type=int, default=768, help="Maximum protein sequence length")
    parser.add_argument(
        "--decode_structure", action="store_true", help="Decode and save predicted structures as PDB files"
    )
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
    evaluator = ProteinInverseFoldingBaselineEvaluator(
        data_dir=args.data_dir,
        pocket_distance_threshold=args.pocket_threshold,
        num_samples=args.num_samples,
        nsteps=args.nsteps,
        device=args.device,
        max_length=max_length,
        decode_structure=args.decode_structure,
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
    print("PROTEIN INVERSE FOLDING BASELINE RESULTS")
    print("=" * 70)
    print(f"\nDataset: {args.data_dir}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Pocket threshold: {args.pocket_threshold} Å")
    print(f"Samples evaluated: {summary['n_samples']}")
    print(f"Mean pocket size: {summary['mean_pocket_size']:.1f} residues")

    print("\n--- Sequence Recovery (AAR) - Protein-Only Model, No Ligand Context ---")
    print(f"\n{'Region':<25} {'AAR':<20}")
    print("-" * 45)
    print(f"{'Overall':<25} {summary['mean_aar_overall']:.2%} ± {summary['std_aar_overall']:.2%}")
    print(f"{'Pocket':<25} {summary['mean_aar_pocket']:.2%} ± {summary['std_aar_pocket']:.2%}")
    print(f"{'Non-pocket':<25} {summary['mean_aar_nonpocket']:.2%} ± {summary['std_aar_nonpocket']:.2%}")
    print("=" * 70)

    print("\nNote: This is a protein-only baseline. Compare with protein-ligand model")
    print("      to see if ligand context improves sequence recovery in the pocket.")

    if args.output_json:
        summary_json = {k: float(v) if hasattr(v, "item") else v for k, v in summary.items()}
        summary_json["checkpoint"] = args.checkpoint
        summary_json["data_dir"] = args.data_dir
        summary_json["pocket_threshold"] = args.pocket_threshold
        summary_json["nsteps"] = args.nsteps
        summary_json["max_length"] = max_length
        summary_json["model_type"] = "protein_only_baseline"

        with open(args.output_json, "w") as f:
            json.dump(summary_json, f, indent=2)
        logger.info(f"Saved summary to {args.output_json}")

    logger.info("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
