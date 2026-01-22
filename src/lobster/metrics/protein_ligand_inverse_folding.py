"""Protein-Ligand Inverse Folding Evaluator.

Evaluates inverse folding on protein-ligand complexes with and without ligand context.
Can be used as a standalone evaluator or within a callback during training.

Key Question: Does providing ligand context improve sequence recovery for binding pocket residues?
"""

import os
from glob import glob
from typing import TYPE_CHECKING

import pandas as pd
import torch
from loguru import logger
from torch import Tensor
from tqdm import tqdm

from lobster.model.latent_generator.utils.residue_constants import (
    convert_lobster_aa_tokenization_to_standard_aa,
)

if TYPE_CHECKING:
    from lightning import LightningModule


class ProteinLigandInverseFoldingEvaluator:
    """Evaluates inverse folding on protein-ligand complexes with/without ligand context.

    This evaluator compares two modes:
    1. Protein-only: Provide only protein structure, predict sequence
    2. Protein+Ligand: Provide protein structure + ligand, predict sequence

    Tracks metrics separately for:
    - Overall sequence recovery
    - Binding pocket residues (within distance threshold of ligand)
    - Non-pocket residues

    Can be used:
    - As standalone evaluation script
    - Within callback during training

    Parameters
    ----------
    data_dir : str
        Path to PDBBind test directory containing *_protein.pt and *_ligand.pt pairs
    pocket_distance_threshold : float
        Distance threshold (Å) for defining binding pocket residues
    num_samples : int, optional
        Limit number of samples to evaluate (None = all)
    num_designs : int
        Number of designs per structure
    nsteps : int
        Number of diffusion steps for generation
    device : str
        Device for computation
    max_length : int
        Maximum combined sequence length (protein + ligand) to process (default: 512).
        Samples exceeding this length will be skipped.
    """

    def __init__(
        self,
        data_dir: str,
        pocket_distance_threshold: float = 5.0,
        num_samples: int | None = None,
        num_designs: int = 1,
        nsteps: int = 100,
        device: str = "cuda",
        max_length: int = 512,
    ):
        self.data_dir = data_dir
        self.pocket_distance_threshold = pocket_distance_threshold
        self.num_samples = num_samples
        self.num_designs = num_designs
        self.nsteps = nsteps
        self.device = device
        self.max_length = max_length

        # Amino acid mapping (lobster tokenization)
        self.aa_map = {
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

        # Element vocabulary (ELEMENT_VOCAB_EXTENDED from residue_constants)
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

    def _atom_names_to_indices(self, atom_names: list) -> Tensor:
        """Convert atom names (e.g., ['C1', 'N2', 'O3']) to element indices."""
        indices = []
        for name in atom_names:
            # Extract element symbol (first 1-2 characters, handling cases like 'Cl', 'Br')
            if len(name) >= 2 and name[:2] in self.element_to_idx:
                elem = name[:2]
            elif name[0] in self.element_to_idx:
                elem = name[0]
            else:
                # Try just the first character uppercase
                elem = name[0].upper()

            idx = self.element_to_idx.get(elem, 2)  # 2 = UNK
            indices.append(idx)

        return torch.tensor(indices, dtype=torch.long, device=self.device)

    def load_test_set(self) -> list[dict]:
        """Load PDBBind test protein-ligand pairs.

        Returns list of dicts with:
        - pdb_id: str
        - protein_coords: Tensor [L, 3, 3]  # N, CA, C backbone
        - protein_sequence: Tensor [L]
        - protein_mask: Tensor [L]
        - protein_indices: Tensor [L]
        - ligand_coords: Tensor [N_atoms, 3]
        - ligand_atom_types: Tensor [N_atoms]
        - ligand_mask: Tensor [N_atoms]
        - ligand_indices: Tensor [N_atoms]
        - bond_matrix: Tensor [N_atoms, N_atoms] (if available)
        """
        # Find protein-ligand pairs
        protein_files = sorted(glob(os.path.join(self.data_dir, "*_protein.pt")))

        if not protein_files:
            raise ValueError(f"No protein files found in {self.data_dir}")

        # Limit samples if specified
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

            # Extract protein data
            protein_coords = protein_data.get("coords_res", protein_data.get("coords"))
            protein_sequence = protein_data.get("sequence")

            if protein_coords is None:
                logger.warning(f"Missing protein coordinates for {pdb_id}, skipping")
                continue

            protein_mask = protein_data.get("mask", torch.ones(protein_coords.shape[0], device=self.device))
            protein_indices = protein_data.get("indices", torch.arange(protein_coords.shape[0], device=self.device))

            # Extract ligand data - handle different key names
            # PDBBind uses: atom_coords, atom_names, atom_indices
            ligand_coords = ligand_data.get("atom_coords", ligand_data.get("coords", ligand_data.get("ligand_coords")))

            if ligand_coords is None:
                logger.warning(f"Missing ligand coordinates for {pdb_id}, skipping")
                continue

            # Handle atom types - may be a list of names or tensor of indices
            atom_names = ligand_data.get("atom_names")
            if atom_names is not None and isinstance(atom_names, list):
                # Convert atom names to element indices
                ligand_atom_types = self._atom_names_to_indices(atom_names)
            else:
                ligand_atom_types = ligand_data.get(
                    "element_indices",
                    ligand_data.get(
                        "ligand_element_indices",
                        torch.full((ligand_coords.shape[0],), 3, dtype=torch.long, device=self.device),
                    ),  # Default to carbon (3)
                )

            ligand_mask = ligand_data.get(
                "mask", ligand_data.get("ligand_mask", torch.ones(ligand_coords.shape[0], device=self.device))
            )
            ligand_indices = ligand_data.get(
                "atom_indices",
                ligand_data.get(
                    "indices",
                    ligand_data.get("ligand_indices", torch.arange(ligand_coords.shape[0], device=self.device)),
                ),
            )
            bond_matrix = ligand_data.get("bond_matrix")

            if protein_sequence is None:
                logger.warning(f"Missing sequence for {pdb_id}, skipping")
                continue

            samples.append(
                {
                    "pdb_id": pdb_id,
                    "protein_coords": protein_coords,
                    "protein_sequence": protein_sequence,
                    "protein_mask": protein_mask,
                    "protein_indices": protein_indices,
                    "ligand_coords": ligand_coords,
                    "ligand_atom_types": ligand_atom_types,
                    "ligand_mask": ligand_mask,
                    "ligand_indices": ligand_indices,
                    "bond_matrix": bond_matrix,
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
        """Compute pocket mask based on distance to ligand.

        A residue is considered part of the binding pocket if any of its
        backbone atoms (N, CA, C) are within the threshold distance of
        any ligand heavy atom.

        Parameters
        ----------
        protein_coords : Tensor
            [L, 3, 3] or [L, 3] backbone coordinates
        ligand_coords : Tensor
            [N_atoms, 3] ligand atom coordinates
        protein_mask : Tensor, optional
            [L] valid residue mask

        Returns
        -------
        pocket_mask : Tensor
            [L] boolean mask, True for pocket residues
        """
        # Handle different coordinate formats
        if protein_coords.dim() == 3:
            # [L, 3, 3] - use CA atoms (index 1)
            ca_coords = protein_coords[:, 1, :]  # [L, 3]
        else:
            # [L, 3] - already CA-like
            ca_coords = protein_coords

        # Compute pairwise distances between CA atoms and ligand atoms
        # ca_coords: [L, 3], ligand_coords: [N_atoms, 3]
        # distances: [L, N_atoms]
        distances = torch.cdist(ca_coords.unsqueeze(0), ligand_coords.unsqueeze(0)).squeeze(0)

        # Min distance from each residue to any ligand atom
        min_distances = distances.min(dim=1).values  # [L]

        # Pocket mask: residues within threshold
        pocket_mask = min_distances < self.pocket_distance_threshold

        # Apply valid mask if provided
        if protein_mask is not None:
            pocket_mask = pocket_mask & protein_mask.bool()

        return pocket_mask

    def inverse_fold(
        self,
        model: "LightningModule",
        sample: dict,
        include_ligand: bool,
    ) -> dict:
        """Run inverse folding with or without ligand context.

        Parameters
        ----------
        model : LightningModule
            The Gen-UME protein-ligand model
        sample : dict
            Sample dictionary from load_test_set()
        include_ligand : bool
            Whether to include ligand context

        Returns
        -------
        dict with:
            - predicted_sequence: Tensor [L]
            - sequence_logits: Tensor [L, vocab_size]
        """
        # Prepare protein inputs - ensure proper dtype
        protein_coords = sample["protein_coords"].unsqueeze(0).float()
        protein_mask = sample["protein_mask"].unsqueeze(0).float()
        # Indices must be long (int64) for indexing operations
        protein_indices = sample["protein_indices"].unsqueeze(0).long()
        length = protein_coords.shape[1]

        # Prepare ligand inputs if needed
        ligand_mask = None
        ligand_atom_tokens = None
        ligand_structure_tokens = None
        bond_matrix = None
        num_atoms = 0

        if include_ligand:
            ligand_coords = sample["ligand_coords"].unsqueeze(0).float()
            ligand_mask = sample["ligand_mask"].unsqueeze(0).float()
            ligand_indices = sample["ligand_indices"].unsqueeze(0).long()
            ligand_atom_tokens = sample["ligand_atom_types"].unsqueeze(0).long()
            num_atoms = ligand_coords.shape[1]

            # Encode ligand structure to tokens
            with torch.no_grad():
                ligand_structure_tokens, _ = model.encode_ligand_structure(ligand_coords, ligand_mask, ligand_indices)

            bond_matrix = sample.get("bond_matrix")
            if bond_matrix is not None:
                bond_matrix = bond_matrix.unsqueeze(0).long()

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
                # Ligand context
                generate_ligand=include_ligand,
                num_atoms=num_atoms if include_ligand else 0,
                input_ligand_atom_tokens=ligand_atom_tokens,
                input_ligand_structure_tokens=ligand_structure_tokens,
                input_bond_matrix=bond_matrix,
            )

        # Get predicted sequence
        sequence_logits = result["sequence_logits"]  # [1, L, vocab_size]

        # Handle both 33-token and 21-token vocab formats
        if sequence_logits.shape[-1] == 33:
            predicted_sequence = convert_lobster_aa_tokenization_to_standard_aa(
                sequence_logits, device=sequence_logits.device
            ).squeeze(0)  # [L]
        else:
            predicted_sequence = sequence_logits.argmax(dim=-1).squeeze(0)  # [L]
            predicted_sequence[predicted_sequence > 21] = 20

        return {
            "predicted_sequence": predicted_sequence,
            "sequence_logits": sequence_logits.squeeze(0),
        }

    def compute_aar(
        self,
        predicted_seq: Tensor,
        ground_truth_seq: Tensor,
        mask: Tensor | None = None,
    ) -> float:
        """Compute amino acid recovery rate.

        Parameters
        ----------
        predicted_seq : Tensor
            [L] predicted sequence tokens
        ground_truth_seq : Tensor
            [L] ground truth sequence tokens
        mask : Tensor, optional
            [L] boolean mask for positions to include

        Returns
        -------
        float
            Amino acid recovery rate (0-1)
        """
        if mask is not None:
            mask = mask.bool()
            if mask.sum() == 0:
                return float("nan")
            predicted_seq = predicted_seq[mask]
            ground_truth_seq = ground_truth_seq[mask]

        if len(predicted_seq) == 0:
            return float("nan")

        return (predicted_seq == ground_truth_seq).float().mean().item()

    def evaluate(
        self,
        model: "LightningModule",
        samples: list[dict] | None = None,
    ) -> dict:
        """Run full evaluation on PDBBind test set.

        Parameters
        ----------
        model : LightningModule
            The Gen-UME protein-ligand model
        samples : list[dict], optional
            Pre-loaded samples (will load if not provided)

        Returns
        -------
        dict with:
            - results_df: DataFrame with per-structure results
            - summary: dict with aggregated metrics
        """
        model.eval()
        model.to(self.device)

        if samples is None:
            samples = self.load_test_set()

        results = []
        skipped_samples = []

        for sample in tqdm(samples, desc="Evaluating inverse folding"):
            pdb_id = sample["pdb_id"]
            gt_seq = sample["protein_sequence"]
            protein_mask = sample["protein_mask"]

            # Check combined protein + ligand length (they are concatenated in the model)
            protein_length = len(gt_seq)
            ligand_length = len(sample["ligand_coords"])
            total_length = protein_length + ligand_length

            if total_length > self.max_length:
                logger.warning(
                    f"Skipping {pdb_id}: total length {total_length} "
                    f"(protein: {protein_length}, ligand: {ligand_length}) exceeds max_length {self.max_length}"
                )
                skipped_samples.append(
                    {
                        "pdb_id": pdb_id,
                        "protein_length": protein_length,
                        "ligand_length": ligand_length,
                        "total_length": total_length,
                    }
                )
                continue

            # Compute binding pocket
            pocket_mask = self.compute_binding_pocket(
                sample["protein_coords"],
                sample["ligand_coords"],
                protein_mask,
            )
            non_pocket_mask = protein_mask.bool() & ~pocket_mask

            # Mode 1: Protein only (no ligand context)
            pred_no_ligand = self.inverse_fold(model, sample, include_ligand=False)
            pred_seq_no_ligand = pred_no_ligand["predicted_sequence"]

            # Mode 2: Protein + Ligand context
            pred_with_ligand = self.inverse_fold(model, sample, include_ligand=True)
            pred_seq_with_ligand = pred_with_ligand["predicted_sequence"]

            # Compute metrics
            result = {
                "pdb_id": pdb_id,
                "length": len(gt_seq),
                "n_pocket_residues": int(pocket_mask.sum().item()),
                "n_nonpocket_residues": int(non_pocket_mask.sum().item()),
                # Protein-only metrics
                "aar_overall_no_ligand": self.compute_aar(pred_seq_no_ligand, gt_seq, protein_mask),
                "aar_pocket_no_ligand": self.compute_aar(pred_seq_no_ligand, gt_seq, pocket_mask),
                "aar_nonpocket_no_ligand": self.compute_aar(pred_seq_no_ligand, gt_seq, non_pocket_mask),
                # With-ligand metrics
                "aar_overall_with_ligand": self.compute_aar(pred_seq_with_ligand, gt_seq, protein_mask),
                "aar_pocket_with_ligand": self.compute_aar(pred_seq_with_ligand, gt_seq, pocket_mask),
                "aar_nonpocket_with_ligand": self.compute_aar(pred_seq_with_ligand, gt_seq, non_pocket_mask),
            }

            results.append(result)

        # Log skipped samples
        if skipped_samples:
            logger.info(f"Skipped {len(skipped_samples)} samples due to length > {self.max_length}")
            logger.debug(f"Skipped samples: {skipped_samples}")

        # Create results DataFrame
        results_df = pd.DataFrame(results)

        # Handle empty results
        if len(results_df) == 0:
            logger.warning("No samples were successfully evaluated")
            summary = {
                "mean_aar_overall_no_ligand": float("nan"),
                "mean_aar_overall_with_ligand": float("nan"),
                "mean_aar_pocket_no_ligand": float("nan"),
                "mean_aar_pocket_with_ligand": float("nan"),
                "mean_aar_nonpocket_no_ligand": float("nan"),
                "mean_aar_nonpocket_with_ligand": float("nan"),
                "mean_aar_overall_delta": float("nan"),
                "mean_aar_pocket_delta": float("nan"),
                "mean_aar_nonpocket_delta": float("nan"),
                "std_aar_pocket_delta": float("nan"),
                "std_aar_nonpocket_delta": float("nan"),
                "n_samples": 0,
                "mean_pocket_size": float("nan"),
            }
            return {"results_df": results_df, "summary": summary}

        # Compute delta metrics (improvement from ligand)
        results_df["aar_overall_delta"] = results_df["aar_overall_with_ligand"] - results_df["aar_overall_no_ligand"]
        results_df["aar_pocket_delta"] = results_df["aar_pocket_with_ligand"] - results_df["aar_pocket_no_ligand"]
        results_df["aar_nonpocket_delta"] = (
            results_df["aar_nonpocket_with_ligand"] - results_df["aar_nonpocket_no_ligand"]
        )

        # Compute summary statistics
        summary = {
            # Overall averages (excluding NaN)
            "mean_aar_overall_no_ligand": results_df["aar_overall_no_ligand"].mean(),
            "mean_aar_overall_with_ligand": results_df["aar_overall_with_ligand"].mean(),
            "mean_aar_pocket_no_ligand": results_df["aar_pocket_no_ligand"].mean(),
            "mean_aar_pocket_with_ligand": results_df["aar_pocket_with_ligand"].mean(),
            "mean_aar_nonpocket_no_ligand": results_df["aar_nonpocket_no_ligand"].mean(),
            "mean_aar_nonpocket_with_ligand": results_df["aar_nonpocket_with_ligand"].mean(),
            # Delta (improvement from ligand)
            "mean_aar_overall_delta": results_df["aar_overall_delta"].mean(),
            "mean_aar_pocket_delta": results_df["aar_pocket_delta"].mean(),
            "mean_aar_nonpocket_delta": results_df["aar_nonpocket_delta"].mean(),
            # Standard deviations
            "std_aar_pocket_delta": results_df["aar_pocket_delta"].std(),
            "std_aar_nonpocket_delta": results_df["aar_nonpocket_delta"].std(),
            # Sample counts
            "n_samples": len(results_df),
            "mean_pocket_size": results_df["n_pocket_residues"].mean(),
        }

        return {"results_df": results_df, "summary": summary}

    def sequence_to_string(self, seq_tensor: Tensor) -> str:
        """Convert sequence tensor to string."""
        return "".join([self.aa_map.get(int(s), "X") for s in seq_tensor.cpu().tolist()])
