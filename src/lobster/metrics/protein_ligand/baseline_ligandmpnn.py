"""LigandMPNN Inverse Folding Baseline Evaluator.

Benchmarks LigandMPNN on the same protein-ligand test set used for LeFlur
inverse folding evaluation. Supports running LigandMPNN locally via subprocess
(default) or via Pylon endpoint.

Co-folding validation (Protenix/Boltz) is handled separately via SLURM batch
jobs -- the output CSV includes ``sequence`` and ``smiles`` columns so that
``submit_cofold_batch.py --eval_csv`` can consume it directly.

Metrics:
- AAR (amino acid recovery): overall and pocket
"""

import os
from glob import glob
from typing import Any

import pandas as pd
import torch
from loguru import logger
from tmtools import tm_align
from torch import Tensor
from tqdm import tqdm

from lobster.metrics import align_and_compute_rmsd
from lobster.metrics.pylon_client import (
    call_ligandmpnn,
    call_ligandmpnn_local,
    ligand_data_to_smiles,
    ligand_sdf_to_smiles,
    LIGANDMPNN_DEFAULT_PATH,
)
from lobster.model.latent_generator.utils.residue_constants import restype_order_with_x_inv


# Standard amino acid mapping (alphabetical order, matching .pt file format)
STANDARD_AA_MAP = {
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

AA_TO_IDX = {v: k for k, v in STANDARD_AA_MAP.items()}


def _seq_str_to_tensor(seq_str: str, device: str = "cpu") -> Tensor:
    """Convert amino acid string to standard-format index tensor."""
    return torch.tensor([AA_TO_IDX.get(c, 20) for c in seq_str], dtype=torch.long, device=device)


class LigandMPNNInverseFoldingBaselineEvaluator:
    """Evaluates LigandMPNN on protein-ligand inverse folding.

    For each sample in the test set:
    1. Read GT complex PDB and run LigandMPNN for sequence design.
    2. Compute AAR (overall and pocket) of designed sequence vs GT sequence.

    Co-folding validation is done as a separate step via SLURM batch jobs.
    The output CSV includes ``sequence`` and ``smiles`` columns for downstream
    use by ``submit_cofold_batch.py --eval_csv``.

    Parameters
    ----------
    data_dir : str
        Path to directory with *_protein.pt and *_ligand.pt pairs.
    raw_data_dir : str
        Path to raw benchmark data with SDF files for SMILES extraction
        (e.g., posebusters_benchmark_set/).
    pocket_distance_threshold : float
        Distance threshold (angstrom) for binding pocket definition.
    num_samples : int, optional
        Limit number of samples to evaluate.
    num_designs : int
        Number of LigandMPNN designs per structure.
    temperature : float
        LigandMPNN sampling temperature.
    device : str
        Device for tensor operations.
    use_local_ligandmpnn : bool
        If True, run LigandMPNN locally via subprocess. If False, use Pylon.
    ligandmpnn_path : str
        Path to local LigandMPNN repo (only used when use_local_ligandmpnn=True).
    max_protein_length : int
        Maximum protein length to evaluate.
    """

    def __init__(
        self,
        data_dir: str,
        raw_data_dir: str,
        pocket_distance_threshold: float = 5.0,
        num_samples: int | None = None,
        num_designs: int = 10,
        temperature: float = 0.1,
        device: str = "cpu",
        use_local_ligandmpnn: bool = True,
        ligandmpnn_path: str = LIGANDMPNN_DEFAULT_PATH,
        max_protein_length: int = 512,
    ):
        self.data_dir = data_dir
        self.raw_data_dir = raw_data_dir
        self.pocket_distance_threshold = pocket_distance_threshold
        self.num_samples = num_samples
        self.num_designs = num_designs
        self.temperature = temperature
        self.device = device
        self.use_local_ligandmpnn = use_local_ligandmpnn
        self.ligandmpnn_path = ligandmpnn_path
        self.max_protein_length = max_protein_length

    def load_test_set(self) -> list[dict]:
        """Load protein-ligand test pairs."""
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
                continue

            protein_mask = protein_data.get("mask", torch.ones(protein_coords.shape[0], device=self.device))

            ligand_coords = ligand_data.get("atom_coords", ligand_data.get("coords"))
            if ligand_coords is None:
                continue

            atom_names = ligand_data.get("atom_names", [])
            bond_matrix = ligand_data.get("bond_matrix")

            # Extract SMILES from raw SDF if available, else reconstruct from bond matrix
            smiles = None
            sdf_path = os.path.join(self.raw_data_dir, pdb_id, f"{pdb_id}_ligand.sdf")
            if os.path.exists(sdf_path):
                try:
                    smiles = ligand_sdf_to_smiles(sdf_path)
                except Exception as e:
                    logger.warning(f"Failed to extract SMILES from SDF for {pdb_id}: {e}")

            if smiles is None and atom_names and bond_matrix is not None:
                try:
                    smiles = ligand_data_to_smiles(atom_names, bond_matrix, ligand_coords)
                except Exception as e:
                    logger.warning(f"Failed to reconstruct SMILES for {pdb_id}: {e}")

            samples.append(
                {
                    "pdb_id": pdb_id,
                    "protein_coords": protein_coords,
                    "protein_sequence": protein_sequence,
                    "protein_mask": protein_mask,
                    "ligand_coords": ligand_coords,
                    "ligand_atom_names": atom_names,
                    "bond_matrix": bond_matrix,
                    "smiles": smiles,
                }
            )

        logger.info(f"Loaded {len(samples)} valid samples")
        return samples

    def _sequence_to_string(self, seq_tensor: Tensor) -> str:
        return "".join([STANDARD_AA_MAP.get(int(s), "X") for s in seq_tensor.cpu().tolist()])

    def _get_pdb_content(self, pdb_id: str) -> str | None:
        """Get PDB content from the raw benchmark set (original PDB files).

        LigandMPNN requires properly formatted PDB files with HETATM records,
        so we use the originals rather than reconstructing from .pt data.
        """
        pdb_path = os.path.join(self.raw_data_dir, pdb_id, f"{pdb_id}_protein.pdb")
        if os.path.exists(pdb_path):
            with open(pdb_path) as f:
                return f.read()
        return None

    def compute_binding_pocket(
        self,
        protein_coords: Tensor,
        ligand_coords: Tensor,
        protein_mask: Tensor | None = None,
    ) -> Tensor:
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

    def compute_aar(self, predicted_seq: Tensor, gt_seq: Tensor, mask: Tensor | None = None) -> float:
        if mask is not None:
            mask = mask.bool()
            if mask.sum() == 0:
                return float("nan")
            predicted_seq = predicted_seq[mask]
            gt_seq = gt_seq[mask]
        if len(predicted_seq) == 0:
            return float("nan")
        return (predicted_seq == gt_seq).float().mean().item()

    def compute_tm_score(
        self,
        pred_coords: Tensor,
        gt_coords: Tensor,
        sequence: Tensor,
        mask: Tensor | None = None,
    ) -> float:
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

    def compute_rmsd(self, pred_coords: Tensor, gt_coords: Tensor, mask: Tensor | None = None) -> float:
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

    def evaluate(
        self,
        samples: list[dict] | None = None,
        structure_path: str | None = None,
    ) -> dict:
        """Run LigandMPNN baseline evaluation.

        Parameters
        ----------
        samples : list[dict], optional
            Pre-loaded samples. Loads from data_dir if not provided.
        structure_path : str, optional
            Directory to save output structures.

        Returns
        -------
        dict with results_df and summary.
        """
        if samples is None:
            samples = self.load_test_set()

        if structure_path:
            os.makedirs(structure_path, exist_ok=True)

        results = []

        for sample in tqdm(samples, desc="LigandMPNN baseline evaluation"):
            pdb_id = sample["pdb_id"]
            gt_seq = sample["protein_sequence"]
            gt_coords = sample["protein_coords"]
            protein_mask = sample["protein_mask"]
            protein_length = len(gt_seq)

            if protein_length > self.max_protein_length:
                logger.warning(f"Skipping {pdb_id}: protein length {protein_length} > {self.max_protein_length}")
                continue

            pocket_mask = self.compute_binding_pocket(gt_coords, sample["ligand_coords"], protein_mask)

            # Get original PDB for LigandMPNN (needs properly formatted PDB with HETATM)
            pdb_content = self._get_pdb_content(pdb_id)
            if pdb_content is None:
                logger.warning(f"No original PDB found for {pdb_id} in {self.raw_data_dir}, skipping")
                continue

            # Call LigandMPNN (local subprocess or Pylon)
            try:
                if self.use_local_ligandmpnn:
                    designed_sequences = call_ligandmpnn_local(
                        structure=pdb_content,
                        batch_size=self.num_designs,
                        number_of_batches=1,
                        temperature=self.temperature,
                        model_type="ligand_mpnn",
                        ligandmpnn_path=self.ligandmpnn_path,
                    )
                else:
                    designed_sequences = call_ligandmpnn(
                        structure=pdb_content,
                        batch_size=self.num_designs,
                        number_of_batches=1,
                        temperature=self.temperature,
                        model_type="ligand_mpnn",
                    )
            except Exception as e:
                logger.warning(f"LigandMPNN failed for {pdb_id}: {e}")
                continue

            if not designed_sequences:
                logger.warning(f"LigandMPNN returned no sequences for {pdb_id}")
                continue

            # LigandMPNN returns WT as sequences[0]; designed sequences start at [1]
            designed_only = designed_sequences[1:] if len(designed_sequences) > 1 else designed_sequences
            if not designed_only:
                logger.warning(f"LigandMPNN returned only WT for {pdb_id}")
                continue
            designed_seq_str = designed_only[0]
            designed_seq_tensor = _seq_str_to_tensor(designed_seq_str, self.device)

            # Truncate/pad to match GT length
            if len(designed_seq_tensor) > protein_length:
                designed_seq_tensor = designed_seq_tensor[:protein_length]
            elif len(designed_seq_tensor) < protein_length:
                logger.warning(
                    f"{pdb_id}: LigandMPNN seq length {len(designed_seq_tensor)} != GT {protein_length}, skipping"
                )
                continue

            result: dict[str, Any] = {
                "pdb_id": pdb_id,
                "length": protein_length,
                "n_pocket_residues": int(pocket_mask.sum().item()),
                "n_designs": len(designed_sequences),
                "sequence": designed_seq_str,
                "smiles": sample.get("smiles", ""),
                "aar_overall": self.compute_aar(designed_seq_tensor, gt_seq, protein_mask),
                "aar_pocket": self.compute_aar(designed_seq_tensor, gt_seq, pocket_mask),
            }

            results.append(result)

        results_df = pd.DataFrame(results)

        if len(results_df) == 0:
            logger.warning("No samples were successfully evaluated")
            return {"results_df": results_df, "summary": {"n_samples": 0}}

        summary: dict[str, Any] = {
            "n_samples": len(results_df),
            "mean_aar_overall": results_df["aar_overall"].mean(),
            "mean_aar_pocket": results_df["aar_pocket"].mean(),
            "mean_pocket_size": results_df["n_pocket_residues"].mean(),
        }

        return {"results_df": results_df, "summary": summary}
