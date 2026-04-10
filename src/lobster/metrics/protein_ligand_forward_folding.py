"""Protein-Ligand Forward Folding Evaluator.

Evaluates forward folding (sequence → structure) on protein-ligand complexes
with and without ligand context.

Key Question: Does providing ligand context improve structure prediction quality
(TM-score, RMSD) for the protein, particularly for binding pocket residues?
"""

import os
from glob import glob
from typing import TYPE_CHECKING

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
from lobster.model.latent_generator.io import writepdb, writepdb_ligand_complex
from lobster.model.latent_generator.utils import minimize_ligand_structure
from lobster.model.latent_generator.utils.residue_constants import restype_order_with_x_inv
from lobster.transforms._structure_transforms import AminoAcidTokenizerTransform

# Mapping from string names to inference schedule classes
INFERENCE_SCHEDULE_MAP = {
    "LinearInferenceSchedule": LinearInferenceSchedule,
    "LogInferenceSchedule": LogInferenceSchedule,
    "PowerInferenceSchedule": PowerInferenceSchedule,
}


def _get_inference_schedule_class(schedule_name: str):
    """Convert string schedule name to class."""
    if schedule_name not in INFERENCE_SCHEDULE_MAP:
        raise ValueError(f"Unknown inference schedule: {schedule_name}. Options: {list(INFERENCE_SCHEDULE_MAP.keys())}")
    return INFERENCE_SCHEDULE_MAP[schedule_name]


if TYPE_CHECKING:
    from lightning import LightningModule


class ProteinLigandForwardFoldingEvaluator:
    """Evaluates forward folding on protein-ligand complexes with/without ligand context.

    This evaluator compares two modes:
    1. Protein-only: Provide only protein sequence, predict structure
    2. Protein+Ligand: Provide protein sequence + ligand, predict structure

    Tracks metrics:
    - Overall TM-score and RMSD
    - Binding pocket RMSD (residues within distance threshold of ligand)
    - Non-pocket RMSD

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
    nsteps : int
        Number of diffusion steps for generation
    device : str
        Device for computation
    max_length : int
        Maximum combined sequence length (protein + ligand) to process (default: 512).
        Samples exceeding this length will be skipped.
    max_protein_length : int
        Maximum protein-only sequence length (default: 512). Samples with protein length
        exceeding this will be skipped entirely.
    temperature_seq : float
        Temperature for sequence sampling
    temperature_struc : float
        Temperature for structure sampling
    save_structures : bool
        Whether to save predicted structures as PDB files (default: False).
    save_gt_structure : bool
        Whether to save ground truth structures as PDB files (default: False).
    minimize_ligand : bool
        Whether to apply geometry correction to decoded ligand structures (default: False).
    minimize_mode : str
        Minimization mode: "bonds_only", "bonds_and_angles", "local", or "full".
    force_field : str
        Force field for minimization: "MMFF94", "MMFF94s", "UFF", etc.
    minimize_steps : int
        Maximum number of minimization steps.
    stochasticity_seq : int
        Stochasticity parameter for sequence sampling (default: 20).
    stochasticity_struc : int
        Stochasticity parameter for structure sampling (default: 20).
    temperature_ligand : float
        Temperature for ligand structure sampling (default: 0.5).
    stochasticity_ligand : int
        Stochasticity parameter for ligand structure sampling (default: 20).
    ligand_context_mode : str
        How to provide ligand context. Options:
        - "structure_tokens": Encode GT ligand structure and provide tokens as fixed context
        - "atom_bond_only": Only provide atom types + bond matrix, model generates ligand structure
    inference_schedule_seq : str
        Inference schedule for sequence generation. Options: "LinearInferenceSchedule",
        "LogInferenceSchedule", "PowerInferenceSchedule" (default: "LogInferenceSchedule").
    inference_schedule_struc : str
        Inference schedule for structure generation. Options: "LinearInferenceSchedule",
        "LogInferenceSchedule", "PowerInferenceSchedule" (default: "LinearInferenceSchedule").
    inference_schedule_ligand_atom : str
        Inference schedule for ligand atom token generation. Options: "LinearInferenceSchedule",
        "LogInferenceSchedule", "PowerInferenceSchedule", or None to use sequence schedule
        (default: None).
    inference_schedule_ligand_struc : str
        Inference schedule for ligand structure token generation. Options: "LinearInferenceSchedule",
        "LogInferenceSchedule", "PowerInferenceSchedule", or None to use structure schedule
        (default: None).
    num_predictions : int
        Number of predictions per sample for best-of-N evaluation (default: 1).
        When > 1, generates multiple predictions and selects the best one.
    best_of_n_metric : str
        Metric to use for best-of-N selection: "rmsd" (lower is better) or "tm_score"
        (higher is better). Default: "rmsd".
    save_all_predictions : bool
        Whether to save all N predicted structures (not just the best). Only applies
        when save_structures=True and num_predictions > 1. Default: False.
    try_reflection : bool
        Whether to try both original and reflected (mirror image) coordinates and
        select the one with higher TM-score. This is useful if the model might
        output mirror images of structures. Default: False.
    use_protenix : bool
        Whether to additionally validate with Protenix co-folding via Pylon endpoint.
        Sends GT sequence + ligand SMILES to Protenix and compares to GT structure.
    use_boltz : bool
        Whether to additionally validate with Boltz-2 co-folding via Pylon endpoint.
        Mutually exclusive with use_protenix; if both are True, Boltz is used.
    raw_data_dir : str, optional
        Path to raw benchmark data with SDF files for SMILES extraction. Required if
        use_protenix=True or use_boltz=True.
    """

    def __init__(
        self,
        data_dir: str,
        pocket_distance_threshold: float = 5.0,
        num_samples: int | None = None,
        nsteps: int = 100,
        device: str = "cuda",
        max_length: int = 512,
        max_protein_length: int = 512,
        temperature_seq: float = 0.5,
        temperature_struc: float = 0.5,
        save_structures: bool = False,
        save_gt_structure: bool = False,
        minimize_ligand: bool = False,
        minimize_mode: str = "bonds_and_angles",
        force_field: str = "MMFF94",
        minimize_steps: int = 500,
        # Additional generation hyperparameters
        stochasticity_seq: int = 20,
        stochasticity_struc: int = 20,
        temperature_ligand: float = 0.5,
        stochasticity_ligand: int = 20,
        ligand_context_mode: str = "structure_tokens",
        inference_schedule_seq: str = "LogInferenceSchedule",
        inference_schedule_struc: str = "LinearInferenceSchedule",
        inference_schedule_ligand_atom: str | None = None,
        inference_schedule_ligand_struc: str | None = None,
        # Best-of-N parameters
        num_predictions: int = 1,
        best_of_n_metric: str = "rmsd",
        save_all_predictions: bool = False,
        # Mirror image handling
        try_reflection: bool = False,
        # Co-folding validation (Protenix or Boltz)
        use_protenix: bool = False,
        use_boltz: bool = False,
        raw_data_dir: str | None = None,
    ):
        self.data_dir = data_dir
        self.pocket_distance_threshold = pocket_distance_threshold
        self.num_samples = num_samples
        self.nsteps = nsteps
        self.device = device
        self.max_length = max_length
        self.max_protein_length = max_protein_length
        self.temperature_seq = temperature_seq
        self.temperature_struc = temperature_struc
        self.save_structures = save_structures
        self.save_gt_structure = save_gt_structure
        self.minimize_ligand = minimize_ligand
        self.minimize_mode = minimize_mode
        self.force_field = force_field
        self.minimize_steps = minimize_steps
        # Additional generation hyperparameters
        self.stochasticity_seq = stochasticity_seq
        self.stochasticity_struc = stochasticity_struc
        self.temperature_ligand = temperature_ligand
        self.stochasticity_ligand = stochasticity_ligand
        self.ligand_context_mode = ligand_context_mode
        self.inference_schedule_seq = inference_schedule_seq
        self.inference_schedule_struc = inference_schedule_struc
        self.inference_schedule_ligand_atom = inference_schedule_ligand_atom
        self.inference_schedule_ligand_struc = inference_schedule_ligand_struc
        # Best-of-N parameters
        self.num_predictions = num_predictions
        self.best_of_n_metric = best_of_n_metric
        self.save_all_predictions = save_all_predictions
        if best_of_n_metric not in ("rmsd", "tm_score"):
            raise ValueError(f"best_of_n_metric must be 'rmsd' or 'tm_score', got {best_of_n_metric}")
        # Mirror image handling
        self.try_reflection = try_reflection

        # Co-folding validation
        self.use_protenix = use_protenix or use_boltz
        self.use_boltz = use_boltz
        self.cofold_backend = "boltz" if use_boltz else "protenix"
        self.raw_data_dir = raw_data_dir

        # Initialize tokenizer transform for sequence conversion
        self.tokenizer_transform = AminoAcidTokenizerTransform(max_length=max_length)

        # Standard amino acid mapping (alphabetical order, matching .pt file format)
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

    def _reflect_coords(self, coords: Tensor) -> Tensor:
        """Create a mirror image of coordinates by negating the x-axis.

        Parameters
        ----------
        coords : Tensor
            Coordinates tensor of shape [..., 3] where the last dimension is (x, y, z)

        Returns
        -------
        Tensor
            Reflected coordinates with x-axis negated
        """
        reflected = coords.clone()
        reflected[..., 0] = -reflected[..., 0]
        return reflected

    def _select_best_orientation(
        self,
        pred_coords: Tensor,
        gt_coords: Tensor,
        sequence: Tensor,
        mask: Tensor | None = None,
        decoded_ligand_coords: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None, bool]:
        """Select best orientation (original or reflected) based on TM-score.

        Parameters
        ----------
        pred_coords : Tensor
            [L, 3, 3] predicted protein backbone coordinates
        gt_coords : Tensor
            [L, 3, 3] ground truth backbone coordinates
        sequence : Tensor
            [L] sequence tokens for TM-align
        mask : Tensor, optional
            [L] boolean mask for positions to include
        decoded_ligand_coords : Tensor, optional
            [N_atoms, 3] predicted ligand coordinates (will be reflected too if needed)

        Returns
        -------
        tuple
            (best_pred_coords, best_ligand_coords, was_reflected)
        """
        # Compute TM-score for original orientation
        tm_original = self.compute_tm_score(pred_coords, gt_coords, sequence, mask)

        # Compute TM-score for reflected orientation
        reflected_coords = self._reflect_coords(pred_coords)
        tm_reflected = self.compute_tm_score(reflected_coords, gt_coords, sequence, mask)

        # Select the orientation with higher TM-score
        if tm_reflected > tm_original:
            # Use reflected coordinates
            reflected_ligand = None
            if decoded_ligand_coords is not None:
                reflected_ligand = self._reflect_coords(decoded_ligand_coords)
            return reflected_coords, reflected_ligand, True
        else:
            # Use original coordinates
            return pred_coords, decoded_ligand_coords, False

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

            # Extract ligand SMILES (for CSV output and downstream co-folding)
            smiles = None
            if self.raw_data_dir:
                sdf_path = os.path.join(self.raw_data_dir, pdb_id, f"{pdb_id}_ligand.sdf")
                if os.path.exists(sdf_path):
                    try:
                        from lobster.metrics.pylon_client import ligand_sdf_to_smiles

                        smiles = ligand_sdf_to_smiles(sdf_path)
                    except Exception as e:
                        logger.warning(f"Failed to extract SMILES for {pdb_id}: {e}")

            samples.append(
                {
                    "pdb_id": pdb_id,
                    "protein_coords": protein_coords,
                    "protein_sequence": protein_sequence,
                    "protein_mask": protein_mask,
                    "protein_indices": protein_indices,
                    "ligand_coords": ligand_coords,
                    "ligand_atom_types": ligand_atom_types,
                    "ligand_atom_names": atom_names,
                    "ligand_mask": ligand_mask,
                    "ligand_indices": ligand_indices,
                    "bond_matrix": bond_matrix,
                    "smiles": smiles,
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

    def forward_fold(
        self,
        model: "LightningModule",
        sample: dict,
        include_ligand: bool,
    ) -> dict:
        """Run forward folding with or without ligand context.

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
            - predicted_coords: Tensor [L, 3, 3] (N, CA, C backbone)
            - structure_tokens: Tensor [L]
        """
        # Prepare protein inputs
        protein_mask = sample["protein_mask"].unsqueeze(0).float()
        protein_indices = sample["protein_indices"].unsqueeze(0).long()
        length = int(protein_mask.sum().item())

        # Tokenize sequence for forward folding
        gt_seq = sample["protein_sequence"]
        tokenized_data = self.tokenizer_transform({"sequence": gt_seq.cpu()})
        tokenized_seq = tokenized_data["sequence"].to(self.device).unsqueeze(0)  # [1, L]

        # Prepare ligand inputs if needed
        ligand_mask = None
        ligand_atom_tokens = None
        ligand_structure_tokens = None
        ligand_structure_embeddings = None
        bond_matrix = None
        num_atoms = 0

        if include_ligand:
            ligand_coords = sample["ligand_coords"].unsqueeze(0).float()
            ligand_mask = sample["ligand_mask"].unsqueeze(0).float()
            ligand_indices = sample["ligand_indices"].unsqueeze(0).long()
            ligand_atom_tokens = sample["ligand_atom_types"].unsqueeze(0).long()
            num_atoms = ligand_coords.shape[1]

            # Conditionally encode ligand structure based on ligand_context_mode
            if self.ligand_context_mode == "structure_tokens":
                # Encode GT ligand structure and provide tokens as fixed context
                with torch.no_grad():
                    encode_result = model.encode_ligand_structure(
                        ligand_coords, ligand_mask, ligand_indices, return_continuous=True
                    )
                    ligand_structure_tokens, _, ligand_structure_embeddings = encode_result
            else:
                # atom_bond_only mode: don't provide structure tokens, model generates ligand structure
                ligand_structure_tokens = None
                ligand_structure_embeddings = None

            bond_matrix = sample.get("bond_matrix")
            if bond_matrix is not None:
                bond_matrix = bond_matrix.unsqueeze(0).long()

        # Get inference schedule classes
        inference_schedule_seq_class = _get_inference_schedule_class(self.inference_schedule_seq)
        inference_schedule_struc_class = _get_inference_schedule_class(self.inference_schedule_struc)
        # Ligand schedules (None to fall back to protein schedules)
        inference_schedule_ligand_atom_class = (
            _get_inference_schedule_class(self.inference_schedule_ligand_atom)
            if self.inference_schedule_ligand_atom
            else None
        )
        inference_schedule_ligand_struc_class = (
            _get_inference_schedule_class(self.inference_schedule_ligand_struc)
            if self.inference_schedule_ligand_struc
            else None
        )

        # Generate sample (forward folding mode)
        with torch.no_grad():
            result = model.generate_sample(
                length=length,
                num_samples=1,
                forward_folding=True,
                nsteps=self.nsteps,
                inference_schedule_seq=inference_schedule_seq_class,
                inference_schedule_struc=inference_schedule_struc_class,
                inference_schedule_ligand_atom=inference_schedule_ligand_atom_class,
                inference_schedule_ligand_struc=inference_schedule_ligand_struc_class,
                temperature_seq=self.temperature_seq,
                temperature_struc=self.temperature_struc,
                stochasticity_seq=self.stochasticity_seq,
                stochasticity_struc=self.stochasticity_struc,
                temperature_ligand=self.temperature_ligand,
                stochasticity_ligand=self.stochasticity_ligand,
                input_sequence_tokens=tokenized_seq,
                input_mask=protein_mask,
                input_indices=protein_indices,
                # Ligand context (fixed conditioning if structure_tokens mode, otherwise model generates)
                generate_ligand=include_ligand,
                num_atoms=num_atoms if include_ligand else 0,
                input_ligand_atom_tokens=ligand_atom_tokens,
                input_ligand_structure_tokens=ligand_structure_tokens,
                input_ligand_structure_embeddings=ligand_structure_embeddings,
                input_bond_matrix=bond_matrix,
                ligand_is_context=include_ligand and self.ligand_context_mode == "structure_tokens",
            )

        # Decode structure
        decoded_x = model.decode_structure(result, protein_mask, ligand_mask=ligand_mask)

        # Extract coordinates from vit_decoder
        predicted_coords = None
        decoded_ligand_coords = None
        for decoder_name in decoded_x:
            if "vit_decoder" == decoder_name:
                vit_output = decoded_x[decoder_name]
                # Handle both tensor output (protein-only) and dict output (protein-ligand)
                if isinstance(vit_output, dict):
                    predicted_coords = vit_output.get("protein_coords")
                    decoded_ligand_coords = vit_output.get("ligand_coords")
                else:
                    predicted_coords = vit_output
                break

        if predicted_coords is None:
            raise RuntimeError("No vit_decoder found in decoded structures")

        # Handle both discrete (structure_tokens) and continuous (structure_embeddings) modes
        structure_tokens = result.get("generated_struc_tokens")
        structure_embeddings = result.get("generated_structure_embeddings")

        # Get predicted bond matrix if available
        predicted_bond_matrix = result.get("predicted_bond_matrix")

        return {
            "predicted_coords": predicted_coords.squeeze(0),  # [L, 3, 3]
            "structure_tokens": structure_tokens.squeeze(0) if structure_tokens is not None else None,
            "structure_embeddings": structure_embeddings.squeeze(0) if structure_embeddings is not None else None,
            "decoded_ligand_coords": decoded_ligand_coords.squeeze(0) if decoded_ligand_coords is not None else None,
            "predicted_bond_matrix": predicted_bond_matrix.squeeze(0) if predicted_bond_matrix is not None else None,
        }

    def forward_fold_best_of_n(
        self,
        model: "LightningModule",
        sample: dict,
        include_ligand: bool,
        n_predictions: int,
    ) -> dict:
        """Run forward folding N times and return the best prediction.

        Parameters
        ----------
        model : LightningModule
            The Gen-UME protein-ligand model
        sample : dict
            Sample dictionary from load_test_set()
        include_ligand : bool
            Whether to include ligand context
        n_predictions : int
            Number of predictions to generate

        Returns
        -------
        dict with:
            - predicted_coords: Tensor [L, 3, 3] (N, CA, C backbone) - best prediction
            - structure_tokens: Tensor [L] - from best prediction
            - all_predictions: list of dicts - all N predictions with their scores
            - best_idx: int - index of the best prediction
            - best_score: float - score of the best prediction
        """
        if n_predictions == 1:
            pred = self.forward_fold(model, sample, include_ligand)
            return {
                **pred,
                "all_predictions": [pred],
                "best_idx": 0,
                "best_score": None,
            }

        gt_coords = sample["protein_coords"]
        gt_seq = sample["protein_sequence"]
        protein_mask = sample["protein_mask"]

        all_predictions = []
        scores = []

        for i in range(n_predictions):
            pred = self.forward_fold(model, sample, include_ligand)
            all_predictions.append(pred)

            # Compute score for selection
            if self.best_of_n_metric == "rmsd":
                score = self.compute_rmsd(pred["predicted_coords"], gt_coords, protein_mask)
            else:  # tm_score
                score = self.compute_tm_score(pred["predicted_coords"], gt_coords, gt_seq, protein_mask)
            scores.append(score)

        # Select best prediction
        if self.best_of_n_metric == "rmsd":
            # Lower RMSD is better
            best_idx = min(range(len(scores)), key=lambda i: scores[i])
        else:
            # Higher TM-score is better
            best_idx = max(range(len(scores)), key=lambda i: scores[i])

        best_pred = all_predictions[best_idx]
        return {
            **best_pred,
            "all_predictions": all_predictions,
            "best_idx": best_idx,
            "best_score": scores[best_idx],
            "all_scores": scores,
        }

    def compute_tm_score(
        self,
        pred_coords: Tensor,
        gt_coords: Tensor,
        sequence: Tensor,
        mask: Tensor | None = None,
    ) -> float:
        """Compute TM-score between predicted and ground truth structures.

        Parameters
        ----------
        pred_coords : Tensor
            [L, 3, 3] predicted backbone coordinates (N, CA, C)
        gt_coords : Tensor
            [L, 3, 3] ground truth backbone coordinates
        sequence : Tensor
            [L] sequence tokens for alignment
        mask : Tensor, optional
            [L] boolean mask for positions to include

        Returns
        -------
        float
            TM-score (0-1, higher is better)
        """
        # Apply mask if provided
        if mask is not None:
            mask = mask.bool()
            pred_coords = pred_coords[mask]
            gt_coords = gt_coords[mask]
            sequence = sequence[mask]

        if len(pred_coords) == 0:
            return float("nan")

        # Get sequence string for TM-align
        sequence_str = "".join([restype_order_with_x_inv.get(int(s), "X") for s in sequence.cpu().tolist()])

        # Use CA atoms (index 1) for TM-align
        pred_ca = pred_coords[:, 1, :].detach().cpu().numpy()
        gt_ca = gt_coords[:, 1, :].detach().cpu().numpy()

        # Calculate TM-Score using TM-align
        tm_out = tm_align(pred_ca, gt_ca, sequence_str, sequence_str)

        return tm_out.tm_norm_chain1

    def compute_rmsd(
        self,
        pred_coords: Tensor,
        gt_coords: Tensor,
        mask: Tensor | None = None,
    ) -> float:
        """Compute RMSD between predicted and ground truth structures.

        Parameters
        ----------
        pred_coords : Tensor
            [L, 3, 3] predicted backbone coordinates (N, CA, C)
        gt_coords : Tensor
            [L, 3, 3] ground truth backbone coordinates
        mask : Tensor, optional
            [L] boolean mask for positions to include

        Returns
        -------
        float
            RMSD in Angstroms (lower is better)
        """
        # Apply mask if provided
        if mask is not None:
            mask = mask.bool()
            pred_coords = pred_coords[mask]
            gt_coords = gt_coords[mask]

        if len(pred_coords) == 0:
            return float("nan")

        # Calculate RMSD using Kabsch alignment (detach to avoid gradient issues)
        rmsd = align_and_compute_rmsd(
            coords1=pred_coords.detach(),
            coords2=gt_coords.detach(),
            mask=None,  # Already masked
            return_aligned=False,
            device=pred_coords.device,
        )

        return float(rmsd)

    def evaluate(
        self,
        model: "LightningModule",
        samples: list[dict] | None = None,
        structure_path: str | None = None,
    ) -> dict:
        """Run full evaluation on PDBBind test set.

        Parameters
        ----------
        model : LightningModule
            The Gen-UME protein-ligand model
        samples : list[dict], optional
            Pre-loaded samples (will load if not provided)
        structure_path : str, optional
            Directory to save predicted structures as PDB files

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

        # Log best-of-N settings
        if self.num_predictions > 1:
            logger.info(f"Using best-of-{self.num_predictions} evaluation (selecting by {self.best_of_n_metric})")

        # Log reflection settings
        if self.try_reflection:
            logger.info(
                "Mirror image handling enabled: will try both original and reflected "
                "coordinates and select based on TM-score"
            )

        # Create output directory if specified
        if structure_path:
            os.makedirs(structure_path, exist_ok=True)

        results = []
        skipped_samples = []

        for sample in tqdm(samples, desc="Evaluating forward folding"):
            pdb_id = sample["pdb_id"]
            gt_seq = sample["protein_sequence"]
            gt_coords = sample["protein_coords"]
            protein_mask = sample["protein_mask"]

            # Check protein and combined lengths
            protein_length = len(gt_seq)
            ligand_length = len(sample["ligand_coords"])
            total_length = protein_length + ligand_length

            # Skip samples exceeding max protein length
            if protein_length > self.max_protein_length:
                logger.warning(
                    f"Skipping {pdb_id}: protein length {protein_length} "
                    f"exceeds max_protein_length {self.max_protein_length}"
                )
                skipped_samples.append(
                    {
                        "pdb_id": pdb_id,
                        "protein_length": protein_length,
                        "ligand_length": ligand_length,
                        "total_length": total_length,
                        "reason": "max_protein_length",
                    }
                )
                continue

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
                        "reason": "max_length",
                    }
                )
                continue

            # Compute binding pocket
            pocket_mask = self.compute_binding_pocket(
                gt_coords,
                sample["ligand_coords"],
                protein_mask,
            )
            non_pocket_mask = protein_mask.bool() & ~pocket_mask

            # Mode 1: Protein only (no ligand context)
            # NOTE: try-catch removed for debugging - will crash on first error to get full traceback
            pred_no_ligand = self.forward_fold_best_of_n(
                model, sample, include_ligand=False, n_predictions=self.num_predictions
            )
            pred_coords_no_ligand = pred_no_ligand["predicted_coords"]

            # Mode 2: Protein + Ligand context
            pred_with_ligand = self.forward_fold_best_of_n(
                model, sample, include_ligand=True, n_predictions=self.num_predictions
            )
            pred_coords_with_ligand = pred_with_ligand["predicted_coords"]
            decoded_ligand_coords_with_ligand = pred_with_ligand.get("decoded_ligand_coords")

            # Try reflection if enabled - select best orientation based on TM-score
            reflected_no_ligand = False
            reflected_with_ligand = False
            if self.try_reflection:
                pred_coords_no_ligand, _, reflected_no_ligand = self._select_best_orientation(
                    pred_coords_no_ligand, gt_coords, gt_seq, protein_mask
                )
                pred_coords_with_ligand, decoded_ligand_coords_with_ligand, reflected_with_ligand = (
                    self._select_best_orientation(
                        pred_coords_with_ligand,
                        gt_coords,
                        gt_seq,
                        protein_mask,
                        decoded_ligand_coords_with_ligand,
                    )
                )
                # Update the prediction dict with potentially reflected coordinates
                pred_no_ligand["predicted_coords"] = pred_coords_no_ligand
                pred_with_ligand["predicted_coords"] = pred_coords_with_ligand
                if decoded_ligand_coords_with_ligand is not None:
                    pred_with_ligand["decoded_ligand_coords"] = decoded_ligand_coords_with_ligand

            # Save structures if requested
            if structure_path and (self.save_structures or self.save_gt_structure):
                ligand_coords = sample["ligand_coords"]
                # Get atom names from original data or generate default names
                atom_names = sample.get("ligand_atom_names")
                if atom_names is None:
                    # Generate default atom names from element indices
                    idx_to_element = {v: k for k, v in self.element_to_idx.items()}
                    ligand_types = sample["ligand_atom_types"]
                    atom_names = [
                        f"{idx_to_element.get(int(t), 'C')}{i + 1}" for i, t in enumerate(ligand_types.cpu().tolist())
                    ]

                # Get bond matrix for CONECT records
                bond_matrix = sample.get("bond_matrix")

                # Save ground truth structures
                if self.save_gt_structure:
                    # Save ground truth protein structure
                    gt_pdb_path = os.path.join(structure_path, f"{pdb_id}_gt_protein.pdb")
                    writepdb(gt_pdb_path, gt_coords, gt_seq)

                    # Save ground truth protein-ligand complex
                    gt_complex_path = os.path.join(structure_path, f"{pdb_id}_gt_complex.pdb")
                    writepdb_ligand_complex(
                        gt_complex_path,
                        protein_atoms=gt_coords,
                        protein_seq=gt_seq,
                        ligand_atoms=ligand_coords,
                        ligand_atom_names=atom_names,
                        ligand_bond_matrix=bond_matrix,
                    )

                # Save predicted structures
                if self.save_structures:
                    # Determine which predictions to save
                    if self.save_all_predictions and self.num_predictions > 1:
                        # Save all N predictions
                        all_preds_no_ligand = pred_no_ligand.get("all_predictions", [pred_no_ligand])
                        all_preds_with_ligand = pred_with_ligand.get("all_predictions", [pred_with_ligand])
                        best_idx_no_ligand = pred_no_ligand.get("best_idx", 0)
                        best_idx_with_ligand = pred_with_ligand.get("best_idx", 0)

                        # Save all predictions without ligand context
                        for i, pred in enumerate(all_preds_no_ligand):
                            suffix = f"_pred{i}" if self.num_predictions > 1 else ""
                            best_marker = "_best" if i == best_idx_no_ligand else ""
                            pred_no_lig_path = os.path.join(
                                structure_path, f"{pdb_id}_pred_no_ligand{suffix}{best_marker}.pdb"
                            )
                            writepdb(pred_no_lig_path, pred["predicted_coords"].detach(), gt_seq)

                        # Save all predictions with ligand context
                        for i, pred in enumerate(all_preds_with_ligand):
                            suffix = f"_pred{i}" if self.num_predictions > 1 else ""
                            best_marker = "_best" if i == best_idx_with_ligand else ""
                            pred_with_lig_path = os.path.join(
                                structure_path, f"{pdb_id}_pred_with_ligand{suffix}{best_marker}.pdb"
                            )
                            decoded_ligand = pred.get("decoded_ligand_coords")
                            if decoded_ligand is not None:
                                pred_bond = pred.get("predicted_bond_matrix")
                                bond_matrix_for_pred = pred_bond if pred_bond is not None else bond_matrix
                                ligand_coords_to_save = decoded_ligand.detach().cpu()
                                if self.minimize_ligand:
                                    try:
                                        ligand_coords_to_save = minimize_ligand_structure(
                                            ligand_coords_to_save,
                                            atom_names,
                                            bond_matrix=bond_matrix_for_pred,
                                            steps=self.minimize_steps,
                                            force_field=self.force_field,
                                            mode=self.minimize_mode,
                                        )
                                    except Exception as e:
                                        logger.warning(f"Ligand minimization failed for {pdb_id} pred{i}: {e}")
                                writepdb_ligand_complex(
                                    pred_with_lig_path,
                                    protein_atoms=pred["predicted_coords"].detach(),
                                    protein_seq=gt_seq,
                                    ligand_atoms=ligand_coords_to_save,
                                    ligand_atom_names=atom_names,
                                    ligand_bond_matrix=bond_matrix_for_pred,
                                )
                            else:
                                writepdb(pred_with_lig_path, pred["predicted_coords"].detach(), gt_seq)
                    else:
                        # Save only the best prediction (default behavior)
                        pred_no_lig_path = os.path.join(structure_path, f"{pdb_id}_pred_no_ligand.pdb")
                        writepdb(pred_no_lig_path, pred_coords_no_ligand.detach(), gt_seq)

                        # Save predicted structure with ligand context (use decoded ligand)
                        pred_with_lig_path = os.path.join(structure_path, f"{pdb_id}_pred_with_ligand.pdb")
                        decoded_ligand_coords = pred_with_ligand.get("decoded_ligand_coords")
                        if decoded_ligand_coords is not None:
                            # Use predicted bond matrix if available, otherwise fall back to GT
                            pred_bond_matrix = pred_with_ligand.get("predicted_bond_matrix")
                            bond_matrix_for_pred = pred_bond_matrix if pred_bond_matrix is not None else bond_matrix

                            # Apply minimization if enabled
                            ligand_coords_to_save = decoded_ligand_coords.detach().cpu()
                            if self.minimize_ligand:
                                try:
                                    ligand_coords_to_save = minimize_ligand_structure(
                                        ligand_coords_to_save,
                                        atom_names,
                                        bond_matrix=bond_matrix_for_pred,
                                        steps=self.minimize_steps,
                                        force_field=self.force_field,
                                        mode=self.minimize_mode,
                                    )
                                except Exception as e:
                                    logger.warning(f"Ligand minimization failed for {pdb_id}: {e}")
                            writepdb_ligand_complex(
                                pred_with_lig_path,
                                protein_atoms=pred_coords_with_ligand.detach(),
                                protein_seq=gt_seq,
                                ligand_atoms=ligand_coords_to_save,
                                ligand_atom_names=atom_names,
                                ligand_bond_matrix=bond_matrix_for_pred,
                            )
                        else:
                            # Fallback to protein-only if no decoded ligand available
                            logger.warning(f"No decoded ligand coords for {pdb_id}, saving protein only")
                            writepdb(pred_with_lig_path, pred_coords_with_ligand.detach(), gt_seq)

            # Compute metrics
            result = {
                "pdb_id": pdb_id,
                "length": len(gt_seq),
                "n_pocket_residues": int(pocket_mask.sum().item()),
                "n_nonpocket_residues": int(non_pocket_mask.sum().item()),
                "sequence": self.sequence_to_string(gt_seq),
                "smiles": sample.get("smiles", ""),
                # Protein-only metrics
                "tm_score_no_ligand": self.compute_tm_score(pred_coords_no_ligand, gt_coords, gt_seq, protein_mask),
                "rmsd_overall_no_ligand": self.compute_rmsd(pred_coords_no_ligand, gt_coords, protein_mask),
                "rmsd_pocket_no_ligand": self.compute_rmsd(pred_coords_no_ligand, gt_coords, pocket_mask),
                "rmsd_nonpocket_no_ligand": self.compute_rmsd(pred_coords_no_ligand, gt_coords, non_pocket_mask),
                # With-ligand metrics
                "tm_score_with_ligand": self.compute_tm_score(pred_coords_with_ligand, gt_coords, gt_seq, protein_mask),
                "rmsd_overall_with_ligand": self.compute_rmsd(pred_coords_with_ligand, gt_coords, protein_mask),
                "rmsd_pocket_with_ligand": self.compute_rmsd(pred_coords_with_ligand, gt_coords, pocket_mask),
                "rmsd_nonpocket_with_ligand": self.compute_rmsd(pred_coords_with_ligand, gt_coords, non_pocket_mask),
            }

            # Ligand placement metrics
            decoded_ligand = pred_with_ligand.get("decoded_ligand_coords")
            gt_ligand_coords = sample["ligand_coords"]
            if decoded_ligand is not None and gt_ligand_coords is not None:
                min_lig_len = min(len(decoded_ligand), len(gt_ligand_coords))
                if min_lig_len > 0:
                    pred_lig = decoded_ligand[:min_lig_len].detach().float()
                    gt_lig = gt_ligand_coords[:min_lig_len].detach().float()

                    # Raw ligand RMSD (no alignment)
                    diff = pred_lig - gt_lig
                    result["ligand_rmsd"] = float(torch.sqrt((diff**2).sum(dim=-1).mean()).item())

                    pred_centroid = pred_lig.mean(dim=0)
                    gt_centroid = gt_lig.mean(dim=0)
                    centroid_dist = (pred_centroid - gt_centroid).norm().item()
                    result["ligand_centroid_distance"] = centroid_dist

                    # Aligned ligand RMSD (align pred protein to GT, apply to ligand)
                    from lobster.metrics._generation_utils import (
                        compute_aligned_ligand_rmsd,
                        compute_protein_ligand_contacts,
                    )

                    aligned_metrics = compute_aligned_ligand_rmsd(
                        pred_coords_with_ligand,
                        gt_coords,
                        pred_lig,
                        gt_lig,
                        protein_mask=protein_mask,
                    )
                    result.update(aligned_metrics)

                    # Protein-ligand contacts (CA within 6A of ligand atoms)
                    contact_metrics = compute_protein_ligand_contacts(
                        pred_coords_with_ligand,
                        pred_lig,
                        contact_threshold=6.0,
                    )
                    result["n_protein_ligand_contacts"] = contact_metrics["n_contacts"]
                    result["frac_residues_contacting_ligand"] = contact_metrics["frac_residues_in_contact"]
                    result["n_ligand_atoms_contacted"] = contact_metrics["n_ligand_atoms_contacted"]
                    result["frac_ligand_atoms_contacted"] = contact_metrics["frac_ligand_atoms_contacted"]
                    result["ligand_contacts_protein"] = contact_metrics["n_contacts"] > 0

                    # Ligand in pocket: pred ligand contacts at least one GT pocket residue
                    if pocket_mask is not None and pocket_mask.any():
                        pocket_contact = contact_metrics["contact_mask"] & pocket_mask.bool()
                        n_pocket_contacts = int(pocket_contact.sum().item())
                        result["n_pocket_contacts"] = n_pocket_contacts
                        result["ligand_in_pocket"] = n_pocket_contacts > 0
                    else:
                        result["n_pocket_contacts"] = 0
                        result["ligand_in_pocket"] = False

            # Co-folding validation (Protenix or Boltz)
            if self.use_protenix and sample.get("smiles"):
                from lobster.metrics.pylon_client import (
                    call_cofold,
                    parse_structure_to_coords,
                    parse_mmcif_ligand_coords,
                )

                gt_seq_str = self.sequence_to_string(gt_seq)
                try:
                    protenix_out = call_cofold(
                        sequence=gt_seq_str,
                        ligand_smiles=sample["smiles"],
                        backend=self.cofold_backend,
                    )
                    confidence = protenix_out.get("confidence", {})
                    result["protenix_iptm"] = confidence.get("iptm", float("nan"))
                    result["protenix_ptm"] = confidence.get("ptm", float("nan"))
                    result["protenix_plddt"] = confidence.get("plddt", float("nan"))

                    structure_text = protenix_out.get("structure")
                    if structure_text:
                        pred_backbone = parse_structure_to_coords(structure_text)
                        min_len = min(len(pred_backbone), len(gt_coords))
                        pred_bb = pred_backbone[:min_len].cpu()
                        gt_bb = gt_coords[:min_len].cpu()
                        seq_for_tm = gt_seq[:min_len].cpu()
                        mask_for_tm = protein_mask[:min_len].cpu()
                        pocket_for_tm = pocket_mask[:min_len].cpu()

                        result["protenix_tm_score"] = self.compute_tm_score(pred_bb, gt_bb, seq_for_tm, mask_for_tm)
                        result["protenix_rmsd"] = self.compute_rmsd(pred_bb, gt_bb, mask_for_tm)
                        if pocket_for_tm.any():
                            result["protenix_rmsd_pocket"] = self.compute_rmsd(pred_bb, gt_bb, pocket_for_tm)

                        # Check Protenix ligand placement
                        try:
                            protenix_lig_coords = parse_mmcif_ligand_coords(structure_text)
                            if len(protenix_lig_coords) > 0 and len(gt_ligand_coords) > 0:
                                min_lig = min(len(protenix_lig_coords), len(gt_ligand_coords))
                                pred_l = protenix_lig_coords[:min_lig].float()
                                gt_l = gt_ligand_coords[:min_lig].cpu().float()
                                diff_l = pred_l - gt_l
                                result["protenix_ligand_rmsd"] = float(
                                    torch.sqrt((diff_l**2).sum(dim=-1).mean()).item()
                                )
                                p_centroid = pred_l.mean(dim=0)
                                g_centroid = gt_l.mean(dim=0)
                                result["protenix_ligand_centroid_dist"] = (p_centroid - g_centroid).norm().item()
                        except Exception:
                            pass
                except Exception as e:
                    logger.warning(f"Protenix co-folding failed for {pdb_id}: {e}")

            # Add best-of-N info if applicable
            if self.num_predictions > 1:
                result["best_idx_no_ligand"] = pred_no_ligand.get("best_idx")
                result["best_idx_with_ligand"] = pred_with_ligand.get("best_idx")
                if pred_no_ligand.get("all_scores"):
                    result["all_scores_no_ligand"] = str(pred_no_ligand["all_scores"])
                if pred_with_ligand.get("all_scores"):
                    result["all_scores_with_ligand"] = str(pred_with_ligand["all_scores"])

            # Add reflection info if try_reflection is enabled
            if self.try_reflection:
                result["reflected_no_ligand"] = reflected_no_ligand
                result["reflected_with_ligand"] = reflected_with_ligand

            results.append(result)

        # Log skipped samples
        if skipped_samples:
            n_protein = sum(1 for s in skipped_samples if s.get("reason") == "max_protein_length")
            n_total = sum(1 for s in skipped_samples if s.get("reason") == "max_length")
            skip_reasons = []
            if n_protein:
                skip_reasons.append(f"{n_protein} due to protein length > {self.max_protein_length}")
            if n_total:
                skip_reasons.append(f"{n_total} due to total length > {self.max_length}")
            logger.info(f"Skipped {len(skipped_samples)} samples: {', '.join(skip_reasons)}")

        # Create results DataFrame
        results_df = pd.DataFrame(results)

        # Handle empty results
        if len(results_df) == 0:
            logger.warning("No samples were successfully evaluated")
            summary = {
                "mean_tm_score_no_ligand": float("nan"),
                "mean_tm_score_with_ligand": float("nan"),
                "mean_tm_score_delta": float("nan"),
                "mean_rmsd_overall_no_ligand": float("nan"),
                "mean_rmsd_overall_with_ligand": float("nan"),
                "mean_rmsd_overall_delta": float("nan"),
                "mean_rmsd_pocket_no_ligand": float("nan"),
                "mean_rmsd_pocket_with_ligand": float("nan"),
                "mean_rmsd_pocket_delta": float("nan"),
                "mean_rmsd_nonpocket_no_ligand": float("nan"),
                "mean_rmsd_nonpocket_with_ligand": float("nan"),
                "mean_rmsd_nonpocket_delta": float("nan"),
                "n_samples": 0,
                "mean_pocket_size": float("nan"),
            }
            return {"results_df": results_df, "summary": summary}

        # Compute delta metrics (improvement from ligand)
        # For TM-score: higher is better, so positive delta = improvement
        results_df["tm_score_delta"] = results_df["tm_score_with_ligand"] - results_df["tm_score_no_ligand"]
        # For RMSD: lower is better, so negative delta = improvement
        results_df["rmsd_overall_delta"] = results_df["rmsd_overall_with_ligand"] - results_df["rmsd_overall_no_ligand"]
        results_df["rmsd_pocket_delta"] = results_df["rmsd_pocket_with_ligand"] - results_df["rmsd_pocket_no_ligand"]
        results_df["rmsd_nonpocket_delta"] = (
            results_df["rmsd_nonpocket_with_ligand"] - results_df["rmsd_nonpocket_no_ligand"]
        )

        # Compute summary statistics
        summary = {
            # TM-score (overall only)
            "mean_tm_score_no_ligand": results_df["tm_score_no_ligand"].mean(),
            "mean_tm_score_with_ligand": results_df["tm_score_with_ligand"].mean(),
            "mean_tm_score_delta": results_df["tm_score_delta"].mean(),
            "std_tm_score_delta": results_df["tm_score_delta"].std(),
            # Overall RMSD
            "mean_rmsd_overall_no_ligand": results_df["rmsd_overall_no_ligand"].mean(),
            "mean_rmsd_overall_with_ligand": results_df["rmsd_overall_with_ligand"].mean(),
            "mean_rmsd_overall_delta": results_df["rmsd_overall_delta"].mean(),
            "std_rmsd_overall_delta": results_df["rmsd_overall_delta"].std(),
            # Pocket RMSD
            "mean_rmsd_pocket_no_ligand": results_df["rmsd_pocket_no_ligand"].mean(),
            "mean_rmsd_pocket_with_ligand": results_df["rmsd_pocket_with_ligand"].mean(),
            "mean_rmsd_pocket_delta": results_df["rmsd_pocket_delta"].mean(),
            "std_rmsd_pocket_delta": results_df["rmsd_pocket_delta"].std(),
            # Non-pocket RMSD
            "mean_rmsd_nonpocket_no_ligand": results_df["rmsd_nonpocket_no_ligand"].mean(),
            "mean_rmsd_nonpocket_with_ligand": results_df["rmsd_nonpocket_with_ligand"].mean(),
            "mean_rmsd_nonpocket_delta": results_df["rmsd_nonpocket_delta"].mean(),
            "std_rmsd_nonpocket_delta": results_df["rmsd_nonpocket_delta"].std(),
            # Sample counts
            "n_samples": len(results_df),
            "mean_pocket_size": results_df["n_pocket_residues"].mean(),
        }

        # Ligand placement summary
        if "ligand_rmsd" in results_df.columns:
            summary["mean_ligand_rmsd"] = results_df["ligand_rmsd"].mean()
            summary["mean_ligand_centroid_distance"] = results_df["ligand_centroid_distance"].mean()
            if "ligand_rmsd_aligned" in results_df.columns:
                summary["mean_ligand_rmsd_aligned"] = results_df["ligand_rmsd_aligned"].mean()
                summary["mean_ligand_centroid_distance_aligned"] = results_df["ligand_centroid_distance_aligned"].mean()
            if "n_protein_ligand_contacts" in results_df.columns:
                summary["mean_protein_ligand_contacts"] = results_df["n_protein_ligand_contacts"].mean()
                summary["mean_frac_ligand_atoms_contacted"] = results_df["frac_ligand_atoms_contacted"].mean()
            if "ligand_contacts_protein" in results_df.columns:
                summary["ligand_contacts_protein_fraction"] = results_df["ligand_contacts_protein"].mean()
            if "ligand_in_pocket" in results_df.columns:
                summary["ligand_in_pocket_fraction"] = results_df["ligand_in_pocket"].mean()
                # Good fold (TM > 0.5) AND ligand in correct pocket
                good_fold = results_df["tm_score_with_ligand"] > 0.5
                summary["good_fold_and_in_pocket_fraction"] = (results_df["ligand_in_pocket"] & good_fold).mean()
            if "n_pocket_contacts" in results_df.columns:
                summary["mean_pocket_contacts"] = results_df["n_pocket_contacts"].mean()

        # Protenix summary
        if "protenix_tm_score" in results_df.columns:
            for col in [
                "protenix_tm_score",
                "protenix_rmsd",
                "protenix_rmsd_pocket",
                "protenix_iptm",
                "protenix_ptm",
                "protenix_plddt",
                "protenix_ligand_rmsd",
                "protenix_ligand_centroid_dist",
            ]:
                if col in results_df.columns:
                    summary[f"mean_{col}"] = results_df[col].mean()

        # Add reflection statistics if try_reflection is enabled
        if self.try_reflection and "reflected_no_ligand" in results_df.columns:
            summary["reflection_rate_no_ligand"] = results_df["reflected_no_ligand"].mean()
            summary["reflection_rate_with_ligand"] = results_df["reflected_with_ligand"].mean()
            summary["n_reflected_no_ligand"] = int(results_df["reflected_no_ligand"].sum())
            summary["n_reflected_with_ligand"] = int(results_df["reflected_with_ligand"].sum())

        return {"results_df": results_df, "summary": summary}

    def sequence_to_string(self, seq_tensor: Tensor) -> str:
        """Convert sequence tensor (in standard format) to string."""
        return "".join([self.standard_aa_map.get(int(s), "X") for s in seq_tensor.cpu().tolist()])
