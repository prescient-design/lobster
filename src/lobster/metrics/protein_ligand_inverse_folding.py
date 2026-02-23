"""Protein-Ligand Inverse Folding Evaluator.

Evaluates inverse folding on protein-ligand complexes with and without ligand context.
Can be used as a standalone evaluator or within a callback during training.

Key Question: Does providing ligand context improve sequence recovery for binding pocket residues?

Optional ESMFold validation: Fold designed sequences with ESMFold to check if the predicted
structure matches the ground truth (designability metric).
"""

import os
from glob import glob
from typing import TYPE_CHECKING

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

from lobster.metrics import align_and_compute_rmsd
from lobster.model.latent_generator.io import writepdb, writepdb_ligand_complex
from lobster.model.latent_generator.utils import apply_se3_augmentation_protein_ligand, minimize_ligand_structure
from lobster.model.latent_generator.utils.residue_constants import (
    convert_lobster_aa_tokenization_to_standard_aa,
)

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
    max_protein_length : int
        Maximum protein-only sequence length (default: 512). Samples with protein length
        exceeding this will be skipped entirely. Also used as the ESMFold max length when
        ESMFold is enabled.
    decode_structure : bool
        Whether to decode and save predicted structures as PDB files (default: False).
        When True, saves decoded structures for both with/without ligand conditions.
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
    save_reconstructed_input : bool
        Whether to save the reconstructed input structures (encode then decode the input
        before generation) to verify token encoding fidelity (default: False).
    use_se3_augmentation : bool
        Whether to apply random SE3 augmentation (rotation + translation) to input
        structures before encoding (default: False). This matches training behavior.
    se3_translation_scale : float
        Scale factor for random translation when SE3 augmentation is enabled (default: 1.0).
    temperature_seq : float
        Temperature for sequence sampling (default: 0.5).
    temperature_struc : float
        Temperature for structure sampling (default: 0.5).
    stochasticity_seq : int
        Stochasticity parameter for sequence sampling (default: 20).
    stochasticity_struc : int
        Stochasticity parameter for structure sampling (default: 20).
    temperature_ligand : float
        Temperature for ligand structure sampling (default: 0.5).
    stochasticity_ligand : int
        Stochasticity parameter for ligand structure sampling (default: 20).
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
    use_esmfold : bool
        Whether to validate designed sequences with ESMFold. When enabled, folds designed
        sequences and computes TM-score, RMSD, pLDDT vs ground truth structure (default: False).
    plm_fold : object, optional
        Pre-loaded LobsterPLMFold model instance. Required if use_esmfold=True.
        Load with: LobsterPLMFold(model_name="esmfold_v1", max_length=512).
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
        max_protein_length: int = 512,
        decode_structure: bool = False,
        save_gt_structure: bool = False,
        minimize_ligand: bool = False,
        minimize_mode: str = "bonds_and_angles",
        force_field: str = "MMFF94",
        minimize_steps: int = 500,
        save_reconstructed_input: bool = False,
        use_se3_augmentation: bool = False,
        se3_translation_scale: float = 1.0,
        # Generation hyperparameters
        temperature_seq: float = 0.5,
        temperature_struc: float = 0.5,
        stochasticity_seq: int = 20,
        stochasticity_struc: int = 20,
        temperature_ligand: float = 0.5,
        stochasticity_ligand: int = 20,
        inference_schedule_seq: str = "LogInferenceSchedule",
        inference_schedule_struc: str = "LinearInferenceSchedule",
        inference_schedule_ligand_atom: str | None = None,
        inference_schedule_ligand_struc: str | None = None,
        # ESMFold validation
        use_esmfold: bool = False,
        plm_fold: object | None = None,
    ):
        self.data_dir = data_dir
        self.pocket_distance_threshold = pocket_distance_threshold
        self.num_samples = num_samples
        self.num_designs = num_designs
        self.nsteps = nsteps
        self.device = device
        self.max_length = max_length
        self.max_protein_length = max_protein_length
        self.decode_structure = decode_structure
        self.save_gt_structure = save_gt_structure
        self.minimize_ligand = minimize_ligand
        self.minimize_mode = minimize_mode
        self.force_field = force_field
        self.minimize_steps = minimize_steps
        self.save_reconstructed_input = save_reconstructed_input
        self.use_se3_augmentation = use_se3_augmentation
        self.se3_translation_scale = se3_translation_scale
        # Generation hyperparameters
        self.temperature_seq = temperature_seq
        self.temperature_struc = temperature_struc
        self.stochasticity_seq = stochasticity_seq
        self.stochasticity_struc = stochasticity_struc
        self.temperature_ligand = temperature_ligand
        self.stochasticity_ligand = stochasticity_ligand
        self.inference_schedule_seq = inference_schedule_seq
        self.inference_schedule_struc = inference_schedule_struc
        self.inference_schedule_ligand_atom = inference_schedule_ligand_atom
        self.inference_schedule_ligand_struc = inference_schedule_ligand_struc
        # ESMFold validation
        self.use_esmfold = use_esmfold
        self.plm_fold = plm_fold
        if use_esmfold and plm_fold is None:
            raise ValueError(
                "plm_fold must be provided when use_esmfold=True. "
                "Load with: LobsterPLMFold(model_name='esmfold_v1', max_length=512)"
            )

        # Standard amino acid mapping (alphabetical order, matching writepdb num2aa)
        # The .pt files store sequences in this STANDARD format
        self.standard_aa_map = {
            0: "A",  # ALA
            1: "R",  # ARG
            2: "N",  # ASN
            3: "D",  # ASP
            4: "C",  # CYS
            5: "Q",  # GLN
            6: "E",  # GLU
            7: "G",  # GLY
            8: "H",  # HIS
            9: "I",  # ILE
            10: "L",  # LEU
            11: "K",  # LYS
            12: "M",  # MET
            13: "F",  # PHE
            14: "P",  # PRO
            15: "S",  # SER
            16: "T",  # THR
            17: "W",  # TRP
            18: "Y",  # TYR
            19: "V",  # VAL
            20: "X",  # UNK
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
        # Used to convert 21-token vocab model outputs to standard format
        self.lobster_to_standard = torch.tensor(
            [
                10,  # 0: L -> LEU (10)
                0,  # 1: A -> ALA (0)
                7,  # 2: G -> GLY (7)
                19,  # 3: V -> VAL (19)
                15,  # 4: S -> SER (15)
                6,  # 5: E -> GLU (6)
                1,  # 6: R -> ARG (1)
                16,  # 7: T -> THR (16)
                9,  # 8: I -> ILE (9)
                3,  # 9: D -> ASP (3)
                14,  # 10: P -> PRO (14)
                11,  # 11: K -> LYS (11)
                5,  # 12: Q -> GLN (5)
                13,  # 13: F -> PHE (13)
                2,  # 14: N -> ASN (2)
                18,  # 15: Y -> TYR (18)
                12,  # 16: M -> MET (12)
                8,  # 17: H -> HIS (8)
                17,  # 18: W -> TRP (17)
                4,  # 19: C -> CYS (4)
                20,  # 20: X -> UNK (20)
            ],
            dtype=torch.long,
            device=device,
        )

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
                    "ligand_atom_names": atom_names,  # Keep original atom names for PDB writing
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

    def decode_input_tokens(
        self,
        model: "LightningModule",
        inverse_fold_result: dict,
    ) -> dict:
        """Decode the input structure tokens from inverse folding back to coordinates.

        This uses the EXACT same tokens that were used for inverse folding,
        ensuring consistent comparison of input vs output structures.

        Parameters
        ----------
        model : LightningModule
            The Gen-UME protein-ligand model
        inverse_fold_result : dict
            Result dictionary from inverse_fold() containing:
            - input_protein_structure_logits: Tensor [L, n_tokens]
            - input_ligand_structure_tokens: Tensor [N] (optional)
            - protein_mask: Tensor [L]
            - ligand_mask: Tensor [N] (optional)

        Returns
        -------
        dict with:
            - reconstructed_protein_coords: Tensor [L, 3, 3] - decoded protein coordinates
            - reconstructed_ligand_coords: Tensor [N, 3] - decoded ligand coordinates (if available)
        """
        # Get the input tokens/logits from inverse_fold result
        protein_structure_logits = inverse_fold_result.get("input_protein_structure_logits")
        ligand_structure_tokens = inverse_fold_result.get("input_ligand_structure_tokens")
        protein_mask = inverse_fold_result.get("protein_mask")
        ligand_mask = inverse_fold_result.get("ligand_mask")

        if protein_structure_logits is None:
            logger.warning("No input_protein_structure_logits in inverse_fold result")
            return {
                "reconstructed_protein_coords": None,
                "reconstructed_ligand_coords": None,
            }

        # Add batch dimension
        protein_structure_logits = protein_structure_logits.unsqueeze(0)  # [1, L, n_tokens]
        protein_mask = protein_mask.unsqueeze(0)  # [1, L]

        # Create decode input dict
        decode_input = {
            "structure_logits": protein_structure_logits,
            "sequence_logits": torch.zeros(
                1, protein_structure_logits.shape[1], 33, device=protein_structure_logits.device
            ),
        }

        # Handle ligand if present
        ligand_mask_batched = None
        if ligand_structure_tokens is not None and ligand_mask is not None:
            ligand_structure_tokens = ligand_structure_tokens.unsqueeze(0)  # [1, N]
            ligand_mask_batched = ligand_mask.unsqueeze(0)  # [1, N]

            # Convert ligand tokens to one-hot logits
            n_tokens = model.quantizer.n_tokens if model.quantizer is not None else 4375
            ligand_structure_logits = torch.zeros(
                1, ligand_structure_tokens.shape[1], n_tokens, device=ligand_structure_tokens.device
            )
            ligand_structure_logits.scatter_(2, ligand_structure_tokens.unsqueeze(-1).long(), 1.0)
            decode_input["ligand_structure_logits"] = ligand_structure_logits

        # Decode to coordinates
        with torch.no_grad():
            decoded_x = model.decode_structure(
                decode_input,
                protein_mask,
                ligand_mask=ligand_mask_batched,
            )

        vit_output = decoded_x.get("vit_decoder")
        if isinstance(vit_output, dict):
            reconstructed_protein_coords = vit_output.get("protein_coords")
            reconstructed_ligand_coords = vit_output.get("ligand_coords")
        else:
            reconstructed_protein_coords = vit_output
            reconstructed_ligand_coords = None

        return {
            "reconstructed_protein_coords": reconstructed_protein_coords.squeeze(0)
            if reconstructed_protein_coords is not None
            else None,
            "reconstructed_ligand_coords": reconstructed_ligand_coords.squeeze(0)
            if reconstructed_ligand_coords is not None
            else None,
        }

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
            - decoded_coords: Tensor [L, 3, 3] (decoded protein structure)
            - decoded_ligand_coords: Tensor [N, 3] (decoded ligand structure, if include_ligand=True)
            - input_protein_structure_tokens: Tensor [L] (input protein structure tokens used)
            - input_protein_structure_logits: Tensor [L, n_tokens] (input protein structure logits)
            - input_ligand_structure_tokens: Tensor [N] (input ligand structure tokens, if include_ligand)
            - protein_mask: Tensor [L] (protein mask used)
            - ligand_mask: Tensor [N] (ligand mask used, if include_ligand)
        """
        # Prepare protein inputs - ensure proper dtype
        protein_coords = sample["protein_coords"].unsqueeze(0).float()
        protein_mask = sample["protein_mask"].unsqueeze(0).float()
        # Indices must be long (int64) for indexing operations
        protein_indices = sample["protein_indices"].unsqueeze(0).long()
        length = protein_coords.shape[1]

        # Prepare ligand inputs if needed
        ligand_coords = None
        ligand_mask = None
        ligand_indices = None
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
            bond_matrix = sample.get("bond_matrix")
            if bond_matrix is not None:
                bond_matrix = bond_matrix.unsqueeze(0).long()

        # Apply SE3 augmentation if enabled
        # Uses standalone function that applies SAME SE3 transform to both protein and ligand
        if self.use_se3_augmentation:
            augmented = apply_se3_augmentation_protein_ligand(
                protein_coords=protein_coords,
                protein_mask=protein_mask,
                ligand_coords=ligand_coords,
                ligand_mask=ligand_mask,
                random_se3=True,
                translation_scale=self.se3_translation_scale,
                backbone_noise=0.0,
            )
            protein_coords = augmented.protein_coords
            if ligand_coords is not None:
                ligand_coords = augmented.ligand_coords

        # Encode protein and ligand structure TOGETHER (joint encoding)
        # This allows protein-ligand interactions during encoding
        input_protein_structure_tokens = None
        input_protein_structure_logits = None
        protein_structure_embeddings = None
        ligand_structure_tokens = None
        ligand_structure_embeddings = None

        with torch.no_grad():
            if include_ligand:
                # Joint encoding using the model's encode_protein_ligand_structure method
                encoded = model.encode_protein_ligand_structure(
                    protein_coords=protein_coords,
                    protein_mask=protein_mask,
                    protein_indices=protein_indices,
                    ligand_coords=ligand_coords,
                    ligand_mask=ligand_mask,
                    ligand_indices=ligand_indices,
                    ligand_atom_types=ligand_atom_tokens.squeeze(0) if ligand_atom_tokens is not None else None,
                    bond_matrix=bond_matrix,
                )

                input_protein_structure_tokens = encoded["protein_tokens"]
                protein_structure_embeddings = encoded["protein_embeddings"]
                ligand_structure_tokens = encoded["ligand_tokens"]
                ligand_structure_embeddings = encoded["ligand_embeddings"]

                # For discrete mode decoding, convert tokens to one-hot logits
                if not model.use_continuous_structure:
                    n_tokens = model.quantizer.n_tokens if model.quantizer is not None else model.num_struc_classes
                    input_protein_structure_logits = torch.zeros(
                        *input_protein_structure_tokens.shape, n_tokens, device=input_protein_structure_tokens.device
                    )
                    input_protein_structure_logits.scatter_(
                        -1, input_protein_structure_tokens.unsqueeze(-1).long(), 1.0
                    )
            else:
                # Protein-only encoding
                protein_structure_logits, _, _ = model.encode_structure(protein_coords, protein_mask, protein_indices)
                input_protein_structure_logits = protein_structure_logits
                input_protein_structure_tokens = protein_structure_logits.argmax(dim=-1)

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

        # Generate sample (inverse folding mode)
        with torch.no_grad():
            result = model.generate_sample(
                length=length,
                num_samples=1,
                inverse_folding=True,
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
                input_structure_coords=protein_coords,
                input_mask=protein_mask,
                input_indices=protein_indices,
                # Ligand context (fixed conditioning, not to be generated)
                generate_ligand=include_ligand,
                num_atoms=num_atoms if include_ligand else 0,
                input_ligand_atom_tokens=ligand_atom_tokens,
                input_ligand_structure_tokens=ligand_structure_tokens,
                input_ligand_structure_embeddings=ligand_structure_embeddings if include_ligand else None,
                input_bond_matrix=bond_matrix,
                ligand_is_context=include_ligand,
            )

            # Decode structure to coordinates (optional)
            decoded_coords = None
            decoded_ligand_coords = None
            if self.decode_structure:
                decoded_x = model.decode_structure(
                    result,
                    protein_mask,
                    ligand_mask=ligand_mask if include_ligand else None,
                )
                vit_output = decoded_x.get("vit_decoder")
                if isinstance(vit_output, dict):
                    decoded_coords = vit_output.get("protein_coords")
                    decoded_ligand_coords = vit_output.get("ligand_coords")
                else:
                    decoded_coords = vit_output

        # Get predicted sequence
        sequence_logits = result["sequence_logits"]  # [1, L, vocab_size]
        uses_33_token_vocab = sequence_logits.shape[-1] == 33

        # Handle both 33-token and 21-token vocab formats
        # Always convert to standard format for consistency with ground truth
        if uses_33_token_vocab:
            # 33-token vocab: convert to standard (alphabetical) format
            predicted_sequence = convert_lobster_aa_tokenization_to_standard_aa(
                sequence_logits, device=sequence_logits.device
            ).squeeze(0)  # [L] in standard format
        else:
            # 21-token vocab: output is in lobster format, convert to standard
            predicted_sequence = sequence_logits.argmax(dim=-1).squeeze(0)  # [L] in lobster format
            predicted_sequence[predicted_sequence > 20] = 20  # Clamp to valid range
            predicted_sequence = self.lobster_to_standard[predicted_sequence.long()]  # Convert to standard

        # Get predicted bond matrix if available
        predicted_bond_matrix = result.get("predicted_bond_matrix")

        # Get output structure tokens for comparison with input
        output_protein_structure_tokens = result.get("structure_tokens")
        output_ligand_structure_tokens = result.get("ligand_structure_tokens")

        return {
            "predicted_sequence": predicted_sequence,  # Always in standard format
            "sequence_logits": sequence_logits.squeeze(0),
            "decoded_coords": decoded_coords.squeeze(0) if decoded_coords is not None else None,
            "decoded_ligand_coords": decoded_ligand_coords.squeeze(0) if decoded_ligand_coords is not None else None,
            "predicted_bond_matrix": predicted_bond_matrix.squeeze(0) if predicted_bond_matrix is not None else None,
            "output_protein_structure_tokens": output_protein_structure_tokens.squeeze(0)
            if output_protein_structure_tokens is not None
            else None,
            "output_ligand_structure_tokens": output_ligand_structure_tokens.squeeze(0)
            if output_ligand_structure_tokens is not None
            else None,
            # Input tokens/embeddings for reconstruction (exact same used for inverse folding)
            "input_protein_structure_tokens": input_protein_structure_tokens.squeeze(0)
            if input_protein_structure_tokens is not None
            else None,
            "input_protein_structure_logits": input_protein_structure_logits.squeeze(0)
            if input_protein_structure_logits is not None
            else None,
            "input_protein_structure_embeddings": protein_structure_embeddings.squeeze(0)
            if protein_structure_embeddings is not None
            else None,
            "input_ligand_structure_tokens": ligand_structure_tokens.squeeze(0)
            if ligand_structure_tokens is not None
            else None,
            "input_ligand_structure_embeddings": ligand_structure_embeddings.squeeze(0)
            if ligand_structure_embeddings is not None
            else None,
            "protein_mask": protein_mask.squeeze(0),
            "ligand_mask": ligand_mask.squeeze(0) if ligand_mask is not None else None,
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

    def fold_with_esmfold(
        self,
        sequence_str: str,
        gt_coords: Tensor,
        protein_mask: Tensor,
    ) -> dict:
        """Fold a designed sequence with ESMFold and compute metrics vs ground truth.

        Parameters
        ----------
        sequence_str : str
            Amino acid sequence string to fold
        gt_coords : Tensor
            [L, 3, 3] ground truth backbone coordinates (N, CA, C)
        protein_mask : Tensor
            [L] valid residue mask

        Returns
        -------
        dict with:
            - esmfold_tm_score: TM-score of ESMFold prediction vs GT
            - esmfold_rmsd: RMSD of ESMFold prediction vs GT
            - esmfold_plddt: mean pLDDT score
            - esmfold_pae: mean predicted aligned error
            - esmfold_coords: Tensor [L_valid, 3, 3] ESMFold predicted coordinates
        """
        from lobster.metrics import get_folded_structure_metrics

        if self.plm_fold is None:
            return {}

        # Tokenize sequence for ESMFold
        tokenized_input = self.plm_fold.tokenizer.encode_plus(
            sequence_str,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].to(self.device)

        # Fold with ESMFold
        with torch.no_grad():
            esmfold_outputs = self.plm_fold.model(tokenized_input)

        # Get reference structure (only valid residues)
        mask_bool = protein_mask.bool()
        ref_coords = gt_coords[mask_bool].unsqueeze(0)  # [1, L_valid, 3, 3]

        # Calculate metrics (ref_coords already filtered, so mask=None)
        folded_metrics, folded_coords = get_folded_structure_metrics(
            esmfold_outputs, ref_coords, [sequence_str], mask=None, device=self.device
        )

        def _to_float(v):
            """Convert tensor/numpy scalar to Python float."""
            if hasattr(v, "item"):
                return v.item()
            return float(v)

        return {
            "esmfold_tm_score": _to_float(folded_metrics["_tm_score"]),
            "esmfold_rmsd": _to_float(folded_metrics["_rmsd"]),
            "esmfold_plddt": _to_float(folded_metrics["_plddt"]),
            "esmfold_pae": _to_float(folded_metrics["_predicted_aligned_error"]),
            "esmfold_coords": folded_coords[0],  # [L_valid, 3, 3]
        }

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
            Directory to save designed sequences as FASTA files

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

        # Create output directory if specified
        if structure_path:
            os.makedirs(structure_path, exist_ok=True)

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

            # Save sequences and structures if structure_path provided
            if structure_path:
                gt_seq_str = self.sequence_to_string(gt_seq)
                no_ligand_seq_str = self.sequence_to_string(pred_seq_no_ligand)
                with_ligand_seq_str = self.sequence_to_string(pred_seq_with_ligand)

                # Save sequences as FASTA
                fasta_path = os.path.join(structure_path, f"{pdb_id}_sequences.fasta")
                with open(fasta_path, "w") as f:
                    f.write(f">{pdb_id}_ground_truth\n{gt_seq_str}\n")
                    f.write(f">{pdb_id}_no_ligand\n{no_ligand_seq_str}\n")
                    f.write(f">{pdb_id}_with_ligand\n{with_ligand_seq_str}\n")

                # Get coordinates and atom names
                protein_coords = sample["protein_coords"]
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

                # Save ground truth structures (optional)
                # NOTE: gt_seq from .pt files is already in standard tokenization format
                if self.save_gt_structure:
                    # Save ground truth protein structure as PDB
                    pdb_path = os.path.join(structure_path, f"{pdb_id}_protein.pdb")
                    writepdb(pdb_path, protein_coords, gt_seq)

                    # Save protein-ligand complex as PDB (ground truth)
                    complex_path = os.path.join(structure_path, f"{pdb_id}_complex.pdb")
                    writepdb_ligand_complex(
                        complex_path,
                        protein_atoms=protein_coords,
                        protein_seq=gt_seq,
                        ligand_atoms=ligand_coords,
                        ligand_atom_names=atom_names,
                        ligand_bond_matrix=bond_matrix,
                    )

                # Save reconstructed input structures (decode the SAME tokens used for inverse folding)
                # This verifies token encoding/decoding fidelity using exact same tokens
                if self.save_reconstructed_input:
                    # Use decode_input_tokens with tokens from pred_with_ligand (same tokens used for IF)
                    recon_result = self.decode_input_tokens(model, pred_with_ligand)

                    # Save reconstructed protein structure
                    recon_protein_coords = recon_result.get("reconstructed_protein_coords")
                    if recon_protein_coords is not None:
                        recon_pdb_path = os.path.join(structure_path, f"{pdb_id}_reconstructed_input_protein.pdb")
                        writepdb(recon_pdb_path, recon_protein_coords, gt_seq)

                    # Save reconstructed protein-ligand complex
                    recon_ligand_coords = recon_result.get("reconstructed_ligand_coords")
                    if recon_protein_coords is not None and recon_ligand_coords is not None:
                        # Apply minimization to reconstructed ligand if enabled
                        recon_ligand_coords_to_save = recon_ligand_coords.cpu()
                        if self.minimize_ligand:
                            try:
                                recon_ligand_coords_to_save = minimize_ligand_structure(
                                    recon_ligand_coords_to_save,
                                    atom_names,
                                    bond_matrix=bond_matrix,
                                    steps=self.minimize_steps,
                                    force_field=self.force_field,
                                    mode=self.minimize_mode,
                                )
                            except Exception as e:
                                logger.warning(f"Reconstructed ligand minimization failed for {pdb_id}: {e}")

                        recon_complex_path = os.path.join(structure_path, f"{pdb_id}_reconstructed_input_complex.pdb")
                        writepdb_ligand_complex(
                            recon_complex_path,
                            protein_atoms=recon_protein_coords,
                            protein_seq=gt_seq,
                            ligand_atoms=recon_ligand_coords_to_save,
                            ligand_atom_names=atom_names,
                            ligand_bond_matrix=bond_matrix,
                        )

                    # Log token info - tokens come directly from inverse_fold result
                    input_protein_tokens = pred_with_ligand.get("input_protein_structure_tokens")
                    input_ligand_tokens = pred_with_ligand.get("input_ligand_structure_tokens")
                    output_protein_tokens = pred_with_ligand.get("output_protein_structure_tokens")
                    output_ligand_tokens = pred_with_ligand.get("output_ligand_structure_tokens")

                    # Compute token preservation rate
                    if input_protein_tokens is not None and output_protein_tokens is not None:
                        protein_token_match = (input_protein_tokens == output_protein_tokens).float().mean().item()
                        logger.info(f"{pdb_id}: Protein structure token preservation: {protein_token_match * 100:.1f}%")
                    if input_ligand_tokens is not None and output_ligand_tokens is not None:
                        ligand_token_match = (input_ligand_tokens == output_ligand_tokens).float().mean().item()
                        logger.info(f"{pdb_id}: Ligand structure token preservation: {ligand_token_match * 100:.1f}%")

                # Save decoded protein structure (no ligand) as PDB
                # NOTE: pred_seq_no_ligand is already in standard format (converted in inverse_fold)
                decoded_coords_no_ligand = pred_no_ligand.get("decoded_coords")
                if decoded_coords_no_ligand is not None:
                    decoded_pdb_path = os.path.join(structure_path, f"{pdb_id}_decoded_no_ligand.pdb")
                    writepdb(decoded_pdb_path, decoded_coords_no_ligand, pred_seq_no_ligand)

                # Save decoded protein structure (with ligand) as PDB - include decoded ligand
                # NOTE: pred_seq_with_ligand is already in standard format (converted in inverse_fold)
                decoded_coords_with_ligand = pred_with_ligand.get("decoded_coords")
                decoded_ligand_coords = pred_with_ligand.get("decoded_ligand_coords")
                if decoded_coords_with_ligand is not None:
                    if decoded_ligand_coords is None:
                        raise ValueError(
                            f"Model did not output decoded ligand coordinates for {pdb_id}. "
                            "Check that the model supports ligand structure decoding."
                        )
                    # Use predicted bond matrix if available, otherwise fall back to GT
                    pred_bond_matrix = pred_with_ligand.get("predicted_bond_matrix")
                    bond_matrix_for_pred = pred_bond_matrix if pred_bond_matrix is not None else bond_matrix

                    # Apply minimization if enabled
                    ligand_coords_to_save = decoded_ligand_coords.cpu()
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
                    decoded_pdb_path = os.path.join(structure_path, f"{pdb_id}_decoded_with_ligand.pdb")
                    writepdb_ligand_complex(
                        decoded_pdb_path,
                        protein_atoms=decoded_coords_with_ligand,
                        protein_seq=pred_seq_with_ligand,
                        ligand_atoms=ligand_coords_to_save,
                        ligand_atom_names=atom_names,
                        ligand_bond_matrix=bond_matrix_for_pred,
                    )

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

            # ESMFold validation (fold designed sequences and compare to GT structure)
            if self.use_esmfold and self.plm_fold is not None:
                gt_coords = sample["protein_coords"]
                gt_seq_masked = gt_seq[protein_mask.bool()]

                # Build pocket mask relative to valid residues only (for pocket RMSD)
                # pocket_mask is [L] over all residues; we need it relative to valid residues
                pocket_mask_valid = pocket_mask[protein_mask.bool()]  # [L_valid]
                gt_coords_valid = gt_coords[protein_mask.bool()]  # [L_valid, 3, 3]

                def _compute_pocket_rmsd(esmfold_coords, gt_coords_v, pocket_mask_v):
                    """Compute pocket RMSD between ESMFold prediction and GT."""
                    if pocket_mask_v.sum() == 0:
                        return float("nan")
                    pred_pocket = esmfold_coords[pocket_mask_v]
                    gt_pocket = gt_coords_v[pocket_mask_v]
                    rmsd = align_and_compute_rmsd(
                        coords1=pred_pocket.detach(),
                        coords2=gt_pocket.detach(),
                        mask=None,
                        return_aligned=False,
                        device=self.device,
                    )
                    return float(rmsd)

                # Fold "no ligand" designed sequence
                no_ligand_seq_str = self.sequence_to_string(pred_seq_no_ligand[protein_mask.bool()])
                esmfold_no_ligand = self.fold_with_esmfold(no_ligand_seq_str, gt_coords, protein_mask)
                result["esmfold_tm_no_ligand"] = esmfold_no_ligand["esmfold_tm_score"]
                result["esmfold_rmsd_no_ligand"] = esmfold_no_ligand["esmfold_rmsd"]
                result["esmfold_plddt_no_ligand"] = esmfold_no_ligand["esmfold_plddt"]
                result["esmfold_pae_no_ligand"] = esmfold_no_ligand["esmfold_pae"]
                result["esmfold_rmsd_pocket_no_ligand"] = _compute_pocket_rmsd(
                    esmfold_no_ligand["esmfold_coords"], gt_coords_valid, pocket_mask_valid
                )

                if structure_path and esmfold_no_ligand.get("esmfold_coords") is not None:
                    esmfold_pdb_path = os.path.join(structure_path, f"{pdb_id}_esmfold_no_ligand.pdb")
                    writepdb(esmfold_pdb_path, esmfold_no_ligand["esmfold_coords"], gt_seq_masked)

                # Fold "with ligand" designed sequence
                with_ligand_seq_str = self.sequence_to_string(pred_seq_with_ligand[protein_mask.bool()])
                esmfold_with_ligand = self.fold_with_esmfold(with_ligand_seq_str, gt_coords, protein_mask)
                result["esmfold_tm_with_ligand"] = esmfold_with_ligand["esmfold_tm_score"]
                result["esmfold_rmsd_with_ligand"] = esmfold_with_ligand["esmfold_rmsd"]
                result["esmfold_plddt_with_ligand"] = esmfold_with_ligand["esmfold_plddt"]
                result["esmfold_pae_with_ligand"] = esmfold_with_ligand["esmfold_pae"]
                result["esmfold_rmsd_pocket_with_ligand"] = _compute_pocket_rmsd(
                    esmfold_with_ligand["esmfold_coords"], gt_coords_valid, pocket_mask_valid
                )

                if structure_path and esmfold_with_ligand.get("esmfold_coords") is not None:
                    esmfold_pdb_path = os.path.join(structure_path, f"{pdb_id}_esmfold_with_ligand.pdb")
                    writepdb(esmfold_pdb_path, esmfold_with_ligand["esmfold_coords"], gt_seq_masked)

                # Fold GT sequence for baseline comparison
                gt_seq_str = self.sequence_to_string(gt_seq_masked)
                esmfold_gt = self.fold_with_esmfold(gt_seq_str, gt_coords, protein_mask)
                result["esmfold_tm_gt"] = esmfold_gt["esmfold_tm_score"]
                result["esmfold_rmsd_gt"] = esmfold_gt["esmfold_rmsd"]
                result["esmfold_plddt_gt"] = esmfold_gt["esmfold_plddt"]
                result["esmfold_pae_gt"] = esmfold_gt["esmfold_pae"]
                result["esmfold_rmsd_pocket_gt"] = _compute_pocket_rmsd(
                    esmfold_gt["esmfold_coords"], gt_coords_valid, pocket_mask_valid
                )

                if structure_path and esmfold_gt.get("esmfold_coords") is not None:
                    esmfold_pdb_path = os.path.join(structure_path, f"{pdb_id}_esmfold_gt.pdb")
                    writepdb(esmfold_pdb_path, esmfold_gt["esmfold_coords"], gt_seq_masked)

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

        # Add ESMFold summary metrics if available
        if self.use_esmfold and "esmfold_tm_no_ligand" in results_df.columns:
            # ESMFold deltas
            results_df["esmfold_tm_delta"] = results_df["esmfold_tm_with_ligand"] - results_df["esmfold_tm_no_ligand"]
            results_df["esmfold_rmsd_delta"] = (
                results_df["esmfold_rmsd_no_ligand"] - results_df["esmfold_rmsd_with_ligand"]
            )
            results_df["esmfold_rmsd_pocket_delta"] = (
                results_df["esmfold_rmsd_pocket_no_ligand"] - results_df["esmfold_rmsd_pocket_with_ligand"]
            )
            results_df["esmfold_plddt_delta"] = (
                results_df["esmfold_plddt_with_ligand"] - results_df["esmfold_plddt_no_ligand"]
            )

            summary.update(
                {
                    # ESMFold: no ligand
                    "mean_esmfold_tm_no_ligand": results_df["esmfold_tm_no_ligand"].mean(),
                    "mean_esmfold_rmsd_no_ligand": results_df["esmfold_rmsd_no_ligand"].mean(),
                    "mean_esmfold_rmsd_pocket_no_ligand": results_df["esmfold_rmsd_pocket_no_ligand"].mean(),
                    "mean_esmfold_plddt_no_ligand": results_df["esmfold_plddt_no_ligand"].mean(),
                    "mean_esmfold_pae_no_ligand": results_df["esmfold_pae_no_ligand"].mean(),
                    # ESMFold: with ligand
                    "mean_esmfold_tm_with_ligand": results_df["esmfold_tm_with_ligand"].mean(),
                    "mean_esmfold_rmsd_with_ligand": results_df["esmfold_rmsd_with_ligand"].mean(),
                    "mean_esmfold_rmsd_pocket_with_ligand": results_df["esmfold_rmsd_pocket_with_ligand"].mean(),
                    "mean_esmfold_plddt_with_ligand": results_df["esmfold_plddt_with_ligand"].mean(),
                    "mean_esmfold_pae_with_ligand": results_df["esmfold_pae_with_ligand"].mean(),
                    # ESMFold: GT sequence baseline
                    "mean_esmfold_tm_gt": results_df["esmfold_tm_gt"].mean(),
                    "mean_esmfold_rmsd_gt": results_df["esmfold_rmsd_gt"].mean(),
                    "mean_esmfold_rmsd_pocket_gt": results_df["esmfold_rmsd_pocket_gt"].mean(),
                    "mean_esmfold_plddt_gt": results_df["esmfold_plddt_gt"].mean(),
                    "mean_esmfold_pae_gt": results_df["esmfold_pae_gt"].mean(),
                    # ESMFold: deltas (improvement from ligand context)
                    "mean_esmfold_tm_delta": results_df["esmfold_tm_delta"].mean(),
                    "mean_esmfold_rmsd_delta": results_df["esmfold_rmsd_delta"].mean(),
                    "mean_esmfold_rmsd_pocket_delta": results_df["esmfold_rmsd_pocket_delta"].mean(),
                    "mean_esmfold_plddt_delta": results_df["esmfold_plddt_delta"].mean(),
                    "std_esmfold_tm_delta": results_df["esmfold_tm_delta"].std(),
                }
            )

        return {"results_df": results_df, "summary": summary}

    def sequence_to_string(self, seq_tensor: Tensor) -> str:
        """Convert sequence tensor (in standard format) to string.

        All sequences (ground truth and predictions) are in standard tokenization format.
        """
        return "".join([self.standard_aa_map.get(int(s), "X") for s in seq_tensor.cpu().tolist()])
