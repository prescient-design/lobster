"""Ligand-Conditioned Protein Generation Evaluator.

Evaluates the model's ability to generate proteins that bind to a given ligand,
starting from scratch (no protein structure or sequence input).

The core metric is **self-consistency**: the model generates both a sequence and
a structure; we then fold the generated sequence with ESMFold and measure how
well the ESMFold-predicted structure agrees with the model-decoded structure.

Metrics:
- scTM (self-consistency TM-score): TM-score between decoded and ESMFold structures
- scRMSD: RMSD between decoded and ESMFold structures
- Pocket scTM / scRMSD: same, restricted to residues near the decoded ligand
- pLDDT: ESMFold confidence score
- PAE: ESMFold predicted aligned error
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
from lobster.model.latent_generator.utils.residue_constants import (
    convert_lobster_aa_tokenization_to_standard_aa,
    restype_order_with_x_inv,
)

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


class LigandConditionedProteinGenerationEvaluator:
    """Evaluates ligand-conditioned protein generation via self-consistency.

    Given only a ligand (atom types + bond matrix), the model generates a protein
    (both sequence and structure) from scratch. The generated sequence is then
    folded with ESMFold, and the self-consistency between the model-decoded
    structure and the ESMFold-predicted structure is measured.

    Parameters
    ----------
    data_dir : str
        Path to directory containing *_ligand.pt files.
    length : int
        Length of the protein to generate (number of residues).
    pocket_distance_threshold : float
        Distance threshold (angstrom) for defining binding pocket residues
        on the decoded structure relative to decoded ligand coordinates.
    num_samples : int, optional
        Limit number of samples to evaluate (None = all).
    nsteps : int
        Number of diffusion steps for generation.
    device : str
        Device for computation.
    max_length : int
        Maximum combined sequence length (protein + ligand) to process.
    temperature_seq : float
        Temperature for sequence sampling.
    temperature_struc : float
        Temperature for structure sampling.
    stochasticity_seq : int
        Stochasticity parameter for sequence sampling.
    stochasticity_struc : int
        Stochasticity parameter for structure sampling.
    temperature_ligand : float
        Temperature for ligand structure sampling.
    stochasticity_ligand : int
        Stochasticity parameter for ligand structure sampling.
    ligand_context_mode : str
        How to provide ligand context: "atom_bond_only" or "structure_tokens".
    inference_schedule_seq : str
        Inference schedule for sequence generation.
    inference_schedule_struc : str
        Inference schedule for structure generation.
    inference_schedule_ligand_atom : str, optional
        Inference schedule for ligand atom token generation.
    inference_schedule_ligand_struc : str, optional
        Inference schedule for ligand structure token generation.
    save_structures : bool
        Whether to save generated structures as PDB files.
    num_designs : int
        Number of designs to generate per ligand. The best design (by scTM)
        is selected for reporting.
    minimize_ligand : bool
        Whether to apply geometry correction to decoded ligand structures.
    minimize_mode : str
        Minimization mode: "bonds_only", "bonds_and_angles", "local", or "full".
    force_field : str
        Force field for minimization: "MMFF94", "MMFF94s", "UFF", etc.
    minimize_steps : int
        Maximum number of minimization steps.
    plm_fold : object
        Pre-loaded LobsterPLMFold model instance for ESMFold prediction.
    """

    def __init__(
        self,
        data_dir: str,
        length: int = 100,
        pocket_distance_threshold: float = 5.0,
        num_samples: int | None = None,
        num_designs: int = 10,
        nsteps: int = 100,
        device: str = "cuda",
        max_length: int = 512,
        temperature_seq: float = 0.5,
        temperature_struc: float = 0.5,
        stochasticity_seq: int = 20,
        stochasticity_struc: int = 20,
        temperature_ligand: float = 0.5,
        stochasticity_ligand: int = 20,
        ligand_context_mode: str = "atom_bond_only",
        inference_schedule_seq: str = "LogInferenceSchedule",
        inference_schedule_struc: str = "LinearInferenceSchedule",
        inference_schedule_ligand_atom: str | None = None,
        inference_schedule_ligand_struc: str | None = None,
        save_structures: bool = False,
        minimize_ligand: bool = False,
        minimize_mode: str = "bonds_and_angles",
        force_field: str = "MMFF94",
        minimize_steps: int = 500,
        plm_fold: object = None,
    ):
        self.data_dir = data_dir
        self.length = length
        self.pocket_distance_threshold = pocket_distance_threshold
        self.num_samples = num_samples
        self.num_designs = num_designs
        self.nsteps = nsteps
        self.device = device
        self.max_length = max_length
        self.temperature_seq = temperature_seq
        self.temperature_struc = temperature_struc
        self.stochasticity_seq = stochasticity_seq
        self.stochasticity_struc = stochasticity_struc
        self.temperature_ligand = temperature_ligand
        self.stochasticity_ligand = stochasticity_ligand
        self.ligand_context_mode = ligand_context_mode
        self.inference_schedule_seq = inference_schedule_seq
        self.inference_schedule_struc = inference_schedule_struc
        self.inference_schedule_ligand_atom = inference_schedule_ligand_atom
        self.inference_schedule_ligand_struc = inference_schedule_ligand_struc
        self.save_structures = save_structures
        self.minimize_ligand = minimize_ligand
        self.minimize_mode = minimize_mode
        self.force_field = force_field
        self.minimize_steps = minimize_steps
        self.plm_fold = plm_fold

        if plm_fold is None:
            raise ValueError(
                "plm_fold is required for self-consistency evaluation. "
                "Load with: LobsterPLMFold(model_name='esmfold_v1', max_length=512)"
            )

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
            if len(name) >= 2 and name[:2] in self.element_to_idx:
                elem = name[:2]
            elif name[0] in self.element_to_idx:
                elem = name[0]
            else:
                elem = name[0].upper()
            idx = self.element_to_idx.get(elem, 2)
            indices.append(idx)
        return torch.tensor(indices, dtype=torch.long, device=self.device)

    def load_test_set(self) -> list[dict]:
        """Load ligand data from the test directory.

        Only ligand .pt files are required. Protein .pt files are used only to
        derive sample IDs (via the *_ligand.pt naming convention).

        Returns list of dicts with:
        - ligand_id: str
        - ligand_coords: Tensor [N_atoms, 3]
        - ligand_atom_types: Tensor [N_atoms]
        - ligand_atom_names: list[str] or None
        - ligand_mask: Tensor [N_atoms]
        - ligand_indices: Tensor [N_atoms]
        - bond_matrix: Tensor [N_atoms, N_atoms] (if available)
        """
        ligand_files = sorted(glob(os.path.join(self.data_dir, "*_ligand.pt")))

        if not ligand_files:
            raise ValueError(f"No ligand files found in {self.data_dir}")

        if self.num_samples is not None:
            ligand_files = ligand_files[: self.num_samples]

        logger.info(f"Loading {len(ligand_files)} ligand samples from {self.data_dir}")

        samples = []
        for lf in tqdm(ligand_files, desc="Loading samples"):
            ligand_id = os.path.basename(lf).replace("_ligand.pt", "")
            ligand_data = torch.load(lf, weights_only=False, map_location=self.device)

            ligand_coords = ligand_data.get(
                "atom_coords",
                ligand_data.get("coords", ligand_data.get("ligand_coords")),
            )

            if ligand_coords is None:
                logger.warning(f"Missing ligand coordinates for {ligand_id}, skipping")
                continue

            atom_names = ligand_data.get("atom_names")
            if atom_names is not None and isinstance(atom_names, list):
                ligand_atom_types = self._atom_names_to_indices(atom_names)
            else:
                ligand_atom_types = ligand_data.get(
                    "element_indices",
                    ligand_data.get(
                        "ligand_element_indices",
                        torch.full(
                            (ligand_coords.shape[0],),
                            3,
                            dtype=torch.long,
                            device=self.device,
                        ),
                    ),
                )

            ligand_mask = ligand_data.get(
                "mask",
                ligand_data.get(
                    "ligand_mask",
                    torch.ones(ligand_coords.shape[0], device=self.device),
                ),
            )
            ligand_indices = ligand_data.get(
                "atom_indices",
                ligand_data.get(
                    "indices",
                    ligand_data.get(
                        "ligand_indices",
                        torch.arange(ligand_coords.shape[0], device=self.device),
                    ),
                ),
            )
            bond_matrix = ligand_data.get("bond_matrix")

            samples.append(
                {
                    "ligand_id": ligand_id,
                    "ligand_coords": ligand_coords,
                    "ligand_atom_types": ligand_atom_types,
                    "ligand_atom_names": atom_names,
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
    ) -> Tensor:
        """Compute pocket mask on the decoded structure.

        A residue is in the pocket if its CA atom is within
        pocket_distance_threshold of any decoded ligand atom.

        Returns boolean mask [L] where True indicates pocket residues.
        """
        if protein_coords.dim() == 3:
            ca_coords = protein_coords[:, 1, :]
        else:
            ca_coords = protein_coords

        distances = torch.cdist(ca_coords.unsqueeze(0), ligand_coords.unsqueeze(0)).squeeze(0)
        min_distances = distances.min(dim=1).values
        return min_distances < self.pocket_distance_threshold

    def generate_protein(
        self,
        model: "LightningModule",
        sample: dict,
    ) -> dict:
        """Generate protein sequence and structure conditioned on ligand.

        No protein information is provided. The model generates both sequence
        and structure from noise, conditioned on the ligand.

        Parameters
        ----------
        model : LightningModule
            The Gen-UME protein-ligand model.
        sample : dict
            Sample dictionary from load_test_set().

        Returns
        -------
        dict with:
            - predicted_sequence: Tensor [L] (in standard AA format)
            - sequence_logits: Tensor [L, vocab_size]
            - decoded_coords: Tensor [L, 3, 3]
            - decoded_ligand_coords: Tensor [N, 3] or None
        """
        length = self.length

        ligand_atom_tokens = sample["ligand_atom_types"].unsqueeze(0).long()
        num_atoms = len(sample["ligand_atom_types"])
        bond_matrix = sample.get("bond_matrix")
        if bond_matrix is not None:
            bond_matrix = bond_matrix.unsqueeze(0).long()

        ligand_structure_tokens = None
        ligand_structure_embeddings = None

        if self.ligand_context_mode == "structure_tokens":
            ligand_coords = sample["ligand_coords"].float()
            ligand_mask_t = sample["ligand_mask"].float()
            ligand_indices = sample["ligand_indices"].long()

            # Center ligand at the origin so the generated protein
            # (which starts from noise around the origin) is spatially
            # close to the ligand.
            valid_mask = ligand_mask_t.bool()
            if valid_mask.any():
                centroid = ligand_coords[valid_mask].mean(dim=0, keepdim=True)
                ligand_coords = ligand_coords - centroid

            ligand_coords = ligand_coords.unsqueeze(0)
            ligand_mask_t = ligand_mask_t.unsqueeze(0)
            ligand_indices = ligand_indices.unsqueeze(0)

            with torch.no_grad():
                encode_result = model.encode_ligand_structure(
                    ligand_coords,
                    ligand_mask_t,
                    ligand_indices,
                    return_continuous=True,
                )
                ligand_structure_tokens = encode_result[0]
                ligand_structure_embeddings = encode_result[2]

        inference_schedule_seq_class = _get_inference_schedule_class(self.inference_schedule_seq)
        inference_schedule_struc_class = _get_inference_schedule_class(self.inference_schedule_struc)
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

        with torch.no_grad():
            result = model.generate_sample(
                length=length,
                num_samples=1,
                inverse_folding=False,
                forward_folding=False,
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
                generate_ligand=True,
                num_atoms=num_atoms,
                input_ligand_atom_tokens=ligand_atom_tokens,
                input_ligand_structure_tokens=ligand_structure_tokens,
                input_ligand_structure_embeddings=ligand_structure_embeddings,
                input_bond_matrix=bond_matrix,
                ligand_is_context=(self.ligand_context_mode == "structure_tokens"),
            )

            protein_mask_batch = torch.ones((1, length), device=self.device)
            ligand_mask_batch = torch.ones((1, num_atoms), device=self.device)
            decoded_x = model.decode_structure(
                result,
                protein_mask_batch,
                ligand_mask=ligand_mask_batch,
            )
            decoded_coords = None
            decoded_ligand_coords = None
            vit_output = decoded_x.get("vit_decoder")
            if isinstance(vit_output, dict):
                decoded_coords = vit_output.get("protein_coords")
                decoded_ligand_coords = vit_output.get("ligand_coords")
            else:
                decoded_coords = vit_output

        sequence_logits = result["sequence_logits"]
        uses_33_token_vocab = sequence_logits.shape[-1] == 33

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
            "decoded_coords": (decoded_coords.squeeze(0) if decoded_coords is not None else None),
            "decoded_ligand_coords": (decoded_ligand_coords.squeeze(0) if decoded_ligand_coords is not None else None),
        }

    def fold_with_esmfold(self, sequence_str: str) -> dict:
        """Fold a sequence with ESMFold and return predicted coords + confidence.

        Returns
        -------
        dict with:
            - esmfold_coords: Tensor [L, 3, 3] (N, CA, C backbone)
            - plddt: float (mean pLDDT)
            - pae: float (mean predicted aligned error)
        """
        tokenized_input = self.plm_fold.tokenizer.encode_plus(
            sequence_str,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].to(self.device)

        with torch.no_grad():
            outputs = self.plm_fold.model(tokenized_input)

        esmfold_coords = outputs["positions"][-1][0, :, :3, :]  # [L, 3, 3]
        plddt = outputs["plddt"].mean().item()
        pae = outputs["predicted_aligned_error"].mean().item()

        return {
            "esmfold_coords": esmfold_coords,
            "plddt": plddt,
            "pae": pae,
        }

    def compute_tm_score(
        self,
        coords1: Tensor,
        coords2: Tensor,
        sequence: Tensor,
        mask: Tensor | None = None,
    ) -> float:
        """Compute TM-score between two structures."""
        if mask is not None:
            mask = mask.bool()
            coords1 = coords1[mask]
            coords2 = coords2[mask]
            sequence = sequence[mask]

        if len(coords1) < 3:
            return float("nan")

        sequence_str = "".join([restype_order_with_x_inv.get(int(s), "X") for s in sequence.cpu().tolist()])

        ca1 = coords1[:, 1, :].detach().cpu().numpy()
        ca2 = coords2[:, 1, :].detach().cpu().numpy()

        tm_out = tm_align(ca1, ca2, sequence_str, sequence_str)
        return tm_out.tm_norm_chain1

    def compute_rmsd(
        self,
        coords1: Tensor,
        coords2: Tensor,
        mask: Tensor | None = None,
    ) -> float:
        """Compute RMSD between two structures (Kabsch-aligned)."""
        if mask is not None:
            mask = mask.bool()
            coords1 = coords1[mask]
            coords2 = coords2[mask]

        if len(coords1) == 0:
            return float("nan")

        rmsd = align_and_compute_rmsd(
            coords1=coords1.detach(),
            coords2=coords2.detach(),
            mask=None,
            return_aligned=False,
            device=coords1.device,
        )
        return float(rmsd)

    def compute_contact_metrics(
        self,
        protein_coords: Tensor,
        ligand_coords: Tensor,
        contact_threshold: float = 4.5,
    ) -> dict:
        """Compute protein-ligand contact statistics.

        Parameters
        ----------
        protein_coords : Tensor [L, 3, 3]
            Backbone coords (N, CA, C) per residue.
        ligand_coords : Tensor [N, 3]
            Ligand atom coords.
        contact_threshold : float
            Distance cutoff in angstrom for defining a contact.

        Returns
        -------
        dict with contact metrics.
        """
        # Use CA atoms for residue-level contacts
        if protein_coords.dim() == 3:
            ca_coords = protein_coords[:, 1, :]
        else:
            ca_coords = protein_coords

        # Pairwise distances: [n_residues, n_ligand_atoms]
        dists = torch.cdist(ca_coords.unsqueeze(0), ligand_coords.unsqueeze(0)).squeeze(0)
        min_dist_per_residue = dists.min(dim=1).values
        min_dist_per_ligand_atom = dists.min(dim=0).values

        n_residues = ca_coords.shape[0]
        n_ligand_atoms = ligand_coords.shape[0]

        residues_in_contact = (min_dist_per_residue < contact_threshold).sum().item()
        ligand_atoms_in_contact = (min_dist_per_ligand_atom < contact_threshold).sum().item()
        n_contacts = (dists < contact_threshold).sum().item()

        return {
            "n_contacts": n_contacts,
            "n_residues_in_contact": int(residues_in_contact),
            "frac_residues_in_contact": residues_in_contact / n_residues,
            "n_ligand_atoms_in_contact": int(ligand_atoms_in_contact),
            "frac_ligand_atoms_in_contact": ligand_atoms_in_contact / n_ligand_atoms,
            "min_protein_ligand_dist": float(dists.min().item()),
            "mean_min_dist_per_residue": float(min_dist_per_residue.mean().item()),
        }

    def sequence_to_string(self, seq_tensor: Tensor) -> str:
        """Convert sequence tensor (standard format) to string."""
        return "".join([self.standard_aa_map.get(int(s), "X") for s in seq_tensor.cpu().tolist()])

    def _evaluate_single_design(
        self,
        model: "LightningModule",
        sample: dict,
        ligand_id: str,
        ligand_length: int,
    ) -> dict | None:
        """Generate one design for a ligand and compute its metrics.

        Returns a dict of metrics, or None if generation failed.
        """
        gen_result = self.generate_protein(model, sample)
        pred_seq = gen_result["predicted_sequence"]
        decoded_coords = gen_result["decoded_coords"]
        decoded_ligand_coords = gen_result["decoded_ligand_coords"]

        if decoded_coords is None:
            return None

        # Minimize decoded ligand geometry if enabled
        if self.minimize_ligand and decoded_ligand_coords is not None:
            atom_names = sample.get("ligand_atom_names")
            if atom_names is None:
                idx_to_element = {v: k for k, v in self.element_to_idx.items()}
                ligand_types = sample["ligand_atom_types"]
                atom_names = [
                    f"{idx_to_element.get(int(t), 'C')}{i + 1}" for i, t in enumerate(ligand_types.cpu().tolist())
                ]
            try:
                decoded_ligand_coords = minimize_ligand_structure(
                    decoded_ligand_coords.cpu(),
                    atom_names,
                    bond_matrix=sample.get("bond_matrix"),
                    steps=self.minimize_steps,
                    force_field=self.force_field,
                    mode=self.minimize_mode,
                ).to(self.device)
            except Exception as e:
                logger.warning(f"Ligand minimization failed for {ligand_id}: {e}")

        # Fold generated sequence with ESMFold
        seq_str = self.sequence_to_string(pred_seq)
        esm_result = self.fold_with_esmfold(seq_str)
        esmfold_coords = esm_result["esmfold_coords"]

        # Compute pocket mask on the decoded structure
        pocket_mask = None
        n_pocket = 0
        if decoded_ligand_coords is not None:
            pocket_mask = self.compute_binding_pocket(decoded_coords, decoded_ligand_coords)
            n_pocket = int(pocket_mask.sum().item())

        # Contact metrics between protein and ligand
        contact_metrics = {}
        if decoded_ligand_coords is not None:
            contact_metrics = self.compute_contact_metrics(decoded_coords, decoded_ligand_coords)

        result = {
            "ligand_id": ligand_id,
            "protein_length": self.length,
            "ligand_length": ligand_length,
            "n_pocket_residues": n_pocket,
            "n_contacts": contact_metrics.get("n_contacts", 0),
            "n_residues_in_contact": contact_metrics.get("n_residues_in_contact", 0),
            "frac_residues_in_contact": contact_metrics.get("frac_residues_in_contact", 0.0),
            "n_ligand_atoms_in_contact": contact_metrics.get("n_ligand_atoms_in_contact", 0),
            "frac_ligand_atoms_in_contact": contact_metrics.get("frac_ligand_atoms_in_contact", 0.0),
            "min_protein_ligand_dist": contact_metrics.get("min_protein_ligand_dist", float("nan")),
            "scTM": self.compute_tm_score(decoded_coords, esmfold_coords, pred_seq),
            "scRMSD": self.compute_rmsd(decoded_coords, esmfold_coords),
            "plddt": esm_result["plddt"],
            "pae": esm_result["pae"],
            "sequence": seq_str,
        }

        if pocket_mask is not None and n_pocket > 0:
            result["pocket_scTM"] = self.compute_tm_score(decoded_coords, esmfold_coords, pred_seq, pocket_mask)
            result["pocket_scRMSD"] = self.compute_rmsd(decoded_coords, esmfold_coords, pocket_mask)
        else:
            result["pocket_scTM"] = float("nan")
            result["pocket_scRMSD"] = float("nan")

        # Attach tensors for optional structure saving (not serialized to CSV)
        result["_pred_seq"] = pred_seq
        result["_decoded_coords"] = decoded_coords
        result["_decoded_ligand_coords"] = decoded_ligand_coords
        result["_esmfold_coords"] = esmfold_coords

        return result

    def evaluate(
        self,
        model: "LightningModule",
        samples: list[dict] | None = None,
        structure_path: str | None = None,
    ) -> dict:
        """Run full self-consistency evaluation on the test set.

        For each ligand, ``num_designs`` proteins are generated. The design
        with the highest scTM is selected as the representative for that
        ligand.  All per-design results are kept in the returned DataFrame
        (with a ``design_idx`` column); summary statistics are computed over
        best-per-ligand rows only.

        Returns
        -------
        dict with:
            - results_df: DataFrame with per-sample, per-design results
            - summary: dict with aggregated metrics (best design per ligand)
        """
        model.eval()
        model.to(self.device)

        if samples is None:
            samples = self.load_test_set()

        if structure_path:
            os.makedirs(structure_path, exist_ok=True)

        all_results = []
        skipped_samples = []

        for sample in tqdm(samples, desc="Evaluating ligand-conditioned generation"):
            ligand_id = sample["ligand_id"]
            ligand_length = len(sample["ligand_coords"])
            total_length = self.length + ligand_length

            if total_length > self.max_length:
                logger.warning(
                    f"Skipping {ligand_id}: total length {total_length} "
                    f"(protein: {self.length}, ligand: {ligand_length}) "
                    f"exceeds max_length {self.max_length}"
                )
                skipped_samples.append(
                    {
                        "ligand_id": ligand_id,
                        "ligand_length": ligand_length,
                        "total_length": total_length,
                        "reason": "max_length",
                    }
                )
                continue

            # Generate num_designs proteins and evaluate each
            design_results = []
            for design_idx in range(self.num_designs):
                result = self._evaluate_single_design(
                    model,
                    sample,
                    ligand_id,
                    ligand_length,
                )
                if result is None:
                    logger.warning(f"No decoded coordinates for {ligand_id} design {design_idx}, skipping design")
                    continue
                result["design_idx"] = design_idx
                design_results.append(result)

            if not design_results:
                logger.warning(f"All {self.num_designs} designs failed for {ligand_id}")
                continue

            # Select best design by scTM (highest)
            best = max(design_results, key=lambda r: r["scTM"])
            for r in design_results:
                r["is_best"] = r is best

            # Save all designs' structures
            if structure_path and self.save_structures:
                for r in design_results:
                    design_suffix = f"_d{r['design_idx']}"
                    self._save_outputs(
                        structure_path,
                        f"{ligand_id}{design_suffix}",
                        sample,
                        r["_pred_seq"],
                        r["_decoded_coords"],
                        r["_decoded_ligand_coords"],
                        r["_esmfold_coords"],
                    )

            # Strip tensor fields before collecting results
            for r in design_results:
                for key in (
                    "_pred_seq",
                    "_decoded_coords",
                    "_decoded_ligand_coords",
                    "_esmfold_coords",
                ):
                    r.pop(key, None)

            all_results.extend(design_results)

        if skipped_samples:
            logger.info(f"Skipped {len(skipped_samples)} samples due to total length > {self.max_length}")

        results_df = pd.DataFrame(all_results)

        if len(results_df) == 0:
            logger.warning("No samples were successfully evaluated")
            return {"results_df": results_df, "summary": self._empty_summary()}

        summary = self._compute_summary(results_df)
        return {"results_df": results_df, "summary": summary}

    def _save_outputs(
        self,
        structure_path: str,
        ligand_id: str,
        sample: dict,
        pred_seq: Tensor,
        decoded_coords: Tensor,
        decoded_ligand_coords: Tensor | None,
        esmfold_coords: Tensor,
    ):
        """Save generated structures to disk."""
        atom_names = sample.get("ligand_atom_names")
        if atom_names is None:
            idx_to_element = {v: k for k, v in self.element_to_idx.items()}
            ligand_types = sample["ligand_atom_types"]
            atom_names = [
                f"{idx_to_element.get(int(t), 'C')}{i + 1}" for i, t in enumerate(ligand_types.cpu().tolist())
            ]
        bond_matrix = sample.get("bond_matrix")

        # Save FASTA
        seq_str = self.sequence_to_string(pred_seq)
        fasta_path = os.path.join(structure_path, f"{ligand_id}_generated.fasta")
        with open(fasta_path, "w") as f:
            f.write(f">{ligand_id}_generated\n{seq_str}\n")

        # Save model-decoded structure
        if decoded_ligand_coords is not None:
            writepdb_ligand_complex(
                os.path.join(structure_path, f"{ligand_id}_decoded.pdb"),
                protein_atoms=decoded_coords,
                protein_seq=pred_seq,
                ligand_atoms=decoded_ligand_coords,
                ligand_atom_names=atom_names,
                ligand_bond_matrix=bond_matrix,
            )
        else:
            writepdb(
                os.path.join(structure_path, f"{ligand_id}_decoded.pdb"),
                decoded_coords,
                pred_seq,
            )

        # Save ESMFold-predicted structure
        writepdb(
            os.path.join(structure_path, f"{ligand_id}_esmfold.pdb"),
            esmfold_coords,
            pred_seq,
        )

    def _empty_summary(self) -> dict:
        """Return summary dict with NaN values for empty results."""
        return {
            "n_ligands": 0,
            "num_designs": self.num_designs,
            "protein_length": self.length,
            "mean_n_contacts": float("nan"),
            "mean_frac_residues_in_contact": float("nan"),
            "mean_frac_ligand_atoms_in_contact": float("nan"),
            "mean_min_protein_ligand_dist": float("nan"),
            "mean_scTM": float("nan"),
            "mean_scRMSD": float("nan"),
            "mean_pocket_scTM": float("nan"),
            "mean_pocket_scRMSD": float("nan"),
            "mean_plddt": float("nan"),
            "mean_pae": float("nan"),
        }

    def _compute_summary(self, results_df: pd.DataFrame) -> dict:
        """Compute aggregated summary statistics from results DataFrame.

        Summary metrics are computed over the best design per ligand only
        (selected by highest scTM).
        """
        best_df = results_df[results_df["is_best"]].copy()
        n_total_designs = len(results_df)
        n_ligands = len(best_df)

        return {
            "n_ligands": n_ligands,
            "n_total_designs": n_total_designs,
            "num_designs": self.num_designs,
            "protein_length": self.length,
            "mean_n_contacts": best_df["n_contacts"].mean(),
            "std_n_contacts": best_df["n_contacts"].std(),
            "mean_n_residues_in_contact": best_df["n_residues_in_contact"].mean(),
            "mean_frac_residues_in_contact": best_df["frac_residues_in_contact"].mean(),
            "mean_frac_ligand_atoms_in_contact": best_df["frac_ligand_atoms_in_contact"].mean(),
            "mean_min_protein_ligand_dist": best_df["min_protein_ligand_dist"].mean(),
            "std_min_protein_ligand_dist": best_df["min_protein_ligand_dist"].std(),
            "mean_scTM": best_df["scTM"].mean(),
            "std_scTM": best_df["scTM"].std(),
            "median_scTM": best_df["scTM"].median(),
            "mean_scRMSD": best_df["scRMSD"].mean(),
            "std_scRMSD": best_df["scRMSD"].std(),
            "median_scRMSD": best_df["scRMSD"].median(),
            "mean_pocket_scTM": best_df["pocket_scTM"].mean(),
            "std_pocket_scTM": best_df["pocket_scTM"].std(),
            "mean_pocket_scRMSD": best_df["pocket_scRMSD"].mean(),
            "std_pocket_scRMSD": best_df["pocket_scRMSD"].std(),
            "mean_plddt": best_df["plddt"].mean(),
            "std_plddt": best_df["plddt"].std(),
            "mean_pae": best_df["pae"].mean(),
            "std_pae": best_df["pae"].std(),
            "mean_pocket_size": best_df["n_pocket_residues"].mean(),
        }
