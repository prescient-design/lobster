"""Callback for decoding and saving protein-ligand complexes during training."""

import os

import lightning
import torch
from loguru import logger

from lobster.model.latent_generator.io import writepdb_ligand_complex
from lobster.model.latent_generator.utils.residue_constants import (
    ELEMENT_VOCAB_EXTENDED,
    convert_lobster_aa_tokenization_to_standard_aa,
)


class ProteinLigandDecodeCallback(lightning.Callback):
    """Callback to decode and save protein-ligand complexes during training.

    This callback saves:
    1. Decoded protein structure
    2. Decoded ligand structure (atom positions, types, bonds)
    3. Combined protein-ligand complex PDB

    Parameters
    ----------
    structure_path : str
        Base path for saving structures
    save_every_n : int
        Save structures every N batches
    save_separate : bool
        Whether to save protein and ligand separately in addition to complex
    """

    def __init__(
        self,
        structure_path: str = None,
        save_every_n: int = 1000,
        save_separate: bool = True,
    ):
        self.structure_path = structure_path
        self.save_every_n = save_every_n
        self.save_separate = save_separate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create output directories
        self.complex_dir = f"{self.structure_path}/complexes"
        os.makedirs(self.complex_dir, exist_ok=True)
        if save_separate:
            os.makedirs(f"{self.structure_path}/proteins", exist_ok=True)
            os.makedirs(f"{self.structure_path}/ligands", exist_ok=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Save decoded structures at specified intervals."""
        # Only run on rank 0 in distributed setting
        if trainer.global_rank != 0:
            return

        current_step = trainer.global_step

        if batch_idx % self.save_every_n != 0:
            return

        has_protein = outputs.get("has_protein", True)  # Default True for backward compat
        has_ligand = outputs.get("has_ligand", False)

        # Need at least decoded protein or ligand structure
        if "decoded_x" not in outputs and "decoded_ligand_x" not in outputs:
            return

        # Get decoded structures from vit_decoder (handles both protein and ligand)
        x_recon_xyz = None
        ligand_coords_from_decoder = None
        seq = None

        if "decoded_x" in outputs:
            x_recon = outputs["decoded_x"]
            for decoder_name in x_recon:
                if "vit_decoder" == decoder_name:
                    vit_output = x_recon[decoder_name]
                    # Handle both old format (tensor) and new format (dict with protein_coords/ligand_coords)
                    if isinstance(vit_output, dict):
                        x_recon_xyz = vit_output.get("protein_coords")
                        ligand_coords_from_decoder = vit_output.get("ligand_coords")
                    else:
                        x_recon_xyz = vit_output

        # Get protein sequence (if protein present)
        # Check if model uses 33-token vocab (from AminoAcidTokenizerTransform)
        uses_33_token_vocab = False
        if has_protein and x_recon_xyz is not None:
            if outputs["unmasked_x"]["sequence_logits"].shape[-1] == 33:
                uses_33_token_vocab = True
                seq = convert_lobster_aa_tokenization_to_standard_aa(
                    outputs["unmasked_x"]["sequence_logits"], device=self.device
                )
            else:
                seq = outputs["unmasked_x"]["sequence_logits"].argmax(dim=-1)
                seq[seq > 21] = 20

        # Get timesteps for filename (use ligand timesteps for ligand-only)
        if has_protein:
            t_seq = outputs["train_timesteps_seq"][0].cpu().numpy()
            t_struc = outputs["train_timesteps_struc"][0].cpu().numpy()
        else:
            # Ligand-only batch
            t_seq = 0.0
            t_struc = outputs.get("train_timesteps_ligand", outputs["train_timesteps_struc"])[0].cpu().numpy()

        # === SAVE LIGAND STRUCTURES ===
        ligand_coords = None
        ligand_atom_types = None

        if has_ligand and "decoded_ligand_x" in outputs:
            decoded_ligand = outputs["decoded_ligand_x"]
            ligand_mask = outputs.get("ligand_mask")

            # Get ligand coordinates from unified vit_decoder
            if ligand_coords_from_decoder is not None:
                ligand_coords = ligand_coords_from_decoder[0]  # First sample (decoded)
            else:
                logger.warning(
                    "No decoded ligand coordinates available - vit_decoder may not be returning ligand_coords"
                )

            # Get ligand atom types
            if "atom_types" in decoded_ligand:
                # Convert token indices to element names
                atom_indices = decoded_ligand["atom_types"][0]
                if ligand_mask is not None:
                    valid_mask = ligand_mask[0].bool()
                    atom_indices = atom_indices[valid_mask]
                ligand_atom_types = self._indices_to_atom_names(atom_indices)
            elif "ligand_atom_names" in batch:
                ligand_atom_types = batch["ligand_atom_names"][0]

        # Save complex PDB (protein + ligand)
        if has_protein and has_ligand and x_recon_xyz is not None and ligand_coords is not None:
            if ligand_atom_types is not None:
                complex_filename = (
                    f"{self.complex_dir}/complex_{batch_idx}_{current_step}_tseq_{t_seq:.2f}_tstruc_{t_struc:.2f}.pdb"
                )

                try:
                    writepdb_ligand_complex(
                        filename=complex_filename,
                        protein_atoms=x_recon_xyz[0],
                        protein_seq=seq[0],
                        protein_chain="A",
                        ligand_atoms=ligand_coords.cpu() if torch.is_tensor(ligand_coords) else ligand_coords,
                        ligand_atom_names=ligand_atom_types,
                        ligand_chain="L",
                        ligand_resname="LIG",
                    )
                    logger.info(f"Saved complex: {complex_filename}")
                except Exception as e:
                    logger.warning(f"Failed to save complex: {e}")

                # Save ground truth complex
                if "coords_res" in batch and "ligand_coords" in batch:
                    gt_filename = f"{self.complex_dir}/complex_{batch_idx}_{current_step}_gt.pdb"
                    gt_ligand_names = batch.get("ligand_atom_names", [["C"] * batch["ligand_coords"].shape[1]])[0]

                    try:
                        gt_seq = batch["sequence"][0]
                        # Handle both 33-token (from AminoAcidTokenizerTransform) and 21-token formats
                        if uses_33_token_vocab:
                            gt_seq = convert_lobster_aa_tokenization_to_standard_aa(
                                gt_seq.unsqueeze(0), device=self.device
                            )[0]
                        else:
                            gt_seq = gt_seq.clone()
                            gt_seq[gt_seq > 21] = 20
                        writepdb_ligand_complex(
                            filename=gt_filename,
                            protein_atoms=batch["coords_res"][0],
                            protein_seq=gt_seq,
                            protein_chain="A",
                            ligand_atoms=batch["ligand_coords"][0].cpu(),
                            ligand_atom_names=gt_ligand_names,
                            ligand_chain="L",
                            ligand_resname="LIG",
                        )
                        logger.info(f"Saved ground truth complex: {gt_filename}")
                    except Exception as e:
                        logger.warning(f"Failed to save GT complex: {e}")

        # Save ligand-only if no protein (GEOM dataset)
        elif not has_protein and has_ligand and ligand_coords is not None and ligand_atom_types is not None:
            ligand_filename = f"{self.structure_path}/ligands/ligand_{batch_idx}_{current_step}_tlig_{t_struc:.2f}.pdb"
            try:
                writepdb_ligand_complex(
                    filename=ligand_filename,
                    protein_atoms=None,  # No protein
                    protein_seq=None,
                    ligand_atoms=ligand_coords.cpu() if torch.is_tensor(ligand_coords) else ligand_coords,
                    ligand_atom_names=ligand_atom_types,
                    ligand_chain="L",
                    ligand_resname="LIG",
                )
                logger.info(f"Saved ligand-only: {ligand_filename}")
            except Exception as e:
                logger.warning(f"Failed to save ligand: {e}")

            # Save ground truth ligand
            if "ligand_coords" in batch:
                gt_filename = f"{self.structure_path}/ligands/ligand_{batch_idx}_{current_step}_gt.pdb"
                gt_ligand_names = batch.get("ligand_atom_names", [["C"] * batch["ligand_coords"].shape[1]])[0]
                try:
                    writepdb_ligand_complex(
                        filename=gt_filename,
                        protein_atoms=None,
                        protein_seq=None,
                        ligand_atoms=batch["ligand_coords"][0].cpu(),
                        ligand_atom_names=gt_ligand_names,
                        ligand_chain="L",
                        ligand_resname="LIG",
                    )
                    logger.info(f"Saved ground truth ligand: {gt_filename}")
                except Exception as e:
                    logger.warning(f"Failed to save GT ligand: {e}")

        # Save protein-only if requested or no ligand
        if has_protein and x_recon_xyz is not None and (self.save_separate or not has_ligand):
            from lobster.model.latent_generator.io import writepdb

            protein_filename = (
                f"{self.structure_path}/proteins/protein_{batch_idx}_{current_step}_"
                f"tseq_{t_seq:.2f}_tstruc_{t_struc:.2f}.pdb"
            )
            try:
                writepdb(protein_filename, x_recon_xyz[0], seq[0])
                logger.info(f"Saved protein: {protein_filename}")
            except Exception as e:
                logger.warning(f"Failed to save protein: {e}")

    def _indices_to_atom_names(self, indices: torch.Tensor) -> list[str]:
        """Convert element indices to atom names."""
        atom_names = []
        for i, idx in enumerate(indices.cpu().tolist()):
            # ELEMENT_VOCAB_EXTENDED is a list, index directly
            if 0 <= idx < len(ELEMENT_VOCAB_EXTENDED):
                element = ELEMENT_VOCAB_EXTENDED[idx]
                # Skip special tokens
                if element in ("PAD", "MASK", "UNK"):
                    element = "C"  # Default to carbon
            else:
                element = "C"  # Default to carbon
            # Create unique atom name: element + number
            atom_names.append(f"{element}{i + 1}")

        return atom_names
