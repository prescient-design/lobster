import logging
import os

import lightning
import torch

from lobster.model.latent_generator.io import writepdb, writepdb_ligand_complex
from lobster.model.latent_generator.utils import residue_constants
from lobster.model.latent_generator.utils import minimize_ligand_structure

logger = logging.getLogger(__name__)

# make dictionary from index to one-letter amino acid code with residue_constants.restype_order_with_x
idx_to_aa = dict(enumerate(residue_constants.restype_order_with_x))


def get_element_name(idx: int, use_extended_vocab: bool = False) -> str:
    """Get element name from index, supporting both standard and extended vocabularies.

    Parameters
    ----------
    idx : int
        Element index.
    use_extended_vocab : bool
        If True, use ELEMENT_VOCAB_EXTENDED (25 tokens).
        If False, use ELEMENT_VOCAB (14 tokens).

    Returns
    -------
    str
        Element name (e.g., 'C', 'N', 'O') or 'X' for unknown.
    """
    if use_extended_vocab:
        vocab = residue_constants.ELEMENT_VOCAB_EXTENDED
    else:
        vocab = residue_constants.ELEMENT_VOCAB

    if 0 <= idx < len(vocab):
        return vocab[idx]
    else:
        return "X"  # Unknown element


def get_seq_from_batch(batch):
    seq = []
    for i in range(batch.shape[0]):
        seq.append("".join([idx_to_aa[int(j)] if int(j) in idx_to_aa else "U" for j in batch[i]]))
    return seq


class BackboneReconstruction(lightning.Callback):
    def __init__(
        self,
        structure_path: str = None,
        target_paths: str = None,
        save_every_n: int = 1000,
        max_total_files: int = 1000,
        use_extended_element_vocab: bool = False,
        minimize_ligand: bool = False,
        minimize_mode: str = "bonds_and_angles",
        force_field: str = "MMFF94",
        minimize_steps: int = 500,
    ):
        """Initialize BackboneReconstruction callback.

        Args:
            structure_path: Path to save reconstructed structures
            target_paths: Target paths (unused)
            save_every_n: Save structures every N batches
            max_total_files: Maximum total number of PDB files to keep. Older files
                will be deleted when this limit is exceeded. If None, keeps all files.
                Default: None
            use_extended_element_vocab: If True, use ELEMENT_VOCAB_EXTENDED (25 tokens)
                for mapping element indices to atom names. If False, use ELEMENT_VOCAB
                (14 tokens). Default: False
            minimize_ligand: If True, apply geometry correction to ligand structures.
                Default: False
            minimize_mode: Minimization mode - "bonds_only" or "bonds_and_angles" (recommended).
                Default: "bonds_and_angles"
            force_field: Force field for minimization - "MMFF94", "MMFF94s", "UFF", etc.
                Default: "MMFF94"
            minimize_steps: Maximum number of minimization steps. Default: 500
        """
        self.target_paths = target_paths
        self.STRUCTURE_PATH = structure_path
        self.save_every_n = save_every_n
        self.max_total_files = max_total_files
        self.use_extended_element_vocab = use_extended_element_vocab
        self.minimize_ligand = minimize_ligand
        self.minimize_mode = minimize_mode
        self.force_field = force_field
        self.minimize_steps = minimize_steps
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs(f"{self.STRUCTURE_PATH}/recon", exist_ok=True)

        if self.max_total_files is not None:
            logger.info(f"Will keep maximum {self.max_total_files} total PDB files (oldest will be deleted)")
        logger.info(f"Using {'extended' if use_extended_element_vocab else 'standard'} element vocabulary")
        if self.minimize_ligand:
            logger.info(f"Ligand minimization enabled: mode={minimize_mode}, force_field={force_field}")

    def _cleanup_old_files(self):
        """Remove oldest PDB files if total count exceeds max_total_files."""
        if self.max_total_files is None:
            return

        recon_dir = f"{self.STRUCTURE_PATH}/recon"
        if not os.path.exists(recon_dir):
            return

        # Get all PDB files with their creation times
        pdb_files = []
        for filename in os.listdir(recon_dir):
            if filename.endswith(".pdb"):
                filepath = os.path.join(recon_dir, filename)
                try:
                    mtime = os.path.getmtime(filepath)
                    pdb_files.append((filepath, mtime))
                except OSError:
                    continue

        # If we exceed the limit, delete oldest files
        if len(pdb_files) > self.max_total_files:
            # Sort by modification time (oldest first)
            pdb_files.sort(key=lambda x: x[1])

            # Calculate how many to delete
            num_to_delete = len(pdb_files) - self.max_total_files

            # Delete oldest files
            for filepath, _ in pdb_files[:num_to_delete]:
                try:
                    os.remove(filepath)
                    logger.debug(f"Deleted old PDB file: {filepath}")
                except OSError as e:
                    logger.warning(f"Failed to delete {filepath}: {e}")

            logger.info(f"Cleaned up {num_to_delete} old PDB files. Total files: {self.max_total_files}")

    def on_train_batch_end(self, trainer, tokenizer, outputs, batch, batch_idx):
        # Only save on rank 0 to avoid file I/O contention in distributed training
        if trainer.is_global_zero:
            self._save_reconstruction(trainer, outputs, batch, batch_idx, prefix="")
            self._cleanup_old_files()

    def on_validation_batch_end(self, trainer, tokenizer, outputs, batch, batch_idx, dataloader_idx=0):
        # Only save on rank 0 to avoid file I/O contention in distributed training
        if trainer.is_global_zero:
            self._save_reconstruction(trainer, outputs, batch, batch_idx, prefix="val_")
            self._cleanup_old_files()

    def _save_reconstruction(self, trainer, outputs, batch, batch_idx, prefix=""):
        current_step = trainer.global_step

        if batch_idx % self.save_every_n != 0:
            return

        # Extract reconstructions
        x_recon = outputs["x_recon"]
        x_recon_xyz = None
        x_recon_ligand = None
        x_recon_element = None

        for decoder_name in x_recon:
            if "vit_decoder" == decoder_name or "vit_decoder_simple" == decoder_name:
                x_recon_xyz = x_recon[decoder_name]
                if isinstance(x_recon_xyz, dict) and "ligand_coords" in x_recon_xyz:
                    x_recon_ligand = x_recon_xyz["ligand_coords"]
                    x_recon_xyz = x_recon_xyz["protein_coords"]
                elif isinstance(x_recon_xyz, dict) and "protein_coords_refinement" in x_recon_xyz:
                    x_recon_xyz = x_recon_xyz["protein_coords_refinement"]
                    x_recon_ligand = None
                else:
                    x_recon_ligand = None
            if "element_decoder" == decoder_name:
                x_recon_element = x_recon[decoder_name].argmax(dim=-1)

        # Determine batch size
        if x_recon_xyz is not None:
            batch_size = x_recon_xyz.shape[0]
        elif x_recon_ligand is not None:
            batch_size = x_recon_ligand.shape[0]
        else:
            return

        # Save all batch entries
        for i in range(batch_size):
            # Save reconstructed structures
            if x_recon_xyz is not None:
                # Apply mask to reconstructed protein (assume mask is in batch)
                protein_mask_i = batch.get("mask", None)
                if protein_mask_i is not None:
                    protein_mask_i = protein_mask_i[i].bool()
                    protein_coords_i = x_recon_xyz[i][protein_mask_i]
                    seq_i = torch.zeros(protein_coords_i.shape[0], dtype=torch.long)
                else:
                    protein_coords_i = x_recon_xyz[i]
                    seq_i = torch.zeros(x_recon_xyz.shape[1], dtype=torch.long)

                filename = f"{self.STRUCTURE_PATH}recon/{prefix}struc_{batch_idx}_{current_step}_gen_item{i}.pdb"

                if x_recon_ligand is not None:
                    # Apply mask to reconstructed ligand
                    ligand_mask_i = batch.get("ligand_mask", None)
                    ligand_atom_names_i = None
                    bond_matrix_i = None

                    if ligand_mask_i is not None:
                        ligand_mask_i = ligand_mask_i[i].bool()
                        ligand_coords_i = x_recon_ligand[i][ligand_mask_i]

                        # Get ligand atom names with masking
                        if x_recon_element is not None:
                            ligand_elements_masked = x_recon_element[i][ligand_mask_i]
                            ligand_atom_names_i = [
                                get_element_name(int(j), self.use_extended_element_vocab)
                                for j in ligand_elements_masked
                            ]

                        # Get bond matrix with masking if available
                        if "ligand_bond_matrix" in batch:
                            full_bond_matrix = batch["ligand_bond_matrix"][i]
                            # Apply mask to bond matrix (select rows and columns for valid atoms)
                            bond_matrix_i = full_bond_matrix[ligand_mask_i][:, ligand_mask_i]
                    else:
                        ligand_coords_i = x_recon_ligand[i]
                        if x_recon_element is not None:
                            ligand_atom_names_i = [
                                get_element_name(int(j), self.use_extended_element_vocab) for j in x_recon_element[i]
                            ]
                        if "ligand_bond_matrix" in batch:
                            bond_matrix_i = batch["ligand_bond_matrix"][i]

                    # Apply ligand minimization if enabled
                    if self.minimize_ligand and ligand_atom_names_i is not None:
                        try:
                            ligand_coords_i = minimize_ligand_structure(
                                ligand_coords_i,
                                ligand_atom_names_i,
                                bond_matrix=bond_matrix_i,
                                steps=self.minimize_steps,
                                force_field=self.force_field,
                                mode=self.minimize_mode,
                            )
                        except Exception as e:
                            logger.warning(f"Ligand minimization failed: {e}")

                    writepdb_ligand_complex(
                        filename,
                        ligand_atoms=ligand_coords_i,
                        ligand_atom_names=ligand_atom_names_i,
                        ligand_chain="L",
                        ligand_resname="LIG",
                        protein_atoms=protein_coords_i,
                        protein_seq=seq_i,
                        ligand_bond_matrix=bond_matrix_i,
                    )
                else:
                    writepdb(filename, protein_coords_i, seq_i)
                logger.info(f"Saved {filename}")

                # Save ground truth
                if "coords_res" in batch:
                    filename_gt = f"{self.STRUCTURE_PATH}recon/{prefix}struc_{batch_idx}_{current_step}_gt_item{i}.pdb"

                    # Apply mask to ground truth protein
                    if protein_mask_i is not None:
                        gt_protein_coords_i = batch["coords_res"][i][protein_mask_i]
                        seq_gt_i = torch.zeros(gt_protein_coords_i.shape[0], dtype=torch.long)
                    else:
                        gt_protein_coords_i = batch["coords_res"][i]
                        seq_gt_i = torch.zeros(batch["coords_res"].shape[1], dtype=torch.long)

                    if "ligand_coords" in batch:
                        # Apply mask to ground truth ligand
                        gt_ligand_mask_i = batch.get("ligand_mask", None)
                        gt_ligand_atom_names = None
                        gt_bond_matrix_i = None

                        if gt_ligand_mask_i is not None:
                            gt_ligand_mask_i = gt_ligand_mask_i[i].bool()
                            gt_ligand_coords_i = batch["ligand_coords"][i][gt_ligand_mask_i]

                            # Get ground truth ligand atom names with masking
                            if "ligand_element_indices" in batch:
                                gt_ligand_elements_masked = batch["ligand_element_indices"][i][gt_ligand_mask_i]
                                gt_ligand_atom_names = [
                                    get_element_name(int(j), self.use_extended_element_vocab)
                                    for j in gt_ligand_elements_masked
                                ]

                            # Get bond matrix with masking if available
                            if "ligand_bond_matrix" in batch:
                                full_bond_matrix = batch["ligand_bond_matrix"][i]
                                gt_bond_matrix_i = full_bond_matrix[gt_ligand_mask_i][:, gt_ligand_mask_i]
                        else:
                            gt_ligand_coords_i = batch["ligand_coords"][i]
                            if "ligand_element_indices" in batch:
                                gt_ligand_atom_names = [
                                    get_element_name(int(j), self.use_extended_element_vocab)
                                    for j in batch["ligand_element_indices"][i]
                                ]
                            if "ligand_bond_matrix" in batch:
                                gt_bond_matrix_i = batch["ligand_bond_matrix"][i]

                        writepdb_ligand_complex(
                            filename_gt,
                            ligand_atoms=gt_ligand_coords_i,
                            ligand_atom_names=gt_ligand_atom_names,
                            ligand_chain="L",
                            ligand_resname="LIG",
                            protein_atoms=gt_protein_coords_i,
                            protein_seq=seq_gt_i,
                            ligand_bond_matrix=gt_bond_matrix_i,
                        )
                    else:
                        writepdb(filename_gt, gt_protein_coords_i, seq_gt_i)
                    logger.info(f"Saved {filename_gt}")

            # Ligand-only case
            elif x_recon_ligand is not None:
                # Apply mask to reconstructed ligand
                ligand_mask_i = batch.get("ligand_mask", None)
                ligand_atom_names_i = None
                bond_matrix_i = None

                if ligand_mask_i is not None:
                    ligand_mask_i = ligand_mask_i[i].bool()
                    ligand_coords_recon_i = x_recon_ligand[i][ligand_mask_i]

                    # Get ligand atom names with masking
                    if x_recon_element is not None:
                        ligand_elements_masked = x_recon_element[i][ligand_mask_i]
                        ligand_atom_names_i = [
                            get_element_name(int(j), self.use_extended_element_vocab) for j in ligand_elements_masked
                        ]

                    # Get bond matrix with masking if available
                    if "ligand_bond_matrix" in batch:
                        full_bond_matrix = batch["ligand_bond_matrix"][i]
                        bond_matrix_i = full_bond_matrix[ligand_mask_i][:, ligand_mask_i]
                else:
                    ligand_coords_recon_i = x_recon_ligand[i]
                    if x_recon_element is not None:
                        ligand_atom_names_i = [
                            get_element_name(int(j), self.use_extended_element_vocab) for j in x_recon_element[i]
                        ]
                    if "ligand_bond_matrix" in batch:
                        bond_matrix_i = batch["ligand_bond_matrix"][i]

                # Apply ligand minimization if enabled
                if self.minimize_ligand and ligand_atom_names_i is not None:
                    try:
                        ligand_coords_recon_i = minimize_ligand_structure(
                            ligand_coords_recon_i,
                            ligand_atom_names_i,
                            bond_matrix=bond_matrix_i,
                            steps=self.minimize_steps,
                            force_field=self.force_field,
                            mode=self.minimize_mode,
                        )
                    except Exception as e:
                        logger.warning(f"Ligand minimization failed: {e}")

                # Save reconstructed ligand
                filename = f"{self.STRUCTURE_PATH}recon/{prefix}struc_{batch_idx}_{current_step}_gen_ligand_item{i}.pdb"
                writepdb_ligand_complex(
                    filename,
                    ligand_atoms=ligand_coords_recon_i,
                    ligand_atom_names=ligand_atom_names_i,
                    ligand_chain="L",
                    ligand_resname="LIG",
                    protein_atoms=None,
                    protein_seq=None,
                    ligand_bond_matrix=bond_matrix_i,
                )
                logger.info(f"Saved {filename}")

                # Save ground truth ligand
                if "ligand_coords" in batch:
                    filename_gt = (
                        f"{self.STRUCTURE_PATH}recon/{prefix}struc_{batch_idx}_{current_step}_gt_ligand_item{i}.pdb"
                    )

                    # Apply mask to ground truth ligand
                    gt_ligand_mask_i = batch.get("ligand_mask", None)
                    gt_ligand_atom_names = None
                    gt_bond_matrix_i = None

                    if gt_ligand_mask_i is not None:
                        gt_ligand_mask_i = gt_ligand_mask_i[i].bool()
                        gt_ligand_coords_i = batch["ligand_coords"][i][gt_ligand_mask_i]

                        # Get ground truth ligand atom names with masking
                        if "ligand_element_indices" in batch:
                            gt_ligand_elements_masked = batch["ligand_element_indices"][i][gt_ligand_mask_i]
                            gt_ligand_atom_names = [
                                get_element_name(int(j), self.use_extended_element_vocab)
                                for j in gt_ligand_elements_masked
                            ]

                        # Get bond matrix with masking if available
                        if "ligand_bond_matrix" in batch:
                            full_bond_matrix = batch["ligand_bond_matrix"][i]
                            gt_bond_matrix_i = full_bond_matrix[gt_ligand_mask_i][:, gt_ligand_mask_i]
                    else:
                        gt_ligand_coords_i = batch["ligand_coords"][i]
                        if "ligand_element_indices" in batch:
                            gt_ligand_atom_names = [
                                get_element_name(int(j), self.use_extended_element_vocab)
                                for j in batch["ligand_element_indices"][i]
                            ]
                        if "ligand_bond_matrix" in batch:
                            gt_bond_matrix_i = batch["ligand_bond_matrix"][i]

                    writepdb_ligand_complex(
                        filename_gt,
                        ligand_atoms=gt_ligand_coords_i,
                        ligand_atom_names=gt_ligand_atom_names,
                        ligand_chain="L",
                        ligand_resname="LIG",
                        protein_atoms=None,
                        protein_seq=None,
                        ligand_bond_matrix=gt_bond_matrix_i,
                    )
                    logger.info(f"Saved {filename_gt}")
