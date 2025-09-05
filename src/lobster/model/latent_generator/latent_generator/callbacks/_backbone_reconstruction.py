from pathlib import Path

import lightning
import pandas as pd
import torch
import s3fs
import os
from loguru import logger
from lobster.model.latent_generator.latent_generator.io import writepdb, writepdb_ligand_complex
from icecream import ic
import numpy as np
from lobster.model.latent_generator.latent_generator.utils import residue_constants
from lobster.model.latent_generator.latent_generator.datasets import StructureResidueTransform
import glob

# make dictionary from index to one-letter amino acid code with residue_constants.restype_order_with_x
idx_to_aa = dict(enumerate(residue_constants.restype_order_with_x))
def get_seq_from_batch(batch):
    seq = []
    for i in range(batch.shape[0]):
        seq.append(''.join([idx_to_aa[int(j)] if int(j) in idx_to_aa else 'U' for j in batch[i]]))
    return seq

class BackboneReconstruction(lightning.Callback):
    def __init__(self,
                 structure_path: str = None,
                 target_paths: str = "/data/lisanzas/structure_tokenizer/casp/processed_atom/pdb/all/",
                 save_every_n: int = 1000):
        self.target_paths = target_paths
        self.STRUCTURE_PATH = structure_path
        self.save_every_n = save_every_n
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs(f"{self.STRUCTURE_PATH}/recon", exist_ok=True)

    def on_train_batch_end(self, trainer, tokenizer, outputs, batch, batch_idx):
        current_step = trainer.global_step
        seq = None
        x_recon_xyz = None

        if batch_idx % self.save_every_n == 0:
            #save ouputs too
            x_recon = outputs["x_recon"]

            x_recon_xyz = None
            seq = None

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
                    x_recon_element = x_recon[decoder_name]
                    x_recon_element = x_recon_element.argmax(dim=-1)
                    ligand_atom_names = [residue_constants.ELEMENT_VOCAB[int(i)] for i in x_recon_element[0]]
                else:
                    ligand_atom_names = None



            #save the pdb file
            if x_recon_xyz is not None:
                if seq is None:
                    seq = torch.zeros(x_recon_xyz.shape[1], dtype=torch.long)[None]
                filename = f"{self.STRUCTURE_PATH}recon/struc_{batch_idx}_{current_step}_gen.pdb"
                if x_recon_ligand is not None:
                    ligand_atoms = x_recon_ligand[0]
                    ligand_chain = "L"
                    ligand_resname = "LIG"
                    writepdb_ligand_complex(filename, ligand_atoms=ligand_atoms, ligand_atom_names=ligand_atom_names, ligand_chain=ligand_chain, ligand_resname=ligand_resname, protein_atoms=x_recon_xyz[0], protein_seq=seq[0])
                else:
                    writepdb(filename, x_recon_xyz[0], seq[0])
                logger.info(f"Saved {filename}")

                #save batch
                filename = f"{self.STRUCTURE_PATH}recon/struc_{batch_idx}_{current_step}_gt.pdb"
                seq = torch.zeros(batch["coords_res"].shape[1], dtype=torch.long)[None]
                if "ligand_coords" in batch:
                    ligand_atoms = batch["ligand_coords"][0]
                    ligand_atom_names = None
                    ligand_chain = "L"
                    ligand_resname = "LIG"
                    writepdb_ligand_complex(filename, ligand_atoms=ligand_atoms, ligand_atom_names=ligand_atom_names, ligand_chain=ligand_chain, ligand_resname=ligand_resname, protein_atoms=batch["coords_res"][0], protein_seq=seq[0])
                else:
                    writepdb(filename, batch["coords_res"][0], seq)
                logger.info(f"Saved {filename}")
            elif x_recon_ligand is not None:
                filename = f"{self.STRUCTURE_PATH}recon/struc_{batch_idx}_{current_step}_gen_ligand.pdb"
                ligand_atoms = x_recon_ligand[0]
                ligand_chain = "L"
                ligand_resname = "LIG"
                writepdb_ligand_complex(filename, ligand_atoms=ligand_atoms, ligand_atom_names=ligand_atom_names, ligand_chain=ligand_chain, ligand_resname=ligand_resname, protein_atoms=None, protein_seq=None)
                logger.info(f"Saved {filename}")
                filename = f"{self.STRUCTURE_PATH}recon/struc_{batch_idx}_{current_step}_gt_ligand.pdb"
                ligand_atoms = batch["ligand_coords"][0]
                ligand_atom_names = [residue_constants.ELEMENT_VOCAB[int(i)] for i in batch["ligand_element_indices"][0]]
                ligand_chain = "L"
                ligand_resname = "LIG"
                writepdb_ligand_complex(filename, ligand_atoms=ligand_atoms, ligand_atom_names=ligand_atom_names, ligand_chain=ligand_chain, ligand_resname=ligand_resname, protein_atoms=None, protein_seq=None)
