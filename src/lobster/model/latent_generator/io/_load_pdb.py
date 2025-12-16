import logging
import os
from typing import Any

import boto3
import numpy as np
import torch
from biopandas.mmcif import PandasMmcif
from rdkit import Chem

from lobster.model.latent_generator.utils import residue_constants

try:
    import cpdb
except ImportError:
    cpdb = None

logger = logging.getLogger(__name__)

aa_3to1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


def load_pdb(filepath: str, add_batch_dim: bool = True) -> dict[str, Any] | None:
    """Convert a PDB file to a PyTorch tensor.

    Args:
        filepath (str): Path to the PDB file. Can be a local path or an S3 URI.

    Returns:
        dict: A dictionary containing the following keys:
            - 'pdb_path': The path to the PDB file.
            - 'sequence': A tensor of shape (1, N) containing the amino acid sequence as integer indices.
            - 'sequence_str': A string representing the amino acid sequence in one-letter codes.
            - 'coords_res': A tensor of shape (1, N, 3, 3) containing the coordinates of the backbone atoms.
            - 'chains_ids': A tensor of shape (1, N) containing the chain IDs.
            - 'indices': A tensor of shape (1, N) containing the residue numbers.
            - 'mask': A tensor of shape (1, N) containing the mask for the coordinates.

    """
    import lobster

    lobster.ensure_package("cpdb", group="struct-gpu (or --extra struct-cpu)")

    if filepath.startswith("s3://"):
        # Parse S3 URI
        s3 = boto3.client("s3")
        bucket, key = filepath[5:].split("/", 1)

        # Download the file locally
        local_file = "/tmp/" + os.path.basename(filepath)
        s3.download_file(bucket, key, local_file)
        filepath = local_file

    # Read PDB to dataframe
    if filepath.endswith(".cif"):
        pmmcif = PandasMmcif()
        df = pmmcif.read_mmcif(filepath).df["ATOM"]
        # rename label_atom_id to atom_name
        df = df.rename(columns={"label_atom_id": "atom_name"})
        df_coords = df[df["atom_name"].isin(["C", "N", "CA"])]
        # rename Cartn_x, Cartn_y, Cartn_z to x_coord, y_coord, z_coord
        df_coords = df_coords.rename(columns={"Cartn_x": "x_coord", "Cartn_y": "y_coord", "Cartn_z": "z_coord"})
        # rename auth_comp_id to residue_name
        df_coords = df_coords.rename(columns={"label_seq_id": "residue_number"})
        df_coords = df_coords.rename(columns={"auth_comp_id": "residue_name"})
        # ensure that residue_number is an integer
        df_coords["residue_number"] = df_coords["residue_number"].astype(int)
        group_chain = df_coords.groupby("auth_asym_id")
    else:
        df = cpdb.parse(filepath, df=True)
        df = df[df["record_name"] == "ATOM"]
        df_coords = df[df["atom_name"].isin(["C", "N", "CA"])]
        group_chain = df_coords.groupby("chain_id")

    backbone_coords = []
    sequence = []
    chains = []
    residue_numbers = []

    for chain_id, chain in group_chain:
        group_residue = chain.groupby("residue_number")
        for residue_number, residue in group_residue:
            x_coords_ca = residue[residue["atom_name"] == "CA"]["x_coord"].values
            y_coords_ca = residue[residue["atom_name"] == "CA"]["y_coord"].values
            z_coords_ca = residue[residue["atom_name"] == "CA"]["z_coord"].values
            coords_ca = np.column_stack((x_coords_ca, y_coords_ca, z_coords_ca))

            x_coords_n = residue[residue["atom_name"] == "N"]["x_coord"].values
            y_coords_n = residue[residue["atom_name"] == "N"]["y_coord"].values
            z_coords_n = residue[residue["atom_name"] == "N"]["z_coord"].values
            coords_n = np.column_stack((x_coords_n, y_coords_n, z_coords_n))

            x_coords_c = residue[residue["atom_name"] == "C"]["x_coord"].values
            y_coords_c = residue[residue["atom_name"] == "C"]["y_coord"].values
            z_coords_c = residue[residue["atom_name"] == "C"]["z_coord"].values
            coords_c = np.column_stack((x_coords_c, y_coords_c, z_coords_c))

            if coords_ca.shape[0] > 1:
                coords_ca = coords_ca[0:1]
                logger.info(
                    f"Warning: {filepath} and residue {residue_number} and chain {chain_id} has multiple CA atoms, taking the first one"
                )

            if coords_ca.shape[0] == coords_n.shape[0] and coords_ca.shape[0] == coords_c.shape[0]:
                backbone_coords.append(np.stack((coords_n[0:1], coords_ca[0:1], coords_c[0:1]), axis=1))
                sequence.append(residue["residue_name"].values[0])
                # if chain_id is an empty string, set it to 'A'
                if chain_id == "":
                    chain_id = "A"
                chains.append(chain_id)
                residue_numbers.append(residue_number)
            else:
                continue

    try:
        backbone_coords = np.stack(backbone_coords)
    except Exception as e:
        logger.error(f"Error in {filepath} and backbone_coords {backbone_coords}: {e}")
        return None

    backbone_coords = torch.tensor(backbone_coords, dtype=torch.float32).squeeze()
    mask = torch.ones(backbone_coords.shape[0], dtype=torch.float32)

    # Convert 3-letter codes to 1-letter codes
    sequence_1letter = [aa_3to1.get(aa, "X") for aa in sequence]

    # Create the string sequence
    sequence_str = "".join(sequence_1letter)

    # Convert to tensor indices
    sequence = [residue_constants.restype_order_with_x[aa] for aa in sequence_1letter]
    sequence = torch.tensor(sequence, dtype=torch.int32)

    # get ord of chains but make sure chain is a character
    chains = [ord(chain[0]) for chain in chains]
    real_chains = torch.tensor(chains, dtype=torch.int32)

    # renumber residue_numbers such that when the chain changes, the residue_numbers are continuous+200
    residue_numbers = torch.tensor(residue_numbers, dtype=torch.int32)
    chain_changes = np.diff(chains, prepend=chains[0]) != 0
    chains = np.cumsum(chain_changes) * 200
    chains = torch.tensor(chains)
    residue_numbers = residue_numbers + chains

    structure_data = {
        "pdb_path": filepath,
        "sequence": sequence,
        "sequence_str": sequence_str,
        "coords_res": backbone_coords,
        "chains_ids": chains,
        "indices": residue_numbers,
        "mask": mask,
        "real_chains": real_chains,
    }

    if add_batch_dim:
        structure_data["sequence"] = structure_data["sequence"][None]
        structure_data["coords_res"] = structure_data["coords_res"][None]
        structure_data["chains_ids"] = structure_data["chains_ids"][None]
        structure_data["indices"] = structure_data["indices"][None]
        structure_data["mask"] = structure_data["mask"][None]
        structure_data["real_chains"] = structure_data["real_chains"][None]
    return structure_data


def reorder_molecule(mol, new_order):
    """
    Create a new molecule with atoms reordered according to new_order.
    new_order[i] gives the original index of atom that should be at position i.
    """
    # Create a new molecule
    new_mol = Chem.RWMol()

    # Add atoms in the new order
    atom_map = {}  # maps old atom idx to new atom idx
    for new_idx, old_idx in enumerate(new_order):
        old_atom = mol.GetAtomWithIdx(old_idx)
        new_atom_idx = new_mol.AddAtom(old_atom)
        atom_map[old_idx] = new_atom_idx

    # Add bonds in the new order
    for bond in mol.GetBonds():
        begin_old = bond.GetBeginAtomIdx()
        end_old = bond.GetEndAtomIdx()
        begin_new = atom_map[begin_old]
        end_new = atom_map[end_old]
        bond_type = bond.GetBondType()
        new_mol.AddBond(begin_new, end_new, bond_type)

    # Copy conformer if it exists
    if mol.GetNumConformers() > 0:
        old_conf = mol.GetConformer()
        # Create a new conformer with the same number of atoms
        new_conf = Chem.Conformer(len(new_order))

        # Copy 3D coordinates
        for new_idx, old_idx in enumerate(new_order):
            pos = old_conf.GetAtomPosition(old_idx)
            new_conf.SetAtomPosition(new_idx, pos)

        # Add the conformer to the new molecule
        new_mol.AddConformer(new_conf)

    return new_mol.GetMol()


def load_ligand(filepath: str, add_batch_dim: bool = True, canonical_order: bool = True) -> dict[str, Any]:
    """Convert a ligand file to a PyTorch tensor.

    Args:
        filepath (str): Path to the ligand file. Can be a local path or an S3 URI.
                       Supports .pdb, .mol2, and .sdf formats.
        add_batch_dim (bool): Whether to add a batch dimension to the output.
        canonical_order (bool): Whether to reorder the atoms to the canonical order.

    Returns:
        dict: A dictionary containing the following keys:
            - 'pdb_path': The path to the ligand file. Could be .pdb or .mol2 or .sdf
            - 'atom_names': A list of strings representing the atom names.
            - 'atom_coords': A tensor of shape (1, N, 3) containing the coordinates of the ligand atoms.
            - 'atom_indices': A tensor of shape (1, N) containing the atom indices.
            - 'mask': A tensor of shape (1, N) containing the mask for the coordinates.
    """
    if filepath.startswith("s3://"):
        # Parse S3 URI
        s3 = boto3.client("s3")
        bucket, key = filepath[5:].split("/", 1)

        # Download the file locally
        local_file = "/tmp/" + os.path.basename(filepath)
        s3.download_file(bucket, key, local_file)
        filepath = local_file

    # Determine file format and parse accordingly
    if filepath.endswith(".mol2") or filepath.endswith(".sdf"):
        # Load using RDKit
        if filepath.endswith(".sdf"):
            mol = Chem.SDMolSupplier(filepath)[0]  # Get first molecule
        elif filepath.endswith(".mol2"):
            mol = Chem.MolFromMol2File(filepath)

        if canonical_order:
            Chem.MolToSmiles(mol)
            canonical_order = mol.GetPropsAsDict(True, True)["_smilesAtomOutputOrder"]
            mol = reorder_molecule(mol, canonical_order)

        if mol is None:
            raise ValueError(f"Could not parse molecule from {filepath}")

        # Get conformer (3D coordinates)
        conf = mol.GetConformer()

        coords = []
        atom_names = []
        atom_numbers = []

        for i, atom in enumerate(mol.GetAtoms()):
            pos = conf.GetAtomPosition(i)
            coords.append([pos.x, pos.y, pos.z])
            atom_names.append(atom.GetSymbol())
            atom_numbers.append(i)

        coords = torch.tensor(coords, dtype=torch.float32)
        atom_numbers = torch.tensor(atom_numbers, dtype=torch.int32)
        mask = torch.ones(coords.shape[0], dtype=torch.float32)

        structure_data = {
            "pdb_path": filepath,
            "atom_names": atom_names,
            "atom_coords": coords,
            "atom_indices": atom_numbers,
            "mask": mask,
        }
        if add_batch_dim:
            structure_data["atom_coords"] = structure_data["atom_coords"][None]
            structure_data["atom_indices"] = structure_data["atom_indices"][None]
            structure_data["mask"] = structure_data["mask"][None]

        return structure_data

    else:
        # Original PDB parsing logic
        # Read PDB to dataframe
        df = cpdb.parse(filepath, df=True)
        # only ligands
        df = df[df["record_name"] == "HETATM"]
        # remove waters
        df = df[df["residue_name"] != "HOH"]
        # reindex
        df = df.reset_index(drop=True)
        # remove metals
        # df = df[~df['atom_name'].isin(['ZN', 'MG', 'CA', 'FE', 'CL', 'NA', 'K'])]

        coords = []
        atom_names = []
        atom_numbers = []

        for index, row in df.iterrows():
            if index == 0:
                residue_number = row["residue_number"]
            else:
                if row["residue_number"] != residue_number:
                    index += 200
            x_coord = row["x_coord"]
            y_coord = row["y_coord"]
            z_coord = row["z_coord"]
            coords.append([x_coord, y_coord, z_coord])
            atom_names.append(row["atom_name"])
            atom_numbers.append(index)

        coords = torch.tensor(coords, dtype=torch.float32)
        atom_numbers = torch.tensor(atom_numbers, dtype=torch.int32)
        mask = torch.ones(coords.shape[0], dtype=torch.float32)

        structure_data = {
            "pdb_path": filepath,
            "atom_names": atom_names,
            "atom_coords": coords,
            "atom_indices": atom_numbers,
            "mask": mask,
        }

        if add_batch_dim:
            structure_data["atom_coords"] = structure_data["atom_coords"][None]
            structure_data["atom_indices"] = structure_data["atom_indices"][None]
            structure_data["mask"] = structure_data["mask"][None]

        return structure_data
