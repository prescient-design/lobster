import logging
import os
from typing import Any

import boto3
import numpy as np
import torch
from biopandas.mmcif import PandasMmcif
from rdkit import Chem

from lobster.model.latent_generator.utils import residue_constants
from lobster.model.latent_generator.utils.residue_constants import (
    ELEMENT_TO_IDX,
    ELEMENT_VOCAB_EXTENDED_TO_IDX,
)

try:
    import cpdb
except ImportError:
    cpdb = None

logger = logging.getLogger(__name__)


# RDKit bond type to integer mapping (matches BOND_TYPES in residue_constants.py)
# 0=none, 1=single, 2=double, 3=triple, 4=aromatic, 5=other
RDKIT_BOND_TYPE_MAP = {
    Chem.BondType.SINGLE: 1,
    Chem.BondType.DOUBLE: 2,
    Chem.BondType.TRIPLE: 3,
    Chem.BondType.AROMATIC: 4,
}


def extract_bond_matrix(mol: Chem.Mol) -> torch.Tensor:
    """Extract bond matrix from RDKit molecule.

    Creates a symmetric matrix where entry [i,j] indicates the bond type
    between atoms i and j.

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule object with atoms and bonds.

    Returns
    -------
    torch.Tensor
        Bond type matrix of shape [N_atoms, N_atoms] with values:
        0 = no bond
        1 = single bond
        2 = double bond
        3 = triple bond
        4 = aromatic bond
        5 = other bond type
    """
    n_atoms = mol.GetNumAtoms()
    bond_matrix = torch.zeros(n_atoms, n_atoms, dtype=torch.long)

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_type = RDKIT_BOND_TYPE_MAP.get(bond.GetBondType(), 5)  # 5 = OTHER
        bond_matrix[i, j] = bond_type
        bond_matrix[j, i] = bond_type  # Symmetric

    return bond_matrix


def extract_element_indices(mol: Chem.Mol, use_extended_vocab: bool = False) -> torch.Tensor:
    """Extract element indices from RDKit molecule.

    Maps each atom's element symbol to its index in the chosen vocabulary.

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule object.
    use_extended_vocab : bool
        If True, use ELEMENT_VOCAB_EXTENDED (25 tokens) to match Gen-UME.
        If False (default), use ELEMENT_VOCAB (14 tokens) for latent generator.

    Returns
    -------
    torch.Tensor
        Element indices of shape [N_atoms] with integer values.

        If use_extended_vocab=False (default, ELEMENT_VOCAB, 14 tokens):
            0=PAD, 1=B, 2=Bi, 3=Br, 4=C, 5=Cl, 6=F, 7=H, 8=I, 9=N, 10=O, 11=P, 12=S, 13=Si

        If use_extended_vocab=True (ELEMENT_VOCAB_EXTENDED, 25 tokens):
            0=PAD, 1=MASK, 2=UNK, 3=C, 4=N, 5=O, 6=S, 7=P, 8=F, 9=Cl, 10=Br, 11=I,
            12=B, 13=Si, 14=Se, 15=As, 16=Zn, 17=Fe, 18=Cu, 19=Mg, 20=Ca, 21=Na,
            22=K, 23=Bi, 24=H
    """
    if use_extended_vocab:
        vocab = ELEMENT_VOCAB_EXTENDED_TO_IDX
        default_idx = 2  # UNK token
    else:
        vocab = ELEMENT_TO_IDX
        default_idx = 0  # PAD token (no UNK in ELEMENT_VOCAB)

    element_indices = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        idx = vocab.get(symbol, default_idx)
        element_indices.append(idx)

    return torch.tensor(element_indices, dtype=torch.long)


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

            if coords_ca.shape[0] == 0:
                continue

            ca_pos = coords_ca[0:1]
            n_pos = coords_n[0:1] if coords_n.shape[0] > 0 else ca_pos
            c_pos = coords_c[0:1] if coords_c.shape[0] > 0 else ca_pos
            backbone_coords.append(np.stack((n_pos, ca_pos, c_pos), axis=1))
            sequence.append(residue["residue_name"].values[0])
            if chain_id == "":
                chain_id = "A"
            chains.append(chain_id)
            residue_numbers.append(residue_number)

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


def load_pdb_atom14(pdb_file, add_batch_dim: bool = True) -> dict[str, Any]:
    """Convert a PDB file to a PyTorch tensor.

    Args:
        filepath (str): Path to the PDB file. Can be a local path or an S3 URI.

    Returns:
        dict: A dictionary containing the following keys:
            - 'pdb_path': The path to the PDB file.
            - 'sequence': A tensor of shape (1, N) containing the amino acid sequence as integer indices.
            - 'sequence_str': A string representing the amino acid sequence in one-letter codes.
            - 'atom14_coords': A tensor of shape (1, N, 14, 3) containing the coordinates of the atom14 atoms.
            - 'chains_ids': A tensor of shape (1, N) containing the chain IDs.
            - 'indices': A tensor of shape (1, N) containing the residue numbers.
            - 'atom14_mask': A tensor of shape (1, N) containing the mask for the coordinates.
            - 'real_chains': A tensor of shape (1, N) containing the real chain IDs.
    """
    if pdb_file.startswith("s3://"):
        # Parse S3 URI
        s3 = boto3.client("s3")
        bucket, key = pdb_file[5:].split("/", 1)

        # Download the file locally
        local_file = "/tmp/" + os.path.basename(pdb_file)
        s3.download_file(bucket, key, local_file)
        pdb_file = local_file

    # Read PDB or CIF file to dataframe
    if pdb_file.endswith(".cif"):
        pmmcif = PandasMmcif()
        df = pmmcif.read_mmcif(pdb_file).df["ATOM"]
        # rename label_atom_id to atom_name
        df = df.rename(columns={"label_atom_id": "atom_name"})
        # rename Cartn_x, Cartn_y, Cartn_z to x_coord, y_coord, z_coord
        df = df.rename(columns={"Cartn_x": "x_coord", "Cartn_y": "y_coord", "Cartn_z": "z_coord"})
        # rename auth_comp_id to residue_name
        df = df.rename(columns={"label_seq_id": "residue_number"})
        df = df.rename(columns={"auth_comp_id": "residue_name"})
        # ensure that residue_number is an integer
        df["residue_number"] = df["residue_number"].astype(int)
        df_coords = df
        group_chain = df_coords.groupby("auth_asym_id")
    else:
        df = cpdb.parse(pdb_file, df=True)
        df = df[df["record_name"] == "ATOM"]
        df_coords = df
        group_chain = df_coords.groupby("chain_id")
    atom14_coords = []
    atom14_mask = []
    sequence = []
    chains = []
    residue_numbers = []

    for chain_id, chain in group_chain:
        group_residue = chain.groupby("residue_number")
        for residue_number, residue in group_residue:
            residue_name = residue["residue_name"].iloc[0]
            # Skip non-standard residues
            if residue_name not in residue_constants.restype_name_to_atom_thin_names:
                logger.warning(
                    f"Skipping non-standard residue {residue_name} at position {residue_number} in chain {chain_id}"
                )
                continue
            atom14_atom_names = residue_constants.restype_name_to_atom_thin_names[residue_name]
            atom14_coords_list = []
            atom14_mask_list = []
            atom14_atom_names_list = []
            for atom_name in atom14_atom_names:
                if atom_name != "":
                    if atom_name in residue["atom_name"].values:
                        coords_x = residue[residue["atom_name"] == atom_name]["x_coord"].values[0]
                        coords_y = residue[residue["atom_name"] == atom_name]["y_coord"].values[0]
                        coords_z = residue[residue["atom_name"] == atom_name]["z_coord"].values[0]
                        atom14_coords_list.append(np.array([coords_x, coords_y, coords_z]))
                        atom14_mask_list.append(1)
                        atom14_atom_names_list.append(atom_name)
                    else:
                        atom14_coords_list.append(np.array([0.0, 0.0, 0.0]))
                        atom14_mask_list.append(0)
                        atom14_atom_names_list.append("")
                else:
                    atom14_coords_list.append(np.array([0.0, 0.0, 0.0]))
                    atom14_mask_list.append(0)
                    atom14_atom_names_list.append("")
            atom14_coords.append(np.array(atom14_coords_list))
            atom14_mask.append(np.array(atom14_mask_list))
            sequence.append(residue["residue_name"].values[0])
            chains.append(chain_id)
            residue_numbers.append(residue_number)
    atom14_coords = np.array(atom14_coords)
    atom14_coords = torch.tensor(atom14_coords, dtype=torch.float32)
    atom14_mask = np.array(atom14_mask)
    atom14_mask = torch.tensor(atom14_mask, dtype=torch.float32)
    residue_numbers = np.array(residue_numbers)

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
        "pdb_path": pdb_file,
        "sequence": sequence,
        "sequence_str": sequence_str,
        "atom14_coords": atom14_coords,
        "chains_ids": chains,
        "indices": residue_numbers,
        "atom14_mask": atom14_mask,
        "real_chains": real_chains,
    }
    if add_batch_dim:
        structure_data["sequence"] = structure_data["sequence"][None]
        structure_data["atom14_coords"] = structure_data["atom14_coords"][None]
        structure_data["atom14_mask"] = structure_data["atom14_mask"][None]
        structure_data["chains_ids"] = structure_data["chains_ids"][None]
        structure_data["indices"] = structure_data["indices"][None]
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


def load_ligand(
    filepath: str,
    add_batch_dim: bool = True,
    canonical_order: bool = True,
    use_extended_element_vocab: bool = False,
) -> dict[str, Any]:
    """Convert a ligand file to a PyTorch tensor.

    Parameters
    ----------
    filepath : str
        Path to the ligand file. Can be a local path or an S3 URI.
        Supports .pdb, .mol2, and .sdf formats.
    add_batch_dim : bool
        Whether to add a batch dimension to the output.
    canonical_order : bool
        Whether to reorder the atoms to the canonical order (mol2/sdf only).
    use_extended_element_vocab : bool
        If True, use ELEMENT_VOCAB_EXTENDED (25 tokens) for element_indices
        to match Gen-UME protein-ligand encoder.
        If False (default), use ELEMENT_VOCAB (14 tokens) for latent generator.

    Returns
    -------
    dict
        A dictionary containing the following keys:
            - 'pdb_path': The path to the ligand file.
            - 'atom_names': A list of strings representing the atom symbols.
            - 'atom_coords': Tensor of shape [N, 3] or [1, N, 3] with coordinates.
            - 'atom_indices': Tensor of shape [N] or [1, N] with atom indices.
            - 'mask': Tensor of shape [N] or [1, N] with validity mask.
            - 'element_indices': Tensor of shape [N] or [1, N] with element type indices
              (only for mol2/sdf files). Uses ELEMENT_VOCAB (14 tokens) by default,
              or ELEMENT_VOCAB_EXTENDED (25 tokens) if use_extended_element_vocab=True.
            - 'bond_matrix': Tensor of shape [N, N] with bond types
              (only for mol2/sdf files). Values: 0=none, 1=single, 2=double,
              3=triple, 4=aromatic, 5=other.
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

        # Extract bond matrix and element indices from RDKit molecule
        bond_matrix = extract_bond_matrix(mol)
        element_indices = extract_element_indices(mol, use_extended_vocab=use_extended_element_vocab)

        structure_data = {
            "pdb_path": filepath,
            "atom_names": atom_names,
            "atom_coords": coords,
            "atom_indices": atom_numbers,
            "mask": mask,
            "element_indices": element_indices,
            "bond_matrix": bond_matrix,
        }
        if add_batch_dim:
            structure_data["atom_coords"] = structure_data["atom_coords"][None]
            structure_data["atom_indices"] = structure_data["atom_indices"][None]
            structure_data["mask"] = structure_data["mask"][None]
            structure_data["element_indices"] = structure_data["element_indices"][None]
            # Note: bond_matrix is NOT batched as it's [N, N] and collation handles it

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
