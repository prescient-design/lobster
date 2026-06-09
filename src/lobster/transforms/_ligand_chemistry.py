"""Ligand chemistry utilities for SMILES <-> graph conversion.

This module provides functions for converting between SMILES strings and
graph representations (atom types + bond matrices) for use in Gen-UME
protein-ligand modeling.

Functions:
    smiles_to_graph: Convert SMILES string to (atom_types, bond_matrix)
    graph_to_smiles: Convert (atom_types, bond_matrix) to SMILES string
    atom_types_to_indices: Convert element strings to vocabulary indices
    indices_to_atom_types: Convert vocabulary indices to element strings
"""

import torch
from rdkit import Chem

from lobster.model.latent_generator.utils.residue_constants import (
    BOND_TYPES,
    ELEMENT_VOCAB_EXTENDED,
    ELEMENT_VOCAB_EXTENDED_TO_IDX,
)


def atom_types_to_indices(atom_types: list[str]) -> torch.Tensor:
    """Convert element strings to vocabulary indices.

    Parameters
    ----------
    atom_types : list[str]
        List of element symbols (e.g., ["C", "N", "O"]).

    Returns
    -------
    torch.Tensor
        Tensor of vocabulary indices with shape (N,).

    Examples
    --------
    >>> atom_types_to_indices(["C", "N", "O"])
    tensor([3, 4, 5])

    Notes
    -----
    Unknown elements are mapped to the UNK token (index 2).
    """
    indices = []
    unk_idx = ELEMENT_VOCAB_EXTENDED_TO_IDX["UNK"]

    for atom in atom_types:
        idx = ELEMENT_VOCAB_EXTENDED_TO_IDX.get(atom, unk_idx)
        indices.append(idx)

    return torch.tensor(indices, dtype=torch.long)


def indices_to_atom_types(indices: torch.Tensor) -> list[str]:
    """Convert vocabulary indices to element strings.

    Parameters
    ----------
    indices : torch.Tensor
        Tensor of vocabulary indices.

    Returns
    -------
    list[str]
        List of element symbols.

    Examples
    --------
    >>> indices_to_atom_types(torch.tensor([3, 4, 5]))
    ['C', 'N', 'O']
    """
    return [ELEMENT_VOCAB_EXTENDED[idx.item()] for idx in indices]


def smiles_to_graph(smiles: str) -> tuple[list[str], torch.Tensor]:
    """Convert SMILES string to graph representation.

    Parameters
    ----------
    smiles : str
        SMILES string representing a molecule.

    Returns
    -------
    tuple[list[str], torch.Tensor]
        atom_types: List of element symbols for each heavy atom.
        bond_matrix: Symmetric NxN tensor where N is number of atoms.
            Values: 0=no bond, 1=single, 2=double, 3=triple, 4=aromatic, 5=other.

    Raises
    ------
    ValueError
        If SMILES string is invalid.

    Examples
    --------
    >>> atom_types, bond_matrix = smiles_to_graph("CCO")
    >>> atom_types
    ['C', 'C', 'O']
    >>> bond_matrix.shape
    torch.Size([3, 3])
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    # Get heavy atoms (non-hydrogen)
    num_atoms = mol.GetNumAtoms()
    atom_types = []

    for atom in mol.GetAtoms():
        atom_types.append(atom.GetSymbol())

    # Build bond matrix
    bond_matrix = torch.zeros(num_atoms, num_atoms, dtype=torch.long)

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        # Map RDKit bond type to our encoding
        bond_type = bond.GetBondType()
        if bond_type == Chem.BondType.SINGLE:
            val = BOND_TYPES["SINGLE"]
        elif bond_type == Chem.BondType.DOUBLE:
            val = BOND_TYPES["DOUBLE"]
        elif bond_type == Chem.BondType.TRIPLE:
            val = BOND_TYPES["TRIPLE"]
        elif bond_type == Chem.BondType.AROMATIC:
            val = BOND_TYPES["AROMATIC"]
        else:
            val = BOND_TYPES["OTHER"]

        # Symmetric
        bond_matrix[i, j] = val
        bond_matrix[j, i] = val

    return atom_types, bond_matrix


def graph_to_smiles(
    atom_types: list[str],
    bond_matrix: torch.Tensor,
    coords: torch.Tensor | None = None,
) -> str:
    """Convert graph representation to SMILES string.

    Parameters
    ----------
    atom_types : list[str]
        List of element symbols for each atom.
    bond_matrix : torch.Tensor
        Symmetric NxN tensor of bond types.
        Values: 0=no bond, 1=single, 2=double, 3=triple, 4=aromatic, 5=other.
    coords : torch.Tensor, optional
        3D coordinates with shape (N, 3). If provided, used for stereochemistry.

    Returns
    -------
    str
        Canonical SMILES string.

    Examples
    --------
    >>> atom_types = ["C", "C", "O"]
    >>> bond_matrix = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    >>> graph_to_smiles(atom_types, bond_matrix)
    'CCO'
    """
    # Create editable molecule
    mol = Chem.RWMol()

    # Add atoms
    for elem in atom_types:
        atom = Chem.Atom(elem)
        mol.AddAtom(atom)

    # Add bonds
    num_atoms = len(atom_types)
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            bond_val = bond_matrix[i, j].item()
            if bond_val == 0:
                continue

            # Map our encoding to RDKit bond type
            if bond_val == BOND_TYPES["SINGLE"]:
                bond_type = Chem.BondType.SINGLE
            elif bond_val == BOND_TYPES["DOUBLE"]:
                bond_type = Chem.BondType.DOUBLE
            elif bond_val == BOND_TYPES["TRIPLE"]:
                bond_type = Chem.BondType.TRIPLE
            elif bond_val == BOND_TYPES["AROMATIC"]:
                bond_type = Chem.BondType.AROMATIC
            else:
                bond_type = Chem.BondType.SINGLE  # Default to single

            mol.AddBond(i, j, bond_type)

    # Convert to regular molecule
    mol = mol.GetMol()

    # Set aromaticity if we have aromatic bonds
    if (bond_matrix == BOND_TYPES["AROMATIC"]).any():
        # Mark atoms with aromatic bonds as aromatic
        for i in range(num_atoms):
            if (bond_matrix[i] == BOND_TYPES["AROMATIC"]).any():
                mol.GetAtomWithIdx(i).SetIsAromatic(True)

    # Add stereochemistry from 3D coordinates if provided
    if coords is not None:
        _assign_stereochemistry_from_coords(mol, coords)

    # Sanitize molecule
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        # If sanitization fails, try without aromaticity perception
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)

    return Chem.MolToSmiles(mol)


def _assign_stereochemistry_from_coords(mol: Chem.Mol, coords: torch.Tensor) -> None:
    """Assign stereochemistry to molecule from 3D coordinates.

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule (modified in place).
    coords : torch.Tensor
        3D coordinates with shape (N, 3).
    """
    # Create conformer
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, coord in enumerate(coords):
        conf.SetAtomPosition(i, coord.tolist())

    mol.AddConformer(conf, assignId=True)

    # Assign stereochemistry from 3D structure
    Chem.AssignStereochemistryFrom3D(mol)


def mol_to_bond_matrix(mol: Chem.Mol) -> torch.Tensor:
    """Convert RDKit molecule to bond matrix.

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule object.

    Returns
    -------
    torch.Tensor
        Symmetric NxN tensor of bond types.

    Notes
    -----
    This is useful for extracting bond matrices from SDF files.
    """
    num_atoms = mol.GetNumAtoms()
    bond_matrix = torch.zeros(num_atoms, num_atoms, dtype=torch.long)

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        bond_type = bond.GetBondType()
        if bond_type == Chem.BondType.SINGLE:
            val = BOND_TYPES["SINGLE"]
        elif bond_type == Chem.BondType.DOUBLE:
            val = BOND_TYPES["DOUBLE"]
        elif bond_type == Chem.BondType.TRIPLE:
            val = BOND_TYPES["TRIPLE"]
        elif bond_type == Chem.BondType.AROMATIC:
            val = BOND_TYPES["AROMATIC"]
        else:
            val = BOND_TYPES["OTHER"]

        bond_matrix[i, j] = val
        bond_matrix[j, i] = val

    return bond_matrix


def sdf_to_bond_matrix(sdf_content: str) -> tuple[list[str], torch.Tensor]:
    """Extract atom types and bond matrix from SDF content.

    Parameters
    ----------
    sdf_content : str
        Content of an SDF file.

    Returns
    -------
    tuple[list[str], torch.Tensor]
        atom_types: List of element symbols.
        bond_matrix: Symmetric NxN tensor of bond types.

    Raises
    ------
    ValueError
        If SDF content is invalid.
    """
    mol = Chem.MolFromMolBlock(sdf_content, removeHs=True)
    if mol is None:
        raise ValueError("Invalid SDF content")

    atom_types = [atom.GetSymbol() for atom in mol.GetAtoms()]
    bond_matrix = mol_to_bond_matrix(mol)

    return atom_types, bond_matrix
