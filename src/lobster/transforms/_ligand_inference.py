"""Inference transforms for ligand processing in Gen-UME.

This module provides transforms for preparing ligand inputs during inference
and reconstructing SMILES from model outputs.

Functions:
    smiles_to_ligand_input: Convert SMILES to model input tensors
    sdf_to_ligand_input: Convert SDF to model input tensors
    reconstruct_smiles: Convert model outputs to SMILES strings
"""

import torch
from rdkit import Chem
from torch import Tensor

from lobster.model.latent_generator.utils.residue_constants import (
    ELEMENT_VOCAB_EXTENDED_TO_IDX,
)

from ._ligand_chemistry import (
    atom_types_to_indices,
    graph_to_smiles,
    indices_to_atom_types,
    mol_to_bond_matrix,
    smiles_to_graph,
)


def smiles_to_ligand_input(
    smiles: str | list[str],
    max_atoms: int | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, Tensor]:
    """Convert SMILES string(s) to model input tensors.

    Parameters
    ----------
    smiles : str or list[str]
        SMILES string or list of SMILES strings.
    max_atoms : int, optional
        Maximum number of atoms (for padding). If None, uses the max
        in the batch without padding.
    device : torch.device or str
        Device to place tensors on.

    Returns
    -------
    dict[str, Tensor]
        Dictionary containing:
        - ligand_atom_input_ids: [B, N_atoms] atom type indices
        - ligand_mask: [B, N_atoms] valid atom mask
        - bond_matrix: [B, N_atoms, N_atoms] bond types
        - atom_types: list[list[str]] original atom type strings
        - smiles: list[str] original SMILES strings

    Examples
    --------
    >>> inputs = smiles_to_ligand_input("CCO")
    >>> inputs["ligand_atom_input_ids"].shape
    torch.Size([1, 3])
    >>> inputs = smiles_to_ligand_input(["CCO", "CC(=O)O"])
    >>> inputs["ligand_atom_input_ids"].shape
    torch.Size([2, 4])
    """
    # Handle single SMILES
    if isinstance(smiles, str):
        smiles = [smiles]

    # batch_size = len(smiles)

    # Parse all SMILES
    all_atom_types = []
    all_bond_matrices = []

    for smi in smiles:
        atom_types, bond_matrix = smiles_to_graph(smi)
        all_atom_types.append(atom_types)
        all_bond_matrices.append(bond_matrix)

    # Determine max atoms
    if max_atoms is None:
        max_atoms = max(len(atoms) for atoms in all_atom_types)

    # Pad and stack
    ligand_atom_ids = []
    ligand_masks = []
    padded_bond_matrices = []

    pad_idx = ELEMENT_VOCAB_EXTENDED_TO_IDX["PAD"]

    for atom_types, bond_matrix in zip(all_atom_types, all_bond_matrices):
        n_atoms = len(atom_types)

        # Convert atom types to indices
        atom_indices = atom_types_to_indices(atom_types)

        # Pad atom indices
        padded_atoms = torch.full((max_atoms,), pad_idx, dtype=torch.long)
        padded_atoms[:n_atoms] = atom_indices
        ligand_atom_ids.append(padded_atoms)

        # Create mask
        mask = torch.zeros(max_atoms)
        mask[:n_atoms] = 1.0
        ligand_masks.append(mask)

        # Pad bond matrix
        padded_bonds = torch.zeros(max_atoms, max_atoms, dtype=torch.long)
        padded_bonds[:n_atoms, :n_atoms] = bond_matrix
        padded_bond_matrices.append(padded_bonds)

    return {
        "ligand_atom_input_ids": torch.stack(ligand_atom_ids).to(device),
        "ligand_mask": torch.stack(ligand_masks).to(device),
        "bond_matrix": torch.stack(padded_bond_matrices).to(device),
        "atom_types": all_atom_types,
        "smiles": smiles,
    }


def sdf_to_ligand_input(
    sdf_path: str | list[str],
    max_atoms: int | None = None,
    device: torch.device | str = "cpu",
    include_coords: bool = True,
) -> dict[str, Tensor]:
    """Convert SDF file(s) to model input tensors.

    Parameters
    ----------
    sdf_path : str or list[str]
        Path to SDF file or list of paths.
    max_atoms : int, optional
        Maximum number of atoms (for padding).
    device : torch.device or str
        Device to place tensors on.
    include_coords : bool
        Whether to include 3D coordinates.

    Returns
    -------
    dict[str, Tensor]
        Dictionary containing:
        - ligand_atom_input_ids: [B, N_atoms] atom type indices
        - ligand_mask: [B, N_atoms] valid atom mask
        - bond_matrix: [B, N_atoms, N_atoms] bond types
        - ligand_coords: [B, N_atoms, 3] 3D coordinates (if include_coords)
        - atom_types: list[list[str]] original atom type strings
        - smiles: list[str] SMILES derived from SDF

    Examples
    --------
    >>> inputs = sdf_to_ligand_input("ligand.sdf")
    >>> inputs["ligand_atom_input_ids"].shape
    torch.Size([1, N])
    """
    # Handle single path
    if isinstance(sdf_path, str):
        sdf_path = [sdf_path]

    # batch_size = len(sdf_path)

    # Parse all SDF files
    all_atom_types = []
    all_bond_matrices = []
    all_coords = []
    all_smiles = []

    for path in sdf_path:
        # Read SDF file
        supplier = Chem.SDMolSupplier(path, removeHs=True)
        mol = next(supplier)

        if mol is None:
            raise ValueError(f"Failed to parse SDF: {path}")

        # Get atom types
        atom_types = [atom.GetSymbol() for atom in mol.GetAtoms()]
        all_atom_types.append(atom_types)

        # Get bond matrix
        bond_matrix = mol_to_bond_matrix(mol)
        all_bond_matrices.append(bond_matrix)

        # Get coordinates
        if include_coords and mol.GetNumConformers() > 0:
            conf = mol.GetConformer()
            coords = torch.tensor(
                [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())],
                dtype=torch.float32,
            )
            all_coords.append(coords)
        elif include_coords:
            # No conformer, create zeros
            all_coords.append(torch.zeros(len(atom_types), 3))

        # Get SMILES
        smiles = Chem.MolToSmiles(mol)
        all_smiles.append(smiles)

    # Determine max atoms
    if max_atoms is None:
        max_atoms = max(len(atoms) for atoms in all_atom_types)

    # Pad and stack
    ligand_atom_ids = []
    ligand_masks = []
    padded_bond_matrices = []
    padded_coords = []

    pad_idx = ELEMENT_VOCAB_EXTENDED_TO_IDX["PAD"]

    for i, (atom_types, bond_matrix) in enumerate(zip(all_atom_types, all_bond_matrices)):
        n_atoms = len(atom_types)

        # Convert atom types to indices
        atom_indices = atom_types_to_indices(atom_types)

        # Pad atom indices
        padded_atoms = torch.full((max_atoms,), pad_idx, dtype=torch.long)
        padded_atoms[:n_atoms] = atom_indices
        ligand_atom_ids.append(padded_atoms)

        # Create mask
        mask = torch.zeros(max_atoms)
        mask[:n_atoms] = 1.0
        ligand_masks.append(mask)

        # Pad bond matrix
        padded_bonds = torch.zeros(max_atoms, max_atoms, dtype=torch.long)
        padded_bonds[:n_atoms, :n_atoms] = bond_matrix
        padded_bond_matrices.append(padded_bonds)

        # Pad coordinates
        if include_coords:
            coords = all_coords[i]
            padded_coord = torch.zeros(max_atoms, 3)
            padded_coord[:n_atoms] = coords
            padded_coords.append(padded_coord)

    result = {
        "ligand_atom_input_ids": torch.stack(ligand_atom_ids).to(device),
        "ligand_mask": torch.stack(ligand_masks).to(device),
        "bond_matrix": torch.stack(padded_bond_matrices).to(device),
        "atom_types": all_atom_types,
        "smiles": all_smiles,
    }

    if include_coords:
        result["ligand_coords"] = torch.stack(padded_coords).to(device)

    return result


def reconstruct_smiles(
    ligand_atom_logits: Tensor,
    bond_logits: Tensor,
    ligand_mask: Tensor | None = None,
    ligand_coords: Tensor | None = None,
    temperature: float = 1.0,
) -> list[str]:
    """Reconstruct SMILES strings from model outputs.

    Parameters
    ----------
    ligand_atom_logits : Tensor
        Atom type logits with shape [B, N_atoms, atom_vocab_size].
    bond_logits : Tensor
        Bond type logits with shape [B, N_atoms, N_atoms, num_bond_types].
    ligand_mask : Tensor, optional
        Valid atom mask with shape [B, N_atoms].
    ligand_coords : Tensor, optional
        3D coordinates with shape [B, N_atoms, 3] for stereochemistry.
    temperature : float
        Temperature for sampling (1.0 = argmax).

    Returns
    -------
    list[str]
        Reconstructed SMILES strings for each sample in batch.

    Examples
    --------
    >>> smiles = reconstruct_smiles(atom_logits, bond_logits, mask)
    >>> print(smiles[0])
    'CCO'
    """
    batch_size = ligand_atom_logits.shape[0]
    reconstructed_smiles = []

    for b in range(batch_size):
        # Get predicted atom types
        if temperature == 1.0:
            atom_indices = ligand_atom_logits[b].argmax(dim=-1)
        else:
            probs = torch.softmax(ligand_atom_logits[b] / temperature, dim=-1)
            atom_indices = torch.multinomial(probs, num_samples=1).squeeze(-1)

        # Get predicted bond matrix
        bond_matrix = bond_logits[b].argmax(dim=-1)

        # Apply mask if provided
        if ligand_mask is not None:
            mask = ligand_mask[b].bool()
            n_valid = mask.sum().item()
            atom_indices = atom_indices[:n_valid]
            bond_matrix = bond_matrix[:n_valid, :n_valid]
        else:
            n_valid = atom_indices.shape[0]

        # Skip if no valid atoms
        if n_valid == 0:
            reconstructed_smiles.append("")
            continue

        # Convert indices to atom types
        atom_types = indices_to_atom_types(atom_indices)

        # Filter out special tokens
        valid_atoms = []
        valid_indices = []
        for i, atom in enumerate(atom_types):
            if atom not in ["PAD", "MASK", "UNK"]:
                valid_atoms.append(atom)
                valid_indices.append(i)

        if len(valid_atoms) == 0:
            reconstructed_smiles.append("")
            continue

        # Extract valid portion of bond matrix
        valid_indices_t = torch.tensor(valid_indices)
        valid_bond_matrix = bond_matrix[valid_indices_t][:, valid_indices_t]

        # Get coordinates if provided
        coords = None
        if ligand_coords is not None:
            coords = ligand_coords[b, valid_indices_t]

        # Convert to SMILES
        try:
            smiles = graph_to_smiles(valid_atoms, valid_bond_matrix, coords)
            reconstructed_smiles.append(smiles)
        except Exception:
            # If reconstruction fails, return empty string
            reconstructed_smiles.append("")

    return reconstructed_smiles


def reconstruct_smiles_from_tokens(
    ligand_atom_tokens: Tensor,
    bond_matrix: Tensor,
    ligand_mask: Tensor | None = None,
    ligand_coords: Tensor | None = None,
) -> list[str]:
    """Reconstruct SMILES from discrete tokens (not logits).

    This is useful when you have the final generated tokens rather
    than probability distributions.

    Parameters
    ----------
    ligand_atom_tokens : Tensor
        Atom type token indices with shape [B, N_atoms].
    bond_matrix : Tensor
        Bond type matrix with shape [B, N_atoms, N_atoms].
    ligand_mask : Tensor, optional
        Valid atom mask with shape [B, N_atoms].
    ligand_coords : Tensor, optional
        3D coordinates with shape [B, N_atoms, 3] for stereochemistry.

    Returns
    -------
    list[str]
        Reconstructed SMILES strings for each sample in batch.
    """
    batch_size = ligand_atom_tokens.shape[0]
    reconstructed_smiles = []

    for b in range(batch_size):
        atom_indices = ligand_atom_tokens[b]
        bonds = bond_matrix[b]

        # Apply mask if provided
        if ligand_mask is not None:
            mask = ligand_mask[b].bool()
            n_valid = mask.sum().item()
            atom_indices = atom_indices[:n_valid]
            bonds = bonds[:n_valid, :n_valid]
        else:
            n_valid = atom_indices.shape[0]

        if n_valid == 0:
            reconstructed_smiles.append("")
            continue

        # Convert indices to atom types
        atom_types = indices_to_atom_types(atom_indices)

        # Filter out special tokens
        valid_atoms = []
        valid_indices = []
        for i, atom in enumerate(atom_types):
            if atom not in ["PAD", "MASK", "UNK"]:
                valid_atoms.append(atom)
                valid_indices.append(i)

        if len(valid_atoms) == 0:
            reconstructed_smiles.append("")
            continue

        # Extract valid portion of bond matrix
        valid_indices_t = torch.tensor(valid_indices, device=bonds.device)
        valid_bond_matrix = bonds[valid_indices_t][:, valid_indices_t]

        # Get coordinates if provided
        coords = None
        if ligand_coords is not None:
            coords = ligand_coords[b, valid_indices_t]

        # Convert to SMILES
        try:
            smiles = graph_to_smiles(valid_atoms, valid_bond_matrix, coords)
            reconstructed_smiles.append(smiles)
        except Exception:
            reconstructed_smiles.append("")

    return reconstructed_smiles
