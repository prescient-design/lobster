#!/usr/bin/env python3
"""
Script to evaluate reconstruction quality of LatentGenerator models.
Takes structures from a directory, encodes and reconstructs them with multiple models,
and reports average aligned RMSD between original and reconstructed structures.
"""

import os
import glob
import torch
import numpy as np
import argparse
from loguru import logger
from tqdm import tqdm
import json

# Import from latent_generator
from lobster.model.latent_generator.cmdline import load_model, encode, decode, methods
from lobster.model.latent_generator.cmdline import minimize_ligand_structure, get_ligand_energy
from lobster.model.latent_generator.io import load_pdb, load_ligand, writepdb_ligand_complex, writepdb
from lobster.model.latent_generator.utils._utils import kabsch_torch_batched
from lobster.transforms._structure_transforms import StructureLigandTransform

ligand_transform = StructureLigandTransform(max_length=512)


def check_file_not_empty(file_path: str) -> bool:
    """Check if a file exists and is not empty."""
    if not os.path.exists(file_path):
        return False

    # Check file size
    if os.path.getsize(file_path) == 0:
        return False

    # For PDB files, check if they contain actual structure data
    if file_path.endswith(".pdb"):
        with open(file_path) as f:
            lines = f.readlines()
            # Check if file contains ATOM or HETATM records
            has_atoms = any(line.startswith(("ATOM", "HETATM")) for line in lines)
            return has_atoms

    # For SDF files, check if they contain structure data
    elif file_path.endswith(".sdf"):
        with open(file_path) as f:
            content = f.read()
            # Check if file contains atom count and coordinates
            return "$$$$" in content and any(line.strip().isdigit() for line in content.split("\n")[:10])

    # For PT files, check if they can be loaded
    elif file_path.endswith(".pt"):
        # Try to load the file to check if it's valid
        data = torch.load(file_path, map_location="cpu")
        return data is not None

    return True


def load_structure_data(file_path: str) -> dict:
    """Load structure data from file (PDB, SDF, or processed PT files)."""
    if file_path.endswith(".pt"):
        # Load already processed data
        structure_data = torch.load(file_path, map_location="cpu")
        if isinstance(structure_data, dict):
            return structure_data
        else:
            # If it's a tensor, wrap it in a dict
            return {"protein_coords": structure_data}

    elif file_path.endswith(".pdb"):
        return load_pdb(file_path)
    elif file_path.endswith(".sdf"):
        return load_ligand(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")


def find_paired_protein_ligand_files(data_dir: str) -> list[tuple[str, str, str]]:
    """Find paired protein and ligand files in a directory.

    Looks for files matching patterns like:
    - {name}_protein.pt and {name}_ligand.pt
    - {name}.protein.pt and {name}.ligand.pt

    Args:
        data_dir: Directory containing protein and ligand files

    Returns:
        List of tuples: (base_name, protein_path, ligand_path)
    """
    paired_files = []

    # Pattern 1: {name}_protein.pt and {name}_ligand.pt
    protein_files = glob.glob(os.path.join(data_dir, "*_protein.pt"))
    for protein_path in protein_files:
        base_name = os.path.basename(protein_path).replace("_protein.pt", "")
        ligand_path = protein_path.replace("_protein.pt", "_ligand.pt")
        if os.path.exists(ligand_path):
            paired_files.append((base_name, protein_path, ligand_path))

    # Pattern 2: {name}.protein.pt and {name}.ligand.pt (if not found with pattern 1)
    if not paired_files:
        protein_files = glob.glob(os.path.join(data_dir, "*.protein.pt"))
        for protein_path in protein_files:
            base_name = os.path.basename(protein_path).replace(".protein.pt", "")
            ligand_path = protein_path.replace(".protein.pt", ".ligand.pt")
            if os.path.exists(ligand_path):
                paired_files.append((base_name, protein_path, ligand_path))

    return paired_files


def load_paired_protein_ligand_data(protein_path: str, ligand_path: str) -> dict:
    """Load and merge paired protein and ligand data into a single dict.

    The encoder expects specific key names:
    - coords_res: protein coordinates (B, L, n_atoms, 3)
    - mask: protein mask (B, L)
    - indices: residue indices (B, L)
    - sequence: amino acid sequence (B, L)
    - ligand_coords: ligand coordinates (B, num_atoms, 3)
    - ligand_mask: ligand mask (B, num_atoms)
    - ligand_residue_index: ligand atom indices (B, num_atoms)

    Args:
        protein_path: Path to protein .pt file
        ligand_path: Path to ligand .pt file

    Returns:
        Combined structure data dict with both protein and ligand information
    """
    # Load protein data
    protein_data = torch.load(protein_path, map_location="cpu")
    if not isinstance(protein_data, dict):
        protein_data = {"coords_res": protein_data}

    # Load ligand data
    ligand_data = torch.load(ligand_path, map_location="cpu")
    if not isinstance(ligand_data, dict):
        ligand_data = {"ligand_coords": ligand_data}

    # Start with combined dict
    combined = {}

    # Map protein coordinates - encoder expects 'coords_res'
    # Expected shape: (B, L, n_atoms, 3)
    if "coords_res" in protein_data:
        coords = protein_data["coords_res"]
    elif "protein_coords" in protein_data:
        coords = protein_data["protein_coords"]
    elif "coords" in protein_data:
        coords = protein_data["coords"]
    else:
        coords = None

    if coords is not None:
        # Add batch dimension if missing (shape is L, n_atoms, 3)
        if coords.dim() == 3:
            coords = coords.unsqueeze(0)  # (1, L, n_atoms, 3)
        combined["coords_res"] = coords  # Key expected by encoder
        combined["protein_coords"] = coords  # Also keep for RMSD computation

    # Map protein mask - encoder expects 'mask'
    if "mask" in protein_data:
        mask = protein_data["mask"]
    elif "protein_mask" in protein_data:
        mask = protein_data["protein_mask"]
    else:
        mask = None

    if mask is not None:
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)  # (1, L)
        combined["mask"] = mask  # Key expected by encoder
        combined["protein_mask"] = mask  # Also keep for compatibility

    # Map sequence - encoder expects 'sequence'
    if "sequence" in protein_data:
        seq = protein_data["sequence"]
        if seq.dim() == 1:
            seq = seq.unsqueeze(0)  # (1, L)
        combined["sequence"] = seq  # Key expected by encoder
    elif "seq" in protein_data:
        seq = protein_data["seq"]
        if seq.dim() == 1:
            seq = seq.unsqueeze(0)
        combined["sequence"] = seq  # Key expected by encoder

    # Map indices - encoder expects 'indices' with batch dim
    if "indices" in protein_data:
        indices = protein_data["indices"]
        if indices.dim() == 1:
            indices = indices.unsqueeze(0)  # (1, L)
        combined["indices"] = indices

    # Copy other protein fields
    for key in ["sequence_str", "chains_ids", "real_chains", "pdb_path"]:
        if key in protein_data:
            combined[key] = protein_data[key]

    # Add ligand data with proper key mapping
    # Handle different possible key names in ligand files
    # Expected ligand_coords shape: (B, num_atoms, 3)
    if "ligand_coords" in ligand_data:
        lig_coords = ligand_data["ligand_coords"]
    elif "atom_coords" in ligand_data:
        lig_coords = ligand_data["atom_coords"]
    elif "coords" in ligand_data:
        lig_coords = ligand_data["coords"]
    else:
        lig_coords = None

    if lig_coords is not None:
        # Add batch dimension if missing (shape is num_atoms, 3)
        if lig_coords.dim() == 2:
            lig_coords = lig_coords.unsqueeze(0)  # (1, num_atoms, 3)
        combined["ligand_coords"] = lig_coords

    # Map ligand mask and add batch dim if needed
    if "ligand_mask" in ligand_data:
        lig_mask = ligand_data["ligand_mask"]
    elif "mask" in ligand_data:
        lig_mask = ligand_data["mask"]
    else:
        lig_mask = None

    if lig_mask is not None:
        if lig_mask.dim() == 1:
            lig_mask = lig_mask.unsqueeze(0)  # (1, num_atoms)
        combined["ligand_mask"] = lig_mask

    # Map ligand indices and add batch dim if needed
    if "ligand_residue_index" in ligand_data:
        lig_idx = ligand_data["ligand_residue_index"]
    elif "atom_indices" in ligand_data:
        lig_idx = ligand_data["atom_indices"]
    elif "residue_index" in ligand_data:
        lig_idx = ligand_data["residue_index"]
    else:
        lig_idx = None

    if lig_idx is not None:
        if lig_idx.dim() == 1:
            lig_idx = lig_idx.unsqueeze(0)  # (1, num_atoms)
        combined["ligand_residue_index"] = lig_idx
        combined["ligand_indices"] = lig_idx

    # Ligand atom names (list, no batch dim needed)
    if "ligand_atom_names" in ligand_data:
        combined["ligand_atom_names"] = ligand_data["ligand_atom_names"]
    elif "atom_names" in ligand_data:
        combined["ligand_atom_names"] = ligand_data["atom_names"]

    # Bond matrix for connectivity (important for geometry idealization)
    if "bond_matrix" in ligand_data:
        combined["bond_matrix"] = ligand_data["bond_matrix"]
    elif "ligand_bond_matrix" in ligand_data:
        combined["bond_matrix"] = ligand_data["ligand_bond_matrix"]

    return combined


def compute_complex_rmsd(
    original_protein_coords: torch.Tensor,
    reconstructed_protein_coords: torch.Tensor,
    original_ligand_coords: torch.Tensor,
    reconstructed_ligand_coords: torch.Tensor,
    protein_mask: torch.Tensor = None,
    ligand_mask: torch.Tensor = None,
) -> tuple[float, float, float]:
    """Compute aligned RMSD for protein-ligand complex (aligned together).

    This aligns the entire complex (protein + ligand) together, preserving
    relative positioning between protein and ligand.

    Args:
        original_protein_coords: Original protein coordinates (B, L, n_atoms, 3)
        reconstructed_protein_coords: Reconstructed protein coordinates (B, L, n_atoms, 3)
        original_ligand_coords: Original ligand coordinates (B, num_atoms, 3)
        reconstructed_ligand_coords: Reconstructed ligand coordinates (B, num_atoms, 3)
        protein_mask: Protein mask (B, L)
        ligand_mask: Ligand mask (B, num_atoms)

    Returns:
        Tuple of (complex_rmsd, protein_rmsd_after_complex_align, ligand_rmsd_after_complex_align)
    """
    device = reconstructed_protein_coords.device
    original_protein_coords = original_protein_coords.to(device)
    original_ligand_coords = original_ligand_coords.to(device)

    B, L, n_atoms, _ = original_protein_coords.shape
    B_lig, num_lig_atoms, _ = original_ligand_coords.shape

    # Create masks if not provided
    if protein_mask is None:
        protein_mask = torch.ones(B, L, device=device)
    else:
        protein_mask = protein_mask.to(device)

    if ligand_mask is None:
        ligand_mask = torch.ones(B_lig, num_lig_atoms, device=device)
    else:
        ligand_mask = ligand_mask.to(device)

    # Flatten protein coordinates (B, L*n_atoms, 3)
    gt_protein_flat = original_protein_coords.reshape(B, -1, 3)
    pred_protein_flat = reconstructed_protein_coords.reshape(B, -1, 3)
    protein_mask_flat = protein_mask.unsqueeze(-1).repeat(1, 1, n_atoms).reshape(B, -1)

    # Concatenate protein + ligand for joint alignment
    gt_complex = torch.cat([gt_protein_flat, original_ligand_coords], dim=1)
    pred_complex = torch.cat([pred_protein_flat, reconstructed_ligand_coords], dim=1)
    mask_complex = torch.cat([protein_mask_flat, ligand_mask], dim=1)

    # Align the ENTIRE complex together using Kabsch
    aligned_complex = kabsch_torch_batched(pred_complex, gt_complex, mask_complex)

    # Split back into protein and ligand
    n_protein_atoms = gt_protein_flat.shape[1]
    aligned_protein_flat = aligned_complex[:, :n_protein_atoms, :]
    aligned_ligand = aligned_complex[:, n_protein_atoms:, :]

    # Compute complex RMSD (all atoms)
    squared_diff_complex = torch.sum((gt_complex - aligned_complex) ** 2, dim=-1)
    masked_squared_diff = squared_diff_complex * mask_complex
    total_atoms = torch.sum(mask_complex)
    if total_atoms > 0:
        complex_rmsd = torch.sqrt(torch.sum(masked_squared_diff) / total_atoms).item()
    else:
        complex_rmsd = float("inf")

    # Compute protein RMSD after complex alignment (CA atoms only)
    aligned_protein = aligned_protein_flat.reshape(B, L, n_atoms, 3)
    gt_protein_ca = original_protein_coords[:, :, 1, :]  # CA atoms
    aligned_protein_ca = aligned_protein[:, :, 1, :]
    squared_diff_protein = torch.sum((gt_protein_ca - aligned_protein_ca) ** 2, dim=-1)
    masked_squared_diff_protein = squared_diff_protein * protein_mask
    total_protein_atoms = torch.sum(protein_mask)
    if total_protein_atoms > 0:
        protein_rmsd = torch.sqrt(torch.sum(masked_squared_diff_protein) / total_protein_atoms).item()
    else:
        protein_rmsd = float("inf")

    # Compute ligand RMSD after complex alignment
    squared_diff_ligand = torch.sum((original_ligand_coords - aligned_ligand) ** 2, dim=-1)
    masked_squared_diff_ligand = squared_diff_ligand * ligand_mask
    total_ligand_atoms = torch.sum(ligand_mask)
    if total_ligand_atoms > 0:
        ligand_rmsd = torch.sqrt(torch.sum(masked_squared_diff_ligand) / total_ligand_atoms).item()
    else:
        ligand_rmsd = float("inf")

    return complex_rmsd, protein_rmsd, ligand_rmsd


def compute_aligned_rmsd(
    original_coords: torch.Tensor,
    reconstructed_coords: torch.Tensor,
    mask: torch.Tensor = None,
    structure_type: str = "protein",
) -> float:
    """Compute aligned RMSD between original and reconstructed coordinates.

    Args:
        original_coords: Original coordinates tensor
        reconstructed_coords: Reconstructed coordinates tensor
        mask: Mask indicating valid positions
        structure_type: "protein" or "ligand" - determines alignment strategy
    """
    if mask is None:
        mask = torch.ones(original_coords.shape[:2], device=original_coords.device)

    B, L = original_coords.shape[:2]
    if structure_type == "protein":
        mask_expanded = mask.unsqueeze(-1).repeat(1, 1, 3).to(reconstructed_coords.device)
    else:
        mask_expanded = mask.to(reconstructed_coords.device)
    original_coords = original_coords.to(reconstructed_coords.device)

    # Align structures using Kabsch algorithm
    aligned_coords = kabsch_torch_batched(
        reconstructed_coords.reshape(B, -1, 3), original_coords.reshape(B, -1, 3), mask_expanded.reshape(B, -1)
    )

    if structure_type == "protein":
        aligned_coords = aligned_coords.reshape(B, L, 3, 3)
        # For proteins, use CA atoms (index 1) for RMSD calculation
        atoms_for_rmsd_original = original_coords[:, :, 1, :]  # CA atoms
        atoms_for_rmsd_aligned = aligned_coords[:, :, 1, :]  # CA atoms
        mask_expanded = mask[:, :].to(original_coords.device)
    elif structure_type == "ligand":
        # For ligands, use all atoms (average across atom dimension)
        atoms_for_rmsd_original = original_coords
        atoms_for_rmsd_aligned = aligned_coords
    else:  # not tested yet
        # Default to all atoms
        atoms_for_rmsd_original = original_coords.reshape(B, L * original_coords.shape[2], 3)
        atoms_for_rmsd_aligned = aligned_coords.reshape(B, L * aligned_coords.shape[2], 3)
        # Adjust mask for flattened coordinates
        mask_expanded = mask.unsqueeze(-1).repeat(1, 1, original_coords.shape[2]).reshape(B, -1)

    # Compute RMSD
    squared_diff = torch.sum((atoms_for_rmsd_original - atoms_for_rmsd_aligned) ** 2, dim=-1)
    masked_squared_diff = squared_diff * mask_expanded
    total_atoms = torch.sum(mask_expanded)

    if total_atoms > 0:
        rmsd = torch.sqrt(torch.sum(masked_squared_diff) / total_atoms)
        return rmsd.item(), original_coords, aligned_coords
    else:
        return float("inf"), None, None


def evaluate_model_on_structure(
    model_name: str,
    structure_data: dict,
    structure_path: str,
    save_structures: bool = False,
    output_dir: str = None,
    use_canonical_pose: bool = False,
    num_steps: int = None,
    minimize_ligand: bool = False,
    minimize_steps: int = 500,
    force_field: str = "MMFF94",
    minimize_mode: str = "bonds_and_angles",
) -> dict:
    """Evaluate a single model on a single structure.

    Parameters
    ----------
    minimize_ligand : bool, default=False
        If True, minimize ligand structure after decoding using Open Babel.
    minimize_steps : int, default=500
        Maximum number of minimization steps.
    force_field : str, default="MMFF94"
        Force field for minimization.
    minimize_mode : str, default="bonds_and_angles"
        Minimization mode: "bonds_only" (ideal bond lengths) or "bonds_and_angles" (recommended).
        Force field for minimization: "MMFF94", "MMFF94s", "UFF", "GAFF", "Ghemical".
    """
    results = {
        "model_name": model_name,
        "structure_path": structure_path,
        "num_steps": num_steps,
        "rmsd": float("inf"),
        "success": False,
        "error": None,
        "minimize_ligand": minimize_ligand,
    }

    # Check protein length and skip if > 512
    protein_length = None
    if "protein_coords" in structure_data:
        protein_length = structure_data["protein_coords"].shape[1]
    elif "coords_res" in structure_data:
        protein_length = structure_data["coords_res"].shape[1]

    if protein_length is not None and protein_length > 512:
        results["error"] = f"Protein length ({protein_length}) exceeds maximum allowed length (512)"
        logger.warning(f"Skipping {structure_path} - protein length {protein_length} > 512")
        return results

    # Handle different model types
    if "Ligand" in model_name:
        structure_data = ligand_transform(structure_data)
        if "conformers" in structure_data:
            structure_data = structure_data["conformers"][np.random.randint(0, len(structure_data["conformers"]))]
        if "atom_coords" in structure_data:
            structure_data["ligand_coords"] = structure_data["atom_coords"][None]
            structure_data["ligand_mask"] = structure_data["mask"][None]
            structure_data["ligand_residue_index"] = structure_data["atom_indices"][None]
            structure_data["ligand_atom_names"] = structure_data["atom_names"]
            structure_data["ligand_indices"] = structure_data["atom_indices"][None]

    # Encode and decode
    if use_canonical_pose:
        frame_type = "mol_frame"
    else:
        frame_type = None

    latents, embeddings = encode(structure_data, return_embeddings=True, frame_type=frame_type)

    # Pass num_steps to decode if provided
    if num_steps is not None:
        decoded_outputs, sequence_outputs = decode(latents, x_emb=embeddings, num_steps=num_steps)
    else:
        decoded_outputs, sequence_outputs = decode(latents, x_emb=embeddings)

    # Extract coordinates from decoded output and determine what was reconstructed
    reconstructed_protein_coords = None
    reconstructed_ligand_coords = None
    refined_reconstructed_coords = None

    if isinstance(decoded_outputs, dict):
        if "protein_coords" in decoded_outputs and decoded_outputs["protein_coords"] is not None:
            reconstructed_protein_coords = decoded_outputs["protein_coords"]
            if "protein_coords_refinement" in decoded_outputs:
                refined_reconstructed_coords = decoded_outputs["protein_coords_refinement"]
        if "ligand_coords" in decoded_outputs and decoded_outputs["ligand_coords"] is not None:
            reconstructed_ligand_coords = decoded_outputs["ligand_coords"]
    else:
        reconstructed_protein_coords = decoded_outputs

    # Apply ligand minimization if requested
    reconstructed_ligand_coords_minimized = None
    if minimize_ligand and reconstructed_ligand_coords is not None:
        try:
            # Get atom types from structure data
            ligand_atom_types = structure_data.get("ligand_atom_names", None)
            if ligand_atom_types is None:
                ligand_atom_types = structure_data.get("atom_names", None)
            if ligand_atom_types is None:
                # Default to carbon for unknown atoms
                num_atoms = reconstructed_ligand_coords.shape[1]
                ligand_atom_types = ["C"] * num_atoms

            # Get bond matrix if available
            bond_matrix = None
            if isinstance(decoded_outputs, dict):
                bond_matrix = decoded_outputs.get("ligand_bond_matrix", None)
            if bond_matrix is None:
                bond_matrix = structure_data.get("ligand_bond_matrix", None)
            if bond_matrix is None:
                bond_matrix = structure_data.get("bond_matrix", None)

            # Calculate energy before minimization
            energy_before = get_ligand_energy(
                reconstructed_ligand_coords[0], ligand_atom_types, bond_matrix, force_field
            )
            results["ligand_energy_before"] = energy_before

            # Minimize ligand structure
            minimized_coords = minimize_ligand_structure(
                reconstructed_ligand_coords[0],
                ligand_atom_types,
                bond_matrix=bond_matrix,
                steps=minimize_steps,
                force_field=force_field,
                method="cg",
                mode=minimize_mode,
            )
            reconstructed_ligand_coords_minimized = minimized_coords.unsqueeze(0)

            # Calculate energy after minimization
            energy_after = get_ligand_energy(minimized_coords, ligand_atom_types, bond_matrix, force_field)
            results["ligand_energy_after"] = energy_after
            results["ligand_energy_reduction"] = energy_before - energy_after

        except Exception as e:
            logger.warning(f"Ligand minimization failed for {structure_path}: {e}")
            results["minimize_error"] = str(e)

    # Compute RMSD for both protein and ligand when both are reconstructed
    protein_rmsd = None
    ligand_rmsd = None
    gt_coords = None
    aligned_pred_coords = None

    # Compute protein RMSD if protein was reconstructed
    if reconstructed_protein_coords is not None:
        # Get original protein coordinates
        if "protein_coords" in structure_data and structure_data["protein_coords"] is not None:
            original_protein_coords = structure_data["protein_coords"]
        elif "coords_res" in structure_data:
            original_protein_coords = structure_data["coords_res"]
            # Add batch dim if needed
            if original_protein_coords.dim() == 3:
                original_protein_coords = original_protein_coords.unsqueeze(0)
        else:
            original_protein_coords = None

        if original_protein_coords is not None:
            # Get protein mask
            protein_mask = structure_data.get("mask")
            if protein_mask is None:
                protein_mask = structure_data.get("protein_mask")

            protein_rmsd, gt_coords, aligned_pred_coords = compute_aligned_rmsd(
                original_protein_coords, reconstructed_protein_coords, protein_mask, "protein"
            )

    # Compute ligand RMSD if ligand was reconstructed
    ligand_rmsd_minimized = None
    if reconstructed_ligand_coords is not None:
        # Get original ligand coordinates
        if "ligand_coords" in structure_data:
            original_ligand_coords = structure_data["ligand_coords"]
        elif "atom_coords" in structure_data:
            original_ligand_coords = structure_data["atom_coords"]
            # Add batch dim if needed
            if original_ligand_coords.dim() == 2:
                original_ligand_coords = original_ligand_coords.unsqueeze(0)
        else:
            original_ligand_coords = None

        if original_ligand_coords is not None:
            # Get ligand mask
            ligand_mask = structure_data.get("ligand_mask")

            ligand_rmsd, _, _ = compute_aligned_rmsd(
                original_ligand_coords, reconstructed_ligand_coords, ligand_mask, "ligand"
            )

            # Also compute RMSD for minimized ligand if available
            if reconstructed_ligand_coords_minimized is not None:
                ligand_rmsd_minimized, _, _ = compute_aligned_rmsd(
                    original_ligand_coords, reconstructed_ligand_coords_minimized, ligand_mask, "ligand"
                )

    # Compute complex RMSD when both protein and ligand are available
    complex_rmsd = None
    complex_protein_rmsd = None
    complex_ligand_rmsd = None
    complex_ligand_rmsd_minimized = None
    if (
        reconstructed_protein_coords is not None
        and reconstructed_ligand_coords is not None
        and original_protein_coords is not None
        and original_ligand_coords is not None
    ):
        complex_rmsd, complex_protein_rmsd, complex_ligand_rmsd = compute_complex_rmsd(
            original_protein_coords,
            reconstructed_protein_coords,
            original_ligand_coords,
            reconstructed_ligand_coords,
            protein_mask,
            ligand_mask,
        )

        # Also compute complex RMSD with minimized ligand if available
        if reconstructed_ligand_coords_minimized is not None:
            (
                complex_rmsd_minimized,
                complex_protein_rmsd_minimized,
                complex_ligand_rmsd_minimized,
            ) = compute_complex_rmsd(
                original_protein_coords,
                reconstructed_protein_coords,
                original_ligand_coords,
                reconstructed_ligand_coords_minimized,
                protein_mask,
                ligand_mask,
            )
            results["complex_rmsd_minimized"] = complex_rmsd_minimized
            results["complex_protein_rmsd_minimized"] = complex_protein_rmsd_minimized
            results["complex_ligand_rmsd_minimized"] = complex_ligand_rmsd_minimized

    # Handle refined reconstruction (protein only for now)
    if refined_reconstructed_coords is not None and original_protein_coords is not None:
        rmsd_refined, gt_coords_refined, aligned_pred_coords_refined = compute_aligned_rmsd(
            original_protein_coords, refined_reconstructed_coords, protein_mask, "protein"
        )
    else:
        rmsd_refined = None

    # Store results - use protein RMSD as primary if available, else ligand
    if protein_rmsd is not None and protein_rmsd != float("inf"):
        results["rmsd"] = protein_rmsd
        results["protein_rmsd"] = protein_rmsd
        results["success"] = True
    if ligand_rmsd is not None and ligand_rmsd != float("inf"):
        results["ligand_rmsd"] = ligand_rmsd
        if results["rmsd"] == float("inf"):  # Only set primary if protein wasn't available
            results["rmsd"] = ligand_rmsd
        results["success"] = True
    # Store minimized ligand RMSD if available
    if ligand_rmsd_minimized is not None and ligand_rmsd_minimized != float("inf"):
        results["ligand_rmsd_minimized"] = ligand_rmsd_minimized
        # Calculate improvement from minimization
        if ligand_rmsd is not None and ligand_rmsd != float("inf"):
            results["ligand_rmsd_improvement"] = ligand_rmsd - ligand_rmsd_minimized
    # Store complex RMSD results
    if complex_rmsd is not None and complex_rmsd != float("inf"):
        results["complex_rmsd"] = complex_rmsd
        results["complex_protein_rmsd"] = complex_protein_rmsd
        results["complex_ligand_rmsd"] = complex_ligand_rmsd
    if rmsd_refined is not None and rmsd_refined != float("inf"):
        results["rmsd_refined"] = rmsd_refined
        results["success_refined"] = True

    # Check if we got any valid RMSD
    if protein_rmsd is None and ligand_rmsd is None:
        results["error"] = "No reconstructed coordinates in model output"
        return results

    # Determine structure type for saving
    has_protein = reconstructed_protein_coords is not None and original_protein_coords is not None
    has_ligand = reconstructed_ligand_coords is not None and original_ligand_coords is not None

    # Save aligned structures if requested
    if save_structures and output_dir is not None:
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            # Generate base filename from structure path
            structure_name = os.path.splitext(os.path.basename(structure_path))[0]
            model_safe_name = model_name.replace(" ", "_").replace("/", "_")

            # Get sequence for protein
            seq = None
            if has_protein:
                seq = structure_data.get("sequence")
                if seq is None:
                    seq = structure_data.get("seq")
                if seq is None:
                    # Create default sequence (UNK) for each residue
                    seq = torch.full((original_protein_coords.shape[1],), 20, dtype=torch.long)

            # Save ground truth structure
            gt_filename = os.path.join(output_dir, f"{structure_name}_{model_safe_name}_gt.pdb")
            if has_protein and has_ligand:
                # Save protein-ligand complex together
                writepdb_ligand_complex(
                    gt_filename,
                    protein_atoms=original_protein_coords[0],  # Remove batch dim
                    protein_seq=seq[0] if seq.dim() > 1 else seq,
                    ligand_atoms=original_ligand_coords[0],  # Remove batch dim
                    ligand_atom_names=structure_data.get("ligand_atom_names", None),
                    ligand_resname="LIG",
                    ligand_bond_matrix=structure_data.get("bond_matrix", None),
                )
            elif has_protein:
                writepdb(gt_filename, original_protein_coords, seq)
            elif has_ligand:
                writepdb_ligand_complex(
                    gt_filename,
                    ligand_atoms=original_ligand_coords[0],
                    ligand_atom_names=structure_data.get("ligand_atom_names", None),
                    ligand_resname="LIG",
                    ligand_bond_matrix=structure_data.get("bond_matrix", None),
                )

            # Save aligned prediction structure
            pred_filename = os.path.join(
                output_dir,
                f"{structure_name}_{model_safe_name}_{num_steps if num_steps is not None else 'default'}_pred.pdb",
            )
            if has_protein and has_ligand:
                # Save protein-ligand complex together
                writepdb_ligand_complex(
                    pred_filename,
                    protein_atoms=reconstructed_protein_coords[0],  # Remove batch dim
                    protein_seq=seq[0] if seq.dim() > 1 else seq,
                    ligand_atoms=reconstructed_ligand_coords[0],  # Remove batch dim
                    ligand_atom_names=structure_data.get("ligand_atom_names", None),
                    ligand_resname="LIG",
                    ligand_bond_matrix=structure_data.get("bond_matrix", None),
                )
            elif has_protein:
                writepdb(pred_filename, reconstructed_protein_coords, seq)
            elif has_ligand:
                writepdb_ligand_complex(
                    pred_filename,
                    ligand_atoms=reconstructed_ligand_coords[0],
                    ligand_atom_names=structure_data.get("ligand_atom_names", None),
                    ligand_resname="LIG",
                    ligand_bond_matrix=structure_data.get("bond_matrix", None),
                )

            results["saved_structures"] = {"ground_truth": gt_filename, "prediction": pred_filename}

            # Save minimized structure if available
            if reconstructed_ligand_coords_minimized is not None:
                minimized_filename = os.path.join(
                    output_dir,
                    f"{structure_name}_{model_safe_name}_{num_steps if num_steps is not None else 'default'}_minimized.pdb",
                )
                if has_protein and has_ligand:
                    writepdb_ligand_complex(
                        minimized_filename,
                        protein_atoms=reconstructed_protein_coords[0],
                        protein_seq=seq[0] if seq.dim() > 1 else seq,
                        ligand_atoms=reconstructed_ligand_coords_minimized[0],
                        ligand_atom_names=structure_data.get("ligand_atom_names", None),
                        ligand_resname="LIG",
                        ligand_bond_matrix=structure_data.get("bond_matrix", None),
                    )
                elif has_ligand:
                    writepdb_ligand_complex(
                        minimized_filename,
                        ligand_atoms=reconstructed_ligand_coords_minimized[0],
                        ligand_atom_names=structure_data.get("ligand_atom_names", None),
                        ligand_resname="LIG",
                        ligand_bond_matrix=structure_data.get("bond_matrix", None),
                    )
                results["saved_structures"]["minimized"] = minimized_filename

        except Exception as e:
            logger.warning(f"Failed to save structures for {structure_path}: {e}")
            results["saved_structures"] = None

    return results


def evaluate_models_on_directory(
    models: list[str],
    data_dir: str,
    output_file: str = "reconstruction_evaluation.json",
    save_structures: bool = False,
    structures_output_dir: str = None,
    use_canonical_pose: bool = False,
    num_steps_list: list[int] = None,
    max_save_structures: int = None,
    minimize_ligand: bool = False,
    minimize_steps: int = 500,
    force_field: str = "MMFF94",
    minimize_mode: str = "bonds_and_angles",
) -> dict:
    """Evaluate multiple models on all structures in a directory.

    Args:
        max_save_structures: Maximum number of structures to save. If None, saves all.
        minimize_ligand: If True, minimize ligand structure after decoding.
        minimize_steps: Maximum number of minimization steps.
        force_field: Force field for minimization.
    """

    if num_steps_list is None:
        num_steps_list = [None]  # Default to no num_steps specified

    # Track how many structures have been saved
    structures_saved_count = 0

    # Check if any model requires protein-ligand pairs
    requires_protein_ligand = any("Ligand" in model and "Protein" in model for model in models)

    # First, check for paired protein-ligand files
    paired_files = find_paired_protein_ligand_files(data_dir)
    use_paired_mode = len(paired_files) > 0 and requires_protein_ligand

    if use_paired_mode:
        logger.info(f"Found {len(paired_files)} paired protein-ligand complexes")
        # Validate paired files
        valid_paired_files = []
        for base_name, protein_path, ligand_path in tqdm(paired_files, desc="Checking paired files"):
            if check_file_not_empty(protein_path) and check_file_not_empty(ligand_path):
                valid_paired_files.append((base_name, protein_path, ligand_path))
            else:
                logger.warning(f"Skipping invalid pair: {base_name}")
        logger.info(f"Total: {len(valid_paired_files)} valid protein-ligand pairs")
        valid_files = valid_paired_files  # List of tuples for paired mode
    else:
        # Find all structure files (including processed PT files)
        pdb_files = glob.glob(os.path.join(data_dir, "*.pdb"))
        sdf_files = glob.glob(os.path.join(data_dir, "*.sdf"))
        pt_files = glob.glob(os.path.join(data_dir, "*.pt"))
        structure_files = pdb_files + sdf_files + pt_files

        # Filter out empty files
        valid_files = []
        for file_path in tqdm(structure_files, desc="Checking files"):
            if check_file_not_empty(file_path):
                valid_files.append(file_path)
            else:
                logger.warning(f"Skipping empty or invalid file: {file_path}")

        logger.info(f"Found {len(pdb_files)} PDB files, {len(sdf_files)} SDF files, and {len(pt_files)} PT files")
        logger.info(f"Total: {len(valid_files)} valid structure files")

    if len(valid_files) == 0:
        logger.error("No valid structure files found!")
        return {}

    # Results storage - now includes step information
    all_results = {"models": {}, "structures": {}, "summary": {}}

    # Initialize model results for each step count
    for model_name in models:
        all_results["models"][model_name] = {}
        for num_steps in num_steps_list:
            step_key = f"steps_{num_steps}" if num_steps is not None else "default"
            all_results["models"][model_name][step_key] = {
                "rmsd_values": [],
                "successful_reconstructions": 0,
                "failed_reconstructions": 0,
                "average_rmsd": float("inf"),
                "std_rmsd": float("inf"),
                # Protein-specific RMSD tracking
                "protein_rmsd_values": [],
                "average_protein_rmsd": float("inf"),
                "std_protein_rmsd": float("inf"),
                # Ligand-specific RMSD tracking
                "ligand_rmsd_values": [],
                "average_ligand_rmsd": float("inf"),
                "std_ligand_rmsd": float("inf"),
                # Minimized ligand RMSD tracking
                "ligand_rmsd_minimized_values": [],
                "average_ligand_rmsd_minimized": float("inf"),
                "std_ligand_rmsd_minimized": float("inf"),
                # Complex RMSD tracking (protein+ligand aligned together)
                "complex_rmsd_values": [],
                "average_complex_rmsd": float("inf"),
                "std_complex_rmsd": float("inf"),
                "complex_protein_rmsd_values": [],
                "average_complex_protein_rmsd": float("inf"),
                "complex_ligand_rmsd_values": [],
                "average_complex_ligand_rmsd": float("inf"),
                # Minimized complex RMSD tracking
                "complex_rmsd_minimized_values": [],
                "average_complex_rmsd_minimized": float("inf"),
                "std_complex_rmsd_minimized": float("inf"),
                "complex_protein_rmsd_minimized_values": [],
                "average_complex_protein_rmsd_minimized": float("inf"),
                "complex_ligand_rmsd_minimized_values": [],
                "average_complex_ligand_rmsd_minimized": float("inf"),
                # Refined reconstruction tracking
                "rmsd_values_refined": [],
                "successful_reconstructions_refined": 0,
                "failed_reconstructions_refined": 0,
                "average_rmsd_refined": float("inf"),
                "std_rmsd_refined": float("inf"),
            }

    # Initialize structures dictionary
    if use_paired_mode:
        for base_name, _, _ in valid_files:
            all_results["structures"][base_name] = {}
    else:
        for structure_path in valid_files:
            structure_name = os.path.basename(structure_path)
            all_results["structures"][structure_name] = {}

    # Process one model at a time
    for model_name in tqdm(models, desc="Processing models"):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Loading and evaluating model: {model_name}")
        logger.info(f"{'=' * 60}")

        # Load the current model
        logger.info(f"Loading model: {model_name}")
        mc = methods[model_name].model_config
        load_model(
            mc.checkpoint,
            mc.config_path,
            mc.config_name,
            overrides=mc.overrides,
            model_class=mc.model_class,
        )
        logger.info(f"Model {model_name} loaded successfully")

        # Evaluate this model on all structures with all step counts
        if use_paired_mode:
            # Paired protein-ligand mode
            for base_name, protein_path, ligand_path in tqdm(valid_files, desc=f"Evaluating {model_name}"):
                structure_name = base_name

                if structure_name not in all_results["structures"]:
                    all_results["structures"][structure_name] = {}
                if model_name not in all_results["structures"][structure_name]:
                    all_results["structures"][structure_name][model_name] = {}

                # Load paired protein-ligand data
                structure_data = load_paired_protein_ligand_data(protein_path, ligand_path)
                structure_path = protein_path  # Use protein path for reference

                # Determine if we should save this structure
                should_save = save_structures and (
                    max_save_structures is None or structures_saved_count < max_save_structures
                )

                # Test each num_steps value
                for num_steps in num_steps_list:
                    step_key = f"steps_{num_steps}" if num_steps is not None else "default"

                    # Evaluate model on this structure with this num_steps
                    result = evaluate_model_on_structure(
                        model_name,
                        structure_data,
                        structure_path,
                        should_save,
                        structures_output_dir,
                        use_canonical_pose,
                        num_steps,
                        minimize_ligand=minimize_ligand,
                        minimize_steps=minimize_steps,
                        force_field=force_field,
                        minimize_mode=minimize_mode,
                    )

                    # Increment saved counter if structure was saved
                    if should_save and result.get("saved_structures"):
                        structures_saved_count += 1

                    all_results["structures"][structure_name][model_name][step_key] = result

                    # Update model statistics
                    if result["success"]:
                        all_results["models"][model_name][step_key]["rmsd_values"].append(result["rmsd"])
                        all_results["models"][model_name][step_key]["successful_reconstructions"] += 1
                        # Track protein RMSD if available
                        if "protein_rmsd" in result:
                            all_results["models"][model_name][step_key]["protein_rmsd_values"].append(
                                result["protein_rmsd"]
                            )
                        # Track ligand RMSD if available
                        if "ligand_rmsd" in result:
                            all_results["models"][model_name][step_key]["ligand_rmsd_values"].append(
                                result["ligand_rmsd"]
                            )
                        # Track minimized ligand RMSD if available
                        if "ligand_rmsd_minimized" in result:
                            all_results["models"][model_name][step_key]["ligand_rmsd_minimized_values"].append(
                                result["ligand_rmsd_minimized"]
                            )
                        # Track complex RMSD if available
                        if "complex_rmsd" in result:
                            all_results["models"][model_name][step_key]["complex_rmsd_values"].append(
                                result["complex_rmsd"]
                            )
                            all_results["models"][model_name][step_key]["complex_protein_rmsd_values"].append(
                                result["complex_protein_rmsd"]
                            )
                            all_results["models"][model_name][step_key]["complex_ligand_rmsd_values"].append(
                                result["complex_ligand_rmsd"]
                            )
                        # Track minimized complex RMSD if available
                        if "complex_rmsd_minimized" in result:
                            all_results["models"][model_name][step_key]["complex_rmsd_minimized_values"].append(
                                result["complex_rmsd_minimized"]
                            )
                            all_results["models"][model_name][step_key]["complex_protein_rmsd_minimized_values"].append(
                                result["complex_protein_rmsd_minimized"]
                            )
                            all_results["models"][model_name][step_key]["complex_ligand_rmsd_minimized_values"].append(
                                result["complex_ligand_rmsd_minimized"]
                            )
                        if "success_refined" in result and result["success_refined"]:
                            all_results["models"][model_name][step_key]["rmsd_values_refined"].append(
                                result["rmsd_refined"]
                            )
                            all_results["models"][model_name][step_key]["successful_reconstructions_refined"] += 1
                    else:
                        all_results["models"][model_name][step_key]["failed_reconstructions"] += 1
                        all_results["models"][model_name][step_key]["failed_reconstructions_refined"] += 1
        else:
            # Single file mode (original behavior)
            for structure_path in tqdm(valid_files, desc=f"Evaluating {model_name}"):
                structure_name = os.path.basename(structure_path)

                if structure_name not in all_results["structures"]:
                    all_results["structures"][structure_name] = {}
                if model_name not in all_results["structures"][structure_name]:
                    all_results["structures"][structure_name][model_name] = {}

                # Load structure data once
                structure_data = load_structure_data(structure_path)

                # Determine if we should save this structure
                should_save = save_structures and (
                    max_save_structures is None or structures_saved_count < max_save_structures
                )

                # Test each num_steps value
                for num_steps in num_steps_list:
                    step_key = f"steps_{num_steps}" if num_steps is not None else "default"

                    # Evaluate model on this structure with this num_steps
                    result = evaluate_model_on_structure(
                        model_name,
                        structure_data,
                        structure_path,
                        should_save,
                        structures_output_dir,
                        use_canonical_pose,
                        num_steps,
                        minimize_ligand=minimize_ligand,
                        minimize_steps=minimize_steps,
                        force_field=force_field,
                        minimize_mode=minimize_mode,
                    )

                    # Increment saved counter if structure was saved
                    if should_save and result.get("saved_structures"):
                        structures_saved_count += 1

                    all_results["structures"][structure_name][model_name][step_key] = result

                    # Update model statistics
                    if result["success"]:
                        all_results["models"][model_name][step_key]["rmsd_values"].append(result["rmsd"])
                        all_results["models"][model_name][step_key]["successful_reconstructions"] += 1
                        # Track protein RMSD if available
                        if "protein_rmsd" in result:
                            all_results["models"][model_name][step_key]["protein_rmsd_values"].append(
                                result["protein_rmsd"]
                            )
                        # Track ligand RMSD if available
                        if "ligand_rmsd" in result:
                            all_results["models"][model_name][step_key]["ligand_rmsd_values"].append(
                                result["ligand_rmsd"]
                            )
                        # Track minimized ligand RMSD if available
                        if "ligand_rmsd_minimized" in result:
                            all_results["models"][model_name][step_key]["ligand_rmsd_minimized_values"].append(
                                result["ligand_rmsd_minimized"]
                            )
                        # Track complex RMSD if available
                        if "complex_rmsd" in result:
                            all_results["models"][model_name][step_key]["complex_rmsd_values"].append(
                                result["complex_rmsd"]
                            )
                            all_results["models"][model_name][step_key]["complex_protein_rmsd_values"].append(
                                result["complex_protein_rmsd"]
                            )
                            all_results["models"][model_name][step_key]["complex_ligand_rmsd_values"].append(
                                result["complex_ligand_rmsd"]
                            )
                        # Track minimized complex RMSD if available
                        if "complex_rmsd_minimized" in result:
                            all_results["models"][model_name][step_key]["complex_rmsd_minimized_values"].append(
                                result["complex_rmsd_minimized"]
                            )
                            all_results["models"][model_name][step_key]["complex_protein_rmsd_minimized_values"].append(
                                result["complex_protein_rmsd_minimized"]
                            )
                            all_results["models"][model_name][step_key]["complex_ligand_rmsd_minimized_values"].append(
                                result["complex_ligand_rmsd_minimized"]
                            )
                        if "success_refined" in result and result["success_refined"]:
                            all_results["models"][model_name][step_key]["rmsd_values_refined"].append(
                                result["rmsd_refined"]
                            )
                            all_results["models"][model_name][step_key]["successful_reconstructions_refined"] += 1
                    else:
                        all_results["models"][model_name][step_key]["failed_reconstructions"] += 1
                        all_results["models"][model_name][step_key]["failed_reconstructions_refined"] += 1

        # Clear model from memory after evaluation
        logger.info(f"Clearing model {model_name} from memory")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Compute summary statistics
    for model_name in models:
        for num_steps in num_steps_list:
            step_key = f"steps_{num_steps}" if num_steps is not None else "default"
            model_results = all_results["models"][model_name][step_key]
            if model_results["rmsd_values"]:
                model_results["average_rmsd"] = np.mean(model_results["rmsd_values"])
                model_results["std_rmsd"] = np.std(model_results["rmsd_values"])
                model_results["min_rmsd"] = np.min(model_results["rmsd_values"])
                model_results["max_rmsd"] = np.max(model_results["rmsd_values"])
                # Compute protein RMSD statistics
                if model_results["protein_rmsd_values"]:
                    model_results["average_protein_rmsd"] = np.mean(model_results["protein_rmsd_values"])
                    model_results["std_protein_rmsd"] = np.std(model_results["protein_rmsd_values"])
                    model_results["min_protein_rmsd"] = np.min(model_results["protein_rmsd_values"])
                    model_results["max_protein_rmsd"] = np.max(model_results["protein_rmsd_values"])
                # Compute ligand RMSD statistics
                if model_results["ligand_rmsd_values"]:
                    model_results["average_ligand_rmsd"] = np.mean(model_results["ligand_rmsd_values"])
                    model_results["std_ligand_rmsd"] = np.std(model_results["ligand_rmsd_values"])
                    model_results["min_ligand_rmsd"] = np.min(model_results["ligand_rmsd_values"])
                    model_results["max_ligand_rmsd"] = np.max(model_results["ligand_rmsd_values"])
                # Compute minimized ligand RMSD statistics
                if model_results["ligand_rmsd_minimized_values"]:
                    model_results["average_ligand_rmsd_minimized"] = np.mean(
                        model_results["ligand_rmsd_minimized_values"]
                    )
                    model_results["std_ligand_rmsd_minimized"] = np.std(model_results["ligand_rmsd_minimized_values"])
                    model_results["min_ligand_rmsd_minimized"] = np.min(model_results["ligand_rmsd_minimized_values"])
                    model_results["max_ligand_rmsd_minimized"] = np.max(model_results["ligand_rmsd_minimized_values"])
                    # Compute improvement from minimization
                    if model_results["ligand_rmsd_values"]:
                        model_results["average_ligand_rmsd_improvement"] = (
                            model_results["average_ligand_rmsd"] - model_results["average_ligand_rmsd_minimized"]
                        )
                # Compute complex RMSD statistics
                if model_results["complex_rmsd_values"]:
                    model_results["average_complex_rmsd"] = np.mean(model_results["complex_rmsd_values"])
                    model_results["std_complex_rmsd"] = np.std(model_results["complex_rmsd_values"])
                    model_results["min_complex_rmsd"] = np.min(model_results["complex_rmsd_values"])
                    model_results["max_complex_rmsd"] = np.max(model_results["complex_rmsd_values"])
                    model_results["average_complex_protein_rmsd"] = np.mean(
                        model_results["complex_protein_rmsd_values"]
                    )
                    model_results["average_complex_ligand_rmsd"] = np.mean(model_results["complex_ligand_rmsd_values"])
                # Compute minimized complex RMSD statistics
                if model_results.get("complex_rmsd_minimized_values"):
                    model_results["average_complex_rmsd_minimized"] = np.mean(
                        model_results["complex_rmsd_minimized_values"]
                    )
                    model_results["std_complex_rmsd_minimized"] = np.std(model_results["complex_rmsd_minimized_values"])
                    model_results["min_complex_rmsd_minimized"] = np.min(model_results["complex_rmsd_minimized_values"])
                    model_results["max_complex_rmsd_minimized"] = np.max(model_results["complex_rmsd_minimized_values"])
                    model_results["average_complex_protein_rmsd_minimized"] = np.mean(
                        model_results["complex_protein_rmsd_minimized_values"]
                    )
                    model_results["average_complex_ligand_rmsd_minimized"] = np.mean(
                        model_results["complex_ligand_rmsd_minimized_values"]
                    )
                if model_results["rmsd_values_refined"]:
                    model_results["average_rmsd_refined"] = np.mean(model_results["rmsd_values_refined"])
                    model_results["std_rmsd_refined"] = np.std(model_results["rmsd_values_refined"])
                    model_results["min_rmsd_refined"] = np.min(model_results["rmsd_values_refined"])
                    model_results["max_rmsd_refined"] = np.max(model_results["rmsd_values_refined"])
            else:
                model_results["average_rmsd"] = float("inf")
                model_results["std_rmsd"] = float("inf")
                model_results["min_rmsd"] = float("inf")
                model_results["max_rmsd"] = float("inf")
                # filler values
                model_results["average_rmsd_refined"] = float("inf")
                model_results["std_rmsd_refined"] = float("inf")
                model_results["min_rmsd_refined"] = float("inf")
                model_results["max_rmsd_refined"] = float("inf")

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("RECONSTRUCTION EVALUATION SUMMARY")
    logger.info("=" * 80)

    for model_name in models:
        for num_steps in num_steps_list:
            step_key = f"steps_{num_steps}" if num_steps is not None else "default"
            model_results = all_results["models"][model_name][step_key]
            logger.info(f"\nModel: {model_name} (Steps: {num_steps if num_steps is not None else 'Default'})")
            logger.info(f"  Successful reconstructions: {model_results['successful_reconstructions']}")
            logger.info(f"  Failed reconstructions: {model_results['failed_reconstructions']}")
            if model_results["average_rmsd"] != float("inf"):
                logger.info(f"  Average RMSD: {model_results['average_rmsd']:.3f} ± {model_results['std_rmsd']:.3f} Å")
                logger.info(f"  Min RMSD: {model_results['min_rmsd']:.3f} Å")
                logger.info(f"  Max RMSD: {model_results['max_rmsd']:.3f} Å")
            else:
                logger.info("  Average RMSD: N/A (no successful reconstructions)")
            # Print protein-specific RMSD if available
            if model_results.get("protein_rmsd_values"):
                logger.info(
                    f"  Average Protein RMSD: {model_results['average_protein_rmsd']:.3f} ± {model_results['std_protein_rmsd']:.3f} Å"
                )
                logger.info(f"  Min Protein RMSD: {model_results['min_protein_rmsd']:.3f} Å")
                logger.info(f"  Max Protein RMSD: {model_results['max_protein_rmsd']:.3f} Å")
            # Print ligand-specific RMSD if available
            if model_results.get("ligand_rmsd_values"):
                logger.info(
                    f"  Average Ligand RMSD: {model_results['average_ligand_rmsd']:.3f} ± {model_results['std_ligand_rmsd']:.3f} Å"
                )
                logger.info(f"  Min Ligand RMSD: {model_results['min_ligand_rmsd']:.3f} Å")
                logger.info(f"  Max Ligand RMSD: {model_results['max_ligand_rmsd']:.3f} Å")
            # Print minimized ligand RMSD if available
            if model_results.get("ligand_rmsd_minimized_values"):
                logger.info(
                    f"  Average Ligand RMSD (minimized): {model_results['average_ligand_rmsd_minimized']:.3f} ± {model_results['std_ligand_rmsd_minimized']:.3f} Å"
                )
                logger.info(f"  Min Ligand RMSD (minimized): {model_results['min_ligand_rmsd_minimized']:.3f} Å")
                logger.info(f"  Max Ligand RMSD (minimized): {model_results['max_ligand_rmsd_minimized']:.3f} Å")
                if "average_ligand_rmsd_improvement" in model_results:
                    improvement = model_results["average_ligand_rmsd_improvement"]
                    logger.info(f"  Ligand RMSD improvement from minimization: {improvement:+.3f} Å")
            # Print complex RMSD if available (protein+ligand aligned together)
            if model_results.get("complex_rmsd_values"):
                logger.info(
                    f"  Average Complex RMSD: {model_results['average_complex_rmsd']:.3f} ± {model_results['std_complex_rmsd']:.3f} Å"
                )
                logger.info(f"  Min Complex RMSD: {model_results['min_complex_rmsd']:.3f} Å")
                logger.info(f"  Max Complex RMSD: {model_results['max_complex_rmsd']:.3f} Å")
                logger.info(
                    f"    (Complex alignment) Protein RMSD: {model_results['average_complex_protein_rmsd']:.3f} Å"
                )
                logger.info(
                    f"    (Complex alignment) Ligand RMSD: {model_results['average_complex_ligand_rmsd']:.3f} Å"
                )
            # Print minimized complex RMSD if available
            if model_results.get("complex_rmsd_minimized_values"):
                logger.info(
                    f"  Average Complex RMSD (minimized): {model_results['average_complex_rmsd_minimized']:.3f} ± {model_results['std_complex_rmsd_minimized']:.3f} Å"
                )
                logger.info(f"  Min Complex RMSD (minimized): {model_results['min_complex_rmsd_minimized']:.3f} Å")
                logger.info(f"  Max Complex RMSD (minimized): {model_results['max_complex_rmsd_minimized']:.3f} Å")
                logger.info(
                    f"    (Complex alignment minimized) Protein RMSD: {model_results['average_complex_protein_rmsd_minimized']:.3f} Å"
                )
                logger.info(
                    f"    (Complex alignment minimized) Ligand RMSD: {model_results['average_complex_ligand_rmsd_minimized']:.3f} Å"
                )
            if model_results["rmsd_values_refined"]:
                logger.info(
                    f"  Successful refined reconstructions: {model_results['successful_reconstructions_refined']}"
                )
                logger.info(f"  Failed refined reconstructions: {model_results['failed_reconstructions_refined']}")
                if model_results["average_rmsd_refined"] != float("inf"):
                    logger.info(
                        f"  Average refined RMSD: {model_results['average_rmsd_refined']:.3f} ± {model_results['std_rmsd_refined']:.3f} Å"
                    )
                    logger.info(f"  Min refined RMSD: {model_results['min_rmsd_refined']:.3f} Å")
                    logger.info(f"  Max refined RMSD: {model_results['max_rmsd_refined']:.3f} Å")
                else:
                    logger.info("  Average refined RMSD: N/A (no successful refined reconstructions)")
    # Save results
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info(f"\nDetailed results saved to: {output_file}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate reconstruction quality of LatentGenerator models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available Models:
{chr(10).join([f"  - {name}" for name in methods.keys()])}

Example usage Protein-only model:
  python src/lobster/metrics/evaluate_reconstruction.py \
    --models "LG full attention 2" \
    --data_dir /cv/data/ai4dd/data2/lisanzas/latent_generator_files/casp_recon/CASP15_merged/ \
    --output_file reconstruction_results_protein_only.json \
    --save_structures \
    --structures_output_dir aligned_structures_protein_only

Example usage Ligand-only model:
  python src/lobster/metrics/evaluate_reconstruction.py \
    --models "LG Ligand 20A" \
    --data_dir /cv/data/ai4dd/data2/lisanzas/geom_12_15_25/test/ \
    --output_file reconstruction_results_ligand_only.json

Example usage Protein-Ligand model (with paired *_protein.pt and *_ligand.pt files):
  python src/lobster/metrics/evaluate_reconstruction.py \
    --models "LG Protein Ligand" \
    --data_dir /cv/data/ai4dd/data2/lisanzas/pdb_bind_12_15_25/test/ \
    --output_file reconstruction_results_protein_ligand.json \
    --save_structures \
    --structures_output_dir aligned_structures_protein_ligand
        """,
    )

    parser.add_argument("--models", nargs="+", required=True, help="List of model names to evaluate")

    parser.add_argument(
        "--data_dir",
        type=str,
        default="/cv/data/ai4dd/data2/lisanzas/latent_generator_files/casp_recon/CASP15_merged/",
        help="Directory containing structure files to evaluate",
    )

    parser.add_argument(
        "--output_file", type=str, default="reconstruction_evaluation.json", help="Output file to save results"
    )

    parser.add_argument(
        "--save_structures",
        action="store_true",
        help="Save aligned ground truth and prediction structures to PDB files",
    )

    parser.add_argument(
        "--structures_output_dir",
        type=str,
        default="aligned_structures",
        help="Directory to save aligned structures (used with --save_structures)",
    )

    parser.add_argument(
        "--max_save_structures",
        type=int,
        default=None,
        help="Maximum number of structures to save (default: save all). Only saves first N structures.",
    )

    parser.add_argument(
        "--use_canonical_pose", action="store_true", help="Make model invariant with mol_frame to get canonical pose"
    )

    parser.add_argument(
        "--num_steps", nargs="+", type=int, help="List of num_steps values to test for each model (e.g., 1 5 10 20)"
    )

    # Minimization options
    parser.add_argument(
        "--minimize_ligand",
        action="store_true",
        help="Minimize ligand structure after decoding using Open Babel force field",
    )
    parser.add_argument(
        "--minimize_steps",
        type=int,
        default=500,
        help="Maximum number of minimization steps (default: 500)",
    )
    parser.add_argument(
        "--force_field",
        type=str,
        default="MMFF94",
        choices=["MMFF94", "MMFF94s", "UFF", "GAFF", "Ghemical"],
        help="Force field for minimization (default: MMFF94)",
    )
    parser.add_argument(
        "--minimize_mode",
        type=str,
        default="bonds_and_angles",
        choices=["bonds_only", "bonds_and_angles"],
        help="Minimization mode: 'bonds_only' (ideal bond lengths), 'bonds_and_angles' (ideal bonds + angles via constrained minimization, recommended)",
    )

    args = parser.parse_args()

    # Validate models
    invalid_models = [model for model in args.models if model not in methods]
    if invalid_models:
        logger.error(f"Invalid model names: {invalid_models}")
        logger.error(f"Available models: {list(methods.keys())}")
        return

    # Validate data directory
    if not os.path.exists(args.data_dir):
        logger.error(f"Data directory does not exist: {args.data_dir}")
        return

    logger.info(f"Evaluating models: {args.models}")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output file: {args.output_file}")
    if args.save_structures:
        logger.info(f"Saving aligned structures to: {args.structures_output_dir}")
        if args.max_save_structures:
            logger.info(f"Limiting to first {args.max_save_structures} structures")
    if args.minimize_ligand:
        logger.info(
            f"Ligand minimization enabled: force_field={args.force_field}, steps={args.minimize_steps}, mode={args.minimize_mode}"
        )

    # Run evaluation
    results = evaluate_models_on_directory(
        args.models,
        args.data_dir,
        args.output_file,
        args.save_structures,
        args.structures_output_dir,
        args.use_canonical_pose,
        args.num_steps,
        args.max_save_structures,
        minimize_ligand=args.minimize_ligand,
        minimize_steps=args.minimize_steps,
        force_field=args.force_field,
        minimize_mode=args.minimize_mode,
    )
    if results:
        logger.info("Evaluation completed successfully!")
    else:
        logger.error("Evaluation failed!")


if __name__ == "__main__":
    main()
