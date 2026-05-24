"""Standalone SE3 augmentation utilities for protein-ligand complexes.

This module provides functions to apply SE(3) transformations (rotation + translation)
to protein-ligand complexes while maintaining their relative positions.
"""

import logging
from typing import Literal, NamedTuple

import torch
from torch import Tensor

from ._kinematics import (
    apply_global_frame_to_coords,
    apply_random_se3_batched,
)

logger = logging.getLogger(__name__)


class SE3AugmentedComplex(NamedTuple):
    """Output of SE3 augmentation for protein-ligand complexes."""

    protein_coords: Tensor | None  # [B, L, n_atoms, 3]
    protein_mask: Tensor | None  # [B, L]
    ligand_coords: Tensor | None  # [B, N_ligand, 3]
    ligand_mask: Tensor | None  # [B, N_ligand]


def apply_se3_augmentation_protein_ligand(
    protein_coords: Tensor | None = None,
    protein_mask: Tensor | None = None,
    ligand_coords: Tensor | None = None,
    ligand_mask: Tensor | None = None,
    random_se3: bool = True,
    only_rot: bool = False,
    only_trans: bool = False,
    translation_scale: float = 1.0,
    rotation_mode: str = "svd",
    frame_type: Literal["norm_frame", "pca_frame", "mol_frame"] | None = None,
    apply_stochastic_fa: bool = False,
    get_all_frames: bool = False,
    backbone_noise: float = 0.0,
) -> SE3AugmentedComplex:
    """Apply SE(3) augmentation to protein-ligand complex.

    This function applies the SAME SE(3) transformation (rotation + translation)
    to both protein and ligand coordinates, ensuring they remain in the same
    reference frame. The transformation is applied as:
    1. Concatenate protein and ligand coordinates
    2. Apply random SE(3) transformation
    3. Optionally apply global frame (PCA/norm/mol frame)
    4. Add optional backbone noise
    5. Split back into protein and ligand coordinates

    Parameters
    ----------
    protein_coords : Tensor | None
        Protein backbone coordinates of shape [B, L, n_atoms, 3] where n_atoms
        is typically 4 (N, CA, C, O) or 3 (N, CA, C). Can be None for ligand-only.
    protein_mask : Tensor | None
        Boolean mask of shape [B, L] indicating valid residues. Can be None.
    ligand_coords : Tensor | None
        Ligand atom coordinates of shape [B, N_ligand, 3]. Can be None for protein-only.
    ligand_mask : Tensor | None
        Boolean mask of shape [B, N_ligand] indicating valid atoms. Can be None.
    random_se3 : bool
        Whether to apply random SE(3) transformation. Default True.
    only_rot : bool
        If True, only apply rotation (no translation). Default False.
    only_trans : bool
        If True, only apply translation (no rotation). Default False.
    translation_scale : float
        Scale factor for random translation. Default 1.0.
    rotation_mode : str
        Method to generate random rotation. One of "svd", "quaternion", or "none".
        Default "svd".
    frame_type : str | None
        Type of global frame to apply. One of "norm_frame", "pca_frame", "mol_frame",
        or None to skip frame application. Default None.
    apply_stochastic_fa : bool
        Whether to apply stochastic frame alignment. Only used with frame_type.
        Default False.
    get_all_frames : bool
        Whether to get all possible frame orientations. Only used with frame_type.
        Default False.
    backbone_noise : float
        Standard deviation of Gaussian noise to add to coordinates. Default 0.0.

    Returns
    -------
    SE3AugmentedComplex
        Named tuple containing:
        - protein_coords: Transformed protein coordinates [B, L, n_atoms, 3] or None
        - protein_mask: Protein mask [B, L] or None
        - ligand_coords: Transformed ligand coordinates [B, N_ligand, 3] or None
        - ligand_mask: Ligand mask [B, N_ligand] or None

    Examples
    --------
    >>> # Protein-only augmentation
    >>> protein = torch.randn(2, 100, 4, 3)
    >>> mask = torch.ones(2, 100, dtype=torch.bool)
    >>> result = apply_se3_augmentation_protein_ligand(
    ...     protein_coords=protein,
    ...     protein_mask=mask,
    ... )

    >>> # Protein-ligand complex augmentation
    >>> protein = torch.randn(2, 100, 4, 3)
    >>> ligand = torch.randn(2, 30, 3)
    >>> result = apply_se3_augmentation_protein_ligand(
    ...     protein_coords=protein,
    ...     protein_mask=torch.ones(2, 100, dtype=torch.bool),
    ...     ligand_coords=ligand,
    ...     ligand_mask=torch.ones(2, 30, dtype=torch.bool),
    ... )
    """
    # Handle edge cases
    has_protein = protein_coords is not None
    has_ligand = ligand_coords is not None

    if not has_protein and not has_ligand:
        raise ValueError("At least one of protein_coords or ligand_coords must be provided")

    # Clone inputs to avoid modifying originals
    if has_protein:
        protein_coords = protein_coords.clone()
        if protein_mask is not None:
            protein_mask = protein_mask.clone()
    if has_ligand:
        ligand_coords = ligand_coords.clone()
        if ligand_mask is not None:
            ligand_mask = ligand_mask.clone()

    # Determine batch size and device
    if has_protein:
        B, L, n_atoms, _ = protein_coords.shape
        device = protein_coords.device
    else:
        B = ligand_coords.shape[0]
        L = 0
        n_atoms = 0
        device = ligand_coords.device

    # Prepare combined coordinates for joint transformation
    if has_protein and has_ligand:
        # Both present - concatenate for joint transformation
        B_ligand = ligand_coords.shape[0]
        assert B == B_ligand, f"Batch size mismatch: protein {B} vs ligand {B_ligand}"

        # Flatten protein coords: [B, L, n_atoms, 3] -> [B, L*n_atoms, 3]
        coords_flat = protein_coords.reshape(B, -1, 3)
        # Concatenate with ligand: [B, L*n_atoms + N_ligand, 3]
        coords = torch.cat([coords_flat, ligand_coords], dim=1)

        # Expand protein mask to atom level: [B, L] -> [B, L*n_atoms]
        if protein_mask is not None:
            seq_mask = protein_mask.unsqueeze(-1).expand(-1, -1, n_atoms).reshape(B, -1)
        else:
            seq_mask = torch.ones(B, L * n_atoms, device=device, dtype=torch.bool)

        # Concatenate masks
        if ligand_mask is not None:
            seq_mask = torch.cat([seq_mask, ligand_mask], dim=1)
        else:
            ligand_n = ligand_coords.shape[1]
            seq_mask = torch.cat([seq_mask, torch.ones(B, ligand_n, device=device, dtype=torch.bool)], dim=1)

    elif has_protein:
        # Protein-only - flatten
        coords = protein_coords.reshape(B, -1, 3)
        if protein_mask is not None:
            seq_mask = protein_mask.unsqueeze(-1).expand(-1, -1, n_atoms).reshape(B, -1)
        else:
            seq_mask = torch.ones(B, L * n_atoms, device=device, dtype=torch.bool)

    else:
        # Ligand-only
        coords = ligand_coords
        if ligand_mask is not None:
            seq_mask = ligand_mask
        else:
            seq_mask = torch.ones(B, ligand_coords.shape[1], device=device, dtype=torch.bool)

    # Apply SE(3) transformation
    if random_se3 and seq_mask.any():
        if only_rot:
            logger.debug("Only applying rotation")
            actual_translation_scale = 0.0
            actual_rotation_mode = rotation_mode
        elif only_trans:
            logger.debug("Only applying translation")
            actual_translation_scale = translation_scale
            actual_rotation_mode = "none"
        else:
            actual_translation_scale = translation_scale
            actual_rotation_mode = rotation_mode

        coords = apply_random_se3_batched(
            coords,
            atom_mask=seq_mask,
            translation_scale=actual_translation_scale,
            rotation_mode=actual_rotation_mode,
        )
    elif not random_se3:
        logger.debug("No SE(3) applied")

    # Apply global frame transformation
    if frame_type is not None and seq_mask.any():
        coords = apply_global_frame_to_coords(
            coords,
            frame_type=frame_type,
            mask=seq_mask,
            apply_stochastic_fa=apply_stochastic_fa,
            get_all_frames=get_all_frames,
        )

    # Add backbone noise
    if backbone_noise > 0:
        coords = coords + backbone_noise * torch.randn_like(coords)

    # Split back into protein and ligand
    if has_protein and has_ligand:
        # Split coordinates
        n_protein_atoms = L * n_atoms
        ligand_coords_out = coords[:, n_protein_atoms:, :]
        protein_coords_flat = coords[:, :n_protein_atoms, :]
        protein_coords_out = protein_coords_flat.reshape(B, L, n_atoms, 3)

        # Reconstruct protein mask from atom-level mask
        protein_mask_out = seq_mask[:, :n_protein_atoms].reshape(B, L, n_atoms).any(dim=-1)
        ligand_mask_out = seq_mask[:, n_protein_atoms:]

        return SE3AugmentedComplex(
            protein_coords=protein_coords_out,
            protein_mask=protein_mask_out,
            ligand_coords=ligand_coords_out,
            ligand_mask=ligand_mask_out,
        )

    elif has_protein:
        # Protein-only - reshape back
        protein_coords_out = coords.reshape(B, L, n_atoms, 3)
        protein_mask_out = seq_mask.reshape(B, L, n_atoms).any(dim=-1) if protein_mask is not None else None

        return SE3AugmentedComplex(
            protein_coords=protein_coords_out,
            protein_mask=protein_mask_out,
            ligand_coords=None,
            ligand_mask=None,
        )

    else:
        # Ligand-only
        return SE3AugmentedComplex(
            protein_coords=None,
            protein_mask=None,
            ligand_coords=coords,
            ligand_mask=seq_mask if ligand_mask is not None else None,
        )


def apply_se3_augmentation_batched(
    coords: Tensor,
    mask: Tensor | None = None,
    random_se3: bool = True,
    only_rot: bool = False,
    only_trans: bool = False,
    translation_scale: float = 1.0,
    rotation_mode: str = "svd",
    frame_type: Literal["norm_frame", "pca_frame", "mol_frame"] | None = None,
    apply_stochastic_fa: bool = False,
    get_all_frames: bool = False,
    backbone_noise: float = 0.0,
) -> Tensor:
    """Apply SE(3) augmentation to a batch of coordinates.

    Simplified interface for single coordinate tensor (either protein or ligand).

    Parameters
    ----------
    coords : Tensor
        Coordinates of shape [B, N, 3] (flat) or [B, L, n_atoms, 3] (structured).
    mask : Tensor | None
        Boolean mask. For flat coords: [B, N], for structured: [B, L].
    random_se3 : bool
        Whether to apply random SE(3) transformation. Default True.
    only_rot : bool
        If True, only apply rotation (no translation). Default False.
    only_trans : bool
        If True, only apply translation (no rotation). Default False.
    translation_scale : float
        Scale factor for random translation. Default 1.0.
    rotation_mode : str
        Method to generate random rotation. One of "svd", "quaternion", "none".
    frame_type : str | None
        Type of global frame to apply. One of "norm_frame", "pca_frame", "mol_frame".
    apply_stochastic_fa : bool
        Whether to apply stochastic frame alignment.
    get_all_frames : bool
        Whether to get all possible frame orientations.
    backbone_noise : float
        Standard deviation of Gaussian noise to add.

    Returns
    -------
    Tensor
        Transformed coordinates with same shape as input.
    """
    coords = coords.clone()
    is_structured = len(coords.shape) == 4

    if is_structured:
        B, L, n_atoms, _ = coords.shape
        device = coords.device

        # Flatten for processing
        coords_flat = coords.reshape(B, -1, 3)

        # Expand mask
        if mask is not None:
            seq_mask = mask.unsqueeze(-1).expand(-1, -1, n_atoms).reshape(B, -1)
        else:
            seq_mask = torch.ones(B, L * n_atoms, device=device, dtype=torch.bool)
    else:
        coords_flat = coords
        if mask is not None:
            seq_mask = mask
        else:
            seq_mask = torch.ones(coords.shape[0], coords.shape[1], device=coords.device, dtype=torch.bool)

    # Apply SE(3) transformation
    if random_se3 and seq_mask.any():
        if only_rot:
            actual_translation_scale = 0.0
            actual_rotation_mode = rotation_mode
        elif only_trans:
            actual_translation_scale = translation_scale
            actual_rotation_mode = "none"
        else:
            actual_translation_scale = translation_scale
            actual_rotation_mode = rotation_mode

        coords_flat = apply_random_se3_batched(
            coords_flat,
            atom_mask=seq_mask,
            translation_scale=actual_translation_scale,
            rotation_mode=actual_rotation_mode,
        )

    # Apply global frame
    if frame_type is not None and seq_mask.any():
        coords_flat = apply_global_frame_to_coords(
            coords_flat,
            frame_type=frame_type,
            mask=seq_mask,
            apply_stochastic_fa=apply_stochastic_fa,
            get_all_frames=get_all_frames,
        )

    # Add noise
    if backbone_noise > 0:
        coords_flat = coords_flat + backbone_noise * torch.randn_like(coords_flat)

    # Reshape if needed
    if is_structured:
        return coords_flat.reshape(B, L, n_atoms, 3)
    return coords_flat
