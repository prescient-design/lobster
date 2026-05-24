"""
Utility functions for binder design generation.

This module provides helper functions for the binder_design generation mode,
including chain information extraction, binder initialization, and mask creation.
"""

import torch


def get_target_chain_info(structure_data: dict, target_chain_letter: str) -> tuple[int, int, int]:
    """
    Get chain information for the target chain.

    Args:
        structure_data: Loaded PDB structure dictionary with 'real_chains' and 'chains_ids'
        target_chain_letter: Chain letter (e.g., "A", "B")

    Returns:
        chain_idx: Chain index (0, 200, 400, etc.)
        start_residue_idx: Starting residue index for this chain
        end_residue_idx: Ending residue index for this chain (exclusive)

    Example:
        For a PDB with chains A (residues 0-99) and B (residues 100-161):
        - real_chains: [65, 65, ..., 66, 66, ...]  # ord('A')=65, ord('B')=66
        - chains_ids: [0, 0, ..., 200, 200, ...]

        get_target_chain_info(data, "A") -> (0, 0, 100)
        get_target_chain_info(data, "B") -> (200, 100, 162)
    """
    # Convert chain letter to ASCII code
    target_chain_ord = ord(target_chain_letter)

    # Get real_chains tensor (contains ASCII codes for chain letters)
    real_chains = structure_data["real_chains"]
    # Note: StructureBackboneTransform renames 'chains_ids' to 'chains'
    chains_ids = structure_data.get("chains", structure_data.get("chains_ids"))

    # Find where this chain appears
    chain_mask = real_chains == target_chain_ord

    if not chain_mask.any():
        available = set(chr(c) for c in real_chains.unique().tolist())
        raise ValueError(f"Chain '{target_chain_letter}' not found in structure. Available chains: {available}")

    # Get the chain index (0, 200, 400, etc.)
    chain_idx = chains_ids[chain_mask][0].item()

    # Find start and end indices in the sequence
    chain_positions = torch.where(chain_mask)[0]
    start_residue_idx = chain_positions[0].item()
    end_residue_idx = chain_positions[-1].item() + 1

    return chain_idx, start_residue_idx, end_residue_idx


def initialize_binder_at_origin(
    binder_length: int, device: torch.device, target_coords: torch.Tensor = None, epitope_indices: list[int] = None
) -> dict:
    """
    Create initial binder structure with coordinates positioned relative to target epitope.

    Args:
        binder_length: Length of binder to create
        device: Torch device
        target_coords: Optional target coordinates tensor (L_target, 3, 3)
        epitope_indices: Optional list of residue indices (in coords_res numbering) defining the epitope.
                        If provided, binder atoms are randomly distributed in a ball of radius 12Å,
                        centered 5Å away from epitope (in direction away from target COM).
                        All binder atoms are constrained to be at least 5Å from target atoms.

    Returns:
        binder_data: Dictionary with keys:
            - 'coords_res': Coordinates tensor (L, 3, 3) initialized based on epitope or COM
            - 'sequence': Sequence tokens (L,) initialized to random valid amino acids
            - 'mask': Validity mask (L,) all ones

    Example:
        For binder_length=100 with epitope:
        coords_res shape: (100, 3, 3)
        Atoms randomly distributed in 12Å ball, centered 5Å from epitope, ≥5Å from target

        sequence shape: (100,)
        Random tokens from 0-19 (valid amino acids, excluding X=20)

        mask shape: (100,)
        All ones (all positions valid)
    """
    # Constants for initialization
    EPITOPE_DISTANCE = 5.0  # Distance from epitope center to ball center
    BALL_RADIUS = 12.0  # Radius of random distribution ball
    MIN_TARGET_DISTANCE = 5.0  # Minimum distance from target atoms
    MAX_ATTEMPTS = 100  # Maximum rejection sampling attempts per atom

    # Calculate initial position
    if target_coords is not None:
        # Calculate center of mass of target structure using CA atoms (index 1)
        ca_coords = target_coords[:, 1, :]  # (L_target, 3)
        center_of_mass = ca_coords.mean(dim=0)  # (3,)

        # Flatten all target atoms for distance checking
        all_target_atoms = target_coords.reshape(-1, 3)  # (L_target * 3, 3)

        if epitope_indices is not None and len(epitope_indices) > 0:
            # Calculate epitope center from specified residues
            epitope_ca_coords = ca_coords[epitope_indices]  # (n_epitope, 3)
            epitope_center = epitope_ca_coords.mean(dim=0)  # (3,)

            # Calculate direction vector from COM to epitope (pointing away from COM)
            direction = epitope_center - center_of_mass  # (3,)
            direction_norm = torch.norm(direction)

            if direction_norm > 1e-6:  # Avoid division by zero
                direction_unit = direction / direction_norm  # Normalize
            else:
                # If epitope is at COM, use arbitrary direction (z-axis)
                direction_unit = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)

            # Calculate ball center: 5Å away from epitope, in direction away from COM
            ball_center = epitope_center + direction_unit * EPITOPE_DISTANCE

            # Initialize coordinates tensor
            coords_res = torch.zeros((binder_length, 3, 3), dtype=torch.float32)

            # Generate random positions for each residue
            for i in range(binder_length):
                for attempt in range(MAX_ATTEMPTS):
                    # Generate random point in unit sphere using rejection sampling
                    random_vec = torch.randn(3)
                    random_vec = random_vec / torch.norm(random_vec)  # Normalize to unit sphere surface

                    # Random radius (uniform in volume, so use cube root)
                    random_radius = BALL_RADIUS * (torch.rand(1).item() ** (1 / 3))

                    # Random point in ball
                    random_point = ball_center + random_vec * random_radius

                    # Check minimum distance to all target atoms
                    distances_to_target = torch.norm(all_target_atoms - random_point.unsqueeze(0), dim=1)
                    min_distance = distances_to_target.min().item()

                    if min_distance >= MIN_TARGET_DISTANCE:
                        # Point is valid, use it for all 3 backbone atoms (N, CA, C)
                        coords_res[i, :, :] = random_point.unsqueeze(0).expand(3, 3)
                        break
                else:
                    # Max attempts reached, use ball center as fallback
                    coords_res[i, :, :] = ball_center.unsqueeze(0).expand(3, 3)

        else:
            # No epitope specified, use center of mass with random distribution
            coords_res = torch.zeros((binder_length, 3, 3), dtype=torch.float32)
            for i in range(binder_length):
                for attempt in range(MAX_ATTEMPTS):
                    # Generate random point around COM
                    random_vec = torch.randn(3)
                    random_vec = random_vec / torch.norm(random_vec)
                    random_radius = BALL_RADIUS * (torch.rand(1).item() ** (1 / 3))
                    random_point = center_of_mass + random_vec * random_radius

                    # Check minimum distance to target
                    distances_to_target = torch.norm(all_target_atoms - random_point.unsqueeze(0), dim=1)
                    min_distance = distances_to_target.min().item()

                    if min_distance >= MIN_TARGET_DISTANCE:
                        coords_res[i, :, :] = random_point.unsqueeze(0).expand(3, 3)
                        break
                else:
                    # Fallback: place at COM + offset in random direction
                    fallback_dir = torch.randn(3)
                    fallback_dir = fallback_dir / torch.norm(fallback_dir)
                    coords_res[i, :, :] = (
                        (center_of_mass + fallback_dir * MIN_TARGET_DISTANCE).unsqueeze(0).expand(3, 3)
                    )
    else:
        # Fall back to origin if no target provided
        coords_res = torch.zeros((binder_length, 3, 3), dtype=torch.float32)

    coords_res = coords_res.to(device)
    # Shape: (L, 3, 3) where:
    #   - First dim: residue index
    #   - Second dim: atom type (0=N, 1=CA, 2=C)
    #   - Third dim: xyz coordinates

    # Initialize sequence with random valid amino acids (0-19, excluding X=20)
    sequence = torch.randint(0, 20, (binder_length,), dtype=torch.int32, device=device)

    # Set first residue to Methionine (M=10 in standard AA ordering)
    # This is the canonical start codon and the first residue is kept fixed for chain break
    METHIONINE_IDX = 10  # M in alphabetical AA ordering: A=0, C=1, ..., M=10, ...
    sequence[0] = METHIONINE_IDX

    # Create validity mask (all ones - all positions are valid)
    mask = torch.ones(binder_length, dtype=torch.float32, device=device)

    return {
        "coords_res": coords_res,
        "sequence": sequence,
        "mask": mask,
    }


def get_next_chain_index(structure_data: dict) -> int:
    """
    Get the next available chain index (200, 400, 600, etc.).

    Args:
        structure_data: Loaded PDB structure dictionary with 'chains' or 'chains_ids'

    Returns:
        next_chain_idx: Next available chain index

    Example:
        If chains contains [0, 0, ..., 200, 200, ...]:
        - Max chain index is 200
        - Next available is 400

        If chains contains [0, 0, ...]:
        - Max chain index is 0
        - Next available is 200
    """
    # Note: StructureBackboneTransform renames 'chains_ids' to 'chains'
    chains_ids = structure_data.get("chains", structure_data.get("chains_ids"))

    # Find max chain index
    max_chain_idx = chains_ids.max().item()

    # Next chain index is max + 200
    next_chain_idx = max_chain_idx + 200

    # Verify it doesn't collide (rare but check)
    while next_chain_idx in chains_ids:
        next_chain_idx += 200

    return next_chain_idx


def create_binder_inpainting_masks(
    chains_ids: torch.Tensor, target_chain_idx: int, binder_chain_idx: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create inpainting masks for binder design.
    Target residues get mask=0 (fixed), binder residues get mask=1 (generate).

    IMPORTANT: The first residue of the binder is kept fixed (mask=0) to preserve
    the chain break token. This tells the model where the new chain starts,
    otherwise it would treat the binder as a continuation of the target chain.

    Args:
        chains_ids: Chain ID tensor for all residues (B, L)
        target_chain_idx: Index of target chain to keep fixed
        binder_chain_idx: Index of binder chain to generate
        device: Torch device

    Returns:
        mask_sequence: Inpainting mask for sequence (B, L)
        mask_structure: Inpainting mask for structure (B, L)

    Example:
        For a complex with:
        - Chain A (target): chain_idx=0, residues 0-99
        - Chain B (binder): chain_idx=200, residues 100-199

        chains_ids: [0,0,...,0, 200,200,...,200]  (shape: 1, 200)
        target_chain_idx: 0
        binder_chain_idx: 200

        Returns masks of shape (1, 200):
        - Positions 0-99: mask=0 (keep target fixed)
        - Position 100: mask=0 (keep first binder token fixed for chain break)
        - Positions 101-199: mask=1 (generate rest of binder)
    """
    # Create masks initialized to zeros
    B, L = chains_ids.shape
    mask_sequence = torch.zeros((B, L), dtype=torch.float32, device=device)
    mask_structure = torch.zeros((B, L), dtype=torch.float32, device=device)

    # Set binder positions to 1 (generate)
    binder_mask = chains_ids == binder_chain_idx
    mask_sequence[binder_mask] = 1.0
    mask_structure[binder_mask] = 1.0

    # Keep first binder residue fixed (mask=0) to preserve chain break token
    # Find the first position of the binder chain in each batch
    for b in range(B):
        binder_positions = torch.where(chains_ids[b] == binder_chain_idx)[0]
        if len(binder_positions) > 0:
            first_binder_idx = binder_positions[0].item()
            mask_sequence[b, first_binder_idx] = 0.0
            mask_structure[b, first_binder_idx] = 0.0

    # Target positions remain 0 (fixed)
    # Any other chains in the structure also remain 0 (fixed)

    return mask_sequence, mask_structure
