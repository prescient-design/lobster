import torch
from lobster.model.latent_generator.io._write_pdb import writepdb


def get_interface_residues(positions, mask, asym_id, interface_threshold):
    """
    Get interface residues based on pairwise CA distances across chains.

    Args:
        positions: CA atom positions [num_res, 3]
        mask: Mask for valid CA atoms [num_res]
        asym_id: Chain IDs [num_res]
        interface_threshold: Distance threshold for interface (e.g., 5.0 Angstroms)

    Returns:
        interface_residues_idxs: Indices of residues at the interface
    """
    # Calculate pairwise distances between CA atoms
    coord_diff = positions[:, None, :] - positions[None, :, :]  # [num_res, num_res, 3]
    pairwise_dists = torch.sqrt(torch.sum(coord_diff**2, dim=-1))  # [num_res, num_res]

    # Mask for different chains and valid CA atoms
    diff_chain_mask = (asym_id[:, None] != asym_id[None, :]).float()  # [num_res, num_res]
    pair_mask = mask[:, None] * mask[None, :]  # [num_res, num_res]
    mask = (diff_chain_mask * pair_mask).bool()  # [num_res, num_res]

    # Find minimum distance to any residue in a different chain
    min_dist_per_res = torch.where(mask, pairwise_dists, torch.inf).min(dim=-1)[0]  # [num_res]

    # Identify interface residues (those with min distance < threshold)
    interface_residues_idxs = torch.nonzero(min_dist_per_res < interface_threshold, as_tuple=True)[0]

    return interface_residues_idxs


def get_spatial_crop_indices(
    x: dict, interface_distance: float = 5.0, crop_size: int = 512
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get the epitope tensor for the input dictionary using interface residue logic.

    Args:
        x: Dictionary containing structure information with keys:
            - 'coords_res': Residue coordinates [num_res, num_atoms, 3]
            - 'chains': Chain IDs [num_res]
            - 'mask': Atom mask [num_res]
        interface_distance: Distance threshold for interface (default: 5.0 Angstroms)
        crop_size: Number of residues to keep in spatial crop (default: 512)

    Returns:
        tuple: (paratope_mask, epitope_mask, spatial_crop_indices)
            - paratope_mask: Mask for paratope residues
            - epitope_mask: Mask for epitope residues
            - spatial_crop_indices: Indices to spatially crop the structure, sorted by distance
    """
    positions = x["coords_res"]  # [num_res, num_atoms, 3]
    asym_id = x["chains"]  # [num_res]
    mask = x["mask"]  # [num_res]

    # Extract CA positions (atom index 1)
    ca_positions = positions[:, 1, :]  # [num_res, 3]

    # Get all interface residues
    interface_residues_idxs = get_interface_residues(ca_positions, mask, asym_id, interface_distance)

    # Pick a random chain to be the paratope
    chains = torch.unique(asym_id)
    paratope_chain = chains[torch.randint(0, len(chains), (1,)).item()]

    # Get paratope and epitope masks
    paratope_mask = asym_id == paratope_chain
    epitope_mask = asym_id != paratope_chain

    # Filter interface residues by chain type
    paratope_interface_indices = interface_residues_idxs[paratope_mask[interface_residues_idxs]]
    epitope_interface_indices = interface_residues_idxs[epitope_mask[interface_residues_idxs]]

    # Logic for spatial cropping
    # Randomly select a target residue from interface residues
    target_res_idx = torch.randint(
        low=0, high=interface_residues_idxs.shape[-1], size=(1,), device=positions.device
    ).item()

    target_res = interface_residues_idxs[target_res_idx]

    # Calculate distances from target residue to all other residues (using CA positions already computed)
    coord_diff = ca_positions[..., None, :] - ca_positions[..., None, :, :]
    ca_pairwise_dists = torch.sqrt(torch.sum(coord_diff**2, dim=-1))
    to_target_distances = ca_pairwise_dists[target_res]

    # Break ties by adding small incremental values based on index
    break_tie = torch.arange(0, to_target_distances.shape[-1], device=positions.device).float() * 1e-3
    to_target_distances = torch.where(mask.bool(), to_target_distances, torch.inf) + break_tie

    # Get indices of closest residues
    sorted_indices = torch.argsort(to_target_distances)

    # Apply crop_size (take top crop_size closest residues)
    spatial_crop_indices = sorted_indices[:crop_size].sort().values

    return paratope_mask, epitope_mask, spatial_crop_indices


def spatial_crop_transform(x: dict, crop_size: int = 512) -> dict:
    """
    Get the spatial crop transform for the input dictionary.
    """
    paratope_mask, epitope_mask, spatial_crop_indices = get_spatial_crop_indices(x, crop_size=crop_size)
    # crop the structure
    x["coords_res"] = x["coords_res"][spatial_crop_indices]
    x["mask"] = x["mask"][spatial_crop_indices]
    x["chains"] = x["chains"][spatial_crop_indices]
    x["sequence"] = x["sequence"][spatial_crop_indices]
    x["epitope_tensor"] = epitope_mask[spatial_crop_indices]
    x["paratope_tensor"] = paratope_mask[spatial_crop_indices]
    return x


if __name__ == "__main__":
    test_file = "/data2/lisanzas/sabdab/train_denovo_processed/10001_6vo3_HL_A.pt"
    x = torch.load(test_file)
    if "chains_ids" in x:
        x["chains"] = x["chains_ids"]
        del x["chains_ids"]
        # make sequence long instead of int
        x["sequence"] = x["sequence"].long()
    # Save the original non-cropped structure
    output_pdb_original = "original_structure.pdb"
    writepdb(filename=output_pdb_original, atoms=x["coords_res"], seq=x["sequence"], idx_pdb=None, bfacts=None)
    print(f"✓ Saved original structure to: {output_pdb_original}")
    print(f"  Original structure: {x['coords_res'].shape[0]} residues\n")

    x = spatial_crop_transform(x, crop_size=512)
    print("x['coords_res'].shape", x["coords_res"].shape)
    print("x['mask'].shape", x["mask"].shape)
    print("x['chains'].shape", x["chains"].shape)
    print("x['sequence'].shape", x["sequence"].shape)
    print("x['epitope_tensor'].shape", x["epitope_tensor"].shape)
    print("x['paratope_tensor'].shape", x["paratope_tensor"].shape)

    # Save the spatially cropped structure to PDB
    output_pdb_cropped = "spatially_cropped_structure.pdb"
    writepdb(filename=output_pdb_cropped, atoms=x["coords_res"], seq=x["sequence"], idx_pdb=None, bfacts=None)
    print(f"\n✓ Saved spatially cropped structure to: {output_pdb_cropped}")
    print(f"  Cropped structure: {x['coords_res'].shape[0]} residues")
    print(f"\n📊 Comparison:")
    print(f"  Original: {output_pdb_original}")
    print(f"  Cropped:  {output_pdb_cropped}")
