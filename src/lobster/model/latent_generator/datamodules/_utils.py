import logging

import torch
import torch.nn.functional as F

from lobster.model.latent_generator.utils import residue_constants

logger = logging.getLogger(__name__)


# Padding values - MUST match what's used in collation functions
PROTEIN_PADDING_VALUES = {
    "coords_res": 0.0,
    "mask": 0.0,
    "indices": -1,
    "sequence": None,  # Set at runtime to residue_constants.PEPTIDE_ALPHABET.index("-")
    "chains": -1,
    "template_coords": 0.0,
    "template_mask": 0.0,
    "3di_states": 0.0,
    "3di_descriptors": 0.0,
    "c6d": 0.0,
    "c6d_mask": False,
    "c6d_binned": 0.0,
    "plm_embeddings": 0.0,
    "graph_label": 0.0,
    "zernlike_descriptors": 0.0,
    "geometric_features": 0.0,
}

LIGAND_PADDING_VALUES = {
    "ligand_coords": 0.0,
    "ligand_mask": 0.0,
    "ligand_indices": -1,
    "ligand_element_indices": 0,
    "radius_of_gyration": 0.0,
    "solvent_accessible_surface_area": 0.0,
}


def get_padding_value(key: str, dtype: torch.dtype | None = None):
    """Get padding value for a field, matching standard collation behavior."""
    if key in PROTEIN_PADDING_VALUES:
        val = PROTEIN_PADDING_VALUES[key]
        if val is None and key == "sequence":
            return residue_constants.PEPTIDE_ALPHABET.index("-")
        return val
    if key in LIGAND_PADDING_VALUES:
        return LIGAND_PADDING_VALUES[key]
    # Default fallbacks
    if "mask" in key:
        return False if dtype == torch.bool else 0.0
    elif "indices" in key or "chains" in key:
        return -1
    else:
        return 0.0


def collate_fn_backbone(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Collate function with unified batch dimensions and validity masks.

    BACKWARDS COMPATIBILITY:
    - Pure protein-only batches: Use original collation (no validity masks)
    - Pure ligand-only batches: Use original collation (no validity masks)
    - Pure paired batches: Use original collation (no validity masks)
    - Mixed batches: Use unified batch with validity masks (NEW)

    Handles:
    - StructureDataset items: {"coords_res": ..., "mask": ..., ...}
    - LigandDataset items: {"protein": None or {...}, "ligand": {...}}
    """
    batch_size = len(batch)

    # Categorize batch items to determine if homogeneous or heterogeneous
    has_structure_items = False  # Pure protein items (StructureDataset)
    has_ligand_only_items = False  # Ligand-only items (protein=None)
    has_paired_items = False  # Protein-ligand pairs (both present)

    for item in batch:
        if "protein" in item and "ligand" in item:
            # LigandDataset format
            if item["protein"] is not None and item["ligand"] is not None:
                has_paired_items = True
            elif item["protein"] is not None:
                # Shouldn't happen, but count as structure
                has_structure_items = True
            else:
                # Ligand only
                has_ligand_only_items = True
        else:
            # StructureDataset format
            has_structure_items = True

    # Check if batch is homogeneous (backwards compatible case)
    num_types = sum([has_structure_items, has_ligand_only_items, has_paired_items])

    if num_types == 1:
        # HOMOGENEOUS BATCH - use original collation for backwards compatibility
        if has_structure_items:
            # Pure protein-only batch - use original collation
            logger.debug("Homogeneous protein-only batch - using original collation")
            return _collate_proteins(batch)
        elif has_ligand_only_items:
            # Pure ligand-only batch - extract ligands and use original collation
            logger.debug("Homogeneous ligand-only batch - using original collation")
            ligand_items = [item["ligand"] for item in batch]
            return collate_fn_ligand(ligand_items)
        elif has_paired_items:
            # Pure paired batch - extract and collate
            logger.debug("Homogeneous paired batch - using original collation")
            protein_items = [item["protein"] for item in batch]
            ligand_items = [item["ligand"] for item in batch]

            protein_collated = _collate_proteins(protein_items)
            ligand_collated = collate_fn_ligand(ligand_items)

            # Combine without validity masks (backwards compatible)
            return {**protein_collated, **ligand_collated}

    # HETEROGENEOUS BATCH - use new unified batch approach with validity masks
    logger.debug(
        f"Heterogeneous batch detected (size={batch_size}): "
        f"structure={has_structure_items}, ligand_only={has_ligand_only_items}, paired={has_paired_items}"
    )

    # Normalize all items to {"protein": ..., "ligand": ...} format
    normalized_batch = []
    for item in batch:
        if "protein" in item and "ligand" in item:
            # LigandDataset format - already normalized
            normalized_batch.append(item)
        else:
            # StructureDataset format - wrap it
            normalized_batch.append({"protein": item, "ligand": None})

    # Extract components and build validity masks
    protein_items = []
    ligand_items = []
    protein_valid_mask = []
    ligand_valid_mask = []

    for item in normalized_batch:
        protein_items.append(item["protein"])
        ligand_items.append(item["ligand"])
        protein_valid_mask.append(item["protein"] is not None)
        ligand_valid_mask.append(item["ligand"] is not None)

    # Convert to tensors
    protein_valid_mask = torch.tensor(protein_valid_mask, dtype=torch.bool)
    ligand_valid_mask = torch.tensor(ligand_valid_mask, dtype=torch.bool)

    logger.debug(
        f"Validity masks: protein={protein_valid_mask.sum().item()}/{batch_size}, "
        f"ligand={ligand_valid_mask.sum().item()}/{batch_size}"
    )

    # Collate and expand
    result = {}

    # Collate valid proteins
    if protein_valid_mask.any():
        valid_protein_items = [p for p in protein_items if p is not None]
        protein_collated = _collate_proteins(valid_protein_items)

        # Expand to full batch size if needed
        if protein_valid_mask.all():
            # All items have protein - no expansion needed
            result.update(protein_collated)
        else:
            # Some items don't have protein - need to expand with padding
            expanded = _expand_protein_to_full_batch(protein_collated, protein_valid_mask, batch_size)
            result.update(expanded)
    else:
        # No proteins in batch - create minimal placeholders
        result.update(_create_empty_protein_batch(batch_size))

    # Collate valid ligands
    if ligand_valid_mask.any():
        valid_ligand_items = [l for l in ligand_items if l is not None]
        ligand_collated = collate_fn_ligand(valid_ligand_items)

        # Expand to full batch size if needed
        if ligand_valid_mask.all():
            # All items have ligand - no expansion needed
            result.update(ligand_collated)
        else:
            # Some items don't have ligand - need to expand with padding
            expanded = _expand_ligand_to_full_batch(ligand_collated, ligand_valid_mask, batch_size)
            result.update(expanded)
    else:
        # No ligands in batch - create minimal placeholders
        result.update(_create_empty_ligand_batch(batch_size))

    # Add validity masks
    result["protein_valid_mask"] = protein_valid_mask
    result["ligand_valid_mask"] = ligand_valid_mask

    return result


def _collate_proteins(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Collate protein-only items (original collate_fn_backbone logic).

    This is the ORIGINAL implementation extracted for backwards compatibility.
    """
    max_length = max(bb_dict["coords_res"].shape[0] for bb_dict in batch)
    padded_coords_res = []
    padded_mask = []
    padded_indices = []
    padded_sequence = []
    padded_chains = []

    if "3di_states" in batch[0]:
        padded_3di_states = []
        padded_3di_descriptors = []

    if "c6d" in batch[0]:
        padded_c6d = []
        padded_c6d_mask = []
        padded_c6d_binned = []

    if "plm_embeddings" in batch[0]:
        padded_plm_embeddings = []

    if "template_coords" in batch[0]:
        padded_template_coords = []
        padded_template_mask = []

    for bb_dict in batch:
        coords_res = bb_dict["coords_res"]
        mask = bb_dict["mask"]
        indices = bb_dict["indices"]
        chains = bb_dict["chains"]

        padded_coords_res.append(
            torch.cat(
                [coords_res, torch.zeros(max_length - coords_res.shape[0], *coords_res.shape[1:])],
                dim=0,
            )
        )
        padded_mask.append(torch.cat([mask, torch.zeros(max_length - mask.shape[0], *mask.shape[1:])], dim=0))
        padded_indices.append(
            torch.cat([indices, torch.full((max_length - indices.shape[0],), -1, dtype=indices.dtype)], dim=0)
        )
        padded_sequence.append(
            torch.cat(
                [
                    bb_dict["sequence"],
                    torch.full(
                        (max_length - bb_dict["sequence"].shape[0],),
                        residue_constants.PEPTIDE_ALPHABET.index("-"),
                        dtype=bb_dict["sequence"].dtype,
                    ),
                ],
                dim=0,
            )
        )
        padded_chains.append(
            torch.cat([chains, torch.full((max_length - chains.shape[0],), -1, dtype=chains.dtype)], dim=0)
        )

        if "template_coords" in batch[0]:
            padded_template_coords.append(
                torch.cat(
                    [
                        bb_dict["template_coords"],
                        torch.zeros(
                            max_length - bb_dict["template_coords"].shape[0],
                            *bb_dict["template_coords"].shape[1:],
                        ),
                    ],
                    dim=0,
                )
            )
            padded_template_mask.append(
                torch.cat(
                    [
                        bb_dict["template_mask"],
                        torch.zeros(
                            max_length - bb_dict["template_mask"].shape[0], *bb_dict["template_mask"].shape[1:]
                        ),
                    ],
                    dim=0,
                )
            )

        if "3di_states" in batch[0]:
            padded_3di_states.append(
                torch.cat(
                    [
                        bb_dict["3di_states"],
                        torch.zeros(max_length - bb_dict["3di_states"].shape[0], *bb_dict["3di_states"].shape[1:]),
                    ],
                    dim=0,
                )
            )
            padded_3di_descriptors.append(
                torch.cat(
                    [
                        bb_dict["3di_descriptors"],
                        torch.zeros(
                            max_length - bb_dict["3di_descriptors"].shape[0], *bb_dict["3di_descriptors"].shape[1:]
                        ),
                    ],
                    dim=0,
                )
            )

        if "c6d" in batch[0]:
            # Pad c6d from [L, L, 4] to [max_len, max_len, 4]
            c6d = bb_dict["c6d"]
            c6d_binned = bb_dict["c6d_binned"]

            padding = max_length - c6d.shape[0]
            pad_depth = (0, 0)
            pad = pad_depth + (0, padding) + (0, padding)
            padded_c6d_ = F.pad(c6d, pad, mode="constant", value=0)
            padded_c6d_binned_ = F.pad(c6d_binned, pad, mode="constant", value=0)
            # Pad c6d_mask from [L, L] to [max_len, max_len]
            c6d_mask = bb_dict["c6d_mask"]
            padding = max_length - c6d_mask.shape[0]
            pad = (0, padding) + (0, padding)
            padded_c6d_mask_ = F.pad(c6d_mask, pad, mode="constant", value=False)
            padded_c6d.append(padded_c6d_)
            padded_c6d_mask.append(padded_c6d_mask_)
            padded_c6d_binned.append(padded_c6d_binned_)

        if "plm_embeddings" in batch[0]:
            padded_plm_embeddings.append(
                torch.cat(
                    [
                        bb_dict["plm_embeddings"],
                        torch.zeros(
                            max_length - bb_dict["plm_embeddings"].shape[0], *bb_dict["plm_embeddings"].shape[1:]
                        ),
                    ],
                    dim=0,
                )
            )

    out = {
        "coords_res": torch.stack(padded_coords_res, dim=0),
        "mask": torch.stack(padded_mask, dim=0),
        "indices": torch.stack(padded_indices, dim=0),
        "sequence": torch.stack(padded_sequence, dim=0),
        "chains": torch.stack(padded_chains, dim=0),
    }

    if "3di_states" in batch[0]:
        out["3di_states"] = torch.stack(padded_3di_states, dim=0)
        out["3di_descriptors"] = torch.stack(padded_3di_descriptors, dim=0)

    if "c6d" in batch[0]:
        out["c6d"] = torch.stack(padded_c6d, dim=0)
        out["c6d_mask"] = torch.stack(padded_c6d_mask, dim=0)
        out["c6d_binned"] = torch.stack(padded_c6d_binned, dim=0)

    if "graph_label" in batch[0]:
        out["graph_label"] = torch.stack([bb_dict["graph_label"] for bb_dict in batch], dim=0)
    if "zernlike_descriptors" in batch[0]:
        out["zernlike_descriptors"] = torch.stack([bb_dict["zernlike_descriptors"] for bb_dict in batch], dim=0)
    if "geometric_features" in batch[0]:
        out["geometric_features"] = torch.stack([bb_dict["geometric_features"] for bb_dict in batch], dim=0)
    if "plm_embeddings" in batch[0]:
        out["plm_embeddings"] = torch.stack(padded_plm_embeddings, dim=0)
    if "template_coords" in batch[0]:
        out["template_coords"] = torch.stack(padded_template_coords, dim=0)
        out["template_mask"] = torch.stack(padded_template_mask, dim=0)

    if "name" in batch[0]:
        out["name"] = [bb_dict.get("name", None) for bb_dict in batch]

    return out


def _expand_protein_to_full_batch(
    collated: dict[str, torch.Tensor], valid_mask: torch.Tensor, batch_size: int
) -> dict[str, torch.Tensor]:
    """Expand collated protein batch to full batch size with padding."""
    result = {}
    n_valid = collated["coords_res"].shape[0]

    assert valid_mask.sum() == n_valid, f"valid_mask count {valid_mask.sum()} doesn't match {n_valid}"

    for key, value in collated.items():
        if key == "name":
            # Handle name specially (list, not tensor)
            expanded_names = [None] * batch_size
            valid_idx = 0
            for i in range(batch_size):
                if valid_mask[i]:
                    expanded_names[i] = value[valid_idx]
                    valid_idx += 1
            result[key] = expanded_names
            continue

        if not isinstance(value, torch.Tensor):
            result[key] = value
            continue

        # Get padding value using centralized function
        pad_value = get_padding_value(key, value.dtype)

        # Create full batch tensor
        full_shape = (batch_size,) + value.shape[1:]
        if value.dtype == torch.bool:
            expanded = torch.zeros(full_shape, dtype=torch.bool, device=value.device)
        else:
            expanded = torch.full(full_shape, pad_value, dtype=value.dtype, device=value.device)

        # Fill in valid positions
        expanded[valid_mask] = value
        result[key] = expanded

    return result


def _expand_ligand_to_full_batch(
    collated: dict[str, torch.Tensor], valid_mask: torch.Tensor, batch_size: int
) -> dict[str, torch.Tensor]:
    """Expand collated ligand batch to full batch size with padding."""
    result = {}
    n_valid = collated["ligand_coords"].shape[0]

    assert valid_mask.sum() == n_valid, f"valid_mask count {valid_mask.sum()} doesn't match {n_valid}"

    for key, value in collated.items():
        if not isinstance(value, torch.Tensor):
            result[key] = value
            continue

        # Get padding value using centralized function
        pad_value = get_padding_value(key, value.dtype)

        # Create full batch tensor
        full_shape = (batch_size,) + value.shape[1:]
        if value.dtype == torch.bool:
            expanded = torch.zeros(full_shape, dtype=torch.bool, device=value.device)
        else:
            expanded = torch.full(full_shape, pad_value, dtype=value.dtype, device=value.device)

        # Fill in valid positions
        expanded[valid_mask] = value
        result[key] = expanded

    return result


def _create_empty_protein_batch(batch_size: int) -> dict[str, torch.Tensor]:
    """Create minimal empty protein batch as placeholder."""
    return {
        "coords_res": torch.zeros(batch_size, 1, 3, 3),
        "mask": torch.zeros(batch_size, 1, dtype=torch.bool),
        "indices": torch.full((batch_size, 1), -1, dtype=torch.long),
        "sequence": torch.full((batch_size, 1), -1, dtype=torch.long),
        "chains": torch.full((batch_size, 1), -1, dtype=torch.long),
    }


def _create_empty_ligand_batch(batch_size: int) -> dict[str, torch.Tensor]:
    """Create minimal empty ligand batch as placeholder."""
    return {
        "ligand_coords": torch.zeros(batch_size, 1, 3),
        "ligand_mask": torch.zeros(batch_size, 1, dtype=torch.bool),
        "ligand_indices": torch.full((batch_size, 1), -1, dtype=torch.long),
    }


def collate_fn_ligand(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Collate fn for batching ligand data.

    Handles:
    - atom_coords: [N_atoms, 3] -> [batch, max_atoms, 3]
    - mask: [N_atoms] -> [batch, max_atoms]
    - atom_indices: [N_atoms] -> [batch, max_atoms]
    - element_indices: [N_atoms] -> [batch, max_atoms] (optional)
    - bond_matrix: [N_atoms, N_atoms] -> [batch, max_atoms, max_atoms] (optional)
    - smiles: str (optional, passed through as list)
    """
    padded_ligand_coords = []
    padded_ligand_mask = []
    padded_ligand_indices = []
    padded_element_indices = []
    padded_bond_matrices = []
    smiles_list = []
    max_length = max(atom_dict["atom_coords"].shape[0] for atom_dict in batch)

    has_element_indices = "element_indices" in batch[0]
    has_bond_matrix = "bond_matrix" in batch[0]
    has_smiles = "smiles" in batch[0]

    for atom_dict in batch:
        ligand_coords = atom_dict["atom_coords"]
        ligand_mask = atom_dict["mask"]
        ligand_indices = atom_dict["atom_indices"]
        n_atoms = ligand_coords.shape[0]
        pad_length = max_length - n_atoms

        padded_ligand_coords.append(
            torch.cat([ligand_coords, torch.zeros(pad_length, *ligand_coords.shape[1:])], dim=0)
        )
        padded_ligand_mask.append(torch.cat([ligand_mask, torch.zeros(pad_length, *ligand_mask.shape[1:])], dim=0))
        padded_ligand_indices.append(
            torch.cat(
                [ligand_indices, torch.full((pad_length,), -1, dtype=ligand_indices.dtype)],
                dim=0,
            )
        )

        # Handle element indices if present
        if has_element_indices:
            element_indices = atom_dict["element_indices"]
            padded_element_indices.append(
                torch.cat(
                    [element_indices, torch.zeros(pad_length, dtype=element_indices.dtype)],
                    dim=0,
                )
            )

        # Handle bond matrix if present
        if has_bond_matrix:
            bond_matrix = atom_dict["bond_matrix"]
            # Pad bond_matrix from [n_atoms, n_atoms] to [max_length, max_length]
            padded_bond = torch.zeros(max_length, max_length, dtype=bond_matrix.dtype)
            padded_bond[:n_atoms, :n_atoms] = bond_matrix
            padded_bond_matrices.append(padded_bond)

        # Handle SMILES if present
        if has_smiles:
            smiles_list.append(atom_dict["smiles"])

    out = {
        "ligand_coords": torch.stack(padded_ligand_coords, dim=0),
        "ligand_mask": torch.stack(padded_ligand_mask, dim=0),
        "ligand_indices": torch.stack(padded_ligand_indices, dim=0),
    }

    if padded_element_indices:
        out["ligand_element_indices"] = torch.stack(padded_element_indices, dim=0)

    if padded_bond_matrices:
        out["bond_matrix"] = torch.stack(padded_bond_matrices, dim=0)

    if smiles_list:
        out["smiles"] = smiles_list

    # Handle additional properties like radius_of_gyration
    if "radius_of_gyration" in batch[0]:
        out["radius_of_gyration"] = torch.tensor(
            [atom_dict["radius_of_gyration"] for atom_dict in batch], dtype=torch.float32
        )

    # Handle SASA property
    if "solvent_accessible_surface_area" in batch[0]:
        out["solvent_accessible_surface_area"] = torch.tensor(
            [atom_dict["solvent_accessible_surface_area"] for atom_dict in batch], dtype=torch.float32
        )

    return out


def collate_fn_backbone_binder_target(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Collate function for batching protein backbone data with binder and target transform."""
    from latent_generator.datasets._transforms import BinderTargetTransform

    # Apply the backbone collation
    collated_batch = collate_fn_backbone(batch)

    # Apply the binder and target transform
    binder_transform = BinderTargetTransform()
    transformed_batch = binder_transform(collated_batch)

    return transformed_batch
