import torch
import torch.nn.functional as F

from lobster.model.latent_generator.utils import residue_constants


def collate_fn_backbone(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Collate fn for batching protein backbone data.

    Handles three cases:
    1. Protein-only samples (from StructureDataset): {"coords_res": ..., "mask": ..., ...}
    2. Protein+ligand samples (from LigandDataset): {"protein": {...}, "ligand": {...}}
    3. Ligand-only samples (from LigandDataset with no protein): {"protein": None, "ligand": {...}}

    When mixing different sample types in a batch, creates validity masks to track
    which samples have valid protein/ligand data. Missing data is padded with zeros
    to maintain consistent batch dimensions.
    """
    # Check if any sample has the nested protein/ligand structure
    has_nested_structure = any("ligand" in item or "protein" in item for item in batch)

    if has_nested_structure:
        # Normalize all samples to the nested structure format
        normalized_batch = []
        for item in batch:
            if "ligand" in item or "protein" in item:
                # Already in nested format
                normalized_batch.append(item)
            else:
                # Protein-only sample from StructureDataset - wrap it
                normalized_batch.append({"protein": item, "ligand": None})

        # Now process the normalized batch
        ligand_list = [bb_dict.get("ligand") for bb_dict in normalized_batch]
        protein_list = [bb_dict.get("protein") for bb_dict in normalized_batch]

        # Create validity masks for mixed batches
        protein_valid_mask = torch.tensor([p is not None for p in protein_list], dtype=torch.bool)
        ligand_valid_mask = torch.tensor([l is not None for l in ligand_list], dtype=torch.bool)

        # Get valid samples for determining padding dimensions
        valid_proteins = [p for p in protein_list if p is not None]
        valid_ligands = [l for l in ligand_list if l is not None]

        # Create padded protein batch
        if valid_proteins:
            # Get max length from valid proteins
            max_protein_len = max(p["coords_res"].shape[0] for p in valid_proteins)

            # Create dummy protein data for samples without protein
            dummy_protein = {
                "coords_res": torch.zeros(max_protein_len, 3, 3),
                "mask": torch.zeros(max_protein_len),
                "indices": torch.full((max_protein_len,), -1, dtype=torch.long),
                "sequence": torch.zeros(max_protein_len, dtype=torch.long),
                "chains": torch.full((max_protein_len,), -1, dtype=torch.long),
            }

            # Replace None with dummy data
            padded_protein_list = [p if p is not None else dummy_protein for p in protein_list]
            protein_batch = collate_fn_backbone(padded_protein_list)
        else:
            protein_batch = {}

        # Create padded ligand batch
        if valid_ligands:
            # Get max length from valid ligands
            max_ligand_len = max(l["atom_coords"].shape[0] for l in valid_ligands)

            # Create dummy ligand data for samples without ligand
            dummy_ligand = {
                "atom_coords": torch.zeros(max_ligand_len, 3),
                "mask": torch.zeros(max_ligand_len),
                "atom_indices": torch.full((max_ligand_len,), -1, dtype=torch.long),
            }
            # Add optional fields if present in valid ligands
            if "element_indices" in valid_ligands[0]:
                dummy_ligand["element_indices"] = torch.zeros(max_ligand_len, dtype=torch.long)
            if "bond_matrix" in valid_ligands[0]:
                dummy_ligand["bond_matrix"] = torch.zeros(max_ligand_len, max_ligand_len, dtype=torch.long)

            # Replace None with dummy data
            padded_ligand_list = [l if l is not None else dummy_ligand for l in ligand_list]
            ligand_batch = collate_fn_ligand(padded_ligand_list)
        else:
            ligand_batch = {}

        # Combine batches
        batch = {**protein_batch, **ligand_batch}

        # Add validity masks
        batch["protein_valid_mask"] = protein_valid_mask
        batch["ligand_valid_mask"] = ligand_valid_mask
        return batch

    max_length = max(bb_dict["coords_res"].shape[0] for bb_dict in batch)
    padded_coords_res = []
    padded_mask = []
    padded_indices = []
    padded_sequence = []
    padded_chains = []
    if "3di_states" in batch[0]:
        padded_3di_states = []
        padded_3di_descriptors = []
    if "3di_partner_index" in batch[0]:
        padded_3di_partner_index = []

    if "c6d" in batch[0]:
        padded_c6d = []
        padded_c6d_mask = []
        padded_c6d_binned = []

    if "plm_embeddings" in batch[0]:
        padded_plm_embeddings = []

    if "template_coords" in batch[0]:
        padded_template_coords = []
        padded_template_mask = []

    if "sequence_tokenized" in batch[0]:
        padded_sequence_tokenized = []

    if "epitope_tensor" in batch[0]:
        padded_epitope_tensor = []
        padded_paratope_tensor = []

    if "chain_ids_for_embedding" in batch[0]:
        padded_chain_ids_for_embedding = []

    # Per-design scalar conditioning bins (per-residue Long, 0=NULL); pad with 0. Use explicit key names
    # + `in` (works for dict and PyG-Data samples; iterating the sample yields values, not keys).
    _sc_all = [
        "scalar_cond__rg_ratio", "scalar_cond__iface_frac", "scalar_cond__iface_helix",
        "scalar_cond__iface_sheet", "scalar_cond__iface_coil", "scalar_cond__frac_arom",
    ]
    sc_keys = [k for k in _sc_all if k in batch[0]]
    padded_scalar_cond = {k: [] for k in sc_keys}

    for bb_dict in batch:
        coords_res = bb_dict["coords_res"]
        mask = bb_dict["mask"]
        indices = bb_dict["indices"]
        chains = bb_dict["chains"]
        padded_coords_res.append(
            torch.cat(
                [
                    coords_res,
                    torch.zeros(max_length - coords_res.shape[0], *coords_res.shape[1:]),
                ],
                dim=0,
            )
        )
        padded_mask.append(
            torch.cat(
                [
                    mask,
                    torch.zeros(max_length - mask.shape[0], *mask.shape[1:]),
                ],
                dim=0,
            )
        )
        padded_indices.append(
            torch.cat(
                [
                    indices,
                    torch.full((max_length - indices.shape[0],), -1, dtype=indices.dtype),
                ],
                dim=0,
            )
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
            torch.cat(
                [
                    chains,
                    torch.full((max_length - chains.shape[0],), -1, dtype=chains.dtype),
                ],
                dim=0,
            )
        )
        if "chain_ids_for_embedding" in batch[0]:
            cie = bb_dict["chain_ids_for_embedding"]
            padded_chain_ids_for_embedding.append(
                torch.cat(
                    [
                        cie,
                        # Pad with 0 -- the encoder's nn.Embedding is created
                        # with `padding_idx=0` so padding positions contribute
                        # a fixed zero vector and are excluded from gradient.
                        torch.zeros(max_length - cie.shape[0], dtype=cie.dtype),
                    ],
                    dim=0,
                )
            )
        if "epitope_tensor" in batch[0]:
            padded_epitope_tensor.append(
                torch.cat(
                    [
                        bb_dict["epitope_tensor"],
                        torch.zeros(
                            max_length - bb_dict["epitope_tensor"].shape[0], *bb_dict["epitope_tensor"].shape[1:]
                        ),
                    ],
                    dim=0,
                )
            )
            padded_paratope_tensor.append(
                torch.cat(
                    [
                        bb_dict["paratope_tensor"],
                        torch.zeros(
                            max_length - bb_dict["paratope_tensor"].shape[0], *bb_dict["paratope_tensor"].shape[1:]
                        ),
                    ],
                    dim=0,
                )
            )
        for _sck in sc_keys:
            _t = bb_dict[_sck]
            padded_scalar_cond[_sck].append(
                torch.cat([_t, torch.zeros(max_length - _t.shape[0], dtype=_t.dtype)], dim=0)
            )
        if "template_coords" in batch[0]:
            padded_template_coords.append(
                torch.cat(
                    [
                        bb_dict["template_coords"],
                        torch.zeros(
                            max_length - bb_dict["template_coords"].shape[0], *bb_dict["template_coords"].shape[1:]
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

        if "3di_partner_index" in batch[0]:
            # Right-pad with zeros (a sentinel residue index that
            # downstream consumers will mask out via seq_mask). Same
            # convention used for `3di_states` above.
            pi = bb_dict["3di_partner_index"]
            padded_3di_partner_index.append(
                torch.cat(
                    [
                        pi,
                        torch.zeros(max_length - pi.shape[0], *pi.shape[1:], dtype=pi.dtype),
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

        if "sequence_tokenized" in batch[0]:
            padded_sequence_tokenized.append(
                torch.cat(
                    [
                        bb_dict["sequence_tokenized"],
                        torch.ones(
                            max_length - bb_dict["sequence_tokenized"].shape[0],
                            *bb_dict["sequence_tokenized"].shape[1:],
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
    if "chain_ids_for_embedding" in batch[0]:
        out["chain_ids_for_embedding"] = torch.stack(padded_chain_ids_for_embedding, dim=0)
    if "3di_states" in batch[0]:
        out["3di_states"] = torch.stack(padded_3di_states, dim=0)
        out["3di_descriptors"] = torch.stack(padded_3di_descriptors, dim=0)
    if "3di_partner_index" in batch[0]:
        out["3di_partner_index"] = torch.stack(padded_3di_partner_index, dim=0)

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
    if "sequence_tokenized" in batch[0]:
        out["sequence_tokenized"] = torch.stack(padded_sequence_tokenized, dim=0)
    if "epitope_tensor" in batch[0]:
        out["epitope_tensor"] = torch.stack(padded_epitope_tensor, dim=0)
        out["paratope_tensor"] = torch.stack(padded_paratope_tensor, dim=0)
    for _sck in sc_keys:
        out[_sck] = torch.stack(padded_scalar_cond[_sck], dim=0)
    if "name" in batch[0]:
        # Heterogeneous batches: when the train mix combines datasets
        # produced by different preprocessing passes (e.g. the leflur-p
        # mix's PDB/AFDB/denovo have a `name` key, the Pinder atom14
        # train set does not), batch[0] may carry the key while a later
        # item does not. ``.get`` keeps the collate non-fatal -- the
        # `name` list ends up sparse (None entries for samples that
        # lack the key) which is purely informational downstream.
        out["name"] = [bb_dict.get("name") for bb_dict in batch]

    return out


def collate_fn_ligand(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Collate fn for batching ligand data."""
    padded_ligand_coords = []
    padded_ligand_mask = []
    padded_ligand_indices = []
    padded_element_indices = []
    padded_bond_matrices = []
    max_length = max(atom_dict["atom_coords"].shape[0] for atom_dict in batch)

    for atom_dict in batch:
        ligand_coords = atom_dict["atom_coords"]
        ligand_mask = atom_dict["mask"]
        ligand_indices = atom_dict["atom_indices"]

        padded_ligand_coords.append(
            torch.cat(
                [
                    ligand_coords,
                    torch.zeros(max_length - ligand_coords.shape[0], *ligand_coords.shape[1:]),
                ],
                dim=0,
            )
        )
        padded_ligand_mask.append(
            torch.cat(
                [
                    ligand_mask,
                    torch.zeros(max_length - ligand_mask.shape[0], *ligand_mask.shape[1:]),
                ],
                dim=0,
            )
        )
        padded_ligand_indices.append(
            torch.cat(
                [
                    ligand_indices,
                    torch.full((max_length - ligand_indices.shape[0],), -1, dtype=ligand_indices.dtype),
                ],
                dim=0,
            )
        )

        # Handle element indices if present
        if "element_indices" in atom_dict:
            element_indices = atom_dict["element_indices"]
            padded_element_indices.append(
                torch.cat(
                    [
                        element_indices,
                        torch.zeros(max_length - element_indices.shape[0], dtype=element_indices.dtype),
                    ],
                    dim=0,
                )
            )

        # Handle bond_matrix if present (shape: [N_atoms, N_atoms])
        if "bond_matrix" in atom_dict:
            bond_matrix = atom_dict["bond_matrix"]
            n_atoms = bond_matrix.shape[0]
            # Pad bond_matrix to [max_length, max_length]
            padded_bond = torch.zeros(max_length, max_length, dtype=bond_matrix.dtype)
            padded_bond[:n_atoms, :n_atoms] = bond_matrix
            padded_bond_matrices.append(padded_bond)

    out = {
        "ligand_coords": torch.stack(padded_ligand_coords, dim=0),
        "ligand_mask": torch.stack(padded_ligand_mask, dim=0),
        "ligand_indices": torch.stack(padded_ligand_indices, dim=0),
    }

    if padded_element_indices:
        out["ligand_element_indices"] = torch.stack(padded_element_indices, dim=0)

    # Add bond_matrix to output if present
    if padded_bond_matrices:
        out["bond_matrix"] = torch.stack(padded_bond_matrices, dim=0)

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
    from lobster.transforms._structure_transforms import BinderTargetTransform

    # Apply the backbone collation
    collated_batch = collate_fn_backbone(batch)

    # Apply the binder and target transform
    binder_transform = BinderTargetTransform()
    transformed_batch = binder_transform(collated_batch)

    return transformed_batch
