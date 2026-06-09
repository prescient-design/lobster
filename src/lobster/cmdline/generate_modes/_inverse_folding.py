"""LeFlur inverse-folding mode.

Generates sequences conditioned on a fixed input structure. The input PDB
provides Cα coordinates that are projected into the structure latent and
held fixed while the sequence stream is sampled. Optionally validates each
generated sequence with ESMFold against the input structure.

Called from :func:`lobster.cmdline.generate.generate` when
``cfg.generation.mode == "inverse_folding"``.
"""

from __future__ import annotations

from pathlib import Path
import glob

from loguru import logger
from omegaconf import (
    DictConfig,
    ListConfig,
)
from tmtools import tm_align
import torch

from lobster.metrics import (
    calculate_percent_identity,
    calculate_aggregate_stats,
    build_multichain_sequence_string,
    predict_structure_with_esmfold,
)
from lobster.model.latent_generator.io import (
    writepdb,
    load_pdb,
)
from lobster.model.latent_generator.utils.residue_constants import (
    convert_lobster_aa_tokenization_to_standard_aa,
    restype_order_with_x_inv,
)
from lobster.transforms._structure_transforms import StructureBackboneTransform

from ._shared import _check_sequence_tokens, _validate_with_esmfold


def _generate_inverse_folding(
    model, cfg: DictConfig, device: torch.device, output_dir: Path, plm_fold=None, csv_writer=None, plotter=None
) -> None:
    """Generate sequences for given structures (inverse folding)."""
    logger.info("Starting inverse folding generation...")

    # Get input structure paths
    input_structures = cfg.generation.input_structures
    if not input_structures:
        raise ValueError("input_structures must be provided for inverse folding mode")

    # Handle different input formats
    structure_paths = []
    if isinstance(input_structures, str):
        # Single path or glob pattern
        if "*" in input_structures or "?" in input_structures:
            # Glob pattern
            structure_paths = glob.glob(input_structures)
        else:
            # Single file or directory
            path = Path(input_structures)
            if path.is_file():
                structure_paths = [str(path)]
            elif path.is_dir():
                # Find all structure files in directory (PDB, CIF, PT)
                structure_paths = list(glob.glob(str(path / "*.pdb")))
                structure_paths.extend(glob.glob(str(path / "*.cif")))
                structure_paths.extend(glob.glob(str(path / "*.pt")))
            else:
                raise ValueError(f"Input path does not exist: {input_structures}")
    elif isinstance(input_structures, (list, tuple, ListConfig)):
        # List of paths (includes OmegaConf ListConfig)
        for path_str in input_structures:
            path = Path(path_str)
            if path.is_file():
                structure_paths.append(str(path))
            else:
                logger.warning(f"Skipping non-existent file: {path_str}")
    else:
        raise ValueError(f"Invalid input_structures format: {type(input_structures)}")

    if not structure_paths:
        raise ValueError("No valid structure files found in input_structures")

    logger.info(f"Found {len(structure_paths)} structure files to process")

    gen_cfg = cfg.generation
    nsteps = gen_cfg.get("nsteps", 100)
    batch_size = gen_cfg.get("batch_size", 1)
    n_trials = gen_cfg.get("n_trials", 1)  # Number of trials for best output selection
    n_designs_per_structure = gen_cfg.get("n_designs_per_structure", 1)  # Number of designs to generate per structure

    logger.info(f"Processing structures with {nsteps} generation steps, batch size {batch_size}, n_trials {n_trials}")
    logger.info(f"Generating {n_designs_per_structure} sequence design(s) per structure")

    # Build ligand file mapping for pocket AAR computation (optional)
    ligand_structures_cfg = gen_cfg.get("ligand_structures", None)
    pocket_distance_threshold = gen_cfg.get("pocket_distance_threshold", 5.0)
    ligand_file_map = {}  # Maps protein path -> ligand path

    if ligand_structures_cfg is not None:
        # Resolve ligand file paths
        ligand_paths = []
        if isinstance(ligand_structures_cfg, str):
            if "*" in ligand_structures_cfg or "?" in ligand_structures_cfg:
                ligand_paths = glob.glob(ligand_structures_cfg)
            else:
                lpath = Path(ligand_structures_cfg)
                if lpath.is_file():
                    ligand_paths = [str(lpath)]
                elif lpath.is_dir():
                    ligand_paths = list(glob.glob(str(lpath / "*ligand.pt")))

        # Build mapping: for each ligand file, find matching protein file
        # Convention: *_ligand.pt <-> *_protein.pt (same prefix)
        ligand_by_prefix = {}
        for lp in ligand_paths:
            prefix = Path(lp).stem.replace("_ligand", "")
            ligand_by_prefix[prefix] = lp

        for sp in structure_paths:
            prefix = Path(sp).stem.replace("_protein", "")
            if prefix in ligand_by_prefix:
                ligand_file_map[sp] = ligand_by_prefix[prefix]

        logger.info(
            f"Pocket AAR enabled: found {len(ligand_file_map)}/{len(structure_paths)} "
            f"matching ligand files (threshold={pocket_distance_threshold} Å)"
        )
    else:
        logger.info("Pocket AAR disabled (no ligand_structures configured)")

    def _compute_pocket_mask(protein_coords, ligand_coords, protein_mask=None, threshold=5.0):
        """Compute pocket mask: residues with CA within threshold of any ligand atom."""
        if protein_coords.dim() == 3:
            ca_coords = protein_coords[:, 1, :]  # CA atoms (index 1)
        else:
            ca_coords = protein_coords
        distances = torch.cdist(ca_coords.unsqueeze(0), ligand_coords.unsqueeze(0)).squeeze(0)
        min_distances = distances.min(dim=1).values
        pocket_mask = min_distances < threshold
        if protein_mask is not None:
            pocket_mask = pocket_mask & protein_mask.bool()
        return pocket_mask

    def _compute_aar(predicted_seq, ground_truth_seq, aar_mask=None):
        """Compute amino acid recovery rate (0-1) with optional mask."""
        if aar_mask is not None:
            aar_mask = aar_mask.bool()
            if aar_mask.sum() == 0:
                return float("nan")
            predicted_seq = predicted_seq[aar_mask]
            ground_truth_seq = ground_truth_seq[aar_mask]
        if len(predicted_seq) == 0:
            return float("nan")
        return (predicted_seq == ground_truth_seq).float().mean().item()

    # Initialize StructureBackboneTransform
    structure_transform = StructureBackboneTransform(max_length=cfg.generation.get("max_length", 512))

    # Initialize aggregate statistics collection
    all_percent_identities = []
    all_plddt_scores = []
    all_predicted_aligned_errors = []
    all_tm_scores = []
    all_rmsd_scores = []

    # Pocket AAR aggregate statistics
    all_aar_overall = []
    all_aar_pocket = []
    all_aar_nonpocket = []
    all_n_pocket_residues = []

    with torch.no_grad():
        # Process structure files in batches
        for batch_start in range(0, len(structure_paths), batch_size):
            batch_end = min(batch_start + batch_size, len(structure_paths))
            batch_paths = structure_paths[batch_start:batch_end]
            batch_idx = batch_start // batch_size

            logger.info(f"Processing batch {batch_idx + 1}/{(len(structure_paths) + batch_size - 1) // batch_size}")

            # Load structures from files
            batch_data = []
            valid_indices = []
            max_len = cfg.generation.get("max_length", 512)

            for i, structure_path in enumerate(batch_paths):
                logger.info(f"Loading {structure_path}")

                # Check file extension to determine loading method
                if structure_path.endswith(".pt"):
                    # Load .pt file directly
                    try:
                        structure_data = torch.load(structure_path, map_location="cpu")
                        if structure_data is not None:
                            # Skip structures that exceed max_length before cropping
                            raw_length = structure_data["coords_res"].shape[0]
                            if raw_length > max_len:
                                logger.info(
                                    f"Skipping structure {structure_path} - too long "
                                    f"({raw_length} residues, maximum {max_len})"
                                )
                                continue
                            # Apply StructureBackboneTransform
                            structure_data = structure_transform(structure_data)
                            batch_data.append(structure_data)
                            valid_indices.append(i)
                        else:
                            logger.warning(f"Failed to load structure from {structure_path} - data is None")
                    except Exception as e:
                        logger.warning(f"Failed to load .pt file {structure_path}: {e}")
                else:
                    # Load PDB/CIF file using existing method
                    structure_data = load_pdb(structure_path, add_batch_dim=False)
                    if structure_data is not None:
                        # Skip structures that exceed max_length before cropping
                        raw_length = structure_data["coords_res"].shape[0]
                        if raw_length > max_len:
                            logger.info(
                                f"Skipping structure {structure_path} - too long "
                                f"({raw_length} residues, maximum {max_len})"
                            )
                            continue
                        # Apply StructureBackboneTransform
                        structure_data = structure_transform(structure_data)
                        batch_data.append(structure_data)
                        valid_indices.append(i)
                    else:
                        logger.warning(f"Failed to load structure from {structure_path}")

            if not batch_data:
                logger.warning(f"No valid structures in batch {batch_idx + 1}, skipping")
                continue

            # Filter structures by minimum length (30 residues) and make sure sequence tensor does not contain more than 10% 20s
            filtered_batch_data = []
            filtered_valid_indices = []
            for i, data in enumerate(batch_data):
                if data["coords_res"].shape[0] >= 30:
                    percent_20s = (data["sequence"] == 20).sum() / data["sequence"].shape[0]
                    if percent_20s > 0.1:
                        logger.info(
                            f"Skipping structure {batch_paths[valid_indices[i]]} - sequence tensor contains more than 10% 20s"
                        )
                        continue
                    filtered_batch_data.append(data)
                    filtered_valid_indices.append(valid_indices[i])
                else:
                    logger.info(
                        f"Skipping structure {batch_paths[valid_indices[i]]} - too short ({data['coords_res'].shape[0]} residues, minimum 30)"
                    )

            if not filtered_batch_data:
                logger.warning(f"No structures with sufficient length in batch {batch_idx + 1}, skipping")
                continue

            # Prepare batch tensors
            max_length = max(data["coords_res"].shape[0] for data in filtered_batch_data)
            B = len(filtered_batch_data)

            # Initialize tensors
            coords_res = torch.zeros((B, max_length, 3, 3), device=device)
            mask = torch.zeros((B, max_length), device=device)
            indices = torch.zeros((B, max_length), device=device, dtype=torch.long)

            # Fill batch tensors
            for i, data in enumerate(filtered_batch_data):
                L = data["coords_res"].shape[0]
                coords_res[i, :L] = data["coords_res"].to(device)
                mask[i, :L] = data["mask"].to(device)
                indices[i, :L] = data["indices"].to(device)

            # Handle NaN coordinates
            nan_indices = torch.isnan(coords_res).any(dim=-1).any(dim=-1)
            mask[nan_indices] = 0
            coords_res[nan_indices] = 0

            logger.info(f"Batch {batch_idx + 1}: {B} structures, max length {max_length}")

            # Load ligand coordinates for pocket AAR (if configured)
            batch_ligand_coords = []
            for fvi in filtered_valid_indices:
                protein_path = batch_paths[fvi]
                ligand_path = ligand_file_map.get(protein_path)
                if ligand_path is not None:
                    try:
                        ligand_data_loaded = torch.load(ligand_path, weights_only=False, map_location="cpu")
                        lig_coords = ligand_data_loaded.get(
                            "atom_coords", ligand_data_loaded.get("coords", ligand_data_loaded.get("ligand_coords"))
                        )
                        if lig_coords is not None:
                            batch_ligand_coords.append(lig_coords.to(device))
                        else:
                            logger.warning(f"No ligand coordinates found in {ligand_path}")
                            batch_ligand_coords.append(None)
                    except Exception as e:
                        logger.warning(f"Failed to load ligand file {ligand_path}: {e}")
                        batch_ligand_coords.append(None)
                else:
                    batch_ligand_coords.append(None)

            # Loop over designs - generate multiple independent designs per structure
            for design_idx in range(n_designs_per_structure):
                if n_designs_per_structure > 1:
                    logger.info("=" * 60)
                    logger.info(f"DESIGN {design_idx + 1}/{n_designs_per_structure} for batch {batch_idx + 1}")
                    logger.info("=" * 60)

                # Run multiple trials and select best based on TM-score
                best_trial_results = []

                for trial in range(n_trials):
                    logger.info(
                        f"Trial {trial + 1}/{n_trials} for batch {batch_idx + 1}, design {design_idx + 1}/{n_designs_per_structure}"
                    )

                    # Retry loop for quality control (like unconditional generation)
                    if gen_cfg.get("enable_sequence_token_check", True):
                        max_retries = gen_cfg.get("sequence_token_check_retries", 10)
                    else:
                        max_retries = 0
                    retry_count = 0
                    valid_sequences_generated = False

                    while retry_count <= max_retries and not valid_sequences_generated:
                        if retry_count > 0:
                            logger.info(f"  Retry attempt {retry_count}/{max_retries}")

                        # Generate sequences
                        generate_sample = model.generate_sample(
                            length=max_length,
                            num_samples=B,
                            inverse_folding=True,
                            nsteps=nsteps,
                            input_structure_coords=coords_res,
                            input_mask=mask,
                            input_indices=indices,
                            temperature_seq=gen_cfg.get("temperature_seq", 0.5),
                            stochasticity_seq=gen_cfg.get("stochasticity_seq", 20),
                            asynchronous_sampling=gen_cfg.get("asynchronous_sampling", False),
                        )

                        # Decode structures
                        decoded_x = model.decode_structure(generate_sample, mask)

                        # Extract coordinates
                        x_recon_xyz = None
                        for decoder_name in decoded_x:
                            if "vit_decoder" == decoder_name:
                                x_recon_xyz = decoded_x[decoder_name]
                                break

                        # Extract sequences
                        if generate_sample["sequence_logits"].shape[-1] == 33:
                            seq = convert_lobster_aa_tokenization_to_standard_aa(
                                generate_sample["sequence_logits"], device=device
                            )
                        else:
                            seq = generate_sample["sequence_logits"].argmax(dim=-1)
                            seq[seq > 21] = 20

                        # Quality control check for invalid tokens
                        is_valid, error_msg = _check_sequence_tokens(seq, mask, "inverse folding")
                        if not is_valid:
                            logger.warning(f"  Quality control FAILED: {error_msg}")
                            retry_count += 1
                            if retry_count > max_retries:
                                logger.warning(
                                    f"  Max retries ({max_retries}) exceeded for trial {trial + 1}. "
                                    f"Using argmax without 'X' token as fallback."
                                )

                                # Store original sequence (with X tokens) for comparison
                                seq_with_x = seq.clone()

                                # Re-extract sequences with X-masked logits to avoid unknown tokens
                                if generate_sample["sequence_logits"].shape[-1] == 33:
                                    # Mask token 24 (X) in 33-token scheme
                                    masked_logits = generate_sample["sequence_logits"].clone()
                                    masked_logits[..., 24] = float("-inf")
                                    seq = convert_lobster_aa_tokenization_to_standard_aa(masked_logits, device=device)
                                else:
                                    # Mask token 20 (X) in standard scheme
                                    masked_logits = generate_sample["sequence_logits"].clone()
                                    masked_logits[..., 20] = float("-inf")
                                    seq = masked_logits.argmax(dim=-1)
                                    seq[seq > 21] = 20

                                # Log sequences for visual inspection
                                logger.info("  Sequence comparison (original with X vs. X-masked):")
                                for i in range(seq.shape[0]):
                                    # Convert token indices to amino acid strings
                                    valid_positions = mask[i] == 1
                                    seq_with_x_str = "".join(
                                        [
                                            restype_order_with_x_inv.get(int(t), "?")
                                            for t in seq_with_x[i, valid_positions].cpu().numpy()
                                        ]
                                    )
                                    seq_masked_str = "".join(
                                        [
                                            restype_order_with_x_inv.get(int(t), "?")
                                            for t in seq[i, valid_positions].cpu().numpy()
                                        ]
                                    )

                                    # Count X tokens
                                    num_x_before = seq_with_x_str.count("X")
                                    num_x_after = seq_masked_str.count("X")

                                    logger.info(f"    Sample {i}:")
                                    logger.info(
                                        f"      Before (X count={num_x_before}): {seq_with_x_str[:100]}{'...' if len(seq_with_x_str) > 100 else ''}"
                                    )
                                    logger.info(
                                        f"      After  (X count={num_x_after}): {seq_masked_str[:100]}{'...' if len(seq_masked_str) > 100 else ''}"
                                    )

                                valid_sequences_generated = True
                                break
                            logger.warning(f"  Regenerating sequences (retry {retry_count}/{max_retries})")
                            continue
                        else:
                            logger.info("  Quality control PASSED: All sequences contain valid amino acids")
                            valid_sequences_generated = True

                    # Extract structure tokens (argmax)
                    structure_tokens = generate_sample["structure_logits"].argmax(dim=-1)  # Shape: [batch_size, length]

                    # Calculate TM-scores for this trial
                    trial_tm_scores = []
                    outputs = None
                    pred_coords = None
                    trial_folded_structure_metrics = None

                    for i in range(B):
                        # Get original coordinates
                        orig_coords = coords_res[i, mask[i] == 1, :, :]  # Original structure

                        # Get generated sequence
                        seq_i = seq[i, mask[i] == 1]

                        # Get chain information for this structure
                        chains_i = filtered_batch_data[i]["chains"].to(device)[mask[i] == 1]

                        # For inverse folding, we need to fold the generated sequence with ESMFold
                        # and compare with the original structure
                        if plm_fold is not None:
                            # Parse chain groups from config
                            esmfold_chain_groups = cfg.generation.get("esmfold_chain_groups", None)

                            # If not specified, use all chains (default behavior for backwards compatibility)
                            if esmfold_chain_groups is None:
                                unique_chains = chains_i.unique().tolist()
                                esmfold_chain_groups = [unique_chains]

                            # Log available chains for debugging
                            available_chains = chains_i.unique().tolist()
                            logger.info(f"Available chains in structure: {available_chains}")
                            logger.info(
                                f"Predicting {len(esmfold_chain_groups)} chain group(s): {esmfold_chain_groups}"
                            )

                            # Run ESMFold prediction for each chain group
                            chain_group_results = []
                            for group_idx, chain_group in enumerate(esmfold_chain_groups):
                                logger.info(
                                    f"ESMFold prediction for chain group {group_idx + 1}/{len(esmfold_chain_groups)}: "
                                    f"{chain_group}"
                                )

                                # Validate chain group
                                invalid_chains = [c for c in chain_group if c not in available_chains]
                                if invalid_chains:
                                    logger.warning(
                                        f"Chain group {chain_group} contains invalid chain IDs: {invalid_chains}. "
                                        f"Available chains: {available_chains}. Skipping this group."
                                    )
                                    continue

                                if not chain_group:
                                    logger.warning("Empty chain group specified, skipping")
                                    continue

                                # Use refactored ESMFold prediction function
                                result = predict_structure_with_esmfold(
                                    plm_fold=plm_fold,
                                    seq_i=seq_i,
                                    chains_i=chains_i,
                                    orig_coords=orig_coords,
                                    gen_coords=None,  # No generated coords for inverse folding
                                    mask_i=mask[i],
                                    cfg=cfg,
                                    device=device,
                                    restype_order_inv=restype_order_with_x_inv,
                                    chain_group=chain_group,  # Specify which chains to predict
                                )

                                # Skip if sequence too long for ESMFold
                                if result is None:
                                    logger.warning(
                                        f"Chain group {chain_group} exceeds ESMFold max length, skipping ESMFold validation"
                                    )
                                    continue

                                chain_group_results.append(result)

                                logger.info(
                                    f"Chain group {chain_group}: TM-score: "
                                    f"{result['folded_structure_metrics']['_tm_score']:.3f}, "
                                    f"Chains: {result['num_chains']}, Residues: {result['num_residues']}"
                                )

                            # Handle results: use first group as primary, store all
                            if chain_group_results:
                                # Use FIRST chain group as primary result (user controls priority by ordering)
                                primary_result = chain_group_results[0]

                                logger.info(
                                    f"Using first chain group {primary_result['chain_group']} as primary result: "
                                    f"TM-score {primary_result['folded_structure_metrics']['_tm_score']:.3f}"
                                )

                                # Log all other results for comparison
                                if len(chain_group_results) > 1:
                                    logger.info("Additional chain group results:")
                                    for idx, result in enumerate(chain_group_results[1:], start=2):
                                        logger.info(
                                            f"  Chain group {idx}/{len(chain_group_results)} {result['chain_group']}: "
                                            f"TM-score {result['folded_structure_metrics']['_tm_score']:.3f}, "
                                            f"Chains: {result['num_chains']}, Residues: {result['num_residues']}"
                                        )

                                # Use primary result for output
                                trial_tm_scores.append(primary_result["folded_structure_metrics"]["_tm_score"])
                                outputs = primary_result["esmfold_outputs"]
                                pred_coords = primary_result["pred_coords"]
                                trial_folded_structure_metrics = primary_result["folded_structure_metrics"]

                                # Store ALL results for later analysis
                                trial_folded_structure_metrics["_all_chain_group_results"] = chain_group_results
                                trial_folded_structure_metrics["_primary_chain_group"] = primary_result["chain_group"]
                            else:
                                # Fallback: if all chain groups invalid, use all chains
                                logger.warning("No valid chain groups found, falling back to all chains")
                                result = predict_structure_with_esmfold(
                                    plm_fold=plm_fold,
                                    seq_i=seq_i,
                                    chains_i=chains_i,
                                    orig_coords=orig_coords,
                                    gen_coords=None,
                                    mask_i=mask[i],
                                    cfg=cfg,
                                    device=device,
                                    restype_order_inv=restype_order_with_x_inv,
                                )

                                # Skip if sequence too long for ESMFold
                                if result is None:
                                    logger.warning(
                                        f"Structure exceeds ESMFold max length ({len(seq_i)} residues with linkers), "
                                        "skipping ESMFold validation for this batch"
                                    )
                                    trial_tm_scores.append(float("nan"))
                                    outputs = None
                                    pred_coords = None
                                    trial_folded_structure_metrics = {"_skipped": True, "_reason": "sequence_too_long"}
                                else:
                                    trial_tm_scores.append(result["folded_structure_metrics"]["_tm_score"])
                                    outputs = result["esmfold_outputs"]
                                    pred_coords = result["pred_coords"]
                                    trial_folded_structure_metrics = result["folded_structure_metrics"]
                                    logger.info(f"TM-score: {result['folded_structure_metrics']['_tm_score']:.3f}")

                        else:
                            # If ESMFold is not available, use generated structure as fallback
                            # Build sequence string for TM-align
                            sequence_str = build_multichain_sequence_string(seq_i, chains_i, restype_order_with_x_inv)

                            gen_coords = x_recon_xyz[i, mask[i] == 1, :, :]  # Generated structure
                            tm_out = tm_align(
                                gen_coords[:, 1, :].cpu().numpy(),  # CA atoms of generated structure
                                orig_coords[:, 1, :].detach().cpu().numpy(),  # CA atoms of original structure
                                sequence_str,
                                sequence_str,
                            )
                            trial_tm_scores.append(tm_out.tm_norm_chain1)

                    # Store trial results
                    best_trial_results.append(
                        {
                            "trial": trial,
                            "tm_scores": trial_tm_scores,
                            "avg_tm_score": sum(trial_tm_scores) / len(trial_tm_scores),
                            "generate_sample": generate_sample,
                            "x_recon_xyz": x_recon_xyz,
                            "seq": seq,
                            "esmfold_outputs": outputs,
                            "esmfold_pred_coords": pred_coords,
                            "folded_structure_metrics": trial_folded_structure_metrics,
                        }
                    )

                # Select best trial based on average TM-score
                best_trial = max(best_trial_results, key=lambda x: x["avg_tm_score"])
                logger.info(
                    f"Selected trial {best_trial['trial'] + 1} with average TM-score: {best_trial['avg_tm_score']:.3f}"
                )

                # Use best trial results
                generate_sample = best_trial["generate_sample"]
                x_recon_xyz = best_trial["x_recon_xyz"]
                seq = best_trial["seq"]

                # Calculate percent identity for inverse folding (compare generated sequence with original)
                # For inverse folding, we need to get the original sequence from the input structure
                original_sequences = []
                for i, valid_idx in enumerate(filtered_valid_indices):
                    structure_path = batch_paths[valid_idx]
                    if structure_path.endswith(".pt"):
                        # For .pt files, the sequence should be in the loaded data
                        structure_data = torch.load(structure_path, map_location="cpu")
                        if "sequence" in structure_data:
                            orig_seq = structure_data["sequence"]
                            if orig_seq.dim() > 1:
                                orig_seq = orig_seq.squeeze()
                            original_sequences.append(orig_seq)
                        else:
                            raise ValueError(f"No sequence found for structure: {structure_path}")
                    else:
                        # For PDB/CIF files, we need to extract sequence from the loaded structure
                        # This is already done in the structure_transform, so we can get it from batch_data
                        if i < len(batch_data) and "sequence" in batch_data[i]:
                            orig_seq = batch_data[i]["sequence"]
                            if orig_seq.dim() > 1:
                                orig_seq = orig_seq.squeeze()
                            original_sequences.append(orig_seq)
                        else:
                            raise ValueError(f"No sequence found for structure: {structure_path}")

                # Calculate percent identity for this batch
                if original_sequences:
                    batch_percent_identities = []

                    for i, (orig_seq, gen_seq) in enumerate(zip(original_sequences, seq)):
                        # Get the actual length of the original sequence (excluding padding)
                        orig_len = len(orig_seq)
                        gen_len = len(gen_seq)

                        # Use the minimum length to avoid dimension mismatches
                        min_len = min(orig_len, gen_len)

                        if min_len > 0:
                            # Truncate both sequences to the same length and ensure they're on the same device
                            orig_seq_truncated = orig_seq[:min_len].to(device)
                            gen_seq_truncated = gen_seq[:min_len].to(device)

                            # Calculate percent identity for this single sequence
                            percent_identity = calculate_percent_identity(
                                orig_seq_truncated.unsqueeze(0), gen_seq_truncated.unsqueeze(0)
                            )
                            batch_percent_identities.append(percent_identity.item())
                        else:
                            # If sequences are empty, set percent identity to 0
                            batch_percent_identities.append(0.0)

                    all_percent_identities.extend(batch_percent_identities)

                # Compute pocket AAR if ligand data is available
                if ligand_file_map and original_sequences:
                    batch_aar_overall = []
                    batch_aar_pocket = []
                    batch_aar_nonpocket = []
                    batch_n_pocket = []

                    for i, (orig_seq, gen_seq) in enumerate(zip(original_sequences, seq)):
                        orig_len = len(orig_seq)
                        gen_len = len(gen_seq)
                        min_len = min(orig_len, gen_len)

                        if min_len > 0:
                            orig_seq_dev = orig_seq[:min_len].to(device)
                            gen_seq_dev = gen_seq[:min_len].to(device)

                            # Overall AAR (0-1 scale)
                            aar_overall = _compute_aar(gen_seq_dev, orig_seq_dev)
                            batch_aar_overall.append(aar_overall)

                            # Pocket / non-pocket AAR
                            lig_coords_i = batch_ligand_coords[i] if i < len(batch_ligand_coords) else None
                            if lig_coords_i is not None:
                                # Get protein coords (unpadded)
                                orig_coords_i = filtered_batch_data[i]["coords_res"][:min_len].to(device)
                                pocket_mask_i = _compute_pocket_mask(
                                    orig_coords_i, lig_coords_i, threshold=pocket_distance_threshold
                                )
                                non_pocket_mask_i = ~pocket_mask_i

                                n_pocket = int(pocket_mask_i.sum().item())
                                batch_n_pocket.append(n_pocket)

                                aar_pocket = _compute_aar(gen_seq_dev, orig_seq_dev, pocket_mask_i)
                                aar_nonpocket = _compute_aar(gen_seq_dev, orig_seq_dev, non_pocket_mask_i)
                                batch_aar_pocket.append(aar_pocket)
                                batch_aar_nonpocket.append(aar_nonpocket)

                                logger.info(
                                    f"  AAR overall: {aar_overall:.3f}, "
                                    f"pocket: {aar_pocket:.3f} ({n_pocket} residues), "
                                    f"non-pocket: {aar_nonpocket:.3f}"
                                )
                            else:
                                batch_aar_pocket.append(float("nan"))
                                batch_aar_nonpocket.append(float("nan"))
                                batch_n_pocket.append(0)
                        else:
                            batch_aar_overall.append(float("nan"))
                            batch_aar_pocket.append(float("nan"))
                            batch_aar_nonpocket.append(float("nan"))
                            batch_n_pocket.append(0)

                    all_aar_overall.extend(batch_aar_overall)
                    all_aar_pocket.extend(batch_aar_pocket)
                    all_aar_nonpocket.extend(batch_aar_nonpocket)
                    all_n_pocket_residues.extend(batch_n_pocket)

                # Write sequences to CSV
                if csv_writer is not None:
                    # Convert generated sequences to strings
                    generated_sequence_strs = []
                    structure_token_strs = []
                    for i in range(B):
                        seq_i = seq[i, mask[i] == 1]
                        sequence_str = "".join([restype_order_with_x_inv[j.item()] for j in seq_i])
                        generated_sequence_strs.append(sequence_str)

                        # Convert structure tokens to comma-separated string
                        tokens_i = structure_tokens[i, mask[i] == 1]
                        tokens_str = ",".join([str(t.item()) for t in tokens_i])
                        structure_token_strs.append(tokens_str)

                    # Convert original sequences to strings
                    original_sequence_strs = []
                    for orig_seq in original_sequences:
                        orig_seq_str = "".join([restype_order_with_x_inv[j.item()] for j in orig_seq])
                        original_sequence_strs.append(orig_seq_str)

                    # Determine run_id based on whether we're generating multiple designs
                    if n_designs_per_structure > 1:
                        run_id = f"inverse_folding_batch_{batch_idx:03d}_design_{design_idx:02d}"
                    else:
                        run_id = f"inverse_folding_batch_{batch_idx:03d}"

                    # Write to sequences CSV
                    csv_writer.write_sequences(
                        sequences=generated_sequence_strs,
                        original_sequences=original_sequence_strs,
                        run_id=run_id,
                        input_structure=[Path(batch_paths[i]).stem for i in filtered_valid_indices],
                        trial_number=best_trial["trial"] + 1,
                        percent_identities=batch_percent_identities,
                        latent_generator_tokens=structure_token_strs,
                    )

                # Save results
                logger.info(f"Saving inverse folding results for batch {batch_idx + 1}, design {design_idx + 1}...")
                for i, valid_idx in enumerate(filtered_valid_indices):
                    original_path = batch_paths[valid_idx]
                    original_name = Path(original_path).stem
                    x_recon_xyz_i_masked = x_recon_xyz[i, mask[i] == 1]
                    seq_i_masked = seq[i, mask[i] == 1]

                    # Save generated structure with design index
                    if n_designs_per_structure > 1:
                        filename = output_dir / f"inverse_folding_{original_name}_design_{design_idx:02d}_generated.pdb"
                    else:
                        filename = output_dir / f"inverse_folding_{original_name}_generated.pdb"
                    writepdb(str(filename), x_recon_xyz_i_masked, seq_i_masked)
                    logger.info(f"Saved: {filename}")

                # Optional ESMFold validation - reuse results from trial selection
                if plm_fold is not None:
                    logger.info(f"Validating batch {batch_idx + 1} with ESMFold (reusing trial results)...")

                    # Check if ESMFold was skipped due to sequence length
                    if best_trial["folded_structure_metrics"].get("_skipped", False):
                        logger.info(
                            f"Skipping ESMFold validation (reason: {best_trial['folded_structure_metrics'].get('_reason', 'unknown')})"
                        )
                        batch_metrics = best_trial["folded_structure_metrics"]
                    # Reuse ESMFold results from the best trial
                    elif (
                        best_trial["folded_structure_metrics"] is not None
                        and best_trial["esmfold_pred_coords"] is not None
                    ):
                        # Use stored metrics without recalculation
                        folded_structure_metrics = best_trial["folded_structure_metrics"]
                        pred_coords = best_trial["esmfold_pred_coords"]

                        # Log metrics
                        logger.info("ESMFold validation metrics:")
                        for key, value in folded_structure_metrics.items():
                            # Skip internal fields that store chain group results
                            if key.startswith("_all_") or key.startswith("_primary_"):
                                continue
                            # Format numeric values
                            if isinstance(value, (int, float)):
                                logger.info(f"  {key}: {value:.4f}")
                            else:
                                logger.info(f"  {key}: {value}")

                        # Save folded structures
                        for i in range(seq.shape[0]):
                            original_name = Path(batch_paths[filtered_valid_indices[i]]).stem

                            # Check if using chain groups (pred_coords is filtered)
                            if "_primary_chain_group" in folded_structure_metrics:
                                # pred_coords only contains the filtered chains
                                # No need to mask - already filtered
                                pred_coords_i = pred_coords[i]

                                # Get the filtered sequence (from filtered chains)
                                chains_i = filtered_batch_data[i]["chains"].to(device)[mask[i] == 1]
                                seq_i_full = seq[i, mask[i] == 1]

                                # Create mask for primary chain group
                                primary_chain_group = folded_structure_metrics["_primary_chain_group"]
                                chain_mask = torch.zeros_like(chains_i, dtype=torch.bool)
                                for chain_id in primary_chain_group:
                                    chain_mask |= chains_i == chain_id

                                seq_i_filtered = seq_i_full[chain_mask]

                                if n_designs_per_structure > 1:
                                    filename = (
                                        output_dir
                                        / f"inverse_folding_{original_name}_design_{design_idx:02d}_esmfold_chains_{'_'.join(map(str, primary_chain_group))}.pdb"
                                    )
                                else:
                                    filename = (
                                        output_dir
                                        / f"inverse_folding_{original_name}_esmfold_chains_{'_'.join(map(str, primary_chain_group))}.pdb"
                                    )
                                writepdb(str(filename), pred_coords_i, seq_i_filtered)
                                logger.info(f"Saved ESMFold structure (chains {primary_chain_group}): {filename}")
                            else:
                                # Using all chains - normal masking
                                pred_coords_i_masked = pred_coords[i, mask[i] == 1]
                                seq_i_masked = seq[i, mask[i] == 1]
                                if n_designs_per_structure > 1:
                                    filename = (
                                        output_dir
                                        / f"inverse_folding_{original_name}_design_{design_idx:02d}_esmfold.pdb"
                                    )
                                else:
                                    filename = output_dir / f"inverse_folding_{original_name}_esmfold.pdb"
                                writepdb(str(filename), pred_coords_i_masked, seq_i_masked)
                                logger.info(f"Saved ESMFold structure: {filename}")

                        batch_metrics = folded_structure_metrics
                    else:
                        # Fallback to original validation if no stored results
                        logger.warning("No stored ESMFold results, running validation...")
                        batch_metrics = _validate_with_esmfold(
                            seq,
                            x_recon_xyz,
                            plm_fold,
                            device,
                            output_dir,
                            f"inverse_folding_batch{batch_idx:03d}",
                            original_paths=[batch_paths[i] for i in filtered_valid_indices],
                            mask=mask,
                            max_length=max_length,
                        )

                    # Collect metrics for aggregate statistics (skip if ESMFold was skipped)
                    if batch_metrics and not batch_metrics.get("_skipped", False):
                        all_plddt_scores.append(batch_metrics["_plddt"])
                        all_predicted_aligned_errors.append(batch_metrics["_predicted_aligned_error"])
                        all_tm_scores.append(batch_metrics["_tm_score"])
                        all_rmsd_scores.append(batch_metrics["_rmsd"])
                        avg_percent_identity = sum(batch_percent_identities) / len(batch_percent_identities)

                        # Write batch metrics to CSV
                        if csv_writer is not None:
                            if n_designs_per_structure > 1:
                                run_id = f"inverse_folding_batch_{batch_idx:03d}_design_{design_idx:02d}"
                            else:
                                run_id = f"inverse_folding_batch_{batch_idx:03d}"
                            csv_writer.write_batch_metrics(
                                batch_metrics,
                                run_id,
                                percent_identity=avg_percent_identity,
                                sequence_length=max_length,
                                input_file=f"batch_{batch_idx:03d}",
                            )

    # Calculate and report aggregate statistics
    logger.info("=" * 80)
    logger.info("INVERSE FOLDING AGGREGATE STATISTICS")
    logger.info("=" * 80)

    if all_percent_identities:
        avg_percent_identity = sum(all_percent_identities) / len(all_percent_identities)
        logger.info(f"Average Percent Identity: {avg_percent_identity:.2f}% (n={len(all_percent_identities)})")
    else:
        logger.warning("No percent identity data collected")

    if all_plddt_scores:
        avg_plddt = sum(all_plddt_scores) / len(all_plddt_scores)
        logger.info(f"Average pLDDT: {avg_plddt:.2f} (n={len(all_plddt_scores)})")
    else:
        logger.warning("No pLDDT data collected")

    if all_predicted_aligned_errors:
        avg_pae = sum(all_predicted_aligned_errors) / len(all_predicted_aligned_errors)
        logger.info(f"Average Predicted Aligned Error: {avg_pae:.2f} (n={len(all_predicted_aligned_errors)})")
    else:
        logger.warning("No Predicted Aligned Error data collected")

    if all_tm_scores:
        avg_tm_score = sum(all_tm_scores) / len(all_tm_scores)
        logger.info(f"Average TM-Score: {avg_tm_score:.3f} (n={len(all_tm_scores)})")
    else:
        logger.warning("No TM-Score data collected")

    # Calculate RMSD pass rate (< 2.0Å threshold)
    rmsd_threshold = 2.0
    rmsd_pass_rates = {}

    if all_rmsd_scores:
        avg_rmsd = sum(all_rmsd_scores) / len(all_rmsd_scores)
        logger.info(f"Average RMSD: {avg_rmsd:.2f} Å (n={len(all_rmsd_scores)})")

        pass_count = sum(1 for rmsd in all_rmsd_scores if rmsd < rmsd_threshold)
        total_count = len(all_rmsd_scores)
        pass_rate = (pass_count / total_count * 100) if total_count > 0 else 0.0
        rmsd_pass_rates["rmsd"] = (pass_count, total_count, pass_rate)
        logger.info(f"RMSD Pass Rate (< {rmsd_threshold:.1f}Å): {pass_count}/{total_count} ({pass_rate:.1f}%)")
    else:
        logger.warning("No RMSD data collected")

    logger.info("=" * 80)

    # Report pocket AAR statistics if available
    if all_aar_overall:
        import math

        valid_aar_overall = [x for x in all_aar_overall if not math.isnan(x)]
        valid_aar_pocket = [x for x in all_aar_pocket if not math.isnan(x)]
        valid_aar_nonpocket = [x for x in all_aar_nonpocket if not math.isnan(x)]
        valid_n_pocket = [x for x in all_n_pocket_residues if x > 0]

        logger.info("")
        logger.info("--- Pocket Amino Acid Recovery (AAR) ---")
        if valid_aar_overall:
            avg_aar = sum(valid_aar_overall) / len(valid_aar_overall)
            logger.info(f"  Overall AAR: {avg_aar:.4f} ({avg_aar * 100:.2f}%) (n={len(valid_aar_overall)})")
        if valid_aar_pocket:
            avg_pocket = sum(valid_aar_pocket) / len(valid_aar_pocket)
            logger.info(f"  Pocket AAR:  {avg_pocket:.4f} ({avg_pocket * 100:.2f}%) (n={len(valid_aar_pocket)})")
        if valid_aar_nonpocket:
            avg_nonpocket = sum(valid_aar_nonpocket) / len(valid_aar_nonpocket)
            logger.info(
                f"  Non-pocket AAR: {avg_nonpocket:.4f} ({avg_nonpocket * 100:.2f}%) (n={len(valid_aar_nonpocket)})"
            )
        if valid_aar_pocket and valid_aar_nonpocket:
            delta = avg_pocket - avg_nonpocket
            logger.info(f"  Delta (pocket - non-pocket): {delta:+.4f} ({delta * 100:+.2f}%)")
        if valid_n_pocket:
            avg_pocket_size = sum(valid_n_pocket) / len(valid_n_pocket)
            logger.info(f"  Average pocket size: {avg_pocket_size:.1f} residues")
        logger.info("=" * 80)

    # Write aggregate statistics to CSV
    if csv_writer is not None:
        logger.info("Writing inverse folding aggregate statistics to CSV...")

        # Collect all metric values
        metric_lists = {
            "percent_identity": all_percent_identities,
            "plddt": all_plddt_scores,
            "predicted_aligned_error": all_predicted_aligned_errors,
            "tm_score": all_tm_scores,
            "rmsd": all_rmsd_scores,
        }

        # Add pocket AAR metrics if available
        if all_aar_overall:
            metric_lists["aar_overall"] = all_aar_overall
            metric_lists["aar_pocket"] = all_aar_pocket
            metric_lists["aar_nonpocket"] = all_aar_nonpocket
            metric_lists["n_pocket_residues"] = [float(x) for x in all_n_pocket_residues]

        # Calculate aggregate statistics
        aggregate_stats = calculate_aggregate_stats(metric_lists)

        # Write aggregate statistics to CSV
        csv_writer.write_aggregate_stats(aggregate_stats)

        # Write pass rate statistics to CSV if available
        if rmsd_pass_rates:
            csv_writer.write_pass_rates(rmsd_pass_rates, threshold=rmsd_threshold)

    # Create plots from CSV data if plotter is available
    if plotter is not None and csv_writer is not None:
        logger.info("Creating box and whisker plots from CSV data...")
        try:
            plotter.create_box_plots_from_csv(csv_writer.csv_path)
            logger.info("✓ Box plots created successfully")
        except Exception as e:
            logger.error(f"Error creating box plots: {e}")

        # Create correlation plots (only for unconditional mode)
        try:
            plotter.create_correlation_plots_from_csv(csv_writer.csv_path)
        except Exception as e:
            logger.debug(f"Correlation plots not applicable: {e}")
