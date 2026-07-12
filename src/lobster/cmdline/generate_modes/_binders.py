"""LeFlur binder-design generation mode.

Generates novel binder proteins for target protein structures via inpainting.
The target chain stays fixed; a binder chain of configurable length is
initialized in 3D space near the target (or near an epitope, when provided)
and then completed via the LeFlur sample loop with inpainting masks that
freeze the target tokens.

Called from :func:`lobster.cmdline.generate.generate` when
``cfg.generation.mode == "binder_design"``.
"""

from __future__ import annotations

import glob
from pathlib import Path

import torch
from loguru import logger
from omegaconf import DictConfig, ListConfig

from lobster.metrics import predict_structure_with_esmfold
from lobster.model.latent_generator.io import load_pdb, writepdb
from lobster.model.latent_generator.utils import apply_random_se3_batched
from lobster.model.latent_generator.utils.residue_constants import (
    convert_lobster_aa_tokenization_to_standard_aa,
    restype_order_with_x_inv,
)
from lobster.model.leflur.binder_utils import (
    create_binder_inpainting_masks,
    get_next_chain_index,
    get_target_chain_info,
    initialize_binder_at_origin,
)
from lobster.transforms._structure_transforms import (
    AminoAcidTokenizerTransform,
    StructureBackboneTransform,
)


def _generate_binders(
    model, cfg: DictConfig, device: torch.device, output_dir: Path, plm_fold=None, csv_writer=None, plotter=None
) -> None:
    """Generate binders for target protein structures."""
    logger.info("Starting binder design generation...")

    # Get input structure paths
    input_structures = cfg.generation.input_structures
    if not input_structures:
        raise ValueError("input_structures must be provided for binder_design mode")

    # Handle different input formats (same as inpainting mode)
    structure_paths = []
    if isinstance(input_structures, str):
        if "*" in input_structures or "?" in input_structures:
            # Glob pattern
            structure_paths = glob.glob(input_structures)
        else:
            # Single file or directory
            path = Path(input_structures)
            if path.is_file():
                structure_paths = [str(path)]
            elif path.is_dir():
                # Find all structure files in directory
                structure_paths = list(glob.glob(str(path / "*.pdb")))
                structure_paths.extend(glob.glob(str(path / "*.cif")))
            else:
                raise ValueError(f"Input path does not exist: {input_structures}")
    elif isinstance(input_structures, (list, tuple, ListConfig)):
        # List of paths
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

    logger.info(f"Found {len(structure_paths)} structure(s) to process")

    # Get configuration parameters
    gen_cfg = cfg.generation
    target_chain = gen_cfg.get("target_chain")
    binder_length = gen_cfg.get("binder_length")
    epitope_indices = gen_cfg.get("epitope_indices", None)
    nsteps = gen_cfg.get("nsteps", 200)
    n_trials = gen_cfg.get("n_trials", 1)
    n_designs_per_structure = gen_cfg.get("n_designs_per_structure", 1)

    if not target_chain:
        raise ValueError("target_chain must be specified for binder_design mode")
    if not binder_length:
        raise ValueError("binder_length must be specified for binder_design mode")

    logger.info(f"Target chain: {target_chain}")
    logger.info(f"Binder length: {binder_length}")
    if epitope_indices:
        logger.info(f"Epitope indices: {epitope_indices}")
    logger.info(f"Generation steps: {nsteps}")
    logger.info(f"Designs per structure: {n_designs_per_structure}")

    # Initialize transforms
    structure_transform = StructureBackboneTransform(max_length=gen_cfg.get("max_length", 512))
    tokenizer_transform = AminoAcidTokenizerTransform(max_length=gen_cfg.get("max_length", 512))

    # Process each structure
    with torch.no_grad():
        for structure_idx, structure_path in enumerate(structure_paths):
            logger.info(f"\n{'=' * 70}")
            logger.info(f"Processing structure {structure_idx + 1}/{len(structure_paths)}")
            logger.info(f"Input: {structure_path}")
            logger.info(f"{'=' * 70}")

            # Load target structure
            logger.info(f"Loading target structure from {structure_path}")
            target_data = load_pdb(structure_path, add_batch_dim=False)

            if target_data is None:
                logger.warning(f"Failed to load structure from {structure_path}, skipping")
                continue

            # Apply transforms
            target_data = structure_transform(target_data)

            # Check minimum length
            if target_data["coords_res"].shape[0] < 30:
                logger.warning(f"Structure too short ({target_data['coords_res'].shape[0]} residues), skipping")
                continue

            # Identify target chain
            try:
                target_chain_idx, target_start, target_end = get_target_chain_info(target_data, target_chain)
                logger.info(f"Target chain '{target_chain}' found:")
                logger.info(f"  Chain index: {target_chain_idx}")
                logger.info(f"  Residue range: {target_start}-{target_end}")
                logger.info(f"  Length: {target_end - target_start} residues")
            except ValueError as e:
                logger.error(str(e))
                continue

            # Extract only target chain from structure
            # Note: StructureBackboneTransform renames 'chains_ids' to 'chains'
            chains_key = "chains" if "chains" in target_data else "chains_ids"
            target_chain_mask = target_data[chains_key] == target_chain_idx
            target_data_filtered = {
                "coords_res": target_data["coords_res"][target_chain_mask],
                "sequence": target_data["sequence"][target_chain_mask],
                chains_key: target_data[chains_key][target_chain_mask],
                "real_chains": target_data["real_chains"][target_chain_mask],
                "indices": target_data["indices"][target_chain_mask],
                "mask": target_data["mask"][target_chain_mask],
            }

            # Initialize binder position
            if epitope_indices:
                logger.info(f"Initializing binder with length {binder_length} near epitope")
                logger.info(f"  Epitope residue indices: {epitope_indices}")
                logger.info("  Ball center: 5Å from epitope, radius: 12Å, min target distance: 5Å")
            else:
                logger.info(f"Initializing binder with length {binder_length} around target center of mass")
                logger.info("  Ball radius: 12Å, min target distance: 5Å")

            binder_data = initialize_binder_at_origin(
                binder_length,
                device="cpu",
                target_coords=target_data_filtered["coords_res"],
                epitope_indices=epitope_indices,
            )

            # Get next chain index for binder
            binder_chain_idx = get_next_chain_index(target_data_filtered)
            logger.info(f"Binder will be assigned chain index: {binder_chain_idx}")

            # Create composite structure (target + binder)
            logger.info("Creating composite structure (target + binder)")

            L_target = target_data_filtered["coords_res"].shape[0]
            L_binder = binder_data["coords_res"].shape[0]
            L_total = L_target + L_binder

            # Check max length
            max_length = gen_cfg.get("max_length", 512)
            if L_total > max_length:
                logger.warning(
                    f"Total length {L_total} (target: {L_target}, binder: {L_binder}) "
                    f"exceeds max_length {max_length}. Skipping structure."
                )
                continue

            # Concatenate all tensors
            coords_res_combined = torch.cat([target_data_filtered["coords_res"], binder_data["coords_res"]], dim=0)

            sequence_combined = torch.cat([target_data_filtered["sequence"], binder_data["sequence"]], dim=0)

            mask_combined = torch.cat([target_data_filtered["mask"], binder_data["mask"]], dim=0)

            # Create chain IDs for binder
            binder_chain_ids = torch.full((L_binder,), binder_chain_idx, dtype=target_data_filtered[chains_key].dtype)
            chains_ids_combined = torch.cat([target_data_filtered[chains_key], binder_chain_ids], dim=0)

            # Create indices for binder
            binder_indices = torch.arange(
                binder_chain_idx, binder_chain_idx + L_binder, dtype=target_data_filtered["indices"].dtype
            )
            indices_combined = torch.cat([target_data_filtered["indices"], binder_indices], dim=0)

            logger.info("Composite structure created:")
            logger.info(f"  Total length: {L_total} ({L_target} target + {L_binder} binder)")
            logger.info(f"  Target chain index: {target_chain_idx}")
            logger.info(f"  Binder chain index: {binder_chain_idx}")

            # Save initial structure (before generation)
            structure_name = Path(structure_path).stem
            initial_structure_path = output_dir / f"{structure_name}_initial_structure.pdb"
            writepdb(str(initial_structure_path), coords_res_combined, sequence_combined)
            logger.info(f"Saved initial structure: {initial_structure_path}")

            # Add batch dimension and move to device
            coords_res = coords_res_combined.unsqueeze(0).to(device)
            sequence = sequence_combined.unsqueeze(0).to(device)
            mask = mask_combined.unsqueeze(0).to(device)
            chains_ids = chains_ids_combined.unsqueeze(0).to(device)
            indices = indices_combined.unsqueeze(0).to(device)

            # Apply tokenizer to sequence
            tokenized_data = tokenizer_transform({"sequence": sequence.squeeze(0).cpu()})
            sequence_tokenized = tokenized_data["sequence"].unsqueeze(0).to(device)

            # Create inpainting masks
            # Note: First binder residue is kept fixed to preserve chain break token
            logger.info("Creating inpainting masks (target=fixed, first binder token=fixed, rest of binder=generate)")

            mask_sequence, mask_structure = create_binder_inpainting_masks(
                chains_ids, target_chain_idx, binder_chain_idx, device
            )

            # Verify masks
            num_fixed = (mask_sequence == 0).sum().item()
            num_generate = (mask_sequence == 1).sum().item()
            logger.info(f"  Fixed residues: {num_fixed} (target + 1 binder chain-break token)")
            logger.info(f"  Generate residues: {num_generate} (binder minus first token)")

            # Optional epitope conditioning: feed the hotspot residues (epitope_indices are
            # antigen-local = target-region positions, since the target chain is concatenated
            # first) into the model's conditioning channel, and pass per-residue chain ids, so
            # a complex/epitope-trained checkpoint actually uses its training. Mirrors
            # _dimer_forward_folding.py. When off (default), both stay None => unchanged behavior.
            use_epitope_conditioning = gen_cfg.get("use_epitope_conditioning", False)
            chain_ids_emb = None
            cond_tensor = None
            if use_epitope_conditioning:
                chain_ids_emb = torch.zeros_like(chains_ids)
                chain_ids_emb[chains_ids == target_chain_idx] = 1
                chain_ids_emb[chains_ids == binder_chain_idx] = 2
                cond_tensor = torch.zeros((1, L_total, 1), device=device)
                if epitope_indices:
                    hot = torch.tensor(
                        [i for i in epitope_indices if 0 <= i < L_target], device=device, dtype=torch.long
                    )
                    if hot.numel() > 0:
                        cond_tensor[0, hot, 0] = 1.0
                logger.info(
                    f"  Epitope conditioning ON: {int((cond_tensor > 0).sum())} hotspot residues "
                    f"(of {len(epitope_indices) if epitope_indices else 0} given); "
                    f"chain_ids remapped target->1, binder->2"
                )

            # TEMPLATE-TARGET mode: condition on the target via its per-chain template
            # (frame-randomized, isolated encode -> fold only, no absolute pose) + hotspot,
            # INSTEAD of pinning the target with fixed structure tokens. The target SEQUENCE
            # stays fixed (known), but its STRUCTURE is generated (mask_structure target->1)
            # guided by the template channel -- i.e. the target is forward-folded from its
            # template while the binder is designed. Requires a template-trained checkpoint.
            template_arg = None
            if gen_cfg.get("template_target", False) and getattr(model, "no_template_idx", None) is not None:
                template_tokens = torch.full((1, L_total), model.no_template_idx, dtype=torch.long, device=device)
                tmask = mask * (chains_ids == target_chain_idx).float()
                if tmask.sum() > 0:
                    coords_c = apply_random_se3_batched(coords_res.clone())  # random frame -> no pose leak
                    xq_c, _, _ = model.encode_structure(coords_c, tmask, indices)
                    tok_c = xq_c.argmax(dim=-1)
                    sel = chains_ids == target_chain_idx
                    template_tokens[sel] = tok_c[sel]
                template_arg = template_tokens
                mask_structure = mask_structure.clone()
                mask_structure[chains_ids == target_chain_idx] = 1.0  # generate target struct from template
                logger.info(
                    "  TEMPLATE-TARGET mode ON: target structure generated from per-chain template "
                    "(seq fixed); target structure tokens NOT pinned"
                )

            # Generate binder designs
            for design_idx in range(n_designs_per_structure):
                if n_designs_per_structure > 1:
                    logger.info(f"\n--- Design {design_idx + 1}/{n_designs_per_structure} ---")

                best_result = None

                for trial in range(n_trials):
                    if n_trials > 1:
                        logger.info(f"Trial {trial + 1}/{n_trials}")

                    # Generate with inpainting
                    generate_sample = model.generate_sample(
                        length=L_total,
                        num_samples=1,
                        nsteps=nsteps,
                        temperature_seq=gen_cfg.get("temperature_seq", 0.5),
                        temperature_struc=gen_cfg.get("temperature_struc", 1.0),
                        stochasticity_seq=gen_cfg.get("stochasticity_seq", 20),
                        stochasticity_struc=gen_cfg.get("stochasticity_struc", 20),
                        inpainting=True,
                        input_structure_coords=coords_res,
                        input_sequence_tokens=sequence_tokenized,
                        input_mask=mask,
                        input_indices=indices,
                        inpainting_mask_sequence=mask_sequence,
                        inpainting_mask_structure=mask_structure,
                        asynchronous_sampling=gen_cfg.get("asynchronous_sampling", False),
                        chain_ids=chain_ids_emb,
                        conditioning_tensor_override=cond_tensor,
                        encode_target_only=gen_cfg.get("encode_target_only", False),
                        template_structure_tokens=template_arg,
                        cfg_weight=float(gen_cfg.get("cfg_weight", 1.0)),
                    )

                    # Decode structures
                    decoded_x = model.decode_structure(generate_sample, mask)

                    # Extract coordinates
                    x_recon_xyz = None
                    for decoder_name in decoded_x:
                        if "vit_decoder" == decoder_name:
                            x_recon_xyz = decoded_x[decoder_name]
                            break

                    if x_recon_xyz is None:
                        logger.error("No vit_decoder output found, skipping this trial")
                        continue

                    # Extract coordinates (B, L, 3, 3) - N, CA, C atoms
                    gen_coords = x_recon_xyz[:, :, [0, 1, 2], :]

                    # Extract sequences
                    if generate_sample["sequence_logits"].shape[-1] == 33:
                        gen_sequence = convert_lobster_aa_tokenization_to_standard_aa(
                            generate_sample["sequence_logits"], device=device
                        )
                    else:
                        gen_sequence = generate_sample["sequence_logits"].argmax(dim=-1)
                        gen_sequence[gen_sequence > 21] = 20

                    # Store result
                    result = {
                        "coords": gen_coords,
                        "sequence": gen_sequence,
                        "mask": mask,
                        "chains_ids": chains_ids,
                        "indices": indices,
                    }

                    # For now, just keep the first/only trial
                    best_result = result
                    if n_trials == 1:
                        break

                # Save outputs
                structure_name = Path(structure_path).stem
                prefix = f"{structure_name}_design{design_idx:03d}"

                gen_coords = best_result["coords"]
                gen_sequence = best_result["sequence"]

                # Save complete complex (binder + target on separate chains so
                # viewers/downstream tools treat them as distinct entities).
                complex_path = output_dir / f"{prefix}_complex.pdb"
                writepdb(str(complex_path), gen_coords[0], gen_sequence[0], chains=best_result["chains_ids"][0])
                logger.info(f"Saved complex: {complex_path}")

                # Save binder alone
                binder_mask = chains_ids[0] == binder_chain_idx
                binder_coords = gen_coords[0, binder_mask]
                binder_sequence = gen_sequence[0, binder_mask]
                binder_path = output_dir / f"{prefix}_binder.pdb"
                writepdb(str(binder_path), binder_coords, binder_sequence)
                logger.info(f"Saved binder: {binder_path}")

                # Save target alone (for reference)
                target_mask = chains_ids[0] == target_chain_idx
                target_coords = gen_coords[0, target_mask]
                target_sequence = gen_sequence[0, target_mask]
                target_path = output_dir / f"{prefix}_target.pdb"
                writepdb(str(target_path), target_coords, target_sequence)
                logger.info(f"Saved target: {target_path}")

                # Validate with ESMFold if enabled
                if gen_cfg.get("use_esmfold", False) and plm_fold is not None:
                    logger.info("Validating with ESMFold...")

                    # Validate the complex (target + binder together)
                    try:
                        # Get chain groups for validation
                        esmfold_chain_groups = gen_cfg.get("esmfold_chain_groups", None)
                        if esmfold_chain_groups is None:
                            # Default: validate target + binder together
                            esmfold_chain_groups = [[target_chain_idx, binder_chain_idx]]

                        # Call ESMFold validation
                        result = predict_structure_with_esmfold(
                            plm_fold=plm_fold,
                            seq_i=gen_sequence[0],
                            chains_i=chains_ids[0],
                            orig_coords=coords_res[0],  # Original composite structure
                            gen_coords=gen_coords[0],  # Generated composite structure
                            mask_i=mask[0],
                            cfg=cfg,
                            device=device,
                            restype_order_inv=restype_order_with_x_inv,
                            inpainting_mask_seq_i=mask_sequence[0],
                            inpainting_mask_struc_i=mask_structure[0],
                            chain_group=esmfold_chain_groups[0] if esmfold_chain_groups else None,
                        )

                        # Skip if sequence too long for ESMFold
                        if result is None:
                            logger.warning("Structure exceeds ESMFold max length, skipping ESMFold validation")
                            continue

                        logger.info("ESMFold validation metrics:")
                        if "folded_structure_metrics" in result:
                            for key, value in result["folded_structure_metrics"].items():
                                if isinstance(value, (int, float)):
                                    logger.info(f"  {key}: {value:.4f}")
                                else:
                                    logger.info(f"  {key}: {value}")

                        # Save ESMFold predicted structure
                        if "pred_coords" in result and result["pred_coords"] is not None:
                            pred_coords = result["pred_coords"]
                            # pred_coords shape is (1, L, 3, 3) or (L, 3, 3)
                            if pred_coords.dim() == 4:
                                pred_coords = pred_coords.squeeze(0)  # Remove batch dim

                            # Save ESMFold predicted complex
                            esmfold_path = output_dir / f"{prefix}_esmfold.pdb"
                            writepdb(str(esmfold_path), pred_coords, gen_sequence[0])
                            logger.info(f"Saved ESMFold prediction: {esmfold_path}")

                            # Save ESMFold predicted binder only
                            if pred_coords.shape[0] == gen_sequence[0].shape[0]:
                                esmfold_binder_coords = pred_coords[binder_mask]
                                esmfold_binder_path = output_dir / f"{prefix}_esmfold_binder.pdb"
                                writepdb(str(esmfold_binder_path), esmfold_binder_coords, binder_sequence)
                                logger.info(f"Saved ESMFold binder prediction: {esmfold_binder_path}")

                    except Exception as e:
                        logger.warning(f"ESMFold validation failed: {e}")

    logger.info("\nBinder design generation completed!")
