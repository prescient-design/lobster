"""LeFlur unconditional generation mode + self-reflection helpers.

De-novo protein generation: jointly samples sequence and Cα backbone tokens
from the LeFlur latent prior, decodes to atom coordinates via the ViT decoder,
and (optionally) validates each sample with ESMFold. Supports a self-reflection
loop where forward and inverse folding rejection passes are used to improve
sample quality.

Called from :func:`lobster.cmdline.generate.generate` when
``cfg.generation.mode == "unconditional"``.
"""

from __future__ import annotations

from pathlib import Path
import csv

from loguru import logger
from omegaconf import (
    DictConfig,
)
from tmtools import tm_align
import torch

from lobster.metrics import (
    get_folded_structure_metrics,
    calculate_percent_identity,
    calculate_aggregate_stats,
    align_and_compute_rmsd,
)
from lobster.metrics.cal_foldseek_clusters import calculate_diversity_for_generation
from lobster.model.latent_generator.io import writepdb
from lobster.model.latent_generator.utils.residue_constants import (
    convert_lobster_aa_tokenization_to_standard_aa,
    restype_order_with_x,
    restype_order_with_x_inv,
)
from lobster.transforms._structure_transforms import AminoAcidTokenizerTransform

from ._shared import _check_sequence_tokens, _get_inference_schedule_class, _validate_with_esmfold


def _save_failed_self_reflection_attempt(
    output_dir: Path,
    current_length: int,
    iteration: int,
    retry_count: int,
    failure_reason: str,
    initial_seq_str: str,
    initial_structure: torch.Tensor | None = None,
    mask_i: torch.Tensor | None = None,
    extra_metrics: dict | None = None,
) -> None:
    """Save an SR failed attempt's input sequence, structure (if available), and metadata.

    Creates ``<output_dir>/failed_self_reflection/`` containing one FASTA + PDB
    pair per failed attempt and a single ``failed_self_reflection.csv`` log.
    The post-hoc workflow can ESMFold the saved FASTA against the saved PDB to
    measure SR-QC vs ESMFold-QC concordance.
    """
    import csv as _csv

    fail_dir = output_dir / "failed_self_reflection"
    fail_dir.mkdir(parents=True, exist_ok=True)
    base = f"length_{current_length}_iter_{iteration:03d}_retry_{retry_count:03d}_{failure_reason}"

    # FASTA with the (initial) unconditional sequence
    fasta_path = fail_dir / f"{base}.fasta"
    with open(fasta_path, "w") as fh:
        fh.write(f">{base}\n{initial_seq_str}\n")

    # Initial backbone PDB (only available post-token-check)
    pdb_path = None
    if initial_structure is not None and mask_i is not None:
        pdb_path = fail_dir / f"{base}.pdb"
        masked_struct = initial_structure[mask_i == 1]
        seq_tokens = torch.tensor(
            [restype_order_with_x.get(c, 20) for c in initial_seq_str],
            device=initial_structure.device,
        )
        writepdb(str(pdb_path), masked_struct, seq_tokens)

    # Append a row to failed_self_reflection.csv (create header if first time)
    csv_path = fail_dir.parent / "failed_self_reflection.csv"
    extras = extra_metrics or {}
    fieldnames = [
        "length",
        "iteration",
        "retry_count",
        "failure_reason",
        "fasta_path",
        "pdb_path",
        "sequence_length",
        "sequence",
        "tm_score_unconditional_to_forward",
        "rmsd_unconditional_to_forward",
        "tm_score_forward_to_inverse",
        "rmsd_forward_to_inverse",
        "percent_identity_self_reflection",
    ]
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        row = {
            "length": current_length,
            "iteration": iteration,
            "retry_count": retry_count,
            "failure_reason": failure_reason,
            "fasta_path": str(fasta_path),
            "pdb_path": str(pdb_path) if pdb_path is not None else "",
            "sequence_length": len(initial_seq_str),
            "sequence": initial_seq_str,
        }
        for k in fieldnames[8:]:
            row[k] = extras.get(k, "")
        w.writerow(row)


def _execute_self_reflection_pipeline(
    model,
    cfg: DictConfig,
    device: torch.device,
    output_dir: Path,
    plm_fold,
    generate_sample: dict,
    mask: torch.Tensor,
    iteration: int,
    batch_size: int,
    current_length: int,
    save_structures: bool = False,
    retry_count: int = 0,
) -> dict[str, float] | None:
    """Execute self-reflection refinement pipeline to improve ESMFold metrics.

    Pipeline: Unconditional → Forward Folding → Inverse Folding

    This function refines unconditionally generated structure-sequence pairs through
    forward folding (sequence → structure) and inverse folding (structure → sequence)
    to improve consistency and ESMFold validation metrics.

    The refined outputs (sequence₂, structure₃) should produce better ESMFold metrics
    (higher pLDDT, higher TM-score, lower RMSD) compared to initial unconditional outputs.

    Args:
        model: genUME model
        cfg: Configuration
        device: torch device
        output_dir: Output directory
        plm_fold: ESMFold model (optional, but recommended for measuring improvement)
        generate_sample: Raw output dict from unconditional model.generate_sample()
        mask: Validity mask (B, L)
        iteration: Current iteration number
        batch_size: Batch size
        current_length: Sequence length
        save_structures: Whether to save structures
    Returns:
        Dictionary containing self-reflection metrics or None if pipeline failed
    """

    gen_cfg = cfg.generation
    sr_cfg = gen_cfg.get("self_reflection", {}) if hasattr(gen_cfg, "self_reflection") else {}
    save_failed = bool(sr_cfg.get("save_failed_attempts", False)) if hasattr(sr_cfg, "get") else False

    def _initial_seq_str(i: int = 0) -> str:
        seq_i = initial_seq[i, mask[i] == 1]
        return "".join([restype_order_with_x_inv[j.item()] for j in seq_i])

    try:
        logger.info("=" * 80)
        logger.info("SELF-REFLECTION REFINEMENT PIPELINE")
        logger.info("=" * 80)

        # Step 0: Extract sequences from unconditional generation (before decoding structure)
        logger.info("Step 0: Extracting unconditional sequences...")

        # Extract sequences
        if generate_sample["sequence_logits"].shape[-1] == 33:
            initial_seq = convert_lobster_aa_tokenization_to_standard_aa(
                generate_sample["sequence_logits"], device=device
            )
        else:
            initial_seq = generate_sample["sequence_logits"].argmax(dim=-1)
            initial_seq[initial_seq > 21] = 20

        logger.info(f"  Initial sequences shape: {initial_seq.shape}")

        # Quality control check for invalid tokens BEFORE decoding structure
        if hasattr(gen_cfg, "self_reflection") and hasattr(gen_cfg.self_reflection, "quality_control"):
            qc_config = gen_cfg.self_reflection.quality_control
            if qc_config.get("enable_sequence_token_check", True):  # Default True
                is_valid, error_msg = _check_sequence_tokens(initial_seq, mask, "unconditional generation")
                if not is_valid:
                    logger.warning(f"  Quality control FAILED: {error_msg}")
                    logger.warning("  Iteration will be retried (invalid sequence tokens)")
                    logger.warning("  Skipping structure decoding and forward/inverse folding")
                    return None
                else:
                    logger.info("  Quality control PASSED: All sequences contain valid amino acids")

        # Now decode structures (only if sequences passed QC)
        logger.info("  Decoding unconditional structures...")
        decoded_x = model.decode_structure(generate_sample, mask)

        # Extract coordinates
        initial_structure = None
        for decoder_name in decoded_x:
            if "vit_decoder" == decoder_name:
                initial_structure = decoded_x[decoder_name]
                break

        if initial_structure is None:
            logger.error("No structure decoder found in model output")
            return None

        logger.info(f"  Initial structures shape: {initial_structure.shape}")

        # Step 1: Prepare data for forward folding
        logger.info("Step 1: Preparing data for forward folding...")
        tokenizer_transform = AminoAcidTokenizerTransform(max_length=cfg.generation.get("max_length", 512))

        # Tokenize initial sequences
        padded_sequences = torch.zeros((batch_size, current_length), device=device, dtype=torch.long)
        for i in range(batch_size):
            seq_i = initial_seq[i, mask[i] == 1]
            tokenized_data = tokenizer_transform({"sequence": seq_i.cpu()})
            tokenized_seq = tokenized_data["sequence"]
            seq_len = min(len(tokenized_seq), current_length)
            padded_sequences[i, :seq_len] = tokenized_seq[:seq_len].to(device)

        # Create indices from mask
        indices = torch.arange(current_length, device=device).unsqueeze(0).expand(batch_size, -1)

        # Step 2: Forward folding
        logger.info("Step 2: Forward folding (sequence → structure refinement)...")
        forward_params = _get_self_reflection_params(cfg, "forward_folding")
        logger.info(f"  Forward folding parameters: {forward_params}")

        # Get inference schedule classes from config (use same as main generation)
        inference_schedule_seq = gen_cfg.get("inference_schedule_seq", "LogInferenceSchedule")
        inference_schedule_struc = gen_cfg.get("inference_schedule_struc", "LinearInferenceSchedule")
        if isinstance(inference_schedule_seq, str):
            inference_schedule_seq = _get_inference_schedule_class(inference_schedule_seq)
        if isinstance(inference_schedule_struc, str):
            inference_schedule_struc = _get_inference_schedule_class(inference_schedule_struc)

        forward_sample = model.generate_sample(
            length=current_length,
            num_samples=batch_size,
            forward_folding=True,
            input_sequence_tokens=padded_sequences,
            input_mask=mask,
            input_indices=indices,
            nsteps=forward_params["nsteps"],
            temperature_seq=forward_params["temperature_seq"],
            temperature_struc=forward_params["temperature_struc"],
            stochasticity_seq=forward_params["stochasticity_seq"],
            stochasticity_struc=forward_params["stochasticity_struc"],
            inference_schedule_seq=inference_schedule_seq,
            inference_schedule_struc=inference_schedule_struc,
            asynchronous_sampling=gen_cfg.get("asynchronous_sampling", False),
        )

        # Decode forward-folded structures
        forward_decoded_x = model.decode_structure(forward_sample, mask)
        forward_structure = None
        for decoder_name in forward_decoded_x:
            if "vit_decoder" == decoder_name:
                forward_structure = forward_decoded_x[decoder_name]
                break

        if forward_structure is None:
            logger.error("No structure decoder found in forward folding output")
            return None

        # Extract forward-folded sequences
        if forward_sample["sequence_logits"].shape[-1] == 33:
            forward_seq = convert_lobster_aa_tokenization_to_standard_aa(
                forward_sample["sequence_logits"], device=device
            )
        else:
            forward_seq = forward_sample["sequence_logits"].argmax(dim=-1)
            forward_seq[forward_seq > 21] = 20

        # Calculate TM-score and RMSD between unconditional and forward-folded
        tm_scores_uncond_to_forward = []
        rmsd_uncond_to_forward = []

        for i in range(batch_size):
            orig_coords = initial_structure[i, mask[i] == 1, :, :]
            forward_coords = forward_structure[i, mask[i] == 1, :, :]
            seq_i = initial_seq[i, mask[i] == 1]
            sequence_str = "".join([restype_order_with_x_inv[j.item()] for j in seq_i])

            # TM-align RMSD
            tm_out = tm_align(
                forward_coords[:, 1, :].cpu().numpy(),
                orig_coords[:, 1, :].detach().cpu().numpy(),
                sequence_str,
                sequence_str,
            )
            tm_scores_uncond_to_forward.append(tm_out.tm_norm_chain1)

            # Kabsch RMSD
            rmsd = align_and_compute_rmsd(
                coords1=forward_coords,
                coords2=orig_coords,
                mask=None,  # Use all positions
                return_aligned=False,
                device=device,
            )
            rmsd_uncond_to_forward.append(rmsd)

        avg_tm_uncond_to_forward = sum(tm_scores_uncond_to_forward) / len(tm_scores_uncond_to_forward)
        avg_rmsd_uncond_to_forward = sum(rmsd_uncond_to_forward) / len(rmsd_uncond_to_forward)

        logger.info(
            f"  Unconditional → Forward: TM-score={avg_tm_uncond_to_forward:.3f}, "
            f"RMSD={avg_rmsd_uncond_to_forward:.2f}Å"
        )

        # Quality control check: Verify forward folding TM-score meets threshold
        if hasattr(gen_cfg, "self_reflection") and hasattr(gen_cfg.self_reflection, "quality_control"):
            qc_config = gen_cfg.self_reflection.quality_control
            if qc_config.get("enable_tm_threshold", False):
                min_tm_score = qc_config.get("min_tm_score_forward", 0.7)
                if avg_tm_uncond_to_forward < min_tm_score:
                    logger.warning(
                        f"  Quality control FAILED: Forward folding TM-score "
                        f"{avg_tm_uncond_to_forward:.3f} < threshold {min_tm_score:.3f}"
                    )
                    logger.warning("  Iteration will be retried")
                    if save_failed:
                        for i in range(batch_size):
                            _save_failed_self_reflection_attempt(
                                output_dir=output_dir,
                                current_length=current_length,
                                iteration=iteration,
                                retry_count=retry_count,
                                failure_reason="forward_tm",
                                initial_seq_str=_initial_seq_str(i),
                                initial_structure=initial_structure[i],
                                mask_i=mask[i],
                                extra_metrics={
                                    "tm_score_unconditional_to_forward": tm_scores_uncond_to_forward[i],
                                    "rmsd_unconditional_to_forward": rmsd_uncond_to_forward[i],
                                },
                            )
                    return None
                else:
                    logger.info(
                        f"  Quality control PASSED: TM-score {avg_tm_uncond_to_forward:.3f} "
                        f">= threshold {min_tm_score:.3f}"
                    )

        # Save forward-folded structures
        if save_structures:
            for i in range(batch_size):
                filename = output_dir / (
                    f"self_reflection_forward_length_{current_length}_iter_{iteration:03d}_sample_{i:02d}.pdb"
                )
                forward_structure_i = forward_structure[i, mask[i] == 1]
                forward_seq_i = forward_seq[i, mask[i] == 1]
                writepdb(str(filename), forward_structure_i, forward_seq_i)

        logger.info("  Saved forward-folded structures")

        # Step 3: Inverse folding
        logger.info("Step 3: Inverse folding (structure → sequence refinement)...")
        inverse_params = _get_self_reflection_params(cfg, "inverse_folding")
        logger.info(f"  Inverse folding parameters: {inverse_params}")

        # Get inference schedule classes from inverse folding parameters
        inference_schedule_seq = inverse_params.get("inference_schedule_seq", "LogInferenceSchedule")
        if isinstance(inference_schedule_seq, str):
            inference_schedule_seq = _get_inference_schedule_class(inference_schedule_seq)

        inverse_sample = model.generate_sample(
            length=current_length,
            num_samples=batch_size,
            inverse_folding=True,
            input_structure_coords=forward_structure,
            input_mask=mask,
            input_indices=indices,
            nsteps=inverse_params["nsteps"],
            temperature_seq=inverse_params["temperature_seq"],
            stochasticity_seq=inverse_params["stochasticity_seq"],
            inference_schedule_seq=inference_schedule_seq,
            asynchronous_sampling=gen_cfg.get("asynchronous_sampling", False),
        )

        # Decode inverse-folded structures
        inverse_decoded_x = model.decode_structure(inverse_sample, mask)
        inverse_structure = None
        for decoder_name in inverse_decoded_x:
            if "vit_decoder" == decoder_name:
                inverse_structure = inverse_decoded_x[decoder_name]
                break

        if inverse_structure is None:
            logger.error("No structure decoder found in inverse folding output")
            return None

        # Extract inverse-folded sequences (refined)
        if inverse_sample["sequence_logits"].shape[-1] == 33:
            refined_seq = convert_lobster_aa_tokenization_to_standard_aa(
                inverse_sample["sequence_logits"], device=device
            )
        else:
            refined_seq = inverse_sample["sequence_logits"].argmax(dim=-1)
            refined_seq[refined_seq > 21] = 20

        # Save inverse-folded (refined) structures
        if save_structures:
            for i in range(batch_size):
                filename = output_dir / (
                    f"self_reflection_inverse_length_{current_length}_iter_{iteration:03d}_sample_{i:02d}.pdb"
                )
                inverse_structure_i = inverse_structure[i, mask[i] == 1]
                refined_seq_i = refined_seq[i, mask[i] == 1]
                writepdb(str(filename), inverse_structure_i, refined_seq_i)

        logger.info("  Saved inverse-folded (refined) structures")

        # Calculate TM-score and RMSD between forward and inverse-folded
        tm_scores_forward_to_inverse = []
        rmsd_forward_to_inverse = []

        for i in range(batch_size):
            forward_coords = forward_structure[i, mask[i] == 1, :, :]
            inverse_coords = inverse_structure[i, mask[i] == 1, :, :]
            seq_i = refined_seq[i, mask[i] == 1]
            sequence_str = "".join([restype_order_with_x_inv[j.item()] for j in seq_i])

            # TM-align RMSD
            tm_out = tm_align(
                inverse_coords[:, 1, :].cpu().numpy(),
                forward_coords[:, 1, :].detach().cpu().numpy(),
                sequence_str,
                sequence_str,
            )
            tm_scores_forward_to_inverse.append(tm_out.tm_norm_chain1)

            # Kabsch RMSD
            rmsd = align_and_compute_rmsd(
                coords1=inverse_coords,
                coords2=forward_coords,
                mask=None,  # Use all positions
                return_aligned=False,
                device=device,
            )
            rmsd_forward_to_inverse.append(rmsd)

        avg_tm_forward_to_inverse = sum(tm_scores_forward_to_inverse) / len(tm_scores_forward_to_inverse)
        avg_rmsd_forward_to_inverse = sum(rmsd_forward_to_inverse) / len(rmsd_forward_to_inverse)

        logger.info(
            f"  Forward → Inverse: TM-score={avg_tm_forward_to_inverse:.3f}, RMSD={avg_rmsd_forward_to_inverse:.2f}Å"
        )

        # Step 4: Sequence recovery metrics
        logger.info("Step 4: Calculating sequence recovery metrics...")
        percent_identities = []

        for i in range(batch_size):
            orig_seq = initial_seq[i, mask[i] == 1]
            ref_seq = refined_seq[i, mask[i] == 1]
            min_len = min(len(orig_seq), len(ref_seq))

            if min_len > 0:
                percent_identity = calculate_percent_identity(
                    orig_seq[:min_len].unsqueeze(0), ref_seq[:min_len].unsqueeze(0)
                )
                percent_identities.append(percent_identity.item())
            else:
                percent_identities.append(0.0)

        avg_percent_identity = sum(percent_identities) / len(percent_identities)
        logger.info(f"  Sequence identity (initial → refined): {avg_percent_identity:.2f}%")

        # Quality control check: Verify percent identity meets thresholds
        if hasattr(gen_cfg, "self_reflection") and hasattr(gen_cfg.self_reflection, "quality_control"):
            qc_config = gen_cfg.self_reflection.quality_control

            # Check minimum percent identity (too low = too much change)
            if qc_config.get("enable_min_percent_identity_threshold", False):
                min_percent_identity = qc_config.get("min_percent_identity", 20.0)
                if avg_percent_identity < min_percent_identity:
                    logger.warning(
                        f"  Quality control FAILED: Percent identity "
                        f"{avg_percent_identity:.2f}% < minimum threshold {min_percent_identity:.2f}%"
                    )
                    logger.warning("  Iteration will be retried (too much sequence change)")
                    return None
                else:
                    logger.info(
                        f"  Quality control PASSED: Percent identity {avg_percent_identity:.2f}% "
                        f">= minimum threshold {min_percent_identity:.2f}%"
                    )

            # Check maximum percent identity (too high = insufficient refinement)
            if qc_config.get("enable_max_percent_identity_threshold", False):
                max_percent_identity = qc_config.get("max_percent_identity", 90.0)
                if avg_percent_identity > max_percent_identity:
                    logger.warning(
                        f"  Quality control FAILED: Percent identity "
                        f"{avg_percent_identity:.2f}% > maximum threshold {max_percent_identity:.2f}%"
                    )
                    logger.warning("  Iteration will be retried (insufficient sequence refinement)")
                    return None
                else:
                    logger.info(
                        f"  Quality control PASSED: Percent identity {avg_percent_identity:.2f}% "
                        f"<= maximum threshold {max_percent_identity:.2f}%"
                    )

        # Step 4.5: ESMFold Validation (if enabled and available)
        esmfold_metrics = {}
        use_esmfold_validation = False
        if hasattr(gen_cfg, "self_reflection"):
            use_esmfold_validation = gen_cfg.self_reflection.get("use_esmfold_validation", False)

        if plm_fold is not None and use_esmfold_validation:
            logger.info("Step 4.5: ESMFold Validation...")

            # Substep A: Fold unconditional sequences (baseline)
            logger.info("  Folding unconditional sequences (baseline)...")
            plddt_unconditional_list = []
            pae_unconditional_list = []
            tm_esmfold_unconditional_list = []
            rmsd_esmfold_unconditional_list = []
            folded_coords_unconditional = []  # Store ESMFold predictions for structure comparison

            for i in range(batch_size):
                # Convert sequence to string
                seq_i = initial_seq[i, mask[i] == 1]
                sequence_str = "".join([restype_order_with_x_inv[j.item()] for j in seq_i])

                # Tokenize sequence
                tokenized_input = plm_fold.tokenizer.encode_plus(
                    sequence_str,
                    padding=True,
                    truncation=True,
                    max_length=cfg.generation.get("max_length", 512),
                    add_special_tokens=False,
                    return_tensors="pt",
                )["input_ids"].to(device)

                # Fold with ESMFold
                with torch.no_grad():
                    esmfold_outputs = plm_fold.model(tokenized_input)

                # Get reference structure
                ref_coords = initial_structure[i, mask[i] == 1, :, :].unsqueeze(0)

                # Calculate metrics
                folded_metrics, folded_coords = get_folded_structure_metrics(
                    esmfold_outputs, ref_coords, [sequence_str], mask=mask[i : i + 1], device=device
                )

                plddt_unconditional_list.append(folded_metrics["_plddt"])
                pae_unconditional_list.append(folded_metrics["_predicted_aligned_error"])
                tm_esmfold_unconditional_list.append(folded_metrics["_tm_score"])
                rmsd_esmfold_unconditional_list.append(folded_metrics["_rmsd"])

                # Store ESMFold predicted coordinates for structure comparison
                folded_coords_unconditional.append(folded_coords[0])

                # Save ESMFold baseline structure
                filename = output_dir / (
                    f"self_reflection_unconditional_esmfold_length_{current_length}_iter_{iteration:03d}_sample_{i:02d}.pdb"
                )
                folded_coords_i = folded_coords[0, mask[i] == 1]
                seq_i_masked = initial_seq[i, mask[i] == 1]
                writepdb(str(filename), folded_coords_i, seq_i_masked)

            avg_plddt_unconditional = sum(plddt_unconditional_list) / len(plddt_unconditional_list)
            avg_pae_unconditional = sum(pae_unconditional_list) / len(pae_unconditional_list)
            avg_tm_esmfold_unconditional = sum(tm_esmfold_unconditional_list) / len(tm_esmfold_unconditional_list)
            avg_rmsd_esmfold_unconditional = sum(rmsd_esmfold_unconditional_list) / len(rmsd_esmfold_unconditional_list)

            logger.info(f"    pLDDT: {avg_plddt_unconditional:.2f}")
            logger.info(
                f"    TM-score: {avg_tm_esmfold_unconditional:.3f}, RMSD: {avg_rmsd_esmfold_unconditional:.2f}Å"
            )

            # Substep B: Compare unconditional structures to ESMFold predictions
            logger.info("  Comparing unconditional structures to ESMFold predictions...")
            tm_scores_unconditional_to_esmfold = []
            rmsd_unconditional_to_esmfold = []

            for i in range(batch_size):
                # Get unconditional structure and ESMFold prediction
                uncond_coords = initial_structure[i, mask[i] == 1, :, :]
                esmfold_coords = folded_coords_unconditional[i][mask[i] == 1]
                seq_i = initial_seq[i, mask[i] == 1]
                sequence_str = "".join([restype_order_with_x_inv[j.item()] for j in seq_i])

                # TM-align
                tm_out = tm_align(
                    uncond_coords[:, 1, :].cpu().numpy(),
                    esmfold_coords[:, 1, :].detach().cpu().numpy(),
                    sequence_str,
                    sequence_str,
                )
                tm_scores_unconditional_to_esmfold.append(tm_out.tm_norm_chain1)

                # Kabsch RMSD
                rmsd = align_and_compute_rmsd(
                    coords1=uncond_coords,
                    coords2=esmfold_coords,
                    mask=None,
                    return_aligned=False,
                    device=device,
                )
                rmsd_unconditional_to_esmfold.append(rmsd)

            avg_tm_unconditional_to_esmfold = sum(tm_scores_unconditional_to_esmfold) / len(
                tm_scores_unconditional_to_esmfold
            )
            avg_rmsd_unconditional_to_esmfold = sum(rmsd_unconditional_to_esmfold) / len(rmsd_unconditional_to_esmfold)

            logger.info(
                f"    Unconditional → ESMFold: TM-score={avg_tm_unconditional_to_esmfold:.3f}, "
                f"RMSD={avg_rmsd_unconditional_to_esmfold:.2f}Å"
            )

            # Substep C: Fold refined sequences (improved)
            logger.info("  Folding refined sequences (improved)...")
            plddt_refined_list = []
            pae_refined_list = []
            tm_esmfold_refined_list = []
            rmsd_esmfold_refined_list = []
            folded_coords_refined = []  # Store ESMFold predictions for structure comparison

            for i in range(batch_size):
                # Convert refined sequence to string
                seq_i = refined_seq[i, mask[i] == 1]
                sequence_str = "".join([restype_order_with_x_inv[j.item()] for j in seq_i])

                # Tokenize sequence
                tokenized_input = plm_fold.tokenizer.encode_plus(
                    sequence_str,
                    padding=True,
                    truncation=True,
                    max_length=cfg.generation.get("max_length", 512),
                    add_special_tokens=False,
                    return_tensors="pt",
                )["input_ids"].to(device)

                # Fold with ESMFold
                with torch.no_grad():
                    esmfold_outputs = plm_fold.model(tokenized_input)

                # Get reference structure (inverse folded structure)
                ref_coords = inverse_structure[i, mask[i] == 1, :, :].unsqueeze(0)

                # Calculate metrics
                folded_metrics, folded_coords = get_folded_structure_metrics(
                    esmfold_outputs, ref_coords, [sequence_str], mask=mask[i : i + 1], device=device
                )

                plddt_refined_list.append(folded_metrics["_plddt"])
                pae_refined_list.append(folded_metrics["_predicted_aligned_error"])
                tm_esmfold_refined_list.append(folded_metrics["_tm_score"])
                rmsd_esmfold_refined_list.append(folded_metrics["_rmsd"])

                # Store ESMFold predicted coordinates for structure comparison
                folded_coords_refined.append(folded_coords[0])

                # Save ESMFold refined structure
                filename = output_dir / (
                    f"self_reflection_refined_esmfold_length_{current_length}_iter_{iteration:03d}_sample_{i:02d}.pdb"
                )
                folded_coords_i = folded_coords[0, mask[i] == 1]
                seq_i_masked = refined_seq[i, mask[i] == 1]
                writepdb(str(filename), folded_coords_i, seq_i_masked)

            avg_plddt_refined = sum(plddt_refined_list) / len(plddt_refined_list)
            avg_pae_refined = sum(pae_refined_list) / len(pae_refined_list)
            avg_tm_esmfold_refined = sum(tm_esmfold_refined_list) / len(tm_esmfold_refined_list)
            avg_rmsd_esmfold_refined = sum(rmsd_esmfold_refined_list) / len(rmsd_esmfold_refined_list)

            logger.info(f"    pLDDT: {avg_plddt_refined:.2f}")
            logger.info(f"    TM-score: {avg_tm_esmfold_refined:.3f}, RMSD: {avg_rmsd_esmfold_refined:.2f}Å")

            # Substep D: Compare forward-folded structures to ESMFold predictions
            logger.info("  Comparing forward-folded structures to ESMFold predictions...")
            tm_scores_forward_to_esmfold = []
            rmsd_forward_to_esmfold = []

            for i in range(batch_size):
                # Get forward-folded structure and ESMFold prediction from unconditional sequence
                forward_coords = forward_structure[i, mask[i] == 1, :, :]
                esmfold_coords = folded_coords_unconditional[i][mask[i] == 1]
                seq_i = initial_seq[i, mask[i] == 1]
                sequence_str = "".join([restype_order_with_x_inv[j.item()] for j in seq_i])

                # TM-align
                tm_out = tm_align(
                    forward_coords[:, 1, :].cpu().numpy(),
                    esmfold_coords[:, 1, :].detach().cpu().numpy(),
                    sequence_str,
                    sequence_str,
                )
                tm_scores_forward_to_esmfold.append(tm_out.tm_norm_chain1)

                # Kabsch RMSD
                rmsd = align_and_compute_rmsd(
                    coords1=forward_coords,
                    coords2=esmfold_coords,
                    mask=None,
                    return_aligned=False,
                    device=device,
                )
                rmsd_forward_to_esmfold.append(rmsd)

            avg_tm_forward_to_esmfold = sum(tm_scores_forward_to_esmfold) / len(tm_scores_forward_to_esmfold)
            avg_rmsd_forward_to_esmfold = sum(rmsd_forward_to_esmfold) / len(rmsd_forward_to_esmfold)

            logger.info(
                f"    Forward-folded → ESMFold: TM-score={avg_tm_forward_to_esmfold:.3f}, "
                f"RMSD={avg_rmsd_forward_to_esmfold:.2f}Å"
            )

            # Substep E: Calculate ESMFold agreement improvement
            tm_esmfold_agreement_improvement = avg_tm_forward_to_esmfold - avg_tm_unconditional_to_esmfold
            rmsd_esmfold_agreement_improvement = avg_rmsd_unconditional_to_esmfold - avg_rmsd_forward_to_esmfold

            logger.info("  ESMFold Agreement Improvement:")
            logger.info(
                f"    TM-score improvement: {tm_esmfold_agreement_improvement:+.3f} "
                f"(Unconditional→Forward better agreement with ESMFold)"
            )
            logger.info(
                f"    RMSD improvement: {rmsd_esmfold_agreement_improvement:+.2f}Å "
                f"(Positive = Forward closer to ESMFold)"
            )

            # Substep F: Calculate baseline improvements
            plddt_improvement = avg_plddt_refined - avg_plddt_unconditional
            pae_improvement = avg_pae_unconditional - avg_pae_refined
            tm_improvement = avg_tm_esmfold_refined - avg_tm_esmfold_unconditional
            rmsd_improvement = avg_rmsd_esmfold_unconditional - avg_rmsd_esmfold_refined

            logger.info("  Improvement Summary:")
            logger.info(
                f"    pLDDT: {plddt_improvement:+.2f} ({plddt_improvement / avg_plddt_unconditional * 100:+.1f}%)"
            )
            logger.info(f"    PAE: {pae_improvement:+.2f}Å")
            logger.info(f"    TM-score: {tm_improvement:+.3f}")
            logger.info(f"    RMSD: {rmsd_improvement:+.2f}Å")

            # Store ESMFold metrics
            esmfold_metrics = {
                "plddt_unconditional": avg_plddt_unconditional,
                "pae_unconditional": avg_pae_unconditional,
                "tm_score_esmfold_unconditional": avg_tm_esmfold_unconditional,
                "rmsd_esmfold_unconditional": avg_rmsd_esmfold_unconditional,
                "plddt_refined": avg_plddt_refined,
                "pae_refined": avg_pae_refined,
                "tm_score_esmfold_refined": avg_tm_esmfold_refined,
                "rmsd_esmfold_refined": avg_rmsd_esmfold_refined,
                "plddt_improvement": plddt_improvement,
                "pae_improvement": pae_improvement,
                "tm_score_improvement": tm_improvement,
                "rmsd_improvement": rmsd_improvement,
                # ESMFold structure comparison metrics
                "tm_score_unconditional_to_esmfold": avg_tm_unconditional_to_esmfold,
                "rmsd_unconditional_to_esmfold": avg_rmsd_unconditional_to_esmfold,
                "tm_score_forward_to_esmfold": avg_tm_forward_to_esmfold,
                "rmsd_forward_to_esmfold": avg_rmsd_forward_to_esmfold,
                "tm_score_esmfold_agreement_improvement": tm_esmfold_agreement_improvement,
                "rmsd_esmfold_agreement_improvement": rmsd_esmfold_agreement_improvement,
            }
        else:
            if plm_fold is None:
                logger.info("Step 4.5: Skipping ESMFold validation (ESMFold model not available)")
            elif not use_esmfold_validation:
                logger.info("Step 4.5: Skipping ESMFold validation (disabled in self_reflection config)")

        # Step 5: Return metrics
        metrics = {
            "percent_identity_self_reflection": avg_percent_identity,
            "tm_score_unconditional_to_forward": avg_tm_uncond_to_forward,
            "rmsd_unconditional_to_forward": avg_rmsd_uncond_to_forward,
            "tm_score_forward_to_inverse": avg_tm_forward_to_inverse,
            "rmsd_forward_to_inverse": avg_rmsd_forward_to_inverse,
        }

        # Add ESMFold metrics if available
        metrics.update(esmfold_metrics)

        logger.info("=" * 80)
        logger.info("Self-reflection refinement pipeline completed successfully")
        logger.info("=" * 80)

        return metrics

    except Exception as e:
        logger.error(f"Self-reflection pipeline failed: {e}")
        import traceback

        traceback.print_exc()
        return None


def _get_self_reflection_params(cfg: DictConfig, stage: str) -> dict:
    """Get generation parameters for self-reflection pipeline stage with fallback.

    Args:
        cfg: Configuration
        stage: Either 'forward_folding' or 'inverse_folding'

    Returns:
        Dictionary of generation parameters
    """
    gen_cfg = cfg.generation

    # Try to get stage-specific parameters
    if hasattr(gen_cfg, "self_reflection") and hasattr(gen_cfg.self_reflection, stage):
        stage_cfg = getattr(gen_cfg.self_reflection, stage)
        return {
            "nsteps": stage_cfg.get("nsteps", 100 if stage == "forward_folding" else 200),
            "temperature_seq": stage_cfg.get("temperature_seq", gen_cfg.get("temperature_seq", 0.5)),
            "temperature_struc": stage_cfg.get("temperature_struc", gen_cfg.get("temperature_struc", 1.0)),
            "stochasticity_seq": stage_cfg.get("stochasticity_seq", gen_cfg.get("stochasticity_seq", 20)),
            "stochasticity_struc": stage_cfg.get("stochasticity_struc", gen_cfg.get("stochasticity_struc", 20)),
        }

    # Fallback to main generation parameters
    return {
        "nsteps": gen_cfg.get("nsteps", 100 if stage == "forward_folding" else 200),
        "temperature_seq": gen_cfg.get("temperature_seq", 0.5),
        "temperature_struc": gen_cfg.get("temperature_struc", 1.0),
        "stochasticity_seq": gen_cfg.get("stochasticity_seq", 20),
        "stochasticity_struc": gen_cfg.get("stochasticity_struc", 20),
    }


def _generate_unconditional(
    model, cfg: DictConfig, device: torch.device, output_dir: Path, plm_fold=None, csv_writer=None, plotter=None
) -> None:
    """Generate structures unconditionally."""
    logger.info("Starting unconditional generation...")

    gen_cfg = cfg.generation
    length = gen_cfg.length
    num_samples = gen_cfg.num_samples
    nsteps = gen_cfg.get("nsteps", 200)
    batch_size = gen_cfg.get("batch_size", 1)

    # Handle both single length and list of lengths
    # Check for ListConfig, list, or tuple
    if hasattr(length, "__iter__") and not isinstance(length, (str, int, float)):
        # Convert ListConfig/list/tuple to regular list if needed
        lengths = list(length)
        logger.info(f"Generating {num_samples} structures for each length in {lengths}")
    else:
        lengths = [int(length)]
        logger.info(f"Generating {num_samples} structures of length {length}")

    # Process each length
    for current_length in lengths:
        # Ensure current_length is an integer
        current_length = int(current_length)

        logger.info("=" * 60)
        logger.info(f"PROCESSING LENGTH: {current_length}")
        logger.info("=" * 60)

        n_iterations = num_samples // batch_size
        logger.info(
            f"Generating {num_samples} structures of length {current_length} with {nsteps} steps, will run with batch size {batch_size} for {n_iterations} iterations"
        )

        # Resume support: build set of already-completed iterations (by PDB existence)
        # This handles gaps from skipped iterations (e.g. max_retries exceeded)
        completed_iterations = set()
        is_resuming = gen_cfg.get("resume", False)
        if is_resuming:
            for check_iter in range(n_iterations):
                all_exist = True
                for check_i in range(batch_size):
                    check_file = (
                        output_dir
                        / f"generated_structure_length_{current_length}_{check_iter * batch_size + check_i:03d}.pdb"
                    )
                    if not check_file.exists():
                        all_exist = False
                        break
                if all_exist:
                    completed_iterations.add(check_iter)
            if completed_iterations:
                logger.info(
                    f"Resuming: {len(completed_iterations)}/{n_iterations} iterations already complete "
                    f"for length {current_length} (will skip them individually)"
                )
            if len(completed_iterations) >= n_iterations:
                logger.info(f"All {n_iterations} iterations already complete for length {current_length}, skipping")
                continue

        # Initialize metrics collection for this length, pre-loading from CSV on resume.
        # Deduplicate by run_id keeping only the latest entry (by row order / timestamp).
        all_metrics = []
        if is_resuming and completed_iterations:
            existing_csvs = sorted(output_dir.glob("*_metrics_*.csv"), key=lambda x: x.stat().st_mtime)
            if existing_csvs:
                csv_path = existing_csvs[-1]
                logger.info(f"Loading prior metrics from {csv_path} for length {current_length}")
                try:
                    csv_col_to_internal_key = {
                        "plddt": "_plddt",
                        "predicted_aligned_error": "_predicted_aligned_error",
                        "tm_score": "_tm_score",
                        "rmsd": "_rmsd",
                    }
                    metrics_by_run_id = {}
                    with open(csv_path, newline="") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            if row.get("sequence_length") and int(float(row["sequence_length"])) == current_length:
                                run_id = row.get("run_id", "")
                                metrics_dict = {}
                                for key, value in row.items():
                                    if key in ("run_id", "timestamp", "mode", "sequence_length", "num_samples"):
                                        continue
                                    if value is not None and value != "":
                                        try:
                                            internal_key = csv_col_to_internal_key.get(key, key)
                                            metrics_dict[internal_key] = float(value)
                                        except (ValueError, TypeError):
                                            pass
                                if metrics_dict:
                                    metrics_by_run_id[run_id] = metrics_dict
                    all_metrics = list(metrics_by_run_id.values())
                    logger.info(f"Pre-loaded {len(all_metrics)} unique metric entries for length {current_length}")
                except Exception as e:
                    logger.warning(f"Failed to load prior metrics from CSV: {e}")
                    all_metrics = []

        # Get quality control config for retry logic
        qc_config = {}
        if hasattr(gen_cfg, "self_reflection") and hasattr(gen_cfg.self_reflection, "quality_control"):
            qc_config = gen_cfg.self_reflection.quality_control

        # Check for independent sequence token check (not tied to self-reflection)
        enable_sequence_token_check = gen_cfg.get("enable_sequence_token_check", True)
        sequence_token_check_retries = gen_cfg.get("sequence_token_check_retries", 10)

        # Enable retries if any QC threshold is enabled (from self-reflection)
        self_reflection_qc_enabled = (
            qc_config.get("enable_tm_threshold", False)
            or qc_config.get("enable_min_percent_identity_threshold", False)
            or qc_config.get("enable_max_percent_identity_threshold", False)
            or qc_config.get("enable_sequence_token_check", True)  # Token check enabled by default
        )

        # Determine max_retries based on what's enabled
        if self_reflection_qc_enabled:
            max_retries = qc_config.get("max_retries", 3)
            if enable_sequence_token_check and not gen_cfg.get("enable_self_reflection", False):
                max_retries = sequence_token_check_retries
        else:
            max_retries = 0

        # Track retry statistics
        total_retries = 0
        max_retries_exceeded = 0

        # Build sequence logit bias tensor from config
        sequence_logit_bias = None
        bias_cfg = gen_cfg.get("sequence_logit_bias", None)
        if bias_cfg:
            from lobster.tokenization._amino_acid import AA_VOCAB

            sequence_logit_bias = torch.zeros(len(AA_VOCAB), device=device)
            for aa, bias_val in bias_cfg.items():
                if aa in AA_VOCAB:
                    sequence_logit_bias[AA_VOCAB[aa]] = float(bias_val)
                else:
                    logger.warning(f"Unknown amino acid '{aa}' in sequence_logit_bias, skipping")
            logger.info(f"Sequence logit bias: {dict(bias_cfg)}")

        sequence_logit_bias_steps = int(gen_cfg.get("sequence_logit_bias_steps", 10))
        if sequence_logit_bias is not None:
            logger.info(f"Sequence logit bias applied for first {sequence_logit_bias_steps} denoising steps")

        for n_iter in range(n_iterations):
            if n_iter in completed_iterations:
                logger.debug(f"Skipping already-completed iteration {n_iter + 1}/{n_iterations}")
                continue
            logger.info(f"Iteration {n_iter + 1}/{n_iterations}")

            # Retry loop for quality control
            retry_count = 0
            iteration_success = False

            while retry_count <= max_retries and not iteration_success:
                if retry_count > 0:
                    logger.info(f"  Retry attempt {retry_count}/{max_retries} for iteration {n_iter + 1}")
                    total_retries += 1

                with torch.no_grad():
                    # Get inference schedule classes from config
                    inference_schedule_seq = gen_cfg.get("inference_schedule_seq", "LogInferenceSchedule")
                    inference_schedule_struc = gen_cfg.get("inference_schedule_struc", "LinearInferenceSchedule")

                    # Convert string names to classes if needed
                    if isinstance(inference_schedule_seq, str):
                        inference_schedule_seq = _get_inference_schedule_class(inference_schedule_seq)
                    if isinstance(inference_schedule_struc, str):
                        inference_schedule_struc = _get_inference_schedule_class(inference_schedule_struc)

                    # Build sequence anchor tensors if enabled
                    anchor_tokens = None
                    anchor_mask = None
                    anchor_cfg = gen_cfg.get("sequence_anchor_fraction", 0.0)
                    if isinstance(anchor_cfg, (dict, DictConfig)):
                        anchor_fraction = float(
                            anchor_cfg.get(current_length, anchor_cfg.get(str(current_length), 0.0))
                        )
                    else:
                        anchor_fraction = float(anchor_cfg)
                    if anchor_fraction > 0.0:
                        num_anchors = max(1, int(current_length * anchor_fraction))
                        anchor_positions = torch.randperm(current_length, device=device)[:num_anchors]
                        # Sample from 19 amino acids excluding Cysteine (index 4)
                        allowed_aa = torch.tensor(
                            [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], device=device
                        )
                        rand_indices = torch.randint(0, len(allowed_aa), (batch_size, current_length), device=device)
                        anchor_tokens = allowed_aa[rand_indices]
                        anchor_mask = torch.ones((batch_size, current_length), device=device)
                        anchor_mask[:, anchor_positions] = 0  # 0 = keep anchored
                        logger.info(
                            f"Sequence anchors enabled: fixing {num_anchors}/{current_length} "
                            f"positions ({anchor_fraction * 100:.0f}%)"
                        )

                    # Generate samples
                    generate_sample = model.generate_sample(
                        length=current_length,
                        num_samples=batch_size,
                        nsteps=nsteps,
                        temperature_seq=gen_cfg.get("temperature_seq", 0.5),
                        temperature_struc=gen_cfg.get("temperature_struc", 1.0),
                        stochasticity_seq=gen_cfg.get("stochasticity_seq", 20),
                        stochasticity_struc=gen_cfg.get("stochasticity_struc", 20),
                        inference_schedule_seq=inference_schedule_seq,
                        inference_schedule_struc=inference_schedule_struc,
                        asynchronous_sampling=gen_cfg.get("asynchronous_sampling", False),
                        sequence_anchor_tokens=anchor_tokens,
                        sequence_anchor_mask=anchor_mask,
                        sequence_logit_bias=sequence_logit_bias,
                        sequence_logit_bias_steps=sequence_logit_bias_steps,
                    )

                    # Create mask for decoding
                    mask = torch.ones((batch_size, current_length), device=device)

                    # Self-reflection refinement pipeline (if enabled)
                    self_reflection_metrics = None
                    if gen_cfg.get("enable_self_reflection", False):
                        logger.info("Executing self-reflection refinement pipeline...")
                        self_reflection_metrics = _execute_self_reflection_pipeline(
                            model=model,
                            cfg=cfg,
                            device=device,
                            output_dir=output_dir,
                            plm_fold=plm_fold,
                            generate_sample=generate_sample,
                            mask=mask,
                            iteration=n_iter,
                            batch_size=batch_size,
                            current_length=current_length,
                            retry_count=retry_count,
                        )

                        if self_reflection_metrics is not None:
                            # Success! Store self-reflection metrics for aggregate statistics
                            all_metrics.append(self_reflection_metrics)

                            # Write to CSV if writer is available
                            if csv_writer is not None:
                                run_id = f"self_reflection_length_{current_length}_iter_{n_iter:03d}"
                                csv_writer.write_batch_metrics(
                                    self_reflection_metrics,
                                    run_id,
                                    sequence_length=current_length,
                                    num_samples=batch_size,
                                )
                            iteration_success = True
                        else:
                            # Quality control failed, will retry
                            retry_count += 1
                            if retry_count > max_retries:
                                logger.warning(
                                    f"  Max retries ({max_retries}) exceeded for iteration {n_iter + 1}. "
                                    f"Using current sequences despite quality control failure. "
                                    f"Skipping self-reflection for this iteration."
                                )
                                max_retries_exceeded += 1
                                iteration_success = True
                            continue
                    elif enable_sequence_token_check:
                        # Extract sequences for validation
                        if generate_sample["sequence_logits"].shape[-1] == 33:
                            check_seq = convert_lobster_aa_tokenization_to_standard_aa(
                                generate_sample["sequence_logits"], device=device
                            )
                        else:
                            check_seq = generate_sample["sequence_logits"].argmax(dim=-1)
                            check_seq[check_seq > 21] = 20

                        # Run sequence token check
                        is_valid, error_msg = _check_sequence_tokens(check_seq, mask, "unconditional generation")
                        if not is_valid:
                            logger.warning(f"  Sequence token check FAILED: {error_msg}")
                            logger.warning("  Iteration will be retried (invalid sequence tokens)")
                            retry_count += 1
                            if retry_count > max_retries:
                                logger.warning(
                                    f"  Max retries ({max_retries}) exceeded for iteration {n_iter + 1}. "
                                    f"Using current sequences despite quality control failure."
                                )
                                max_retries_exceeded += 1
                                iteration_success = True
                            continue
                        else:
                            logger.info("  Sequence token check PASSED: All sequences contain valid amino acids")
                            iteration_success = True
                    else:
                        # No quality control at all
                        iteration_success = True

                    # Only proceed with normal flow if iteration succeeded or max retries exceeded
                    if not iteration_success and retry_count <= max_retries:
                        continue

                    # Decode structures
                    decoded_x = model.decode_structure(generate_sample, mask)

                    # Extract coordinates
                    x_recon_xyz = None
                    for decoder_name in decoded_x:
                        if "vit_decoder" == decoder_name:
                            x_recon_xyz = decoded_x[decoder_name]
                            break

                    if x_recon_xyz is None:
                        raise RuntimeError("No structure decoder found in model output")

                    # Extract sequences
                    if generate_sample["sequence_logits"].shape[-1] == 33:
                        seq = convert_lobster_aa_tokenization_to_standard_aa(
                            generate_sample["sequence_logits"], device=device
                        )
                    else:
                        seq = generate_sample["sequence_logits"].argmax(dim=-1)
                        seq[seq > 21] = 20

                    # Extract structure tokens (argmax)
                    structure_tokens = generate_sample["structure_logits"].argmax(dim=-1)  # Shape: [batch_size, length]

                    # Write sequences to CSV
                    # Note: For self-reflection mode, we only store initial unconditional sequences (not forward/inverse intermediates)
                    if csv_writer is not None:
                        # Convert sequences to strings
                        sequence_strs = []
                        structure_token_strs = []
                        for i in range(batch_size):
                            seq_i = seq[i, mask[i] == 1]
                            sequence_str = "".join([restype_order_with_x_inv[j.item()] for j in seq_i])
                            sequence_strs.append(sequence_str)

                            # Convert structure tokens to comma-separated string
                            tokens_i = structure_tokens[i, mask[i] == 1]
                            tokens_str = ",".join([str(t.item()) for t in tokens_i])
                            structure_token_strs.append(tokens_str)

                        # Write to sequences CSV
                        csv_writer.write_sequences(
                            sequences=sequence_strs,
                            run_id=f"unconditional_length_{current_length}_iter_{n_iter:03d}",
                            iteration=n_iter,
                            sequence_type="unconditional",
                            latent_generator_tokens=structure_token_strs,
                        )

                    # Save generated structures
                    logger.info("Saving generated structures...")
                    for i in range(batch_size):
                        filename = (
                            output_dir
                            / f"generated_structure_length_{current_length}_{n_iter * batch_size + i:03d}.pdb"
                        )
                        writepdb(str(filename), x_recon_xyz[i], seq[i])
                        logger.info(f"Saved: {filename}")

                    # Optional ESMFold validation
                    if plm_fold is not None:
                        logger.info("Validating structures with ESMFold...")
                        batch_metrics = _validate_with_esmfold(
                            seq,
                            x_recon_xyz,
                            plm_fold,
                            device,
                            output_dir,
                            f"generated_structure_length_{current_length}_{n_iter * batch_size + i:03d}",
                            max_length=current_length,
                        )

                        # Log metrics for unconditional generation
                        if batch_metrics and not batch_metrics.get("_skipped", False):
                            logger.info("ESMFold validation metrics for unconditional generation:")
                            for key, value in batch_metrics.items():
                                if isinstance(value, (int, float)):
                                    logger.info(f"  {key}: {value:.4f}")

                            # Store metrics for CSV logging
                            if csv_writer is not None:
                                run_id = f"unconditional_length_{current_length}_iter_{n_iter:03d}"
                                csv_writer.write_batch_metrics(
                                    batch_metrics, run_id, sequence_length=current_length, num_samples=batch_size
                                )

                            # Always collect metrics for aggregate statistics
                            all_metrics.append(batch_metrics)

        # Calculate and log aggregate statistics for this length
        if all_metrics:
            logger.info(f"Calculating aggregate statistics for length {current_length}...")

            # Collect all metric values
            metric_lists = {
                "_plddt": [],
                "_predicted_aligned_error": [],
                "_tm_score": [],
                "_rmsd": [],
                # Self-reflection refinement metrics
                "percent_identity_self_reflection": [],
                "tm_score_unconditional_to_forward": [],
                "rmsd_unconditional_to_forward": [],
                "tm_score_forward_to_inverse": [],
                "rmsd_forward_to_inverse": [],
                # ESMFold baseline metrics
                "plddt_unconditional": [],
                "pae_unconditional": [],
                "tm_score_esmfold_unconditional": [],
                "rmsd_esmfold_unconditional": [],
                # ESMFold refined metrics
                "plddt_refined": [],
                "pae_refined": [],
                "tm_score_esmfold_refined": [],
                "rmsd_esmfold_refined": [],
                # ESMFold improvement metrics
                "plddt_improvement": [],
                "pae_improvement": [],
                "tm_score_improvement": [],
                "rmsd_improvement": [],
                # ESMFold structure comparison metrics
                "tm_score_unconditional_to_esmfold": [],
                "rmsd_unconditional_to_esmfold": [],
                "tm_score_forward_to_esmfold": [],
                "rmsd_forward_to_esmfold": [],
                "tm_score_esmfold_agreement_improvement": [],
                "rmsd_esmfold_agreement_improvement": [],
            }

            for metrics in all_metrics:
                for key in metric_lists:
                    if key in metrics:
                        metric_lists[key].append(metrics[key])

            # Calculate aggregate statistics
            aggregate_stats = calculate_aggregate_stats(metric_lists)

            # Calculate RMSD pass rates (< 2.0Å threshold)
            rmsd_pass_rates = {}
            rmsd_threshold = 2.0

            # Check each RMSD metric in metric_lists
            rmsd_metrics = [
                "_rmsd",
                "rmsd_unconditional_to_forward",
                "rmsd_forward_to_inverse",
                "rmsd_esmfold_unconditional",
                "rmsd_esmfold_refined",
                "rmsd_unconditional_to_esmfold",
                "rmsd_forward_to_esmfold",
            ]

            for rmsd_metric in rmsd_metrics:
                if rmsd_metric in metric_lists and metric_lists[rmsd_metric]:
                    rmsd_values = metric_lists[rmsd_metric]
                    total_count = len(rmsd_values)
                    pass_count = sum(1 for rmsd in rmsd_values if rmsd < rmsd_threshold)
                    pass_rate = (pass_count / total_count * 100) if total_count > 0 else 0.0
                    rmsd_pass_rates[rmsd_metric] = (pass_count, total_count, pass_rate)

            # Log aggregate statistics
            logger.info("=" * 80)
            logger.info(f"UNCONDITIONAL GENERATION AGGREGATE STATISTICS - LENGTH {current_length}")
            logger.info("=" * 80)

            for metric_name, (avg_value, count) in aggregate_stats.items():
                logger.info(f"Average {metric_name}: {avg_value:.4f} (n={count})")

            # Log RMSD pass rates
            if rmsd_pass_rates:
                logger.info("")
                logger.info(f"RMSD Pass Rates (< {rmsd_threshold:.1f}Å):")
                for rmsd_metric, (pass_count, total_count, pass_rate) in rmsd_pass_rates.items():
                    logger.info(f"  {rmsd_metric}: {pass_count}/{total_count} ({pass_rate:.1f}%)")

            logger.info("=" * 80)

            # Log quality control statistics if enabled
            if max_retries > 0 and gen_cfg.get("enable_self_reflection", False):
                logger.info("")
                logger.info("SELF-REFLECTION QUALITY CONTROL SUMMARY")
                logger.info("=" * 80)
                logger.info(f"Total iterations: {n_iterations}")
                logger.info(f"Retries required: {total_retries}")
                logger.info(f"Max retries exceeded: {max_retries_exceeded}")
                if "tm_score_unconditional_to_forward" in metric_lists:
                    forward_tm_scores = metric_lists["tm_score_unconditional_to_forward"]
                    if forward_tm_scores:
                        avg_forward_tm = sum(forward_tm_scores) / len(forward_tm_scores)
                        logger.info(f"Average forward TM-score: {avg_forward_tm:.3f}")
                if "percent_identity_self_reflection" in metric_lists:
                    percent_identities = metric_lists["percent_identity_self_reflection"]
                    if percent_identities:
                        avg_percent_id = sum(percent_identities) / len(percent_identities)
                        logger.info(f"Average percent identity: {avg_percent_id:.2f}%")
            logger.info("=" * 80)

            # Foldseek Diversity Analysis
            if cfg.generation.get("calculate_foldseek_diversity", False):
                logger.info("")
                logger.info("FOLDSEEK DIVERSITY ANALYSIS")
                logger.info("=" * 80)

                foldseek_bin_path = cfg.generation.get(
                    "foldseek_bin_path",
                    str(Path(__file__).resolve().parent.parent / "metrics" / "foldseek" / "bin"),
                )

                try:
                    diversity_metrics = calculate_diversity_for_generation(
                        output_dir=output_dir,
                        length=current_length,
                        rmsd_threshold=cfg.generation.get("rmsd_threshold_for_diversity", 2.0),
                        foldseek_bin_path=foldseek_bin_path,
                        tmscore_threshold=cfg.generation.get("foldseek_tmscore_threshold", 0.5),
                    )

                    if diversity_metrics:
                        logger.info(f"Total structures passing RMSD threshold: {diversity_metrics['total_structures']}")
                        logger.info(
                            f"Number of Foldseek clusters (TM-score ≥ {diversity_metrics['tmscore_threshold']}): {diversity_metrics['num_clusters']}"
                        )
                        logger.info(f"Diversity percentage: {diversity_metrics['diversity_percentage']:.2f}%")

                        # Write to CSV if available
                        if csv_writer is not None:
                            csv_writer.write_diversity_metrics(
                                diversity_metrics=diversity_metrics, length=current_length
                            )

                except Exception as e:
                    logger.error(f"Foldseek diversity analysis failed: {e}")
                    import traceback

                    traceback.print_exc()

                logger.info("=" * 80)

            # Write aggregate statistics to CSV if writer is available
            if csv_writer is not None:
                csv_writer.write_aggregate_stats(aggregate_stats, length=current_length)

                # Write pass rate statistics to CSV if available
                if rmsd_pass_rates:
                    csv_writer.write_pass_rates(rmsd_pass_rates, length=current_length, threshold=rmsd_threshold)

    # Create plots from CSV data if plotter is available
    if plotter is not None and csv_writer is not None:
        logger.info("Creating box and whisker plots from CSV data...")
        try:
            plotter.create_box_plots_from_csv(csv_writer.csv_path)
            logger.info("✓ Box plots created successfully")
        except Exception as e:
            logger.error(f"Error creating box plots: {e}")

        # Create correlation plots (only for self-reflection enabled runs)
        logger.info("Creating correlation plots from CSV data...")
        try:
            plotter.create_correlation_plots_from_csv(csv_writer.csv_path)
            logger.info("✓ Correlation plots created successfully")
        except Exception as e:
            logger.error(f"Error creating correlation plots: {e}")
