import logging
from pathlib import Path
import glob

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from lobster.model.latent_generator.io import writepdb, load_pdb
from lobster.model.latent_generator.utils.residue_constants import (
    convert_lobster_aa_tokenization_to_standard_aa,
    restype_order_with_x_inv,
)
from lobster.metrics import (
    get_folded_structure_metrics,
    calculate_percent_identity,
    parse_mask_indices,
    MetricsPlotter,
    MetricsCSVWriter,
    calculate_aggregate_stats,
    align_and_compute_rmsd,
    _is_sequence_pattern,
    _create_sequence_pattern_masks,
    build_multichain_sequence_string,
    predict_structure_with_esmfold,
)
from lobster.metrics.cal_foldseek_clusters import calculate_diversity_for_generation
from lobster.transforms._structure_transforms import StructureBackboneTransform, AminoAcidTokenizerTransform
from lobster.model import LobsterPLMFold

logger = logging.getLogger(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="../hydra_config", config_name="generate")
def generate(cfg: DictConfig) -> None:
    """Generate protein structures using genUME model.

    This command-line interface supports:
    - Unconditional generation: Generate novel protein structures from scratch
    - Inverse folding: Generate sequences for given protein structures
    - Optional ESMFold validation of generated structures
    """
    logger.info("Starting genUME structure generation")
    logger.info("Config:\n %s", OmegaConf.to_yaml(cfg))

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Set seed for reproducibility
    if cfg.get("seed"):
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cfg.seed)
        logger.info(f"Set random seed to {cfg.seed}")

    # Load model
    logger.info("Loading genUME model...")
    if hasattr(cfg.model, "ckpt_path") and cfg.model.ckpt_path is not None:
        logger.info(f"Loading model from checkpoint: {cfg.model.ckpt_path}")
        model_cls = hydra.utils.get_class(cfg.model._target_)
        model = model_cls.load_from_checkpoint(cfg.model.ckpt_path)
    else:
        logger.info("Instantiating fresh model (no checkpoint provided)")
        model = hydra.utils.instantiate(cfg.model)

    model.to(device)
    model.eval()
    logger.info("✓ Model loaded successfully")

    # Initialize ESMFold if requested
    plm_fold = None
    if cfg.generation.get("use_esmfold", False):
        logger.info("Loading ESMFold for structure validation...")

        plm_fold = LobsterPLMFold(model_name="esmfold_v1", max_length=cfg.generation.get("max_length", 512))
        plm_fold.to(device)
        logger.info("✓ ESMFold loaded successfully")

    # Create output directory
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Initialize CSV logging and plotting if enabled
    csv_writer = None
    plotter = None
    if cfg.generation.get("save_csv_metrics", True):
        generation_mode = cfg.generation.mode
        csv_writer = MetricsCSVWriter(output_dir, generation_mode)
        logger.info(f"CSV metrics logging enabled for {generation_mode} mode")

        # Initialize plotter if plotting is enabled
        if cfg.generation.get("create_plots", True):
            plotter = MetricsPlotter(output_dir, generation_mode)
            logger.info(f"Plotting enabled for {generation_mode} mode")

    # Generate structures
    generation_mode = cfg.generation.mode
    logger.info(f"Generation mode: {generation_mode}")

    if generation_mode == "unconditional":
        _generate_unconditional(model, cfg, device, output_dir, plm_fold, csv_writer, plotter)
    elif generation_mode == "inverse_folding":
        _generate_inverse_folding(model, cfg, device, output_dir, plm_fold, csv_writer, plotter)
    elif generation_mode == "forward_folding":
        _generate_forward_folding(model, cfg, device, output_dir, plm_fold, csv_writer, plotter)
    elif generation_mode == "inpainting":
        _generate_inpainting(model, cfg, device, output_dir, plm_fold, csv_writer, plotter)
    else:
        raise ValueError(f"Unknown generation mode: {generation_mode}")

    logger.info("Generation completed successfully!")


def _check_sequence_tokens(
    sequences: torch.Tensor, mask: torch.Tensor, stage_name: str = "generation"
) -> tuple[bool, str]:
    """Check if generated sequences contain only valid amino acid tokens.

    Valid tokens are 0-19 (standard 20 amino acids).
    Invalid tokens include:
    - Token 20 (X = unknown amino acid)
    - Tokens > 20 (mask/special tokens)
    - Negative values

    Args:
        sequences: Sequence tensor (B, L) with amino acid token indices
        mask: Validity mask (B, L) indicating which positions are valid
        stage_name: Name of the generation stage for logging

    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if all sequences contain only valid tokens
        - error_message: Description of the issue if invalid, empty string if valid
    """
    batch_size = sequences.shape[0]

    for i in range(batch_size):
        # Get valid positions for this sample
        valid_positions = mask[i] == 1
        seq_i = sequences[i, valid_positions]

        # Check for unknown tokens (token 20 = X)
        num_unknown = (seq_i == 20).sum().item()
        if num_unknown > 0:
            return False, f"Sample {i} in {stage_name} contains {num_unknown} unknown 'X' tokens (token 20)"

        # Check for mask/special tokens (> 20)
        num_mask = (seq_i > 20).sum().item()
        if num_mask > 0:
            return False, f"Sample {i} in {stage_name} contains {num_mask} mask/special tokens (> 20)"

        # Check for negative values (should never happen, but check anyway)
        num_negative = (seq_i < 0).sum().item()
        if num_negative > 0:
            return False, f"Sample {i} in {stage_name} contains negative token values"

    return True, ""


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
    from lobster.transforms._structure_transforms import AminoAcidTokenizerTransform

    try:
        from tmtools import tm_align
    except ImportError as e:
        raise ImportError(
            "tmtools is required. Install with `uv sync --extra struct-gpu` or `uv sync --extra struct-cpu`"
        ) from e

    gen_cfg = cfg.generation

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

        # Initialize metrics collection for this length
        all_metrics = []

        # Get quality control config for retry logic
        qc_config = {}
        if hasattr(gen_cfg, "self_reflection") and hasattr(gen_cfg.self_reflection, "quality_control"):
            qc_config = gen_cfg.self_reflection.quality_control

        # Enable retries if any QC threshold is enabled
        qc_enabled = (
            qc_config.get("enable_tm_threshold", False)
            or qc_config.get("enable_min_percent_identity_threshold", False)
            or qc_config.get("enable_max_percent_identity_threshold", False)
            or qc_config.get("enable_sequence_token_check", True)  # Token check enabled by default
        )
        max_retries = qc_config.get("max_retries", 3) if qc_enabled else 0

        # Track retry statistics
        total_retries = 0
        max_retries_exceeded = 0

        for n_iter in range(n_iterations):
            logger.info(f"Iteration {n_iter + 1}/{n_iterations}")

            # Retry loop for quality control
            retry_count = 0
            iteration_success = False

            while retry_count <= max_retries and not iteration_success:
                if retry_count > 0:
                    logger.info(f"  Retry attempt {retry_count}/{max_retries} for iteration {n_iter + 1}")
                    total_retries += 1

                with torch.no_grad():
                    # Generate samples
                    generate_sample = model.generate_sample(
                        length=current_length,
                        num_samples=batch_size,
                        nsteps=nsteps,
                        temperature_seq=gen_cfg.get("temperature_seq", 0.5),
                        temperature_struc=gen_cfg.get("temperature_struc", 1.0),
                        stochasticity_seq=gen_cfg.get("stochasticity_seq", 20),
                        stochasticity_struc=gen_cfg.get("stochasticity_struc", 20),
                        asynchronous_sampling=gen_cfg.get("asynchronous_sampling", False),
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
                                logger.error(
                                    f"  Max retries ({max_retries}) exceeded for iteration {n_iter + 1}. "
                                    f"Skipping self-reflection for this iteration."
                                )
                                max_retries_exceeded += 1
                                iteration_success = True
                            continue
                    else:
                        # Self-reflection disabled, no quality control
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

                    # Write sequences to CSV
                    # Note: For self-reflection mode, we only store initial unconditional sequences (not forward/inverse intermediates)
                    if csv_writer is not None:
                        # Convert sequences to strings
                        sequence_strs = []
                        for i in range(batch_size):
                            seq_i = seq[i, mask[i] == 1]
                            sequence_str = "".join([restype_order_with_x_inv[j.item()] for j in seq_i])
                            sequence_strs.append(sequence_str)

                        # Write to sequences CSV
                        csv_writer.write_sequences(
                            sequences=sequence_strs,
                            run_id=f"unconditional_length_{current_length}_iter_{n_iter:03d}",
                            iteration=n_iter,
                            sequence_type="unconditional",
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
                        if batch_metrics:
                            logger.info("ESMFold validation metrics for unconditional generation:")
                            for key, value in batch_metrics.items():
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
                    "/homefs/home/lisanzas/scratch/Develop/lobster/src/lobster/metrics/foldseek/bin",
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


def _generate_inverse_folding(
    model, cfg: DictConfig, device: torch.device, output_dir: Path, plm_fold=None, csv_writer=None, plotter=None
) -> None:
    """Generate sequences for given structures (inverse folding)."""
    try:
        from tmtools import tm_align
    except ImportError as e:
        raise ImportError(
            "tmtools is required. Install with `uv sync --extra struct-gpu` or `uv sync --extra struct-cpu`"
        ) from e

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
    elif isinstance(input_structures, (list, tuple)):
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

    logger.info(f"Found {len(structure_paths)} structure files to process")

    gen_cfg = cfg.generation
    nsteps = gen_cfg.get("nsteps", 100)
    batch_size = gen_cfg.get("batch_size", 1)
    n_trials = gen_cfg.get("n_trials", 1)  # Number of trials for best output selection
    n_designs_per_structure = gen_cfg.get("n_designs_per_structure", 1)  # Number of designs to generate per structure

    logger.info(f"Processing structures with {nsteps} generation steps, batch size {batch_size}, n_trials {n_trials}")
    logger.info(f"Generating {n_designs_per_structure} sequence design(s) per structure")

    # Initialize StructureBackboneTransform
    structure_transform = StructureBackboneTransform(max_length=cfg.generation.get("max_length", 512))

    # Initialize aggregate statistics collection
    all_percent_identities = []
    all_plddt_scores = []
    all_predicted_aligned_errors = []
    all_tm_scores = []
    all_rmsd_scores = []

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

            for i, structure_path in enumerate(batch_paths):
                logger.info(f"Loading {structure_path}")

                # Check file extension to determine loading method
                if structure_path.endswith(".pt"):
                    # Load .pt file directly
                    try:
                        structure_data = torch.load(structure_path, map_location="cpu")
                        if structure_data is not None:
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

                # Write sequences to CSV
                if csv_writer is not None:
                    # Convert generated sequences to strings
                    generated_sequence_strs = []
                    for i in range(B):
                        seq_i = seq[i, mask[i] == 1]
                        sequence_str = "".join([restype_order_with_x_inv[j.item()] for j in seq_i])
                        generated_sequence_strs.append(sequence_str)

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

                    # Reuse ESMFold results from the best trial
                    if (
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

                    # Collect metrics for aggregate statistics
                    if batch_metrics:
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


def _generate_forward_folding(
    model, cfg: DictConfig, device: torch.device, output_dir: Path, plm_fold=None, csv_writer=None, plotter=None
) -> None:
    """Generate structures from given input structures (forward folding)."""
    try:
        from tmtools import tm_align
    except ImportError as e:
        raise ImportError(
            "tmtools is required. Install with `uv sync --extra struct-gpu` or `uv sync --extra struct-cpu`"
        ) from e
    logger.info("Starting forward folding generation...")

    # Get input structure paths
    input_structures = cfg.generation.input_structures
    if not input_structures:
        raise ValueError("input_structures must be provided for forward folding mode")

    # Handle different input formats (same as inverse folding)
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
    elif isinstance(input_structures, (list, tuple)):
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

    logger.info(f"Found {len(structure_paths)} structure files to process")

    gen_cfg = cfg.generation
    nsteps = gen_cfg.get("nsteps", 200)  # More steps for forward folding
    batch_size = gen_cfg.get("batch_size", 1)
    n_trials = gen_cfg.get("n_trials", 1)  # Number of trials for best output selection

    # Initialize transforms
    structure_transform = StructureBackboneTransform(max_length=cfg.generation.get("max_length", 512))
    tokenizer_transform = AminoAcidTokenizerTransform(max_length=cfg.generation.get("max_length", 512))

    # Initialize aggregate statistics collection
    all_tm_scores = []
    all_rmsd_scores = []

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

            for i, structure_path in enumerate(batch_paths):
                logger.info(f"Loading {structure_path}")

                # Check file extension to determine loading method
                if structure_path.endswith(".pt"):
                    # Load .pt file directly
                    structure_data = torch.load(structure_path, map_location="cpu")
                    if structure_data is not None:
                        # Apply StructureBackboneTransform
                        structure_data = structure_transform(structure_data)
                        batch_data.append(structure_data)
                        valid_indices.append(i)
                    else:
                        raise ValueError(f"Failed to load structure from {structure_path} - data is None")

                else:
                    # Load PDB/CIF file using existing method
                    structure_data = load_pdb(structure_path, add_batch_dim=False)
                    if structure_data is not None:
                        # Apply StructureBackboneTransform
                        structure_data = structure_transform(structure_data)
                        batch_data.append(structure_data)
                        valid_indices.append(i)
                    else:
                        raise ValueError(f"Failed to load structure from {structure_path}")

            if not batch_data:
                raise ValueError(f"No valid structures in batch {batch_idx + 1}, skipping")

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

            # Extract and tokenize sequences from input structures for forward folding
            input_sequences = []
            for i, data in enumerate(filtered_batch_data):
                if "sequence" in data:
                    seq_tensor = data["sequence"]
                    if seq_tensor.dim() > 1:
                        seq_tensor = seq_tensor.squeeze()

                    # Apply tokenizer transform to the sequence
                    tokenized_data = tokenizer_transform({"sequence": seq_tensor})
                    tokenized_seq = tokenized_data["sequence"]
                    input_sequences.append(tokenized_seq)
                else:
                    raise ValueError(f"No sequence found for structure: {structure_path}")

            # Pad sequences to same length
            padded_sequences = torch.zeros((B, max_length), device=device, dtype=torch.long)
            for i, seq in enumerate(input_sequences):
                seq_len = min(len(seq), max_length)
                padded_sequences[i, :seq_len] = seq[:seq_len]

            # Run multiple trials and select best based on TM-score
            best_trial_results = []

            for trial in range(n_trials):
                logger.info(f"Trial {trial + 1}/{n_trials} for batch {batch_idx + 1}")

                # Generate new structures (forward folding)
                generate_sample = model.generate_sample(
                    length=max_length,
                    num_samples=B,
                    nsteps=nsteps,
                    temperature_seq=gen_cfg.get("temperature_seq", 0.5),
                    temperature_struc=gen_cfg.get("temperature_struc", 1.0),
                    stochasticity_seq=gen_cfg.get("stochasticity_seq", 20),
                    stochasticity_struc=gen_cfg.get("stochasticity_struc", 20),
                    forward_folding=True,
                    input_sequence_tokens=padded_sequences,
                    input_mask=mask,
                    input_indices=indices,
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

                # Calculate TM-scores for this trial
                trial_tm_scores = []
                for i in range(B):
                    # Get original and generated coordinates
                    orig_coords = coords_res[i, mask[i] == 1, :, :]  # Original structure
                    gen_coords = x_recon_xyz[i, mask[i] == 1, :, :]  # Generated structure

                    # Get sequence for TM-align
                    seq_i = seq[i, mask[i] == 1]
                    sequence_str = "".join([restype_order_with_x_inv[j.item()] for j in seq_i])

                    # Calculate TM-Score using TM-align

                    tm_out = tm_align(
                        gen_coords[:, 1, :].cpu().numpy(),  # CA atoms of generated structure
                        orig_coords[:, 1, :].detach().cpu().numpy(),  # CA atoms of original structure
                        sequence_str,
                        sequence_str,
                    )
                    trial_tm_scores.append(tm_out.tm_norm_chain1)
                    logger.info(f"TM-Score: {tm_out.tm_norm_chain1:.3f}, RMSD: {tm_out.rmsd:.2f} Å")

                # Store trial results
                best_trial_results.append(
                    {
                        "trial": trial,
                        "tm_scores": trial_tm_scores,
                        "avg_tm_score": sum(trial_tm_scores) / len(trial_tm_scores),
                        "generate_sample": generate_sample,
                        "x_recon_xyz": x_recon_xyz,
                        "seq": seq,
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

            # Write sequences to CSV
            if csv_writer is not None:
                # Convert generated sequences to strings
                generated_sequence_strs = []
                for i in range(B):
                    seq_i = seq[i, mask[i] == 1]
                    sequence_str = "".join([restype_order_with_x_inv[j.item()] for j in seq_i])
                    generated_sequence_strs.append(sequence_str)

                # Convert original sequences to strings (from input structures)
                original_sequence_strs = []
                for i, data in enumerate(filtered_batch_data):
                    orig_seq = data["sequence"]
                    if orig_seq.dim() > 1:
                        orig_seq = orig_seq.squeeze()
                    orig_seq_str = "".join([restype_order_with_x_inv[j.item()] for j in orig_seq])
                    original_sequence_strs.append(orig_seq_str)

                # Write to sequences CSV
                csv_writer.write_sequences(
                    sequences=generated_sequence_strs,
                    original_sequences=original_sequence_strs,
                    run_id=f"forward_folding_batch_{batch_idx:03d}",
                    input_structure=[Path(batch_paths[i]).stem for i in filtered_valid_indices],
                    trial_number=best_trial["trial"] + 1,
                )

            # Save generated and original structures
            logger.info(f"Saving forward folding results for batch {batch_idx + 1}...")
            for i, valid_idx in enumerate(filtered_valid_indices):
                original_path = batch_paths[valid_idx]
                original_name = Path(original_path).stem
                x_recon_xyz_i_masked = x_recon_xyz[i, mask[i] == 1]
                seq_i_masked = seq[i, mask[i] == 1]

                # Get original structure coordinates and sequence
                orig_coords_i_masked = coords_res[i, mask[i] == 1, :, :]

                # Save generated structure
                generated_filename = output_dir / f"forward_folding_{original_name}_generated.pdb"
                writepdb(str(generated_filename), x_recon_xyz_i_masked, seq_i_masked)
                logger.info(f"Saved generated: {generated_filename}")

                # Save original structure
                original_filename = output_dir / f"forward_folding_{original_name}_original.pdb"
                writepdb(str(original_filename), orig_coords_i_masked, seq_i_masked)
                logger.info(f"Saved original: {original_filename}")

            # Calculate TM-Score and RMSD between generated and original structures
            logger.info(f"Calculating structural metrics for batch {batch_idx + 1}...")
            batch_tm_scores = []
            batch_rmsd_scores = []

            for i, valid_idx in enumerate(filtered_valid_indices):
                # Get original and generated coordinates
                orig_coords = coords_res[i, mask[i] == 1, :, :]  # Original structure
                gen_coords = x_recon_xyz[i, mask[i] == 1, :, :]  # Generated structure

                # Get sequence for TM-align
                seq_i = seq[i, mask[i] == 1]
                sequence_str = "".join([restype_order_with_x_inv[j.item()] for j in seq_i])

                # Calculate TM-Score and RMSD using TM-align

                tm_out = tm_align(
                    gen_coords[:, 1, :].cpu().numpy(),  # CA atoms of generated structure
                    orig_coords[:, 1, :].detach().cpu().numpy(),  # CA atoms of original structure
                    sequence_str,
                    sequence_str,
                )
                logger.info(f"Sequence: {sequence_str}")
                logger.info(f"TM-Score: {tm_out.tm_norm_chain1:.3f}, RMSD: {tm_out.rmsd:.2f} Å")
                batch_tm_scores.append(tm_out.tm_norm_chain1)
                batch_rmsd_scores.append(tm_out.rmsd)

            # Collect metrics for aggregate statistics
            all_tm_scores.extend(batch_tm_scores)
            all_rmsd_scores.extend(batch_rmsd_scores)

            # Write batch metrics to CSV
            if csv_writer is not None:
                run_id = f"forward_folding_batch_{batch_idx:03d}"
                batch_metrics = {
                    "tm_score": sum(batch_tm_scores) / len(batch_tm_scores) if batch_tm_scores else 0.0,
                    "rmsd": sum(batch_rmsd_scores) / len(batch_rmsd_scores) if batch_rmsd_scores else 0.0,
                }
                csv_writer.write_batch_metrics(
                    batch_metrics, run_id, sequence_length=max_length, input_file=f"batch_{batch_idx:03d}"
                )

    # Calculate and report aggregate statistics
    logger.info("=" * 80)
    logger.info("FORWARD FOLDING AGGREGATE STATISTICS")
    logger.info("=" * 80)

    if all_tm_scores:
        avg_tm_score = sum(all_tm_scores) / len(all_tm_scores)
        logger.info(f"Average TM-Score: {avg_tm_score:.3f} (n={len(all_tm_scores)})")
    else:
        logger.warning("No TM-Score data collected")

    # Calculate RMSD pass rate (< 2.0Å threshold)
    rmsd_threshold = 2.0
    rmsd_pass_rates = {}

    if all_rmsd_scores:
        # Filter out infinite RMSD values
        valid_rmsd = [r for r in all_rmsd_scores if r != float("inf")]
        if valid_rmsd:
            avg_rmsd = sum(valid_rmsd) / len(valid_rmsd)
            logger.info(f"Average RMSD: {avg_rmsd:.2f} Å (n={len(valid_rmsd)})")

            pass_count = sum(1 for rmsd in valid_rmsd if rmsd < rmsd_threshold)
            total_count = len(valid_rmsd)
            pass_rate = (pass_count / total_count * 100) if total_count > 0 else 0.0
            rmsd_pass_rates["rmsd"] = (pass_count, total_count, pass_rate)
            logger.info(f"RMSD Pass Rate (< {rmsd_threshold:.1f}Å): {pass_count}/{total_count} ({pass_rate:.1f}%)")
        else:
            logger.warning("No valid RMSD data collected")
    else:
        logger.warning("No RMSD data collected")

    logger.info("=" * 80)

    # Write aggregate statistics to CSV
    if csv_writer is not None:
        logger.info("Writing forward folding aggregate statistics to CSV...")

        # Collect all metric values
        metric_lists = {"tm_score": all_tm_scores, "rmsd": all_rmsd_scores}

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


def _generate_inpainting(
    model, cfg: DictConfig, device: torch.device, output_dir: Path, plm_fold=None, csv_writer=None, plotter=None
) -> None:
    """Generate structures using inpainting (mask and regenerate specific positions)."""
    try:
        from tmtools import tm_align
    except ImportError as e:
        raise ImportError(
            "tmtools is required. Install with `uv sync --extra struct-gpu` or `uv sync --extra struct-cpu`"
        ) from e
    logger.info("Starting inpainting generation...")

    # Get input structure paths
    input_structures = cfg.generation.input_structures
    if not input_structures:
        raise ValueError("input_structures must be provided for inpainting mode")

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
    elif isinstance(input_structures, (list, tuple)):
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

    logger.info(f"Found {len(structure_paths)} structure files to process")

    gen_cfg = cfg.generation
    nsteps = gen_cfg.get("nsteps", 200)
    batch_size = gen_cfg.get("batch_size", 1)
    n_trials = gen_cfg.get("n_trials", 1)  # Number of trials for best output selection
    n_designs_per_structure = gen_cfg.get("n_designs_per_structure", 1)  # Number of designs to generate per structure

    # Get inpainting masks from configuration
    mask_indices_seq = gen_cfg.get("mask_indices_sequence", "")
    mask_indices_struc = gen_cfg.get("mask_indices_structure", "")

    logger.info("Inpainting (joint mode)")
    logger.info(f"Sequence mask indices: {mask_indices_seq if mask_indices_seq else '(no masking)'}")
    logger.info(f"Structure mask indices: {mask_indices_struc if mask_indices_struc else '(no masking)'}")
    logger.info(f"Processing structures with {nsteps} generation steps, batch size {batch_size}, n_trials {n_trials}")
    logger.info(f"Generating {n_designs_per_structure} design(s) per structure")

    # Initialize transforms
    structure_transform = StructureBackboneTransform(max_length=cfg.generation.get("max_length", 512))
    tokenizer_transform = AminoAcidTokenizerTransform(max_length=cfg.generation.get("max_length", 512))

    # Initialize aggregate statistics collection
    all_percent_identities_masked = []
    all_percent_identities_unmasked = []
    all_plddt_scores = []
    all_predicted_aligned_errors = []
    all_tm_scores = []
    all_rmsd_scores = []
    all_rmsd_inpainted = []

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

            for i, structure_path in enumerate(batch_paths):
                logger.info(f"Loading {structure_path}")

                # Check file extension to determine loading method
                if structure_path.endswith(".pt"):
                    # Load .pt file directly
                    try:
                        structure_data = torch.load(structure_path, map_location="cpu")
                        if structure_data is not None:
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

            # Extract and tokenize sequences from input structures
            input_sequences = []
            original_sequences = []
            for i, data in enumerate(filtered_batch_data):
                if "sequence" in data:
                    seq_tensor = data["sequence"]
                    if seq_tensor.dim() > 1:
                        seq_tensor = seq_tensor.squeeze()

                    # Store original sequence for metrics
                    original_sequences.append(seq_tensor)

                    # Apply tokenizer transform to the sequence
                    tokenized_data = tokenizer_transform({"sequence": seq_tensor})
                    tokenized_seq = tokenized_data["sequence"]
                    input_sequences.append(tokenized_seq)
                else:
                    raise ValueError(f"No sequence found for structure: {batch_paths[filtered_valid_indices[i]]}")

            # Pad sequences to same length
            padded_sequences = torch.zeros((B, max_length), device=device, dtype=torch.long)
            for i, seq in enumerate(input_sequences):
                seq_len = min(len(seq), max_length)
                padded_sequences[i, :seq_len] = seq[:seq_len]

            # Parse mask indices and create inpainting masks
            # Handle three types: patterns (e.g., "GGGG"), random specs, or index specs

            # Check if sequence mask is a pattern
            is_pattern_seq = _is_sequence_pattern(mask_indices_seq) if mask_indices_seq else False
            is_random_seq = (
                isinstance(mask_indices_seq, str)
                and "|" in mask_indices_seq
                and mask_indices_seq.strip().lower().startswith("random")
            )

            # Check if structure mask is a pattern
            is_pattern_struc = _is_sequence_pattern(mask_indices_struc) if mask_indices_struc else False
            is_random_struc = (
                isinstance(mask_indices_struc, str)
                and "|" in mask_indices_struc
                and mask_indices_struc.strip().lower().startswith("random")
            )

            # Generate sequence mask
            if is_pattern_seq:
                logger.info(f"Using sequence pattern-based masking: '{mask_indices_seq}'")
                # Use original_sequences (pre-tokenized, raw AA indices)
                inpainting_mask_seq = _create_sequence_pattern_masks(
                    pattern=mask_indices_seq,
                    sequences=original_sequences,
                    max_length=max_length,
                    device=device,
                )
            elif is_random_seq:
                inpainting_mask_seq = parse_mask_indices(mask_indices_seq, max_length, device)
                inpainting_mask_seq = inpainting_mask_seq.expand(B, -1)
            else:
                inpainting_mask_seq = parse_mask_indices(mask_indices_seq, max_length, device)
                inpainting_mask_seq = inpainting_mask_seq.expand(B, -1)

            # Generate structure mask
            if is_pattern_struc:
                logger.info(f"Using structure pattern-based masking: '{mask_indices_struc}'")
                # Use original_sequences (pre-tokenized, raw AA indices)
                inpainting_mask_struc = _create_sequence_pattern_masks(
                    pattern=mask_indices_struc,
                    sequences=original_sequences,
                    max_length=max_length,
                    device=device,
                )
            elif is_random_struc:
                inpainting_mask_struc = parse_mask_indices(mask_indices_struc, max_length, device)
                inpainting_mask_struc = inpainting_mask_struc.expand(B, -1)
            else:
                inpainting_mask_struc = parse_mask_indices(mask_indices_struc, max_length, device)
                inpainting_mask_struc = inpainting_mask_struc.expand(B, -1)

            # Calculate masked positions (may vary per sample for pattern-based masks)
            num_masked_seq = inpainting_mask_seq.sum(dim=1).float().mean().item()
            num_masked_struc = inpainting_mask_struc.sum(dim=1).float().mean().item()

            if num_masked_seq > 0:
                logger.info(
                    f"Sequence inpainting mask: {num_masked_seq:.1f} positions to generate (average per sample)"
                )
            if num_masked_struc > 0:
                logger.info(
                    f"Structure inpainting mask: {num_masked_struc:.1f} positions to generate (average per sample)"
                )

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

                    # Generate with inpainting
                    generate_sample = model.generate_sample(
                        length=max_length,
                        num_samples=B,
                        nsteps=nsteps,
                        temperature_seq=gen_cfg.get("temperature_seq", 0.5),
                        temperature_struc=gen_cfg.get("temperature_struc", 1.0),
                        stochasticity_seq=gen_cfg.get("stochasticity_seq", 20),
                        stochasticity_struc=gen_cfg.get("stochasticity_struc", 20),
                        inpainting=True,
                        input_structure_coords=coords_res,
                        input_sequence_tokens=padded_sequences,
                        input_mask=mask,
                        input_indices=indices,
                        inpainting_mask_sequence=inpainting_mask_seq,
                        inpainting_mask_structure=inpainting_mask_struc,
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

                    # Calculate TM-scores for this trial
                    trial_tm_scores = []
                    trial_rmsd_inpainted = []
                    outputs = None
                    pred_coords = None
                    trial_folded_structure_metrics = None

                    for i in range(B):
                        # Get original and generated coordinates
                        orig_coords = coords_res[i, mask[i] == 1, :, :]  # Original structure
                        gen_coords = x_recon_xyz[i, mask[i] == 1, :, :]  # Generated structure

                        # Get sequence for TM-align
                        seq_i = seq[i, mask[i] == 1]

                        # Get chain information for this structure
                        chains_i = filtered_batch_data[i]["chains"].to(device)[mask[i] == 1]

                        if plm_fold is not None:
                            # Prepare inpainting masks
                            inpaint_mask_seq_i = (
                                inpainting_mask_seq[i, mask[i] == 1] if inpainting_mask_seq is not None else None
                            )
                            inpaint_mask_struc_i = (
                                inpainting_mask_struc[i, mask[i] == 1] if inpainting_mask_struc is not None else None
                            )

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

                            # Warn if using chain-specific groups (coordinates won't be updated)
                            using_chain_subset = len(esmfold_chain_groups) > 1 or sorted(
                                esmfold_chain_groups[0]
                            ) != sorted(available_chains)
                            if using_chain_subset:
                                logger.warning(
                                    "Chain-specific groups detected: Generated coordinates will NOT be updated "
                                    "with ESMFold-aligned predictions (incompatible reference frames). "
                                    "ESMFold metrics are for analysis only."
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

                                # Use refactored ESMFold prediction function with alignment
                                result = predict_structure_with_esmfold(
                                    plm_fold=plm_fold,
                                    seq_i=seq_i,
                                    chains_i=chains_i,
                                    orig_coords=orig_coords,
                                    gen_coords=gen_coords,  # Include generated coords for alignment
                                    mask_i=mask[i],
                                    cfg=cfg,
                                    device=device,
                                    restype_order_inv=restype_order_with_x_inv,
                                    inpainting_mask_seq_i=inpaint_mask_seq_i,
                                    inpainting_mask_struc_i=inpaint_mask_struc_i,
                                    chain_group=chain_group,  # Specify which chains to predict
                                )

                                chain_group_results.append(result)

                                logger.info(
                                    f"Chain group {chain_group}: TM-score: "
                                    f"{result['folded_structure_metrics']['_tm_score']:.3f}, "
                                    f"Inpainted RMSD: {result['rmsd_inpainted']:.3f} Å, "
                                    f"Chains: {result['num_chains']}, Residues: {result['num_residues']}"
                                )

                            # Handle results: use first group as primary, store all
                            if chain_group_results:
                                # Use FIRST chain group as primary result (user controls priority by ordering)
                                primary_result = chain_group_results[0]

                                logger.info(
                                    f"Using first chain group {primary_result['chain_group']} as primary result: "
                                    f"RMSD {primary_result['rmsd_inpainted']:.3f} Å, "
                                    f"TM-score {primary_result['folded_structure_metrics']['_tm_score']:.3f}"
                                )

                                # Log all other results for comparison
                                if len(chain_group_results) > 1:
                                    logger.info("Additional chain group results:")
                                    for idx, result in enumerate(chain_group_results[1:], start=2):
                                        logger.info(
                                            f"  Chain group {idx}/{len(chain_group_results)} {result['chain_group']}: "
                                            f"RMSD {result['rmsd_inpainted']:.3f} Å, "
                                            f"TM-score {result['folded_structure_metrics']['_tm_score']:.3f}, "
                                            f"Chains: {result['num_chains']}, Residues: {result['num_residues']}"
                                        )

                                # Note: When using chain groups, we do NOT update coordinates
                                # because the aligned coords are in a different reference frame
                                # (only the filtered chains). Updating would not make physical sense.
                                # The generated coordinates remain unchanged, and the ESMFold prediction
                                # is used only for metrics and analysis.

                                # Use primary result for output
                                trial_tm_scores.append(primary_result["folded_structure_metrics"]["_tm_score"])
                                trial_rmsd_inpainted.append(primary_result["rmsd_inpainted"])
                                outputs = primary_result["esmfold_outputs"]
                                pred_coords = primary_result["pred_coords"]
                                trial_folded_structure_metrics = primary_result["folded_structure_metrics"]

                                # Store ALL results for later analysis
                                if trial_folded_structure_metrics is not None:
                                    trial_folded_structure_metrics["_all_chain_group_results"] = chain_group_results
                                    trial_folded_structure_metrics["_primary_chain_group"] = primary_result[
                                        "chain_group"
                                    ]
                            else:
                                # Fallback: if all chain groups invalid, use all chains
                                logger.warning("No valid chain groups found, falling back to all chains")
                                result = predict_structure_with_esmfold(
                                    plm_fold=plm_fold,
                                    seq_i=seq_i,
                                    chains_i=chains_i,
                                    orig_coords=orig_coords,
                                    gen_coords=gen_coords,
                                    mask_i=mask[i],
                                    cfg=cfg,
                                    device=device,
                                    restype_order_inv=restype_order_with_x_inv,
                                    inpainting_mask_seq_i=inpaint_mask_seq_i,
                                    inpainting_mask_struc_i=inpaint_mask_struc_i,
                                )

                                # Update coordinates with aligned version
                                x_recon_xyz[i, mask[i] == 1] = result["gen_coords_aligned"]

                                trial_tm_scores.append(result["folded_structure_metrics"]["_tm_score"])
                                trial_rmsd_inpainted.append(result["rmsd_inpainted"])
                                outputs = result["esmfold_outputs"]
                                pred_coords = result["pred_coords"]
                                trial_folded_structure_metrics = result["folded_structure_metrics"]

                                logger.info(
                                    f"TM-score: {result['folded_structure_metrics']['_tm_score']:.3f}, "
                                    f"Inpainted RMSD: {result['rmsd_inpainted']:.3f} Å"
                                )

                        else:
                            # Calculate TM-Score using TM-align
                            # Build sequence string for TM-align
                            sequence_str = build_multichain_sequence_string(seq_i, chains_i, restype_order_with_x_inv)

                            tm_out = tm_align(
                                gen_coords[:, 1, :].cpu().numpy(),  # CA atoms of generated structure
                                orig_coords[:, 1, :].detach().cpu().numpy(),  # CA atoms of original structure
                                sequence_str,
                                sequence_str,
                            )
                            trial_tm_scores.append(tm_out.tm_norm_chain1)
                            trial_rmsd_inpainted.append(0.0)  # No ESMFold, no inpainted RMSD
                            logger.info(f"Sample {i}: TM-Score: {tm_out.tm_norm_chain1:.3f}, RMSD: {tm_out.rmsd:.2f} Å")

                    # Store trial results
                    best_trial_results.append(
                        {
                            "trial": trial,
                            "tm_scores": trial_tm_scores,
                            "rmsd_inpainted": trial_rmsd_inpainted,
                            "avg_tm_score": sum(trial_tm_scores) / len(trial_tm_scores),
                            "avg_rmsd_inpainted": sum(trial_rmsd_inpainted) / len(trial_rmsd_inpainted)
                            if trial_rmsd_inpainted
                            else 0.0,
                            "generate_sample": generate_sample,
                            "x_recon_xyz": x_recon_xyz,
                            "seq": seq,
                            "esmfold_outputs": outputs,
                            "esmfold_pred_coords": pred_coords,
                            "folded_structure_metrics": trial_folded_structure_metrics,
                        }
                    )

                # Select best trial based on average RMSD inpainted
                best_trial = min(best_trial_results, key=lambda x: x["avg_rmsd_inpainted"])
                logger.info(
                    f"Selected trial {best_trial['trial'] + 1} with average RMSD inpainted: {best_trial['avg_rmsd_inpainted']:.3f}"
                )

                # Use best trial results
                generate_sample = best_trial["generate_sample"]
                x_recon_xyz = best_trial["x_recon_xyz"]
                seq = best_trial["seq"]

                # Calculate percent identity for inpainting (compare generated sequence with original)
                batch_percent_identities_masked = []
                batch_percent_identities_unmasked = []

                for i, orig_seq in enumerate(original_sequences):
                    gen_seq = seq[i]
                    actual_mask = mask[i] == 1

                    # Get the actual length
                    orig_len = actual_mask.sum().item()
                    orig_seq_masked = orig_seq[:orig_len].to(device)
                    gen_seq_masked = gen_seq[:orig_len].to(device)

                    # Calculate percent identity for masked positions
                    if inpainting_mask_seq is not None:
                        mask_positions = inpainting_mask_seq[i, :orig_len].bool()
                        if mask_positions.sum() > 0:
                            percent_identity_masked = calculate_percent_identity(
                                orig_seq_masked[mask_positions].unsqueeze(0),
                                gen_seq_masked[mask_positions].unsqueeze(0),
                            )
                            batch_percent_identities_masked.append(percent_identity_masked.item())
                        else:
                            batch_percent_identities_masked.append(0.0)
                    else:
                        batch_percent_identities_masked.append(0.0)

                    # Calculate percent identity for unmasked positions
                    if inpainting_mask_seq is not None:
                        unmask_positions = ~inpainting_mask_seq[i, :orig_len].bool()
                        if unmask_positions.sum() > 0:
                            percent_identity_unmasked = calculate_percent_identity(
                                orig_seq_masked[unmask_positions].unsqueeze(0),
                                gen_seq_masked[unmask_positions].unsqueeze(0),
                            )
                            batch_percent_identities_unmasked.append(percent_identity_unmasked.item())
                        else:
                            batch_percent_identities_unmasked.append(100.0)
                    else:
                        # If no sequence mask, all positions are unmasked
                        percent_identity_unmasked = calculate_percent_identity(
                            orig_seq_masked.unsqueeze(0), gen_seq_masked.unsqueeze(0)
                        )
                        batch_percent_identities_unmasked.append(percent_identity_unmasked.item())

                # Write sequences to CSV
                if csv_writer is not None:
                    # Convert full generated sequences to strings
                    generated_sequence_strs = []
                    for i in range(B):
                        seq_i = seq[i, mask[i] == 1]
                        sequence_str = "".join([restype_order_with_x_inv[j.item()] for j in seq_i])
                        generated_sequence_strs.append(sequence_str)

                    # Convert full original sequences to strings
                    original_sequence_strs = []
                    for orig_seq in original_sequences:
                        orig_seq_str = "".join([restype_order_with_x_inv[j.item()] for j in orig_seq])
                        original_sequence_strs.append(orig_seq_str)

                    # Extract ONLY the inpainted region sequences (generated and original)
                    inpainted_region_strs = []
                    original_inpainted_region_strs = []
                    masked_positions_per_seq = []

                    for i in range(B):
                        if inpainting_mask_seq is not None and inpainting_mask_seq[i].sum() > 0:
                            # Get the masked positions for this sequence
                            seq_i = seq[i, mask[i] == 1]  # Generated sequence
                            seq_len = (mask[i] == 1).sum().item()  # Actual sequence length
                            orig_seq_i = original_sequences[i][:seq_len]  # Original sequence
                            mask_bool = inpainting_mask_seq[i, mask[i] == 1].bool()

                            # Find contiguous regions of masked positions
                            masked_indices = mask_bool.nonzero(as_tuple=True)[0].cpu().tolist()

                            if not masked_indices:
                                inpainted_region_strs.append("")
                                original_inpainted_region_strs.append("")
                                masked_positions_per_seq.append([])
                                continue

                            # Store all masked positions for this sequence
                            masked_positions_per_seq.append(masked_indices)

                            # Split into contiguous regions
                            regions = []
                            current_region = [masked_indices[0]]

                            for idx in masked_indices[1:]:
                                if idx == current_region[-1] + 1:
                                    # Contiguous - add to current region
                                    current_region.append(idx)
                                else:
                                    # Non-contiguous - start new region
                                    regions.append(current_region)
                                    current_region = [idx]
                            regions.append(current_region)  # Add the last region

                            # Extract generated and original sequences for each region
                            generated_region_sequences = []
                            original_region_sequences = []
                            for region_indices in regions:
                                # Generated (inpainted) residues
                                region_residues = seq_i[region_indices]
                                region_seq = "".join([restype_order_with_x_inv[j.item()] for j in region_residues])
                                generated_region_sequences.append(region_seq)

                                # Original residues (before inpainting)
                                original_region_residues = orig_seq_i[region_indices]
                                original_region_seq = "".join(
                                    [restype_order_with_x_inv[j.item()] for j in original_region_residues]
                                )
                                original_region_sequences.append(original_region_seq)

                            # Join multiple regions with comma separator
                            inpainted_region_str = ",".join(generated_region_sequences)
                            original_inpainted_region_str = ",".join(original_region_sequences)
                            inpainted_region_strs.append(inpainted_region_str)
                            original_inpainted_region_strs.append(original_inpainted_region_str)
                        else:
                            # No sequence masking, empty strings
                            inpainted_region_strs.append("")
                            original_inpainted_region_strs.append("")
                            masked_positions_per_seq.append([])

                    # Determine run_id based on whether we are generating multiple designs
                    if n_designs_per_structure > 1:
                        run_id = f"inpainting_batch_{batch_idx:03d}_design_{design_idx:02d}"
                    else:
                        run_id = f"inpainting_batch_{batch_idx:03d}"

                    # Write to sequences CSV
                    csv_writer.write_sequences(
                        sequences=generated_sequence_strs,
                        original_sequences=original_sequence_strs,
                        inpainted_sequences=inpainted_region_strs,
                        original_inpainted_sequences=original_inpainted_region_strs,
                        run_id=run_id,
                        input_structure=[Path(batch_paths[i]).stem for i in filtered_valid_indices],
                        trial_number=best_trial["trial"] + 1,
                        percent_identities=batch_percent_identities_masked,
                        masked_positions=masked_positions_per_seq,
                    )

                # Save results
                logger.info(f"Saving inpainting results for batch {batch_idx + 1}, design {design_idx + 1}...")
                for i, valid_idx in enumerate(filtered_valid_indices):
                    original_path = batch_paths[valid_idx]
                    original_name = Path(original_path).stem
                    x_recon_xyz_i_masked = x_recon_xyz[i, mask[i] == 1]
                    seq_i_masked = seq[i, mask[i] == 1]

                    # Save generated structure with design index
                    if n_designs_per_structure > 1:
                        filename = output_dir / f"inpainting_{original_name}_design_{design_idx:02d}_generated.pdb"
                    else:
                        filename = output_dir / f"inpainting_{original_name}_generated.pdb"
                    writepdb(str(filename), x_recon_xyz_i_masked, seq_i_masked)
                    logger.info(f"Saved: {filename}")

                # Optional ESMFold validation - reuse results from trial selection
                if plm_fold is not None:
                    logger.info(f"Validating batch {batch_idx + 1} with ESMFold...")

                    # Reuse ESMFold results from the best trial
                    if (
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
                                        / f"inpainting_{original_name}_design_{design_idx:02d}_esmfold_chains_{'_'.join(map(str, primary_chain_group))}.pdb"
                                    )
                                else:
                                    filename = (
                                        output_dir
                                        / f"inpainting_{original_name}_esmfold_chains_{'_'.join(map(str, primary_chain_group))}.pdb"
                                    )
                                writepdb(str(filename), pred_coords_i, seq_i_filtered)
                                logger.info(f"Saved ESMFold structure (chains {primary_chain_group}): {filename}")
                            else:
                                # Using all chains - normal masking
                                pred_coords_i_masked = pred_coords[i, mask[i] == 1]
                                seq_i_masked = seq[i, mask[i] == 1]
                                if n_designs_per_structure > 1:
                                    filename = (
                                        output_dir / f"inpainting_{original_name}_design_{design_idx:02d}_esmfold.pdb"
                                    )
                                else:
                                    filename = output_dir / f"inpainting_{original_name}_esmfold.pdb"
                                writepdb(str(filename), pred_coords_i_masked, seq_i_masked)
                                logger.info(f"Saved ESMFold structure: {filename}")

                        batch_metrics = folded_structure_metrics
                    else:
                        # Fallback to full ESMFold validation
                        batch_metrics = _validate_with_esmfold(
                            seq,
                            x_recon_xyz,
                            plm_fold,
                            device,
                            output_dir,
                            f"inpainting_batch{batch_idx:03d}",
                            original_paths=[batch_paths[i] for i in filtered_valid_indices],
                            mask=mask,
                            max_length=max_length,
                        )

                    # Collect metrics for aggregate statistics
                    if batch_metrics:
                        all_plddt_scores.append(batch_metrics["_plddt"])
                        all_predicted_aligned_errors.append(batch_metrics["_predicted_aligned_error"])
                        all_tm_scores.append(batch_metrics["_tm_score"])
                        all_rmsd_scores.append(batch_metrics["_rmsd"])

                        all_percent_identities_masked.extend(batch_percent_identities_masked)
                        all_percent_identities_unmasked.extend(batch_percent_identities_unmasked)

                        # Collect RMSD from inpainted region (from best trial)
                        if "rmsd_inpainted" in best_trial and best_trial["rmsd_inpainted"]:
                            all_rmsd_inpainted.extend(best_trial["rmsd_inpainted"])

                        # Write batch metrics to CSV
                        if csv_writer is not None:
                            if n_designs_per_structure > 1:
                                run_id = f"inpainting_batch_{batch_idx:03d}_design_{design_idx:02d}"
                            else:
                                run_id = f"inpainting_batch_{batch_idx:03d}"
                            avg_percent_identity_masked = sum(batch_percent_identities_masked) / len(
                                batch_percent_identities_masked
                            )
                            avg_percent_identity_unmasked = sum(batch_percent_identities_unmasked) / len(
                                batch_percent_identities_unmasked
                            )

                            num_masked_seq = (
                                inpainting_mask_seq[0].sum().item() if inpainting_mask_seq is not None else 0
                            )
                            num_masked_struc = (
                                inpainting_mask_struc[0].sum().item() if inpainting_mask_struc is not None else 0
                            )

                            # Calculate average RMSD for inpainted region
                            avg_rmsd_inpainted = (
                                best_trial["avg_rmsd_inpainted"] if "avg_rmsd_inpainted" in best_trial else 0.0
                            )

                            csv_writer.write_batch_metrics(
                                batch_metrics,
                                run_id,
                                percent_identity_masked=avg_percent_identity_masked,
                                percent_identity_unmasked=avg_percent_identity_unmasked,
                                rmsd_inpainted=avg_rmsd_inpainted,
                                sequence_length=max_length,
                                num_masked_seq=num_masked_seq,
                                num_masked_struc=num_masked_struc,
                                input_file=f"batch_{batch_idx:03d}",
                            )

    # Calculate and report aggregate statistics
    logger.info("=" * 80)
    logger.info(
        f"INPAINTING ({'joint' if mask_indices_seq or mask_indices_struc else 'no masking'}) AGGREGATE STATISTICS"
    )
    logger.info("=" * 80)

    if all_percent_identities_masked:
        avg_percent_identity_masked = sum(all_percent_identities_masked) / len(all_percent_identities_masked)
        logger.info(
            f"Average Percent Identity (Masked Positions): {avg_percent_identity_masked:.2f}% (n={len(all_percent_identities_masked)})"
        )
    else:
        logger.warning("No masked percent identity data collected")

    if all_percent_identities_unmasked:
        avg_percent_identity_unmasked = sum(all_percent_identities_unmasked) / len(all_percent_identities_unmasked)
        logger.info(
            f"Average Percent Identity (Unmasked Positions): {avg_percent_identity_unmasked:.2f}% (n={len(all_percent_identities_unmasked)})"
        )
    else:
        logger.warning("No unmasked percent identity data collected")

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

    # Calculate RMSD pass rates (< 2.0Å threshold)
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

    if all_rmsd_inpainted:
        avg_rmsd_inpainted = sum(all_rmsd_inpainted) / len(all_rmsd_inpainted)
        logger.info(f"Average RMSD (Inpainted Region): {avg_rmsd_inpainted:.3f} Å (n={len(all_rmsd_inpainted)})")

        pass_count_inpainted = sum(1 for rmsd in all_rmsd_inpainted if rmsd < rmsd_threshold)
        total_count_inpainted = len(all_rmsd_inpainted)
        pass_rate_inpainted = (pass_count_inpainted / total_count_inpainted * 100) if total_count_inpainted > 0 else 0.0
        rmsd_pass_rates["rmsd_inpainted"] = (pass_count_inpainted, total_count_inpainted, pass_rate_inpainted)
        logger.info(
            f"RMSD Pass Rate Inpainted (< {rmsd_threshold:.1f}Å): {pass_count_inpainted}/{total_count_inpainted} ({pass_rate_inpainted:.1f}%)"
        )
    else:
        logger.warning("No inpainted region RMSD data collected")

    logger.info("=" * 80)

    # Write aggregate statistics to CSV
    if csv_writer is not None:
        logger.info("Writing inpainting aggregate statistics to CSV...")

        # Collect all metric values
        metric_lists = {
            "percent_identity_masked": all_percent_identities_masked,
            "percent_identity_unmasked": all_percent_identities_unmasked,
            "rmsd_inpainted": all_rmsd_inpainted,
            "plddt": all_plddt_scores,
            "predicted_aligned_error": all_predicted_aligned_errors,
            "tm_score": all_tm_scores,
            "rmsd": all_rmsd_scores,
        }

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


def _generate_binders(model, cfg: DictConfig, device: torch.device, output_dir: Path, plm_fold=None) -> None:
    """Generate binders."""
    raise NotImplementedError("Binder generation is not implemented")


def _validate_with_esmfold(
    seq: torch.Tensor,
    x_recon_xyz: torch.Tensor,
    plm_fold,
    device: torch.device,
    output_dir: Path,
    prefix: str,
    original_paths: list[str] | None = None,
    mask: torch.Tensor | None = None,
    max_length: int | None = 512,
) -> dict[str, float] | None:
    """Validate generated structures using ESMFold."""
    # Convert sequences to strings
    sequence_str = []
    for i in range(seq.shape[0]):
        if mask is not None:
            # do not include the padded positions in the sequence
            seq_i = seq[i, mask[i] == 1]
            sequence_str.append("".join([restype_order_with_x_inv[j.item()] for j in seq_i]))
        else:
            sequence_str.append("".join([restype_order_with_x_inv[j.item()] for j in seq[i]]))

    # Tokenize sequences
    tokenized_input = plm_fold.tokenizer.batch_encode_plus(
        sequence_str,
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"].to(device)

    # Fold with ESMFold
    with torch.no_grad():
        outputs = plm_fold.model(tokenized_input)

    # Get folding metrics
    folded_structure_metrics, pred_coords = get_folded_structure_metrics(
        outputs, x_recon_xyz, sequence_str, mask=mask, device=device
    )

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
    for i in range(len(sequence_str)):
        if original_paths and i < len(original_paths):
            # Use original filename for inverse folding
            original_name = Path(original_paths[i]).stem
            filename = output_dir / f"{prefix}_{original_name}_esmfold.pdb"
        else:
            # Use generic naming for unconditional generation
            filename = output_dir / f"{prefix}_esmfold_{i:03d}.pdb"
        if mask is not None:
            pred_coords_i_masked = pred_coords[i, mask[i] == 1]
            seq_i_masked = seq[i, mask[i] == 1]
        else:
            pred_coords_i_masked = pred_coords[i]
            seq_i_masked = seq[i]
        writepdb(str(filename), pred_coords_i_masked, seq_i_masked)
        logger.info(f"Saved ESMFold structure: {filename}")

    # Return the metrics for aggregate statistics
    return folded_structure_metrics


if __name__ == "__main__":
    generate()
