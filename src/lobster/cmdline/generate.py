import logging
from pathlib import Path
import glob

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from loguru import logger

from lobster.model.latent_generator.io import writepdb, load_pdb
from lobster.model.latent_generator.utils.residue_constants import (
    convert_lobster_aa_tokenization_to_standard_aa,
    restype_order_with_x_inv,
)
from lobster.callbacks._folding_structure_utils import get_folded_structure_metrics

# Set up logging
logging.basicConfig(level=logging.INFO)
# warning emsfolding does not take into account multiple chains
# todo: add support for multiple chains


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
        from lobster.model import LobsterPLMFold

        plm_fold = LobsterPLMFold(model_name="esmfold_v1", max_length=cfg.generation.get("max_length", 512))
        plm_fold.to(device)
        logger.info("✓ ESMFold loaded successfully")

    # Create output directory
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Generate structures
    generation_mode = cfg.generation.mode
    logger.info(f"Generation mode: {generation_mode}")

    if generation_mode == "unconditional":
        _generate_unconditional(model, cfg, device, output_dir, plm_fold)
    elif generation_mode == "inverse_folding":
        _generate_inverse_folding(model, cfg, device, output_dir, plm_fold)
    else:
        raise ValueError(f"Unknown generation mode: {generation_mode}")

    logger.info("Generation completed successfully!")


def _generate_unconditional(model, cfg: DictConfig, device: torch.device, output_dir: Path, plm_fold=None) -> None:
    """Generate structures unconditionally."""
    logger.info("Starting unconditional generation...")

    gen_cfg = cfg.generation
    length = gen_cfg.length
    num_samples = gen_cfg.num_samples
    nsteps = gen_cfg.get("nsteps", 200)

    logger.info(f"Generating {num_samples} structures of length {length} with {nsteps} steps")

    with torch.no_grad():
        # Generate samples
        generate_sample = model.generate_sample(
            length=length,
            num_samples=num_samples,
            nsteps=nsteps,
            temperature_seq=gen_cfg.get("temperature_seq", 0.5),
            temperature_struc=gen_cfg.get("temperature_struc", 1.0),
            stochasticity_seq=gen_cfg.get("stochasticity_seq", 20),
            stochasticity_struc=gen_cfg.get("stochasticity_struc", 20),
        )

        # Create mask for decoding
        mask = torch.ones((num_samples, length), device=device)

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
            seq = convert_lobster_aa_tokenization_to_standard_aa(generate_sample["sequence_logits"], device=device)
        else:
            seq = generate_sample["sequence_logits"].argmax(dim=-1)
            seq[seq > 21] = 20

        # Save generated structures
        logger.info("Saving generated structures...")
        for i in range(num_samples):
            filename = output_dir / f"generated_structure_{i:03d}.pdb"
            writepdb(str(filename), x_recon_xyz[i], seq[i])
            logger.info(f"Saved: {filename}")

        # Optional ESMFold validation
        if plm_fold is not None:
            logger.info("Validating structures with ESMFold...")
            _validate_with_esmfold(seq, x_recon_xyz, plm_fold, device, output_dir, "generated", max_length=length)


def _generate_inverse_folding(model, cfg: DictConfig, device: torch.device, output_dir: Path, plm_fold=None) -> None:
    """Generate sequences for given structures (inverse folding)."""
    logger.info("Starting inverse folding generation...")

    # Get input structure paths
    input_structures = cfg.generation.input_structures
    if not input_structures:
        raise ValueError("input_structures must be provided for inverse folding mode")

    # Handle different input formats
    pdb_paths = []
    if isinstance(input_structures, str):
        # Single path or glob pattern
        if "*" in input_structures or "?" in input_structures:
            # Glob pattern
            pdb_paths = glob.glob(input_structures)
        else:
            # Single file or directory
            path = Path(input_structures)
            if path.is_file():
                pdb_paths = [str(path)]
            elif path.is_dir():
                # Find all PDB files in directory
                pdb_paths = list(glob.glob(str(path / "*.pdb")))
                pdb_paths.extend(glob.glob(str(path / "*.cif")))
            else:
                raise ValueError(f"Input path does not exist: {input_structures}")
    elif isinstance(input_structures, (list, tuple)):
        # List of paths
        for path_str in input_structures:
            path = Path(path_str)
            if path.is_file():
                pdb_paths.append(str(path))
            else:
                logger.warning(f"Skipping non-existent file: {path_str}")
    else:
        raise ValueError(f"Invalid input_structures format: {type(input_structures)}")

    if not pdb_paths:
        raise ValueError("No valid PDB files found in input_structures")

    logger.info(f"Found {len(pdb_paths)} PDB files to process")

    gen_cfg = cfg.generation
    nsteps = gen_cfg.get("nsteps", 100)
    batch_size = gen_cfg.get("batch_size", 1)

    logger.info(f"Processing structures with {nsteps} generation steps, batch size {batch_size}")

    with torch.no_grad():
        # Process PDB files in batches
        for batch_start in range(0, len(pdb_paths), batch_size):
            batch_end = min(batch_start + batch_size, len(pdb_paths))
            batch_paths = pdb_paths[batch_start:batch_end]
            batch_idx = batch_start // batch_size

            logger.info(f"Processing batch {batch_idx + 1}/{(len(pdb_paths) + batch_size - 1) // batch_size}")

            # Load structures from PDB files
            batch_data = []
            valid_indices = []

            for i, pdb_path in enumerate(batch_paths):
                logger.info(f"Loading {pdb_path}")
                structure_data = load_pdb(pdb_path, add_batch_dim=False)
                if structure_data is not None:
                    batch_data.append(structure_data)
                    valid_indices.append(i)
                else:
                    logger.warning(f"Failed to load structure from {pdb_path}")

            if not batch_data:
                logger.warning(f"No valid structures in batch {batch_idx + 1}, skipping")
                continue

            # Prepare batch tensors
            max_length = max(data["coords_res"].shape[0] for data in batch_data)
            B = len(batch_data)

            # Initialize tensors
            coords_res = torch.zeros((B, max_length, 3, 3), device=device)
            mask = torch.zeros((B, max_length), device=device)
            indices = torch.zeros((B, max_length), device=device, dtype=torch.long)

            # Fill batch tensors
            for i, data in enumerate(batch_data):
                L = data["coords_res"].shape[0]
                coords_res[i, :L] = data["coords_res"].to(device)
                mask[i, :L] = data["mask"].to(device)
                indices[i, :L] = data["indices"].to(device)

            # Handle NaN coordinates
            nan_indices = torch.isnan(coords_res).any(dim=-1).any(dim=-1)
            mask[nan_indices] = 0
            coords_res[nan_indices] = 0

            logger.info(f"Batch {batch_idx + 1}: {B} structures, max length {max_length}")

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
                seq = convert_lobster_aa_tokenization_to_standard_aa(generate_sample["sequence_logits"], device=device)
            else:
                seq = generate_sample["sequence_logits"].argmax(dim=-1)
                seq[seq > 21] = 20

            # Save results
            logger.info(f"Saving inverse folding results for batch {batch_idx + 1}...")
            for i, valid_idx in enumerate(valid_indices):
                original_path = batch_paths[valid_idx]
                original_name = Path(original_path).stem
                x_recon_xyz_i_masked = x_recon_xyz[i, mask[i] == 1]
                seq_i_masked = seq[i, mask[i] == 1]

                # Save generated structure
                filename = output_dir / f"inverse_folding_{original_name}_generated.pdb"
                writepdb(str(filename), x_recon_xyz_i_masked, seq_i_masked)
                logger.info(f"Saved: {filename}")

            # Optional ESMFold validation
            if plm_fold is not None:
                logger.info(f"Validating batch {batch_idx + 1} with ESMFold...")
                _validate_with_esmfold(
                    seq,
                    x_recon_xyz,
                    plm_fold,
                    device,
                    output_dir,
                    f"inverse_folding_batch{batch_idx:03d}",
                    original_paths=[batch_paths[i] for i in valid_indices],
                    mask=mask,
                    max_length=max_length,
                )


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
) -> None:
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
    folded_structure_metrics, pred_coords = get_folded_structure_metrics(outputs, x_recon_xyz, sequence_str, mask=mask)

    # Log metrics
    logger.info("ESMFold validation metrics:")
    for key, value in folded_structure_metrics.items():
        logger.info(f"  {key}: {value:.4f}")

    # Save folded structures
    for i in range(len(sequence_str)):
        if original_paths and i < len(original_paths):
            # Use original filename for inverse folding
            original_name = Path(original_paths[i]).stem
            filename = output_dir / f"inverse_folding_{original_name}_esmfold.pdb"
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


if __name__ == "__main__":
    generate()
