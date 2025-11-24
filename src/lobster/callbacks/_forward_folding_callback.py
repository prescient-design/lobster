import lightning
import os
import torch
import glob
from loguru import logger

from lobster.model.latent_generator.io import writepdb
from lobster.model.latent_generator.utils.residue_constants import (
    convert_lobster_aa_tokenization_to_standard_aa,
    restype_order_with_x_inv,
)
from lobster.metrics import align_and_compute_rmsd
from lobster.transforms._structure_transforms import StructureBackboneTransform, AminoAcidTokenizerTransform
from tmtools import tm_align
import tqdm


class ForwardFoldingCallback(lightning.Callback):
    """Callback for evaluating forward folding (sequence → structure) on CAMEO dataset during training.

    This callback:
    - Loads CAMEO validation structures
    - Extracts ground truth sequences
    - Generates structures from sequences using the model
    - Compares generated vs ground truth structures
    - Logs TM-score and RMSD metrics to WandB
    """

    def __init__(
        self,
        structure_path: str = None,
        cameo_data_path: str = "/data2/lisanzas/AFDB/valid_cameo_processed",
        save_every_n: int = 1000,
        num_samples: int = 127,
        max_length: int = 512,
        nsteps: int = 200,
        temperature_seq: float = 0.3610371899835548,
        temperature_struc: float = 0.2195534567490864,
        stochasticity_seq: int = 1,
        stochasticity_struc: int = 20,
        cache_dir: str | None = None,
    ):
        """Initialize forward folding callback.

        Args:
            structure_path: Directory to save generated structures
            cameo_data_path: Path to CAMEO dataset directory (glob pattern supported)
            save_every_n: Evaluate every N training steps
            num_samples: Number of samples to evaluate per callback
            max_length: Maximum sequence length to process
            nsteps: Number of diffusion steps for generation
            temperature_seq: Temperature for sequence sampling
            temperature_struc: Temperature for structure sampling
            stochasticity_seq: Stochasticity parameter for sequence
            stochasticity_struc: Stochasticity parameter for structure
            cache_dir: Cache directory for datasets
        """
        self.structure_path = structure_path
        self.cameo_data_path = cameo_data_path
        self.save_every_n = save_every_n
        self.num_samples = num_samples
        self.max_length = max_length
        self.nsteps = nsteps
        self.temperature_seq = temperature_seq
        self.temperature_struc = temperature_struc
        self.stochasticity_seq = stochasticity_seq
        self.stochasticity_struc = stochasticity_struc
        self.cache_dir = cache_dir
        self.cameo_structures = None
        self.tokenizer_transform = None
        self.structure_transform = None

        if self.structure_path and not os.path.exists(f"{self.structure_path}/forward_folding"):
            os.makedirs(f"{self.structure_path}/forward_folding", exist_ok=True)

    def _load_cameo_structures(self):
        """Load CAMEO .pt files directly.

        Loads preprocessed CAMEO structures from .pt files directly,
        similar to how generate.py handles them.
        """
        # Use glob to find all .pt files
        if "*" in self.cameo_data_path:
            dataset_paths = glob.glob(self.cameo_data_path)
        else:
            dataset_paths = glob.glob(os.path.join(self.cameo_data_path, "*.pt"))

        if not dataset_paths:
            raise ValueError(f"No .pt files found in {self.cameo_data_path}")

        logger.info(f"Found {len(dataset_paths)} CAMEO dataset files")

        # Load structures from first few files (limit to avoid memory issues)
        max_files = self.num_samples
        dataset_paths = sorted(dataset_paths)[:max_files]

        structures = []
        for pt_path in dataset_paths:
            logger.info(f"Loading structures from {pt_path}")
            structure_data = torch.load(pt_path, map_location="cpu")

            # Apply StructureBackboneTransform to ensure consistent format
            structure_data = self.structure_transform(structure_data)

            # Filter by minimum length and valid sequences
            if structure_data["coords_res"].shape[0] >= 30:
                percent_unknown = (structure_data["sequence"] == 20).sum().float() / structure_data["sequence"].shape[0]
                if percent_unknown <= 0.1:  # Less than 10% unknown
                    structures.append(structure_data)

            # Limit total structures for efficiency
            if len(structures) >= self.num_samples * 3:  # Load 3x more than needed
                break

        logger.info(f"Loaded {len(structures)} valid structures from CAMEO dataset")
        return structures

    def setup(self, trainer, pl_module, stage):
        """Setup method to load CAMEO structures for evaluation."""
        # Only setup on rank 0 (CUDA device 0) in multinode/multi-GPU settings
        if trainer.global_rank != 0:
            return

        # Initialize transforms
        self.structure_transform = StructureBackboneTransform(max_length=self.max_length)
        self.tokenizer_transform = AminoAcidTokenizerTransform(max_length=self.max_length)

        # Load CAMEO structures directly
        self.cameo_structures = self._load_cameo_structures()

        logger.info(f"Loaded {len(self.cameo_structures)} CAMEO structures for evaluation")

    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
        """Called at the end of each training batch."""
        # Only run on rank 0 (CUDA device 0) in multinode/multi-GPU settings
        if trainer.global_rank != 0:
            return

        current_step = trainer.global_step
        device = batch["sequence"].device

        if batch_idx % self.save_every_n == 0 and self.cameo_structures is not None:
            # Perform forward folding on CAMEO validation examples
            with torch.no_grad():
                self._perform_forward_folding(trainer, model, device, batch_idx, current_step)
            torch.cuda.empty_cache()

    def _perform_forward_folding(self, trainer, model, device, batch_idx, current_step):
        """Perform forward folding on CAMEO validation examples.

        Process:
        1. Load ground truth sequences and structures from CAMEO
        2. Tokenize sequences and prepare inputs
        3. Generate structures using model.generate_sample(forward_folding=True)
        4. Compare generated structures to ground truth
        5. Calculate TM-score and RMSD
        6. Save structures and log metrics
        """
        # Initialize lists to accumulate metrics
        all_tm_scores = []
        all_rmsd_scores = []

        # Process CAMEO structures in batches
        batch_size = 10
        num_structures = min(len(self.cameo_structures), self.num_samples)

        logger.info(f"Evaluating forward folding on {num_structures} CAMEO structures")

        for batch_start in tqdm.tqdm(
            range(0, num_structures, batch_size),
            desc="Forward Folding Evaluation",
        ):
            batch_end = min(batch_start + batch_size, num_structures)
            batch_structures = self.cameo_structures[batch_start:batch_end]

            # Prepare batch tensors
            max_length = max(s["coords_res"].shape[0] for s in batch_structures)
            B = len(batch_structures)

            sequence = torch.zeros((B, max_length), dtype=torch.long, device=device)
            coords_res = torch.zeros((B, max_length, 3, 3), device=device)
            mask = torch.zeros((B, max_length), device=device)
            indices = torch.zeros((B, max_length), dtype=torch.long, device=device)

            # Fill batch tensors
            for i, structure in enumerate(batch_structures):
                L = structure["coords_res"].shape[0]
                sequence[i, :L] = structure["sequence"].to(device)
                coords_res[i, :L] = structure["coords_res"].to(device)
                mask[i, :L] = structure["mask"].to(device)
                indices[i, :L] = structure["indices"].to(device)

            mask_orig = mask.clone()

            # Handle NaN coordinates
            nan_indices = torch.isnan(coords_res).any(dim=-1).any(dim=-1)
            mask[nan_indices] = 0
            coords_res[nan_indices] = 0

            # Tokenize sequences for forward folding
            # Note: sequence is already in standard format from dataset
            # Apply tokenizer transform to convert to model input format
            tokenized_sequences = torch.zeros((B, max_length), device=device, dtype=torch.long)
            for i in range(B):
                seq_i = sequence[i, mask[i] == 1]
                # Apply tokenizer transform
                tokenized_data = self.tokenizer_transform({"sequence": seq_i.cpu()})
                tokenized_seq = tokenized_data["sequence"].to(device)
                seq_len = min(len(tokenized_seq), max_length)
                tokenized_sequences[i, :seq_len] = tokenized_seq[:seq_len]

            # Generate structures from sequences (forward folding)
            logger.info(f"Generating structures for batch (batch {batch_start}-{batch_end}, {B} samples)...")
            generate_sample = model.generate_sample(
                length=max_length,
                num_samples=B,
                forward_folding=True,
                nsteps=self.nsteps,
                temperature_seq=self.temperature_seq,
                temperature_struc=self.temperature_struc,
                stochasticity_seq=self.stochasticity_seq,
                stochasticity_struc=self.stochasticity_struc,
                input_sequence_tokens=tokenized_sequences,
                input_mask=mask,
                input_indices=indices,
            )

            # Decode structures
            decoded_x = model.decode_structure(generate_sample, mask)

            # Extract coordinates from vit_decoder
            x_recon_xyz = None
            for decoder_name in decoded_x:
                if "vit_decoder" == decoder_name:
                    x_recon_xyz = decoded_x[decoder_name]
                    break

            if x_recon_xyz is None:
                logger.error("No vit_decoder found in decoded structures")
                continue

            # Extract sequences (should match input, but check)
            if generate_sample["sequence_logits"].shape[-1] == 33:
                seq = convert_lobster_aa_tokenization_to_standard_aa(generate_sample["sequence_logits"], device=device)
            else:
                seq = generate_sample["sequence_logits"].argmax(dim=-1)
                seq[seq > 21] = 20

            # Only save structures for the first batch
            if batch_start == 0:
                # Save generated and ground truth structures
                for i in range(min(5, B)):  # Save first 5 samples
                    seq_i = seq[i, mask_orig[i] == 1]

                    # Save generated structure
                    filename_gen = (
                        f"{self.structure_path}/forward_folding/struc_{batch_idx}_{current_step}_{i}_generated.pdb"
                    )
                    writepdb(filename_gen, x_recon_xyz[i], seq_i)
                    logger.info(f"Saved generated: {filename_gen}")

                    # Save ground truth structure
                    filename_gt = (
                        f"{self.structure_path}/forward_folding/struc_{batch_idx}_{current_step}_{i}_ground_truth.pdb"
                    )
                    writepdb(filename_gt, coords_res[i], seq_i)
                    logger.info(f"Saved ground truth: {filename_gt}")

            # Calculate TM-score and RMSD for all samples in batch
            logger.info("Calculating structural metrics...")
            batch_tm_scores = []
            batch_rmsd_scores = []

            for i in range(B):
                # Get ground truth and generated coordinates (masked)
                gt_coords = coords_res[i, mask_orig[i] == 1, :, :]  # Ground truth
                gen_coords = x_recon_xyz[i, mask_orig[i] == 1, :, :]  # Generated

                # Get sequence string for TM-align
                seq_i = seq[i, mask_orig[i] == 1]
                sequence_str = "".join([restype_order_with_x_inv[j.item()] for j in seq_i])

                # Calculate TM-Score using TM-align (CA atoms only)
                tm_out = tm_align(
                    gen_coords[:, 1, :].cpu().numpy(),  # CA atoms of generated structure
                    gt_coords[:, 1, :].detach().cpu().numpy(),  # CA atoms of ground truth
                    sequence_str,
                    sequence_str,
                )
                batch_tm_scores.append(tm_out.tm_norm_chain1)

                # Calculate RMSD using Kabsch alignment (all backbone atoms)
                rmsd = align_and_compute_rmsd(
                    coords1=gen_coords,
                    coords2=gt_coords,
                    mask=None,  # Use all positions
                    return_aligned=False,
                    device=device,
                )
                batch_rmsd_scores.append(rmsd)

            # Accumulate metrics
            all_tm_scores.extend(batch_tm_scores)
            all_rmsd_scores.extend(batch_rmsd_scores)

            logger.info(
                f"Batch {batch_start}-{batch_end}: Avg TM-score={sum(batch_tm_scores) / len(batch_tm_scores):.3f}, "
                f"Avg RMSD={sum(batch_rmsd_scores) / len(batch_rmsd_scores):.2f} Å"
            )

        # Calculate averaged metrics across all validation batches
        if not all_tm_scores:
            logger.warning("No metrics collected for forward folding callback")
            return

        avg_tm_score = sum(all_tm_scores) / len(all_tm_scores)
        avg_rmsd = sum(all_rmsd_scores) / len(all_rmsd_scores)

        # Log averaged metrics to WandB
        metrics_to_log = {
            "forward_folding/tm_score": avg_tm_score,
            "forward_folding/rmsd": avg_rmsd,
            "forward_folding/num_samples": len(all_tm_scores),
        }
        model.log_dict(metrics_to_log, batch_size=1)

        logger.info(f"Forward Folding Validation Results (step {current_step}):")
        logger.info(f"  Average TM-score: {avg_tm_score:.3f} (n={len(all_tm_scores)})")
        logger.info(f"  Average RMSD: {avg_rmsd:.2f} Å")
