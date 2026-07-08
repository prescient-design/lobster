import lightning
import os
import json
import subprocess
import tempfile
import torch
import glob
from loguru import logger

from lobster.model.latent_generator.io import writepdb
from lobster.model.latent_generator.utils.residue_constants import (
    convert_lobster_aa_tokenization_to_standard_aa,
    restype_order_with_x_inv,
)
from lobster.metrics import align_and_compute_rmsd
from lobster.transforms._structure_transforms import (
    AminoAcidTokenizerTransform,
    Atom14ToBackboneTransform,
    StructureBackboneTransform,
    StructureComplexTransform,
)
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
        batch_size: int = 10,
        temperature_seq: float = 0.3610371899835548,
        temperature_struc: float = 0.2195534567490864,
        stochasticity_seq: int = 1,
        stochasticity_struc: int = 20,
        cache_dir: str | None = None,
        # Complex-aware extensions (Pinder dimer val sets):
        use_chain_ids: bool = False,
        use_epitope_conditioning: bool = False,
        metric_prefix: str | None = None,
        max_length_filter: int | None = None,
        min_chains: int = 1,
        # DockQ interface-quality scoring (complex eval only). Runs in an
        # isolated venv (DockQ pins numpy<2) via subprocess, so it never
        # touches the training env. Only fires for multi-chain structures.
        use_dockq: bool = False,
        dockq_python: str | None = None,
        dockq_script: str | None = None,
    ):
        """Initialize forward folding callback.

        Args:
            structure_path: Directory to save generated structures
            cameo_data_path: Path to CAMEO dataset directory (glob pattern supported)
            save_every_n: Evaluate every N training steps
            num_samples: Number of samples to evaluate per callback
            max_length: Maximum sequence length to process
            nsteps: Number of diffusion steps for generation
            batch_size: Structures per eval batch (reduce for protein-ligand to avoid OOM)
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
        self.eval_batch_size = batch_size
        self.temperature_seq = temperature_seq
        self.temperature_struc = temperature_struc
        self.stochasticity_seq = stochasticity_seq
        self.stochasticity_struc = stochasticity_struc
        self.cache_dir = cache_dir
        # Complex-aware extensions. Default-off keeps single-chain CAMEO
        # callers (the historical caller) bit-identical to the prior
        # behaviour.
        #
        # use_chain_ids: build per-residue chain_ids tensor from each
        #   structure's `chains` field (remapped via the
        #   `RemapChainIdsForEmbedding` convention: chain_idx_0 -> 1,
        #   chain_idx_1 -> 2, monomers -> 0 / dead embedding) and pass
        #   it to ``generate_sample(chain_ids=...)``. Required for any
        #   dimer-aware forward folding on a model trained with
        #   `max_num_chains > 0`.
        #
        # use_epitope_conditioning: use each structure's `epitope_tensor`
        #   (populated by `StructureComplexTransform`) as the
        #   `conditioning_tensor_override` for `generate_sample`. Pair
        #   with `use_chain_ids=True` and a Pinder val set when the
        #   training pipeline learns epitope conditioning.
        #
        # metric_prefix: namespace the W&B keys to disambiguate multiple
        #   forward-folding callbacks in the same run (e.g.
        #   "forward_folding_pinder_cond" vs ..._uncond).
        #
        # max_length_filter / min_chains: post-load filters applied to
        #   structures. Useful for restricting a complex-callback to
        #   "dimers with L <= 512 only" so the eval respects the same
        #   crop boundary as training.
        self.use_chain_ids = use_chain_ids
        self.use_epitope_conditioning = use_epitope_conditioning
        self.metric_prefix = metric_prefix or "forward_folding"
        self.max_length_filter = max_length_filter
        self.min_chains = min_chains
        self.use_dockq = use_dockq
        self.dockq_python = dockq_python or os.environ.get(
            "DOCKQ_PYTHON", "/cv/scratch/u/lisanzas/.dockq_venv/bin/python"
        )
        self.dockq_script = dockq_script or os.environ.get(
            "DOCKQ_SCRIPT", "/cv/home/lisanzas/lobster/scripts/_dockq_pair.py"
        )
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

            # Pinder pt files are atom14; convert to backbone-3 first.
            if "atom14_coords" in structure_data:
                structure_data = self._atom14_to_backbone(structure_data)
            # Apply the structure transform (StructureBackbone or
            # StructureComplex per init).
            structure_data = self.structure_transform(structure_data)

            # Length filter (e.g. dimers <= 512 for in-distribution eval).
            L = structure_data["coords_res"].shape[0]
            if L < 30:
                continue
            if self.max_length_filter is not None and L > self.max_length_filter:
                continue
            # Chain count filter (e.g. dimers only).
            chains_t = structure_data.get("chains")
            if chains_t is not None and self.min_chains > 1:
                if int(torch.unique(chains_t).numel()) < self.min_chains:
                    continue
            # Sequence quality filter (drop structures with > 10% UNK).
            percent_unknown = (structure_data["sequence"] == 20).sum().float() / structure_data["sequence"].shape[0]
            if percent_unknown > 0.1:
                continue
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

        # Initialize transforms. When chain_ids or epitope conditioning is
        # requested we MUST use `StructureComplexTransform` (with the
        # interface-anchored crop) so each batch carries the per-residue
        # `chains` + `epitope_tensor` keys that downstream needs. For the
        # historical monomer-CAMEO callback we keep the lighter
        # `StructureBackboneTransform` to avoid changing baseline runs.
        if self.use_chain_ids or self.use_epitope_conditioning:
            self.structure_transform = StructureComplexTransform(
                max_length=self.max_length, crop_strategy="interface_anchored"
            )
        else:
            self.structure_transform = StructureBackboneTransform(max_length=self.max_length)
        # Apply atom14 -> backbone before the transforms (Pinder is stored
        # as atom14; CAMEO is stored as backbone-3). Putting this in front
        # is a no-op for backbone-3 inputs.
        self._atom14_to_backbone = Atom14ToBackboneTransform()
        self.tokenizer_transform = AminoAcidTokenizerTransform(max_length=self.max_length)

        # Load structures directly
        self.cameo_structures = self._load_cameo_structures()

        logger.info(
            f"Loaded {len(self.cameo_structures)} structures for evaluation "
            f"(prefix={self.metric_prefix} chain_ids={self.use_chain_ids} "
            f"epitope={self.use_epitope_conditioning})"
        )

    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
        """Called at the end of each training batch."""
        current_step = trainer.global_step

        # Check if this is a step where forward folding should run
        # Note: cameo_structures is only loaded on rank 0, so we use batch_idx check for all ranks
        is_forward_folding_step = batch_idx % self.save_every_n == 0

        # All ranks must synchronize BEFORE rank 0 does forward folding
        # This prevents NCCL timeouts from internal collectives in generate_sample
        if is_forward_folding_step and trainer.world_size > 1:
            torch.distributed.barrier()

        # Only rank 0 actually runs the callback (it has the loaded structures)
        if is_forward_folding_step and trainer.global_rank == 0 and self.cameo_structures is not None:
            # Get device from model or from any available tensor in batch
            device = next(model.parameters()).device

            # Perform forward folding on CAMEO validation examples
            # Use model.module to avoid DDP wrapper during inference
            with torch.no_grad():
                unwrapped_model = model.module if hasattr(model, "module") else model
                self._perform_forward_folding(trainer, unwrapped_model, device, batch_idx, current_step)
            torch.cuda.empty_cache()

        # Final barrier to ensure all ranks are synced before continuing training
        if is_forward_folding_step and trainer.world_size > 1:
            torch.distributed.barrier()

    def _dockq_pair(self, gen_coords, gt_coords, seq_i, chains_i):
        """Score a predicted complex interface against native with DockQ.

        Writes pred + native 2-chain PDBs (chains from ``chains_i``) to a temp
        dir and runs the DockQ scorer in its isolated venv via subprocess.
        Returns a metrics dict (DockQ/fnat/irmsd/lrmsd) or None on failure.
        """
        try:
            with tempfile.TemporaryDirectory() as td:
                mp = os.path.join(td, "model.pdb")
                npdb = os.path.join(td, "native.pdb")
                writepdb(mp, gen_coords, seq_i, chains=chains_i)
                writepdb(npdb, gt_coords, seq_i, chains=chains_i)
                out = subprocess.run(
                    [self.dockq_python, self.dockq_script, mp, npdb],
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                lines = out.stdout.strip().splitlines()
                line = lines[-1] if lines else ""
                d = json.loads(line) if line.startswith("{") else {}
                if "DockQ" in d:
                    return d
                logger.warning(f"DockQ no score: {d.get('error', out.stderr[-200:])}")
                return None
        except Exception as e:  # noqa: BLE001
            logger.warning(f"DockQ failed: {e}")
            return None

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
        all_dockq, all_fnat, all_irmsd, all_lrmsd = [], [], [], []

        # Process CAMEO structures in batches
        batch_size = self.eval_batch_size
        num_structures = min(len(self.cameo_structures), self.num_samples)

        logger.info(f"Evaluating forward folding on {num_structures} CAMEO structures (batch_size={batch_size})")

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
            chain_ids = torch.zeros((B, max_length), dtype=torch.long, device=device)
            cond_tensor = torch.zeros((B, max_length, 1), device=device)
            any_chain_signal = False
            any_cond_signal = False

            # Fill batch tensors
            for i, structure in enumerate(batch_structures):
                L = structure["coords_res"].shape[0]
                sequence[i, :L] = structure["sequence"].to(device)
                coords_res[i, :L] = structure["coords_res"].to(device)
                mask[i, :L] = structure["mask"].to(device)
                indices[i, :L] = structure["indices"].to(device)
                # Complex-aware: build per-residue chain-id signal for the
                # encoder's chain_embedding layer. Mirrors
                # `RemapChainIdsForEmbedding` (chain_idx_0 -> 1,
                # chain_idx_1 -> 2; monomers stay 0 / dead embedding).
                if self.use_chain_ids and "chains" in structure:
                    raw_chains = structure["chains"]
                    uniques = sorted(torch.unique(raw_chains).tolist())[:2]
                    for cls, raw in enumerate(uniques, start=1):
                        chain_ids[i, :L][raw_chains == raw] = cls
                    if len(uniques) > 1:
                        any_chain_signal = True
                # Complex-aware: epitope conditioning (1 on epitope residues,
                # 0 elsewhere). Falls back to all-zeros for samples without
                # an epitope_tensor (monomers, or dimers cropped to a single
                # chain by the transform).
                if self.use_epitope_conditioning and "epitope_tensor" in structure:
                    ep = structure["epitope_tensor"].float().to(device)
                    cond_tensor[i, :L, 0] = ep
                    if ep.any():
                        any_cond_signal = True

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

            # Generate structures from sequences (forward folding). Pass
            # chain_ids / conditioning_tensor_override only when the
            # callback was configured for complex-aware eval AND at least
            # one sample in the batch carries the corresponding signal.
            # Passing all-zeros is equivalent to passing None (both leave
            # the respective embedding rows dead) but explicit None keeps
            # the model's autograph cleaner.
            logger.info(f"Generating structures for batch (batch {batch_start}-{batch_end}, {B} samples)...")
            chain_ids_arg = chain_ids if (self.use_chain_ids and any_chain_signal) else None
            cond_arg = cond_tensor if (self.use_epitope_conditioning and any_cond_signal) else None
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
                chain_ids=chain_ids_arg,
                conditioning_tensor_override=cond_arg,
            )

            # Decode structures
            decoded_x = model.decode_structure(generate_sample, mask)

            # Extract coordinates from vit_decoder
            x_recon_xyz = None
            for decoder_name in decoded_x:
                if "vit_decoder" == decoder_name:
                    vit_output = decoded_x[decoder_name]
                    # Handle both tensor output (protein-only) and dict output (protein-ligand)
                    if isinstance(vit_output, dict):
                        x_recon_xyz = vit_output.get("protein_coords")
                    else:
                        x_recon_xyz = vit_output
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

            # Only save structures for the first batch, and only when a real
            # structure_path is set. structure_path=None means metrics-only
            # (compute + log TM/RMSD, skip PDB dumps): without this guard the
            # save path formats to the literal string "None/forward_folding/..."
            # and writepdb crashes with FileNotFoundError.
            if self.structure_path is not None and batch_start == 0:
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

                # DockQ interface-quality score (multi-chain complexes only).
                if self.use_dockq:
                    ch_i = chain_ids[i, mask_orig[i] == 1]
                    if int(ch_i.unique().numel()) >= 2:
                        dq = self._dockq_pair(gen_coords, gt_coords, seq_i, ch_i)
                        if dq is not None:
                            all_dockq.append(dq["DockQ"])
                            all_fnat.append(dq["fnat"])
                            all_irmsd.append(dq["irmsd"])
                            all_lrmsd.append(dq["lrmsd"])

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

        # Log averaged metrics to WandB. Namespace by `metric_prefix` so
        # multiple forward-folding callbacks in the same training run
        # (e.g. monomer CAMEO + complex Pinder conditional / unconditional)
        # log to distinct keys.
        prefix = self.metric_prefix.rstrip("/") if self.metric_prefix else "forward_folding"
        metrics_to_log = {
            f"{prefix}/tm_score": avg_tm_score,
            f"{prefix}/rmsd": avg_rmsd,
            f"{prefix}/num_samples": len(all_tm_scores),
        }
        # DockQ interface metrics (only when complexes were scored).
        if all_dockq:
            metrics_to_log.update(
                {
                    f"{prefix}/dockq": sum(all_dockq) / len(all_dockq),
                    f"{prefix}/dockq_fnat": sum(all_fnat) / len(all_fnat),
                    f"{prefix}/dockq_irmsd": sum(all_irmsd) / len(all_irmsd),
                    f"{prefix}/dockq_lrmsd": sum(all_lrmsd) / len(all_lrmsd),
                    f"{prefix}/dockq_acceptable_frac": sum(1 for q in all_dockq if q >= 0.23) / len(all_dockq),
                    f"{prefix}/dockq_num": len(all_dockq),
                }
            )
        model.log_dict(metrics_to_log, batch_size=1)

        logger.info(f"Forward Folding Validation Results (step {current_step}):")
        logger.info(f"  Average TM-score: {avg_tm_score:.3f} (n={len(all_tm_scores)})")
        logger.info(f"  Average RMSD: {avg_rmsd:.2f} Å")
        if all_dockq:
            logger.info(
                f"  Average DockQ: {sum(all_dockq) / len(all_dockq):.3f} "
                f"(n={len(all_dockq)}, acceptable≥0.23: "
                f"{100 * sum(1 for q in all_dockq if q >= 0.23) / len(all_dockq):.0f}%)"
            )
        # Free GPU memory before returning to training (reduces OOM risk)
        torch.cuda.empty_cache()
