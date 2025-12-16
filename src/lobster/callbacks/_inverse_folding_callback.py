import lightning
import os
import torch
from lobster.model.latent_generator.io import writepdb
from loguru import logger
from lobster.model.latent_generator.utils.residue_constants import (
    convert_lobster_aa_tokenization_to_standard_aa,
    restype_order_with_x_inv,
)
from lobster.metrics import get_folded_structure_metrics, calculate_percent_identity
from lobster.data._coord_structure_datamodule import StructureLightningDataModule
import tqdm
from lobster.model import LobsterPLMFold
from torch.utils.data import DataLoader
import pooch


class InverseFoldingCallback(lightning.Callback):
    def __init__(
        self,
        structure_path: str = None,
        save_every_n: int = 1000,
        length: int = 100,
        num_samples: int = 10,
        use_plm_fold: bool = True,
        max_length: int = 512,
        use_hf_datasets: bool = True,
        cache_dir: str | None = None,
    ):
        self.structure_path = structure_path
        self.save_every_n = save_every_n
        self.length = length
        self.num_samples = num_samples
        self.use_plm_fold = use_plm_fold
        self.val_dataset = None
        self.val_dataloader = None
        self.plm_fold = None
        self.max_length = max_length
        self.eval_datamodule = None
        self.use_hf_datasets = use_hf_datasets
        self.cache_dir = cache_dir

        if not os.path.exists(f"{self.structure_path}/inverse_folding"):
            os.makedirs(f"{self.structure_path}/inverse_folding", exist_ok=True)

    def _download_cath_datasets(self) -> list[str]:
        """Download CATH datasets from Hugging Face using pooch.

        Returns
        -------
        list[str]
            List of paths to the downloaded dataset files.
        """
        base_url = "https://huggingface.co/datasets/Sidney-Lisanza/cath-v4-3-structures/resolve/main/"

        splits = ["train", "val", "test"]
        downloaded_paths = []

        cache_path = self.cache_dir or pooch.os_cache("lobster")

        for split in splits:
            file_path = f"{split}/cath_{split}.pt"
            url = f"{base_url}{file_path}"

            logger.info(f"Downloading {split} dataset from Hugging Face...")
            fname = pooch.retrieve(
                url=url,
                known_hash=None,  # Can add SHA256 hash for verification if needed
                path=cache_path,
                fname=file_path,
            )
            downloaded_paths.append(fname)
            logger.info(f"Downloaded {split} dataset to {fname}")

        return downloaded_paths

    def _create_eval_datamodule(self):
        """Create a separate datamodule for evaluation."""
        if self.use_hf_datasets:
            # Download from Hugging Face
            logger.info("Using Hugging Face datasets for CATH data")
            dataset_paths = self._download_cath_datasets()
            root_dir = self.cache_dir or pooch.os_cache("lobster")
        else:
            # Use local paths
            logger.info("Using local datasets for CATH data")
            dataset_paths = [
                "/data2/lisanzas/CATH_v4_3/processed_structures_pt/train/cath_train.pt",
                "/data2/lisanzas/CATH_v4_3/processed_structures_pt/val/cath_val.pt",
                "/data2/lisanzas/CATH_v4_3/processed_structures_pt/test/cath_test.pt",
            ]
            root_dir = "/data2/lisanzas/CATH_v4_3/temp/"

        # Directly instantiate the datamodule without Hydra
        self.eval_datamodule = StructureLightningDataModule(
            path_to_datasets=dataset_paths,
            root=root_dir,
            cluster_file=None,
            files_to_keep=None,
            batch_size=10,
            num_workers=1,
            testing=False,
            use_shards=False,
            shuffle=False,
        )
        self.eval_datamodule.setup(stage="fit")
        logger.info("Created evaluation datamodule for CATH validation data")

    def setup(self, trainer, pl_module, stage):
        """Setup method to initialize PLM fold model and load validation examples."""
        # Only setup on rank 0 (CUDA device 0) in multinode/multi-GPU settings
        if trainer.global_rank != 0:
            return

        if self.use_plm_fold and self.plm_fold is None:
            self.plm_fold = LobsterPLMFold(model_name="esmfold_v1", max_length=self.max_length)
            logger.info("Loaded ESMFold model for inverse folding evaluation")

        # Create separate evaluation datamodule
        self._create_eval_datamodule()
        # Get validation dataset directly (avoid trainer dependency in val_dataloader)
        self.val_dataset = self.eval_datamodule._val_dataset

        # Create our own dataloader to avoid trainer state issues
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=20,
            shuffle=False,
            num_workers=0,  # Use 0 to avoid multiprocessing issues in callbacks
            collate_fn=self.eval_datamodule._collate_fn,
        )

        logger.info(f"Created validation dataloader with {len(self.val_dataset)} examples")

    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
        # Only run on rank 0 (CUDA device 0) in multinode/multi-GPU settings
        if trainer.global_rank != 0:
            return

        current_step = trainer.global_step
        device = batch["sequence"].device

        if self.use_plm_fold and self.plm_fold is not None:
            self.plm_fold.to(device)

        if batch_idx % self.save_every_n == 0 and self.val_dataloader is not None:
            # Perform inverse folding on validation examples
            self._perform_inverse_folding(trainer, model, device, batch_idx, current_step)

    def _perform_inverse_folding(self, trainer, model, device, batch_idx, current_step):
        """Perform inverse folding on validation examples."""
        # Initialize lists to accumulate metrics across all validation batches
        all_percent_identities = []
        all_folded_structure_metrics = []

        for val_batch_idx, val_example in enumerate(tqdm.tqdm(self.val_dataloader)):
            B, L = val_example["mask"].shape
            sequence = val_example["sequence"].to(device)
            mask = val_example["mask"].to(device)
            mask_orig = mask.clone()
            indices = val_example["indices"].to(device)
            coords_res = val_example["coords_res"].to(device)
            # Find the nans in coords_res and set the mask to 0 for those indices
            nan_indices = torch.isnan(coords_res).any(dim=-1).any(dim=-1)
            mask[nan_indices] = 0
            # set the coords_res to 0 for those indices
            coords_res[nan_indices] = 0

            generate_sample = model.generate_sample(
                length=L,
                num_samples=B,
                inverse_folding=True,
                nsteps=100,
                input_structure_coords=coords_res,
                input_mask=mask,
                input_indices=indices,
            )
            decoded_x = model.decode_structure(generate_sample, mask)

            for decoder_name in decoded_x:
                if "vit_decoder" == decoder_name:
                    x_recon_xyz = decoded_x[decoder_name]
            if generate_sample["sequence_logits"].shape[-1] == 33:
                seq = convert_lobster_aa_tokenization_to_standard_aa(generate_sample["sequence_logits"], device=device)
            else:
                seq = generate_sample["sequence_logits"].argmax(dim=-1)
                seq[seq > 21] = 20
            # Only save structures for the first validation batch
            if val_batch_idx == 0:
                # save the generated structure
                for i in range(min(10, B)):  # Use min to handle cases where B < 10
                    filename = f"{self.structure_path}/inverse_folding/struc_{batch_idx}_{current_step}_{i}_inverse_folding.pdb"
                    writepdb(filename, x_recon_xyz[i], seq[i])
                    logger.info(f"Saved {filename}")

            # Only perform ESMFold folding for the first validation batch for speed
            if val_batch_idx == 0 and self.plm_fold is not None:
                # folding with ESMFold
                sequence_str = []
                for i in range(seq.shape[0]):
                    seq_i = seq[i, mask_orig[i] == 1]
                    sequence_str.append("".join([restype_order_with_x_inv[j.item()] for j in seq_i]))

                tokenized_input = self.plm_fold.tokenizer.batch_encode_plus(
                    sequence_str,
                    padding=True,
                    truncation=True,
                    max_length=max(len(seq_i) for seq_i in sequence_str),
                    add_special_tokens=False,
                    return_tensors="pt",
                )["input_ids"].to(device)
                with torch.no_grad():
                    outputs = self.plm_fold.model(tokenized_input)
                folded_structure_metrics, pred_coords = get_folded_structure_metrics(
                    outputs, x_recon_xyz, sequence_str, prefix="inverse_folding", mask=mask_orig
                )
            else:
                # Skip ESMFold for subsequent batches
                folded_structure_metrics = {}
                pred_coords = None

            # Calculate percent identity between ground truth and generated sequences
            percent_identities = calculate_percent_identity(sequence, seq, mask)

            # Accumulate metrics for averaging across the entire validation set
            all_percent_identities.extend(percent_identities.cpu().tolist())

            # Only accumulate folded structure metrics and save folded structures for the first validation batch
            if val_batch_idx == 0 and folded_structure_metrics:
                all_folded_structure_metrics.append(folded_structure_metrics)

                # save the folded structure
                for i in range(min(10, B)):  # Use min to handle cases where B < 10
                    filename = f"{self.structure_path}/inverse_folding/struc_{batch_idx}_{current_step}_{i}_inverse_folding_folded.pdb"
                    writepdb(filename, pred_coords[i], seq[i])
                    logger.info(f"Saved {filename}")

        # Calculate averaged metrics across the entire validation set
        avg_percent_identity = sum(all_percent_identities) / len(all_percent_identities)

        # Average the folded structure metrics
        avg_folded_metrics = {}
        if all_folded_structure_metrics:
            # Get all metric keys from the first batch
            metric_keys = all_folded_structure_metrics[0].keys()
            for key in metric_keys:
                values = [metrics[key] for metrics in all_folded_structure_metrics if key in metrics]
                avg_folded_metrics[key] = sum(values) / len(values)

        # Log averaged metrics
        total_loss = 0.0
        metrics_to_log = {
            "inverse_folding_loss": total_loss,
            "sequence_percent_identity": avg_percent_identity,
            **avg_folded_metrics,
        }
        model.log_dict(metrics_to_log, batch_size=1)  # Use batch_size=1 since we're logging aggregated metrics

        logger.info(f"Validation metrics averaged over {len(all_folded_structure_metrics)} batches:")
        logger.info(f"Average sequence percent identity: {avg_percent_identity:.2f}%")
        for key, value in avg_folded_metrics.items():
            logger.info(f"Average {key}: {value:.4f}")
