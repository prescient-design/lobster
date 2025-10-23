import lightning
import os
import torch
from lobster.model.latent_generator.io import writepdb
from loguru import logger
from lobster.model.latent_generator.utils.residue_constants import (
    convert_lobster_aa_tokenization_to_standard_aa,
    restype_order_with_x_inv,
)
from lobster.model import LobsterPLMFold
from lobster.metrics import get_folded_structure_metrics


class UnconditionalGenerationCallback(lightning.Callback):
    def __init__(
        self,
        structure_path: str = None,
        save_every_n: int = 1000,
        length: int = 100,
        num_samples: int = 10,
    ):
        self.structure_path = structure_path
        self.save_every_n = save_every_n
        self.length = length
        self.num_samples = num_samples
        self.plm_fold = None  # Initialize as None, will be loaded in setup
        if not os.path.exists(f"{self.structure_path}/unconditional"):
            os.makedirs(f"{self.structure_path}/unconditional", exist_ok=True)

    def setup(self, trainer, pl_module, stage):
        """Setup method to initialize PLM fold model only on rank 0."""
        # Only setup on rank 0 (CUDA device 0) in multinode/multi-GPU settings
        if trainer.global_rank != 0:
            return

        if self.plm_fold is None:
            self.plm_fold = LobsterPLMFold(model_name="esmfold_v1", max_length=self.length)
            logger.info("Loaded ESMFold model for unconditional generation evaluation")

    def on_train_batch_end(self, trainer, gen_ume, outputs, batch, batch_idx):
        # Only run on rank 0 (CUDA device 0) in multinode/multi-GPU settings
        if trainer.global_rank != 0:
            return

        current_step = trainer.global_step
        device = batch["sequence"].device

        if self.plm_fold is not None:
            self.plm_fold.to(device)

        if batch_idx % self.save_every_n == 0 and self.plm_fold is not None:
            # Perform unconditional generation
            self._perform_unconditional_generation(trainer, gen_ume, device, batch_idx, current_step)

    def _perform_unconditional_generation(self, trainer, gen_ume, device, batch_idx, current_step):
        """Perform unconditional generation and folding."""
        generate_sample = gen_ume.generate_sample(length=self.length, num_samples=self.num_samples)
        mask = torch.ones((self.num_samples, self.length), device=device)
        decoded_x = gen_ume.decode_structure(generate_sample, mask)

        for decoder_name in decoded_x:
            if "vit_decoder" == decoder_name:
                x_recon_xyz = decoded_x[decoder_name]
        if generate_sample["sequence_logits"].shape[-1] == 33:
            seq = convert_lobster_aa_tokenization_to_standard_aa(generate_sample["sequence_logits"], device=device)
        else:
            seq = generate_sample["sequence_logits"].argmax(dim=-1)
            seq[seq > 21] = 20
        # save the generated structure
        for i in range(10):
            filename = f"{self.structure_path}/unconditional/struc_{batch_idx}_{current_step}_{i}_unconditional.pdb"
            writepdb(filename, x_recon_xyz[i], seq[i])
            logger.info(f"Saved {filename}")

        # folding with ESMFold
        sequence_str = []
        for i in range(seq.shape[0]):
            sequence_str.append("".join([restype_order_with_x_inv[j.item()] for j in seq[i]]))

        tokenized_input = self.plm_fold.tokenizer.batch_encode_plus(
            sequence_str,
            padding=True,
            truncation=True,
            max_length=self.length,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].to(device)
        with torch.no_grad():
            outputs = self.plm_fold.model(tokenized_input)
        folded_structure_metrics, pred_coords = get_folded_structure_metrics(
            outputs, x_recon_xyz, sequence_str, prefix="unconditional"
        )
        total_loss = 0.0
        # save the folded structure
        for i in range(10):
            filename = (
                f"{self.structure_path}/unconditional/struc_{batch_idx}_{current_step}_{i}_unconditional_folded.pdb"
            )
            writepdb(filename, pred_coords[i], seq[i])
            logger.info(f"Saved {filename}")

        gen_ume.log_dict({"uncoditional_loss": total_loss, **folded_structure_metrics}, batch_size=mask.shape[0])
