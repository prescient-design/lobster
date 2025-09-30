import lightning
import os
import torch
from lobster.model.latent_generator.io import writepdb
from loguru import logger
from lobster.model.latent_generator.utils import convert_lobster_aa_tokenization_to_standard_aa
# TODO add folding with esmfold


class UnconditionalGenerationCallback(lightning.Callback):
    def __init__(
        self,
        structure_path: str = None,
        save_every_n: int = 1000,
        length: int = 100,
        num_samples: int = 10,
    ):
        self.STRUCTURE_PATH = structure_path
        self.save_every_n = save_every_n
        self.length = length
        self.num_samples = num_samples
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not os.path.exists(f"{self.STRUCTURE_PATH}/unconditional"):
            os.makedirs(f"{self.STRUCTURE_PATH}/unconditional", exist_ok=True)

    def on_train_batch_end(self, trainer, gen_ume, outputs, batch, batch_idx):
        current_step = trainer.global_step

        if batch_idx % self.save_every_n == 0:
            generate_sample = gen_ume.generate_sample(length=self.length, num_samples=self.num_samples)
            mask = torch.ones((self.num_samples, self.length), device=self.device)
            decoded_x = gen_ume.decode_structure(generate_sample, mask)

            for decoder_name in decoded_x:
                if "vit_decoder" == decoder_name:
                    x_recon_xyz = decoded_x[decoder_name]
            if generate_sample["sequence_logits"].shape[-1] == 33:
                seq = convert_lobster_aa_tokenization_to_standard_aa(
                    generate_sample["sequence_logits"], device=self.device
                )
            else:
                seq = generate_sample["sequence_logits"].argmax(dim=-1)
                seq[seq > 21] = 20
            for i in range(10):
                filename = f"{self.STRUCTURE_PATH}unconditional/struc_{batch_idx}_{current_step}_{i}_unconditional.pdb"
                writepdb(filename, x_recon_xyz[i], seq[i])
                logger.info(f"Saved {filename}")
