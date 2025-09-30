import lightning
import os
import torch
from lobster.model.latent_generator.io import writepdb
from loguru import logger
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
                from lobster.tokenization._amino_acid import AA_VOCAB
                from lobster.model.latent_generator.utils import residue_constants

                # custom tokenized sequence need to convert to standard amino acid tokenization
                seq_argmax = generate_sample["sequence_logits"].argmax(dim=-1)
                # multi_word tokens: 0, 1, 2, 3, 31, 32 change to 30
                seq_argmax[seq_argmax == 0] = 30
                seq_argmax[seq_argmax == 1] = 30
                seq_argmax[seq_argmax == 2] = 30
                seq_argmax[seq_argmax == 3] = 30
                seq_argmax[seq_argmax == 31] = 30
                seq_argmax[seq_argmax == 32] = 30
                AA_VOCAB_INV = {v: k for k, v in AA_VOCAB.items()}
                standard_seq_tokens = []
                for i in range(seq_argmax.shape[0]):
                    aa_string = "".join([AA_VOCAB_INV[j.item()] for j in seq_argmax[i]])
                    aa_standard = [residue_constants.restype_order_with_x.get(j, 20) for j in aa_string]
                    aa_standard = torch.tensor(aa_standard, device=self.device)
                    standard_seq_tokens.append(aa_standard)
                seq = torch.stack(standard_seq_tokens, dim=0)
                seq[seq > 21] = 20
            else:
                seq = generate_sample["sequence_logits"].argmax(dim=-1)
                seq[seq > 21] = 20
            for i in range(10):
                filename = f"{self.STRUCTURE_PATH}unconditional/struc_{batch_idx}_{current_step}_{i}_unconditional.pdb"
                writepdb(filename, x_recon_xyz[i], seq[i])
                logger.info(f"Saved {filename}")
