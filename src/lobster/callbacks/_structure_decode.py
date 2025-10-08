import lightning
import os
import torch
from lobster.model.latent_generator.io import writepdb
from lobster.model.latent_generator.utils.residue_constants import convert_lobster_aa_tokenization_to_standard_aa
from loguru import logger


class StructureDecodeCallback(lightning.Callback):
    def __init__(self, structure_path: str = None, save_every_n: int = 1000):
        self.structure_path = structure_path
        self.save_every_n = save_every_n
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not os.path.exists(f"{self.structure_path}/decode"):
            os.makedirs(f"{self.structure_path}/decode", exist_ok=True)

    def on_train_batch_end(self, trainer, laten_mlm, outputs, batch, batch_idx):
        current_step = trainer.global_step
        seq = None
        x_recon_xyz = None

        if batch_idx % self.save_every_n == 0:
            x_recon = outputs["decoded_x"]
            if "train_t" in outputs:
                t = outputs["train_t"]
                t = t[0].cpu().numpy()
            else:
                t_seq = outputs["train_timesteps_seq"]
                t_struc = outputs["train_timesteps_struc"]
                t_seq = t_seq[0].cpu().numpy()
                t_struc = t_struc[0].cpu().numpy()
                t = None
            conditioning = outputs["conditioning"]

            x_recon_xyz = None

            for decoder_name in x_recon:
                if "vit_decoder" == decoder_name:
                    x_recon_xyz = x_recon[decoder_name]

            # save the pdb file
            if x_recon_xyz is not None:
                if outputs["unmasked_x"]["sequence_logits"].shape[-1] == 33:
                    seq = convert_lobster_aa_tokenization_to_standard_aa(
                        outputs["unmasked_x"]["sequence_logits"], device=self.device
                    )
                else:
                    seq = outputs["unmasked_x"]["sequence_logits"].argmax(dim=-1)
                    seq[seq > 21] = 20
                if t is not None:
                    filename = f"{self.structure_path}decode/struc_{batch_idx}_{current_step}_t{str(t)}_cond{conditioning}_decode.pdb"
                else:
                    filename = f"{self.structure_path}decode/struc_{batch_idx}_{current_step}_tseq_{str(t_seq)}_tstruc_{str(t_struc)}_cond{conditioning}_decode.pdb"
                writepdb(filename, x_recon_xyz[0], seq[0])
                logger.info(f"Saved {filename}")

                # save batch
                if t is not None:
                    filename = f"{self.structure_path}decode/struc_{batch_idx}_{current_step}_t{str(t)}_cond{conditioning}_gt.pdb"
                else:
                    filename = f"{self.structure_path}decode/struc_{batch_idx}_{current_step}_tseq_{str(t_seq)}_tstruc_{str(t_struc)}_cond{conditioning}_gt.pdb"
                seq = batch["sequence"][0]
                # if naything >21, set to 20
                seq[seq > 21] = 20
                writepdb(filename, batch["coords_res"][0], seq)
                logger.info(f"Saved {filename}")
