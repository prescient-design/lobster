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
        if batch_idx % self.save_every_n != 0:
            return

        current_step = trainer.global_step
        seq = None
        x_recon_xyz = None

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
                vit_output = x_recon[decoder_name]
                # Handle both old format (tensor) and new format (dict with protein_coords/ligand_coords)
                if isinstance(vit_output, dict):
                    x_recon_xyz = vit_output.get("protein_coords")
                else:
                    x_recon_xyz = vit_output

        # save the pdb file
        if x_recon_xyz is not None:
            if outputs["unmasked_x"]["sequence_logits"].shape[-1] == 33:
                seq = convert_lobster_aa_tokenization_to_standard_aa(
                    outputs["unmasked_x"]["sequence_logits"], device=self.device
                )
            else:
                seq = outputs["unmasked_x"]["sequence_logits"].argmax(dim=-1)
                seq[seq > 21] = 20
            # Skip if seq has incorrect shape (needs to be at least 2D: batch x seq_len)
            if seq.dim() < 2:
                logger.warning(f"Skipping structure decode save: seq has unexpected shape {seq.shape}")
                return

            if t is not None:
                filename = f"{self.structure_path}decode/struc_{batch_idx}_{current_step}_t{str(t)}_cond{conditioning}_decode.pdb"
            else:
                filename = f"{self.structure_path}decode/struc_{batch_idx}_{current_step}_tseq_{str(t_seq)}_tstruc_{str(t_struc)}_cond{conditioning}_decode.pdb"
            writepdb(filename, x_recon_xyz[0], seq[0])
            logger.info(f"Saved {filename}")

            # save batch
            if t is not None:
                filename = (
                    f"{self.structure_path}decode/struc_{batch_idx}_{current_step}_t{str(t)}_cond{conditioning}_gt.pdb"
                )
            else:
                filename = f"{self.structure_path}decode/struc_{batch_idx}_{current_step}_tseq_{str(t_seq)}_tstruc_{str(t_struc)}_cond{conditioning}_gt.pdb"
            seq = batch["sequence"][0]
            # if anything >21, set to 20
            seq[seq > 21] = 20
            writepdb(filename, batch["coords_res"][0], seq)
            logger.info(f"Saved {filename}")
