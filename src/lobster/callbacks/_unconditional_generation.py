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
from tmtools import tm_align


def get_folded_structure_metrics(outputs, ref_coords, ref_seq):
    """
    Get the metrics of the folded structure.
    Args:
        outputs: The outputs of the ESMFold model.
        ref_coords: The reference coordinates of the structure. [B, L, 3, 3]
        ref_seq: The reference sequence list of strings.
    Returns:
        Dictionary containing the following keys:
            plddt: The  average pLDDT scores of the batch
            predicted_aligned_error: The average predicted aligned error of the batch
            tm_score: The average TM-score of the predicted structure vs the reference structure of the batch
            rmsd: The average RMSD of the predicted structure vs the reference structure of the batch
        pred_coords: The predicted coordinates of the structure [B, L, 3, 3]

    """
    pred_coords = outputs["positions"][-1][:, :, :3, :]  # [B, L, 3, 3]
    plddt_scores = outputs["plddt"].mean(dim=(-1, -2))  # [B]
    predicted_aligned_error = outputs["predicted_aligned_error"].mean(dim=(-1, -2))  # [B]
    tm_score = []
    rmsd = []
    for i in range(pred_coords.shape[0]):
        tm_out = tm_align(
            pred_coords[i, :, 1, :].cpu().numpy(), ref_coords[i, :, 1, :].detach().cpu().numpy(), ref_seq[i], ref_seq[i]
        )
        tm_score.append(tm_out.tm_norm_chain1)
        rmsd.append(tm_out.rmsd)
    tm_score = torch.tensor(tm_score).to(pred_coords.device)
    rmsd = torch.tensor(rmsd).to(pred_coords.device)
    return {
        "plddt": plddt_scores.mean(),
        "predicted_aligned_error": predicted_aligned_error.mean(),
        "tm_score": tm_score.mean(),
        "rmsd": rmsd.mean(),
    }, pred_coords


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
        self.plm_fold = LobsterPLMFold(model_name="esmfold_v1", max_length=length)
        if not os.path.exists(f"{self.structure_path}/unconditional"):
            os.makedirs(f"{self.structure_path}/unconditional", exist_ok=True)

    def on_train_batch_end(self, trainer, gen_ume, outputs, batch, batch_idx):
        current_step = trainer.global_step
        device = batch["sequence"].device
        self.plm_fold.to(device)

        if batch_idx % self.save_every_n == 0:
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
                filename = f"{self.structure_path}unconditional/struc_{batch_idx}_{current_step}_{i}_unconditional.pdb"
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
            folded_structure_metrics, pred_coords = get_folded_structure_metrics(outputs, x_recon_xyz, sequence_str)
            total_loss = 0.0
            # save the folded structure
            for i in range(10):
                filename = (
                    f"{self.structure_path}unconditional/struc_{batch_idx}_{current_step}_{i}_unconditional_folded.pdb"
                )
                writepdb(filename, pred_coords[i], seq[i])
                logger.info(f"Saved {filename}")

            gen_ume.log_dict({"uncoditional_loss": total_loss, **folded_structure_metrics}, batch_size=mask.shape[0])
