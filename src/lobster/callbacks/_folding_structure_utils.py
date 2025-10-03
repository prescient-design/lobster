import torch
from tmtools import tm_align


def get_folded_structure_metrics(outputs, ref_coords, ref_seq, prefix=""):
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
        f"{prefix}_plddt": plddt_scores.mean(),
        f"{prefix}_predicted_aligned_error": predicted_aligned_error.mean(),
        f"{prefix}_tm_score": tm_score.mean(),
        f"{prefix}_rmsd": rmsd.mean(),
    }, pred_coords
