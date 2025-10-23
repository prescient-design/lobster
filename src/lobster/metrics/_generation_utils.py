import torch
from tmtools import tm_align


def calculate_percent_identity(
    ground_truth_seq: torch.Tensor, generated_seq: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Calculate percent identity between ground truth and generated sequences.

    Parameters
    ----------
    ground_truth_seq : torch.Tensor
        Ground truth sequence tensor of shape (B, L).
    generated_seq : torch.Tensor
        Generated sequence tensor of shape (B, L).
    mask : torch.Tensor, optional
        Optional mask tensor of shape (B, L) to ignore padded positions. Default is None.

    Returns
    -------
    torch.Tensor
        Tensor of percent identities for each sequence in the batch.
    """
    # Ensure both sequences have the same shape
    assert ground_truth_seq.shape == generated_seq.shape, "Sequences must have the same shape"

    # Calculate matches
    matches = (ground_truth_seq == generated_seq).float()

    if mask is not None:
        # Only consider positions where mask is 1
        matches = matches * mask.float()
        valid_positions = mask.sum(dim=1).float()
        # Avoid division by zero
        valid_positions = torch.clamp(valid_positions, min=1.0)
        percent_identity = (matches.sum(dim=1) / valid_positions) * 100.0
    else:
        # Consider all positions
        sequence_length = ground_truth_seq.shape[1]
        percent_identity = (matches.sum(dim=1) / sequence_length) * 100.0

    return percent_identity


def get_folded_structure_metrics(outputs, ref_coords, ref_seq, prefix="", mask=None):
    """Get the metrics of the folded structure.

    Parameters
    ----------
    outputs : dict
        The outputs of the ESMFold model.
    ref_coords : torch.Tensor
        The reference coordinates of the structure. Shape [B, L, 3, 3].
    ref_seq : list of str
        The reference sequence list of strings.
    prefix : str, optional
        Optional prefix for the returned metric keys. Default is "".
    mask : torch.Tensor, optional
        The mask of the structure. Shape [B, L]. Default is None.

    Returns
    -------
    dict
        Dictionary containing the following keys:
        - plddt: The average pLDDT scores of the batch
        - predicted_aligned_error: The average predicted aligned error of the batch
        - tm_score: The average TM-score of the predicted structure vs the reference structure of the batch
        - rmsd: The average RMSD of the predicted structure vs the reference structure of the batch
    torch.Tensor
        The predicted coordinates of the structure. Shape [B, L, 3, 3].
    """
    pred_coords = outputs["positions"][-1][:, :, :3, :]  # [B, L, 3, 3]
    plddt_scores = outputs["plddt"].mean(dim=(-1, -2))  # [B]
    predicted_aligned_error = outputs["predicted_aligned_error"].mean(dim=(-1, -2))  # [B]
    tm_score = []
    rmsd = []
    for i in range(pred_coords.shape[0]):
        if mask is not None:
            pred_coords_i = pred_coords[i, mask[i] == 1, :, :]
            ref_coords_i = ref_coords[i, mask[i] == 1, :, :]
            # get correct index for string
            ref_seq_i = ref_seq[i]
            ref_seq_i = "".join([ref_seq_i[j] for j in range(len(ref_seq_i)) if mask[i][j] == 1])

        else:
            pred_coords_i = pred_coords[i, :, :, :]
            ref_coords_i = ref_coords[i, :, :, :]
            ref_seq_i = ref_seq[i]
        tm_out = tm_align(
            pred_coords_i[:, 1, :].cpu().numpy(), ref_coords_i[:, 1, :].detach().cpu().numpy(), ref_seq_i, ref_seq_i
        )
        tm_score.append(tm_out.tm_norm_chain1)
        rmsd.append(tm_out.rmsd)
    tm_score = torch.tensor(tm_score).to(pred_coords.device)
    rmsd = torch.tensor(rmsd).to(pred_coords.device)

    # set masked coords to 0
    if mask is not None:
        pred_coords[mask == 0] = 0

    return {
        f"{prefix}_plddt": plddt_scores.mean(),
        f"{prefix}_predicted_aligned_error": predicted_aligned_error.mean(),
        f"{prefix}_tm_score": tm_score.mean(),
        f"{prefix}_rmsd": rmsd.mean(),
    }, pred_coords
