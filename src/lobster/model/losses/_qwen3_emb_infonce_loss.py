"""
References
----------
Qwen3-Embedding Technical Report
    https://arxiv.org/pdf/2506.05176
    https://github.com/QwenLM/Qwen3-Embedding
"""

import torch
import torch.nn.functional as F
from torch import nn


class Qwen3ContrastiveLoss(nn.Module):
    """Improved contrastive loss with weak in-batch negatives and false negative masking.

    Parameters
    ----------
    temperature : float, optional
        Temperature parameter for scaling similarities, by default 0.02
    """

    def __init__(self, temperature: float = 0.02):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        query_embeds: torch.Tensor,
        positive_embeds: torch.Tensor,
        negative_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the contrastive loss.

        Parameters
        ----------
        query_embeds : torch.Tensor
            Query embeddings of shape (batch_size, embed_dim)
        positive_embeds : torch.Tensor
            Positive document embeddings of shape (batch_size, embed_dim)
        negative_embeds : torch.Tensor
            Hard negative embeddings of shape (batch_size, num_negatives, embed_dim)

        Returns
        -------
        torch.Tensor
            Scalar loss value
        """
        batch_size = query_embeds.shape[0]
        num_negatives = negative_embeds.shape[1]

        # Normalize embeddings
        query_embeds = F.normalize(query_embeds, p=2, dim=1)
        positive_embeds = F.normalize(positive_embeds, p=2, dim=1)
        negative_embeds = F.normalize(negative_embeds, p=2, dim=2)

        # Compute (qi, di+) - query with its positive
        pos_sim = torch.sum(query_embeds * positive_embeds, dim=1)  # (batch_size,)

        # Compute numerator: exp(sim(qi, di+) / τ)
        numerator = torch.exp(pos_sim / self.temperature)  # (batch_size,)

        # Compute denominator Z
        # 1. Hard negatives: (qi, di-) for each sample
        hard_neg_sim = torch.einsum("be,bne->bn", query_embeds, negative_embeds)  # (batch_size, num_negatives)

        # 2. Weak in-batch negatives
        # (qi, qj) - query with other queries
        query_query_sim = torch.matmul(query_embeds, query_embeds.T)  # (batch_size, batch_size)

        # (d+i, dj-) - positive with all hard negatives
        pos_neg_sim = torch.einsum(
            "be,bne->bn", positive_embeds, negative_embeds.reshape(batch_size, num_negatives, -1)
        )  # (batch_size, num_negatives)

        # (qi, dj-) - query with all hard negatives from other samples
        query_neg_sim = torch.einsum("be,bne->bn", query_embeds, negative_embeds)  # (batch_size, num_negatives)

        # (qi, dj+) - query with other positives
        query_pos_sim = torch.matmul(query_embeds, positive_embeds.T)  # (batch_size, batch_size)

        # Apply false negative mask m_ij
        # Mask out in-batch negatives that are more similar than positive
        pos_sim_expanded = pos_sim.unsqueeze(1)  # (batch_size, 1)

        # Mask for query-query similarities
        qq_mask = (query_query_sim <= pos_sim_expanded).float()
        # Don't mask diagonal (self-similarity)
        qq_mask.fill_diagonal_(0)

        # Mask for query-positive similarities
        qp_mask = (query_pos_sim <= pos_sim_expanded).float()
        # Don't mask diagonal (own positive)
        qp_mask.fill_diagonal_(0)

        # Compute denominator components
        # exp(sim / τ) for all negative pairs
        hard_neg_exp = torch.exp(hard_neg_sim / self.temperature).sum(dim=1)  # (batch_size,)
        pos_neg_exp = torch.exp(pos_neg_sim / self.temperature).sum(dim=1)  # (batch_size,)
        query_neg_exp = torch.exp(query_neg_sim / self.temperature).sum(dim=1)  # (batch_size,)

        # Apply masks and compute weak in-batch negative contributions
        query_query_exp = (torch.exp(query_query_sim / self.temperature) * qq_mask).sum(dim=1)  # (batch_size,)
        query_pos_exp = (torch.exp(query_pos_sim / self.temperature) * qp_mask).sum(dim=1)  # (batch_size,)

        # Total denominator Z
        denominator = (
            numerator + hard_neg_exp + pos_neg_exp + query_neg_exp + query_query_exp + query_pos_exp
        )  # (batch_size,)

        # Compute loss: -log(numerator / Z)
        loss = -torch.log(numerator / denominator).mean()

        return loss

