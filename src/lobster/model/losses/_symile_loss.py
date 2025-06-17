"""Adapted from https://github.com/rajesh-lab/symile

Reference:
    @inproceedings{saporta2024symile,
    title = {Contrasting with Symile: Simple Model-Agnostic Representation Learning for Unlimited Modalities}
    author = {Saporta, Adriel and Puli, Aahlad and Goldstein, Mark and Ranganath, Rajesh}
    booktitle = {Advances in Neural Information Processing Systems},
    year = {2024}
    }
"""

import torch
import torch.nn as nn
from torch import Tensor


class SymileLoss(nn.Module):
    """Symile loss for contrastive learning with multiple views.

    Parameters
    ----------
    temperature : float, default=0.07
        Temperature parameter for scaling the logits.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: list[Tensor]) -> Tensor:
        """Compute Symile loss between multiple sets of embeddings.

        Parameters
        ----------
        embeddings : list[Tensor]
            List of normalized embeddings, each of shape (batch_size, hidden_size)

        Returns
        -------
        Tensor
            The computed Symile loss
        """
        n_views = len(embeddings)
        if n_views < 2:
            raise ValueError("Symile loss requires at least 2 views")

        # Compute similarity matrices between all pairs of views
        total_loss = 0.0
        n_pairs = 0

        for i in range(n_views):
            for j in range(i + 1, n_views):
                # Compute similarity matrix using temperature
                similarities = embeddings[i] @ embeddings[j].T * self.temperature

                # Create labels (diagonal should be positive)
                labels = torch.arange(embeddings[i].shape[0], device=embeddings[i].device)

                # InfoNCE loss in both directions
                loss_ij = nn.functional.cross_entropy(similarities, labels)
                loss_ji = nn.functional.cross_entropy(similarities.T, labels)

                total_loss += (loss_ij + loss_ji) / 2
                n_pairs += 1

        return total_loss / n_pairs

    def compute_weighted_loss(self, contrastive_loss: Tensor, mlm_loss: Tensor, contrastive_weight: float) -> Tensor:
        """Compute weighted combination of contrastive and MLM losses.

        Parameters
        ----------
        contrastive_loss : Tensor
            The contrastive loss value
        mlm_loss : Tensor
            The MLM loss value
        contrastive_weight : float
            Weight for the contrastive loss (between 0 and 1)

        Returns
        -------
        Tensor
            The weighted combination of losses
        """
        return (1 - contrastive_weight) * mlm_loss + contrastive_weight * contrastive_loss
