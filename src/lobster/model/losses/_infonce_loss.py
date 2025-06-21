"""InfoNCE loss implementation for contrastive learning."""

import torch
import torch.nn as nn
from torch import Tensor

from .._disco_clip import Gather
from .._distributed_utils import get_rank, is_distributed


class InfoNCELoss(nn.Module):
    """InfoNCE loss for contrastive learning.

    Parameters
    ----------
    temperature : float, default=0.07
        Temperature parameter for scaling the logits.
    use_disco : bool, default=False
        Whether to use distributed contrastive loss for memory efficiency.
    """

    def __init__(self, temperature: float = 0.07, use_disco: bool = False) -> None:
        super().__init__()
        self.temperature = temperature
        self.use_disco = use_disco

    def _standard_loss(self, embeddings_a: Tensor, embeddings_b: Tensor) -> Tensor:
        """Compute InfoNCE loss using the standard approach with full similarity matrix.

        Parameters
        ----------
        embeddings_a : Tensor
            First set of normalized embeddings, shape (batch_size, hidden_size)
        embeddings_b : Tensor
            Second set of normalized embeddings, shape (batch_size, hidden_size)

        Returns
        -------
        Tensor
            InfoNCE loss
        """
        # Compute similarity matrix using temperature
        similarities = embeddings_a @ embeddings_b.T / self.temperature

        # Create labels (diagonal should be positive)
        labels = torch.arange(embeddings_a.shape[0], device=embeddings_a.device)

        # InfoNCE loss in both directions
        loss_a = nn.functional.cross_entropy(similarities, labels)
        loss_b = nn.functional.cross_entropy(similarities.T, labels)

        return (loss_a + loss_b) / 2

    def _disco_loss(self, embeddings_a: Tensor, embeddings_b: Tensor) -> Tensor:
        """Compute InfoNCE loss using DisCo-CLIP distributed approach for memory efficiency.

        Parameters
        ----------
        embeddings_a : Tensor
            First set of normalized embeddings, shape (local_batch_size, hidden_size)
        embeddings_b : Tensor
            Second set of normalized embeddings, shape (local_batch_size, hidden_size)

        Returns
        -------
        Tensor
            InfoNCE loss
        """
        # Gather embeddings from all GPUs using DisCo-CLIP
        all_embeddings_a = Gather(embeddings_a)
        all_embeddings_b = Gather(embeddings_b)

        # Get local batch size and rank for label calculation
        local_batch_size = embeddings_a.shape[0]
        rank = get_rank()

        # Compute local similarities using DisCo-CLIP approach with slicing
        # This is more memory-efficient: we only compute (local_batch x total_batch)
        # instead of (total_batch x total_batch)
        logits_a = (
            all_embeddings_a[local_batch_size * rank : local_batch_size * (rank + 1)]
            @ all_embeddings_b.T
            / self.temperature
        )
        logits_b = (
            all_embeddings_b[local_batch_size * rank : local_batch_size * (rank + 1)]
            @ all_embeddings_a.T
            / self.temperature
        )

        # Create labels - positive pairs are at positions offset by rank * local_batch_size
        labels = torch.arange(local_batch_size, device=embeddings_a.device) + rank * local_batch_size

        # InfoNCE loss in both directions
        loss_a = nn.functional.cross_entropy(logits_a, labels)
        loss_b = nn.functional.cross_entropy(logits_b, labels)

        return (loss_a + loss_b) / 2

    def forward(self, embeddings_a: Tensor, embeddings_b: Tensor) -> Tensor:
        """Compute InfoNCE loss between two sets of embeddings.

        Parameters
        ----------
        embeddings_a : Tensor
            First set of normalized embeddings
        embeddings_b : Tensor
            Second set of normalized embeddings

        Returns
        -------
        Tensor
            The computed InfoNCE loss
        """
        if self.use_disco and is_distributed():
            return self._disco_loss(embeddings_a, embeddings_b)
        return self._standard_loss(embeddings_a, embeddings_b)

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
