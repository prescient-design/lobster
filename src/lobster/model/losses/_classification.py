import torch
import torch.nn as nn
import torch.nn.functional as F

"""Classification loss functions for fine-tuning.

Note: For standard cross-entropy and binary cross-entropy losses, 
use PyTorch's built-in functions directly:
- torch.nn.CrossEntropyLoss (supports label_smoothing, weight, reduction)
- torch.nn.functional.cross_entropy (functional version)
- torch.nn.functional.binary_cross_entropy_with_logits (supports pos_weight, reduction)

This module provides specialized losses not available in PyTorch.
"""

class FocalLoss(nn.Module):
    """Focal loss for addressing class imbalance.

    Focal loss down-weights easy examples and focuses on hard examples and hard tokens (e.g. Ab HCDR3s).
    Useful for highly imbalanced molecular classification tasks.

    Parameters
    ----------
    alpha : float or torch.Tensor, optional
        Weighting factor for rare class, by default 1.0
    gamma : float, optional
        Focusing parameter, by default 2.0
    reduction : str, optional
        Reduction method, by default "mean"

    References
    ----------
    Focal Loss: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    Like torchvision.ops.sigmoid_focal_loss but for multi-class classification:
        https://docs.pytorch.org/vision/main/generated/torchvision.ops.sigmoid_focal_loss.html

    Examples
    --------
    >>> loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
    >>> pred = torch.randn(32, 2)  # Binary classification
    >>> target = torch.randint(0, 2, (32,))
    >>> loss = loss_fn(pred, target)
    """

    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 2.0,
        reduction: str = "mean"
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        input : torch.Tensor
            Predicted logits of shape (N, C) where N is batch size and C is number of classes
        target : torch.Tensor
            Target class indices of shape (N,)

        Returns
        -------
        torch.Tensor
            Focal loss value
        """
        # Compute cross entropy
        ce_loss = F.cross_entropy(input, target, reduction="none")

        # Compute probabilities
        pt = torch.exp(-ce_loss)

        # Compute focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss