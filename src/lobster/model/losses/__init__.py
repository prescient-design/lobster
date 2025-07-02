"""Loss functions for contrastive learning."""

from ._infonce_loss import InfoNCELoss
from ._symile_loss import SymileLoss

__all__ = ["InfoNCELoss", "SymileLoss"]
