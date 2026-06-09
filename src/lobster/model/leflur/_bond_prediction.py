"""Bond matrix prediction head for LeFlur protein-ligand modeling.

This module predicts bond types between atom pairs from encoder output
features. Used for SMILES reconstruction from generated atom types.
"""

import torch
import torch.nn as nn

from lobster.model.latent_generator.utils.residue_constants import NUM_BOND_TYPES


class BondMatrixPredictionHead(nn.Module):
    """Predict bond matrix from atom features.

    Given atom features from the encoder, predicts bond types between
    all atom pairs. Output can be used with cross-entropy loss against
    ground truth bond matrices.

    This implementation uses a memory-efficient outer product approach
    rather than explicit pairwise tensor construction to reduce GPU memory.

    Parameters
    ----------
    hidden_size : int
        Dimension of input atom features.
    num_bond_types : int, optional
        Number of bond type classes (default: 6).
        0=none, 1=single, 2=double, 3=triple, 4=aromatic, 5=other.
    symmetric : bool, optional
        If True, enforce symmetric predictions (default: True).
        Bonds are inherently symmetric (A-B = B-A).

    Examples
    --------
    >>> head = BondMatrixPredictionHead(hidden_size=64)
    >>> atom_features = torch.randn(2, 10, 64)
    >>> logits = head(atom_features)
    >>> logits.shape
    torch.Size([2, 10, 10, 6])

    Notes
    -----
    Uses outer product of projected features for memory efficiency.
    """

    def __init__(
        self,
        hidden_size: int,
        num_bond_types: int = NUM_BOND_TYPES,
        symmetric: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_bond_types = num_bond_types
        self.symmetric = symmetric

        # Project atoms to smaller dimension for efficiency
        proj_dim = min(hidden_size, 64)
        self.proj_dim = proj_dim

        # Separate projections for source and destination atoms
        self.proj_src = nn.Linear(hidden_size, proj_dim * num_bond_types)
        self.proj_dst = nn.Linear(hidden_size, proj_dim * num_bond_types)

    def forward(
        self,
        atom_features: torch.Tensor,
        atom_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict bond types between all atom pairs.

        Parameters
        ----------
        atom_features : torch.Tensor
            Atom features from encoder with shape [B, N_atoms, H].
        atom_mask : torch.Tensor, optional
            Valid atom mask with shape [B, N_atoms].

        Returns
        -------
        torch.Tensor
            Bond type logits with shape [B, N_atoms, N_atoms, num_bond_types].
        """
        batch_size, num_atoms, _ = atom_features.shape

        # Project to [B, N, proj_dim * num_bond_types]
        src = self.proj_src(atom_features)
        dst = self.proj_dst(atom_features)

        # Reshape to [B, N, num_bond_types, proj_dim]
        src = src.view(batch_size, num_atoms, self.num_bond_types, self.proj_dim)
        dst = dst.view(batch_size, num_atoms, self.num_bond_types, self.proj_dim)

        # Compute outer product via einsum: [B, N_i, K, D] x [B, N_j, K, D] -> [B, N_i, N_j, K]
        # This is equivalent to sum over D of src[i,k,:] * dst[j,k,:]
        logits = torch.einsum("biku,bjku->bijk", src, dst)

        # Enforce symmetry if requested
        if self.symmetric:
            # Average logits[i,j] and logits[j,i]
            logits = (logits + logits.transpose(1, 2)) / 2

        # Mask diagonal (no self-bonds) by setting to large negative for "no bond"
        diag_mask = torch.eye(num_atoms, device=logits.device, dtype=torch.bool)
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1, self.num_bond_types)

        # Set diagonal to favor "no bond" (index 0)
        no_bond_logits = torch.zeros(self.num_bond_types, device=logits.device, dtype=logits.dtype)
        no_bond_logits[0] = 10.0  # High logit for "no bond"
        no_bond_logits[1:] = -10.0  # Low logit for actual bonds

        logits = torch.where(diag_mask, no_bond_logits, logits)

        return logits


class BondMatrixLoss(nn.Module):
    """Compute loss for bond matrix prediction.

    Parameters
    ----------
    ignore_diagonal : bool, optional
        If True, ignore diagonal elements in loss (default: True).
    class_weights : torch.Tensor, optional
        Weights for each bond type class. Useful for handling class imbalance.
    """

    def __init__(
        self,
        ignore_diagonal: bool = True,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.ignore_diagonal = ignore_diagonal
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, reduction="none")

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        atom_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute bond prediction loss.

        Parameters
        ----------
        logits : torch.Tensor
            Predicted logits with shape [B, N, N, num_bond_types].
        target : torch.Tensor
            Target bond types with shape [B, N, N].
        atom_mask : torch.Tensor, optional
            Valid atom mask with shape [B, N].

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        batch_size, num_atoms, _, num_classes = logits.shape

        # Flatten for cross-entropy (use reshape instead of view for non-contiguous tensors)
        logits_flat = logits.reshape(-1, num_classes)
        target_flat = target.reshape(-1)

        # Compute per-element loss
        loss = self.ce_loss(logits_flat, target_flat)
        loss = loss.reshape(batch_size, num_atoms, num_atoms)

        # Create mask
        if atom_mask is not None:
            # Only compute loss where both atoms are valid
            pair_mask = atom_mask.unsqueeze(-1) * atom_mask.unsqueeze(-2)
        else:
            pair_mask = torch.ones(batch_size, num_atoms, num_atoms, device=loss.device)

        # Ignore diagonal
        if self.ignore_diagonal:
            diag_mask = ~torch.eye(num_atoms, device=loss.device, dtype=torch.bool)
            diag_mask = diag_mask.unsqueeze(0).expand(batch_size, -1, -1)
            pair_mask = pair_mask * diag_mask.float()

        # Apply mask and compute mean
        loss = (loss * pair_mask).sum() / (pair_mask.sum() + 1e-8)

        return loss
