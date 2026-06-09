"""Bond matrix embedding module for LeFlur protein-ligand modeling.

This module embeds bond matrix information into atom features using an
encoder-agnostic design. Bond information is added to input features
rather than modifying attention, allowing any encoder to be used.
"""

import torch
import torch.nn as nn

from lobster.model.latent_generator.utils.residue_constants import NUM_BOND_TYPES


class BondMatrixEmbedding(nn.Module):
    """Embed bond matrix information into atom features.

    For each atom, this module:
    1. Looks at bonded neighbors in the bond matrix
    2. Embeds the bond types (single, double, triple, aromatic)
    3. Aggregates into a single vector and adds to atom embedding

    The transformer's attention handles longer-range topology naturally
    (layer 1 sees neighbors, layer 2 sees neighbors-of-neighbors, etc.)

    Parameters
    ----------
    hidden_size : int
        Dimension of atom embeddings.
    num_bond_types : int, optional
        Number of bond types (default: 6).
        0=none, 1=single, 2=double, 3=triple, 4=aromatic, 5=other.

    Examples
    --------
    >>> embed = BondMatrixEmbedding(hidden_size=64)
    >>> atom_embeddings = torch.randn(2, 10, 64)
    >>> bond_matrix = torch.randint(0, 5, (2, 10, 10))
    >>> atom_mask = torch.ones(2, 10)
    >>> enriched = embed(atom_embeddings, bond_matrix, atom_mask)
    >>> enriched.shape
    torch.Size([2, 10, 64])

    Notes
    -----
    Design decision: We use SUM (not MEAN) for aggregation because atom degree
    is chemically informative - a terminal -CH3 with 1 bond behaves differently
    from a ring carbon with 3 bonds. LayerNorm in the transformer handles
    magnitude differences.
    """

    def __init__(
        self,
        hidden_size: int,
        num_bond_types: int = NUM_BOND_TYPES,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_bond_types = num_bond_types

        # Bond type embeddings
        self.bond_type_embedding = nn.Embedding(num_bond_types, hidden_size)

        # Project aggregated bond info
        self.bond_proj = nn.Linear(hidden_size, hidden_size)

        # LayerNorm for stable training
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        atom_embeddings: torch.Tensor,
        bond_matrix: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Enrich atom embeddings with direct bond information.

        Parameters
        ----------
        atom_embeddings : torch.Tensor
            Base atom embeddings with shape [B, N_atoms, H].
        bond_matrix : torch.Tensor
            Bond type matrix with shape [B, N_atoms, N_atoms].
            Values: 0=none, 1=single, 2=double, 3=triple, 4=aromatic, 5=other.
        atom_mask : torch.Tensor
            Valid atom mask with shape [B, N_atoms].

        Returns
        -------
        torch.Tensor
            Enriched atom embeddings with shape [B, N_atoms, H].
        """
        # Embed all bonds: [B, N, N] -> [B, N, N, H]
        bond_embeds = self.bond_type_embedding(bond_matrix)

        # Mask out padding atoms: create 2D mask for atom pairs
        mask_2d = atom_mask.unsqueeze(-1) * atom_mask.unsqueeze(-2)  # [B, N, N]
        bond_embeds = bond_embeds * mask_2d.unsqueeze(-1)  # [B, N, N, H]

        # Sum over neighbors where bond exists (bond_type > 0)
        # [B, N, N, H] -> [B, N, H]
        # NOTE: We use SUM not MEAN because atom degree is chemically informative
        bond_exists = (bond_matrix > 0).float().unsqueeze(-1)  # [B, N, N, 1]
        neighbor_bonds = (bond_embeds * bond_exists).sum(dim=2)  # [B, N, H]

        # Project bond context
        bond_context = self.bond_proj(neighbor_bonds)  # [B, N, H]

        # Add to atom embeddings (residual connection)
        enriched = atom_embeddings + bond_context

        # Normalize
        enriched = self.layer_norm(enriched)

        return enriched
