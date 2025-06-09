"""
E(3)-Equivariant Graph Neural Network implementation for molecular 3D structure encoding.

Adapted from: https://github.com/terraytherapeutics/COATI/blob/main/coati/models/encoding/e3gnn_clip.py
by Terray Therapeutics.
"""

import torch
import torch.nn as nn

from ._equivariant_layer import E3EquivariantLayer


class E3GNN(torch.nn.Module):
    """
    E(3)-Equivariant Graph Neural Network for molecular structure encoding.

    This is a simplified version designed for CLIP-style contrastive learning.
    It processes molecular 3D structures and outputs fixed-size embeddings.
    """

    def __init__(
        self,
        input_dim: int = 119,  # Size of atomic feature vector
        hidden_dim: int = 128,
        device: str = "cpu",
        act_fn: str = "SiLU",
        n_layers: int = 5,
        instance_norm: bool = True,
        message_cutoff: float = 5.0,
        dtype: torch.dtype = torch.float,
        use_atomic_embedding: bool = False,
        residual: bool = False,
        dropout: float = 0.1,
        output_dim: int = None,
    ):
        """
        Parameters
        ----------
        input_dim : int, optional
            Input node feature dimension (atomic features)
        hidden_dim : int, optional
            Hidden dimension for the network
        device : str, optional
            Device to place the model on
        act_fn : str, optional
            Activation function name ("SiLU" or "GELU")
        n_layers : int, optional
            Number of E(3)-equivariant layers
        instance_norm : bool, optional
            Whether to use instance normalization
        message_cutoff : float, optional
            Distance cutoff for message passing
        dtype : torch.dtype, optional
            Data type for computations
        use_atomic_embedding : bool, optional
            Whether to use learnable atomic embeddings
        residual : bool, optional
            Whether to use residual connections
        dropout : float, optional
            Dropout probability
        output_dim : int, optional
            Output embedding dimension (defaults to hidden_dim)
        """
        super().__init__()

        self.dtype = dtype
        self.hidden_dim = hidden_dim
        self.device = device
        self.n_layers = n_layers
        self.instance_norm = instance_norm
        self.message_cutoff = torch.tensor(message_cutoff, requires_grad=False)
        self.use_atomic_embedding = use_atomic_embedding

        if output_dim is None:
            output_dim = hidden_dim
        self.output_dim = output_dim

        assert 0.0 <= dropout < 1.0
        self.dropout = dropout

        # Activation function
        if act_fn == "SiLU":
            self.act_fn = nn.SiLU()
        elif act_fn == "GELU":
            self.act_fn = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation function: {act_fn}")

        # Node feature embedding
        if self.use_atomic_embedding:
            # Use learnable embeddings for atomic numbers
            self.input_dim = hidden_dim
            self.atomic_embedding = nn.Embedding(input_dim, hidden_dim, device=device, dtype=dtype)
            self.embedding = torch.nn.Identity()
        else:
            # Use one-hot or other explicit atomic features
            self.input_dim = input_dim
            self.atomic_embedding = None
            self.embedding = nn.Linear(input_dim, hidden_dim)

        # Instance normalization for embeddings
        if instance_norm:
            self.embedding_norm = torch.nn.InstanceNorm1d(hidden_dim)
        else:
            self.embedding_norm = torch.nn.Identity()

        # Final node decoder
        self.node_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            self.act_fn,
            nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(hidden_dim, output_dim),
        )

        # E(3)-equivariant layers
        for i in range(n_layers):
            self.add_module(
                f"e3_layer_{i}",
                E3EquivariantLayer(
                    input_nf=hidden_dim,
                    act_fn=self.act_fn,
                    residual=residual,
                    attention=False,
                    instance_norm=instance_norm,
                    residual_nf=(input_dim if residual else 0),
                    dropout=dropout,
                    message_cutoff=message_cutoff,
                    prop_coords=False,  # Don't update coordinates for CLIP
                ),
            )

        self.to(self.device)

    def forward(self, atoms: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the E3GNN.

        Parameters
        ----------
        atoms : torch.Tensor
            Atomic numbers or features [batch_size, max_atoms, features]
        coords : torch.Tensor
            3D coordinates [batch_size, max_atoms, 3]

        Returns
        -------
        torch.Tensor
            Molecular embeddings [batch_size, output_dim]
        """
        # Create node mask (atoms > 0 exist)
        if atoms.dtype in [torch.long, torch.int]:
            node_mask = (atoms > 0).to(atoms.device, torch.float)

            if self.use_atomic_embedding:
                # Use learnable atomic embeddings
                assert atoms.max().item() < self.input_dim, (
                    f"Atomic number too large for embedding (max: {self.input_dim})"
                )
                nodes = self.atomic_embedding(atoms.long())
            else:
                # Convert atomic numbers to one-hot or other features
                # This would need to be implemented based on your atomic feature scheme
                raise NotImplementedError("Non-embedding atomic features not implemented")
        else:
            # Assuming atoms already contains features
            nodes = atoms
            node_mask = (atoms.sum(dim=-1) > 0).to(atoms.device, torch.float)

        # Ensure finite values
        assert nodes.isfinite().all(), "Node features contain non-finite values"
        assert coords.isfinite().all(), "Coordinates contain non-finite values"
        assert node_mask.isfinite().all(), "Node mask contains non-finite values"

        # Initial embedding
        h = self.embedding_norm(self.embedding(nodes))

        # Apply E(3)-equivariant layers
        for i in range(self.n_layers):
            h, _ = self._modules[f"e3_layer_{i}"](h, coords, node_mask, h0=nodes)

        # Final node decoding
        h = self.node_decoder(h)

        # Mask out non-existent atoms
        h = h * node_mask.unsqueeze(-1)

        # Global aggregation (sum pooling with normalization)
        n_atoms = torch.maximum(node_mask.sum(-1), torch.ones_like(node_mask.sum(-1)))
        molecular_embedding = torch.sum(h, dim=1) / n_atoms.unsqueeze(-1)

        return molecular_embedding
