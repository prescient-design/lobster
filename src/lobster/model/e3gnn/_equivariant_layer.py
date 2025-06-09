"""
Adapted from: https://github.com/terraytherapeutics/COATI/blob/main/coati/models/encoding/e3gnn_clip.py
by Terray Therapeutics.
"""

from collections.abc import Callable

import torch
import torch.nn as nn

from ._utils import cubic_cutoff, make_neighborlist


class E3EquivariantLayer(nn.Module):
    """
    E(3)-Equivariant Graph Convolutional Layer.

    This layer maintains E(3) equivariance (rotation, translation, reflection)
    while updating both node features and coordinates.

    Based on equations (3)-(6) from "E(n) Equivariant Graph Neural Networks"
    (Satorras et al., ICML 2021).
    """

    def __init__(
        self,
        input_nf: int,
        output_nf: int = None,
        hidden_nf: int = None,
        act_fn: Callable = nn.SiLU(),
        recurrent: bool = True,
        residual: bool = True,
        attention: bool = False,
        instance_norm: bool = False,
        residual_nf: int = 0,
        message_cutoff: float = 5.0,
        dropout: float = 0.0,
        prop_coords: bool = True,
    ):
        """
        Parameters
        ----------
        input_nf : int
            Input node feature dimension
        output_nf : int, optional
            Output node feature dimension (defaults to input_nf)
        hidden_nf : int, optional
            Hidden dimension for MLPs (defaults to input_nf)
        act_fn : Callable, optional
            Activation function
        recurrent : bool, optional
            Whether to use residual connections for node features
        residual : bool, optional
            Whether to include initial node features in updates
        attention : bool, optional
            Whether to use attention mechanism (not recommended)
        instance_norm : bool, optional
            Whether to use instance normalization
        residual_nf : int, optional
            Dimension of residual features
        message_cutoff : float, optional
            Distance cutoff for messages
        dropout : float, optional
            Dropout probability
        prop_coords : bool, optional
            Whether to update coordinates
        """
        super().__init__()

        self.message_cutoff = torch.tensor(message_cutoff, requires_grad=False)

        if output_nf is None:
            output_nf = input_nf
        if hidden_nf is None:
            hidden_nf = input_nf

        self.residual_nf = residual_nf
        self.residual = residual
        self.recurrent = recurrent
        self.attention = attention
        self.prop_coords = prop_coords
        self.dropout = dropout

        # Instance normalization
        if instance_norm:
            self.instance_norm = torch.nn.InstanceNorm1d(hidden_nf)
        else:
            self.instance_norm = torch.nn.Identity()

        # Edge model (φ_e in the paper)
        input_edge = input_nf * 2
        edge_coords_nf = 1  # Squared distance

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf, hidden_nf),
            act_fn,
            nn.Dropout(p=self.dropout) if self.dropout else nn.Identity(),
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            nn.Dropout(p=self.dropout) if self.dropout else nn.Identity(),
        )

        # Node model (φ_h in the paper)
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + self.residual_nf + input_nf, hidden_nf),
            act_fn,
            nn.Dropout(p=self.dropout) if self.dropout else nn.Identity(),
            nn.Linear(hidden_nf, output_nf),
        )

        # Coordinate model (φ_x in the paper)
        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        self.coord_mlp = nn.Sequential(*coord_mlp)

        # Optional attention mechanism
        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(input_nf, hidden_nf),
                act_fn,
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid(),
            )

        self.act_fn = act_fn

    def edge_model(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        node_mask: torch.Tensor,
        distance_gradient: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute edge messages between neighboring atoms.

        Parameters
        ----------
        h : torch.Tensor
            Node features [n_batch, n_atoms, n_features]
        x : torch.Tensor
            Coordinates [n_batch, n_atoms, 3]
        node_mask : torch.Tensor
            Node existence mask [n_batch, n_atoms]
        distance_gradient : bool, optional
            Whether to compute gradients w.r.t. distances

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            Tuple of (messages, batch_indices, atom_i_indices, atom_j_indices, distances)
        """
        if distance_gradient:
            Is, Js, Ks, Ds = make_neighborlist(x, node_mask, self.message_cutoff)
        else:
            with torch.no_grad():
                Is, Js, Ks, Ds = make_neighborlist(x, node_mask, self.message_cutoff)

        # Concatenate node features and squared distances
        h2 = torch.cat([h[Is, Js, :], h[Is, Ks, :], (Ds * Ds).unsqueeze(-1)], -1)

        # Apply smooth cutoff
        msg_mask = cubic_cutoff(Ds, self.message_cutoff).unsqueeze(-1)

        # Compute messages
        mij = self.edge_mlp(h2) * msg_mask

        # Optional attention
        if self.attention:
            att_val = self.att_mlp(mij)
            mij = mij * att_val * msg_mask

        return mij, Is, Js, Ks, Ds

    def coord_model(
        self,
        x: torch.Tensor,
        mij: torch.Tensor,
        Is: torch.Tensor,
        Js: torch.Tensor,
        Ks: torch.Tensor,
        Ds: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Update coordinates while maintaining E(3) equivariance.

        Parameters
        ----------
        x : torch.Tensor
            Current coordinates [n_batch, n_atoms, 3]
        mij : torch.Tensor
            Edge messages [n_messages, n_hidden]
        Is : torch.Tensor
            Batch indices
        Js : torch.Tensor
            Source atom indices
        Ks : torch.Tensor
            Target atom indices
        Ds : torch.Tensor
            Distances
        node_mask : torch.Tensor
            Node existence mask

        Returns
        -------
        torch.Tensor
            Updated coordinates [n_batch, n_atoms, 3]
        """
        nb = x.shape[0]
        na = x.shape[1]
        C = 1.0 / (na - 1.0)

        # Compute coordinate updates
        phi_x_mij = self.coord_mlp(mij)
        x_update = torch.zeros(nb, na, 3, dtype=x.dtype, device=x.device)
        x_update[Is, Js, :] += C * (x[Is, Js, :] - x[Is, Ks, :]) * phi_x_mij

        out = x + x_update
        return torch.clamp(out, -1000.0, 1000.0)

    def node_model(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        mij: torch.Tensor,
        Is: torch.Tensor,
        Js: torch.Tensor,
        Ks: torch.Tensor,
        Ds: torch.Tensor,
        node_mask: torch.Tensor,
        h0: torch.Tensor,
    ) -> torch.Tensor:
        """
        Update node features based on aggregated messages.

        Parameters
        ----------
        h : torch.Tensor
            Current node features [n_batch, n_atoms, n_features]
        x : torch.Tensor
            Coordinates [n_batch, n_atoms, 3]
        mij : torch.Tensor
            Edge messages [n_messages, n_hidden]
        Is : torch.Tensor
            Batch indices
        Js : torch.Tensor
            Source atom indices
        Ks : torch.Tensor
            Target atom indices
        Ds : torch.Tensor
            Distances
        node_mask : torch.Tensor
            Node existence mask
        h0 : torch.Tensor
            Initial node features (for residual connections)

        Returns
        -------
        torch.Tensor
            Updated node features [n_batch, n_atoms, n_features]
        """
        nb = h.shape[0]
        na = h.shape[1]
        nh = h.shape[2]

        # Aggregate messages for each node
        mi = (
            torch.zeros(nb * na, nh, device=h.device, dtype=h.dtype)
            .scatter_add_(0, (na * Is + Js).unsqueeze(-1).tile(1, nh), mij)
            .reshape(nb, na, nh)
        )

        # Update node features
        if self.residual_nf:
            out = self.node_mlp(torch.cat([h, mi, h0], dim=-1))
        else:
            out = self.node_mlp(torch.cat([h, mi], dim=-1))

        if self.recurrent:
            out = h + out

        return out

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        node_mask: torch.Tensor,
        h0: torch.Tensor,
        distance_gradient: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the E(3)-equivariant layer.

        Parameters
        ----------
        h : torch.Tensor
            Node features [n_batch, n_atoms, n_features]
        x : torch.Tensor
            Coordinates [n_batch, n_atoms, 3]
        node_mask : torch.Tensor
            Node existence mask [n_batch, n_atoms]
        h0 : torch.Tensor
            Initial node features
        distance_gradient : bool, optional
            Whether to compute gradients w.r.t. distances

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Tuple of (updated_node_features, updated_coordinates)
        """
        # Compute edge messages
        mij, Is, Js, Ks, Ds = self.edge_model(h, x, node_mask, distance_gradient=distance_gradient)

        # Update node features
        h_new = self.instance_norm(self.node_model(h, x, mij, Is, Js, Ks, Ds, node_mask, h0))

        # Update coordinates (if enabled)
        if self.prop_coords:
            x_new = self.coord_model(x, mij, Is, Js, Ks, Ds, node_mask)
        else:
            x_new = x

        return h_new, x_new
