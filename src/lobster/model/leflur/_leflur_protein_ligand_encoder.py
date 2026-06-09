"""LeFlur Protein-Ligand Encoder Module.

This module extends the protein-only LeFlur encoder to support both protein-only and
protein-ligand modeling. It maintains backward compatibility with existing
protein-only tasks while adding new capabilities for protein-ligand complexes.

Key features:
- Unified architecture handling both protein and ligand modalities
- Bond matrix embedding for molecular topology
- Bond matrix prediction for SMILES reconstruction
- Ligand-optional forward pass (behaves like the protein-only LeFlur encoder when no ligand)
"""

import logging
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

from lobster.model.latent_generator.utils.residue_constants import (
    ELEMENT_VOCAB_EXTENDED,
    NUM_BOND_TYPES,
)

from ._bond_embedding import BondMatrixEmbedding
from ._bond_prediction import BondMatrixPredictionHead
from ..neobert import NeoBERTModule

logger = logging.getLogger(__name__)


@dataclass
class AuxiliaryTask:
    """Configuration for auxiliary tasks."""

    name: str
    output_dim: int
    task_type: Literal["regression"] = "regression"
    pooling: Literal["cls", "mean"] = "mean"
    hidden_size: int | None = None
    dropout: float = 0.1
    num_layers: int = 2
    loss_weight: float = 1.0


class LeFlurProteinLigandEncoderModule(nn.Module):
    """Unified encoder for protein and protein-ligand modeling.

    This encoder handles:
    - Protein sequence tokens (per-residue, 33 vocabulary)
    - Protein structure tokens (per-residue, 4375 vocabulary)
    - Ligand atom type tokens (per-atom, 25 vocabulary) - OPTIONAL
    - Ligand structure tokens (per-atom, 4375 vocabulary) - OPTIONAL
    - Bond matrix embedding (NxN, embedded into features) - OPTIONAL

    When ligand inputs are None, the encoder behaves identically to the
    protein-only LeFlur encoder for backward compatibility.

    Parameters
    ----------
    sequence_token_vocab_size : int
        Size of protein sequence vocabulary (default: 33).
    structure_token_vocab_size : int
        Size of structure token vocabulary (default: 4375 for FSQ).
    ligand_atom_vocab_size : int
        Size of ligand atom type vocabulary (default: 25 from ELEMENT_VOCAB_EXTENDED).
    ligand_structure_vocab_size : int
        Size of ligand structure token vocabulary (default: 4375).
    sequence_token_pad_token_id : int
        Padding token ID for sequences (default: 1).
    structure_token_pad_token_id : int
        Padding token ID for structures (default: 4374).
    ligand_atom_pad_token_id : int
        Padding token ID for ligand atoms (default: 0 for PAD).
    conditioning_input_dim : int
        Dimension of conditioning input (default: 1).
    **neobert_kwargs
        Additional arguments for NeoBERTModule.

    Examples
    --------
    >>> encoder = LeFlurProteinLigandEncoderModule(hidden_size=256, num_hidden_layers=6)
    >>> # Protein-only forward (backward compatible)
    >>> output = encoder(seq_ids, struct_ids, mask, cond)
    >>> # Protein-ligand forward
    >>> output = encoder(seq_ids, struct_ids, mask, cond,
    ...                  ligand_atom_input_ids=atom_ids,
    ...                  ligand_structure_input_ids=lig_struct_ids,
    ...                  ligand_mask=lig_mask, bond_matrix=bonds)
    """

    def __init__(
        self,
        auxiliary_tasks: list[AuxiliaryTask] | None = None,
        model_ckpt: str | None = None,
        cache_dir: str | None = None,
        sequence_token_vocab_size: int = 33,
        structure_token_vocab_size: int = 4375,
        ligand_atom_vocab_size: int = len(ELEMENT_VOCAB_EXTENDED),
        ligand_structure_vocab_size: int = 4375,
        sequence_token_pad_token_id: int = 1,
        structure_token_pad_token_id: int = 4374,
        ligand_atom_pad_token_id: int = 0,  # PAD in ELEMENT_VOCAB_EXTENDED
        ligand_structure_pad_token_id: int = 4374,
        conditioning_input_dim: int = 1,
        num_bond_types: int = NUM_BOND_TYPES,
        **neobert_kwargs,
    ) -> None:
        super().__init__()

        # Store config
        self.sequence_token_vocab_size = sequence_token_vocab_size
        self.structure_token_vocab_size = structure_token_vocab_size
        self.ligand_atom_vocab_size = ligand_atom_vocab_size
        self.ligand_structure_vocab_size = ligand_structure_vocab_size
        self.num_bond_types = num_bond_types

        # Initialize NeoBERT backbone
        self.neobert = NeoBERTModule(**neobert_kwargs)
        hidden_size = self.neobert.config.hidden_size
        self.hidden_size = hidden_size  # Store for continuous projection initialization

        # === PROTEIN EMBEDDINGS (same as original) ===
        self.sequence_embedding = nn.Embedding(
            sequence_token_vocab_size,
            hidden_size,
            padding_idx=sequence_token_pad_token_id,
        )
        self.structure_embedding = nn.Embedding(
            structure_token_vocab_size,
            hidden_size,
            padding_idx=structure_token_pad_token_id,
        )
        self.conditioning_embedding = nn.Linear(conditioning_input_dim, hidden_size, bias=False)
        self.combine_embedding = nn.Linear(hidden_size * 3, hidden_size)

        # === LIGAND EMBEDDINGS (new) ===
        self.ligand_atom_embedding = nn.Embedding(
            ligand_atom_vocab_size,
            hidden_size,
            padding_idx=ligand_atom_pad_token_id,
        )
        self.ligand_structure_embedding = nn.Embedding(
            ligand_structure_vocab_size,
            hidden_size,
            padding_idx=ligand_structure_pad_token_id,
        )
        self.ligand_combine_embedding = nn.Linear(hidden_size * 2, hidden_size)

        # === CONTINUOUS STRUCTURE EMBEDDING PROJECTION (for diffusion-based generation) ===
        # When using continuous structure embeddings (from DiffusionLoss), project to hidden_size
        # This is set dynamically if continuous_structure_dim is provided
        self.continuous_structure_proj: nn.Linear | None = None
        self.continuous_ligand_structure_proj: nn.Linear | None = None

        # === BOND MATRIX MODULES (new) ===
        self.bond_embedding = BondMatrixEmbedding(
            hidden_size=hidden_size,
            num_bond_types=num_bond_types,
        )
        self.bond_prediction_head = BondMatrixPredictionHead(
            hidden_size=hidden_size,
            num_bond_types=num_bond_types,
            symmetric=True,
        )

        # === MODALITY EMBEDDING (new) ===
        # 0 = protein, 1 = ligand
        self.modality_embedding = nn.Embedding(2, hidden_size)

        # === OUTPUT HEADS ===
        # Protein outputs (same as original)
        self.sequence_output = nn.Linear(hidden_size, sequence_token_vocab_size)
        self.structure_output = nn.Linear(hidden_size, structure_token_vocab_size)

        # Ligand outputs (new)
        self.ligand_atom_output = nn.Linear(hidden_size, ligand_atom_vocab_size)
        self.ligand_structure_output = nn.Linear(hidden_size, ligand_structure_vocab_size)

        # Handle auxiliary tasks
        self.auxiliary_tasks = None
        if auxiliary_tasks is not None:
            from ..ume2 import AuxiliaryRegressionTaskHead

            self.auxiliary_tasks = nn.ModuleDict(
                {
                    task.name: AuxiliaryRegressionTaskHead(
                        input_dim=hidden_size,
                        output_dim=task.output_dim,
                        task_name=task.name,
                        hidden_size=task.hidden_size,
                        dropout=task.dropout,
                        num_layers=task.num_layers,
                        pooling=task.pooling,
                    )
                    for task in auxiliary_tasks
                }
            )

    def init_continuous_structure_proj(self, continuous_dim: int) -> None:
        """Initialize projection layers for continuous structure embeddings.

        This is called when using DiffusionLoss for structure tokens.
        Projects continuous embeddings to hidden_size for input to transformer.

        Parameters
        ----------
        continuous_dim : int
            Dimension of continuous structure embeddings (e.g., 256).
        """
        self.continuous_structure_proj = nn.Linear(continuous_dim, self.hidden_size)
        self.continuous_ligand_structure_proj = nn.Linear(continuous_dim, self.hidden_size)

    def _embed_protein(
        self,
        sequence_input_ids: Tensor,
        structure_input_ids: Tensor | None,
        conditioning_tensor: Tensor | None,
        structure_embeddings: Tensor | None = None,
    ) -> Tensor:
        """Embed protein sequence and structure tokens.

        Parameters
        ----------
        sequence_input_ids : Tensor
            Sequence token IDs with shape [B, N_res].
        structure_input_ids : Tensor, optional
            Structure token IDs with shape [B, N_res]. Can be None if
            structure_embeddings is provided.
        conditioning_tensor : Tensor, optional
            Conditioning input with shape [B, N_res, C].
        structure_embeddings : Tensor, optional
            Pre-computed continuous structure embeddings with shape [B, N_res, D].
            For MAR-style generation: zeros at uncommitted positions, sampled
            embeddings at committed positions. The mask embedding from
            structure_input_ids is used for uncommitted positions.

        Returns
        -------
        Tensor
            Combined protein embedding with shape [B, N_res, H].
        """
        batch_size, seq_len = sequence_input_ids.shape
        device = sequence_input_ids.device

        seq_emb = self.sequence_embedding(sequence_input_ids)

        # Handle structure embeddings
        if structure_embeddings is not None and self.continuous_structure_proj is not None:
            # MAR-style: blend continuous embeddings with mask embeddings
            # structure_embeddings has zeros for uncommitted positions
            # We need mask embeddings from structure_input_ids for those positions

            # Get discrete embedding (mask tokens for uncommitted positions)
            discrete_struct_emb = self.structure_embedding(structure_input_ids)

            # Project continuous embeddings to hidden_size
            continuous_struct_emb = self.continuous_structure_proj(structure_embeddings)

            # Create blend mask: 1 where continuous embeddings are non-zero (committed)
            # This is approximate - assumes committed positions have non-zero embeddings
            committed_mask = (structure_embeddings.abs().sum(dim=-1, keepdim=True) > 1e-6).float()

            # Blend: discrete for uncommitted, continuous for committed
            struct_emb = discrete_struct_emb * (1 - committed_mask) + continuous_struct_emb * committed_mask
        elif structure_input_ids is not None:
            struct_emb = self.structure_embedding(structure_input_ids)
        else:
            # Fallback: if only continuous embeddings provided without projection
            struct_emb = structure_embeddings

        if conditioning_tensor is None:
            conditioning_tensor = torch.zeros(batch_size, seq_len, 1, device=device)
        cond_emb = self.conditioning_embedding(conditioning_tensor)

        # Combine: [seq; struct; cond] -> H
        combined = torch.cat([seq_emb, struct_emb, cond_emb], dim=-1)
        protein_emb = self.combine_embedding(combined)

        # Add modality embedding (protein = 0)
        modality_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
        protein_emb = protein_emb + self.modality_embedding(modality_ids)

        return protein_emb

    def _embed_ligand(
        self,
        ligand_atom_input_ids: Tensor,
        ligand_structure_input_ids: Tensor | None,
        bond_matrix: Tensor,
        ligand_mask: Tensor,
        ligand_structure_embeddings: Tensor | None = None,
    ) -> Tensor:
        """Embed ligand atom types and structure tokens with bond information.

        Parameters
        ----------
        ligand_atom_input_ids : Tensor
            Atom type token IDs with shape [B, N_atoms].
        ligand_structure_input_ids : Tensor, optional
            Ligand structure token IDs with shape [B, N_atoms]. Can be None if
            ligand_structure_embeddings is provided.
        bond_matrix : Tensor
            Bond type matrix with shape [B, N_atoms, N_atoms].
        ligand_mask : Tensor
            Valid atom mask with shape [B, N_atoms].
        ligand_structure_embeddings : Tensor, optional
            Pre-computed continuous structure embeddings with shape [B, N_atoms, D].
            For MAR-style generation: zeros at uncommitted positions, sampled
            embeddings at committed positions.

        Returns
        -------
        Tensor
            Combined ligand embedding with shape [B, N_atoms, H].
        """
        batch_size, num_atoms = ligand_atom_input_ids.shape
        device = ligand_atom_input_ids.device

        # Embed atom types
        atom_emb = self.ligand_atom_embedding(ligand_atom_input_ids)

        # Handle structure embeddings
        if ligand_structure_embeddings is not None and self.continuous_ligand_structure_proj is not None:
            # MAR-style: blend continuous embeddings with mask embeddings
            discrete_struct_emb = self.ligand_structure_embedding(ligand_structure_input_ids)
            continuous_struct_emb = self.continuous_ligand_structure_proj(ligand_structure_embeddings)

            # Create blend mask: 1 where continuous embeddings are non-zero (committed)
            committed_mask = (ligand_structure_embeddings.abs().sum(dim=-1, keepdim=True) > 1e-6).float()

            # Blend: discrete for uncommitted, continuous for committed
            struct_emb = discrete_struct_emb * (1 - committed_mask) + continuous_struct_emb * committed_mask
        elif ligand_structure_input_ids is not None:
            struct_emb = self.ligand_structure_embedding(ligand_structure_input_ids)
        else:
            struct_emb = ligand_structure_embeddings

        # Combine: [atom; struct] -> H
        combined = torch.cat([atom_emb, struct_emb], dim=-1)
        ligand_emb = self.ligand_combine_embedding(combined)

        # Add bond information
        ligand_emb = self.bond_embedding(ligand_emb, bond_matrix, ligand_mask)

        # Add modality embedding (ligand = 1)
        modality_ids = torch.ones(batch_size, num_atoms, dtype=torch.long, device=device)
        ligand_emb = ligand_emb + self.modality_embedding(modality_ids)

        return ligand_emb

    def forward(
        self,
        sequence_input_ids: Tensor,
        structure_input_ids: Tensor | None,
        attention_mask: Tensor,
        conditioning_tensor: Tensor | None = None,
        position_ids: Tensor | None = None,
        # Ligand inputs (optional)
        ligand_atom_input_ids: Tensor | None = None,
        ligand_structure_input_ids: Tensor | None = None,
        ligand_mask: Tensor | None = None,
        bond_matrix: Tensor | None = None,
        # Validity masks for mixed batches
        protein_valid_mask: Tensor | None = None,
        ligand_valid_mask: Tensor | None = None,
        # Other
        timesteps: Tensor | None = None,
        return_auxiliary_tasks: bool = False,
        # Continuous structure embeddings (for MAR-style generation)
        structure_embeddings: Tensor | None = None,
        ligand_structure_embeddings: Tensor | None = None,
        **kwargs,
    ) -> dict[str, Tensor]:
        """Forward pass supporting both protein-only and protein-ligand inputs.

        Parameters
        ----------
        sequence_input_ids : Tensor
            Protein sequence token IDs with shape [B, N_res].
        structure_input_ids : Tensor, optional
            Protein structure token IDs with shape [B, N_res]. Can be None if
            structure_embeddings is provided.
        attention_mask : Tensor
            Attention mask for protein with shape [B, N_res].
        conditioning_tensor : Tensor, optional
            Conditioning input with shape [B, N_res, C].
        position_ids : Tensor, optional
            Position IDs (not currently used).
        ligand_atom_input_ids : Tensor, optional
            Ligand atom type IDs with shape [B, N_atoms].
        ligand_structure_input_ids : Tensor, optional
            Ligand structure token IDs with shape [B, N_atoms].
        ligand_mask : Tensor, optional
            Valid atom mask with shape [B, N_atoms].
        bond_matrix : Tensor, optional
            Bond type matrix with shape [B, N_atoms, N_atoms].
        protein_valid_mask : Tensor, optional
            Which samples have valid protein with shape [B].
        ligand_valid_mask : Tensor, optional
            Which samples have valid ligand with shape [B].
        timesteps : Tensor, optional
            Timesteps for flow matching.
        return_auxiliary_tasks : bool
            Whether to return auxiliary task outputs.
        structure_embeddings : Tensor, optional
            Pre-computed continuous protein structure embeddings [B, N_res, D].
            If provided, bypasses the structure_embedding layer. Used for
            MAR-style iterative generation.
        ligand_structure_embeddings : Tensor, optional
            Pre-computed continuous ligand structure embeddings [B, N_atoms, D].

        Returns
        -------
        dict[str, Tensor]
            Output dictionary containing:
            - sequence_logits: [B, N_res, vocab_size]
            - structure_logits: [B, N_res, struct_vocab_size]
            - last_hidden_state: [B, N_total, H]
            - ligand_atom_logits: [B, N_atoms, atom_vocab_size] (if ligand present)
            - ligand_structure_logits: [B, N_atoms, struct_vocab_size] (if ligand present)
            - bond_logits: [B, N_atoms, N_atoms, num_bond_types] (if ligand present)
        """
        batch_size = sequence_input_ids.shape[0]
        device = sequence_input_ids.device
        seq_len = sequence_input_ids.shape[1]

        # Check if we have ligand inputs
        has_ligand = (
            ligand_atom_input_ids is not None
            and ligand_atom_input_ids.numel() > 0
            and ligand_atom_input_ids.shape[1] > 0
        )

        # === EMBED PROTEIN ===
        if seq_len > 0:
            protein_emb = self._embed_protein(
                sequence_input_ids,
                structure_input_ids,
                conditioning_tensor,
                structure_embeddings=structure_embeddings,
            )
        else:
            protein_emb = torch.empty(batch_size, 0, self.neobert.config.hidden_size, device=device)

        # === EMBED LIGAND (if present) ===
        if has_ligand:
            num_atoms = ligand_atom_input_ids.shape[1]
            if ligand_mask is None:
                ligand_mask = torch.ones(batch_size, num_atoms, device=device)
            if bond_matrix is None:
                bond_matrix = torch.zeros(batch_size, num_atoms, num_atoms, dtype=torch.long, device=device)

            ligand_emb = self._embed_ligand(
                ligand_atom_input_ids,
                ligand_structure_input_ids,
                bond_matrix,
                ligand_mask,
                ligand_structure_embeddings=ligand_structure_embeddings,
            )
        else:
            ligand_emb = torch.empty(batch_size, 0, self.neobert.config.hidden_size, device=device)
            num_atoms = 0

        # === CONCATENATE AND CREATE COMBINED MASK ===
        # [protein_emb; ligand_emb] -> [B, N_res + N_atoms, H]
        combined_emb = torch.cat([protein_emb, ligand_emb], dim=1)

        # Combined attention mask
        if has_ligand:
            combined_mask = torch.cat([attention_mask, ligand_mask], dim=1)
        else:
            combined_mask = attention_mask

        # === RUN THROUGH TRANSFORMER ===
        # Position IDs are not used (NeoBERT handles internally)
        neobert_output = self.neobert(
            input_ids=None,
            inputs_embeds=combined_emb,
            position_ids=None,
            attention_mask=combined_mask,
            **kwargs,
        )

        hidden_state = neobert_output["last_hidden_state"]

        # === SPLIT HIDDEN STATE BACK ===
        protein_hidden = hidden_state[:, :seq_len, :]
        ligand_hidden = hidden_state[:, seq_len:, :] if has_ligand else None

        # === COMPUTE PROTEIN OUTPUTS ===
        output = {}
        if seq_len > 0:
            output["sequence_logits"] = self.sequence_output(protein_hidden)
            output["structure_logits"] = self.structure_output(protein_hidden)
        else:
            output["sequence_logits"] = torch.empty(batch_size, 0, self.sequence_token_vocab_size, device=device)
            output["structure_logits"] = torch.empty(batch_size, 0, self.structure_token_vocab_size, device=device)

        output["last_hidden_state"] = hidden_state
        # Expose protein and ligand hidden states for DiffusionLoss conditioning
        output["protein_hidden_states"] = protein_hidden

        # === COMPUTE LIGAND OUTPUTS (if present) ===
        if has_ligand and ligand_hidden is not None:
            output["ligand_atom_logits"] = self.ligand_atom_output(ligand_hidden)
            output["ligand_structure_logits"] = self.ligand_structure_output(ligand_hidden)
            output["bond_logits"] = self.bond_prediction_head(ligand_hidden, ligand_mask)
            # Expose ligand hidden states for DiffusionLoss conditioning
            output["ligand_hidden_states"] = ligand_hidden
        else:
            # Empty tensors for consistency
            output["ligand_atom_logits"] = torch.empty(batch_size, 0, self.ligand_atom_vocab_size, device=device)
            output["ligand_structure_logits"] = torch.empty(
                batch_size, 0, self.ligand_structure_vocab_size, device=device
            )
            output["bond_logits"] = torch.empty(batch_size, 0, 0, self.num_bond_types, device=device)
            output["ligand_hidden_states"] = None

        # === AUXILIARY TASKS ===
        if self.auxiliary_tasks is not None and return_auxiliary_tasks:
            for task_name, task_head in self.auxiliary_tasks.items():
                output[task_name] = task_head(hidden_state)

        return output
