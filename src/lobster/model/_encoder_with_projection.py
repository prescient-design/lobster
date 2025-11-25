from os import PathLike
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from lobster.model._ume_contrastive_triplets import UMEContrastiveTriplets
from lobster.model.ume2.ume_sequence_encoder_lightning_module import UMESequenceEncoderLightningModule
from lobster.model._pooler import LMAttentionPool1D, LMClsPooler


class EncoderWithProjection(nn.Module):
    """
    Wrapper that ensures both BEFORE and AFTER contrastive training checkpoints have
    identical architecture (encoder + projection head) for fair comparison.
    
    For BEFORE checkpoints (UMESequenceEncoderLightningModule):
        - Loads pretrained encoder
        - Adds fresh (untrained) projection head matching UMEContrastiveTriplets architecture
    
    For AFTER checkpoints (UMEContrastiveTriplets):
        - Loads full model with trained encoder + projection head
    
    This allows us to compare:
    - BEFORE: pretrained encoder + random projection
    - AFTER: contrastive-trained encoder + contrastive-trained projection
    
    Parameters
    ----------
    checkpoint_path : str | PathLike
        Path to checkpoint (either UMESequenceEncoderLightningModule or UMEContrastiveTriplets)
    is_contrastive_checkpoint : bool, default=False
        Whether checkpoint is from UMEContrastiveTriplets (True) or UMESequenceEncoderLightningModule (False)
    embedding_dim : int, default=256
        Dimension of projection head output
    pooling_type : Literal["mean", "cls", "attention"], default="mean"
        Type of pooling to use for sequence embeddings
    map_location : str, default="cpu"
        Device to load checkpoint to
    """
    
    def __init__(
        self,
        checkpoint_path: str | PathLike,
        is_contrastive_checkpoint: bool = False,
        embedding_dim: int = 256,
        pooling_type: Literal["mean", "cls", "attention"] = "mean",
        map_location: str = "cpu",
    ):
        super().__init__()
        
        self._projection_dim = embedding_dim  # Projection head output dimension (256)
        self.pooling_type = pooling_type
        
        if is_contrastive_checkpoint:
            # Load full UMEContrastiveTriplets (encoder + trained projection)
            model = UMEContrastiveTriplets.load_from_checkpoint(
                checkpoint_path,
                map_location=map_location
            )
            self.encoder = model.encoder
            self.projector = model.projector
            self.pooler = model.pooler
            self._input_dim = model._input_dim
            
        else:
            # Load encoder and add fresh projection head
            self.encoder = UMESequenceEncoderLightningModule.load_from_checkpoint(
                checkpoint_path,
                map_location=map_location
            )
            
            # Get encoder hidden size
            input_dim = self.encoder.encoder.neobert.config.hidden_size
            self._input_dim = input_dim
            
            # Create same projection architecture as UMEContrastiveTriplets
            self.projector = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.ReLU(),
                nn.Linear(input_dim // 2, self._projection_dim),
            )
            
            # Create pooler based on pooling_type
            if pooling_type == "cls":
                pooler_config = type('Config', (), {'hidden_size': input_dim})()
                self.pooler = LMClsPooler(pooler_config)
            elif pooling_type == "attention":
                pooler_config = type('Config', (), {'hidden_size': input_dim})()
                self.pooler = LMAttentionPool1D(pooler_config)
            else:  # mean pooling
                self.pooler = None  # Will use default mean pooling from encoder
    
    @property
    def embedding_dim(self):
        """
        Return the encoder's native hidden dimension for PropertyRegression.
        
        This property is checked by PropertyRegression to determine the input dimension
        for building the regression head. We return the encoder's native dimension
        (not the projection dimension) so that PropertyRegression can build its head
        on top of the raw encoder outputs.
        """
        return self._input_dim
    
    @property
    def hidden_size(self):
        """Return the encoder's native hidden dimension (alias for embedding_dim)."""
        return self._input_dim
    
    def embed(self, input_ids, attention_mask=None, aggregate=None, **kwargs):
        """
        Embed sequences using encoder + optional projection head.
        Matches UMEContrastiveTriplets.embed() signature and accepts PropertyRegression parameters.
        
        Parameters
        ----------
        input_ids : torch.Tensor or dict
            Tokenized input, shape (batch_size, seq_length) or dict with 'input_ids' and 'attention_mask'
        attention_mask : torch.Tensor, optional
            Attention mask, same shape as input_ids
        aggregate : bool, optional
            If False, returns unpooled embeddings (batch_size, seq_length, hidden_dim) WITHOUT projection
                      for PropertyRegression to build its own regression head on native encoder dims.
            If True, returns pooled+projected+normalized embeddings (batch_size, embedding_dim)
                     for contrastive evaluation.
            If None, defaults to True.
        **kwargs : dict
            Additional keyword arguments (e.g., 'ignore_padding')
        
        Returns
        -------
        embeddings : torch.Tensor
            If aggregate=False: unpooled embeddings (batch_size, seq_length, hidden_dim) - NO projection
            If aggregate=True: normalized projected embeddings (batch_size, embedding_dim)
        """
        # Handle dict input (from PropertyRegression)
        if isinstance(input_ids, dict):
            attention_mask = input_ids.get('attention_mask', attention_mask)
            input_ids = input_ids['input_ids']
        
        # Default to aggregate=True for backward compatibility
        if aggregate is None:
            aggregate = True
        
        # If PropertyRegression wants unpooled embeddings, return them directly WITHOUT projection
        # This allows PropertyRegression to build its regression head on the encoder's native dimension
        if aggregate is False:
            # Return raw encoder outputs without pooling or projection
            embeddings = self.encoder.embed(
                {'input_ids': input_ids, 'attention_mask': attention_mask},
                aggregate=False,
                ignore_padding=kwargs.get('ignore_padding', True)
            )
            return embeddings
        
        # Otherwise, do full pooling + projection + normalization for contrastive evaluation
        # Get encoder embeddings with pooling
        if self.pooler is not None:
            # Use custom pooling (cls or attention)
            embeddings = self.encoder.embed(
                {'input_ids': input_ids, 'attention_mask': attention_mask},
                aggregate=False,
                ignore_padding=kwargs.get('ignore_padding', True)
            )
            # Apply custom pooling
            embeddings = self.pooler(embeddings, attention_mask)
        else:
            # Use default mean pooling
            embeddings = self.encoder.embed(
                {'input_ids': input_ids, 'attention_mask': attention_mask},
                aggregate=True,
                ignore_padding=kwargs.get('ignore_padding', True)
            )
        
        # Apply projection (only when aggregate=True for contrastive evaluation)
        embeddings = self.projector(embeddings)
        
        # Normalize
        embeddings = F.normalize(embeddings, dim=-1)
        
        return embeddings
    
    def forward(self, input_ids, attention_mask):
        """Forward pass - just calls embed()."""
        return self.embed(input_ids, attention_mask)

