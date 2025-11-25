from os import PathLike
from typing import Literal

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch import nn

from lobster.model._pooler import LMAttentionPool1D, LMClsPooler
from lobster.model.losses import Qwen3ContrastiveLoss
from lobster.model.ume2.ume_sequence_encoder_lightning_module import UMESequenceEncoderLightningModule


class UMEContrastiveTriplets(pl.LightningModule):
    """
    UME model with Qwen3-style contrastive loss for triplet learning.
    
    This model encodes query, positive, and negative sequences and uses
    contrastive loss with hard negatives and in-batch negatives.
    
    Parameters
    ----------
    ume_model_path : str | PathLike
        Path to pretrained UME model checkpoint
    embedding_dim : int, default=256
        Dimension of the final embedding space
    lr : float, default=1e-3
        Learning rate
    freeze_encoder : bool, default=False
        Whether to freeze the UME encoder weights
    temperature : float, default=0.02
        Temperature parameter for contrastive loss (Qwen3 default)
    pooling_type : Literal["mean", "cls", "attention"], default="mean"
        Type of pooling to use for sequence embeddings
    weight_decay : float, default=0.0
        Weight decay for optimizer
    warmup_steps : int, default=0
        Number of warmup steps for learning rate scheduler
    """
    
    def __init__(
        self,
        pretrained_encoder_path: str | PathLike,
        embedding_dim: int = 256,
        lr: float = 1e-3,
        freeze_encoder: bool = False,
        temperature: float = 0.02,
        pooling_type: Literal["mean", "cls", "attention"] = "mean",
        weight_decay: float = 0.0,
        warmup_steps: int = 0,
        ckpt_path: str | PathLike | None = None,  # For Lightning resumption, not used in __init__
    ):
        """Initialize UME Contrastive Triplets model.
        
        Parameters
        ----------
        pretrained_encoder_path : str | PathLike
            Path to the UME encoder checkpoint (UMESequenceEncoderLightningModule).
            This is the pre-trained UME2 model that will be fine-tuned.
        ckpt_path : str | PathLike | None, optional
            Path to resume training from a UMEContrastiveTriplets checkpoint.
            This is handled by Lightning's trainer.fit() and not used in __init__.
        """
        super().__init__()
        # Don't save ckpt_path in hyperparameters (used by Lightning for resumption)
        self.save_hyperparameters(ignore=['ckpt_path'])

        # Load pre-trained UME encoder (UMESequenceEncoderLightningModule)
        self.encoder = UMESequenceEncoderLightningModule.load_from_checkpoint(
            pretrained_encoder_path,
            map_location="cpu",  # Load to CPU first, Lightning will move to correct device
        )

        # Verify the loaded model has the expected structure
        if not hasattr(self.encoder, 'encoder') or not hasattr(self.encoder.encoder, 'neobert'):
            raise ValueError(
                f"Checkpoint at {pretrained_encoder_path} does not appear to be a valid "
                f"UMESequenceEncoderLightningModule checkpoint. "
                f"Please ensure you're loading a UME2 checkpoint."
            )
        
        input_dim = self.encoder.encoder.neobert.config.hidden_size
        self._input_dim = input_dim
        
        # Freeze encoder if requested
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Create pooler based on pooling_type
        if pooling_type == "cls":
            pooler_config = type('Config', (), {'hidden_size': input_dim})()
            self.pooler = LMClsPooler(pooler_config)
        elif pooling_type == "attention":
            pooler_config = type('Config', (), {'hidden_size': input_dim})()
            self.pooler = LMAttentionPool1D(pooler_config)
        else:  # mean pooling
            self.pooler = None  # Will use default mean pooling from encoder
        
        # Projection head
        self.projector = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, embedding_dim),
        )
        
        # Qwen3-style contrastive loss
        self.loss_fn = Qwen3ContrastiveLoss(temperature=temperature)
    
    def embed(self, input_ids, attention_mask):
        """
        Embed sequences using UME encoder.
        
        Parameters
        ----------
        input_ids : torch.Tensor
            Tokenized input, shape (batch_size, seq_length) or (batch_size, num_seqs, seq_length)
        attention_mask : torch.Tensor
            Attention mask, same shape as input_ids
        
        Returns
        -------
        embeddings : torch.Tensor
            Normalized embeddings, shape (batch_size, embedding_dim) or (batch_size, num_seqs, embedding_dim)
        """
        original_shape = input_ids.shape
        
        # Handle 3D tensors (batch, num_seqs, seq_len) by flattening
        if input_ids.ndim == 3:
            batch_size, num_seqs, seq_len = original_shape
            input_ids = input_ids.reshape(batch_size * num_seqs, seq_len)
            attention_mask = attention_mask.reshape(batch_size * num_seqs, seq_len)
        
        # Get embeddings from UME encoder
        with torch.set_grad_enabled(not self.hparams.freeze_encoder):
            if self.pooler is not None:
                # Use custom pooling (cls or attention)
                embeddings = self.encoder.embed(
                    {'input_ids': input_ids, 'attention_mask': attention_mask},
                    aggregate=False,
                    ignore_padding=True
                )
                # Apply custom pooling
                embeddings = self.pooler(embeddings, attention_mask)
            else:
                # Use default mean pooling
                embeddings = self.encoder.embed(
                    {'input_ids': input_ids, 'attention_mask': attention_mask},
                    aggregate=True,
                    ignore_padding=True
                )
        
        # Project to embedding space
        embeddings = self.projector(embeddings)
        
        # Normalize
        embeddings = F.normalize(embeddings, dim=-1)
        
        # Restore original shape if needed
        if len(original_shape) == 3:
            embeddings = embeddings.reshape(batch_size, num_seqs, -1)
        
        return embeddings
    
    def forward(self, batch):
        """
        Forward pass through the model.
        
        Parameters
        ----------
        batch : dict
            Batch from GredAffinityTripletsDataModule containing:
            - query_input_ids, query_attention_mask
            - positive_input_ids, positive_attention_mask
            - negative_input_ids, negative_attention_masks
        
        Returns
        -------
        dict
            Dictionary with query_embeds, positive_embeds, negative_embeds
        """
        # Embed query sequences
        query_embeds = self.embed(batch['query_input_ids'], batch['query_attention_mask'])
        
        # Embed positive sequences
        positive_embeds = self.embed(batch['positive_input_ids'], batch['positive_attention_mask'])
        
        # Embed negative sequences (handle 3D shape)
        negative_embeds = self.embed(batch['negative_input_ids'], batch['negative_attention_masks'])
        
        return {
            'query_embeds': query_embeds,
            'positive_embeds': positive_embeds,
            'negative_embeds': negative_embeds,
        }
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        batch_size = batch['query_input_ids'].shape[0]
        
        # Get embeddings
        embeddings = self(batch)
        
        # Compute contrastive loss
        loss = self.loss_fn(
            query_embeds=embeddings['query_embeds'],
            positive_embeds=embeddings['positive_embeds'],
            negative_embeds=embeddings['negative_embeds'],
        )
        
        # Compute additional metrics
        with torch.no_grad():
            # Query-positive similarity
            query_pos_sim = torch.sum(
                embeddings['query_embeds'] * embeddings['positive_embeds'], dim=1
            ).mean()
            
            # Query-negative similarity (average over all negatives)
            if embeddings['negative_embeds'].numel() > 0:
                query_neg_sim = torch.einsum(
                    'be,bne->bn',
                    embeddings['query_embeds'],
                    embeddings['negative_embeds']
                ).mean()
            else:
                query_neg_sim = torch.tensor(0.0, device=loss.device)
            
            # Number of negatives per sample
            n_negatives = batch['n_negatives'].float().mean()
        
        # Log metrics
        self.log("train/loss", loss, sync_dist=True, batch_size=batch_size, prog_bar=True)
        self.log("train/query_pos_sim", query_pos_sim, sync_dist=True, batch_size=batch_size)
        self.log("train/query_neg_sim", query_neg_sim, sync_dist=True, batch_size=batch_size)
        self.log("train/n_negatives", n_negatives, sync_dist=True, batch_size=batch_size)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        batch_size = batch['query_input_ids'].shape[0]
        
        # Get embeddings
        embeddings = self(batch)
        
        # Compute contrastive loss
        loss = self.loss_fn(
            query_embeds=embeddings['query_embeds'],
            positive_embeds=embeddings['positive_embeds'],
            negative_embeds=embeddings['negative_embeds'],
        )
        
        # Compute additional metrics
        with torch.no_grad():
            # Query-positive similarity
            query_pos_sim = torch.sum(
                embeddings['query_embeds'] * embeddings['positive_embeds'], dim=1
            ).mean()
            
            # Query-negative similarity (average over all negatives)
            if embeddings['negative_embeds'].numel() > 0:
                query_neg_sim = torch.einsum(
                    'be,bne->bn',
                    embeddings['query_embeds'],
                    embeddings['negative_embeds']
                ).mean()
            else:
                query_neg_sim = torch.tensor(0.0, device=loss.device)
            
            # Number of negatives per sample
            n_negatives = batch['n_negatives'].float().mean()
        
        # Log metrics
        self.log("val/loss", loss, sync_dist=True, batch_size=batch_size, prog_bar=True)
        self.log("val/query_pos_sim", query_pos_sim, sync_dist=True, batch_size=batch_size)
        self.log("val/query_neg_sim", query_neg_sim, sync_dist=True, batch_size=batch_size)
        self.log("val/n_negatives", n_negatives, sync_dist=True, batch_size=batch_size)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        
        if self.hparams.warmup_steps > 0:
            # Linear warmup + cosine decay
            from torch.optim.lr_scheduler import LambdaLR
            
            def lr_lambda(current_step):
                if current_step < self.hparams.warmup_steps:
                    # Linear warmup
                    return float(current_step) / float(max(1, self.hparams.warmup_steps))
                else:
                    # Cosine decay (simplified)
                    return 0.5 * (1.0 + torch.cos(
                        torch.tensor(
                            (current_step - self.hparams.warmup_steps) / 
                            max(1, self.trainer.max_steps - self.hparams.warmup_steps) * 3.14159
                        )
                    )).item()
            
            scheduler = LambdaLR(optimizer, lr_lambda)
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }
        
        return optimizer
    
    @property
    def num_trainable_parameters(self):
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

