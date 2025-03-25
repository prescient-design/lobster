from typing import Callable, Dict, List, Literal

import lightning as L
import torch
from torch import Tensor

from lobster.constants import Modality, ModalityType
from lobster.model.modern_bert import FlexBERT
from lobster.tokenization import UmeTokenizerTransform


class Ume(L.LightningModule):
    """Universal Molecular Encoder.

    A light wrapper around FlexBert model with useful high-level functions
    for molecular encoding across different modalities.

    Parameters
    ----------
    freeze : bool, default=True
        Whether to freeze the model parameters.

    Attributes
    ----------
    freeze : bool
        Indicates whether the model parameters are frozen.
    model : FlexBERT or None
        The underlying FlexBERT model for encoding.
    tokenizers : Dict[Modality, Callable]
        Dictionary mapping modality enums to their respective tokenizers.
    modalities : List[str]
        List of supported modalities.

    Examples
    --------
    >>> # Initialize and load from a checkpoint
    >>> encoder = Ume.load_from_checkpoint("path/to/checkpoint.ckpt")
    >>>
    >>> # Get embeddings for protein sequences
    >>> sequences = ["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"]
    >>> embeddings = encoder.get_embeddings(sequences, "amino_acid")
    >>> print(embeddings.shape)
    torch.Size([1, 768])
    """

    def __init__(
        self,
        model_name: Literal["UME_mini", "UME_small", "UME_medium", "UME_large"] = "UME_mini",
        max_length: int = 512,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.98,
        eps: float = 1e-12,
        mask_percentage: float = 0.25,
        scheduler: Literal[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
            "inverse_sqrt",
            "reduce_lr_on_plateau",
            "cosine_with_min_lr",
            "warmup_stable_decay",
        ] = "constant_with_warmup",
        num_training_steps: int | None = None,
        num_warmup_steps: int | None = 1_000,
        model_kwargs: dict = None,
        scheduler_kwargs: dict = None,
        ckpt_path: str = None,
    ) -> None:
        """Initialize the Universal Molecular Encoder"""
        super().__init__()

        self.save_hyperparameters()

        # Instantiate tokenizer transforms for each modality
        self.tokenizer_transforms = {
            modality: UmeTokenizerTransform(modality, max_length=max_length, return_modality=True)
            for modality in Modality
        }

        # Get any tokenizer to get the special tokens
        tokenizer = list(self.tokenizer_transforms.values())[0].tokenizer

        # Instantiate the model
        self.model = FlexBERT(
            model_name=model_name,
            max_length=max_length,
            vocab_size=len(self.get_vocab()),
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            num_training_steps=num_training_steps,
            num_warmup_steps=num_warmup_steps,
            mask_percentage=mask_percentage,
            scheduler=scheduler,
            model_kwargs=model_kwargs,
            scheduler_kwargs=scheduler_kwargs,
            pad_token_id=tokenizer.pad_token_id,
            mask_token_id=tokenizer.mask_token_id,
            cls_token_id=tokenizer.cls_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        self.embedding_dim = max_length
        self.frozen = False

    @property
    def modalities(self) -> List[str]:
        """List of supported modalities.

        Returns
        -------
        List[str]
            The list of supported modality names as strings.

        Examples
        --------
        >>> encoder = Ume.load_from_checkpoint("path/to/checkpoint.ckpt")
        >>> print(encoder.modalities)
        ['SMILES', 'amino_acid', 'nucleotide', '3d_coordinates']
        """
        return [modality.value for modality in Modality]

    def get_tokenizer(self, modality: ModalityType | Modality) -> Callable:
        """Get the appropriate tokenizer for the given modality.

        Parameters
        ----------
        modality : Literal["SMILES", "amino_acid", "nucleotide", "3d_coordinates"] | Modality
            The modality to use for encoding.

        Returns
        -------
        Callable
            The appropriate tokenizer for the specified modality.

        Examples
        --------
        >>> encoder = Ume.load_from_checkpoint("path/to/checkpoint.ckpt")
        >>> # Get tokenizer for amino acid sequences
        >>> tokenizer = encoder.get_tokenizer("amino_acid")
        >>> sequences = ["MKTVRQERLKSIVRILERSKEPVSGAQL"]
        >>> tokens = tokenizer(sequences, return_tensors="pt")
        >>> print(tokens.keys())
        dict_keys(['input_ids', 'attention_mask'])
        >>>
        >>> # Get tokenizer for nucleotide sequences
        >>> dna_tokenizer = encoder.get_tokenizer("nucleotide")
        >>> dna_sequences = ["ATGCATTGCA"]
        >>> dna_tokens = dna_tokenizer(dna_sequences, return_tensors="pt")
        """
        modality_enum = Modality(modality) if isinstance(modality, str) else modality

        return self.tokenizer_transforms[modality_enum].tokenizer

    def get_vocab(self) -> Dict[int, str]:
        """Get a consolidated vocabulary from all tokenizers.

        Returns
        -------
        Dict[int, str]
            A dictionary mapping token IDs to token strings, sorted by token ID.
            Reserved tokens are excluded.
            Important! Tokens are not unique across modalities and may overlap.
            If the vocabulary is reversed where token strings are keys,
            information will be lost. Use with caution.

        Examples
        --------
        >>> encoder = Ume.load_from_checkpoint("path/to/checkpoint.ckpt")
        >>> vocab = encoder.get_vocab()
        >>> print(len(vocab))  # Size of vocabulary
        1536  # Example size
        """
        tokenizers = [transform.tokenizer for transform in self.tokenizer_transforms.values()]

        vocab = {
            token_id: token
            for tokenizer in tokenizers
            for token, token_id in tokenizer.get_vocab().items()
            if "reserved" not in token
        }

        return dict(sorted(vocab.items(), key=lambda item: item[0]))

    def freeze(self) -> None:
        """Freeze the model parameters."""
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()
        self.frozen = True

    def unfreeze(self) -> None:
        """Unfreeze the model parameters."""
        for param in self.model.parameters():
            param.requires_grad = True

        self.model.train()
        self.frozen = False

    def get_embeddings(self, inputs: List[str], modality: ModalityType, aggregate: bool = True) -> Tensor:
        """Get embeddings for the provided inputs using the specified modality.

        Parameters
        ----------
        inputs : List[str]
            List of input strings to encode.
        modality : Literal["SMILES", "amino_acid", "nucleotide", "3d_coordinates"]
            The modality to use for encoding.
        aggregate : bool, default=True
            Whether to average pool over the sequence length dimension.

        Returns
        -------
        Tensor
            Tensor of embeddings. If aggregate=True, shape is (batch_size, hidden_size).
            Otherwise, shape is (batch_size, seq_len, hidden_size).

        Raises
        ------
        ValueError
            If the model has not been initialized with a checkpoint.

        Examples
        --------
        >>> # Get protein embeddings
        >>> encoder = Ume.load_from_checkpoint("path/to/checkpoint.ckpt")
        >>> sequences = ["MKTVQRERL", "ACDEFGHIKL"]
        >>> embeddings = encoder.get_embeddings(sequences, "amino_acid")
        >>> print(embeddings.shape)
        torch.Size([2, 768])
        >>>
        >>> # Get token-level embeddings for DNA sequences
        >>> dna_seqs = ["ATGCATGC", "GCTAGCTA"]
        >>> token_embeddings = encoder.get_embeddings(dna_seqs, "nucleotide", aggregate=False)
        >>> print(token_embeddings.shape)
        torch.Size([2, 8, 768])  # [batch_size, seq_len, hidden_dim]
        >>>
        >>> # Get SMILES embeddings
        >>> smiles = ["CC(=O)OC1=CC=CC=C1C(=O)O", "CCO"]
        >>> smiles_embeddings = encoder.get_embeddings(smiles, "SMILES")
        >>> print(smiles_embeddings.shape)
        torch.Size([2, 768])
        """
        if self.model is None:
            raise ValueError(
                "Model has not been initialized. Load a model from a checkpoint. "
                "Example: `Ume.load_from_checkpoint('path/to/checkpoint.ckpt')`"
            )

        modality_enum = Modality(modality)
        tokenizer_transform = self.tokenizer_transforms[modality_enum]

        tokenized_inputs = tokenizer_transform(inputs)

        x = {k: v.to(self.model.device) for k, v in tokenized_inputs.items() if isinstance(v, Tensor)}

        if self.frozen:
            with torch.no_grad():
                embeddings = self.model.tokens_to_latents(**x)
        else:
            embeddings = self.model.tokens_to_latents(**x)

        # Reshape to (batch_size, seq_len, hidden_size)
        batch_size = x["input_ids"].size(0)
        seq_len = x["input_ids"].size(-1)
        embeddings = embeddings.view(batch_size, seq_len, -1)

        if aggregate:
            # Simple mean pooling over sequence length dimension
            return embeddings.mean(dim=1)
        else:
            return embeddings

    def configure_optimizers(self) -> dict[str, object]:
        return self.model.configure_optimizers()

    def _step(self, batch: dict[str, Tensor | list[Modality]], stage: Literal["train", "val"]) -> Tensor:
        modalities = batch.pop("modality")

        loss, per_sample_loss = self.model._compute_loss(batch, return_per_sample_loss=True)

        perplexity = torch.exp(loss)
        per_sample_perplexity = torch.exp(per_sample_loss)

        self.log(f"{stage}_loss", loss, sync_dist=True)
        self.log(f"{stage}_perplexity", perplexity, sync_dist=True)

        # Log loss and perplexity for each modality
        for modality in set(modalities):
            mask = torch.tensor([m == modality for m in modalities], device=self.device, dtype=torch.bool)

            modality_loss = per_sample_loss[mask].mean()
            modality_perplexity = per_sample_perplexity[mask].mean()

            self.log(f"{stage}_loss_{modality.value}", modality_loss, sync_dist=True)
            self.log(f"{stage}_perplexity_{modality.value}", modality_perplexity, sync_dist=True)

        return loss

    def training_step(self, batch: dict[str, Tensor | list[Modality]], batch_idx: int) -> Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: dict[str, Tensor | list[Modality]], batch_idx: int) -> Tensor:
        return self._step(batch, "val")
