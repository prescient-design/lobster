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
    model_name : Literal["UME_mini", "UME_small", "UME_medium", "UME_large"], default="UME_mini"
        Name of the model to initialize.
    max_length : int, default=512
        Maximum sequence length for tokenization.
    lr : float, default=1e-3
        Learning rate for optimizer.
    beta1 : float, default=0.9
        Beta1 parameter for Adam optimizer.
    beta2 : float, default=0.98
        Beta2 parameter for Adam optimizer.
    eps : float, default=1e-12
        Epsilon parameter for Adam optimizer.
    mask_percentage : float, default=0.25
        Percentage of tokens to mask during training.
    scheduler : str, default="constant_with_warmup"
        Type of learning rate scheduler to use.
    num_training_steps : int | None, default=None
        Total number of training steps.
    num_warmup_steps : int | None, default=1_000
        Number of warmup steps for learning rate scheduler.
    model_kwargs : dict | None, default=None
        Additional keyword arguments to pass to the FlexBERT model.
    scheduler_kwargs : dict | None, default=None
        Additional keyword arguments to pass to the learning rate scheduler.
    ckpt_path : str | None, default=None
        Path to a checkpoint file to load.

    Attributes
    ----------
    model : FlexBERT
        The underlying FlexBERT model for encoding.
    tokenizer_transforms : Dict[Modality, UmeTokenizerTransform]
        Dictionary mapping modality enums to their respective tokenizer transforms.
    embedding_dim : int
        Dimension of the output embeddings.
    frozen : bool
        Indicates whether model parameters are frozen.

    Examples
    --------
    >>> # Initialize a new model
    >>> encoder = Ume(model_name="UME_mini", max_length=256)
    >>>
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
        model_kwargs: dict | None = None,
        scheduler_kwargs: dict | None = None,
        ckpt_path: str | None = None,
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
        if ckpt_path is not None:
            self.model = FlexBERT.load_from_checkpoint(ckpt_path)
        else:
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
        >>> encoder = Ume(model_name="UME_mini")
        >>> print(encoder.modalities)
        ['SMILES', 'amino_acid', 'nucleotide', '3d_coordinates']

        >>> # Load from checkpoint and check supported modalities
        >>> encoder = Ume(ckpt_path="path/to/checkpoint.ckpt")
        >>> print(encoder.modalities)
        ['SMILES', 'amino_acid', 'nucleotide', '3d_coordinates']
        """
        return [modality.value for modality in Modality]

    def get_tokenizer(self, modality: ModalityType | Modality) -> Callable:
        """Get the appropriate tokenizer for the given modality.

        Parameters
        ----------
        modality : str | Modality
            The modality to use for encoding. Can be a string ("SMILES", "amino_acid",
            "nucleotide", "3d_coordinates") or a Modality enum.

        Returns
        -------
        Callable
            The appropriate tokenizer for the specified modality.

        Examples
        --------
        >>> encoder = Ume(model_name="UME_mini")
        >>>
        >>> # Get tokenizer for amino acid sequences
        >>> tokenizer = encoder.get_tokenizer("amino_acid")
        >>> sequences = ["MKTVRQERLKSIVRILERSKEPVSGAQL"]
        >>> tokens = tokenizer(sequences, return_tensors="pt")
        >>> print(tokens.keys())
        dict_keys(['input_ids', 'attention_mask'])
        >>>
        >>> # Get tokenizer for nucleotide sequences using Modality enum
        >>> from lobster.constants import Modality
        >>> dna_tokenizer = encoder.get_tokenizer(Modality.NUCLEOTIDE)
        >>> dna_sequences = ["ATGCATTGCA"]
        >>> dna_tokens = dna_tokenizer(dna_sequences, return_tensors="pt")
        >>> print(dna_tokens["input_ids"].shape)
        torch.Size([1, 12])  # Including special tokens
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
        >>> encoder = Ume(model_name="UME_mini")
        >>> vocab = encoder.get_vocab()
        >>> print(len(vocab))  # Size of vocabulary
        1536  # Example size
        >>>
        >>> # Check some token mappings
        >>> some_token_id = 42
        >>> if some_token_id in vocab:
        ...     print(f"Token ID {some_token_id} corresponds to: {vocab[some_token_id]}")
        Token ID 42 corresponds to: C
        >>>
        >>> # Filter vocab by a specific pattern
        >>> amino_acid_tokens = {id: token for id, token in vocab.items() if token in "ACDEFGHIKLMNPQRSTVWY"}
        >>> print(f"Number of amino acid tokens: {len(amino_acid_tokens)}")
        Number of amino acid tokens: 20
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
        """Freeze the model parameters.

        This method sets requires_grad=False for all model parameters
        and puts the model in evaluation mode.

        Examples
        --------
        >>> encoder = Ume(model_name="UME_mini")
        >>> # Check if model is trainable
        >>> print(f"Before freezing - Parameter grad enabled: {next(encoder.model.parameters()).requires_grad}")
        Before freezing - Parameter grad enabled: True
        >>>
        >>> # Freeze the model
        >>> encoder.freeze()
        >>> print(f"After freezing - Parameter grad enabled: {next(encoder.model.parameters()).requires_grad}")
        After freezing - Parameter grad enabled: False
        >>> print(f"Model is frozen: {encoder.frozen}")
        Model is frozen: True
        >>>
        >>> # Now you can use it for inference without gradient computation
        >>> import torch
        >>> with torch.no_grad():
        ...     embeddings = encoder.get_embeddings(["ACDEFGHIK"], "amino_acid")
        """
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()
        self.frozen = True

    def unfreeze(self) -> None:
        """Unfreeze the model parameters.

        This method sets requires_grad=True for all model parameters
        and puts the model in training mode.

        Examples
        --------
        >>> encoder = Ume(model_name="UME_mini")
        >>> # First freeze the model
        >>> encoder.freeze()
        >>> print(f"Model is frozen: {encoder.frozen}")
        Model is frozen: True
        >>>
        >>> # Now unfreeze it
        >>> encoder.unfreeze()
        >>> print(f"After unfreezing - Parameter grad enabled: {next(encoder.model.parameters()).requires_grad}")
        After unfreezing - Parameter grad enabled: True
        >>> print(f"Model is frozen: {encoder.frozen}")
        Model is frozen: False
        >>>
        >>> # Now you can continue training the model
        >>> optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-4)
        >>> # ... train the model ...
        """
        for param in self.model.parameters():
            param.requires_grad = True

        self.model.train()
        self.frozen = False

    def get_embeddings(self, inputs: List[str], modality: ModalityType | Modality, aggregate: bool = True) -> Tensor:
        """Get embeddings for the provided inputs using the specified modality.

        Parameters
        ----------
        inputs : List[str]
            List of input strings to encode.
        modality : str | Modality
            The modality to use for encoding. Can be a string ("SMILES", "amino_acid",
            "nucleotide", "3d_coordinates") or a Modality enum.
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
        >>> encoder = Ume(model_name="UME_mini")
        >>> sequences = ["MKTVQRERL", "ACDEFGHIKL"]
        >>> embeddings = encoder.get_embeddings(sequences, "amino_acid")
        >>> print(embeddings.shape)
        torch.Size([2, 768])
        >>>
        >>> # Get token-level embeddings for DNA sequences
        >>> dna_seqs = ["ATGCATGC", "GCTAGCTA"]
        >>> token_embeddings = encoder.get_embeddings(dna_seqs, "nucleotide", aggregate=False)
        >>> print(token_embeddings.shape)
        torch.Size([2, 10, 768])  # [batch_size, seq_len, hidden_dim] (includes special tokens)
        >>>
        >>> # Get SMILES embeddings using Modality enum
        >>> from lobster.constants import Modality
        >>> smiles = ["CC(=O)OC1=CC=CC=C1C(=O)O", "CCO"]
        >>> smiles_embeddings = encoder.get_embeddings(smiles, Modality.SMILES)
        >>> print(smiles_embeddings.shape)
        torch.Size([2, 768])
        >>>
        >>> # Process 3D coordinates
        >>> coords = ["0.0 0.0 0.0;1.0 0.0 0.0;0.0 1.0 0.0", "0.0 0.0 0.0;1.0 1.0 0.0"]
        >>> coord_embeddings = encoder.get_embeddings(coords, "3d_coordinates")
        >>> print(coord_embeddings.shape)
        torch.Size([2, 768])
        """
        if self.model is None:
            raise ValueError(
                "Model has not been initialized. Load a model from a checkpoint. "
                "Example: `Ume.load_from_checkpoint('path/to/checkpoint.ckpt')`"
            )

        modality_enum = Modality(modality) if isinstance(modality, str) else modality
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
        """Configure optimizers and learning rate schedulers.

        Returns
        -------
        dict[str, object]
            Dictionary containing optimizer and learning rate scheduler.

        Examples
        --------
        >>> encoder = Ume(model_name="UME_mini", lr=5e-5, num_training_steps=10000)
        >>> optimizers_config = encoder.configure_optimizers()
        >>> print(type(optimizers_config["optimizer"]))
        <class 'torch.optim.adam.Adam'>
        >>> print(type(optimizers_config["lr_scheduler"]["scheduler"]))
        <class 'transformers.optimization.get_constant_schedule_with_warmup.<locals>.ConstantLRWithWarmup'>
        """
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

            self.log(f"{stage}_loss/{modality.value}", modality_loss, sync_dist=True)
            self.log(f"{stage}_perplexity/{modality.value}", modality_perplexity, sync_dist=True)

        return loss

    def training_step(self, batch: dict[str, Tensor | list[Modality]], batch_idx: int) -> Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: dict[str, Tensor | list[Modality]], batch_idx: int) -> Tensor:
        return self._step(batch, "val")
