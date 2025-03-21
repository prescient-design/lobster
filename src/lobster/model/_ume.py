from typing import Callable, Dict, List, Literal

import lightning as L
import torch
from torch import Tensor

from lobster.constants import Modality
from lobster.model.modern_bert import FlexBERT
from lobster.tokenization import (
    UmeAminoAcidTokenizerFast,
    UmeLatentGenerator3DCoordTokenizerFast,
    UmeNucleotideTokenizerFast,
    UmeSmilesTokenizerFast,
)

ModalityType = Literal["SMILES", "amino_acid", "nucleotide", "3d_coordinates"]


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

    def __init__(self) -> None:
        """Initialize the Universal Molecular Encoder.

        Examples
        --------
        >>> # Create an instance with default settings
        >>> encoder = Ume()
        >>> encoder.modalities
        ['SMILES', 'amino_acid', 'nucleotide', '3d_coordinates']

        >>> encoder.get_vocab()
        {0: '<cls>', 1: '<pad>', 2: '<eos>', ...}
        """
        super().__init__()

        self.model: FlexBERT | None = None
        self.freeze: bool | None = None

        # Initialize tokenizers for all modalities
        self.tokenizers: Dict[Modality, Callable] = {
            Modality.AMINO_ACID: UmeAminoAcidTokenizerFast(),
            Modality.NUCLEOTIDE: UmeNucleotideTokenizerFast(),
            Modality.SMILES: UmeSmilesTokenizerFast(),
            Modality.COORDINATES_3D: UmeLatentGenerator3DCoordTokenizerFast(),
        }

    @classmethod
    def load_from_checkpoint(cls, checkpoint: str, freeze: bool = True) -> "Ume":
        """Load the Ume model from a checkpoint.

        Parameters
        ----------
        checkpoint : str
            Path to the model checkpoint file.
        freeze : bool, default=True
            Whether to freeze the model parameters.

        Returns
        -------
        Ume
            An instance of the Ume class initialized with the checkpoint.

        Examples
        --------
        >>> # Load with default frozen parameters
        >>> encoder = Ume.load_from_checkpoint("path/to/checkpoint.ckpt")
        >>>
        >>> # Load with unfrozen parameters for fine-tuning
        >>> encoder_trainable = Ume.load_from_checkpoint("path/to/checkpoint.ckpt", freeze=False)
        """
        instance = cls()
        instance.model = FlexBERT.load_from_checkpoint(checkpoint)
        instance.freeze = freeze

        # Apply freezing if requested
        if freeze:
            for param in instance.model.model.parameters():
                param.requires_grad = False

        return instance

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

    def get_tokenizer(self, modality: ModalityType) -> Callable:
        """Get the appropriate tokenizer for the given modality.

        Parameters
        ----------
        modality : Literal["SMILES", "amino_acid", "nucleotide", "3d_coordinates"]
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
        modality_enum = Modality(modality)

        return self.tokenizers[modality_enum]

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
        vocab = {
            token_id: token
            for tokenizer in self.tokenizers.values()
            for token, token_id in tokenizer.get_vocab().items()
            if "reserved" not in token
        }

        return dict(sorted(vocab.items(), key=lambda item: item[0]))

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

        tokenizer = self.get_tokenizer(modality)

        items = tokenizer(
            inputs, return_tensors="pt", padding="max_length", truncation=True, max_length=self.model.max_length
        )

        with torch.no_grad():
            x = {k: v.to(self.model.device) for k, v in items.items()}

            # Get token-level embeddings
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
