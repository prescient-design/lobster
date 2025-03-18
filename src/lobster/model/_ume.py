from typing import Callable, Literal

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
    checkpoint : str
        Path to model checkpoint file.
    freeze : bool, default=True
        Whether to freeze the model parameters.

    Examples
    --------
    >>> # Initialize with checkpoint
    >>> encoder = Ume("path/to/checkpoint.ckpt")
    >>>
    >>> # Get embeddings for protein sequences
    >>> sequences = ["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"]
    >>> embeddings = encoder.get_embeddings(sequences, "amino_acid")
    >>> print(embeddings.shape)
    torch.Size([1, 768])
    """

    def __init__(self, checkpoint: str, freeze: bool = True) -> None:
        super().__init__()

        self.freeze = freeze
        self.model = FlexBERT.load_from_checkpoint(checkpoint)

        # Freeze the model parameters
        if self.freeze:
            for param in self.model.model.parameters():
                param.requires_grad = False

    def get_tokenizer(self, inputs: list[str], modality: ModalityType) -> Callable:
        """Get the appropriate tokenizer for the given modality.

        Parameters
        ----------
        inputs : list[str]
            List of input strings to encode.
        modality : Literal["SMILES", "amino_acid", "nucleotide", "3d_coordinates"]
            The modality to use for encoding.

        Returns
        -------
        Callable
            The appropriate tokenizer for the specified modality.

        Examples
        --------
        >>> encoder = Ume("path/to/checkpoint.ckpt")
        >>> sequences = ["MKTVRQERLKSIVRILERSKEPVSGAQL"]
        >>> tokenizer = encoder.get_tokenizer(sequences, "amino_acid")
        >>> tokens = tokenizer(sequences, return_tensors="pt")
        """
        mod_enum = Modality(modality)

        if mod_enum == Modality.AMINO_ACID:
            tokenizer_class = UmeAminoAcidTokenizerFast

        elif mod_enum == Modality.NUCLEOTIDE:
            tokenizer_class = UmeNucleotideTokenizerFast

        elif mod_enum == Modality.SMILES:
            tokenizer_class = UmeSmilesTokenizerFast

        elif mod_enum == Modality.COORDINATES_3D:
            tokenizer_class = UmeLatentGenerator3DCoordTokenizerFast

        return tokenizer_class()

    def get_embeddings(self, inputs: list[str], modality: ModalityType, aggregate: bool = True) -> Tensor:
        """Get embeddings for the provided inputs using the specified modality.

        Parameters
        ----------
        inputs : list[str]
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

        Examples
        --------
        >>> # Get protein embeddings
        >>> encoder = Ume("path/to/checkpoint.ckpt")
        >>> sequences = ["MKTVQRERL", "ACDEFGHIKL"]
        >>> embeddings = encoder.get_embeddings(sequences, "amino_acid")
        >>> print(embeddings.shape)
        torch.Size([2, 768])

        >>> # Get token-level embeddings for DNA sequences
        >>> dna_seqs = ["ATGCATGC", "GCTAGCTA"]
        >>> token_embeddings = encoder.get_embeddings(dna_seqs, "nucleotide", aggregate=False)
        >>> print(token_embeddings.shape)
        torch.Size([2, 512, 768])
        """
        tokenizer = self.get_tokenizer(inputs, modality)

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
