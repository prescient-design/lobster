import warnings
from collections import defaultdict

from torch import Tensor
import torch
from torch.nn import Module

from lobster.constants import Modality, ModalityType
from ._detect_modality import detect_modality
from ._ume_tokenizers import (
    UMEAminoAcidTokenizerFast,
    UMESmilesTokenizerFast,
    UMENucleotideTokenizerFast,
    get_special_tokens,
)


class UMETokenizerTransform(Module):
    """
    Enhanced UME tokenizer transform supporting automatic modality detection and mixed modalities.

    This tokenizer instantiates all three tokenizers and allows you to specify modality at tokenization time,
    making it highly reusable across different sequences and batches.

    Usage modes:
    1. Explicit modality: Specify modality parameter in forward() call
    2. Auto-detection: Pass modality=None to automatically detect each sequence's modality
    3. Mixed modality: Auto-detect and handle different modalities in the same batch

    Examples
    --------
    # Create one tokenizer instance for all use cases
    >>> tokenizer = UMETokenizerTransform(max_length=12, return_modality=True)

    # Use with explicit modality
    >>> result = tokenizer("MYK", modality="amino_acid")

    # Use with auto-detection for single sequence
    >>> result = tokenizer("CCO", modality=None)  # Auto-detects as SMILES

    # Use with mixed modalities in batch
    >>> result = tokenizer(["MYK", "CCO", "ATGC"], modality=None)  # Auto-detects each

    # Use with explicit modality for batch
    >>> result = tokenizer(["MYK", "AVYK"], modality="amino_acid")

    Parameters
    ----------
    max_length : int or None
        Maximum sequence length. If None, no padding/truncation.
    return_modality : bool, optional
        Whether to return modality info in output. Default False.
    add_special_tokens : bool, optional
        Add special tokens. Default True.
    padding : str, optional
        Padding strategy. Default "max_length".
    validate_modality : bool, optional
        Whether to validate detected modalities using BioPython/RDKit. Default False.
    """

    def __init__(
        self,
        max_length: int | None,
        return_modality: bool = False,
        add_special_tokens: bool = True,
        padding: str = "max_length",
        validate_modality: bool = False,
        seed: int | None = 0,
    ):
        super().__init__()

        if max_length is None:
            warnings.warn(
                "UMETokenizerTransform did not receive `max_length` parameter.Padding and truncation will not be applied.",
                UserWarning,
                stacklevel=2,
            )

        self.max_length = max_length
        self.return_modality = return_modality
        self.add_special_tokens = add_special_tokens
        self.padding = padding
        self.validate_modality = validate_modality
        self.seed = seed

        self.tokenizers = {
            Modality.AMINO_ACID: UMEAminoAcidTokenizerFast(),
            Modality.SMILES: UMESmilesTokenizerFast(),
            Modality.NUCLEOTIDE: UMENucleotideTokenizerFast(),
        }

    def _encode(self, items: str | list[str], modality: Modality) -> dict[str, Tensor]:
        """Encode items using a specific modality tokenizer."""
        tokenizer = self.tokenizers[modality]

        return tokenizer(
            items,
            max_length=self.max_length,
            padding=self.padding if self.max_length else False,
            truncation=True if self.max_length else False,
            add_special_tokens=self.add_special_tokens,
            return_tensors="pt",
        )

    def _encode_mixed_modalities(self, items: list[str]) -> dict[str, Tensor | list[Modality]]:
        """
        Encode mixed modality items by grouping, tokenizing separately, then reassembling.
        """
        # Step 1: Detect modalities and group sequences
        modality_groups = defaultdict(list)
        modality_indices = defaultdict(list)
        sequence_modalities = []

        for idx, item in enumerate(items):
            modality = detect_modality(item)
            modality_groups[modality].append(item)
            modality_indices[modality].append(idx)
            sequence_modalities.append(modality)

        # Step 2: Tokenize each modality group
        all_input_ids = [None] * len(items)
        all_attention_masks = [None] * len(items)

        for modality, sequences in modality_groups.items():
            encoded = self._encode(sequences, modality)

            for i, orig_idx in enumerate(modality_indices[modality]):
                all_input_ids[orig_idx] = encoded["input_ids"][i]
                all_attention_masks[orig_idx] = encoded["attention_mask"][i]

        stacked_input_ids = torch.stack(all_input_ids)
        stacked_attention_masks = torch.stack(all_attention_masks)

        result = {
            "input_ids": stacked_input_ids,
            "attention_mask": stacked_attention_masks,
        }

        if self.return_modality:
            result["modality"] = sequence_modalities

        return result

    def forward(
        self,
        item: str | list[str],
        modality: ModalityType | str | None = None,
    ) -> dict[str, Tensor | Modality | list[Modality]]:
        """
        Tokenize input with specified or automatically detected modality.

        Parameters
        ----------
        item : str or list[str]
            Input to tokenize. Examples: "MYK", ["MYK", "CCO", "ATGC"]
        modality : Literal["amino_acid", "smiles", "nucleotide"], str, or None
            Modality to use for tokenization. If None, automatically detects modality for each input.

        Returns
        -------
        dict
            Tokenized output with keys:
                - "input_ids": Tensor of token IDs
                - "attention_mask": Tensor of attention masks
                - "modality": Modality type or list of modalities (if return_modality is True)
        """
        if modality is not None and isinstance(modality, str):
            modality = Modality(modality)

        if isinstance(item, str):
            single_sequence = True
            item = [item]
        else:
            single_sequence = False

        if modality is not None:
            output = self._encode(item, modality)

            if self.return_modality:
                output["modality"] = modality if single_sequence else [modality] * len(item)

        else:
            output = self._encode_mixed_modalities(item)

            if self.return_modality and single_sequence:
                output["modality"] = output["modality"][0]

        return output

    def get_special_tokens(self) -> dict[str, int]:
        """Get special tokens for all modalities."""
        return get_special_tokens()

    def get_vocab(self) -> dict[str, int]:
        """Get vocabulary for all modalities."""
        vocab = {
            token_id: token
            for tokenizer in self.tokenizers.values()
            for token, token_id in tokenizer.get_vocab().items()
            if "reserved" not in token
        }

        return dict(sorted(vocab.items(), key=lambda item: item[0]))

    @property
    def vocab_size(self) -> int:
        return len(self.get_vocab())

    @property
    def pad_token_id(self) -> int:
        return self.tokenizers[Modality.AMINO_ACID].pad_token_id

    @property
    def pad_token(self) -> str:
        return self.tokenizers[Modality.AMINO_ACID].pad_token

    @property
    def mask_token_id(self) -> int:
        return self.tokenizers[Modality.AMINO_ACID].mask_token_id

    @property
    def mask_token(self) -> str:
        return self.tokenizers[Modality.AMINO_ACID].mask_token

    @property
    def eos_token_id(self) -> int:
        return self.tokenizers[Modality.AMINO_ACID].eos_token_id

    @property
    def eos_token(self) -> str:
        return self.tokenizers[Modality.AMINO_ACID].eos_token

    @property
    def unk_token_id(self) -> int:
        return self.tokenizers[Modality.AMINO_ACID].unk_token_id

    @property
    def unk_token(self) -> str:
        return self.tokenizers[Modality.AMINO_ACID].unk_token

    @property
    def cls_token_id(self) -> int:
        return self.tokenizers[Modality.AMINO_ACID].cls_token_id

    @property
    def cls_token(self) -> str:
        return self.tokenizers[Modality.AMINO_ACID].cls_token

    @property
    def sep_token_id(self) -> int:
        return self.tokenizers[Modality.AMINO_ACID].sep_token_id

    @property
    def sep_token(self) -> str:
        return self.tokenizers[Modality.AMINO_ACID].sep_token
