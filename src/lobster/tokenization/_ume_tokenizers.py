"""Ume tokenizers for amino acids, SMILES, nucleotides, and 3D latent generator coordinates.

Creates tokenizers with shared special tokens and reserved tokens to make sure there are
no overlapping tokens between different modalities.

Vocabulary structure:
- Special tokens: ["<cls>", "<eos>", "<unk>", "<pad>"]
- Conversion and interaction tokens: ["<convert>", "<interact>"]
- Extra special tokens to get % 64 == 0: ["<extra_special_token_2>", "<extra_special_token_3>", ...]
- Amino acid tokenizer: [special_tokens] + ["A", "C", "D", ...]
- SMILES tokenizer: [special_tokens] + [reserved_for_amino_acids] + ["C", "O", "N", ...]
- Nucleotide tokenizer: [special_tokens] + [reserved_for_amino_acids] + [reserved_for_SMILES] + ["A", "C", "G", ...]
- Latent generator tokenizer: [special_tokens] + [reserved_for_amino_acids] + [reserved_for_SMILES] +
  [reserved_for_nucleotides] + ["gh", "fh", "ds", ...]

To create the tokenizers, run

```python
    from lobster.tokenization._ume_tokenizers import (
        _make_ume_tokenizers,
        UmeAminoAcidTokenizerFast,
        UmeSmilesTokenizerFast,
        UmeNucleotideTokenizerFast,
        UmeLatentGenerator3DCoordTokenizerFast,
    )
    # Create and save tokenizers
    _make_ume_tokenizers()

    tokenizers = [
            UmeAminoAcidTokenizerFast(),
            UmeSmilesTokenizerFast(),
            UmeNucleotideTokenizerFast(),
            UmeLatentGenerator3DCoordTokenizerFast(),
    ]

    # Compute the total vocabulary size
    vocab = {
        token_id: token for tokenizer in tokenizers
        for token, token_id in tokenizer.get_vocab().items()
        if "reserved" not in token
    }

    print(f"Total vocabulary size = {len(vocab)}")  # 1536
    print(f"Vocab size % 64 = {len(vocab) % 64}")  # 0
```
"""

import importlib.resources
import warnings
from pathlib import Path
from typing import Literal

import torch
from tokenizers.models import BPE, WordLevel
from tokenizers.normalizers import Lowercase
from tokenizers.pre_tokenizers import Sequence as PreTokenizerSequence
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.processors import TemplateProcessing
from torch import Tensor
from torch.nn import Module
from transformers import PreTrainedTokenizerFast

import lobster.transforms.functional
from lobster.constants import Modality, ModalityType

from ._latent_generator_3d_coord_tokenizer import VOCAB_PATH as LATENT_GENERATOR_VOCAB_PATH
from ._load_vocab_file import load_vocab_file
from ._make_pretrained_tokenizer_fast import make_pretrained_tokenizer_fast
from ._smiles_tokenizer import VOCAB_PATH as SMILES_VOCAB_PATH

TOKENIZERS_PATH = importlib.resources.files("lobster") / "assets" / "ume_tokenizers"

# Tokenizer names for saving
AMINO_ACID_TOKENIZER = "amino_acid_tokenizer"
SMILES_TOKENIZER = "smiles_tokenizer"
NUCLEOTIDE_TOKENIZER = "nucleotide_tokenizer"
LATENT_GENERATOR_TOKENIZER = "latent_generator_tokenizer"

AMINO_ACID_VOCAB_PATH = TOKENIZERS_PATH / AMINO_ACID_TOKENIZER / "vocab.txt"
NUCLEOTIDE_VOCAB_PATH = TOKENIZERS_PATH / NUCLEOTIDE_TOKENIZER / "vocab.txt"

# Special tokens
CLS_TOKEN = "<cls>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
SEP_TOKEN = "<sep>"
MASK_TOKEN = "<mask>"
SPECIAL_TOKENS = {CLS_TOKEN, EOS_TOKEN, UNK_TOKEN, PAD_TOKEN, SEP_TOKEN, MASK_TOKEN}


def _load_file(filepath: str | Path, remove_special_tokens: bool = False) -> list[str]:
    vocab = load_vocab_file(filepath)

    if remove_special_tokens:
        return [token for token in vocab if token not in SPECIAL_TOKENS]

    return vocab


def _load_vocabularies() -> dict[str, list[str]]:
    return {
        "special_tokens": _load_file(TOKENIZERS_PATH / "special_tokens.txt"),
        "amino_acid_tokenizer": _load_file(AMINO_ACID_VOCAB_PATH, remove_special_tokens=True),
        "smiles_tokenizer": _load_file(SMILES_VOCAB_PATH, remove_special_tokens=True),
        "nucleotide_tokenizer": _load_file(NUCLEOTIDE_VOCAB_PATH, remove_special_tokens=True),
        "latent_generator_tokenizer": _load_file(LATENT_GENERATOR_VOCAB_PATH, remove_special_tokens=True),
    }


def _add_reserved_tokens(vocabs: dict[str, list[str]]) -> dict[str, list[str]]:
    """
    Add reserved tokens <reserved_special_token_i> to maintain index compatibility
    across tokenizers.

    This function constructs the full vocabulary for each tokenizer by combining:
    - Special tokens (shared across all tokenizers)
    - Reserved/dummy tokens (to maintain index compatibility)
    - Domain-specific tokens for each tokenizer type

    Ordering of tokenizers is important for reserved token construction! Corresponds
    to the order in `TOKENIZER_ORDER`.

    Parameters
    ----------
    vocabs : Dict[str, List[str]]
        Dictionary mapping tokenizer names to their vocabularies

    Returns
    -------
    Dict[str, List[str]]
        Dictionary mapping tokenizer names to their complete vocabularies
        with reserved tokens
    """
    # Find the highest reserved token index in the special tokens
    highest_reserved_index = -1
    for token in vocabs["special_tokens"]:
        if token.startswith("<extra_special_token_"):
            index = int(token.split("_")[-1][:-1])
            highest_reserved_index = max(highest_reserved_index, index)

    # Start reserving from the next index
    next_reserved_index = highest_reserved_index + 1

    # Amino acid tokenizer: special_tokens + amino acid vocab
    vocab_amino_acid = vocabs["special_tokens"] + vocabs["amino_acid_tokenizer"]

    # SMILES tokenizer: special_tokens + [reserved for amino acid tokens] + smiles vocab
    amino_acid_len = len(vocabs["amino_acid_tokenizer"])
    vocab_smiles = (
        vocabs["special_tokens"]
        + [
            f"<reserved_for_amino_acids_special_token_{i}>"
            for i in range(next_reserved_index, next_reserved_index + amino_acid_len)
        ]
        + vocabs["smiles_tokenizer"]
    )

    # Nucleotide tokenizer: special_tokens + [reserved for amino acid] + [reserved for SMILES] + nucleotide vocav
    smiles_len = len(vocabs["smiles_tokenizer"])
    vocab_nucleotide = (
        vocabs["special_tokens"]
        + [
            f"<reserved_for_amino_acids_special_token_{i}>"
            for i in range(next_reserved_index, next_reserved_index + amino_acid_len)
        ]
        + [
            f"<reserved_for_smiles_special_token_{i}>"
            for i in range(next_reserved_index + amino_acid_len, next_reserved_index + amino_acid_len + smiles_len)
        ]
        + vocabs["nucleotide_tokenizer"]
    )

    # Latent generator tokenizer: special_tokens + [reserved for amino acid] + [reserved for SMILES] +
    # [reserved for nucleotide] + latent generator 3D vocab
    nucleotide_len = len(vocabs["nucleotide_tokenizer"])
    vocab_latent = (
        vocabs["special_tokens"]
        + [
            f"<reserved_for_amino_acids_special_token_{i}>"
            for i in range(next_reserved_index, next_reserved_index + amino_acid_len)
        ]
        + [
            f"<reserved_for_smiles_special_token_{i}>"
            for i in range(next_reserved_index + amino_acid_len, next_reserved_index + amino_acid_len + smiles_len)
        ]
        + [
            f"<reserved_for_nucleotides_special_token_{i}>"
            for i in range(
                next_reserved_index + amino_acid_len + smiles_len,
                next_reserved_index + amino_acid_len + smiles_len + nucleotide_len,
            )
        ]
        + vocabs["latent_generator_tokenizer"]
    )

    return {
        AMINO_ACID_TOKENIZER: vocab_amino_acid,
        SMILES_TOKENIZER: vocab_smiles,
        NUCLEOTIDE_TOKENIZER: vocab_nucleotide,
        LATENT_GENERATOR_TOKENIZER: vocab_latent,
    }


def _create_post_processor() -> TemplateProcessing:
    """
    Create a template processor for tokenization.

    The processor formats token sequences by adding special tokens like CLS and EOS
    for both single sequences and pairs of sequences.

    Returns
    -------
    TemplateProcessing
        Configured template processor for token sequence formatting
    """
    return TemplateProcessing(
        single=f"{CLS_TOKEN} $A {EOS_TOKEN}",
        pair=f"{CLS_TOKEN} $A {EOS_TOKEN} $B:1 {EOS_TOKEN}:1",
        special_tokens=[
            (CLS_TOKEN, 0),  # MUST match the order of special tokens in its vocabulary txt file
            (EOS_TOKEN, 2),
        ],
    )


def _make_amino_acid_tokenizer_fast(vocab: list[str]) -> PreTrainedTokenizerFast:
    """
    Create a fast tokenizer for amino acid sequences.

    Parameters
    ----------
    vocab : List[str]
        Complete vocabulary including special and reserved tokens

    Returns
    -------
    PreTrainedTokenizerFast
        Configured fast tokenizer for amino acid sequences
    """
    tokenizer_model = BPE(
        {token: i for i, token in enumerate(vocab)}, merges=[], unk_token=UNK_TOKEN, ignore_merges=True
    )
    post_processor = _create_post_processor()

    return make_pretrained_tokenizer_fast(
        tokenizer_model=tokenizer_model,
        post_processor=post_processor,
        save_dirpath=str(TOKENIZERS_PATH / AMINO_ACID_TOKENIZER),
        cls_token=CLS_TOKEN,
        eos_token=EOS_TOKEN,
        unk_token=UNK_TOKEN,
        sep_token=SEP_TOKEN,
        mask_token=MASK_TOKEN,
        pad_token=PAD_TOKEN,
        bos_token=None,
    )


def _make_smiles_tokenizer_fast(vocab: list[str]) -> PreTrainedTokenizerFast:
    """
    Create a fast tokenizer for SMILES chemical notations.

    Parameters
    ----------
    vocab : List[str]
        Complete vocabulary including special and reserved tokens

    Returns
    -------
    PreTrainedTokenizerFast
        Configured fast tokenizer for SMILES chemical notations
    """
    tokenizer_model = WordLevel(vocab={token: i for i, token in enumerate(vocab)}, unk_token=UNK_TOKEN)
    pre_tokenizer = PreTokenizerSequence([WhitespaceSplit()])
    post_processor = _create_post_processor()

    return make_pretrained_tokenizer_fast(
        tokenizer_model=tokenizer_model,
        pre_tokenizer=pre_tokenizer,
        post_processor=post_processor,
        save_dirpath=str(TOKENIZERS_PATH / SMILES_TOKENIZER),
        cls_token=CLS_TOKEN,
        eos_token=EOS_TOKEN,
        unk_token=UNK_TOKEN,
        sep_token=SEP_TOKEN,
        mask_token=MASK_TOKEN,
        pad_token=PAD_TOKEN,
        bos_token=None,
    )


def _make_nucleotide_tokenizer_fast(vocab: list[str]) -> PreTrainedTokenizerFast:
    """
    Create a fast tokenizer for nucleotide sequences.

    Parameters
    ----------
    vocab : List[str]
        Complete vocabulary including special and reserved tokens

    Returns
    -------
    PreTrainedTokenizerFast
        Configured fast tokenizer for nucleotide sequences
    """
    tokenizer_model = BPE(
        {token: i for i, token in enumerate(vocab)}, merges=[], unk_token=UNK_TOKEN, ignore_merges=True
    )
    post_processor = _create_post_processor()

    return make_pretrained_tokenizer_fast(
        tokenizer_model=tokenizer_model,
        normalizer=Lowercase(),
        post_processor=post_processor,
        save_dirpath=str(TOKENIZERS_PATH / NUCLEOTIDE_TOKENIZER),
        cls_token=CLS_TOKEN,
        eos_token=EOS_TOKEN,
        unk_token=UNK_TOKEN,
        sep_token=SEP_TOKEN,
        mask_token=MASK_TOKEN,
        pad_token=PAD_TOKEN,
        bos_token=None,
    )


def _make_latent_generator_3d_coord_tokenizer_fast(vocab: list[str]) -> PreTrainedTokenizerFast:
    """
    Create a fast tokenizer for 3D latent generator coordinates.

    Parameters
    ----------
    vocab : List[str]
        Complete vocabulary including special and reserved tokens

    Returns
    -------
    PreTrainedTokenizerFast
        Configured fast tokenizer for 3D latent generator coordinates
    """
    tokenizer_model = WordLevel({token: i for i, token in enumerate(vocab)}, unk_token=UNK_TOKEN)
    pre_tokenizer = PreTokenizerSequence([WhitespaceSplit()])
    post_processor = _create_post_processor()

    return make_pretrained_tokenizer_fast(
        tokenizer_model=tokenizer_model,
        pre_tokenizer=pre_tokenizer,
        post_processor=post_processor,
        save_dirpath=str(TOKENIZERS_PATH / LATENT_GENERATOR_TOKENIZER),
        cls_token=CLS_TOKEN,
        eos_token=EOS_TOKEN,
        unk_token=UNK_TOKEN,
        sep_token=SEP_TOKEN,
        mask_token=MASK_TOKEN,
        pad_token=PAD_TOKEN,
        bos_token=None,
    )


def _make_ume_tokenizers() -> None:
    """
    Create and save tokenizers for amino acids, SMILES, nucleotides,
    and 3D latent generator coordinates.

    We don't want overlapping tokens between different tokenizers, so we need to add reserved
    tokens to each tokenizer's
    vocabulary to ensure that the token indices are compatible across all tokenizers.

    For each tokenizer, creates a complete vocabulary that includes:
         a) Special tokens (shared across all tokenizers)
         b) Reserved tokens (placeholders to maintain index compatibility)
         c) Domain-specific tokens for that particular tokenizer

    Returns
    -------
    None
        The function saves tokenizers to disk but does not return any values.
    """
    # Load vocabularies without reserved tokens
    vocabs = _load_vocabularies()

    # Add reserved tokens to create complementary vocabularies
    complete_vocabs = _add_reserved_tokens(vocabs)

    # Create and save individual tokenizers
    _make_amino_acid_tokenizer_fast(complete_vocabs[AMINO_ACID_TOKENIZER])
    _make_smiles_tokenizer_fast(complete_vocabs[SMILES_TOKENIZER])
    _make_nucleotide_tokenizer_fast(complete_vocabs[NUCLEOTIDE_TOKENIZER])
    _make_latent_generator_3d_coord_tokenizer_fast(complete_vocabs[LATENT_GENERATOR_TOKENIZER])


class UmeAminoAcidTokenizerFast(PreTrainedTokenizerFast):
    padding_side = "right"
    truncation_side = "right"
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self):
        super().__init__(
            tokenizer_file=str(TOKENIZERS_PATH / AMINO_ACID_TOKENIZER / "tokenizer.json"),
            bos_token=None,
            eos_token=EOS_TOKEN,
            unk_token=UNK_TOKEN,
            sep_token=SEP_TOKEN,
            pad_token=PAD_TOKEN,
            cls_token=CLS_TOKEN,
            mask_token=MASK_TOKEN,
        )


class UmeSmilesTokenizerFast(PreTrainedTokenizerFast):
    padding_side = "right"
    truncation_side = "right"
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self):
        super().__init__(
            tokenizer_file=str(TOKENIZERS_PATH / SMILES_TOKENIZER / "tokenizer.json"),
            bos_token=None,
            eos_token=EOS_TOKEN,
            unk_token=UNK_TOKEN,
            sep_token=SEP_TOKEN,
            pad_token=PAD_TOKEN,
            cls_token=CLS_TOKEN,
            mask_token=MASK_TOKEN,
        )


class UmeNucleotideTokenizerFast(PreTrainedTokenizerFast):
    padding_side = "right"
    truncation_side = "right"
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self):
        super().__init__(
            tokenizer_file=str(TOKENIZERS_PATH / NUCLEOTIDE_TOKENIZER / "tokenizer.json"),
            bos_token=None,
            eos_token=EOS_TOKEN,
            unk_token=UNK_TOKEN,
            sep_token=SEP_TOKEN,
            pad_token=PAD_TOKEN,
            cls_token=CLS_TOKEN,
            mask_token=MASK_TOKEN,
        )


class UmeLatentGenerator3DCoordTokenizerFast(PreTrainedTokenizerFast):
    padding_side = "right"
    truncation_side = "right"
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self):
        super().__init__(
            tokenizer_file=str(TOKENIZERS_PATH / LATENT_GENERATOR_TOKENIZER / "tokenizer.json"),
            bos_token=None,
            eos_token=EOS_TOKEN,
            unk_token=UNK_TOKEN,
            sep_token=SEP_TOKEN,
            pad_token=PAD_TOKEN,
            cls_token=CLS_TOKEN,
            mask_token=MASK_TOKEN,
        )


def _get_modality_tokenizer(modality: ModalityType | str) -> PreTrainedTokenizerFast:
    modality = Modality(modality) if isinstance(modality, str) else modality

    match modality:
        case Modality.AMINO_ACID:
            tokenizer = UmeAminoAcidTokenizerFast()
        case Modality.SMILES:
            tokenizer = UmeSmilesTokenizerFast()
        case Modality.NUCLEOTIDE:
            tokenizer = UmeNucleotideTokenizerFast()
        case Modality.COORDINATES_3D:
            tokenizer = UmeLatentGenerator3DCoordTokenizerFast()

    return tokenizer


class UmeTokenizerTransform(Module):
    """
    Ume tokenizer transform for single or dual modality inputs.

    For single modality:
        Tokenizes inputs using the specified modality tokenizer
        with vocabulary that's aware of reserved tokens.

    For dual modality:
        Combines inputs as: [CLS] [input1] [SEP] <task_token> [input2] [SEP]
        where task_token is either <interact> or <convert>.

    Examples
    --------
    >>> # Single modality tokenization
    >>> tokenizer = UmeTokenizerTransform(
    ...     modality="amino_acid",
    ...     max_length=12,
    ...     mode="interact",
    ...     return_modality=True,
    ... )
    >>> out = tokenizer("MYK")
    >>> out
    {'input_ids': tensor([[ 0, 41, 40, 36, 2, 1, 1, 1, 1, 1, 1, 1]]),
    'attention_mask': tensor([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]]),
    'modality': <Modality.AMINO_ACID: 'amino_acid'>}

    >>> # Dual modality tokenization
    >>> # Corresponds to: <cls> [input1] [sep] <task_token> [input2] [sep]
    >>> tokenizer = UmeTokenizerTransform(
    ...     modality=("amino_acid", "SMILES"),
    ...     max_length=16,
    ...     mode="interact",
    ...     return_modality=True,
    ... )
    >>> out = tokenizer(("MYK", "CCO"))
    >>> out
    {'input_ids': tensor([ 0, 41, 40, 36, 5, 7, 52, 52, 56, 5, 1, 1, 1, 1, 1, 1]),
    'attention_mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]),
    'modality1': <Modality.AMINO_ACID: 'amino_acid'>,
    'modality2': <Modality.SMILES: 'SMILES'>,
    'mode': 'interact'}

    Parameters
    ----------
    modality : ModalityType or str or tuple
        Single modality or tuple of two modalities for dual mode.
        Examples: "amino_acid", "smiles", "nucleotide",
        or ("amino_acid", "smiles"),...
    max_length : int or None
        Maximum sequence length. If None, no padding/truncation.
    return_modality : bool, optional
        Whether to return modality info in output. Default False.
    add_special_tokens : bool, optional
        Add special tokens for single modality. Default True.
    padding : str, optional
        Padding strategy. Default "max_length".
    mode : {'interact', 'convert'}, optional
        Task mode for dual modality. Can be "interact" or "convert" or None.
    """

    def __init__(
        self,
        modality: ModalityType | str | tuple[ModalityType | str, ModalityType | str],
        max_length: int | None,
        return_modality: bool = False,
        add_special_tokens: bool = True,
        padding: str = "max_length",
        mode: Literal["interact", "convert"] | None = None,
        seed: int | None = 0,
    ):
        super().__init__()

        # Handle max_length warning
        if max_length is None:
            warnings.warn(
                "UmeTokenizerTransform did not receive `max_length` parameter. Padding and truncation will not be applied.",
                UserWarning,
                stacklevel=2,
            )

        self.max_length = max_length
        self.return_modality = return_modality
        self.add_special_tokens = add_special_tokens
        self.padding = padding
        self.mode = mode
        self.seed = seed

        # Single modality
        if not isinstance(modality, tuple):
            self.is_dual_modality = False
            self.modality = Modality(modality) if isinstance(modality, str) else modality
            self.tokenizer = _get_modality_tokenizer(self.modality)

        # Dual modality
        else:
            if self.mode is None or self.mode not in {"interact", "convert"}:
                raise ValueError("For dual modality, `mode` must be specified as 'interact' or 'convert'")

            self.is_dual_modality = True
            self.modality1 = Modality(modality[0]) if isinstance(modality[0], str) else modality[0]
            self.modality2 = Modality(modality[1]) if isinstance(modality[1], str) else modality[1]

            self.tokenizer1 = _get_modality_tokenizer(self.modality1)
            self.tokenizer2 = _get_modality_tokenizer(self.modality2)

            self.task_token = f"<{self.mode}>"
            self.task_token_id = self.tokenizer1.convert_tokens_to_ids(self.task_token)

            if self.task_token not in self.tokenizer1.get_vocab() or self.task_token not in self.tokenizer2.get_vocab():
                raise ValueError(f"Task token '{self.task_token}' not found in tokenizer vocabularies. ")

            self.pad_id = self.tokenizer1.pad_token_id

    def _check_and_sample_3d_coordinates(self, item: list[str] | list[list[str]]) -> None:
        """Validate 3D coordinate input structure.

        3D Coordinates should be either:
        - A list of 4 tokenized canonical poses: ['ge be', 'fh ds', 'gh ge', 'ds be']
        - A list of such lists: [['ge be', 'fh ds', 'gh ge', 'ds be'], [...]]
        """

        def _validate_single_pose(pose: list[str]) -> None:
            if not (isinstance(pose, list) and len(pose) == 4 and all(isinstance(i, str) for i in pose)):
                raise ValueError(f"For 3D coordinates, input must be a list of 4 strings. Got: {pose} instead.")

        # Handle nested lists (batch input)
        if isinstance(item, list) and all(isinstance(i, list) for i in item):
            items = []

            for pose in item:
                _validate_single_pose(pose)
                sampled_pose = lobster.transforms.functional.sample_item(pose, seed=self.seed)
                items.append(sampled_pose)

            return items
        else:
            # Handle single list (non-batch input)
            _validate_single_pose(item)
            return lobster.transforms.functional.sample_item(item, seed=self.seed)

    def _encode(self, item: str | list[str]) -> dict[str, Tensor]:
        """Tokenize and encode single modality input."""

        if self.modality == Modality.COORDINATES_3D:
            item = self._check_and_sample_3d_coordinates(item)
            print(f"item after sampling: {item}")

        return self.tokenizer(
            item,
            max_length=self.max_length,
            padding=self.padding if self.max_length else False,
            truncation=True if self.max_length else False,
            add_special_tokens=self.add_special_tokens,
            return_tensors="pt",
        )

    def _encode_no_padding_no_special_tokens(
        self, item: str | list[str], tokenizer, modality: Modality
    ) -> dict[str, list[int]]:
        """Encode an item for dual modality without padding or special tokens."""
        if modality == Modality.COORDINATES_3D:
            item = self._check_and_sample_3d_coordinates(item)

        return tokenizer(item, padding="do_not_pad", truncation=False, add_special_tokens=False)

    def _combine_and_pad(self, input_ids1: list[int], input_ids2: list[int]) -> list[int]:
        """Combine token ID sequences for multiple modalities and apply padding/truncation."""

        # [CLS] [input1] [SEP] <task_token> [input2] [SEP]
        combined_ids = [
            self.tokenizer1.cls_token_id,
            *input_ids1,
            self.tokenizer1.sep_token_id,
            self.task_token_id,
            *input_ids2,
            self.tokenizer1.sep_token_id,
        ]

        if self.max_length:
            # Truncate
            if len(combined_ids) > self.max_length:
                combined_ids = combined_ids[: self.max_length]

            # Pad
            elif len(combined_ids) < self.max_length:
                combined_ids.extend([self.pad_id] * (self.max_length - len(combined_ids)))

        return combined_ids

    def forward(
        self,
        item: str | list[str] | tuple[str, str],
    ) -> dict[str, Tensor | Modality | str]:
        """
        Tokenize input(s) based on single or dual modality mode.

        Parameters
        ----------
        item : str, list[str], or tuple
            - For single modality: str, list[str]
              Examples: "MYK", ["MYK", "AVYK"]
            - For dual modality: tuple[str, str]
              Examples: ("MYK", "CCO")
              Batch processing is NOT supported for dual modality (because of easy mistakes with LG
              coordinates which are themselves a list of strings).
        Returns
        -------
        dict
            Tokenized output with keys:
                - "input_ids": Tensor of token IDs
                - "attention_mask": Tensor of attention masks
                - "modality": Modality type (if return_modality is True)
                - "modality1": Modality type for first input (if dual modality)
                - "modality2": Modality type for second input (if dual modality)
                - "mode": Task mode (if dual modality and return_modality is True)
        """
        # Single modality
        if not self.is_dual_modality:
            output = self._encode(item)

            if self.return_modality:
                output["modality"] = self.modality

            return output

        # Dual modality
        # Doesn't support batch processing for now
        if not isinstance(item, tuple) and len(item) != 2:
            raise NotImplementedError(
                f"Dual modality doesn't support batch processing and only accepts a tuple of two items. "
                "Example: ('MYK', 'CCO'). "
                f"Got: {item} instead."
            )

        # Encode inputs without padding, truncation, or special tokens
        encoded1 = self._encode_no_padding_no_special_tokens(item[0], self.tokenizer1, self.modality1)
        encoded2 = self._encode_no_padding_no_special_tokens(item[1], self.tokenizer2, self.modality2)

        combined_input_ids = self._combine_and_pad(encoded1["input_ids"], encoded2["input_ids"])

        input_ids = torch.tensor(combined_input_ids).unsqueeze(0)

        output = {
            "input_ids": input_ids,
            "attention_mask": (input_ids != self.pad_id).long(),
        }

        if self.return_modality:
            output = {
                **output,
                "modality1": self.modality1,
                "modality2": self.modality2,
                "mode": self.mode,
            }

        return output
