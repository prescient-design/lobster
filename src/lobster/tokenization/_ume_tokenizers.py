"""Ume tokenizers for amino acids, SMILES, nucleotides, and 3D latent generator coordinates.

Creates tokenizers with shared special tokens and reserved tokens to make sure there are
no overlapping tokens between different modalities.

Vocabulary structure:
- Special tokens: ["<cls>", "<eos>", "<unk>", "<pad>", ...]
- Conversion and interaction tokens: ["<convert>", "<interact>"]
- Extra special tokens to get % 64 == 0: ["<extra2>", "<extra3>", ...]
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
from tokenizers import Regex
from tokenizers.models import BPE, WordLevel
from tokenizers.normalizers import Lowercase
from tokenizers.pre_tokenizers import Sequence as PreTokenizerSequence
from tokenizers.pre_tokenizers import Split, WhitespaceSplit
from tokenizers.processors import TemplateProcessing
from torch import Tensor
from torch.nn import Module
from transformers import PreTrainedTokenizerFast

import lobster.transforms.functional
from lobster.constants import Modality, ModalityType

from ._latent_generator_3d_coord_tokenizer import VOCAB_PATH as LATENT_GENERATOR_VOCAB_PATH
from ._load_vocab_file import load_vocab_file
from ._make_pretrained_tokenizer_fast import make_pretrained_tokenizer_fast
from ._smiles_tokenizer import SMILES_REGEX_PATTERN
from ._smiles_tokenizer import VOCAB_PATH as SMILES_VOCAB_PATH

TOKENIZERS_PATH = importlib.resources.files("lobster") / "assets" / "ume_tokenizers"

# Tokenizer names for saving
AMINO_ACID_TOKENIZER = "amino_acid_tokenizer"
SMILES_TOKENIZER = "smiles_tokenizer"
NUCLEOTIDE_TOKENIZER = "nucleotide_tokenizer"
COORDINATES_3D_TOKENIZER = "coordinates_3d_tokenizer"
SPECIAL_TOKENS_NAME = "special_tokens"

AMINO_ACID_VOCAB_PATH = TOKENIZERS_PATH / AMINO_ACID_TOKENIZER / "vocab.txt"
NUCLEOTIDE_VOCAB_PATH = TOKENIZERS_PATH / NUCLEOTIDE_TOKENIZER / "vocab.txt"

# Special tokens
CLS_TOKEN = "<cls>"
CLS_TOKEN_AMINO_ACID = "<cls_amino_acid>"
CLS_TOKEN_SMILES = "<cls_smiles>"
CLS_TOKEN_NUCLEOTIDE = "<cls_nucleotide>"
CLS_TOKEN_3D_COORDINATES = "<cls_3d_coordinates>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
SEP_TOKEN = "<sep>"
MASK_TOKEN = "<mask>"
CONVERT_TOKEN = "<cls_convert>"
INTERACT_TOKEN = "<cls_interact>"
NUM_EXTRA_SPECIAL_TOKENS = 11


def _get_special_tokens() -> list[str]:
    # Add extra special tokens to the next multiple of 64
    # NOTE: To figure out how much we need to add, run `_make_ume_tokenizers()` first, check the vocab size,
    # update `NUM_EXTRA_SPECIAL_TOKENS` to the number of extra special tokens needed
    # and run `_make_ume_tokenizers()` again.
    extra_special_tokens = [f"<extra_special_token_{i}>" for i in range(NUM_EXTRA_SPECIAL_TOKENS)]

    return [
        CLS_TOKEN,
        CLS_TOKEN_AMINO_ACID,
        CLS_TOKEN_SMILES,
        CLS_TOKEN_NUCLEOTIDE,
        CLS_TOKEN_3D_COORDINATES,
        EOS_TOKEN,
        UNK_TOKEN,
        PAD_TOKEN,
        SEP_TOKEN,
        MASK_TOKEN,
        CONVERT_TOKEN,
        INTERACT_TOKEN,
        *extra_special_tokens,
    ]


def _load_file(filepath: str | Path, remove_special_tokens: bool = False) -> list[str]:
    vocab = load_vocab_file(filepath)

    if remove_special_tokens:
        special_tokens = _get_special_tokens()
        return [token for token in vocab if token not in set(special_tokens)]

    return vocab


def _load_vocabularies() -> dict[str, list[str]]:
    return {
        SPECIAL_TOKENS_NAME: _get_special_tokens(),
        AMINO_ACID_TOKENIZER: _load_file(AMINO_ACID_VOCAB_PATH, remove_special_tokens=True),
        SMILES_TOKENIZER: _load_file(SMILES_VOCAB_PATH, remove_special_tokens=True),
        NUCLEOTIDE_TOKENIZER: _load_file(NUCLEOTIDE_VOCAB_PATH, remove_special_tokens=True),
        COORDINATES_3D_TOKENIZER: _load_file(LATENT_GENERATOR_VOCAB_PATH, remove_special_tokens=True),
    }


def _add_reserved_tokens(vocabs: dict[str, list[str]]) -> dict[str, list[str]]:
    """
    Add reserved tokens <reservedi> to maintain index compatibility
    across tokenizers.

    This function constructs the full vocabulary for each tokenizer by combining:
    - Special tokens (shared across all tokenizers)
    - Reserved/dummy tokens (to maintain index compatibility)
    - Domain-specific tokens for each tokenizer type

    Ordering of tokenizers is important for reserved token construction!

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
    # Amino acid tokenizer: special_tokens + amino acid vocab
    vocab_size_amino_acid = len(vocabs[AMINO_ACID_TOKENIZER])
    vocab_amino_acid = vocabs[SPECIAL_TOKENS_NAME] + vocabs[AMINO_ACID_TOKENIZER]

    # SMILES tokenizer: special_tokens + [reserved for amino acid tokens] + smiles vocab
    vocab_size_smiles = len(vocabs[SMILES_TOKENIZER])
    vocab_smiles = (
        vocabs[SPECIAL_TOKENS_NAME]
        + [f"<reserved_for_amino_acids_{i}>" for i in range(vocab_size_amino_acid)]
        + vocabs[SMILES_TOKENIZER]
    )

    # Nucleotide tokenizer: special_tokens + [reserved for amino acid] + [reserved for SMILES] + nucleotide vocab
    vocab_size_nucleotide = len(vocabs[NUCLEOTIDE_TOKENIZER])
    vocab_nucleotide = (
        vocabs[SPECIAL_TOKENS_NAME]
        + [f"<reserved_for_amino_acids_{i}>" for i in range(vocab_size_amino_acid)]
        + [f"<reserved_for_smiles_{i}>" for i in range(vocab_size_smiles)]
        + vocabs[NUCLEOTIDE_TOKENIZER]
    )

    # Latent generator tokenizer: special_tokens + [reserved for amino acid] + [reserved for SMILES] +
    # [reserved for nucleotide] + latent generator 3D vocab
    vocab_coordinates_3d = (
        vocabs[SPECIAL_TOKENS_NAME]
        + [f"<reserved_for_amino_acids_{i}>" for i in range(vocab_size_amino_acid)]
        + [f"<reserved_for_smiles_{i}>" for i in range(vocab_size_smiles)]
        + [f"<reserved_for_nucleotide_{i}>" for i in range(vocab_size_nucleotide)]
        + vocabs[COORDINATES_3D_TOKENIZER]
    )

    return {
        SPECIAL_TOKENS_NAME: vocabs[SPECIAL_TOKENS_NAME],
        AMINO_ACID_TOKENIZER: vocab_amino_acid,
        SMILES_TOKENIZER: vocab_smiles,
        NUCLEOTIDE_TOKENIZER: vocab_nucleotide,
        COORDINATES_3D_TOKENIZER: vocab_coordinates_3d,
    }


def _create_post_processor(cls_token: str) -> TemplateProcessing:
    """
    Create a template processor for tokenization.

    The processor formats token sequences by adding special tokens like CLS and EOS
    for both single sequences and pairs of sequences.

    Returns
    -------
    TemplateProcessing
        Configured template processor for token sequence formatting
    """
    special_tokens = _get_special_tokens()
    cls_token_index = special_tokens.index(cls_token)
    eos_token_index = special_tokens.index(EOS_TOKEN)

    return TemplateProcessing(
        single=f"{cls_token} $A {EOS_TOKEN}",
        pair=f"{cls_token} $A {EOS_TOKEN} $B:1 {EOS_TOKEN}:1",
        special_tokens=[
            (cls_token, cls_token_index),
            (EOS_TOKEN, eos_token_index),
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
    post_processor = _create_post_processor(cls_token=CLS_TOKEN_AMINO_ACID)

    return make_pretrained_tokenizer_fast(
        tokenizer_model=tokenizer_model,
        post_processor=post_processor,
        save_dirpath=str(TOKENIZERS_PATH / AMINO_ACID_TOKENIZER),
        cls_token=CLS_TOKEN_AMINO_ACID,
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
    pre_tokenizer = Split(pattern=Regex(SMILES_REGEX_PATTERN), behavior="isolated")
    post_processor = _create_post_processor(cls_token=CLS_TOKEN_SMILES)

    return make_pretrained_tokenizer_fast(
        tokenizer_model=tokenizer_model,
        pre_tokenizer=pre_tokenizer,
        post_processor=post_processor,
        save_dirpath=str(TOKENIZERS_PATH / SMILES_TOKENIZER),
        cls_token=CLS_TOKEN_SMILES,
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
    post_processor = _create_post_processor(cls_token=CLS_TOKEN_NUCLEOTIDE)

    return make_pretrained_tokenizer_fast(
        tokenizer_model=tokenizer_model,
        normalizer=Lowercase(),
        post_processor=post_processor,
        save_dirpath=str(TOKENIZERS_PATH / NUCLEOTIDE_TOKENIZER),
        cls_token=CLS_TOKEN_NUCLEOTIDE,
        eos_token=EOS_TOKEN,
        unk_token=UNK_TOKEN,
        sep_token=SEP_TOKEN,
        mask_token=MASK_TOKEN,
        pad_token=PAD_TOKEN,
        bos_token=None,
    )


def _make_3d_coordinates_tokenizer_fast(vocab: list[str]) -> PreTrainedTokenizerFast:
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
    post_processor = _create_post_processor(cls_token=CLS_TOKEN_3D_COORDINATES)

    return make_pretrained_tokenizer_fast(
        tokenizer_model=tokenizer_model,
        pre_tokenizer=pre_tokenizer,
        post_processor=post_processor,
        save_dirpath=str(TOKENIZERS_PATH / COORDINATES_3D_TOKENIZER),
        cls_token=CLS_TOKEN_3D_COORDINATES,
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
    _make_3d_coordinates_tokenizer_fast(complete_vocabs[COORDINATES_3D_TOKENIZER])


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
            cls_token=CLS_TOKEN_AMINO_ACID,
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
            cls_token=CLS_TOKEN_SMILES,
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
            cls_token=CLS_TOKEN_NUCLEOTIDE,
            mask_token=MASK_TOKEN,
        )


class UmeLatentGenerator3DCoordTokenizerFast(PreTrainedTokenizerFast):
    padding_side = "right"
    truncation_side = "right"
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self):
        super().__init__(
            tokenizer_file=str(TOKENIZERS_PATH / COORDINATES_3D_TOKENIZER / "tokenizer.json"),
            bos_token=None,
            eos_token=EOS_TOKEN,
            unk_token=UNK_TOKEN,
            sep_token=SEP_TOKEN,
            pad_token=PAD_TOKEN,
            cls_token=CLS_TOKEN_3D_COORDINATES,
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

            self.task_token = f"<cls_{self.mode}>"
            self.task_token_id = self.tokenizer1.convert_tokens_to_ids(self.task_token)

            if self.task_token not in self.tokenizer1.get_vocab() or self.task_token not in self.tokenizer2.get_vocab():
                raise ValueError(f"Task token '{self.task_token}' not found in tokenizer vocabularies. ")

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

        # [CLS_interact or  CLS_convert] [CLS_modality1] [input1] [SEP] [CLS_modality2] [input2] [SEP]
        combined_ids = [
            self.task_token_id,
            self.tokenizer1.cls_token_id,
            *input_ids1,
            self.tokenizer1.sep_token_id,
            self.tokenizer2.cls_token_id,
            *input_ids2,
            self.tokenizer1.sep_token_id,
        ]

        if self.max_length:
            # Truncate
            if len(combined_ids) > self.max_length:
                combined_ids = combined_ids[: self.max_length]

            # Pad
            elif len(combined_ids) < self.max_length:
                combined_ids.extend([self.tokenizer1.pad_token_id] * (self.max_length - len(combined_ids)))

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
            "attention_mask": (input_ids != self.tokenizer1.pad_token_id).long(),
        }

        if self.return_modality:
            output = {
                **output,
                "modality1": self.modality1,
                "modality2": self.modality2,
                "mode": self.mode,
            }

        return output
