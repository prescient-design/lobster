"""UME tokenizers for amino acids, SMILES, and nucleotides.

Creates tokenizers with shared special tokens and reserved tokens to make sure there are
no overlapping tokens between different modalities.

Vocabulary structure:
- Special tokens: ["<cls>", "<eos>", "<unk>", "<pad>", ...]
- Conversion and interaction tokens: ["<convert>", "<interact>"]
- Extra special tokens to get % 64 == 0: ["<extra2>", "<extra3>", ...]
- Amino acid tokenizer: [special_tokens] + ["A", "C", "D", ...]
- SMILES tokenizer: [special_tokens] + [reserved_for_amino_acids] + ["C", "O", "N", ...]
- Nucleotide tokenizer: [special_tokens] + [reserved_for_amino_acids] + [reserved_for_SMILES] + ["A", "C", "G", ...]

To create the tokenizers, run

```python
    from lobster.tokenization._ume_tokenizers import (
        _make_ume_tokenizers,
        UMEAminoAcidTokenizerFast,
        UMESmilesTokenizerFast,
        UMENucleotideTokenizerFast,
    )
    # Create and save tokenizers
    _make_ume_tokenizers()

    tokenizers = [
            UMEAminoAcidTokenizerFast(),
            UMESmilesTokenizerFast(),
            UMENucleotideTokenizerFast(),
    ]

    # Compute the total vocabulary size
    vocab = {
        token_id: token for tokenizer in tokenizers
        for token, token_id in tokenizer.get_vocab().items()
        if "reserved" not in token
    }

    print(f"Total vocabulary size = {len(vocab)}")
    print(f"Vocab size % 64 = {len(vocab) % 64}")
```
"""

import importlib.resources
import warnings
from pathlib import Path
from typing import Literal

from tokenizers import Regex
from tokenizers.models import BPE, WordLevel
from tokenizers.normalizers import Lowercase
from tokenizers.pre_tokenizers import Split
from tokenizers.processors import TemplateProcessing
from torch import Tensor
from torch.nn import Module
from transformers import PreTrainedTokenizerFast

from lobster.constants import Modality, ModalityType

from ._load_vocab_file import load_vocab_file
from ._make_pretrained_tokenizer_fast import make_pretrained_tokenizer_fast
from ._smiles_tokenizer import SMILES_REGEX_PATTERN
from ._smiles_tokenizer import VOCAB_PATH as SMILES_VOCAB_PATH

TOKENIZERS_PATH = importlib.resources.files("lobster") / "assets" / "ume_tokenizers"

# Tokenizer names for saving
AMINO_ACID_TOKENIZER = "amino_acid_tokenizer"
SMILES_TOKENIZER = "smiles_tokenizer"
NUCLEOTIDE_TOKENIZER = "nucleotide_tokenizer"
SPECIAL_TOKENS_NAME = "special_tokens"

AMINO_ACID_VOCAB_PATH = TOKENIZERS_PATH / AMINO_ACID_TOKENIZER / "vocab.txt"
NUCLEOTIDE_VOCAB_PATH = TOKENIZERS_PATH / NUCLEOTIDE_TOKENIZER / "vocab.txt"

# Special tokens
CLS_TOKEN = "<cls>"
CLS_TOKEN_AMINO_ACID = "<cls_amino_acid>"
CLS_TOKEN_SMILES = "<cls_smiles>"
CLS_TOKEN_NUCLEOTIDE = "<cls_nucleotide>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
SEP_TOKEN = "<sep>"
MASK_TOKEN = "<mask>"
CONVERT_TOKEN = "<cls_convert>"
INTERACT_TOKEN = "<cls_interact>"
NUM_EXTRA_SPECIAL_TOKENS = 14


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
    }


def _add_reserved_tokens(vocabs: dict[str, list[str]]) -> dict[str, list[str]]:
    """
    Add reserved tokens <reserved_token_i> to maintain index compatibility
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
    vocab_nucleotide = (
        vocabs[SPECIAL_TOKENS_NAME]
        + [f"<reserved_for_amino_acids_{i}>" for i in range(vocab_size_amino_acid)]
        + [f"<reserved_for_smiles_{i}>" for i in range(vocab_size_smiles)]
        + vocabs[NUCLEOTIDE_TOKENIZER]
    )

    return {
        SPECIAL_TOKENS_NAME: vocabs[SPECIAL_TOKENS_NAME],
        AMINO_ACID_TOKENIZER: vocab_amino_acid,
        SMILES_TOKENIZER: vocab_smiles,
        NUCLEOTIDE_TOKENIZER: vocab_nucleotide,
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


def _make_ume_tokenizers() -> None:
    """
    Create and save tokenizers for amino acids, SMILES, and nucleotides.

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


class UMEAminoAcidTokenizerFast(PreTrainedTokenizerFast):
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


class UMESmilesTokenizerFast(PreTrainedTokenizerFast):
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


class UMENucleotideTokenizerFast(PreTrainedTokenizerFast):
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


class UMETokenizerTransform(Module):
    """
    UME tokenizer transform for single modality inputs.

    Tokenizes inputs using the specified modality tokenizer
    with vocabulary that's aware of reserved tokens.

    Examples
    --------
    >>> tokenizer = UMETokenizerTransform(
    ...     modality="amino_acid",
    ...     max_length=12,
    ...     return_modality=True,
    ... )
    >>> out = tokenizer("MYK")
    >>> out
    {'input_ids': tensor([[ 0, 41, 40, 36, 2, 1, 1, 1, 1, 1, 1, 1]]),
    'attention_mask': tensor([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]]),
    'modality': <Modality.AMINO_ACID: 'amino_acid'>}

    Parameters
    ----------
    modality : ModalityType or Literal["amino_acid", "smiles", "nucleotide"]
        Modality to tokenize.
    max_length : int or None
        Maximum sequence length. If None, no padding/truncation.
    return_modality : bool, optional
        Whether to return modality info in output. Default False.
    add_special_tokens : bool, optional
        Add special tokens. Default True.
    padding : str, optional
        Padding strategy. Default "max_length".
    """

    def __init__(
        self,
        modality: ModalityType | Literal["amino_acid", "smiles", "nucleotide"],
        max_length: int | None,
        return_modality: bool = False,
        add_special_tokens: bool = True,
        padding: str = "max_length",
        seed: int | None = 0,
    ):
        super().__init__()

        if max_length is None:
            warnings.warn(
                "UMETokenizerTransform did not receive `max_length` parameter. Padding and truncation will not be applied.",
                UserWarning,
                stacklevel=2,
            )

        self.max_length = max_length
        self.return_modality = return_modality
        self.add_special_tokens = add_special_tokens
        self.padding = padding
        self.seed = seed

        self.modality = Modality(modality) if isinstance(modality, str) else modality

        modality = Modality(modality) if isinstance(modality, str) else modality

        match modality:
            case Modality.AMINO_ACID:
                self.tokenizer = UMEAminoAcidTokenizerFast()
            case Modality.SMILES:
                self.tokenizer = UMESmilesTokenizerFast()
            case Modality.NUCLEOTIDE:
                self.tokenizer = UMENucleotideTokenizerFast()

    def _encode(self, item: str | list[str]) -> dict[str, Tensor]:
        """Tokenize and encode input."""
        return self.tokenizer(
            item,
            max_length=self.max_length,
            padding=self.padding if self.max_length else False,
            truncation=True if self.max_length else False,
            add_special_tokens=self.add_special_tokens,
            return_tensors="pt",
        )

    def forward(
        self,
        item: str | list[str],
    ) -> dict[str, Tensor | Modality]:
        """
        Tokenize input.

        Parameters
        ----------
        item : str or list[str]
            Input to tokenize. Examples: "MYK", ["MYK", "AVYK"]

        Returns
        -------
        dict
            Tokenized output with keys:
                - "input_ids": Tensor of token IDs
                - "attention_mask": Tensor of attention masks
                - "modality": Modality type (if return_modality is True)
        """
        output = self._encode(item)

        if self.return_modality:
            output["modality"] = self.modality

        return output
