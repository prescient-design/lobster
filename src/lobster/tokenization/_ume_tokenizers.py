"""Ume tokenizers for amino acids, SMILES, nucleotides, and 3D latent generator coordinates.

Creates tokenizers with shared special tokens and reserved tokens to make sure there are
no overlapping tokens between different modalities.

Example structure:
- Special tokens: ["<cls>", "<eos>", "<unk>", "<pad>"]
- Amino acid tokenizer: [special_tokens] + ["A", "C", "D", ...]
- SMILES tokenizer: [special_tokens] + [reserved_for_amino_acids] + ["C", "O", "N", ...]
- Nucleotide tokenizer: [special_tokens] + [reserved_for_amino_acids] + [reserved_for_SMILES] + ["A", "C", "G", ...]
- Latent generator tokenizer: [special_tokens] + [reserved_for_amino_acids] + [reserved_for_SMILES] +
  [reserved_for_nucleotides] + ["gh", "fh", "ds", ...]

To create the tokenizers, run

```python
    # Create and save tokenizers
    _make_ume_tokenizers()

    # Compute the total vocabulary size
    tokenizers = [
        UmeAminoAcidTokenizerFast(),
        UmeSmilesTokenizerFast(),
        UmeNucleotideTokenizerFast(),
        UmeLatentGenerator3DCoordTokenizerFast(),
    ]
    vocab = set(token for tokenizer in tokenizers for token in tokenizer.get_vocab().keys())

    print(f"Total vocabulary size = {len(vocab)}")  # 1472
    print(f"Vocab size % 64 = {len(vocab) % 64}")  # 0
```
"""

import importlib.resources
from pathlib import Path
from typing import Dict, List, Union

from tokenizers.models import BPE, WordLevel
from tokenizers.pre_tokenizers import Sequence, WhitespaceSplit
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

from ._make_pretrained_tokenizer_fast import make_pretrained_tokenizer_fast

TOKENIZERS_PATH = importlib.resources.files("lobster") / "assets" / "ume_tokenizers"

# Tokenizer names for saving
AMINO_ACID_TOKENIZER = "amino_acid_tokenizer"
SMILES_TOKENIZER = "smiles_tokenizer"
NUCLEOTIDE_TOKENIZER = "nucleotide_tokenizer"
LATENT_GENERATOR_TOKENIZER = "latent_generator_tokenizer"

# Special tokens
CLS_TOKEN = "<cls>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
SEP_TOKEN = "<sep>"
MASK_TOKEN = "<mask>"


# Order of tokenizers, important for vocabulary construction
# DO NOT change the order of tokenizers without making corresponding
# changes to the `_add_reserved_tokens` func
TOKENIZER_ORDER = [
    "special_tokens",
    AMINO_ACID_TOKENIZER,
    SMILES_TOKENIZER,
    NUCLEOTIDE_TOKENIZER,
    LATENT_GENERATOR_TOKENIZER,
]


def _load_file(filepath: Union[str, Path]) -> List[str]:
    with open(filepath, "r") as f:
        return f.read().splitlines()


def _load_vocabularies() -> Dict[str, List[str]]:
    return {
        "special_tokens": _load_file(TOKENIZERS_PATH / "special_tokens.txt"),
        "amino_acid_tokenizer": _load_file(TOKENIZERS_PATH / AMINO_ACID_TOKENIZER / "vocab.txt"),
        "smiles_tokenizer": _load_file(TOKENIZERS_PATH / SMILES_TOKENIZER / "vocab.txt"),
        "nucleotide_tokenizer": _load_file(TOKENIZERS_PATH / NUCLEOTIDE_TOKENIZER / "vocab.txt"),
        "latent_generator_tokenizer": _load_file(TOKENIZERS_PATH / LATENT_GENERATOR_TOKENIZER / "vocab.txt"),
    }


def _add_reserved_tokens(vocabs: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Add reserved tokens <reserved_special_token_i> to maintain index compatibility
    across tokenizers.

    This function constructs the full vocabulary for each tokenizer by combining:
    - Special tokens (shared across all tokenizers)
    - Reserved/dummy tokens (to maintain index compatibility)
    - Domain-specific tokens for each tokenizer type


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
        if token.startswith("<reserved_special_token_"):
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
        + [f"<reserved_special_token_{i}>" for i in range(next_reserved_index, next_reserved_index + amino_acid_len)]
        + vocabs["smiles_tokenizer"]
    )

    # Nucleotide tokenizer: special_tokens + [reserved for amino acid] + [reserved for SMILES] + nucleotide vocav
    smiles_len = len(vocabs["smiles_tokenizer"])
    vocab_nucleotide = (
        vocabs["special_tokens"]
        + [f"<reserved_special_token_{i}>" for i in range(next_reserved_index, next_reserved_index + amino_acid_len)]
        + [
            f"<reserved_special_token_{i}>"
            for i in range(next_reserved_index + amino_acid_len, next_reserved_index + amino_acid_len + smiles_len)
        ]
        + vocabs["nucleotide_tokenizer"]
    )

    # Latent generator tokenizer: special_tokens + [reserved for amino acid] + [reserved for SMILES] +
    # [reserved for nucleotide] + latent generator 3D vocab
    nucleotide_len = len(vocabs["nucleotide_tokenizer"])
    vocab_latent = (
        vocabs["special_tokens"]
        + [f"<reserved_special_token_{i}>" for i in range(next_reserved_index, next_reserved_index + amino_acid_len)]
        + [
            f"<reserved_special_token_{i}>"
            for i in range(next_reserved_index + amino_acid_len, next_reserved_index + amino_acid_len + smiles_len)
        ]
        + [
            f"<reserved_special_token_{i}>"
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


def _make_amino_acid_tokenizer_fast(vocab: List[str]) -> PreTrainedTokenizerFast:
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


def _make_smiles_tokenizer_fast(vocab: List[str]) -> PreTrainedTokenizerFast:
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
    tokenizer_model = BPE(
        {token: i for i, token in enumerate(vocab)}, merges=[], unk_token=UNK_TOKEN, ignore_merges=True
    )
    pre_tokenizer = Sequence([WhitespaceSplit()])
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


def _make_nucleotide_tokenizer_fast(vocab: List[str]) -> PreTrainedTokenizerFast:
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


def _make_latent_generator_3d_coord_tokenizer_fast(vocab: List[str]) -> PreTrainedTokenizerFast:
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
    pre_tokenizer = Sequence([WhitespaceSplit()])
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
