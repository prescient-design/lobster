"""UME tokenizer for HuggingFace Hub integration.

Self-contained tokenization module for Universal Molecular Encoder (UME)
that supports amino acids, SMILES, and nucleotide sequences.
"""

import os

import re
from huggingface_hub import hf_hub_download
from transformers import PreTrainedTokenizer


# HuggingFace repository configuration
HF_UME_REPO_ID = "karina-zadorozhny/ume"

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

# SMILES tokenization pattern
SMILES_REGEX_PATTERN = (
    r"""(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"""
)


def _load_vocab_from_hub(vocab_filename: str, repo_id: str = HF_UME_REPO_ID) -> list[str]:
    """
    Load vocabulary from HuggingFace Hub.

    Parameters
    ----------
    vocab_filename : str
        Name of the vocabulary file in the vocabs/ folder
    repo_id : str
        HuggingFace repository ID

    Returns
    -------
    List[str]
        List of vocabulary tokens
    """
    try:
        vocab_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"vocabs/{vocab_filename}",
            cache_dir=None,  # Use default cache
        )

        with open(vocab_path, encoding="utf-8") as f:
            vocab = [line.strip() for line in f.readlines() if line.strip()]

        return vocab

    except Exception as e:
        raise Exception(f"Failed to download vocabulary {vocab_filename} from {repo_id}.") from e


def _get_special_tokens() -> list[str]:
    """Get all special tokens including extra tokens to make vocab size divisible by 64."""
    # Add extra special tokens to reach next multiple of 64
    extra_special_tokens = [f"<extra_special_token_{i}>" for i in range(14)]

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


def _create_vocabularies() -> dict[str, list[str]]:
    """Create vocabularies for each modality with reserved tokens for compatibility."""
    special_tokens = _get_special_tokens()

    # Load actual vocabularies from HuggingFace Hub
    amino_acid_vocab = _load_vocab_from_hub("amino_acid_vocab.txt")
    smiles_vocab = _load_vocab_from_hub("smiles_vocab.txt")
    nucleotide_vocab = _load_vocab_from_hub("nucleotide_vocab.txt")

    # Remove special tokens from downloaded vocabs if they exist
    # (the downloaded vocabs might include special tokens that we want to control)
    amino_acid_vocab = [token for token in amino_acid_vocab if token not in special_tokens]
    smiles_vocab = [token for token in smiles_vocab if token not in special_tokens]
    nucleotide_vocab = [token for token in nucleotide_vocab if token not in special_tokens]

    # Amino acid tokenizer: special_tokens + amino acid vocab
    vocab_amino_acid = special_tokens + amino_acid_vocab

    # SMILES tokenizer: special_tokens + [reserved for amino acid tokens] + smiles vocab
    vocab_smiles = (
        special_tokens + [f"<reserved_for_amino_acids_{i}>" for i in range(len(amino_acid_vocab))] + smiles_vocab
    )

    # Nucleotide tokenizer: special_tokens + [reserved for amino acid] + [reserved for SMILES] + nucleotide vocab
    vocab_nucleotide = (
        special_tokens
        + [f"<reserved_for_amino_acids_{i}>" for i in range(len(amino_acid_vocab))]
        + [f"<reserved_for_smiles_{i}>" for i in range(len(smiles_vocab))]
        + nucleotide_vocab
    )

    return {
        "amino_acid": vocab_amino_acid,
        "smiles": vocab_smiles,
        "nucleotide": vocab_nucleotide,
    }


def _tokenize_smiles(text: str) -> list[str]:
    """Tokenize SMILES string using regex pattern matching."""
    pattern = SMILES_REGEX_PATTERN
    tokens = re.findall(pattern, text)
    return tokens


def _filter_special_tokens_from_kwargs(kwargs: dict) -> dict:
    """Filter out special token parameters from kwargs to avoid conflicts with explicit parameters."""
    special_token_keys = {"bos_token", "eos_token", "unk_token", "sep_token", "pad_token", "cls_token", "mask_token"}
    return {k: v for k, v in kwargs.items() if k not in special_token_keys}


class UMEAminoAcidTokenizer(PreTrainedTokenizer):
    """Slow tokenizer for amino acid sequences that matches original behavior."""

    def __init__(self, **kwargs):
        self.vocab_dict = {token: i for i, token in enumerate(_create_vocabularies()["amino_acid"])}
        self.ids_to_tokens = {i: token for token, i in self.vocab_dict.items()}

        super().__init__(
            bos_token=None,
            eos_token=EOS_TOKEN,
            unk_token=UNK_TOKEN,
            sep_token=SEP_TOKEN,
            pad_token=PAD_TOKEN,
            cls_token=CLS_TOKEN_AMINO_ACID,
            mask_token=MASK_TOKEN,
            **_filter_special_tokens_from_kwargs(kwargs),
        )

    @property
    def vocab_size(self) -> int:
        return len(self.vocab_dict)

    def get_vocab(self) -> dict[str, int]:
        return self.vocab_dict.copy()

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize amino acid sequence into individual characters."""
        return list(text.strip())

    def _convert_token_to_id(self, token: str) -> int:
        """Convert token to id using vocabulary."""
        return self.vocab_dict.get(token, self.vocab_dict.get(self.unk_token, 0))

    def _convert_id_to_token(self, index: int) -> str:
        """Convert id to token using vocabulary."""
        return self.ids_to_tokens.get(index, self.unk_token)

    def build_inputs_with_special_tokens(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = None
    ) -> list[int]:
        """Add special tokens around the input sequences."""
        cls_token_id = self.vocab_dict.get(self.cls_token, 0)
        eos_token_id = self.vocab_dict.get(self.eos_token, 2)

        if token_ids_1 is None:
            return [cls_token_id] + token_ids_0 + [eos_token_id]
        else:
            return [cls_token_id] + token_ids_0 + [eos_token_id] + token_ids_1 + [eos_token_id]

    def save_vocabulary(self, save_directory: str, filename_prefix: str | None = None) -> tuple[str]:
        """Save the vocabulary to a file."""
        prefix = filename_prefix + "-" if filename_prefix else ""
        vocab_file = os.path.join(
            save_directory,
            f"{prefix}vocab_amino_acid.txt",
        )

        # Sort tokens by their IDs to maintain order
        sorted_tokens = [self.ids_to_tokens[i] for i in sorted(self.ids_to_tokens.keys())]

        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write("\n".join(sorted_tokens))

        return (vocab_file,)


class UMESmilesTokenizer(PreTrainedTokenizer):
    """Slow tokenizer for SMILES chemical notations that matches original behavior."""

    def __init__(self, **kwargs):
        self.vocab_dict = {token: i for i, token in enumerate(_create_vocabularies()["smiles"])}
        self.ids_to_tokens = {i: token for token, i in self.vocab_dict.items()}

        super().__init__(
            bos_token=None,
            eos_token=EOS_TOKEN,
            unk_token=UNK_TOKEN,
            sep_token=SEP_TOKEN,
            pad_token=PAD_TOKEN,
            cls_token=CLS_TOKEN_SMILES,
            mask_token=MASK_TOKEN,
            **_filter_special_tokens_from_kwargs(kwargs),
        )

    @property
    def vocab_size(self) -> int:
        return len(self.vocab_dict)

    def get_vocab(self) -> dict[str, int]:
        return self.vocab_dict.copy()

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize SMILES string using regex pattern matching."""
        return _tokenize_smiles(text.strip())

    def _convert_token_to_id(self, token: str) -> int:
        """Convert token to id using vocabulary."""
        return self.vocab_dict.get(token, self.vocab_dict.get(self.unk_token, 0))

    def _convert_id_to_token(self, index: int) -> str:
        """Convert id to token using vocabulary."""
        return self.ids_to_tokens.get(index, self.unk_token)

    def build_inputs_with_special_tokens(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = None
    ) -> list[int]:
        """Add special tokens around the input sequences."""
        cls_token_id = self.vocab_dict.get(self.cls_token, 0)
        eos_token_id = self.vocab_dict.get(self.eos_token, 2)

        if token_ids_1 is None:
            return [cls_token_id] + token_ids_0 + [eos_token_id]
        else:
            return [cls_token_id] + token_ids_0 + [eos_token_id] + token_ids_1 + [eos_token_id]

    def save_vocabulary(self, save_directory: str, filename_prefix: str | None = None) -> tuple[str]:
        """Save the vocabulary to a file."""
        prefix = filename_prefix + "-" if filename_prefix else ""
        vocab_file = os.path.join(
            save_directory,
            f"{prefix}vocab_smiles.txt",
        )

        # Sort tokens by their IDs to maintain order
        sorted_tokens = [self.ids_to_tokens[i] for i in sorted(self.ids_to_tokens.keys())]

        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write("\n".join(sorted_tokens))

        return (vocab_file,)


class UMENucleotideTokenizer(PreTrainedTokenizer):
    """Slow tokenizer for nucleotide sequences that matches original behavior."""

    def __init__(self, **kwargs):
        self.vocab_dict = {token: i for i, token in enumerate(_create_vocabularies()["nucleotide"])}
        self.ids_to_tokens = {i: token for token, i in self.vocab_dict.items()}

        super().__init__(
            bos_token=None,
            eos_token=EOS_TOKEN,
            unk_token=UNK_TOKEN,
            sep_token=SEP_TOKEN,
            pad_token=PAD_TOKEN,
            cls_token=CLS_TOKEN_NUCLEOTIDE,
            mask_token=MASK_TOKEN,
            **_filter_special_tokens_from_kwargs(kwargs),
        )

    @property
    def vocab_size(self) -> int:
        return len(self.vocab_dict)

    def get_vocab(self) -> dict[str, int]:
        return self.vocab_dict.copy()

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize nucleotide sequence into individual characters with lowercase normalization."""
        return list(text.strip().lower())

    def _convert_token_to_id(self, token: str) -> int:
        """Convert token to id using vocabulary."""
        return self.vocab_dict.get(token, self.vocab_dict.get(self.unk_token, 0))

    def _convert_id_to_token(self, index: int) -> str:
        """Convert id to token using vocabulary."""
        return self.ids_to_tokens.get(index, self.unk_token)

    def build_inputs_with_special_tokens(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = None
    ) -> list[int]:
        """Add special tokens around the input sequences."""
        cls_token_id = self.vocab_dict.get(self.cls_token, 0)
        eos_token_id = self.vocab_dict.get(self.eos_token, 2)

        if token_ids_1 is None:
            return [cls_token_id] + token_ids_0 + [eos_token_id]
        else:
            return [cls_token_id] + token_ids_0 + [eos_token_id] + token_ids_1 + [eos_token_id]

    def save_vocabulary(self, save_directory: str, filename_prefix: str | None = None) -> tuple[str]:
        """Save the vocabulary to a file."""
        prefix = filename_prefix + "-" if filename_prefix else ""
        vocab_file = os.path.join(
            save_directory,
            f"{prefix}vocab_nucleotide.txt",
        )

        # Sort tokens by their IDs to maintain order
        sorted_tokens = [self.ids_to_tokens[i] for i in sorted(self.ids_to_tokens.keys())]

        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write("\n".join(sorted_tokens))

        return (vocab_file,)
