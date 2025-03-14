from lobster.tokenization._ume_tokenizers import (
    UmeAminoAcidTokenizerFast,
    UmeLatentGenerator3DCoordTokenizerFast,
    UmeNucleotideTokenizerFast,
    UmeSmilesTokenizerFast,
    _add_reserved_tokens,
)


def test_add_reserved_tokens():
    mock_vocabs = {
        "special_tokens": [
            "<cls>",
            "<eos>",
            "<unk>",
            "<pad>",
            "<reserved_special_token_0>",
            "<reserved_special_token_1>",
            "<reserved_special_token_2>",
        ],
        "amino_acid_tokenizer": ["A"],  # 1 amino acid tokens
        "smiles_tokenizer": ["C", "O"],  # 2 SMILES tokens
        "nucleotide_tokenizer": ["A", "C", "G"],  # 3 nucleotide tokens
        "latent_generator_tokenizer": ["X1", "Y1", "Z1", "X2"],  # 4 latent tokens
    }

    result = _add_reserved_tokens(mock_vocabs)

    assert result["amino_acid_tokenizer"] == [
        "<cls>",
        "<eos>",
        "<unk>",
        "<pad>",
        "<reserved_special_token_0>",  # reserved from special tokens
        "<reserved_special_token_1>",  # reserved from special tokens
        "<reserved_special_token_2>",  # reserved from special tokens
        "A",
    ]
    assert result["smiles_tokenizer"] == [
        "<cls>",
        "<eos>",
        "<unk>",
        "<pad>",
        "<reserved_special_token_0>",  # reserved from special tokens
        "<reserved_special_token_1>",  # reserved from special tokens
        "<reserved_special_token_2>",  # reserved from special tokens
        "<reserved_special_token_3>",  # reserved for amino acids
        "C",
        "O",
    ]
    assert result["nucleotide_tokenizer"] == [
        "<cls>",
        "<eos>",
        "<unk>",
        "<pad>",
        "<reserved_special_token_0>",  # reserved from special tokens
        "<reserved_special_token_1>",  # reserved from special tokens
        "<reserved_special_token_2>",  # reserved from special tokens
        "<reserved_special_token_3>",  # reserved for amino acids
        "<reserved_special_token_4>",  # reserved for SMILES
        "<reserved_special_token_5>",  # reserved for SMILES
        "A",
        "C",
        "G",
    ]
    assert result["latent_generator_tokenizer"] == [
        "<cls>",
        "<eos>",
        "<unk>",
        "<pad>",
        "<reserved_special_token_0>",  # reserved from special tokens
        "<reserved_special_token_1>",  # reserved from special tokens
        "<reserved_special_token_2>",  # reserved from special tokens
        "<reserved_special_token_3>",  # reserved for amino acids
        "<reserved_special_token_4>",  # reserved for SMILES
        "<reserved_special_token_5>",  # reserved for SMILES
        "<reserved_special_token_6>",  # reserved for nucleotides
        "<reserved_special_token_7>",  # reserved for nucleotides
        "<reserved_special_token_8>",  # reserved for nucleotides
        "X1",
        "Y1",
        "Z1",
        "X2",
    ]


def test_ume_aminio_acid_tokenizer():
    tokenizer = UmeAminoAcidTokenizerFast()
    assert tokenizer.tokenize("VYF") == ["V", "Y", "F"]
    assert tokenizer.encode("VYF", padding="do_not_pad", add_special_tokens=True) == [0, 23, 35, 34, 2]


def test_ume_smiles_tokenizer():
    tokenizer = UmeSmilesTokenizerFast()
    assert tokenizer.tokenize("CCO") == ["C", "C", "O"]
    assert tokenizer.encode("CCO", padding="do_not_pad", add_special_tokens=True) == [0, 46, 46, 49, 2]


def test_ume_nucleotide_tokenizer():
    tokenizer = UmeNucleotideTokenizerFast()
    assert tokenizer.tokenize("ACGT") == ["A", "C", "G", "T"]
    assert tokenizer.encode("ACGT", padding="do_not_pad", add_special_tokens=True) == [0, 623, 624, 625, 626, 2]


def test_ume_latent_generator_tokenizer():
    tokenizer = UmeLatentGenerator3DCoordTokenizerFast()
    assert tokenizer.tokenize("gd fh ds") == ["gd", "fh", "ds"]
    assert tokenizer.encode("gd fh ds", padding="do_not_pad", add_special_tokens=True) == [0, 816, 794, 753, 2]
