import pytest
import torch

from lobster.constants import Modality
from lobster.tokenization._ume_tokenizers import (
    UMEAminoAcidTokenizerFast,
    UMENucleotideTokenizerFast,
    UMESmilesTokenizerFast,
    UMETokenizerTransform,
    _add_reserved_tokens,
)


def test_add_reserved_tokens():
    mock_vocabs = {
        "special_tokens": [
            "<cls>",
            "<eos>",
            "<unk>",
            "<pad>",
            "<extra_special_token_0>",
            "<extra_special_token_1>",
        ],
        "amino_acid_tokenizer": ["A"],  # 1 amino acid tokens
        "smiles_tokenizer": ["C", "O"],  # 2 SMILES tokens
        "nucleotide_tokenizer": ["A", "C", "G"],  # 3 nucleotide tokens
    }

    result = _add_reserved_tokens(mock_vocabs)

    assert result["amino_acid_tokenizer"] == [
        "<cls>",
        "<eos>",
        "<unk>",
        "<pad>",
        "<extra_special_token_0>",  # reserved from special tokens
        "<extra_special_token_1>",  # reserved from special tokens
        "A",
    ]
    assert result["smiles_tokenizer"] == [
        "<cls>",
        "<eos>",
        "<unk>",
        "<pad>",
        "<extra_special_token_0>",  # reserved from special tokens
        "<extra_special_token_1>",  # reserved from special tokens
        "<reserved_for_amino_acids_0>",  # reserved for amino acids
        "C",
        "O",
    ]
    assert result["nucleotide_tokenizer"] == [
        "<cls>",
        "<eos>",
        "<unk>",
        "<pad>",
        "<extra_special_token_0>",  # reserved from special tokens
        "<extra_special_token_1>",  # reserved from special tokens
        "<reserved_for_amino_acids_0>",  # reserved from special tokens
        "<reserved_for_smiles_0>",  # reserved for SMILES
        "<reserved_for_smiles_1>",  # reserved for SMILES
        "A",
        "C",
        "G",
    ]


def test_ume_amino_acid_tokenizer():
    tokenizer = UMEAminoAcidTokenizerFast()
    encoded = [1, 30, 42, 41, 4]

    assert tokenizer.tokenize("VYF") == ["V", "Y", "F"]
    assert tokenizer.encode("VYF", padding="do_not_pad", add_special_tokens=True) == encoded
    assert tokenizer.decode(encoded) == "<cls_amino_acid> V Y F <eos>"


def test_ume_smiles_tokenizer():
    tokenizer = UMESmilesTokenizerFast()
    encoded = [2, 54, 54, 58, 4]

    assert tokenizer.tokenize("CCO") == ["C", "C", "O"]
    assert tokenizer.encode("CCO", padding="do_not_pad", add_special_tokens=True) == encoded
    assert tokenizer.decode(encoded) == "<cls_smiles> C C O <eos>"
    assert tokenizer.tokenize("C[C@@H]") == ["C", "[C@@H]"]


def test_ume_nucleotide_tokenizer():
    tokenizer = UMENucleotideTokenizerFast()
    encoded = [3, 1274, 1275, 1276, 1277, 4]
    assert tokenizer.tokenize("acGT") == ["a", "c", "g", "t"]
    assert tokenizer.encode("acGT", padding="do_not_pad", add_special_tokens=True) == encoded
    assert tokenizer.decode(encoded) == "<cls_nucleotide> a c g t <eos>"


class TestUMETokenizerTransform:
    def test__init__(self):
        with pytest.warns(UserWarning, match="UMETokenizerTransform did not receive `max_length` parameter"):
            transform = UMETokenizerTransform(modality="SMILES", max_length=None, return_modality=False)

        assert transform.modality == Modality.SMILES
        assert transform.max_length is None
        assert isinstance(transform.tokenizer, UMESmilesTokenizerFast)

    @pytest.mark.parametrize(
        "modality,input_text,max_length,expected_input_ids,expected_attention_mask,expected_modality",
        [
            pytest.param("SMILES", "[C@@H]", 6, [2, 66, 4, 6, 6, 6], [1, 1, 1, 0, 0, 0], Modality.SMILES, id="smiles"),
            pytest.param(
                "amino_acid", "V", 6, [1, 30, 4, 6, 6, 6], [1, 1, 1, 0, 0, 0], Modality.AMINO_ACID, id="amino_acid"
            ),
            pytest.param(
                "nucleotide",
                "acT",
                6,
                [3, 1274, 1275, 1277, 4, 6],
                [1, 1, 1, 1, 1, 0],
                Modality.NUCLEOTIDE,
                id="nucleotide",
            ),
        ],
    )
    def test_single_modalities(
        self, modality, input_text, max_length, expected_input_ids, expected_attention_mask, expected_modality
    ):
        transform = UMETokenizerTransform(modality=modality, max_length=max_length, return_modality=True)
        out = transform(input_text)

        assert out["input_ids"].tolist()[0] == expected_input_ids
        assert out["attention_mask"].tolist()[0] == expected_attention_mask
        assert out["modality"] == expected_modality

    @pytest.mark.parametrize(
        "modality,input_batch,max_length,expected_input_ids",
        [
            pytest.param(
                "SMILES", ["C", "CCO"], 6, torch.tensor([[2, 54, 4, 6, 6, 6], [2, 54, 54, 58, 4, 6]]), id="smiles_batch"
            ),
            pytest.param(
                "amino_acid",
                ["AR", "VYK"],
                6,
                torch.tensor([[1, 28, 33, 4, 6, 6], [1, 30, 42, 38, 4, 6]]),
                id="amino_acid_batch",
            ),
            pytest.param(
                "nucleotide",
                ["at", "Cg"],
                6,
                torch.tensor([[3, 1274, 1277, 4, 6, 6], [3, 1275, 1276, 4, 6, 6]]),
                id="nucleotide_batch",
            ),
        ],
    )
    def test_single_modalities_batch_input(self, modality, input_batch, max_length, expected_input_ids):
        transform = UMETokenizerTransform(modality=modality, max_length=max_length, return_modality=True)
        out = transform(input_batch)
        assert torch.equal(out["input_ids"], expected_input_ids)
