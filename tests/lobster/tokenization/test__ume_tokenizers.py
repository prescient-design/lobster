import pytest
import torch

from lobster.constants import Modality
from lobster.tokenization._ume_tokenizers import (
    UmeAminoAcidTokenizerFast,
    UmeLatentGenerator3DCoordTokenizerFast,
    UmeNucleotideTokenizerFast,
    UmeSmilesTokenizerFast,
    UmeTokenizerTransform,
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
        "coordinates_3d_tokenizer": ["X1", "Y1", "Z1", "X2"],  # 4 latent tokens
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
    assert result["coordinates_3d_tokenizer"] == [
        "<cls>",
        "<eos>",
        "<unk>",
        "<pad>",
        "<extra_special_token_0>",  # reserved from special tokens
        "<extra_special_token_1>",  # reserved from special tokens
        "<reserved_for_amino_acids_0>",  # reserved from special tokens
        "<reserved_for_smiles_0>",  # reserved for SMILES
        "<reserved_for_smiles_1>",  # reserved for SMILES
        "<reserved_for_nucleotide_0>",  # reserved for nucleotides
        "<reserved_for_nucleotide_1>",  # reserved for nucleotides
        "<reserved_for_nucleotide_2>",  # reserved for nucleotides
        "X1",
        "Y1",
        "Z1",
        "X2",
    ]


def test_ume_amino_acid_tokenizer():
    tokenizer = UmeAminoAcidTokenizerFast()
    encoded = [1, 28, 40, 39, 5]

    assert tokenizer.tokenize("VYF") == ["V", "Y", "F"]
    assert tokenizer.encode("VYF", padding="do_not_pad", add_special_tokens=True) == encoded
    assert tokenizer.decode(encoded) == "<cls_amino_acid> V Y F <eos>"


def test_ume_smiles_tokenizer():
    tokenizer = UmeSmilesTokenizerFast()
    encoded = [2, 52, 52, 56, 5]

    assert tokenizer.tokenize("CCO") == ["C", "C", "O"]
    assert tokenizer.encode("CCO", padding="do_not_pad", add_special_tokens=True) == encoded
    assert tokenizer.decode(encoded) == "<cls_smiles> C C O <eos>"
    assert tokenizer.tokenize("C[C@@H]") == ["C", "[C@@H]"]


def test_ume_nucleotide_tokenizer():
    tokenizer = UmeNucleotideTokenizerFast()
    encoded = [3, 1272, 1273, 1274, 1275, 5]
    assert tokenizer.tokenize("acGT") == ["a", "c", "g", "t"]
    assert tokenizer.encode("acGT", padding="do_not_pad", add_special_tokens=True) == encoded
    assert tokenizer.decode(encoded) == "<cls_nucleotide> a c g t <eos>"


def test_ume_latent_generator_tokenizer():
    tokenizer = UmeLatentGenerator3DCoordTokenizerFast()
    encoded = [4, 1465, 1443, 1402, 5]
    assert tokenizer.tokenize("gd fh ds") == ["gd", "fh", "ds"]
    assert tokenizer.encode("gd fh ds", padding="do_not_pad", add_special_tokens=True) == encoded
    assert tokenizer.decode(encoded) == "<cls_3d_coordinates> gd fh ds <eos>"


class TestUmeTokenizerTransform:
    def test__init__(self):
        with pytest.warns(UserWarning, match="UmeTokenizerTransform did not receive `max_length` parameter"):
            transform = UmeTokenizerTransform(modality="SMILES", max_length=None, return_modality=False)

        assert transform.modality == Modality.SMILES
        assert transform.max_length is None
        assert isinstance(transform.tokenizer, UmeSmilesTokenizerFast)

    @pytest.mark.parametrize(
        "modality,input_text,max_length,expected_input_ids,expected_attention_mask,expected_modality",
        [
            pytest.param("SMILES", "[C@@H]", 6, [2, 64, 5, 7, 7, 7], [1, 1, 1, 0, 0, 0], Modality.SMILES, id="smiles"),
            pytest.param(
                "amino_acid", "V", 6, [1, 28, 5, 7, 7, 7], [1, 1, 1, 0, 0, 0], Modality.AMINO_ACID, id="amino_acid"
            ),
            pytest.param(
                "nucleotide",
                "acT",
                6,
                [3, 1272, 1273, 1275, 5, 7],
                [1, 1, 1, 1, 1, 0],
                Modality.NUCLEOTIDE,
                id="nucleotide",
            ),
            pytest.param(
                "3d_coordinates",
                ["gd ad", "gd ad", "gd ad", "gd ad"],
                6,
                [4, 1465, 1309, 5, 7, 7],
                [1, 1, 1, 1, 0, 0],
                Modality.COORDINATES_3D,
                id="3d_coordinates_list",
            ),
        ],
    )
    def test_single_modalities(
        self, modality, input_text, max_length, expected_input_ids, expected_attention_mask, expected_modality
    ):
        transform = UmeTokenizerTransform(modality=modality, max_length=max_length, return_modality=True)
        out = transform(input_text)

        assert out["input_ids"].tolist()[0] == expected_input_ids
        assert out["attention_mask"].tolist()[0] == expected_attention_mask
        assert out["modality"] == expected_modality

    @pytest.mark.parametrize(
        "modality,input_batch,max_length,expected_input_ids",
        [
            pytest.param(
                "SMILES", ["C", "CCO"], 6, torch.tensor([[2, 52, 5, 7, 7, 7], [2, 52, 52, 56, 5, 7]]), id="smiles_batch"
            ),
            pytest.param(
                "amino_acid",
                ["AR", "VYK"],
                6,
                torch.tensor([[1, 26, 31, 5, 7, 7], [1, 28, 40, 36, 5, 7]]),
                id="amino_acid_batch",
            ),
            pytest.param(
                "nucleotide",
                ["at", "Cg"],
                6,
                torch.tensor([[3, 1272, 1275, 5, 7, 7], [3, 1273, 1274, 5, 7, 7]]),
                id="nucleotide_batch",
            ),
        ],
    )
    def test_single_modalities_batch_input(self, modality, input_batch, max_length, expected_input_ids):
        transform = UmeTokenizerTransform(modality=modality, max_length=max_length, return_modality=True)
        out = transform(input_batch)
        assert torch.equal(out["input_ids"], expected_input_ids)

    @pytest.mark.parametrize(
        "modalities,max_length,mode,input_data,expected_input_ids,expected_decoded_tokens1,expected_decoded_tokens2",
        [
            pytest.param(
                ("amino_acid", "SMILES"),
                10,
                "interact",
                ("AWY", "C"),
                torch.tensor([[11, 1, 26, 43, 40, 8, 2, 52, 8, 7]]),
                "<cls_interact> <cls_amino_acid> A W Y <sep> <cls_smiles> <sep> <pad>",
                "<cls_interact> <cls_amino_acid> <reserved_for_amino_acids_3> <reserved_for_amino_acids_20> <reserved_for_amino_acids_17> <sep> <cls_smiles> C <sep> <pad>",
                id="amino_acid_smiles_interact",
            ),
            pytest.param(
                ("nucleotide", "3d_coordinates"),
                10,
                "convert",
                ("a", ["cz ge", "cz ge", "cz ge", "cz ge"]),
                torch.tensor([[10, 3, 1272, 8, 4, 1383, 1466, 8, 7, 7]]),
                "<cls_convert> <cls_nucleotide> a <sep> <cls_3d_coordinates> <sep> <pad> <pad>",
                "<cls_convert> <cls_nucleotide> <reserved_for_nucleotide_2> <sep> <cls_3d_coordinates> cz ge <sep> <pad> <pad>",
                id="nucleotide_3d_coordinates_convert",
            ),
        ],
    )
    def test_dual_modalities(
        self,
        modalities,
        max_length,
        mode,
        input_data,
        expected_input_ids,
        expected_decoded_tokens1,
        expected_decoded_tokens2,
    ):
        transform = UmeTokenizerTransform(modality=modalities, max_length=max_length, return_modality=True, mode=mode)
        out = transform(input_data)

        assert torch.equal(out["input_ids"], expected_input_ids)

        decoded_tokens1 = transform.tokenizer1.decode(out["input_ids"][0])
        decoded_tokens2 = transform.tokenizer2.decode(out["input_ids"][0])

        assert decoded_tokens1 == expected_decoded_tokens1
        assert decoded_tokens2 == expected_decoded_tokens2
