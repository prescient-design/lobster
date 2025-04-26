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
        "latent_generator_tokenizer": ["X1", "Y1", "Z1", "X2"],  # 4 latent tokens
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
        "<reserved_for_amino_acids_special_token_2>",  # reserved for amino acids
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
        "<reserved_for_amino_acids_special_token_2>",  # reserved from special tokens
        "<reserved_for_smiles_special_token_3>",  # reserved for SMILES
        "<reserved_for_smiles_special_token_4>",  # reserved for SMILES
        "A",
        "C",
        "G",
    ]
    assert result["latent_generator_tokenizer"] == [
        "<cls>",
        "<eos>",
        "<unk>",
        "<pad>",
        "<extra_special_token_0>",  # reserved from special tokens
        "<extra_special_token_1>",  # reserved from special tokens
        "<reserved_for_amino_acids_special_token_2>",  # reserved from special tokens
        "<reserved_for_smiles_special_token_3>",  # reserved for SMILES
        "<reserved_for_smiles_special_token_4>",  # reserved for SMILES
        "<reserved_for_nucleotides_special_token_5>",  # reserved for nucleotides
        "<reserved_for_nucleotides_special_token_6>",  # reserved for nucleotides
        "<reserved_for_nucleotides_special_token_7>",  # reserved for nucleotides
        "X1",
        "Y1",
        "Z1",
        "X2",
    ]


def test_ume_aminio_acid_tokenizer():
    tokenizer = UmeAminoAcidTokenizerFast()
    assert tokenizer.tokenize("VYF") == ["V", "Y", "F"]
    assert tokenizer.encode("VYF", padding="do_not_pad", add_special_tokens=True) == [0, 28, 40, 39, 2]
    assert tokenizer.convert_tokens_to_ids("<convert>") == 6
    assert tokenizer.convert_tokens_to_ids("<interact>") == 7


def test_ume_smiles_tokenizer():
    tokenizer = UmeSmilesTokenizerFast()
    assert tokenizer.tokenize("CCO") == ["C", "C", "O"]
    assert tokenizer.encode("CCO", padding="do_not_pad", add_special_tokens=True) == [0, 52, 52, 56, 2]
    assert tokenizer.convert_tokens_to_ids("<convert>") == 6
    assert tokenizer.convert_tokens_to_ids("<interact>") == 7


def test_ume_nucleotide_tokenizer():
    tokenizer = UmeNucleotideTokenizerFast()
    assert tokenizer.tokenize("acGT") == ["a", "c", "g", "t"]
    assert tokenizer.encode("acGT", padding="do_not_pad", add_special_tokens=True) == [0, 1272, 1273, 1274, 1275, 2]
    assert tokenizer.convert_tokens_to_ids("<convert>") == 6
    assert tokenizer.convert_tokens_to_ids("<interact>") == 7


def test_ume_latent_generator_tokenizer():
    tokenizer = UmeLatentGenerator3DCoordTokenizerFast()
    assert tokenizer.tokenize("gd fh ds") == ["gd", "fh", "ds"]
    assert tokenizer.encode("gd fh ds", padding="do_not_pad", add_special_tokens=True) == [0, 1465, 1443, 1402, 2]
    assert tokenizer.convert_tokens_to_ids("<convert>") == 6
    assert tokenizer.convert_tokens_to_ids("<interact>") == 7


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
            pytest.param("SMILES", "C", 6, [0, 52, 2, 1, 1, 1], [1, 1, 1, 0, 0, 0], Modality.SMILES, id="smiles"),
            pytest.param(
                "amino_acid", "V", 6, [0, 28, 2, 1, 1, 1], [1, 1, 1, 0, 0, 0], Modality.AMINO_ACID, id="amino_acid"
            ),
            pytest.param(
                "nucleotide",
                "acT",
                6,
                [0, 1272, 1273, 1275, 2, 1],
                [1, 1, 1, 1, 1, 0],
                Modality.NUCLEOTIDE,
                id="nucleotide",
            ),
            pytest.param(
                "3d_coordinates",
                ["gd ad", "gd ad", "gd ad", "gd ad"],
                6,
                [0, 1465, 1309, 2, 1, 1],
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
                "SMILES", ["C", "CCO"], 6, torch.tensor([[0, 52, 2, 1, 1, 1], [0, 52, 52, 56, 2, 1]]), id="smiles_batch"
            ),
            pytest.param(
                "amino_acid",
                ["AR", "VYK"],
                6,
                torch.tensor([[0, 26, 31, 2, 1, 1], [0, 28, 40, 36, 2, 1]]),
                id="amino_acid_batch",
            ),
            pytest.param(
                "nucleotide",
                ["at", "Cg"],
                6,
                torch.tensor([[0, 1272, 1275, 2, 1, 1], [0, 1273, 1274, 2, 1, 1]]),
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
                torch.tensor([[0, 26, 43, 40, 5, 7, 52, 5, 1, 1]]),
                ["<cls>", "A", "W", "Y", "<sep>", "<interact>", None, "<sep>", "<pad>", "<pad>"],
                [
                    "<cls>",
                    "<reserved_for_amino_acids_special_token_20>",
                    "<reserved_for_amino_acids_special_token_37>",
                    "<reserved_for_amino_acids_special_token_34>",
                    "<sep>",
                    "<interact>",
                    "C",
                    "<sep>",
                    "<pad>",
                    "<pad>",
                ],
                id="amino_acid_smiles_interact",
            ),
            pytest.param(
                ("nucleotide", "3d_coordinates"),
                8,
                "convert",
                ("a", ["cz ge", "cz ge", "cz ge", "cz ge"]),
                torch.tensor([[0, 1272, 5, 6, 1383, 1466, 5, 1]]),
                ["<cls>", "a", "<sep>", "<convert>", None, None, "<sep>", "<pad>"],
                [
                    "<cls>",
                    "<reserved_for_nucleotides_special_token_1266>",
                    "<sep>",
                    "<convert>",
                    "cz",
                    "ge",
                    "<sep>",
                    "<pad>",
                ],
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

        decoded_tokens1 = transform.tokenizer1.convert_ids_to_tokens(out["input_ids"][0])
        decoded_tokens2 = transform.tokenizer2.convert_ids_to_tokens(out["input_ids"][0])

        assert decoded_tokens1 == expected_decoded_tokens1
        assert decoded_tokens2 == expected_decoded_tokens2
