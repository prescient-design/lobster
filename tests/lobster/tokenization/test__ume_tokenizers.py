import pytest
import torch
from lobster.constants import Modality
from lobster.tokenization._ume_tokenizers import (
    Ume2ModTokenizerTransform,
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
        assert transform._max_length is None
        assert isinstance(transform.tokenizer, UmeSmilesTokenizerFast)

    def test_forward(self):
        transform = UmeTokenizerTransform(modality="SMILES", max_length=4, return_modality=True)

        out = transform("C")

        assert out["input_ids"].tolist()[0] == [0, 52, 2, 1]
        assert out["attention_mask"].tolist()[0] == [1, 1, 1, 0]
        assert out["modality"] == Modality.SMILES


class TestUme2ModTokenizerTransform:
    def test__init__(self):
        """Test initialization of the Ume2ModTokenizerTransform."""
        # Test normal initialization
        tokenizer = Ume2ModTokenizerTransform(
            modality1="amino_acid", modality2="SMILES", max_length=20, mode="interact"
        )

        assert tokenizer.modality1 == Modality.AMINO_ACID
        assert tokenizer.modality2 == Modality.SMILES
        assert tokenizer.max_length == 20
        assert tokenizer.mode == "interact"
        assert tokenizer.task_token == "<interact>"

        # Test with invalid mode
        with pytest.raises(ValueError, match="mode must be either 'interact' or 'convert'"):
            Ume2ModTokenizerTransform(modality1="amino_acid", modality2="SMILES", max_length=20, mode="invalid_mode")

    def test_forward(self):
        """Test forward method with amino acid and SMILES inputs."""
        tokenizer = Ume2ModTokenizerTransform(
            modality1="amino_acid", modality2="SMILES", max_length=20, mode="interact"
        )

        # Test with the example input
        output = tokenizer(("MYK", "CCO"))

        # Check output structure
        assert "input_ids" in output
        assert "attention_mask" in output
        assert "modality1" in output
        assert "modality2" in output
        assert "mode" in output

        # Check types
        assert isinstance(output["input_ids"], torch.Tensor)
        assert isinstance(output["attention_mask"], torch.Tensor)
        assert output["modality1"] == Modality.AMINO_ACID
        assert output["modality2"] == Modality.SMILES
        assert output["mode"] == "interact"

        # Check tensor shapes
        assert output["input_ids"].shape == (20,)
        assert output["attention_mask"].shape == (20,)

        # Check the token sequence structure
        input_ids = output["input_ids"].tolist()
        attention_mask = output["attention_mask"].tolist()

        # Should start with CLS token
        assert input_ids[0] == tokenizer.cls_id

        # Check that padding works correctly (attention mask)
        assert 0 in attention_mask  # Should have padding

        # Check that the sequence contains the task token
        assert tokenizer.task_token_id in input_ids

        # Verify tokens can be decoded
        tokens1 = tokenizer.tokenizer1.convert_ids_to_tokens(input_ids)
        tokens2 = tokenizer.tokenizer2.convert_ids_to_tokens(input_ids)

        # Basic check that decoding works without errors
        assert isinstance(tokens1, list)
        assert isinstance(tokens2, list)
        assert len(tokens1) == 20
        assert len(tokens2) == 20

    @pytest.mark.parametrize(
        "modality1,modality2,input_pair,mode",
        [
            pytest.param("amino_acid", "SMILES", ("MYK", "CCO"), "interact", id="amino_acid_SMILES_interact"),
            pytest.param("amino_acid", "nucleotide", ("MYK", "ATCG"), "convert", id="amino_acid_nucleotide_convert"),
        ],
    )
    def test_forward_different_modalities(self, modality1, modality2, input_pair, mode):
        """Test forward method with different modality combinations."""
        tokenizer = Ume2ModTokenizerTransform(modality1=modality1, modality2=modality2, max_length=30, mode=mode)

        output = tokenizer(input_pair)

        # Check output structure and types
        assert isinstance(output["input_ids"], torch.Tensor)
        assert isinstance(output["attention_mask"], torch.Tensor)
        assert output["modality1"] == Modality(modality1)
        assert output["modality2"] == Modality(modality2)
        assert output["mode"] == mode

        # Check shapes
        assert output["input_ids"].shape == (30,)
        assert output["attention_mask"].shape == (30,)
