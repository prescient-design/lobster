from unittest.mock import patch

import pytest
from lobster.tokenization._smiles_tokenizer import SmilesTokenizerFast, _make_smiles_tokenizer
from transformers import PreTrainedTokenizerFast


@pytest.fixture
def tokenizer():
    return SmilesTokenizerFast()


@patch("lobster.tokenization._smiles_tokenizer.load_vocab_file")
def test__make_smiles_tokenizer(mock_load_vocab_file):
    mock_vocab = ["<pad>", "<unk>", "<cls>", "<sep>", "<mask>", "<eos>", "c", "C", "(", ")", "O", "1", "2", "=", "N"]
    mock_load_vocab_file.return_value = mock_vocab

    tokenizer = _make_smiles_tokenizer()

    assert isinstance(tokenizer, PreTrainedTokenizerFast)

    assert tokenizer.cls_token == "<cls>"
    assert tokenizer.eos_token == "<eos>"
    assert tokenizer.unk_token == "<unk>"
    assert tokenizer.pad_token == "<pad>"
    assert tokenizer.sep_token == "<sep>"
    assert tokenizer.mask_token == "<mask>"

    assert tokenizer.vocab_size == len(mock_vocab)

    ids = tokenizer.encode("CCO")
    assert ids == [2, 7, 7, 10, 5]
    assert tokenizer.decode(ids) == "<cls> C C O <eos>"
    assert tokenizer.special_tokens_map == {
        "eos_token": "<eos>",
        "unk_token": "<unk>",
        "sep_token": "<sep>",
        "pad_token": "<pad>",
        "cls_token": "<cls>",
        "mask_token": "<mask>",
    }


class TestSmilesTokenizerFast:
    def test_smiles_tokenizer_fast(self, tokenizer):
        assert isinstance(tokenizer, PreTrainedTokenizerFast)

        assert tokenizer.cls_token == "<cls>"
        assert tokenizer.eos_token == "<eos>"
        assert tokenizer.unk_token == "<unk>"
        assert tokenizer.pad_token == "<pad>"
        assert tokenizer.sep_token == "<sep>"
        assert tokenizer.mask_token == "<mask>"

        ids = tokenizer.encode("CCO")
        assert ids == [2, 7, 7, 10, 5]
        assert tokenizer.decode(ids) == "<cls> C C O <eos>"
        assert tokenizer.special_tokens_map == {
            "eos_token": "<eos>",
            "unk_token": "<unk>",
            "sep_token": "<sep>",
            "pad_token": "<pad>",
            "cls_token": "<cls>",
            "mask_token": "<mask>",
        }
