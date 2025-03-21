import pytest
from lobster.tokenization._smiles_tokenizer import SmilesTokenizerFast, _make_smiles_tokenizer
from transformers import PreTrainedTokenizerFast


@pytest.fixture
def tokenizer():
    return SmilesTokenizerFast()


def test__make_smiles_tokenizer():
    tokenizer = _make_smiles_tokenizer(save_dirpath=None)

    assert isinstance(tokenizer, PreTrainedTokenizerFast)

    assert tokenizer.cls_token == "<cls>"
    assert tokenizer.eos_token == "<eos>"
    assert tokenizer.unk_token == "<unk>"
    assert tokenizer.pad_token == "<pad>"
    assert tokenizer.sep_token == "<sep>"
    assert tokenizer.mask_token == "<mask>"

    ids = tokenizer.encode("CCO")
    assert ids == [0, 8, 8, 12, 2]
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
        assert ids == [0, 8, 8, 12, 2]
        assert tokenizer.decode(ids) == "<cls> C C O <eos>"
        assert tokenizer.special_tokens_map == {
            "eos_token": "<eos>",
            "unk_token": "<unk>",
            "sep_token": "<sep>",
            "pad_token": "<pad>",
            "cls_token": "<cls>",
            "mask_token": "<mask>",
        }
