from transformers import PreTrainedTokenizerFast

from lobster.tokenization._nucleotide_tokenizer import NucleotideTokenizerFast, _make_nucleotide_tokenizer


def test__make_nucleotide_tokenizer():
    tokenizer = _make_nucleotide_tokenizer(save_dirpath=None)

    assert isinstance(tokenizer, PreTrainedTokenizerFast)

    assert tokenizer.cls_token == "<cls>"
    assert tokenizer.eos_token == "<eos>"
    assert tokenizer.unk_token == "<unk>"
    assert tokenizer.pad_token == "<pad>"
    assert tokenizer.sep_token == "<sep>"
    assert tokenizer.mask_token == "<mask>"

    assert tokenizer.vocab_size == 12

    assert tokenizer.special_tokens_map == {
        "eos_token": "<eos>",
        "unk_token": "<unk>",
        "sep_token": "<sep>",
        "pad_token": "<pad>",
        "cls_token": "<cls>",
        "mask_token": "<mask>",
    }

    tokenized_output = tokenizer("ATcg")
    assert tokenized_output.input_ids == [0, 6, 9, 7, 8, 2]
    assert tokenizer.decode(tokenized_output.input_ids) == "<cls> a t c g <eos>"


class TestNucleotideTokenizerFaset:
    def test__init__(self):
        tokenizer = NucleotideTokenizerFast()

        assert isinstance(tokenizer, PreTrainedTokenizerFast)

        assert tokenizer.vocab_size == 12

        assert tokenizer.cls_token == "<cls>"
        assert tokenizer.eos_token == "<eos>"
        assert tokenizer.unk_token == "<unk>"
        assert tokenizer.pad_token == "<pad>"
        assert tokenizer.sep_token == "<sep>"
        assert tokenizer.mask_token == "<mask>"

        tokenized_output = tokenizer("atCG")
        assert tokenized_output.input_ids == [0, 6, 9, 7, 8, 2]
        assert tokenizer.decode(tokenized_output.input_ids) == "<cls> a t c g <eos>"

        tokenized_output = tokenizer("RAW")
        assert tokenized_output.input_ids == [0, 3, 6, 3, 2]
        assert tokenizer.decode(tokenized_output.input_ids) == "<cls> <unk> a <unk> <eos>"

        assert tokenizer.special_tokens_map == {
            "eos_token": "<eos>",
            "unk_token": "<unk>",
            "sep_token": "<sep>",
            "pad_token": "<pad>",
            "cls_token": "<cls>",
            "mask_token": "<mask>",
        }
