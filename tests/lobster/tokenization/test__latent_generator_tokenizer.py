from transformers import PreTrainedTokenizerFast

from lobster.tokenization._latent_generator_tokenizer import LatentGeneratorTokenizerFast, _make_latent_generator_tokenizer


def test__make_latent_generator_tokenizer():
    tokenizer = _make_latent_generator_tokenizer()

    assert isinstance(tokenizer, PreTrainedTokenizerFast)

    assert tokenizer.cls_token == "<cls>"
    assert tokenizer.eos_token == "<eos>"
    assert tokenizer.unk_token == "<unk>"
    assert tokenizer.pad_token == "<pad>"
    assert tokenizer.mask_token == "<mask>"

    assert tokenizer.vocab_size == 262

    assert tokenizer.special_tokens_map == {
        "eos_token": "<eos>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "cls_token": "<cls>",
        "mask_token": "<mask>",
    }

    tokenized_output = tokenizer("gd fh ds fh ad gf fe cz ek ds cq")

    assert tokenized_output.input_ids == [0, 191, 169, 128, 169, 35, 193, 166, 109, 146, 128, 100, 2]
    assert tokenizer.decode(tokenized_output.input_ids) == "<cls> gd fh ds fh ad gf fe cz ek ds cq <eos>"

    tokenized_output = tokenizer("GD FH DS FH AD GF FE CZ EK DS CQ")

    assert tokenized_output.input_ids == [0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2]
    assert tokenizer.decode(tokenized_output.input_ids) == "<cls> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <eos>"

    tokenized_output = tokenizer("R A gd fh ds")
    assert tokenized_output.input_ids == [0, 3, 3, 191, 169, 128, 2]
    assert tokenizer.decode(tokenized_output.input_ids) == "<cls> <unk> <unk> gd fh ds <eos>"


class TestLatentGeneratorTokenizerFast:
    def test__init__(self):
        tokenizer = LatentGeneratorTokenizerFast()

        assert isinstance(tokenizer, PreTrainedTokenizerFast)

        assert tokenizer.vocab_size == 262

        assert tokenizer.cls_token == "<cls>"
        assert tokenizer.eos_token == "<eos>"
        assert tokenizer.unk_token == "<unk>"
        assert tokenizer.pad_token == "<pad>"
        assert tokenizer.mask_token == "<mask>"

        tokenized_output = tokenizer("gd fh ds fh ad gf fe cz ek ds cq")
        assert tokenized_output.input_ids == [0, 191, 169, 128, 169, 35, 193, 166, 109, 146, 128, 100, 2]
        assert tokenizer.decode(tokenized_output.input_ids) == "<cls> gd fh ds fh ad gf fe cz ek ds cq <eos>"

        tokenized_output = tokenizer("GD FH DS FH AD GF FE CZ EK DS CQ")
        assert tokenized_output.input_ids == [0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2]
        assert tokenizer.decode(tokenized_output.input_ids) == "<cls> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <eos>"

        tokenized_output = tokenizer("R A gd fh ds")
        assert tokenized_output.input_ids == [0, 3, 3, 191, 169, 128, 2]
        assert tokenizer.decode(tokenized_output.input_ids) == "<cls> <unk> <unk> gd fh ds <eos>"

        assert tokenizer.special_tokens_map == {
            "eos_token": "<eos>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
            "cls_token": "<cls>",
            "mask_token": "<mask>",
        }

if __name__ == "__main__":
    test__make_latent_generator_tokenizer()
    TestLatentGeneratorTokenizerFast().test__init__()