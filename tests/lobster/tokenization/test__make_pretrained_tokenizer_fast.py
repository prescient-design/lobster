from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

from lobster.tokenization._make_pretrained_tokenizer_fast import make_pretrained_tokenizer_fast


def test_make_pretrained_tokenizer_fast():
    model = BPE(
        vocab={
            "<cls>": 0,
            "<eos>": 1,
            "<unk>": 2,
            "Hello": 3,
            "world": 4,
        },
        ignore_merges=True,
        merges=[],
    )
    pre_tokenizer = Whitespace()
    post_processor = TemplateProcessing(
        single="<cls> $A <eos>",
        pair="<cls> $A <eos> $B:1 <eos>:1",
        special_tokens=[("<cls>", 0), ("<eos>", 1)],
    )
    tokenizer = make_pretrained_tokenizer_fast(
        tokenizer_model=model,
        pre_tokenizer=pre_tokenizer,
        post_processor=post_processor,
        cls_token="<cls>",
        eos_token="<eos>",
        unk_token="<unk>",
    )

    assert isinstance(tokenizer, PreTrainedTokenizerFast)
    assert isinstance(tokenizer._tokenizer.pre_tokenizer, Whitespace)

    assert tokenizer.cls_token == "<cls>"
    assert tokenizer.eos_token == "<eos>"
    assert tokenizer.unk_token == "<unk>"

    string = "Hello world"

    assert tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(string) == [("Hello", (0, 5)), ("world", (6, 11))]
    assert tokenizer.encode(string) == [0, 3, 4, 1]


def test_saving_pretrained(tmp_path):
    make_pretrained_tokenizer_fast(tokenizer_model=BPE(), save_dirpath=str(tmp_path))

    assert (tmp_path / "tokenizer.json").exists()
