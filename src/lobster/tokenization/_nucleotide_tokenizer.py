import importlib.resources

from tokenizers.models import BPE
from tokenizers.normalizers import Lowercase
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

from ._make_pretrained_tokenizer_fast import make_pretrained_tokenizer_fast

NT_VOCAB = {
    "<cls>": 0,
    "<pad>": 1,
    "<eos>": 2,
    "<unk>": 3,
    "<mask>": 4,
    "<sep>": 5,
    "a": 6,
    "c": 7,
    "g": 8,
    "t": 9,
    "n": 10,
    "u": 11,
}

PRETRAINED_TOKENIZER_PATH = importlib.resources.files("lobster") / "assets" / "nucleotide_tokenizer"


def _make_nucleotide_tokenizer(save_dirpath: str | None = PRETRAINED_TOKENIZER_PATH) -> PreTrainedTokenizerFast:
    tokenizer_model = BPE(NT_VOCAB, merges=[], unk_token="<unk>", ignore_merges=True)

    cls_token = "<cls>"
    pad_token = "<pad>"
    eos_token = "<eos>"
    unk_token = "<unk>"
    mask_token = "<mask>"
    sep_token = "<sep>"

    post_processor = TemplateProcessing(
        single=f"{cls_token} $A {eos_token}",
        pair=f"{cls_token} $A {eos_token} $B:1 {eos_token}:1",
        special_tokens=[
            (cls_token, 0),
            (eos_token, 2),
        ],
    )

    return make_pretrained_tokenizer_fast(
        tokenizer_model=tokenizer_model,
        save_dirpath=save_dirpath,
        post_processor=post_processor,
        normalizer=Lowercase(),
        cls_token=cls_token,
        pad_token=pad_token,
        eos_token=eos_token,
        unk_token=unk_token,
        mask_token=mask_token,
        sep_token=sep_token,
    )


class NucleotideTokenizerFast(PreTrainedTokenizerFast):
    padding_side = "right"
    truncation_side = "right"
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self):
        super().__init__(
            tokenizer_file=str(PRETRAINED_TOKENIZER_PATH / "tokenizer.json"),
            bos_token=None,
            eos_token="<eos>",
            unk_token="<unk>",
            sep_token="<sep>",
            pad_token="<pad>",
            cls_token="<cls>",
            mask_token="<mask>",
        )
