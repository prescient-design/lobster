import importlib.resources

from tokenizers.models import BPE
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

from ._make_pretrained_tokenizer_fast import make_pretrained_tokenizer_fast

AA_VOCAB = {
    "<cls>": 0,
    "<pad>": 1,
    "<eos>": 2,
    "<unk>": 3,
    "L": 4,
    "A": 5,
    "G": 6,
    "V": 7,
    "S": 8,
    "E": 9,
    "R": 10,
    "T": 11,
    "I": 12,
    "D": 13,
    "P": 14,
    "K": 15,
    "Q": 16,
    "N": 17,
    "F": 18,
    "Y": 19,
    "M": 20,
    "H": 21,
    "W": 22,
    "C": 23,
    "X": 24,
    "B": 25,
    "U": 26,
    "Z": 27,
    "O": 28,
    ".": 29,
    "-": 30,
    "<null_1>": 31,
    "<mask>": 32,
}

PRETRAINED_TOKENIZER_PATH = importlib.resources.files("lobster") / "assets" / "amino_acid_tokenizer"


def _make_amino_acid_tokenizer() -> PreTrainedTokenizerFast:
    """Create a `PreTrainedTokenizerFast` object for tokenization of protein sequences.

    To create the tokenizer config stored under lobster/assets/amino_acid_tokenizer we run

    ```
    tokenizer = _make_amino_acid_tokenizer()
    tokenizer.save_pretrained("src/lobster/assets/amino_acid_tokenizer")
    ```

    This can now be loaded using
    `PreTrainedTokenizerFast.from_pretrained("src/lobster/assets/amino_acid_tokenizer")`
    """

    # BPE with no merges => just use input vocab
    tokenizer_model = BPE(AA_VOCAB, merges=[], unk_token="<unk>", ignore_merges=True)

    # bert style post processing
    post_processor = TemplateProcessing(
        single="<cls> $A <eos>",
        pair="<cls> $A <eos> $B:1 <eos>:1",
        special_tokens=[("<cls>", 0), ("<eos>", 2)],  # NOTE must match ids from AA_VOCAB
    )

    return make_pretrained_tokenizer_fast(
        tokenizer_model=tokenizer_model,
        post_processor=post_processor,
        eos_token="<eos>",
        unk_token="<unk>",
        pad_token="<pad>",
        cls_token="<cls>",
        mask_token="<mask>",
    )


class AminoAcidTokenizerFast(PreTrainedTokenizerFast):
    padding_side = "right"
    truncation_side = "right"
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self):
        super().__init__(
            tokenizer_file=str(PRETRAINED_TOKENIZER_PATH / "tokenizer.json"),
            bos_token=None,
            eos_token="<eos>",
            unk_token="<unk>",
            sep_token=None,
            pad_token="<pad>",
            cls_token="<cls>",
            mask_token="<mask>",
        )
