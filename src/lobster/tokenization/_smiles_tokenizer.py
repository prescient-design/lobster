import importlib.resources
from pathlib import Path

from tokenizers import Regex
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

from ._load_vocab_file import load_vocab_file
from ._make_pretrained_tokenizer_fast import make_pretrained_tokenizer_fast

SMILES_REGEX_PATTERN = (
    r"""(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"""
)

PRETRAINED_TOKENIZER_PATH = importlib.resources.files("lobster") / "assets" / "smiles_tokenizer"
VOCAB_PATH = PRETRAINED_TOKENIZER_PATH / "vocab.txt"


def _make_smiles_tokenizer(save_dirpath: str | Path | None = PRETRAINED_TOKENIZER_PATH) -> PreTrainedTokenizerFast:
    """Create PreTrainedTokenizerFast for SMILES Regex tokenization.

    Usage:
    ```python
    tokenizer = _make_ume_tokenizer(save_dirpath=PRETRAINED_TOKENIZER_PATH)

    fast_tokenizer = PreTrainedTokenizerFast.from_pretrained(PRETRAINED_TOKENIZER_PATH)
    """

    cls_token = "<cls>"
    eos_token = "<eos>"
    unk_token = "<unk>"
    pad_token = "<pad>"
    sep_token = "<sep>"
    mask_token = "<mask>"

    vocab = load_vocab_file(VOCAB_PATH)
    vocab = {v: k for k, v in enumerate(vocab)}

    tokenizer_model = WordLevel(vocab=vocab, unk_token=unk_token)

    pre_tokenizer = Split(pattern=Regex(SMILES_REGEX_PATTERN), behavior="isolated")

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
        pre_tokenizer=pre_tokenizer,
        post_processor=post_processor,
        cls_token=cls_token,
        eos_token=eos_token,
        unk_token=unk_token,
        sep_token=sep_token,
        mask_token=mask_token,
        pad_token=pad_token,
        bos_token=None,
    )


class SmilesTokenizerFast(PreTrainedTokenizerFast):
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
