import importlib.resources
from typing import Optional

from tokenizers import Regex, Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Split
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

from ._load_vocab_file import load_vocab_file

SMILES_REGEX_PATTERN = (
    r"""(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"""
)

PRETRAINED_TOKENIZER_PATH = importlib.resources.files("lobster") / "assets" / "smiles_tokenizer"
VOCAB_PATH = PRETRAINED_TOKENIZER_PATH / "vocab.txt"


def _make_smiles_tokenizer(
    cls_token: str = "<cls>",
    eos_token: str = "<eos>",
    unk_token: str = "<unk>",
    pad_token: str = "<pad>",
    sep_token: str = "<sep>",
    mask_token: str = "<mask>",
    vocab_file: Optional[str] = None,
) -> PreTrainedTokenizerFast:
    """Create PreTrainedTokenizerFast for SMILES Regex tokenization."""

    vocab = load_vocab_file(VOCAB_PATH if vocab_file is None else vocab_file)
    vocab = {v: k for k, v in enumerate(vocab)}

    tok = Tokenizer(BPE(vocab, merges=[], unk_token="<unk>", ignore_merges=True))

    tok.pre_tokenizer = Split(pattern=Regex(SMILES_REGEX_PATTERN), behavior="isolated")

    tok.post_processor = TemplateProcessing(
        single=f"{cls_token} $A {eos_token}",
        pair=f"{cls_token} $A {eos_token} $B:1 {eos_token}:1",
        special_tokens=[
            (pad_token, 0),
            (unk_token, 1),
            (cls_token, 2),
            (sep_token, 3),
            (mask_token, 4),
            (eos_token, 5),
        ],
    )

    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        padding_side="right",
        truncation_side="left",
        bos_token=None,
        sep_token=sep_token,
        eos_token=eos_token,
        pad_token=pad_token,
        unk_token=unk_token,
        cls_token=cls_token,
        mask_token=mask_token,
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
