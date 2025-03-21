import importlib.resources
from pathlib import Path

from tokenizers import pre_tokenizers
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

from ._load_vocab_file import load_vocab_file
from ._make_pretrained_tokenizer_fast import make_pretrained_tokenizer_fast

PRETRAINED_TOKENIZER_PATH = importlib.resources.files("lobster") / "assets" / "latent_generator_tokenizer"
VOCAB_PATH = PRETRAINED_TOKENIZER_PATH / "vocab.txt"
vocab = load_vocab_file(VOCAB_PATH)
LG_VOCAB = {v: k for k, v in enumerate(vocab)}


def _make_latent_generator_3d_coord_tokenizer(
    save_dirpath: str | Path | None = PRETRAINED_TOKENIZER_PATH,
) -> PreTrainedTokenizerFast:
    """Create a `PreTrainedTokenizerFast` object for tokenization of protein structure 3d coordinate to tokens via Latent Generator.

    To create the tokenizer config stored under lobster/assets/latent_generator_tokenizer we run

    ```
    tokenizer = _make_latent_generator_3d_coord_tokenizer()
    tokenizer.save_pretrained("src/lobster/assets/latent_generator_tokenizer")
    ```

    This can now be loaded using
    `PreTrainedTokenizerFast.from_pretrained("src/lobster/assets/latent_generator_tokenizer")`
    """

    # WordLevel tokenizer
    tokenizer_model = WordLevel(LG_VOCAB, unk_token="<unk>")

    # pretokenizers
    pre_tokenizer = pre_tokenizers.Sequence([WhitespaceSplit()])

    # bert style post processing
    post_processor = TemplateProcessing(
        single="<cls> $A <eos>",
        pair="<cls> $A <eos> $B:1 <eos>:1",
        special_tokens=[("<cls>", 0), ("<eos>", 2)],  # NOTE must match ids from AA_VOCAB
    )

    return make_pretrained_tokenizer_fast(
        tokenizer_model=tokenizer_model,
        save_dirpath=save_dirpath,
        post_processor=post_processor,
        pre_tokenizer=pre_tokenizer,
        eos_token="<eos>",
        unk_token="<unk>",
        pad_token="<pad>",
        cls_token="<cls>",
        mask_token="<mask>",
    )


class LatentGenerator3DCoordTokenizerFast(PreTrainedTokenizerFast):
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
