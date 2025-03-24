from collections.abc import Iterator
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import Model
from tokenizers.normalizers import Normalizer
from tokenizers.pre_tokenizers import PreTokenizer
from tokenizers.processors import PostProcessor
from tokenizers.trainers import Trainer
from transformers import PreTrainedTokenizerFast


def make_pretrained_tokenizer_fast(
    *,
    tokenizer_model: Model,
    save_dirpath: str | Path | None = None,
    trainer: Trainer | None = None,
    data_iterator: Iterator | None = None,
    pre_tokenizer: PreTokenizer | None = None,
    post_processor: PostProcessor | None = None,
    normalizer: Normalizer | None = None,
    padding_side: str = "right",
    truncation_side: str = "left",
    **kwargs,
) -> PreTrainedTokenizerFast:
    """Create an instance of PreTrainedTokenizerFast from tokenizers."""

    tokenizer = Tokenizer(tokenizer_model)

    if pre_tokenizer is not None:
        tokenizer.pre_tokenizer = pre_tokenizer

    if normalizer is not None:
        tokenizer.normalizer = normalizer

    if post_processor is not None:
        tokenizer.post_processor = post_processor

    # Check that if either trainer or data_iterator is provided, both are provided
    if (trainer is not None and data_iterator is None) or (data_iterator is not None and trainer is None):
        raise ValueError("Both `trainer` and `data_iterator` must be provided to train the tokenizer.")

    if trainer is not None:
        tokenizer.train_from_iterator(data_iterator, trainer=trainer)

    tokenizer_fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, padding_side=padding_side, truncation_side=truncation_side, **kwargs
    )

    if save_dirpath:
        tokenizer_fast.save_pretrained(save_dirpath)

    return tokenizer_fast
