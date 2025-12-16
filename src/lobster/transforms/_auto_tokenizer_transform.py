from os import PathLike
from typing import Any

from transformers import AutoTokenizer
from transformers.tokenization_utils_base import (
    BatchEncoding,
    PaddingStrategy,
    TruncationStrategy,
)

from ._transform import Transform


class AutoTokenizerTransform(Transform):
    """Transform using HuggingFace's AutoTokenizer for any pretrained model.

    Uses HuggingFace's AutoTokenizer to automatically load the correct tokenizer
    for any supported model (ESM, ProtBERT, BERT, GPT, etc.).

    Parameters
    ----------
    pretrained_model_name_or_path : str | PathLike
        Name or path of pretrained model (e.g., 'facebook/esm2_t33_650M_UR50D',
        'Rostlab/prot_bert', 'bert-base-uncased')
    padding : bool | str | PaddingStrategy, default=False
        Padding strategy
    truncation : bool | str | TruncationStrategy, default=False
        Truncation strategy
    max_length : int | None, default=None
        Maximum sequence length
    return_token_type_ids : bool | None, default=None
        Whether to return token type IDs
    return_attention_mask : bool | None, default=None
        Whether to return attention mask
    return_overflowing_tokens : bool, default=False
        Whether to return overflowing tokens
    return_special_tokens_mask : bool, default=False
        Whether to return special tokens mask
    return_offsets_mapping : bool, default=False
        Whether to return offsets mapping
    return_length : bool, default=False
        Whether to return length
    verbose : bool, default=True
        Whether to print verbose output

    Examples
    --------
    >>> # ESM2 model
    >>> transform = AutoTokenizerTransform(
    ...     pretrained_model_name_or_path="facebook/esm2_t33_650M_UR50D",
    ...     max_length=512,
    ...     padding="max_length",
    ...     truncation=True,
    ... )
    >>> result = transform("ACDEFGHIKLMNPQRSTVWY")
    >>>
    >>> # ProtBERT model
    >>> transform = AutoTokenizerTransform(
    ...     pretrained_model_name_or_path="Rostlab/prot_bert",
    ...     max_length=512,
    ... )
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str | PathLike,
        padding: bool | str | PaddingStrategy = False,
        truncation: bool | str | TruncationStrategy = False,
        max_length: int | None = None,
        return_token_type_ids: bool | None = None,
        return_attention_mask: bool | None = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
    ):
        super().__init__()

        self._pretrained_model_name_or_path = pretrained_model_name_or_path
        self._padding = padding
        self._truncation = truncation
        self._max_length = max_length
        self._return_token_type_ids = return_token_type_ids
        self._return_attention_mask = return_attention_mask
        self._return_overflowing_tokens = return_overflowing_tokens
        self._return_special_tokens_mask = return_special_tokens_mask
        self._return_offsets_mapping = return_offsets_mapping
        self._return_length = return_length
        self._verbose = verbose

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._pretrained_model_name_or_path,
        )

    def _transform(
        self,
        text: str | list[str] | list[int],
        parameters: dict[str, Any],
    ) -> BatchEncoding:
        """Tokenize input text.

        Parameters
        ----------
        text : str | list[str] | list[int]
            Input text or sequence to tokenize
        parameters : dict[str, Any]
            Additional parameters (unused)

        Returns
        -------
        BatchEncoding
            Tokenized output with input_ids, attention_mask, etc.
        """
        tokenized = self._tokenizer(
            text,
            padding=self._padding,
            truncation=self._truncation,
            max_length=self._max_length,
            return_tensors="pt",
            return_token_type_ids=self._return_token_type_ids,
            return_attention_mask=self._return_attention_mask,
            return_overflowing_tokens=self._return_overflowing_tokens,
            return_special_tokens_mask=self._return_special_tokens_mask,
            return_offsets_mapping=self._return_offsets_mapping,
            return_length=self._return_length,
            verbose=self._verbose,
        )

        return tokenized

    def _check_inputs(self, inputs: list[Any]) -> None:
        """Check inputs - required by base Transform class."""
        pass

    @property
    def tokenizer(self):
        """Get the underlying tokenizer."""
        return self._tokenizer
