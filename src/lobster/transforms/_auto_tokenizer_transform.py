from os import PathLike
from typing import Any, Dict, List, Optional, Union

from transformers import AutoTokenizer
from transformers.tokenization_utils_base import (
    BatchEncoding,
    PaddingStrategy,
    TruncationStrategy,
)

from ._transform import Transform


class AutoTokenizerTransform(Transform):
    def __init__(
        self,
        pretrained_model_name_or_path: Union[str, PathLike],
        padding: Union[bool, str, "PaddingStrategy"] = False,
        truncation: Union[bool, str, "TruncationStrategy"] = False,
        max_length: Optional[int] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
    ):
        super().__init__()

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

        self._auto_tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            do_lower_case=False,
            use_fast=True,
        )

        # self._auto_tokenizer.pad_token = self._auto_tokenizer.eos_token

    def transform(
        self,
        text: Union[str, List[str], List[int]],
        parameters: dict[str, Any],
    ) -> "BatchEncoding":
        tokenized = self._auto_tokenizer(
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

        tokenized["labels"] = tokenized["input_ids"].clone()

        tokenized["labels"][:-1] = tokenized["input_ids"][1:]
        tokenized["labels"][-1] = -100

        return tokenized

    def _transform(self, input: Any, parameters: Dict[str, Any]) -> Any:
        return self.transform(input, parameters)

    def validate(self, flat_inputs: list[Any]) -> None:
        pass

    def _check_inputs(self, inputs: List[Any]) -> None:
        pass
