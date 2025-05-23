from unittest.mock import Mock

import pytest
import torch
from transformers import PreTrainedTokenizerFast
from transformers.tokenization_utils_base import BatchEncoding

from lobster.transforms import TokenizerTransform


@pytest.fixture
def mock_tokenizer():
    tokenizer_mock = Mock(spec=PreTrainedTokenizerFast)
    tokenizer_mock.return_value = BatchEncoding(
        {"input_ids": torch.tensor([[1, 2, 3, 0, 0]]), "attention_mask": torch.tensor([[1, 1, 1, 0, 0]])}
    )
    return tokenizer_mock


def test_tokenizer_transform_initialization(mock_tokenizer):
    transform = TokenizerTransform(mock_tokenizer)

    assert transform.tokenizer == mock_tokenizer
    assert transform._padding is False
    assert transform._truncation is False
    assert transform._max_length is None


def test_tokenizer_transform_forward(mock_tokenizer):
    transform = TokenizerTransform(mock_tokenizer)

    text = "This is a test"
    result = transform(text)

    mock_tokenizer.assert_called_once_with(
        text,
        padding=False,
        truncation=False,
        max_length=None,
        return_tensors="pt",
        return_token_type_ids=None,
        return_attention_mask=None,
        return_overflowing_tokens=False,
        return_special_tokens_mask=False,
        return_offsets_mapping=False,
        return_length=False,
        add_special_tokens=True,
        verbose=True,
    )

    assert isinstance(result, BatchEncoding)
    assert "input_ids" in result
    assert "attention_mask" in result
    assert torch.equal(result["input_ids"], torch.tensor([[1, 2, 3, 0, 0]]))
    assert torch.equal(result["attention_mask"], torch.tensor([[1, 1, 1, 0, 0]]))
