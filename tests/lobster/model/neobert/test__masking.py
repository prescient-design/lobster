import torch
import pytest
from lobster.model.neobert._masking import (
    _create_special_tokens_mask,
    _validate_input_shapes,
    _validate_probabilities,
    mask_tokens,
)


def test__create_special_tokens_mask():
    input_ids = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]])
    special_token_ids = [1, 4]

    mask = _create_special_tokens_mask(input_ids, special_token_ids)

    expected = torch.tensor([[True, False, False, True], [False, False, True, False]])
    assert torch.equal(mask, expected)


def test__validate_input_shapes():
    input_ids = torch.tensor([[1, 2, 3]])
    attention_mask = torch.tensor([[1, 1, 1]])
    _validate_input_shapes(input_ids, attention_mask)

    with pytest.raises(ValueError, match="input_ids must be a 2D tensor"):
        _validate_input_shapes(torch.tensor([1, 2, 3]), attention_mask)


def test__validate_probabilities():
    _validate_probabilities(0.5)

    with pytest.raises(ValueError, match="mask_probability must be between 0 and 1"):
        _validate_probabilities(1.5)


def test_mask_tokens():
    input_ids = torch.tensor(
        [[1, 2, 3, 4, 1, 2, 3]]
    )  # 4 should be masked since the rest are special tokens and probability is 1.0
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1]])
    mask_token_id = 999

    result = mask_tokens(
        input_ids=input_ids,
        attention_mask=attention_mask,
        mask_token_id=mask_token_id,
        mask_probability=1.0,
        special_token_ids=[1, 2, 3],
    )

    assert "input_ids" in result
    assert "labels" in result
    assert "attention_mask" in result

    assert result["input_ids"].shape == input_ids.shape
    assert result["labels"].shape == input_ids.shape
    assert result["attention_mask"].shape == attention_mask.shape

    assert result["input_ids"].tolist() == [[1, 2, 3, 999, 1, 2, 3]]
