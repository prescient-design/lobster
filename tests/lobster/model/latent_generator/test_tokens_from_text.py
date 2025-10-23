import pytest
import torch

from lobster.model.latent_generator.io import (
    LG_END_TOK,
    LG_START_TOK,
    LG_TOK_TEMPLATE,
    parse_tokens_from_text,
)


def make_tok_str(
    tokens: list[int],
) -> str:
    return "".join([LG_TOK_TEMPLATE % tok for tok in tokens])


class TestParseTokensFromText:
    """Test parsing LG token sequences from text."""

    def test_empty_text_returns_empty_list(self):
        """Empty text should return an empty list."""
        text = ""
        tokens = parse_tokens_from_text(text, struc_token_codebook_size=256)
        assert tokens == []

    def test_no_token_sequences(self):
        """Text without LG tokens should return an empty list."""
        text = "The protein is stable and well-folded."
        tokens = parse_tokens_from_text(text, struc_token_codebook_size=256)
        assert tokens == []

    def test_single_empty_lg_sequence(self):
        """An LG sequence with no tokens between START and END should return an empty tensor."""
        text = f"{LG_START_TOK}{LG_END_TOK}"
        tokens = parse_tokens_from_text(text, struc_token_codebook_size=256)
        assert len(tokens) == 1
        assert isinstance(tokens[0], torch.Tensor)
        assert tokens[0].numel() == 0

    def test_single_valid_sequence(self):
        """A valid LG sequence with tokens should be parsed correctly."""
        expected_tokens = [3, 12, 5]

        tok_str = make_tok_str(expected_tokens)
        text = f"{LG_START_TOK}{tok_str}{LG_END_TOK}"
        print(text)

        tokens = parse_tokens_from_text(text, struc_token_codebook_size=256)
        print(tokens)
        assert len(tokens) == 1
        assert torch.equal(tokens[0], torch.tensor(expected_tokens))

    def test_multiple_sequences_in_text(self):
        """Multiple LG sequences in one text should each be parsed separately."""
        expected_tokens = [[1, 2], [9, 10]]
        tok_str1 = make_tok_str(expected_tokens[0])
        tok_str2 = make_tok_str(expected_tokens[1])

        text = f"prefix {LG_START_TOK}{tok_str1}{LG_END_TOK} middle text {LG_START_TOK}{tok_str2}{LG_END_TOK} suffix"
        tokens = parse_tokens_from_text(text, struc_token_codebook_size=256)
        assert len(tokens) == 2
        assert torch.equal(tokens[0], torch.tensor(expected_tokens[0]))
        assert torch.equal(tokens[1], torch.tensor(expected_tokens[1]))

    def test_token_index_out_of_bounds(self):
        """If a token index is >= codebook size, a ValueError should be raised."""
        text = f"{LG_START_TOK}{make_tok_str([300])}{LG_END_TOK}"
        with pytest.raises(ValueError, match="expected all tokens to be in"):
            parse_tokens_from_text(text, struc_token_codebook_size=256)

    def test_malformed_missing_end_token(self):
        """Malformed LG string missing END token should not match."""
        text = "<|LG_START|><|LG_1|><|LG_2|>"
        text = f"{LG_START_TOK}{make_tok_str([1, 2])}"
        with pytest.raises(ValueError, match="mismatched number of start tokens"):
            parse_tokens_from_text(text, struc_token_codebook_size=256)

    def test_multiple_adjacent_sequences(self):
        """Handles back-to-back LG sequences with no whitespace."""
        expected_tokens = [[3, 4], [7, 8]]
        tok_str1 = make_tok_str(expected_tokens[0])
        tok_str2 = make_tok_str(expected_tokens[1])

        text = f"{LG_START_TOK}{tok_str1}{LG_END_TOK}{LG_START_TOK}{tok_str2}{LG_END_TOK}"
        tokens = parse_tokens_from_text(text, struc_token_codebook_size=256)
        assert len(tokens) == 2
        assert torch.equal(tokens[0], torch.tensor(expected_tokens[0]))
        assert torch.equal(tokens[1], torch.tensor(expected_tokens[1]))
