import re
import torch

# Expected representation of LG tokens as text
LG_START_TOK = "<|LG_start|>"
LG_END_TOK = "<|LG_end|>"
LG_TOK_TEMPLATE = "<|LG_%d|>"

# Regex pattern for matching full LG token strings that look like:
#  <|LG_START|><|LG_72|><|LG_186|><|LG_END|>
start_pat = re.escape(LG_START_TOK)
end_pat = re.escape(LG_END_TOK)
mid_pat = re.escape(LG_TOK_TEMPLATE).replace(r"%d", r"\d+")
lg_pattern = re.compile(f"{start_pat}(?:{mid_pat})*{end_pat}")

# Regex pattern for extracting just numeric token index from LG token strings,
# e.g. <|LG_42|> -> 42
tok_pattern = re.compile(re.escape(LG_TOK_TEMPLATE).replace(r"%d", r"(\d+)"))


def parse_tokens_from_text(
    text: str,
    struc_token_codebook_size: int,
) -> list[torch.Tensor]:
    """Extract LG tokens from a string containing textified LG tokens.

    Args:
        - text (str): Arbitrary text string that contains any number of LG
        token strings structured as "{LG_START_TOK}{LG_TOK_TEMPLATE}}*{LG_END_TOK}",
        for example, text generated from a LLM trained to generate LG tokens.
        - struc_token_codebook_size (int): Expected LG token codebook size, used for
        error checking parsed tokens.
    Returns:
        List of torch.Tensor, each element is the integer indices of parsed tokens,
        one element per LG string found in the text.
    """
    num_starts = len(re.findall(start_pat, text))
    num_ends = len(re.findall(end_pat, text))
    if num_starts != num_ends:
        raise ValueError(f"mismatched number of start tokens ({num_starts}) and end tokens ({num_ends}): {text}")

    lg_strs = lg_pattern.findall(text)

    parsed_tokens = []
    for lg_str in lg_strs:
        token_idxs = map(int, tok_pattern.findall(lg_str))
        token_tensor = torch.tensor(list(token_idxs))
        if (token_tensor >= struc_token_codebook_size).any():
            raise ValueError(f"expected all tokens to be in [0, {struc_token_codebook_size}), got: {token_tensor}")
        parsed_tokens.append(token_tensor)
    return parsed_tokens
