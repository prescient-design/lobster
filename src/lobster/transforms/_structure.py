import torch


def trim_or_pad(tensor: torch.Tensor, pad_to: int, pad_idx: int = 0):
    """Trim or pad a tensor with shape (L, ...) to a given length."""
    L = tensor.shape[0]
    if L >= pad_to:
        # trim, assuming first dimension is the dim to trim
        tensor = tensor[:pad_to]
    elif L < pad_to:
        padding = torch.full(
            size=(pad_to - tensor.shape[0], *tensor.shape[1:]),
            fill_value=pad_idx,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        tensor = torch.concat((tensor, padding), dim=0)
    return tensor
