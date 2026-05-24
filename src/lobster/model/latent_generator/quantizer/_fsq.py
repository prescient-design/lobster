"""
Torch implementation of Finite Scalar Quantization
https://arxiv.org/abs/2309.15505, Appendix 1
"""

import torch


def round_ste(z):
    """Round with straight through gradients."""
    zhat = torch.round(z)
    return z + (zhat - z).detach()


class FiniteScalarQuantizer(torch.nn.Module):
    def __init__(self, levels: list[int], return_oh_like: bool = True):
        super().__init__()

        levels = torch.tensor(levels)
        basis = torch.cat([torch.tensor([1]), torch.cumprod(levels[:-1], dim=0)]).to(dtype=torch.int32)
        self.levels = levels
        self.basis = basis
        self.return_oh_like = return_oh_like
        # number of dimensions expect from inputs
        self.num_dimensions = len(levels)

        # size of the codebook
        self.codebook_size = torch.prod(levels)
        self.implicit_codebook = self.indexes_to_codes(torch.arange(self.codebook_size))
        self.n_tokens = self.codebook_size
        print("Codebook size:", self.codebook_size)

    @property
    def codebook(self):
        return self.implicit_codebook

    def bound(self, z, eps=1e-3):
        """Bound z, an array of shape (..., d)."""
        levels = self.levels.to(z.device)
        half_l = (levels - 1) * (1 - eps) / 2
        offset = torch.where(levels % 2 == 1, 0.0, 0.5)
        shift = torch.tan(offset / half_l)
        return torch.tanh(z + shift) * half_l - offset

    def _quantize(self, z, mask=None, **kwargs):
        """Quanitzes z, returns quantized zhat as codewords, same shape as z."""
        quantized = round_ste(self.bound(z))
        half_width = self.levels // 2  # Renormalize to [-1, 1].
        half_width = half_width.to(z.device)
        z_tokens = quantized / half_width
        return z_tokens, z, mask

    def _scale_and_shift(self, zhat_normalized):
        levels = self.levels.to(zhat_normalized.device)
        half_width = levels // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat):
        levels = self.levels.to(zhat.device)
        half_width = levels // 2
        return (zhat - half_width) / half_width

    def codes_to_indexes(self, zhat):
        # assert zhat.shape[-1] == len(self.levels)
        basis = self.basis.to(zhat.device)
        zhat = self._scale_and_shift(zhat)
        return (zhat * basis).sum(axis=-1)

    def indexes_to_codes(self, indices):
        indices = indices.unsqueeze(-1)

        # def _maybe_cast_shape(input_arr, target_arr)
        #     # both should have 2 dimensions
        #     # but user-specified indices might be batched
        #     if input_arr.shape != target_arr.shape:
        #         return input_arr.expand_as(target_arr)
        #     else:
        #         return input_arr

        # basis = _maybe_cast_shape(self.basis, indices)
        # levels = _maybe_cast_shape(self.levels, indices)
        basis = self.basis.to(indices.device)
        levels = self.levels.to(indices.device)
        codes_non_centered = torch.remainder(torch.floor_divide(indices, basis), levels)
        return self._scale_and_shift_inverse(codes_non_centered)

    def quantize(self, z, mask=None, **kwargs):
        z_tokens, z, mask = self._quantize(z, mask=mask, **kwargs)
        if self.return_oh_like:
            # Get continuous indexes (B, L)
            continuous_indexes = self.codes_to_indexes(z_tokens)

            # Get codebook entries (codebook_size, num_dimensions)
            codebook = self.implicit_codebook.to(z_tokens.device)

            # Compute similarity between z_tokens and each codebook entry
            # z_tokens: (B, L, num_dimensions)
            # codebook: (codebook_size, num_dimensions)
            # Result: (B, L, codebook_size)
            oh_like = torch.matmul(z_tokens, codebook.T)

            # Optionally: use straight-through with discrete one-hot for sharper distribution
            # Get discrete indexes for one-hot
            codebook_size_int = int(self.codebook_size.item())
            discrete_indexes = torch.round(continuous_indexes).long().clamp(0, codebook_size_int - 1)
            discrete_oh = torch.nn.functional.one_hot(discrete_indexes, num_classes=codebook_size_int).float()

            # Use straight-through: gradients flow through oh_like, forward uses discrete_oh
            oh_like = oh_like + (discrete_oh - oh_like).detach()

            return oh_like, z, mask
        else:
            return z_tokens, z, mask
