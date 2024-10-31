import torch
from cachetools import LRUCache, cached
from torch import LongTensor, Tensor
from transformers import BertTokenizerFast


class CachedBertTokenizerFast(BertTokenizerFast):
    """
    This class is a wrapper around the BertTokenizerFast class from the transformers library.
    It adds an additional cached encoding method for faster runtimes on smaller datasets.
    It also provides attributes indicating which tokens can be corrupted and sampled by denoising models,
    and a convenience method for getting a mask of corruptible tokens from a sequence of token ids.
    """

    def __init__(
        self,
        vocab_file: str = None,
        tokenizer_file: str = None,
        do_lower_case: bool = False,
        unk_token: str = "[UNK]",
        sep_token: str = "[SEP]",
        pad_token: str = "[PAD]",
        cls_token: str = "[CLS]",
        mask_token: str = "[MASK]",
        **kwargs,
    ):
        super().__init__(
            vocab_file=vocab_file,
            tokenizer_file=tokenizer_file,
            do_lower_case=do_lower_case,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs,
        )
        self.padding_idx = self.vocab[pad_token]
        self.masking_idx = self.vocab[mask_token]

        # prevent utility token input corruption
        utility_tokens = [
            unk_token,
            sep_token,
            pad_token,
            cls_token,
            mask_token,
        ]
        self.corruption_vocab_excluded = set(utility_tokens)
        self.sampling_vocab_excluded = set(utility_tokens)

    @property
    def corruption_vocab_included(self):
        return set(self.vocab.keys()) - self.corruption_vocab_excluded

    @property
    def sampling_vocab_included(self):
        return set(self.vocab.keys()) - self.sampling_vocab_excluded

    @cached(cache=LRUCache(maxsize=int(1e6)))
    def cached_encode(self, text: str):
        res = self.convert_tokens_to_ids(text.split(" "))
        return res

    def get_corruptible_mask(self, token_batch: LongTensor) -> Tensor:
        """
        Args:
        ----
            token_batch: a batch of token ids (LongTensor).

        Returns:
        -------
            a boolean mask tensor of corruptible tokens (corrupt if True).

        """
        excluded_idxs = (
            torch.tensor([self.vocab[tok] for tok in self.corruption_vocab_excluded]).view(-1, 1, 1).to(token_batch)
        )
        is_corruptible = token_batch.ne(excluded_idxs).prod(dim=0).bool()
        return is_corruptible
