"""Adapted from https://github.com/huggingface/transformers/tree/v4.23.1/src/transformers/models"""

import importlib.resources
import os
from typing import List, Optional, Union

from transformers.tokenization_utils import PreTrainedTokenizer, Trie
from transformers.tokenization_utils_base import AddedToken
from transformers.utils import logging

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

VOCAB_PATH = importlib.resources.files("lobster") / "assets" / "mgm_tokenizer" / "vocab.txt"


def load_vocab_file(vocab_file):
    with open(vocab_file, "r") as f:
        lines = f.read().splitlines()
    return [ll.strip() for ll in lines]


class MgmTokenizer(PreTrainedTokenizer):
    """
    Constructs a MGM Mlm tokenizer.
    """

    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        model_max_length: int,
        vocab_file=VOCAB_PATH,
        bos_token="<bos>",
        sep_token="<sep>",
        unk_token="<unk>",
        cls_token="<cls>",
        cls_aa_token="<cls_aa>",
        cls_nt_token="<cls_nt>",
        cls_sf_token="<cls_sf>",
        cls_struct_token="<cls_struct>",
        pad_token="<pad>",
        mask_token="<mask>",
        eos_token="<eos>",
        **kwargs,
    ):
        self._model_max_length = model_max_length
        self.all_tokens = load_vocab_file(vocab_file)
        self._id_to_token = dict(enumerate(self.all_tokens))
        super().__init__(**kwargs)

        self._token_to_id = {tok: ind for ind, tok in enumerate(self.all_tokens)}

        self.cls_token = cls_token
        self.cls_token_id = self._token_to_id[cls_token]
        self.cls_aa_token = cls_aa_token
        self.cls_aa_token_id = self._token_to_id[cls_aa_token]
        self.cls_nt_token = cls_nt_token
        self.cls_nt_token_id = self._token_to_id[cls_nt_token]
        self.cls_sf_token = cls_sf_token
        self.cls_sf_token_id = self._token_to_id[cls_sf_token]
        self.cls_struct_token = cls_struct_token
        self.cls_struct_token_id = self._token_to_id[cls_struct_token]

        self.bos_token = bos_token
        self.bos_token_id = self._token_to_id[bos_token]
        self.eos_token = eos_token
        self.eos_token_id = self._token_to_id[eos_token]
        self.mask_token = mask_token
        self.mask_token_id = self._token_to_id[mask_token]
        self.pad_token = pad_token
        self.pad_token_id = self._token_to_id[pad_token]
        self.sep_token = sep_token
        self.sep_token_id = self._token_to_id[sep_token]

        self.unk_token = unk_token
        self.unk_token_id = self._token_to_id[unk_token]

        self.unique_no_split_tokens = self.all_tokens
        self._create_trie(self.unique_no_split_tokens)

    def _convert_id_to_token(self, index: int) -> str:
        return self._id_to_token.get(index, self.unk_token)

    def _convert_token_to_id(self, token: str) -> int:
        return self._token_to_id.get(token, self._token_to_id.get(self.unk_token))

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)

    def _create_trie(self, unique_no_split_tokens):
        trie = Trie()
        for token in unique_no_split_tokens:
            if hasattr(self, "do_lower_case") and self.do_lower_case and token not in self.all_special_tokens:
                trie.add(token.lower())
            else:
                trie.add(token)
        self.tokens_trie = trie

    def _tokenize(self, text: str) -> List[str]:
        return text.split()

    def get_vocab_size(self, with_added_tokens=False):
        return len(self._id_to_token)

    def get_vocab(self):
        return {token: i for i, token in enumerate(self.all_tokens)}

    def token_to_id(self, token: str) -> int:
        return self._token_to_id.get(token, self._token_to_id.get(self.unk_token))

    def id_to_token(self, index: int) -> str:
        return self._id_to_token.get(index, self.unk_token)

    """
    def build_inputs_with_special_tokens(self, token_ids: List[int]) -> List[int]:
        '''
        A MGM sequence has the same format as a BERT sequence: `[CLS] X [SEP]`.
        '''
        cls = [self.cls_token_id]
        eos = [self.eos_token_id]
        return cls + token_ids + eos
    """

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        eos = [self.eos_token_id]

        if token_ids_1 is None:
            if self.eos_token_id is None:
                return cls + token_ids_0
            else:
                return cls + token_ids_0 + eos
        elif self.eos_token_id is None:
            raise ValueError("Cannot tokenize multiple sequences when EOS token is not set!")
        return cls + token_ids_0 + sep + token_ids_1 + eos  # Multiple inputs always have an EOS token

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True,
            )

        result = ([0] * len(token_ids_0)) + [1]
        if token_ids_1 is not None:
            result += ([0] * len(token_ids_1)) + [1]
        return result

    def save_vocabulary(self, save_directory, filename_prefix):
        vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + "vocab.txt",
        )
        with open(vocab_file, "w") as f:
            f.write("\n".join(self.all_tokens))
        return (vocab_file,)

    @property
    def vocab_size(self) -> int:
        return self.get_vocab_size(with_added_tokens=False)

    def _add_tokens(
        self,
        new_tokens: Union[List[str], List[AddedToken]],
        special_tokens: bool = False,
    ) -> int:
        return super()._add_tokens(new_tokens, special_tokens=True)
