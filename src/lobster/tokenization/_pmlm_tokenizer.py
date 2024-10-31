"""Adapted from https://github.com/huggingface/transformers/tree/v4.23.1/src/transformers/models"""

import importlib.resources
import os
from itertools import islice
from typing import List, Optional, Union

from datasets import load_dataset
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
from transformers.tokenization_utils import PreTrainedTokenizer, Trie
from transformers.tokenization_utils_base import AddedToken
from transformers.utils import logging

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

VOCAB_PATH = importlib.resources.files("lobster") / "assets" / "pmlm_tokenizer" / "vocab.txt"

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "facebook/esm2_t6_8M_UR50D": "https://huggingface.co/facebook/esm2_t6_8M_UR50D/resolve/main/vocab.txt",
        "facebook/esm2_t12_35M_UR50D": "https://huggingface.co/facebook/esm2_t12_35M_UR50D/resolve/main/vocab.txt",
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "facebook/esm2_t6_8M_UR50D": 1024,
    "facebook/esm2_t12_35M_UR50D": 1024,
}


def load_vocab_file(vocab_file):
    with open(vocab_file, "r") as f:
        lines = f.read().splitlines()
    return [ll.strip() for ll in lines]


class PmlmTokenizer(PreTrainedTokenizer):
    """
    Constructs a Pmlm tokenizer.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file=VOCAB_PATH,
        unk_token="<unk>",
        cls_token="<cls>",
        pad_token="<pad>",
        mask_token="<mask>",
        eos_token="<eos>",
        **kwargs,
    ):
        self.all_tokens = load_vocab_file(vocab_file)
        self._id_to_token = dict(enumerate(self.all_tokens))
        super().__init__(**kwargs)

        self._token_to_id = {tok: ind for ind, tok in enumerate(self.all_tokens)}
        self.unk_token = unk_token
        self.cls_token = cls_token
        self.pad_token = pad_token
        self.mask_token = mask_token
        self.eos_token = eos_token
        self.unique_no_split_tokens = self.all_tokens
        self._create_trie(self.unique_no_split_tokens)

    def _convert_id_to_token(self, index: int) -> str:
        return self._id_to_token.get(index, self.unk_token)

    def _convert_token_to_id(self, token: str) -> int:
        return self._token_to_id.get(token, self._token_to_id.get(self.unk_token))

    def _tokenize(self, text, **kwargs):
        return text.split()

    def get_vocab_size(self, with_added_tokens=False):
        return len(self._id_to_token)

    def get_vocab(self):
        return {token: i for i, token in enumerate(self.all_tokens)}

    def token_to_id(self, token: str) -> int:
        return self._token_to_id.get(token, self._token_to_id.get(self.unk_token))

    def id_to_token(self, index: int) -> str:
        return self._id_to_token.get(index, self.unk_token)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        cls = [self.cls_token_id]
        sep = [self.eos_token_id]  # No sep token in ESM vocabulary
        if token_ids_1 is None:
            if self.eos_token_id is None:
                return cls + token_ids_0
            else:
                return cls + token_ids_0 + sep
        elif self.eos_token_id is None:
            raise ValueError("Cannot tokenize multiple sequences when EOS token is not set!")
        return cls + token_ids_0 + sep + token_ids_1 + sep  # Multiple inputs always have an EOS token

    def get_special_tokens_mask(
        self,
        token_ids_0: List,
        token_ids_1: Optional[List] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` or `encode_plus` methods.

        Args:
        ----
            token_ids_0 (`List[int]`):
                List of ids of the first sequence.
            token_ids_1 (`List[int]`, *optional*):
                List of ids of the second sequence.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
        -------
            A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.

        """
        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formatted with special tokens for the model."
                )

            return [1 if token in self.all_special_ids else 0 for token in token_ids_0]
        mask = [1] + ([0] * len(token_ids_0)) + [1]
        if token_ids_1 is not None:
            mask += [0] * len(token_ids_1) + [1]
        return mask

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

    def _create_trie(self, unique_no_split_tokens):
        trie = Trie()
        for token in unique_no_split_tokens:
            if hasattr(self, "do_lower_case") and self.do_lower_case and token not in self.all_special_tokens:
                trie.add(token.lower())
            else:
                trie.add(token)
        self.tokens_trie = trie


class TrainablePmlmTokenizer(PmlmTokenizer):
    def __init__(self, **kwargs):
        self._tokenizer, self._trainer = self._build_tokenizer(**kwargs)
        os.environ["TOKENIZERS_PARALLELISM"] = "true"

    def _build_tokenizer(self, **kwargs):
        pad_token = kwargs.get("pad_token", "<pad>")
        unk_token = kwargs.get("unk_token", "<unk>")
        max_vocab_size = kwargs.get("max_vocab_size", 1280)

        tokenizer = Tokenizer(models.BPE(unk_token=unk_token))
        # tokenizer.normalizer = normalizers.BertNormalizer()
        tokenizer.normalizer = normalizers.NFKC()
        # tokenizer.pre_tokenizer = Sequence([Digits(), Punctuation(), WhitespaceSplit()])
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=max_vocab_size,
            initial_alphabet=[
                "A",
                "R",
                "N",
                "D",
                "C",
                "E",
                "Q",
                "G",
                "H",
                "I",
                "L",
                "K",
                "M",
                "F",
                "P",
                "S",
                "T",
                "W",
                "Y",
                "V",
                ".",
                "-",
            ],
            special_tokens=["<pad>", "<cls>", "<eos>", "<unk>", "<mask>"],
        )

        tokenizer.special_tokens_map = {"pad_token": pad_token, "unk_token": unk_token}
        return tokenizer, trainer

    @staticmethod
    def _batch_iterator(hf_dataset, batch_size, text_column):
        for i in range(0, len(hf_dataset), batch_size):
            yield hf_dataset[i : i + batch_size][text_column]

    @staticmethod
    def _batch_txt_to_hf_iterator(txt_file, batch_size, text_column="text"):
        hf_dataset = load_dataset(text_column, data_files=[txt_file])
        for i in range(0, len(hf_dataset["train"]), batch_size):
            yield hf_dataset["train"][i : i + batch_size][text_column]

    @staticmethod
    def _batch_txt_iterator(txt_file, num_lines):
        with open(txt_file, "r") as f:
            return list(islice(f, num_lines))

    def fit(self, txt_file, num_lines=100):
        self._tokenizer.train_from_iterator(
            self._batch_txt_iterator(txt_file, num_lines),
            trainer=self._trainer,
            # length=len(hf_dataset),
        )
        super().__init__(tokenizer_object=self._tokenizer)
        # setattr(self, "model_input_names", ["input_ids"])
        self.model_input_names = ["input_ids"]
        for k, v in self._tokenizer.special_tokens_map.items():
            setattr(self, k, v)


# class PMLMTokenizer(BertTokenizer):
#     """
#     Constructs a PMLM tokenizer.
#     """

#     def __init__(
#         self,
#         vocab_file: str = None,
#         tokenizer_file: str = None,
#         do_lower_case: bool = False,
#         unk_token: str = "<unk>",
#         sep_token: str = "<sep>",
#         cls_token: str = "<cls>",
#         pad_token: str = "<pad>",
#         mask_token: str = "<mask>",
#         eos_token: str = "<eos>",
#         **kwargs,
#     ):
#         super().__init__(
#             vocab_file=vocab_file,
#             tokenizer_file=tokenizer_file,
#             do_lower_case=do_lower_case,
#             unk_token=unk_token,
#             sep_token=sep_token,
#             pad_token=pad_token,
#             cls_token=cls_token,
#             mask_token=mask_token,
#             eos_token=eos_token,
#             **kwargs,
#         )

#         self.all_tokens = load_vocab_file(vocab_file)
#         self._id_to_token = dict(enumerate(self.all_tokens))
#         self._token_to_id = {tok: ind for ind, tok in enumerate(self.all_tokens)}
#         self.unique_no_split_tokens = self.all_tokens
#         self._create_trie(self.unique_no_split_tokens)

#     def tokenize(self, text, **kwargs) -> List[str]:
#         """
#         Converts a string in a sequence of tokens, using the tokenizer.

#         Split in words for word-based vocabulary or sub-words for sub-word-based vocabularies
#         (BPE/SentencePieces/WordPieces). Takes care of added tokens.

#         Args:
#             text (`str`):
#                 The sequence to be encoded.
#             **kwargs (additional keyword arguments):
#                 Passed along to the model-specific `prepare_for_tokenization` preprocessing method.

#         Returns:
#             `List[str]`: The list of tokens.
#         """
#         # Simple mapping string => AddedToken for special tokens with specific tokenization behaviors
#         all_special_tokens_extended = {
#             str(t): t for t in self.all_special_tokens_extended if isinstance(t, AddedToken)
#         }

#         text, kwargs = self.prepare_for_tokenization(text, **kwargs)

#         if kwargs:
#             logger.warning(f"Keyword arguments {kwargs} not recognized.")

#         if hasattr(self, "do_lower_case") and self.do_lower_case:
#             # convert non-special tokens to lowercase
#             escaped_special_toks = [
#                 re.escape(s_tok)
#                 for s_tok in (self.unique_no_split_tokens + self.all_special_tokens)
#             ]
#             pattern = r"(" + r"|".join(escaped_special_toks) + r")|" + r"(.+?)"
#             text = re.sub(pattern, lambda m: m.groups()[0] or m.groups()[1].lower(), text)

#         no_split_token = set(self.unique_no_split_tokens)
#         tokens = self.tokens_trie.split(text)
#         # ["This is something", "<special_token_1>", "  else"]
#         for i, token in enumerate(tokens):
#             if token in no_split_token:
#                 tok_extended = all_special_tokens_extended.get(token, None)
#                 left = tokens[i - 1] if i > 0 else None
#                 right = tokens[i + 1] if i < len(tokens) - 1 else None
#                 if isinstance(tok_extended, AddedToken):
#                     if tok_extended.rstrip and right:
#                         # A bit counter-intuitive but we strip the left of the string
#                         # since tok_extended.rstrip means the special token is eating all white spaces on its right
#                         tokens[i + 1] = right.lstrip()
#                     # Strip white spaces on the left
#                     if tok_extended.lstrip and left:
#                         tokens[i - 1] = left.rstrip()  # Opposite here
#                 else:
#                     # We strip left and right by default
#                     if right:
#                         tokens[i + 1] = right.lstrip()
#                     if left:
#                         tokens[i - 1] = left.rstrip()
#         # ["This is something", "<special_token_1>", "else"]
#         tokenized_text = []
#         for token in tokens:
#             # Need to skip eventual empty (fully stripped) tokens
#             if not token:
#                 continue
#             if token in no_split_token:
#                 tokenized_text.append(token)
#             else:
#                 tokenized_text.extend(self._tokenize(token))
#         # ["This", " is", " something", "<special_token_1>", "else"]
#         return tokenized_text

#     def _tokenize(self, text, **kwargs):
#         return text.split()

#     def prepare_for_tokenization(
#         self, text: str, is_split_into_words: bool = False, **kwargs
#     ) -> Tuple[str, Dict[str, Any]]:
#         """
#         Performs any necessary transformations before tokenization.

#         This method should pop the arguments from kwargs and return the remaining `kwargs` as well. We test the
#         `kwargs` at the end of the encoding process to be sure all the arguments have been used.

#         Args:
#             text (`str`):
#                 The text to prepare.
#             is_split_into_words (`bool`, *optional*, defaults to `False`):
#                 Whether or not the input is already pre-tokenized (e.g., split into words). If set to `True`, the
#                 tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace)
#                 which it will tokenize. This is useful for NER or token classification.
#             kwargs:
#                 Keyword arguments to use for the tokenization.

#         Returns:
#             `Tuple[str, Dict[str, Any]]`: The prepared text and the unused kwargs.
#         """
#         return (text, kwargs)

#     def _convert_id_to_token(self, index: int) -> str:
#         return self._id_to_token.get(index, self.unk_token)

#     def _convert_token_to_id(self, token: str) -> int:
#         return self._token_to_id.get(token, self._token_to_id.get(self.unk_token))

#     def _tokenize(self, text, **kwargs):
#         return text.split()

#     def get_vocab_size(self, with_added_tokens=False):
#         return len(self._id_to_token)

#     def get_vocab(self):
#         return {token: i for i, token in enumerate(self.all_tokens)}

#     def token_to_id(self, token: str) -> int:
#         return self._token_to_id.get(token, self._token_to_id.get(self.unk_token))

#     def id_to_token(self, index: int) -> str:
#         return self._id_to_token.get(index, self.unk_token)

#     def build_inputs_with_special_tokens(
#         self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
#     ) -> List[int]:
#         cls = [self.cls_token_id]
#         sep = [self.eos_token_id]  # No sep token in pmlm vocabulary
#         if token_ids_1 is None:
#             if self.eos_token_id is None:
#                 return cls + token_ids_0
#             else:
#                 return cls + token_ids_0 + sep
#         elif self.eos_token_id is None:
#             raise ValueError("Cannot tokenize multiple sequences when EOS token is not set!")
#         return (
#             cls + token_ids_0 + sep + token_ids_1 + sep
#         )  # Multiple inputs always have an EOS token

#     def get_special_tokens_mask(
#         self,
#         token_ids_0: List,
#         token_ids_1: Optional[List] = None,
#         already_has_special_tokens: bool = False,
#     ) -> List[int]:
#         """
#         Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
#         special tokens using the tokenizer `prepare_for_model` or `encode_plus` methods.

#         Args:
#             token_ids_0 (`List[int]`):
#                 List of ids of the first sequence.
#             token_ids_1 (`List[int]`, *optional*):
#                 List of ids of the second sequence.
#             already_has_special_tokens (`bool`, *optional*, defaults to `False`):
#                 Whether or not the token list is already formatted with special tokens for the model.

#         Returns:
#             A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
#         """
#         if already_has_special_tokens:
#             if token_ids_1 is not None:
#                 raise ValueError(
#                     "You should not supply a second sequence if the provided sequence of "
#                     "ids is already formatted with special tokens for the model."
#                 )

#             return [1 if token in self.all_special_ids else 0 for token in token_ids_0]
#         mask = [1] + ([0] * len(token_ids_0)) + [1]
#         if token_ids_1 is not None:
#             mask += [0] * len(token_ids_1) + [1]
#         return mask

#     def save_vocabulary(self, save_directory, filename_prefix):
#         vocab_file = os.path.join(
#             save_directory, (filename_prefix + "-" if filename_prefix else "") + "vocab.txt"
#         )
#         with open(vocab_file, "w") as f:
#             f.write("\n".join(self.all_tokens))
#         return (vocab_file,)

#     @property
#     def vocab_size(self) -> int:
#         return self.get_vocab_size(with_added_tokens=False)

#     def _add_tokens(
#         self, new_tokens: Union[List[str], List[AddedToken]], special_tokens: bool = False
#     ) -> int:
#         return super()._add_tokens(new_tokens, special_tokens=True)
