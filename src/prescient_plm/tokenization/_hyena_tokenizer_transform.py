import importlib.resources
from os import PathLike
from typing import Any, List, Optional, Union

import torch
from prescient.transforms import Transform
from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, TruncationStrategy

from ._hyena_tokenizer import HyenaTokenizer

# TODO: update with real tables
DNA_CODON_DICT = {
    "A": ["GCT", "GCC", "GCA", "GCG"],
    "R": ["CGT", "CGC", "CGA", "CGG", "AGA", "AGG"],
    "N": ["AAT", "AAC"],
    "D": ["GAT", "GAC"],
    "C": ["TGT", "TGC"],
    "Q": ["CAA", "CAG"],
    "E": ["GAA", "GAG"],
    "G": ["GGT", "GGC", "GGA", "GGG"],
    "H": ["CAT", "CAC"],
    "I": ["ATT", "ATC", "ATA"],
    "L": ["TTA", "TTG", "CTT", "CTC", "CTA", "CTG"],
    "K": ["AAA", "AAG"],
    "M": ["ATG"],
    "F": ["TTT", "TTC"],
    "P": ["CCT", "CCC", "CCA", "CCG"],
    "S": ["TCT", "TCC", "TCA", "TCG", "AGT", "AGC"],
    "T": ["ACT", "ACC", "ACA", "ACG"],
    "W": ["TGG"],
    "Y": ["TAT", "TAC"],
    "V": ["GTT", "GTC", "GTA", "GTG"],
    "STOP": ["TAA", "TAG", "TGA"],
}


class HyenaTokenizerTransform(Transform):
    def __init__(
        self,
        pretrained_model_name_or_path: Union[str, PathLike] = None,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = False,
        max_length: Optional[int] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = False,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        tokenizer_dir: Optional[str] = "hyena_tokenizer",
        mlm: bool = False,
        aa_to_dna: bool = False,
    ):
        super().__init__()

        self._pretrained_model_name_or_path = pretrained_model_name_or_path
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
        self._tokenizer_dir = tokenizer_dir
        self._mlm = mlm
        self._aa_to_dna = aa_to_dna

        if self._pretrained_model_name_or_path is not None:
            self._auto_tokenizer = HyenaTokenizer.from_pretrained(
                self._pretrained_model_name_or_path,
                do_lower_case=False,
                add_special_tokens=True,
                padding_side="left",  # since HyenaDNA is causal, we pad on the left
                use_fast=True,
            )
        elif self._tokenizer_dir is not None:
            path = importlib.resources.files("prescient_plm") / "assets" / self._tokenizer_dir
            self._auto_tokenizer = HyenaTokenizer.from_pretrained(
                path,
                do_lower_case=False,
                add_special_tokens=True,
                padding_side="left",  # since HyenaDNA is causal, we pad on the left
                use_fast=True,
            )

    def transform(
        self,
        text: Union[str, List[str], List[int]],
        parameters: dict[str, Any],
    ) -> BatchEncoding:
        if self._aa_to_dna:
            text = [self.translate_aa_to_dna(seq) for seq in text]

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

        labels = tokenized["input_ids"].clone()
        if self._auto_tokenizer.pad_token_id is not None:
            labels[labels == self._auto_tokenizer.pad_token_id] = -100  # ignore in loss
        tokenized["labels"] = labels

        return tokenized

    def validate(self, flat_inputs: list[Any]) -> None:
        pass

    def translate_aa_to_dna(self, aa_sequence: str) -> str:
        # TODO: update DNA frequencies
        dna_sequence = "".join(
            [
                DNA_CODON_DICT[aa][torch.randint(0, len(DNA_CODON_DICT[aa]), (1,)).item()]
                for aa in aa_sequence
            ]
        )
        return dna_sequence
