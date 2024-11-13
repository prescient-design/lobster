import importlib
from typing import Optional, Sequence, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence

from lobster.tokenization import PmlmTokenizer


class EsmBatchConverter(object):
    """Callable to convert an unprocessed (labels + strings) batch to a
    processed (labels + tensor) batch.
    """

    def __init__(
        self,
        model_name: str = "esm2_t6_8M_UR50D",
        truncation_seq_length: int = 1024,
        prepend_bos=True,  # same as esm2 alphabet
        append_eos=True,  # same as esm2 alphabet
        tokenizer_dir: Optional[str] = "pmlm_tokenizer",
    ):
        self.model_name = model_name
        self.truncation_seq_length = truncation_seq_length
        self.prepend_bos = prepend_bos
        self.append_eos = append_eos
        self._tokenizer_dir = tokenizer_dir

        # self.tokenizer = AutoTokenizer.from_pretrained(f"facebook/{model_name}") # TODO - why not this line?
        path = importlib.resources.files("lobster") / "assets" / self._tokenizer_dir
        self.tokenizer = PmlmTokenizer.from_pretrained(path, do_lower_case=False)

    def __call__(self, raw_batch: Sequence[Tuple[str, str]]):
        # RoBERTa uses an eos token, while ESM-1 does not.
        batch_size = len(raw_batch)
        batch_labels, seq_str_list = zip(*raw_batch)

        seq_encoded_list = [self.tokenizer.encode(seq_str) for seq_str in seq_str_list]  # adds cls and eos
        if self.truncation_seq_length:
            seq_encoded_list = [seq_str[: self.truncation_seq_length] for seq_str in seq_encoded_list]
        max_len = max(len(seq_encoded) for seq_encoded in seq_encoded_list)
        tokens = torch.empty(
            (batch_size, max_len),  # + int(self.prepend_bos) + int(self.append_eos),
            dtype=torch.int64,
        )
        tokens.fill_(self.tokenizer.pad_token_id)
        labels = []
        strs = []

        for i, (label, seq_str, seq_encoded) in enumerate(zip(batch_labels, seq_str_list, seq_encoded_list)):
            labels.append(label)
            strs.append(seq_str)
            # if self.prepend_bos:  # already added in tokenization
            #     tokens[i, 0] = self.tokenizer.cls_token_id
            seq = torch.tensor(seq_encoded, dtype=torch.int64)
            tokens[
                i,
                # int(self.prepend_bos) : len(seq_encoded) + int(self.prepend_bos),
                0 : len(seq_encoded),
            ] = seq
            # if self.append_eos: # already added in tokenization
            #     tokens[i, len(seq_encoded) + int(self.prepend_bos)] = self.tokenizer.eos_token_id

        return labels, strs, tokens


class ESMBatchConverterPPI(EsmBatchConverter):
    def __init__(
        self,
        model_name: str = "esm2_t6_8M_UR50D",
        truncation_seq_length: int = 1024,
        contact_maps: bool = False,
        prepend_bos=True,  # same as esm2 alphabet
        append_eos=True,  # same as esm2 alphabet
        tokenizer_dir: Optional[str] = "pmlm_tokenizer",
    ):
        super().__init__(
            model_name=model_name,
            truncation_seq_length=truncation_seq_length,
            prepend_bos=prepend_bos,
            append_eos=append_eos,
        )

        self._tokenizer_dir = tokenizer_dir

        path = importlib.resources.files("lobster") / "assets" / self._tokenizer_dir
        self.tokenizer = PmlmTokenizer.from_pretrained(path, do_lower_case=False)
        self._contact_maps = contact_maps

    def __call__(self, raw_batch: Sequence[Tuple[str, str]]):
        """Adapted for PPI Data"""
        # NOTE - Removed nulls that came from transforms, might need to write custom dataloader sampler if batch sizes too small
        if self._contact_maps:
            # Flatten the output of Atom3D transforms
            flattened_batch = [(a, b, c) for ((a, b), c) in raw_batch if (a is not None) and (b is not None)]
            batch_size = len(flattened_batch)
            if batch_size == 0:
                return None
            seq1_tokenized, seq2_tokenized, contact_map = zip(*flattened_batch)

        else:
            tokens, batch_labels = zip(*raw_batch)
            seq1_tokenized, seq2_tokenized = zip(*tokens)

        # print(f"seq1 tokenized: {seq1_tokenized}")

        if self.truncation_seq_length:
            # NOTE - This removes eos token for long sequences. Should we re-add eos or keep as is?
            seq1_tokenized = [seq[: self.truncation_seq_length] for seq in seq1_tokenized]
            seq2_tokenized = [seq[: self.truncation_seq_length] for seq in seq2_tokenized]

        tokens1 = pad_sequence(seq1_tokenized, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        tokens2 = pad_sequence(seq2_tokenized, batch_first=True, padding_value=self.tokenizer.pad_token_id)

        #  Get attention masks
        attention_mask_1 = (tokens1 != self.tokenizer.pad_token_id).to(torch.int)
        attention_mask_2 = (tokens2 != self.tokenizer.pad_token_id).to(torch.int)

        # NOTE - might be worth having an output class with this many outputs
        if self._contact_maps:
            return {
                "tokens1": tokens1,
                "tokens2": tokens2,
                "attention_mask1": attention_mask_1,
                "attention_mask2": attention_mask_2,
                "contact_map": contact_map,
            }
        else:
            return {
                "tokens1": tokens1,
                "tokens2": tokens2,
                "attention_mask1": attention_mask_1,
                "attention_mask2": attention_mask_2,
                "labels": torch.tensor(batch_labels),
            }
