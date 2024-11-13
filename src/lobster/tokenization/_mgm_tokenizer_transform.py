import importlib.resources
from os import PathLike
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import selfies as sf
import torch
from transformers.tokenization_utils_base import (
    BatchEncoding,
    PaddingStrategy,
    TruncationStrategy,
)
from transformers.utils import logging

from lobster.tokenization import MgmTokenizer
from lobster.transforms import (
    Transform,
    convert_aa_to_nt,
    convert_aa_to_selfies,
    convert_nt_to_aa,
    convert_nt_to_selfies,
    convert_selfies_to_aa,
    convert_selfies_to_nt,
    invert_residue_to_codon_mapping,
    json_load,
    random_boolean_choice,
    sample_list_with_probs,
    uniform_sample,
)

logger = logging.get_logger(__name__)


class MgmTokenizerTransform(Transform):
    def __init__(
        self,
        pretrained_model_name_or_path: Union[str, PathLike] = None,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = False,
        max_length: Optional[int] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        tokenizer_dir: Optional[str] = "mgm_tokenizer",
        codon_tables_dir: Optional[str] = "codon_tables",
        codon_to_residue_file: Optional[str] = "codon_table.json",
        mlm: bool = True,
        codon_sampling_strategy: Callable = uniform_sample,
        input_modality: Literal["nt", "aa", "selfies"] = "nt",
        p_aa: float = None,  # 0.111,
        p_nt: float = None,  # 0.111,
        p_sf: float = None,  # 0.111,
        p_aa_nt: float = None,  # 0.111,
        p_nt_aa: float = None,  # 0.111,
        p_aa_sf: float = None,  # 0.111,
        p_sf_aa: float = None,  # 0.111,
        p_nt_sf: float = None,  # 0.111,
        p_sf_nt: float = None,  # 0.111,
        mask_percentage: float = None,  # 0.25,
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
        self._codon_tables_dir = codon_tables_dir
        self._mlm = mlm
        self._mask_percentage = mask_percentage
        self._input_modality = input_modality
        self._codon_sampling_strategy = codon_sampling_strategy

        self._modality_probs = [
            p_aa,
            p_nt,
            p_sf,
            p_aa_nt,
            p_nt_aa,
            p_aa_sf,
            p_sf_aa,
            p_nt_sf,
            p_sf_nt,
        ]
        self._modalities = [
            "aa",
            "nt",
            "selfies",
            "aa_nt",
            "nt_aa",
            "aa_sf",
            "sf_aa",
            "nt_sf",
            "sf_nt",
        ]

        self._n_modalities = {}
        for k in self._modalities:
            self._n_modalities[k] = 2 if "_" in k else 1

        self._n_modalities_to_n_special_toks = {
            1: 3,  # cls, cls_modality, eos
            2: 5,  # cls, cls_modality * 2, sep, eos
        }

        self._aa_toks_per_residue = 1
        self._nt_toks_per_residue = 3
        self._sf_toks_per_residue = 13

        self._modalities_to_toks_per_res = {
            "aa": self._aa_toks_per_residue,
            "nt": self._nt_toks_per_residue,
            "selfies": self._sf_toks_per_residue,
            "aa_nt": self._aa_toks_per_residue + self._nt_toks_per_residue,
            "nt_aa": self._nt_toks_per_residue + self._aa_toks_per_residue,
            "aa_sf": self._aa_toks_per_residue + self._sf_toks_per_residue,
            "sf_aa": self._sf_toks_per_residue + self._aa_toks_per_residue,
            "nt_sf": self._nt_toks_per_residue + self._sf_toks_per_residue,
            "sf_nt": self._sf_toks_per_residue + self._nt_toks_per_residue,
        }

        if self._pretrained_model_name_or_path is not None:
            self._auto_tokenizer = MgmTokenizer.from_pretrained(
                self._pretrained_model_name_or_path,
                do_lower_case=False,
                use_fast=True,
            )
        elif self._tokenizer_dir is not None:
            path = importlib.resources.files("lobster") / "assets" / self._tokenizer_dir
            self._auto_tokenizer = MgmTokenizer.from_pretrained(
                path,
                do_lower_case=False,
                use_fast=True,
            )
            self._cls_token_id = self._auto_tokenizer.cls_token_id
            self._pad_token_id = self._auto_tokenizer.pad_token_id
            self._eos_token_id = self._auto_tokenizer.eos_token_id
            self._sep_token_id = self._auto_tokenizer.sep_token_id
            self._mask_token_id = self._auto_tokenizer.mask_token_id
            self._cls_aa_token_id = self._auto_tokenizer.cls_aa_token_id
            self._cls_nt_token_id = self._auto_tokenizer.cls_nt_token_id
            self._cls_sf_token_id = self._auto_tokenizer.cls_sf_token_id

            cls_aa_token = self._auto_tokenizer.cls_aa_token
            cls_nt_token = self._auto_tokenizer.cls_nt_token
            cls_sf_token = self._auto_tokenizer.cls_sf_token

            self._modality_cls_mapping = {
                "aa": {0: cls_aa_token},
                "nt": {0: cls_nt_token},
                "selfies": {0: cls_sf_token},
                "aa_nt": {0: cls_aa_token, 1: cls_nt_token},
                "nt_aa": {0: cls_nt_token, 1: cls_aa_token},
                "aa_sf": {0: cls_aa_token, 1: cls_sf_token},
                "sf_aa": {0: cls_sf_token, 1: cls_aa_token},
                "nt_sf": {0: cls_nt_token, 1: cls_sf_token},
                "sf_nt": {0: cls_sf_token, 1: cls_nt_token},
            }

        if self._codon_tables_dir is not None:
            path = importlib.resources.files("lobster") / "assets" / self._codon_tables_dir / codon_to_residue_file
            self._residue_to_codon = json_load(path)
            self._codon_to_residue = invert_residue_to_codon_mapping(self._residue_to_codon)
            self._allowed_aa = set(self._residue_to_codon.keys())
            self._allowed_nt = set(self._codon_to_residue.keys())
            self._stop_codons = self._residue_to_codon["STOP"]

    def sample_modalities(self) -> str:
        return sample_list_with_probs(self._modalities, self._modality_probs)

    def compute_max_seq_len(self, n_modalities: int) -> int:
        n_special_toks = self._n_modalities_to_n_special_toks[n_modalities]
        return self._max_length - n_special_toks

    def preprocess_seq(self, seq: str) -> str:
        if self._input_modality in ["aa", "sf"]:
            return seq.upper()

        # truncate nt seq if doesn't have 3n nt
        if len(seq) < 3:
            return None

        if len(seq) % 3 == 1:
            seq = seq[:-1]

        if len(seq) % 3 == 2:
            seq = seq[:-2]

        # truncate nt seq in case of early STOP codon
        for i in range(0, len(seq), 3):
            codon = seq[i : i + 3].upper()
            if (codon in self._stop_codons) and (i < len(seq) - 3):
                seq = seq[: i + 3]
        seq = seq.upper()  # codon table is in capital letters
        return seq

    def get_selfies_length(self, seq: str) -> int:
        symbols = list(sf.split_selfies(seq))
        return len(symbols)

    def crop_selfies(self, seq: str, max_len: int, crop_left: bool) -> str:
        symbols = list(sf.split_selfies(seq))
        if crop_left:
            symbols = symbols[-max_len:]
        else:
            symbols = symbols[:max_len]
        return "".join(symbols)

    def crop_non_selfies(self, seq: str, max_len: int, crop_left: bool) -> str:
        if crop_left:
            return seq[-max_len:]
        else:
            return seq[:max_len]

    def compute_max_res(self, max_len: int, in_toks_per_res: int, out_toks_per_res: int) -> int:
        """
        Computes the max # of residues we can encode given the sampled modality
        combination, and returns the max # of tokens we can keep in the input seq.

        Example:
            - max_len = 60
            - input_modality = "nt" -> in_toks_per_res = 3
            - output_modality = "aa_sf" -> out_toks_per_res = 1 + 13 = 14

        How many residues can we encode at most?
            -> max_encoded_res = int(60/14) = 4
        How many tokens in the input seq should we keep to encode max_encoded_res?
            -> max_input_toks = 3*4 = 12
        => We can keep 12 tokens in the input seq.
        """
        max_encoded_res = int(max_len / out_toks_per_res)
        max_input_toks = in_toks_per_res * max_encoded_res
        return max_input_toks

    def crop_seq(
        self,
        seq: str,
        modality_combination: str,
        n_modalities: int,
        max_len: int,
        crop_left: bool,
    ) -> str:
        """
        Computes the max # of tokens we can keep in the input seq, randomly
        sample side of seq which we be cropped, and returns cropped seq.
        """
        # compute max # of input tokens we can keep

        input_toks_per_res = self._modalities_to_toks_per_res[self._input_modality]
        output_toks_per_res = self._modalities_to_toks_per_res[modality_combination]
        max_input_toks = self.compute_max_res(max_len, input_toks_per_res, output_toks_per_res)
        # crop seq
        if self._input_modality == "selfies":
            return self.crop_selfies(seq, max_input_toks, crop_left)
        else:
            return self.crop_non_selfies(seq, max_input_toks, crop_left)

    def postprocess_selfies_inputs(
        self,
        inputs: List[str],
        max_len: int,
        crop_left: bool,
        modality_0: str,
        modality_1: str,
    ) -> List[str]:
        """
        Inputs: 2 seqs from different modalities, one of which is selfies. Their total length is > max_len.
        Computes the max # of residues which can be encoded between the 2 seqs, and crops them such that:
            - both seqs encode roughly the same # of residues,
            - their total length <= max_len.
        """

        if modality_0 == "sf":
            sf_seq = inputs[0]
            other_seq = inputs[1]
            other_toks_per_res = self._modalities_to_toks_per_res[modality_1]
        else:
            other_seq = inputs[0]
            sf_seq = inputs[1]
            other_toks_per_res = self._modalities_to_toks_per_res[modality_0]

        sf_len = self.get_selfies_length(sf_seq)
        sf_symbols_per_res = (sf_len / len(other_seq)) / other_toks_per_res
        max_res = int(max_len / (sf_symbols_per_res + other_toks_per_res))
        max_sf_len = int(max_res * sf_symbols_per_res)
        max_other_len = max_res * other_toks_per_res

        sf_seq = self.crop_selfies(sf_seq, max_sf_len, crop_left)
        other_seq = self.crop_non_selfies(other_seq, max_other_len, crop_left)

        if modality_0 == "sf":
            return [sf_seq, other_seq]
        else:
            return [other_seq, sf_seq]

    def prep_input(
        self,
        input: Union[str, List[str]],
        modality_combination: str,
        n_modalities: int,
        max_len: int,
        crop_left: bool,
    ) -> List[str]:
        """Converts input sequence into the sampled modality combination, truncates the resulting
        sequence(s) according to the maximum possible sequence length, and adds modality-specific
        cls tokens as such:
        - single modality: `[CLS_MODALITY] X`
        - modality pair: `[CLS_MODALITY_A] A [SEP] [CLS_MODALITY_B] B`

        Parameters
        ----------
        input : Union[str, List[str]]
            Input sequence(s) to be converted.
        modality_combination : str
            Sampled modality combination of the form 'modality_0' or 'modality_0_modality_1'
            where 'modality_0' and 'modality_1' are in ['aa', 'nt', 'sf'].
        n_modalities : int
            Number of modalities in the sampled combination.
        max_len : int
            Maximum sequence length.
        crop_left : bool
            Whether to crop the left or right side of the sequence.
        """

        if isinstance(input, str):
            input = [input]

        outputs = []

        modality_conversion_map = {
            "aa": {
                "aa": lambda text: [text.upper()],
                "nt": lambda text: [
                    convert_aa_to_nt(text, self._residue_to_codon, self._codon_sampling_strategy).lower()
                ],
                "selfies": lambda text: [convert_aa_to_selfies(text, self._allowed_aa)],
                "aa_nt": lambda text: [
                    text.upper(),
                    convert_aa_to_nt(text, self._residue_to_codon, self._codon_sampling_strategy).lower(),
                ],
                "nt_aa": lambda text: [
                    convert_aa_to_nt(text, self._residue_to_codon, self._codon_sampling_strategy).lower(),
                    text.upper(),
                ],
                "aa_sf": lambda text: [text.upper(), convert_aa_to_selfies(text, self._allowed_aa)],
                "sf_aa": lambda text: [convert_aa_to_selfies(text, self._allowed_aa), text.upper()],
                "nt_sf": lambda text: [
                    convert_aa_to_nt(text, self._residue_to_codon, self._codon_sampling_strategy).lower(),
                    convert_aa_to_selfies(text, self._allowed_aa),
                ],
                "sf_nt": lambda text: [
                    convert_aa_to_selfies(text, self._allowed_aa),
                    convert_aa_to_nt(text, self._residue_to_codon, self._codon_sampling_strategy).lower(),
                ],
            },
            "nt": {
                "aa": lambda text: [convert_nt_to_aa(text, self._codon_to_residue).upper()],
                "nt": lambda text: [text.lower()],
                "selfies": lambda text: [convert_nt_to_selfies(text, self._codon_to_residue, self._allowed_aa)],
                "aa_nt": lambda text: [convert_nt_to_aa(text, self._codon_to_residue).upper(), text.lower()],
                "nt_aa": lambda text: [text.lower(), convert_nt_to_aa(text, self._codon_to_residue).upper()],
                "aa_sf": lambda text: [
                    convert_nt_to_aa(text, self._codon_to_residue).upper(),
                    convert_nt_to_selfies(text, self._codon_to_residue, self._allowed_aa),
                ],
                "sf_aa": lambda text: [
                    convert_nt_to_selfies(text, self._codon_to_residue, self._allowed_aa),
                    convert_nt_to_aa(text, self._codon_to_residue).upper(),
                ],
                "nt_sf": lambda text: [
                    text.lower(),
                    convert_nt_to_selfies(text, self._codon_to_residue, self._allowed_aa),
                ],
                "sf_nt": lambda text: [
                    convert_nt_to_selfies(text, self._codon_to_residue, self._allowed_aa),
                    text.lower(),
                ],
            },
            "sf": {
                "aa": lambda text: [convert_selfies_to_aa(text)],
                "nt": lambda text: [
                    convert_selfies_to_nt(text, self._residue_to_codon, self._codon_sampling_strategy).lower()
                ],
                "selfies": lambda text: [text],
                "aa_nt": lambda text: [
                    convert_selfies_to_aa(text),
                    convert_selfies_to_nt(text, self._residue_to_codon, self._codon_sampling_strategy).lower(),
                ],
                "nt_aa": lambda text: [
                    convert_selfies_to_nt(text, self._residue_to_codon, self._codon_sampling_strategy).lower(),
                    convert_selfies_to_aa(text),
                ],
                "aa_sf": lambda text: [convert_selfies_to_aa(text), text],
                "sf_aa": lambda text: [text, convert_selfies_to_aa(text)],
                "nt_sf": lambda text: [
                    convert_selfies_to_nt(text, self._residue_to_codon, self._codon_sampling_strategy).lower(),
                    text,
                ],
                "sf_nt": lambda text: [
                    text,
                    convert_selfies_to_nt(text, self._residue_to_codon, self._codon_sampling_strategy).lower(),
                ],
            },
        }

        cls_modality_0 = self._modality_cls_mapping[modality_combination][0]
        if n_modalities == 2:
            cls_modality_1 = self._modality_cls_mapping[modality_combination][1]
            sep = self._auto_tokenizer.sep_token
        if n_modalities > 2:
            raise NotImplementedError("Cannot have > 2 modalities.")

        for seq in input:
            # prepare input seq
            seq = self.preprocess_seq(seq)
            cropped_seq = self.crop_seq(seq, modality_combination, n_modalities, max_len, crop_left)

            # convert input seq into sampled modalities
            out = modality_conversion_map[self._input_modality].get(modality_combination, lambda text: None)(
                cropped_seq
            )

            # postprocess inputs with selfies
            if (n_modalities == 2) and ("sf" in modality_combination):
                modality_0, modality_1 = modality_combination.split("_")

                if modality_0 == "sf":
                    sf_len = self.get_selfies_length(out[0])
                    other_seq_len = len(out[1])
                else:
                    sf_len = self.get_selfies_length(out[1])
                    other_seq_len = len(out[0])
                seqs_total_len = sf_len + other_seq_len

                n_special_toks = self._n_modalities_to_n_special_toks[n_modalities]
                if seqs_total_len + n_special_toks > self._max_length:
                    out = self.postprocess_selfies_inputs(out, max_len, crop_left, modality_0, modality_1)

            seq0 = out[0]

            # add modality-specific tokens
            if n_modalities == 1:
                seq = cls_modality_0 + seq0
            elif n_modalities == 2:
                seq1 = out[1]
                seq = cls_modality_0 + seq0 + sep + cls_modality_1 + seq1
            outputs.append(seq)

        return outputs

    def mask_single_modality_inputs(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Randomly mask mask % of input_ids, excluding special tokens."""

        masked_inputs = input_ids.clone()

        # Do not mask special tokens: cls, pad, eos
        special_tokens_mask = (
            (input_ids == self._cls_token_id) | (input_ids == self._pad_token_id) | (input_ids == self._eos_token_id)
        )

        # Get the indices of the non-special tokens
        non_special_indices = (~special_tokens_mask).nonzero(as_tuple=True)[1]

        # Determine the number of elements to replace (mask % of the non-special elements)
        num_to_replace = int(non_special_indices.numel() * self._mask_percentage)

        # Generate random indices to replace from the non-special indices
        replace_indices = non_special_indices[torch.randperm(non_special_indices.numel())[:num_to_replace]]

        # Replace the selected indices with the mask id
        masked_inputs.view(-1)[replace_indices] = self._mask_token_id  # 7

        del special_tokens_mask
        return masked_inputs

    def mask_two_modality_inputs(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Find the index of the sep_token_id
        sep_index = (input_ids == self._sep_token_id).nonzero(as_tuple=True)[1]
        if not len(sep_index):
            return self.mask_single_modality_inputs(input_ids)

        masked_inputs = input_ids.clone()

        # Do not mask special tokens: cls, pad, eos + modality-specific cls
        special_tokens_mask = (
            (input_ids == self._cls_token_id)
            | (input_ids == self._pad_token_id)
            | (input_ids == self._eos_token_id)
            | (input_ids == self._cls_aa_token_id)
            | (input_ids == self._cls_nt_token_id)
            | (input_ids == self._cls_sf_token_id)
        )

        # Randomly choose to fill left or right of the sep_token_id
        fill_left = random_boolean_choice()

        # Create a mask for the side to be fully replaced with mask token
        if fill_left:
            full_replace_mask = torch.arange(input_ids.size(1)) < sep_index
        else:
            full_replace_mask = torch.arange(input_ids.size(1)) > sep_index
        full_replace_mask = full_replace_mask.unsqueeze(0)

        # Ensure special tokens are not replaced
        full_replace_mask &= ~special_tokens_mask

        # Apply the full replacement with 7 on the chosen side
        masked_inputs[full_replace_mask] = self._mask_token_id

        # Create a mask for the side to be partially replaced with 7
        partial_replace_mask = ~full_replace_mask & ~special_tokens_mask

        # Get the indices of the non-special tokens on the partial side
        non_special_indices = partial_replace_mask.nonzero(as_tuple=True)[1]

        # Determine the number of elements to replace (25% of the non-special elements)
        num_to_replace = int(non_special_indices.numel() * self._mask_percentage)

        # Generate random indices to replace from the non-special indices
        replace_indices = non_special_indices[torch.randperm(non_special_indices.numel())[:num_to_replace]]

        # Replace the selected indices with the value 7
        masked_inputs.view(-1)[replace_indices] = self._mask_token_id

        del special_tokens_mask, full_replace_mask, partial_replace_mask
        return masked_inputs

    def compute_labels(self, toks: torch.Tensor, masked_toks: torch.Tensor) -> torch.Tensor:
        labels = toks.clone()
        labels[masked_toks != self._mask_token_id] = -100
        return labels

    def transform(
        self,
        text: Union[str, List[str], List[int]],
        parameters: dict[str, Any],
    ) -> BatchEncoding:
        modality_combination = self.sample_modalities()
        # modality_combination = "sf_nt"
        n_modalities = self._n_modalities[modality_combination]
        max_seq_len = self.compute_max_seq_len(n_modalities)
        crop_left = random_boolean_choice()
        inputs = self.prep_input(text, modality_combination, n_modalities, max_seq_len, crop_left)
        tokenized = self._auto_tokenizer(
            inputs,
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

        # ignore labels in loss
        if self._mlm:
            toks = tokenized["input_ids"]
            masked_toks_0 = self.mask_single_modality_inputs(toks)
            masked_toks_1 = self.mask_two_modality_inputs(toks)
            labels_0 = self.compute_labels(toks, masked_toks_0)
            labels_1 = self.compute_labels(toks, masked_toks_1)

            tokenized["masked_ids_0"] = masked_toks_0
            tokenized["masked_ids_1"] = masked_toks_1
            tokenized["labels_0"] = labels_0
            tokenized["labels_1"] = labels_1

        else:
            labels = tokenized["input_ids"].clone()
            # ignore padding in loss
            if self._auto_tokenizer.pad_token_id is not None:
                labels[labels == self._auto_tokenizer.pad_token_id] = -100
            tokenized["labels"] = labels
        return tokenized

    def _reverse_text(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        if isinstance(text, str):
            return text[::-1]
        elif isinstance(text, list):
            return [t[::-1] for t in text]

    def _transform(self, input: Any, parameters: Dict[str, Any]) -> Any:
        return self.transform(input, parameters)

    def validate(self, flat_inputs: list[Any]) -> None:
        pass

    def _check_inputs(self, inputs: List[Any]) -> None:
        pass
