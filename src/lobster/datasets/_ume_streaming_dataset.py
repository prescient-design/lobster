import logging
import os
from collections.abc import Callable
from typing import Any

import litdata
import torch
from litdata import StreamingDataset
from litdata.streaming.item_loader import ParquetLoader
from upath import UPath

from lobster.constants import Modality, Split
from lobster.tokenization import UMETokenizerTransform

logger = logging.getLogger(__name__)


class UMEStreamingDataset(StreamingDataset):
    """
    Base class for UME streaming datasets that handles tokenization and data loading
    with litdata.

    This dataset is designed to work with different biological sequence modalities
    and provides functionality for tokenizing sequences for model input. It supports
    streaming data from various sources (e.g., S3) and handles both single and multi-sequence
    tokenization scenarios.

    Class Attributes
    ---------------
    MODALITY : Modality
        The biological sequence modality this dataset handles (e.g., AMINO_ACID, SMILES)
    SPLITS : dict[Split, str]
        Mapping of dataset splits to their filepaths (e.g., {Split.TRAIN: "s3://lobster/ume-dataset/train"})
    OPTIMIZED_SPLITS : dict[Split, str] | None
        Optional mapping of dataset splits to their optimized filepaths
    SEQUENCE_KEY : str
        The key used to access the sequence data in the dataset items
    TRAIN_SIZE : int | None
        Total size of the training split, if known
    VAL_SIZE : int | None
        Total size of the validation split, if known
    LIMIT_TRAIN_SIZE : int | None
        Optional limit on the training split size
    LIMIT_VAL_SIZE : int | None
        Optional limit on the validation split size
    """

    MODALITY: Modality
    SPLITS: dict[Split, str]  # Filepaths to the splits (e.g. {Split.TRAIN: "s3://lobster/ume-dataset/train"})
    OPTIMIZED_SPLITS: dict[Split, str] | None = None  # Filepaths to the lit-optimized splits
    SEQUENCE_KEY: str

    TRAIN_SIZE: int | None = None
    VAL_SIZE: int | None = None
    LIMIT_TRAIN_SIZE: int | None = None
    LIMIT_VAL_SIZE: int | None = None

    def __init__(
        self,
        split: Split | str,
        seed: int = 0,
        cache_dir: str | None = None,
        transform_fn: Callable | None = None,
        tokenize: bool = True,
        use_optimized: bool = False,
        max_length: int | None = 8192,
    ) -> None:
        """
        Initialize the UMEStreamingDataset.

        This constructor sets up the dataset with the specified configuration, including
        tokenization settings, data transformation, and caching options. It handles both
        optimized and non-optimized dataset formats.

        Parameters
        ----------
        split : Split | str
            Dataset split to use (train, validation, etc.). Can be provided as a string
            which will be converted to a Split enum.
        seed : int, default=0
            Random seed for reproducibility of data loading and shuffling
        cache_dir : str | None, default=None
            Directory for caching dataset files. If None, no caching is performed.
        transform_fn : Callable | None, default=None
            Optional function to transform sequences before tokenization. The function
            should accept a sequence and return a transformed sequence.
        tokenize : bool, default=True
            Whether to tokenize sequences. If False, raw sequences will be returned.
        use_optimized : bool, default=False
            Whether to use optimized dataset format. Requires OPTIMIZED_SPLITS to be defined.
        max_length : int | None, default=8192
            Maximum sequence length for tokenization. Required if tokenize is True.
        **kwargs : Any
            Additional arguments passed to StreamingDataset constructor

        Raises
        ------
        ValueError
            If split is not found in available splits
            If use_optimized is True but OPTIMIZED_SPLITS is not defined
            If tokenize is True but max_length is None
        """
        split = Split(split) if isinstance(split, str) else split
        subsample = self._calculate_subsample_rate(split)
        s3_uri = self._get_and_validate_uri(split, use_optimized)

        super().__init__(
            s3_uri,
            item_loader=ParquetLoader() if not use_optimized else None,
            subsample=subsample,
            drop_last=True,
            shuffle=split == Split.TRAIN,
            seed=seed,
            cache_dir=cache_dir,
            force_override_state_dict=True,
        )

        self.transform_fn = transform_fn
        self.tokenize = tokenize
        self.max_length = max_length
        self.use_optimized = use_optimized
        self.subsample = subsample

        if tokenize:
            self._setup_tokenizers(max_length)
        else:
            logger.warning(
                f"Tokenization is disabled for {self.__class__.__name__}. Please make sure this is intentional."
            )

    def _calculate_subsample_rate(self, split: Split) -> float:
        """
        Calculate the subsample rate based on split and dataset size limits.

        This method determines the appropriate subsample rate to use based on the
        dataset split and any size limits that have been set. It ensures that
        the dataset size stays within specified limits while maintaining
        proportional sampling.

        Parameters
        ----------
        split : Split
            The dataset split to calculate subsample rate for

        Returns
        -------
        float
            The subsample rate to use (between 0 and 1)
            Returns 1.0 if no size limits are set for the given split
        """
        if split == Split.TRAIN and self.TRAIN_SIZE is not None and self.LIMIT_TRAIN_SIZE is not None:
            return self.LIMIT_TRAIN_SIZE / self.TRAIN_SIZE
        elif split == Split.VALIDATION and self.VAL_SIZE is not None and self.LIMIT_VAL_SIZE is not None:
            return self.LIMIT_VAL_SIZE / self.VAL_SIZE
        return 1.0

    def _get_and_validate_uri(self, split: Split, use_optimized: bool) -> str:
        """
        Get and validate the S3 URI for the given split.

        This method retrieves the appropriate URI for the dataset split and performs
        validation checks, including verifying the existence of required index files.
        It can handle both optimized and non-optimized dataset formats.

        Parameters
        ----------
        split : Split
            The dataset split to get the URI for
        use_optimized : bool
            Whether to use optimized dataset format

        Returns
        -------
        str
            The validated S3 URI for the dataset split

        Raises
        ------
        ValueError
            If OPTIMIZED_SPLITS is not defined when use_optimized is True
            If the split is not found in available splits
            If the index file is missing for an optimized dataset
        """
        if use_optimized and self.OPTIMIZED_SPLITS is None:
            raise ValueError("Please define OPTIMIZED_SPLITS when seeting `use_optimized=True`")

        splits_map = self.OPTIMIZED_SPLITS if use_optimized else self.SPLITS
        s3_uri = splits_map.get(split)

        if s3_uri is None:
            raise ValueError(f"Split {split} not found in available split: {splits_map}")

        if not UPath(os.path.join(s3_uri, "index.json")).exists():
            if not use_optimized:
                logging.info(f"Index file not found at {s3_uri}, will create one now and reuse it later")
                litdata.index_parquet_dataset(s3_uri)
            else:
                raise ValueError(
                    f"Index file not found at {s3_uri}. Is the dataset use_optimized? Hint: run optimization or select `use_optimized=False`"
                )

        return s3_uri

    def _setup_tokenizers(self, max_length: int | None) -> None:
        """
        Set up tokenizers for different modalities.

        This method initializes the tokenizer registry with appropriate tokenizers
        for each supported modality. Each tokenizer is configured with the specified
        maximum sequence length.

        Parameters
        ----------
        max_length : int | None
            Maximum sequence length for tokenization

        Raises
        ------
        ValueError
            If max_length is None
        """
        if max_length is None:
            raise ValueError("max_length must be provided when tokenize is True")

        self.tokenizer_registry = {
            Modality.AMINO_ACID: UMETokenizerTransform(
                modality=Modality.AMINO_ACID, max_length=max_length, return_modality=False
            ),
            Modality.SMILES: UMETokenizerTransform(
                modality=Modality.SMILES, max_length=max_length, return_modality=False
            ),
            Modality.NUCLEOTIDE: UMETokenizerTransform(
                modality=Modality.NUCLEOTIDE, max_length=max_length, return_modality=False
            ),
        }

    def _tokenize_single(self, sequence: str) -> tuple[torch.Tensor, torch.Tensor, str]:
        """
        Tokenize a single sequence.

        This method tokenizes a single biological sequence using the appropriate
        tokenizer for the dataset's modality. It returns the tokenized sequence
        along with its attention mask and modality information.

        Parameters
        ----------
        sequence : str
            The biological sequence to tokenize

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, str]
            A tuple containing:
            - input_ids: Token IDs for the sequence
            - attention_mask: Attention mask for the sequence
            - modality: The modality of the sequence
        """
        encoded = self.tokenizer_registry[self.MODALITY](sequence)

        return encoded["input_ids"], encoded["attention_mask"], self.MODALITY.value

    def _tokenize_multiple(
        self, sequence: tuple[str, ...], modalities: tuple[str, ...]
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[str, ...]]:
        """
        Tokenize multiple sequences with their respective modalities.

        This method handles tokenization of multiple sequences, each potentially
        having a different modality. It ensures consistent tensor shapes by padding
        sequences to the maximum length in the batch.

        Parameters
        ----------
        sequence : tuple[str, ...]
            Tuple of sequences to tokenize
        modalities : tuple[str, ...]
            Tuple of modality values for each sequence

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, tuple[str, ...]]
            A tuple containing:
            - input_ids: Stacked tensor of token IDs for all sequences
            - attention_mask: Stacked tensor of attention masks for all sequences
            - modalities: Tuple of modality values for each sequence
        """
        encoded_list = []
        for seq, mod in zip(sequence, modalities):
            mod_obj = Modality(mod)
            encoded = self.tokenizer_registry[mod_obj](seq)
            # Detach and clone to avoid storage issues
            encoded = {k: v.detach().clone().squeeze(0) for k, v in encoded.items()}
            encoded_list.append(encoded)

        # Create separate tensors to avoid storage issues
        max_length = max(enc["input_ids"].size(0) for enc in encoded_list)
        batch_size = len(encoded_list)

        # Create new tensors with consistent shape
        input_ids = torch.zeros((batch_size, max_length), dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)

        # Copy data into the new tensors
        for i, enc in enumerate(encoded_list):
            length = enc["input_ids"].size(0)
            input_ids[i, :length] = enc["input_ids"]
            attention_mask[i, :length] = enc["attention_mask"]

        return input_ids, attention_mask, modalities

    def __next__(self) -> dict[str, Any]:
        """
        Get the next item from the dataset with tokenization applied if enabled.

        This method retrieves the next item from the dataset, applies any specified
        transformations, and tokenizes the sequence(s) if tokenization is enabled.
        It handles both single and multiple sequence cases, and skips items with
        None sequences.

        Returns
        -------
        dict[str, Any]
            A dictionary containing:
            - input_ids: Token IDs (None if tokenization is disabled)
            - attention_mask: Attention mask (None if tokenization is disabled)
            - sequence: The original or transformed sequence(s)
            - modality: The modality of the sequence(s)
            - dataset: The name of the dataset class
            - Additional metadata from the original item

        Raises
        ------
        StopIteration
            When the dataset is exhausted
        ValueError
            If transform_fn is provided but doesn't specify output modalities
            for multiple sequence cases
        """
        item: dict = super().__next__()

        sequence: str = item.pop(self.SEQUENCE_KEY)

        if sequence is None:
            return self.__next__()

        if self.transform_fn:
            sequence: str | tuple[str | None, ...] | list[str | None] | None = self.transform_fn(sequence)

        if sequence is None or (isinstance(sequence, list | tuple) and any(seq is None for seq in sequence)):
            return self.__next__()

        if not self.tokenize:
            return {
                "input_ids": None,
                "attention_mask": None,
                "sequence": sequence,
                "modality": self.MODALITY.value,
                "dataset": self.__class__.__name__,
                **item,
            }

        if isinstance(sequence, (tuple, list)):
            if len(sequence) == 1:
                # Single sequence case
                input_ids, attention_mask, modality = self._tokenize_single(sequence[0])
            else:
                # Multiple sequences case
                if hasattr(self.transform_fn, "output_modalities"):
                    modalities = self.transform_fn.output_modalities
                else:
                    raise ValueError(f"Transform {self.transform_fn} does not specify output_modalities")

                input_ids, attention_mask, modality = self._tokenize_multiple(sequence, modalities)
        else:
            input_ids, attention_mask, modality = self._tokenize_single(sequence)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "sequence": sequence,
            "modality": modality,
            "dataset": self.__class__.__name__,
            **item,
        }
