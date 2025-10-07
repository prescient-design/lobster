import logging
from typing import Any

from litdata.streaming.dataset import StreamingDataset
from nodehaha.config.load_from import LoadFrom
from sethaha.sources.gred_affinity_tabular import GredAffinityTabularDataset

from lobster.constants import Modality, Split
from lobster.datasets.s3_datasets import UMEStreamingDataset

logger = logging.getLogger(__name__)


class GredAffinityUMEStreamingDataset(UMEStreamingDataset):
    """
    Thin UME Streaming Dataset wrapper for the GRED Affinity tabular dataset.

    This implementation exposes the affinity regression task by combining the
    heavy and light antibody chains with the antigen sequence and returning the
    requested regression target value.

    Example
    -------
    ```python
    dataset = GredAffinityUMEStreamingDataset(
        split="train",
        target_column="affinity_pkd",
        max_length=1024,
    )
    sample = next(iter(dataset))
    print(sample["target"], sample["sequence"])
    ```
    """
    
    # Dataset configuration - get from existing GredAffinityTabularDataset
    MODALITY = Modality.AMINO_ACID
    SEQUENCE_KEY = "fv_heavy_aho"  # Raw key used only when parent logic is invoked
    
    def __init__(
        self,
        split: Split | str,
        target_column: str = "affinity_pkd",
        significant_figures: int = 3,
        seed: int = 0,
        cache_dir: str | None = None,
        transform_fn: Any = None,
        tokenize: bool = True,
        use_optimized: bool = False,
        max_length: int | None = 1024,
        # Optional explicit validation split key (e.g., "val" or "test") if dataset lacks a canonical val split
        val_split_key: str | None = None,
    ) -> None:
        """
        Initialize the GRED Affinity UME Streaming Dataset.
        
        This is a thin wrapper that creates a GredAffinityTabularDataset instance
        and uses its configuration and processing logic.
        
        Parameters
        ----------
        split : Split | str
            Dataset split to use (train, validation, test)
        target_column : str, default="affinity_pkd"
            Target column for regression tasks. Options: "expression_yield", "affinity_pkd"
        significant_figures : int, default=3
            Number of significant figures for numerical values
        seed : int, default=0
            Random seed for reproducibility
        cache_dir : str | None, default=None
            Directory for caching dataset files
        transform_fn : Any, default=None
            Optional transform function (will be set up internally)
        tokenize : bool, default=True
            Whether to tokenize sequences
        use_optimized : bool, default=False
            Whether to use optimized dataset format (not available for GRED)
        max_length : int | None, default=1024
            Maximum sequence length for tokenization
        """
        self.target_column = target_column
        self.significant_figures = significant_figures
        
        # Create the underlying GredAffinityTabularDataset to get configuration
        # Map Split to expected underlying strings (validation -> val)
        if isinstance(split, str):
            split_str = split
        else:
            split_str = split.value.lower()
            if split_str == "validation":
                # Respect explicit override if provided; otherwise map to canonical 'val'
                split_str = val_split_key if val_split_key is not None else "val"
        self._gred_dataset = GredAffinityTabularDataset(
            split=split_str,
            task_type="raw",  # GredAffinityTabularDataset only supports "raw" task type
            seed=seed,
            load_from=LoadFrom.S3,
        )
        
        # Set class attributes BEFORE calling super().__init__()
        # Resolve S3 split keys robustly across possible naming schemes
        s3_prefixes = self._gred_dataset.S3_PREFIXES

        def _resolve_split_key(candidates: list[str]) -> str:
            for key in candidates:
                if key in s3_prefixes:
                    return key
            available_keys = ", ".join(sorted(s3_prefixes.keys()))
            raise KeyError(
                f"None of the candidate split keys {candidates} found in S3_PREFIXES. Available: {available_keys}"
            )

        train_key = _resolve_split_key(["train", "training"])  # prefer canonical
        # Validation: if explicit override provided, use it; else, prefer val-like keys; else, fall back to test
        if val_split_key is not None:
            if val_split_key not in s3_prefixes:
                available_keys = ", ".join(sorted(s3_prefixes.keys()))
                raise KeyError(
                    f"Provided val_split_key='{val_split_key}' not found in S3_PREFIXES. Available: {available_keys}"
                )
            val_key = val_split_key
        else:
            try:
                val_key = _resolve_split_key(["val", "validation", "valid", "dev"])  # handle variations
            except KeyError:
                val_key = _resolve_split_key(["test"])  # reuse test if no validation split exists

        # Test: prefer explicit test; otherwise reuse val key
        try:
            test_key = _resolve_split_key(["test", "eval"])  # prefer canonical
        except KeyError:
            test_key = val_key

        self.__class__.SPLITS = {
            Split.TRAIN: s3_prefixes[train_key],
            Split.VALIDATION: s3_prefixes[val_key],
            Split.TEST: s3_prefixes[test_key],
        }
        
        # Set sequence key to an actual column in the parquet data (not used in our custom __next__)
        self.__class__.SEQUENCE_KEY = "fv_heavy_aho"
        
        # Do not use a transform function; handle processing in __next__.
        # Keep tokenize=True so tokenizer registry is available.
        super().__init__(
            split=split,
            seed=seed,
            cache_dir=cache_dir,
            transform_fn=None,
            tokenize=tokenize,
            use_optimized=use_optimized,
            max_length=max_length,
        )

    def __iter__(self):
        """Ensure StreamingDataset internal state (e.g., stop_length) is initialized."""
        return StreamingDataset.__iter__(self)
    
    def __next__(self) -> dict[str, Any]:
        """Get the next affinity regression sample from the dataset."""
        # Ensure iterator state is initialized (required by litdata StreamingDataset)
        if not hasattr(self, "stop_length"):
            StreamingDataset.__iter__(self)

        # Get the raw parquet row directly from litdata
        raw_item: dict = StreamingDataset.__next__(self)
        
        try:
            # Process the raw item using GredAffinityTabularDataset logic
            processed_item = self._gred_dataset.process_example({"data": raw_item, "metadata": {}})
            if processed_item is None:
                return self.__next__()

            heavy_chain = processed_item.get("heavy") or ""
            light_chain = processed_item.get("light") or ""
            antigen = processed_item.get("antigen") or ""
            target = processed_item.get(self.target_column)

            if not (heavy_chain and light_chain and antigen):
                return self.__next__()
            if target is None:
                return self.__next__()

            sequence = f"{heavy_chain}<sep>{light_chain}<sep>{antigen}"

            encoded = self.tokenizer_registry[self.MODALITY](sequence)
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]

            result = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "sequence": sequence,
                "modality": self.MODALITY.value,
                "dataset": self.__class__.__name__,
                "target": target,
            }

            metadata = processed_item.get("metadata")
            if metadata:
                result["metadata"] = metadata

            return result

        except Exception as e:
            logger.warning(f"Error processing GRED Affinity item: {e}")
            return self.__next__()
