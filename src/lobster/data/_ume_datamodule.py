import logging
from collections.abc import Callable, Iterator, Sequence
from typing import Any, cast

import torch.utils.data
from lightning import LightningDataModule
from litdata import CombinedStreamingDataset, StreamingDataLoader, StreamingDataset
from omegaconf import DictConfig
from torch import Generator, Tensor

import lobster.datasets.s3_datasets
from lobster.datasets.s3_datasets import UMEStreamingDataset
from lobster.constants import Modality, Split
from lobster.transforms import Transform

logger = logging.getLogger(__name__)
logging.getLogger("botocore.credentials").setLevel(logging.WARNING)


class UMELightningDataModule(LightningDataModule):
    def __init__(
        self,
        *,
        datasets: Sequence[str],
        transforms: dict[str, list[Transform | Callable]] | None = None,
        root: str | None = None,
        max_length: int | None = 1024,
        weights: None | Sequence[float] | dict[str, float] = None,
        seed: int = 0,
        batch_size: int = 1,
        pin_memory: bool = False,
        num_workers: int = 0,
        dataset_kwargs: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        super().__init__()

        self.dataset_names = datasets
        self.batch_size = batch_size
        self.max_length = max_length

        self.root = root
        self.generator = Generator().manual_seed(seed)
        self.seed = seed
        self.pin_memory = pin_memory
        self.num_workers = num_workers

        self.transforms = transforms if transforms is not None else {}
        self.weights = weights
        self.dataset_kwargs = dataset_kwargs if dataset_kwargs is not None else {}

        self._validate_and_process_weights()
        self._validate_transforms()
        self._initialize_datasets()

    def _validate_and_process_weights(self) -> None:
        """Validate and process weights, converting dict to sequence if needed."""
        if self.weights:
            if len(self.weights) != len(self.dataset_names):
                raise ValueError(
                    f"Number of weights ({len(self.weights)}) must match number of datasets ({len(self.dataset_names)})."
                )
            if isinstance(self.weights, (dict, DictConfig)):  # noqa: UP038
                self.weights = [self.weights[dataset] for dataset in self.dataset_names]

            assert isinstance(self.weights, Sequence), f"Weights must be a sequence of floats but got {self.weights}"
            assert all(isinstance(w, float) for w in self.weights), (
                f"Weights must be a sequence of floats but got {self.weights}"
            )
            assert len(self.weights) == len(self.dataset_names), (
                f"Mismatch between weights and datasets: {len(self.weights)} != {len(self.dataset_names)}"
            )

        self.weights = cast(Sequence[float] | None, self.weights)

    def _validate_transforms(self) -> None:
        """Validate transforms configuration."""
        if self.transforms:
            extra = set(self.transforms.keys()) - set(self.dataset_names)
            # check that those transforms are not None
            if extra and any(self.transforms[dataset] is not None for dataset in extra):
                raise ValueError(f"Transforms for non-existent datasets: {', '.join(extra)}")

    def _initialize_datasets(self) -> None:
        """Initialize empty dataset containers."""
        self.train_datasets: list[StreamingDataset] = []
        self.val_datasets: list[StreamingDataset] = []

        self.train_dataset: CombinedStreamingDataset | None = None
        self.val_dataset: CombinedStreamingDataset | None = None

    def _get_dataset(self, dataset_name: str, split: Split) -> Iterator[UMEStreamingDataset]:
        logging.info(f"""Initializing dataset {dataset_name} for split {split.value}...""")

        ds_class = getattr(lobster.datasets.s3_datasets, dataset_name)
        transforms = self.transforms.get(dataset_name, [None])

        kwargs = {
            "max_length": self.max_length,
            "split": split,
            "seed": self.seed,
            "cache_dir": self.root,
            "use_optimized": True,
        } | dict(self.dataset_kwargs.get(dataset_name, {}))

        for transform in transforms:
            try:
                ds_kwargs = kwargs.copy()
                dataset_class_name = ds_class.__name__

                ds_kwargs = {**kwargs, "transform_fn": transform}

                ds = ds_class(**ds_kwargs)
                transform_msg = f" with transform {transform.__class__.__name__}" if transform else " with no transform"

                logging.info(
                    f"Initialized dataset {dataset_class_name} for split {split.value}{transform_msg}: size={len(ds)}"
                )

                yield ds

            except Exception as e:
                raise RuntimeError(f"Failed to initialize dataset {dataset_name} for split {split.value}") from e

    def setup(self, stage: str | None = None) -> None:
        """Set up the dataset nodes for training and validation."""

        self.train_datasets = []
        self.val_datasets = []

        for dataset_name in self.dataset_names:
            for ds in self._get_dataset(dataset_name, Split.TRAIN):
                self.train_datasets.append(ds)

            for ds in self._get_dataset(dataset_name, Split.VALIDATION):
                self.val_datasets.append(ds)

        # Combine the datasets
        self.train_dataset = CombinedStreamingDataset(
            self.train_datasets,
            seed=self.seed,
            weights=self.weights,
            iterate_over_all=False,
        )
        logging.info(f"Initialized training dataset: {self.train_dataset}")

        self.val_dataset = CombinedStreamingDataset(self.val_datasets, seed=self.seed, iterate_over_all=False)

        logging.info(f"Initialized validation dataset: {self.val_dataset}")

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Return the training dataloader."""
        if self.train_dataset is None:
            raise ValueError("train_dataset is not initialized. Call setup() first.")

        return StreamingDataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_with_modality,
            pin_memory=self.pin_memory,
            generator=self.generator,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader | None:
        """Return the validation dataloader."""
        if self.val_dataset is None:
            raise ValueError("val_dataset is not initialized. Call setup() first.")

        return StreamingDataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_with_modality,
            pin_memory=self.pin_memory,
            generator=self.generator,
        )


def collate_with_modality(batch: list[dict[str, Tensor | Modality]]) -> dict[str, Tensor | list[Modality] | list[str]]:
    modalities = [item.get("modality") for item in batch]
    sequences = [item.get("sequence", "") for item in batch]
    dataset = [item.get("dataset", "") for item in batch]

    # Use default collate function for tensors
    tensor_batch = [{key: item[key] for key in item if isinstance(item[key], Tensor)} for item in batch]
    tensor_batch = torch.utils.data.default_collate(tensor_batch)

    return {
        **tensor_batch,
        "modality": modalities,
        "sequence": sequences,
        "dataset": dataset,
    }
