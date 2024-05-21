import random
from typing import Any, Callable, Iterable, Optional, Sequence, TypeVar, Union

import torch
from lightning import LightningDataModule
from torch import Generator
from torch.utils.data import DataLoader, Sampler

from lobster.data._collate import ESMBatchConverterPPI
from lobster.datasets._ab_ag_sequence_ppi_dataset import AbAgSequencePPIDataset
from lobster.transforms._atom3d_ppi_transforms import PairedSequenceToTokens

T = TypeVar("T")


class PPISequenceDataModule(LightningDataModule):
    def __init__(
        self,
        data=None,
        source=None,
        *,
        cache_sequence_indicies: bool = True,
        remove_nulls: Optional[bool] = False,
        transform_fn: Optional[Callable] = None,
        target_transform_fn: Optional[Callable] = None,
        joint_transform_fn: Optional[Callable] = None,
        lengths: Optional[Sequence[float]] = None,
        generator: Optional[Generator] = None,
        seed: int = 0xDEADBEEF,
        batch_size: int = 1,
        shuffle: bool = True,
        sampler: Optional[Union[Iterable, Sampler]] = None,
        batch_sampler: Optional[Union[Iterable[Sequence], Sampler[Sequence]]] = None,
        num_workers: int = 0,
        collate_fn: Optional[
            Callable[["list[T]"], Any]
        ] = None,  # Hydra note -- should be data._collate.ESMBatchConverterPPI
        pin_memory: bool = True,
        drop_last: bool = False,
        truncation_seq_length=512,
        tokenizer_dir="pmlm_tokenizer",
        contact_maps=False,
        sequence1_cols: tuple = ("fv_heavy", "fv_light"),
        sequence2_cols: tuple = ("antigen_sequence",),
        label_col: Optional[str] = None,
    ) -> None:
        super().__init__()

        if generator is None:
            generator = Generator().manual_seed(seed)
        self._data = data
        self._source = source
        self._cache_sequence_indicies = cache_sequence_indicies
        self._transform_fn = transform_fn
        self._target_transform_fn = target_transform_fn
        self._joint_transform_fn = joint_transform_fn
        self._lengths = lengths
        self._generator = generator
        self._seed = seed
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._sampler = sampler
        self._batch_sampler = batch_sampler
        self._num_workers = num_workers
        self._collate_fn = collate_fn
        self._pin_memory = pin_memory
        self._drop_last = drop_last
        self._remove_nulls = remove_nulls
        self._sequence1_cols = sequence1_cols
        self._sequence2_cols = sequence2_cols
        self._label_col = label_col

        self._dataset = None
        self._data = data

        if collate_fn is not None:
            self._collate_fn = collate_fn
        else:
            self._collate_fn = ESMBatchConverterPPI(
                truncation_seq_length=truncation_seq_length,
                contact_maps=contact_maps,
                tokenizer_dir=tokenizer_dir,
            )

    def prepare_data(self) -> None:
        # Load in Dataset, transform sequences
        input_dataset = AbAgSequencePPIDataset(
            transform_fn=PairedSequenceToTokens().transform,
            target_transform_fn=None,
            data=self._data,
            source=self._source,
            sequence1_cols=self._sequence1_cols,
            sequence2_cols=self._sequence2_cols,
            label_col=self._label_col,
        )

        self._dataset = input_dataset

    def setup(self, stage: str = "fit") -> None:  # noqa: ARG002
        """NOTE - writing v0 assuming that a transform exists mapping Atom3D atoms_neighbrs --> seq1, seq2, interactions"""
        # Set random seeds
        random.seed(self._seed)
        torch.manual_seed(self._seed)

        if self._dataset is None:
            self.prepare_data()

        if stage == "fit":
            (
                self._train_dataset,
                self._val_dataset,
                self._test_dataset,
            ) = torch.utils.data.random_split(
                self._dataset,
                lengths=self._lengths,
                generator=self._generator,
            )

        if stage == "predict":
            self._predict_dataset = self._dataset

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_dataset,
            batch_size=self._batch_size,
            shuffle=self._shuffle,
            sampler=self._sampler,
            num_workers=self._num_workers,
            collate_fn=self._collate_fn,
            pin_memory=self._pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            sampler=self._sampler,
            num_workers=self._num_workers,
            collate_fn=self._collate_fn,
            pin_memory=self._pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._test_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            sampler=self._sampler,
            num_workers=self._num_workers,
            collate_fn=self._collate_fn,
            pin_memory=self._pin_memory,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self._predict_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            sampler=self._sampler,
            num_workers=self._num_workers,
            collate_fn=self._collate_fn,
            pin_memory=self._pin_memory,
        )
