import random
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence, TypeVar, Union

import pandas as pd
import torch
from datasets import Dataset
from lightning import LightningDataModule
from yeji.datasets._neglog_dataset import NegLogDataset
from torch import Generator
from torch.utils.data import DataLoader, Sampler

from lobster.data._collate import ESMBatchConverterPPI
from lobster.transforms._atom3d_ppi_transforms import PairedSequenceToTokens

T = TypeVar("T")


class NegLogDataModule(LightningDataModule):
    def __init__(
        self,
        root: Union[str, Path] = None,
        *,
        cache_sequence_indicies: bool = True,
        download: bool = False,
        remove_nulls: Optional[bool] = False,
        transform_fn: Optional[Callable] = None,
        target_transform_fn: Optional[Callable] = None,
        joint_transform_fn: Optional[Callable] = None,
        lengths: Optional[Sequence[float]] = (0.9, 0.05, 0.05),
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
    ) -> None:
        """
        :param path_to_fdata: path to .hdf file

        :param model_name: name of esm model

        :param root: Root directory where the dataset subdirectory exists or,
            if :attr:`download` is ``True``, the directory where the dataset
            subdirectory will be created and the dataset downloaded.

        :param cache_sequence_indicies: If ``True``, caches the sequence
            indicies to disk for faster re-initialization (default: ``True``).

        :param download: If ``True``, download the dataset and to the
            :attr:`root` directory (default: ``False``). If the dataset is
            already downloaded, it is not redownloaded.

        :param transform_fn: A ``Callable`` that maps a sequence to a
            transformed sequence (default: ``None``).

        :param target_transform_fn: ``Callable`` that maps a target (a cluster
            identifier) to a transformed target (default: ``None``).

        :param joint_transform_fn: ``Callable`` that maps a feature and target
            to a transformed feature, target (default: ``None``). Needed in cases
            where the target transform is dependent on the features or vice versa

        :param lengths: Fractions of splits to generate. Unsupervised contact map prediction only requires 20 "training point."
            Here we define

        :param generator: Generator used for the random permutation (default:
            ``None``).

        :param seed: Desired seed. Value must be within the inclusive range
            ``[-0x8000000000000000, 0xFFFFFFFFFFFFFFFF]`` (default:
            ``0xDEADBEEF``). Otherwise, a ``RuntimeError`` is raised. Negative
            inputs are remapped to positive values with the formula
            ``0xFFFFFFFFFFFFFFFF + seed``.

        :param batch_size: Samples per batch (default: ``1``).

        :param shuffle: If ``True``, reshuffle datasets at every epoch (default:
            ``True``).

        :param sampler: Strategy to draw samples from the dataset (default:
            ``None``). Can be any ``Iterable`` with ``__len__`` implemented.
            If specified, :attr:`shuffle` must be ``False``.

        :param batch_sampler: :attr:`sampler`, but returns a batch of indices
            (default: ``None``). Mutually exclusive with :attr:`batch_size`,
            :attr:`shuffle`, :attr:`sampler`, and :attr:`drop_last`.

        :param num_workers: Subprocesses to use (default: ``0``). ``0`` means
            that the datasets will be loaded in the main process.

        :param collate_fn: Merges samples to form a mini-batch of Tensor(s)
            (default: ``None``).

        :param pin_memory: If ``True``, Tensors are copied to the device's
            (e.g., CUDA) pinned memory before returning them (default:
            ``True``).

        :param drop_last: If ``True``, drop the last incomplete batch, if the
            dataset size is not divisible by the batch size (default:
            ``False``). If ``False`` and the size of dataset is not divisible
            by the batch size, then the last batch will be smaller.
        """
        super().__init__()

        if generator is None:
            generator = Generator().manual_seed(seed)

        self._root = root
        self._cache_sequence_indicies = cache_sequence_indicies
        self._download = download
        if transform_fn is not None:
            self._transform_fn = transform_fn
        else:
            self._transform_fn = PairedSequenceToTokens().transform
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
        if collate_fn is not None:
            self._collate_fn = collate_fn
        else:
            self._collate_fn = ESMBatchConverterPPI(
                truncation_seq_length=truncation_seq_length,
                contact_maps=contact_maps,
                tokenizer_dir=tokenizer_dir,
            )
        self._pin_memory = pin_memory
        self._drop_last = drop_last
        self._remove_nulls = remove_nulls

        self._dataset = None

    def prepare_data(self) -> None:
        # Load in Dataset, transform sequences
        dataset = NegLogDataset(
            root=self._root,
            transform_fn=self._transform_fn,
            target_transform_fn=self._target_transform_fn,
            filter_nulls=True,
        )

        self._dataset = dataset

    def setup(self, stage: str = "fit") -> None:  # noqa: ARG002

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

    def _clm_data_wrangle(self, dataset) -> Dataset:
        seqs_for_dl = []
        for pair in dataset:
            seqs_for_dl.append(tuple(pair))
        seq_dict = dict(seqs_for_dl)
        seq_dict_df = pd.DataFrame(seq_dict.items(), columns=["input_ids", "Labels"])
        seq_dict_df = Dataset.from_pandas(seq_dict_df)
        return seq_dict_df
