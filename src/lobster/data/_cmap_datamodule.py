import random
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence, TypeVar, Union

import pandas as pd
import torch
import torch.utils.data
from datasets import Dataset
from lightning import LightningDataModule

from lobster.data import _PRESCIENT_AVAILABLE

if _PRESCIENT_AVAILABLE:
    from prescient.datasets._atom3d_ppi_dataset import ATOM3DPPIDataset

from torch import Generator
from torch.utils.data import DataLoader, Sampler, Subset
from tqdm import tqdm

from lobster.transforms import Atom3DPPIToSequenceAndContactMap, PairedSequenceToTokens

from ._constants import CLM_MODEL_NAMES, ESM_MODEL_NAMES
from ._utils import load_pickle

T = TypeVar("T")


def is_null(f, t):
    return t is None


class ContactMapDataModule(LightningDataModule):
    def __init__(
        self,
        model_name: str = "esm2_t6_8M_UR50D",
        root: Union[str, Path] = None,
        *,
        cache_sequence_indicies: bool = True,
        download: bool = False,
        remove_nulls: Optional[bool] = False,
        transform_fn: Optional[
            Callable
        ] = None,  # Hydra note -- should be transforms._atom3d_ppi_transforms.PairedSequenceToTokens().transform
        target_transform_fn: Optional[Callable] = None,
        joint_transform_fn: Optional[
            Callable
        ] = None,  # Hydra note -- should be transforms._atom3d_ppi_transforms.Atom3DPPIToSequenceAndContactMap().transform
        lengths: Optional[Sequence[float]] = (
            0.05,
            0.05,
            0.90,
        ),  # TODO - set to default None?
        generator: Optional[Generator] = None,
        seed: int = 0xDEADBEEF,
        batch_size: int = 1,
        shuffle: bool = True,
        sampler: Optional[Union[Iterable, Sampler]] = None,
        batch_sampler: Optional[Union[Iterable[Sequence], Sampler[Sequence]]] = None,
        num_workers: int = 0,
        collate_fn: Optional[
            Callable[[list[T]], Any]
        ] = None,  # Hydra note -- should be data._collate.ESMBatchConverterPPI
        pin_memory: bool = True,
        drop_last: bool = False,
        null_path: str = None,
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

        self._model_name = model_name
        self._root = root
        self._cache_sequence_indicies = cache_sequence_indicies
        self._download = download
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
        self._dataset = None
        self._null_path = null_path

        # Set default transforms
        if transform_fn is None:
            self._transform_fn = PairedSequenceToTokens().transform
        if joint_transform_fn is None:
            self._joint_transform_fn = Atom3DPPIToSequenceAndContactMap().transform

    def prepare_data(self) -> None:
        # Check if a data download is needed
        dir_path = (
            Path(self._root) / "ATOM3DPPI"
        )  # directory with top-level dir. So if data at self._root/ATOM3DPPI/raw/DIPS[DB5]/data, dir_path should be self._root/ATOM3DPPI
        download_ppi = True
        if dir_path.exists() and dir_path.is_dir():
            download_ppi = False

        # Load in Dataset with a joint transform
        dataset = ATOM3DPPIDataset(
            root=self._root,
            download=download_ppi,
            joint_transform_fn=self._joint_transform_fn,
            transform_fn=self._transform_fn,
        )

        original_dataset_len = len(dataset)
        # Remove nulls
        if self._null_path:
            nulls_ix = load_pickle(self._null_path)  # TODO - allow loading from S3
            non_nulls_ix = [i for i in range(original_dataset_len) if i not in nulls_ix]
            dataset = Subset(dataset, non_nulls_ix)

        if self._remove_nulls:
            mask = [is_null(*item) for item in tqdm(dataset)]
            subset_indices = [i for i, exclude in enumerate(mask) if not exclude]
            dataset = Subset(dataset, subset_indices)

        new_dataset_len = len(dataset)

        print(
            "Using {} non-null of {} original datapoints".format(
                new_dataset_len, original_dataset_len
            )
        )

        self._dataset = dataset

    def setup(self, stage: str = "fit") -> None:  # noqa: ARG002
        """NOTE - writing v0 assuming that a transform exists mapping Atom3D atoms_neighbrs --> seq1, seq2, interactions"""
        if self._model_name not in ESM_MODEL_NAMES + CLM_MODEL_NAMES:
            raise ValueError(
                f"model_name not one of {ESM_MODEL_NAMES} or {CLM_MODEL_NAMES}"
            )

        random.seed(self._seed)
        torch.manual_seed(self._seed)

        if self._dataset is None:
            self.prepare_data()

        # Only a small subset of data is used for fitting params - Set training & val set sizes to be 20
        if self._lengths is None:
            self._train_length = 20
            self._val_length = 20
            self._test_length = (
                len(self._dataset) - self._train_length - self._val_length
            )
            self._lengths = (self._train_length, self._val_length, self._test_length)

            # Make sure batch size smaller than training, val set sizes
            assert torch.all(
                self._batch_size < torch.tensor(self._lengths)
            ), "batch size is greater than one of the dataset splits"
            # if torch.any(self._batch_size > torch.tensor(self._lengths)): self._batch_size = min(self._lengths)

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
