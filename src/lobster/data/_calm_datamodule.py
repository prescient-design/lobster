import importlib
import random
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence, TypeVar, Union

import torch.utils.data
from lightning import LightningDataModule
from torch import Generator
from torch.utils.data import DataLoader, Sampler

from lobster.datasets._calm_dataset import CalmDataset
from lobster.tokenization import PmlmTokenizerTransform
from lobster.transforms import Transform

T = TypeVar("T")


class CalmLightningDataModule(LightningDataModule):
    def __init__(
        self,
        root: Union[str, Path] = None,
        *,
        cache_sequence_indicies: bool = True,
        download: bool = False,
        transform_fn: Union[Callable, Transform, None] = None,
        lengths: Optional[Sequence[float]] = (0.9, 0.05, 0.05),
        generator: Optional[Generator] = None,
        seed: int = 0xDEADBEEF,
        batch_size: int = 1,
        shuffle: bool = True,
        sampler: Optional[Union[Iterable, Sampler]] = None,
        batch_sampler: Optional[Union[Iterable[Sequence], Sampler[Sequence]]] = None,
        num_workers: int = 0,
        collate_fn: Optional[Callable[[list[T]], Any]] = None,
        max_length: int = 512,
        pin_memory: bool = True,
        drop_last: bool = False,
        train: bool = True,
        is_relative_model: bool = False,
        tokenizer_dir: Optional[str] = "cdna_tokenizer",
        mlm: bool = True,
    ) -> None:
        """
        :param model_name: name of esm model

        :param root: Root directory where the dataset subdirectory exists or,
            if :attr:`download` is ``True``, the directory where the dataset
            subdirectory will be created and the dataset downloaded.

        :param cache_sequence_indicies: If ``True``, caches the sequence
            indicies to disk for faster re-initialization (default: ``True``).

        :param download: If ``True``, download the dataset and to the
            :attr:`root` directory (default: ``False``). If the dataset is
            already downloaded, it is not redownloaded.

        :param use_transform_fn: If ``True``, use transform_fn for dataset
            tokenization, else no transform.

        :param lengths: Fractions of splits to generate.

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


        :param is_relative_model: If ``True``, assumes training between two sequences
            and calls a relative representation data loader

        :param tokenizer_dir: a tokenizer saved to src/lobster/assets.
            default pmlm_tokenizer is compatible with esm2 models
        """
        super().__init__()

        if lengths is None:
            lengths = [0.4, 0.4, 0.2]

        if generator is None:
            generator = Generator().manual_seed(seed)

        self._root = root
        self._cache_sequence_indicies = cache_sequence_indicies
        self._download = download
        self._lengths = lengths
        self._generator = generator
        self._seed = seed
        self._batch_size = batch_size
        self._max_length = max_length
        self._shuffle = shuffle
        self._sampler = sampler
        self._batch_sampler = batch_sampler
        self._num_workers = num_workers
        self._collate_fn = collate_fn
        self._pin_memory = pin_memory
        self._drop_last = drop_last
        self._is_relative_model = is_relative_model
        self._tokenizer_dir = tokenizer_dir
        self._mlm = mlm
        self._train = train
        self._dataset = None

        path = importlib.resources.files("lobster") / "assets" / self._tokenizer_dir
        self._transform_fn = transform_fn or PmlmTokenizerTransform(
            path,
            padding="max_length",
            truncation=True,
            max_length=self._max_length,
            mlm=self._mlm,
        )

    def prepare_data(self) -> None:
        # Load in Dataset, transform sequences
        dataset = CalmDataset(
            root=self._root,
            cache_sequence_indicies=True,
            download=self._download,
            train=self._train,
            transform_fn=self._transform_fn,
        )

        return dataset

    def setup(self, stage: str = "fit") -> None:  # noqa: ARG002
        super().__init__()
        random.seed(self._seed)
        torch.manual_seed(self._seed)

        if self._dataset is None:
            self._dataset = self.prepare_data()

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
