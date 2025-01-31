from typing import Any, Callable, Iterable, Optional, Sequence, TypeVar, Union

from beignet.datasets import ChEMBLDataset
from lightning import LightningDataModule
from lightning.pytorch.utilities import CombinedLoader
from torch import Generator
from torch.utils.data import ConcatDataset, DataLoader, Sampler

from lobster.datasets import M320MDataset

T = TypeVar("T")


class UmeLightningDataModule(LightningDataModule):
    def __init__(
        self,
        *,
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
    ) -> None:
        super().__init__()

        if lengths is None:
            lengths = [0.4, 0.4, 0.2]

        if generator is None:
            generator = Generator().manual_seed(seed)

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
        self._train = train
        self._dataset = None

    def prepare_data(self) -> None:
        self.smiles_dataset = ConcatDataset(
            [
                M320MDataset(
                    root=self._root,
                    download=self._download,
                    use_text_descriptions=False,
                ),
                ChEMBLDataset(
                    root=self._root,
                    download=self._download,
                ),
            ]
        )
        # TODO: Dummy datasets for now
        self.amino_acid_dataset = ConcatDataset(
            [
                M320MDataset(
                    root=self._root,
                    download=self._download,
                    use_text_descriptions=False,
                ),
                ChEMBLDataset(
                    root=self._root,
                    download=self._download,
                ),
            ]
        )
        # nucleotide_dataset = ConcatDataset([
        #     PlaceholderDataset(),
        #     PlaceholderDataset()
        # ])

    def setup(self, stage: str = "fit") -> None:
        # TODO: handle splits
        pass

    def train_dataloader(self) -> DataLoader:
        dataloaders = {
            "smiles": DataLoader(
                self._train_dataset,
                batch_size=self._batch_size,
                shuffle=self._shuffle,
                sampler=self._sampler,
                num_workers=self._num_workers,
                collate_fn=self._collate_fn,
                pin_memory=self._pin_memory,
            ),
            "amino_acid": DataLoader(
                self._train_dataset,
                batch_size=self._batch_size,
                shuffle=self._shuffle,
                sampler=self._sampler,
                num_workers=self._num_workers,
                collate_fn=self._collate_fn,
                pin_memory=self._pin_memory,
            ),
        }

        return CombinedLoader(dataloaders, "max_size_cycle")
