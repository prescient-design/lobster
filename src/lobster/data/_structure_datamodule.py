import os
import random
import subprocess
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence, TypeVar, Union

import numpy as np
import torch
from beignet.datasets import SizedSequenceDataset
from beignet.io._thread_safe_file import ThreadSafeFile
from lightning import LightningDataModule
from torch import Generator
from torch.utils.data import DataLoader, Sampler

from lobster.transforms import StructureFeaturizer

PathLike = Union[Path, str]
T = TypeVar("T")


class FastaStructureDataset(SizedSequenceDataset):
    """
    Base class for loading FASTA datasets and grab the associated PDB file in its header.

    Must implement how the structure features are to be loaded.

    Modified from github.com/facebookresearch/fairseq/blob/main/fairseq/data/fasta_dataset.py
    """

    def __init__(self, fasta_file: PathLike, cache_indices: bool = False, *args, **kwargs):
        self.data_file = Path(fasta_file)
        if not self.data_file.exists():
            raise FileNotFoundError
        self.file = ThreadSafeFile(fasta_file, open)
        self.cache = Path(f"{fasta_file}.idx.npy")
        if cache_indices:
            if self.cache.exists():
                self.offsets, sizes = np.load(self.cache)
            else:
                self.offsets, sizes = self._build_index()
                np.save(self.cache, np.stack([self.offsets, sizes]))
        else:
            self.offsets, sizes = self._build_index()

        super().__init__(self.data_file, sizes)

    def __len__(self):
        return self.offsets.size

    def _build_index(self):
        # Use grep and awk to get 100M/s on local SSD.
        # Should process your enormous 100G fasta in ~10 min single core...
        bytes_offsets = subprocess.check_output(
            f"cat {self.data_file} | tqdm --bytes --total $(wc -c < {self.data_file})"
            "| grep --byte-offset '^>' -o | cut -d: -f1",
            shell=True,
        )
        fasta_lengths = subprocess.check_output(
            f"cat {self.data_file} | tqdm --bytes --total $(wc -c < {self.data_file})"
            '| awk \'/^>/ {print "";next;} { printf("%s",$0);}\' | tail -n+2 | awk '
            "'{print length($1)}'",
            shell=True,
        )
        bytes_np = np.fromstring(bytes_offsets, dtype=np.int64, sep=" ")
        sizes_np = np.fromstring(fasta_lengths, dtype=np.int64, sep=" ")
        return bytes_np, sizes_np

    def get_fasta_sequence(self, idx: int):
        self.file.seek(self.offsets[idx])
        if idx == len(self) - 1:
            data = self.file.read()
        else:
            data = self.file.read(self.offsets[idx + 1] - self.offsets[idx])
        desc, *seq = data.split("\n")
        return desc[1:], "".join(seq)

    def get_structure(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError


class CATHFastaPDBDataset(FastaStructureDataset):
    """Implements structural feature loading for the CATH dataset."""

    def __init__(
        self,
        fasta_file: PathLike,
        pdb_root_dir: PathLike,
        cache_indices: bool = False,
        seq_len: int = 512,
        *args,
        **kwargs,
    ):
        super().__init__(fasta_file, cache_indices, *args, **kwargs)
        self.pdb_root_dir = Path(pdb_root_dir)
        self.structure_featurizer = StructureFeaturizer()
        self.seq_len = seq_len

    def get_structure_features(self, pdb_id):
        structure_fpath = self.pdb_root_dir / pdb_id
        with open(structure_fpath, "r") as f:
            pdb_str = f.read()
        return self.structure_featurizer(pdb_str, self.seq_len, pdb_id)

    def __getitem__(self, idx):
        header, sequence = super().get_fasta_sequence(idx)
        pdb_id = header.split("|")[2].split("/")[0]
        structure_features = self.get_structure_features(pdb_id)
        return sequence, structure_features


class PDBDataset(torch.utils.data.Dataset):
    """Parses structure PDB files into features expected to calculate frame aligned point error (FAPE) calculation."""

    def __init__(
        self,
        root: PathLike,
        max_length: int = 512,
        pdb_id_to_filename_fn: Optional[Callable[[str], str]] = lambda x: x,
    ):
        super().__init__()
        self._root = root
        self._max_length = max_length
        self._pdb_id_to_filename_fn = pdb_id_to_filename_fn

        self.structure_featurizer = StructureFeaturizer()
        self.all_ids = os.listdir(self._root)
        random.shuffle(self.all_ids)

    def get_structure_features(self, pdb_id):
        structure_fpath = Path(self._root) / self._pdb_id_to_filename_fn(pdb_id)
        with open(structure_fpath, "r") as f:
            pdb_str = f.read()
        return self.structure_featurizer(pdb_str, self._max_length, pdb_id)

    def __len__(self):
        return len(self.all_ids)

    def __getitem__(self, idx):
        pdb_id = self.all_ids[idx]
        return self.get_structure_features(pdb_id)


class PDBDataModule(LightningDataModule):
    def __init__(
        self,
        root: PathLike,
        pdb_id_to_filename_fn: Optional[Callable[[str], str]] = lambda x: x,
        lengths: Optional[Sequence[float]] = (0.9, 0.05, 0.05),
        generator: Optional[Generator] = None,
        seed: int = 0xDEADBEEF,
        batch_size: int = 1,
        shuffle: bool = True,
        sampler: Optional[Union[Iterable, Sampler]] = None,
        batch_sampler: Optional[Union[Iterable[Sequence], Sampler[Sequence]]] = None,
        num_workers: int = 0,
        collate_fn: Optional[Callable[[list[T]], Any]] = None,
        pin_memory: bool = True,
        max_length: int = 512,
        drop_last: bool = False,
        transform_fn: Optional[Callable] = None,
    ):
        super().__init__()
        if lengths is None:
            lengths = [0.4, 0.4, 0.2]

        if generator is None:
            generator = Generator().manual_seed(seed)

        self._dataset = PDBDataset(root, max_length, pdb_id_to_filename_fn)

        self._root = root
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
        self._pdb_id_to_filename_fn = pdb_id_to_filename_fn

    def setup(self, stage: str = "fit") -> None:
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
        elif stage == "predict":
            self._predict_dataset = self._dataset
        else:
            raise ValueError(f"Invalid value for stage {stage}.")

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
