import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import numpy
from beignet.datasets._sized_sequence_dataset import SizedSequenceDataset
from beignet.io import ThreadSafeFile

T = TypeVar("T")


class FASTADataset(SizedSequenceDataset):
    def __init__(
        self,
        root: str | Path,
        *,
        transform: Callable | None = None,
        use_text_descriptions: bool = True,
        offsets_arr: Optional[numpy.ndarray] = None,
    ) -> None:
        if isinstance(root, str):
            root = Path(root)

        self.root = root

        self.root = self.root.resolve()

        if not self.root.exists():
            raise FileNotFoundError

        self._use_text_descriptions = use_text_descriptions

        self.data = ThreadSafeFile(self.root, open)

        if offsets_arr is None:
            offsets_path = Path(f"{self.root}.offsets.npy")
            if offsets_path.exists():
                self.offsets, sizes = numpy.load(f"{offsets_path}")
            else:
                self.offsets, sizes = self._build_index()
                numpy.save(f"{offsets_path}", numpy.stack([self.offsets, sizes]))

        else:
            self.offsets = offsets_arr[0, :]
            sizes = offsets_arr[1, :]

        self.transform = transform

        super().__init__(self.root, sizes)

    def __getitem__(self, index: int) -> tuple[str, str]:
        x = self.get(index)
        if self.transform:
            x = self.transform(x)

        return x

    def __len__(self) -> int:
        return self.offsets.size

    def get(self, index: int) -> tuple[str, str]:
        self.data.seek(self.offsets[index])

        if index == len(self) - 1:
            data = self.data.read()
        else:
            data = self.data.read(self.offsets[index + 1] - self.offsets[index])

        description, *sequence = data.split("\n")

        sequence = "".join(sequence)

        if self._use_text_descriptions:
            return sequence, description

        return sequence

    def _build_index(self) -> tuple[numpy.ndarray, numpy.ndarray]:
        return (
            numpy.fromstring(
                subprocess.check_output(
                    f"cat {self.root} | tqdm --bytes --total $(wc -c < {self.root})"
                    "| grep --byte-offset '^>' -o | cut -d: -f1",
                    shell=True,
                ),
                dtype=numpy.int64,
                sep=" ",
            ),
            numpy.fromstring(
                subprocess.check_output(
                    f"cat {self.root} | tqdm --bytes --total $(wc -c < {self.root})"
                    '| awk \'/^>/ {print "";next;} { printf("%s",$0);}\' | tail -n+2 | awk '
                    "'{print length($1)}'",
                    shell=True,
                ),
                dtype=numpy.int64,
                sep=" ",
            ),
        )
