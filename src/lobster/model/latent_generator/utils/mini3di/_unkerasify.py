"""Parser for extracting weights from Keras files.

Adapted from `moof2k/kerasify <https://github.com/moof2k/kerasify>`_.

"""

from __future__ import annotations

import enum
import itertools
import struct
import typing
from typing import BinaryIO

import numpy

from lobster.model.latent_generator.utils.mini3di._layers import DenseLayer, Layer


class LayerType(enum.IntEnum):
    DENSE = 1
    CONVOLUTION2D = 2
    FLATTEN = 3
    ELU = 4
    ACTIVATION = 5
    MAXPOOLING2D = 6
    LSTM = 7
    EMBEDDING = 8


class ActivationType(enum.IntEnum):
    LINEAR = 1
    RELU = 2
    SOFTPLUS = 3
    SIGMOID = 4
    TANH = 5
    HARD_SIGMOID = 6


class KerasifyParser:
    """An incomplete parser for model files serialized with `kerasify`.

    Note:
        Only dense layers are supported, since the ``foldseek`` VQ-VAE model
        is only using 3 dense layers.

    """

    def __init__(self, file: typing.BinaryIO) -> None:
        self.file = file
        self.buffer = bytearray(1024)
        (self.n_layers,) = self._get("I")

    def __iter__(self) -> KerasifyParser:
        return self

    def __next__(self) -> Layer:
        layer = self.read()
        if layer is None:
            raise StopIteration
        return layer

    def _read(self, format: str) -> memoryview:
        n = struct.calcsize(format)
        if len(self.buffer) < n:
            self.buffer.extend(itertools.islice(itertools.repeat(0), n - len(self.buffer)))
        v = memoryview(self.buffer)[:n]
        self.file.readinto(v)  # type: ignore
        return v

    def _get(self, format: str) -> tuple[typing.Any, ...]:
        v = self._read(format)
        return struct.unpack(format, v)

    def read(self) -> Layer | None:
        if self.n_layers == 0:
            return None

        self.n_layers -= 1
        layer_type = LayerType(self._get("I")[0])
        if layer_type == LayerType.DENSE:
            (w0,) = self._get("I")
            (w1,) = self._get("I")
            (b0,) = self._get("I")
            weights = numpy.frombuffer(self._read(f"={w0 * w1}f"), dtype="f4").reshape(w0, w1).copy()
            biases = numpy.frombuffer(self._read(f"={b0}f"), dtype="f4").copy()
            activation = ActivationType(self._get("I")[0])
            if activation not in (ActivationType.LINEAR, ActivationType.RELU):
                raise NotImplementedError(f"Unsupported activation type: {activation!r}")
            return DenseLayer(weights, biases, activation == ActivationType.RELU)
        else:
            raise NotImplementedError(f"Unsupported layer type: {layer_type!r}")


def load(fh: BinaryIO) -> list[Layer]:
    return list(KerasifyParser(fh))
