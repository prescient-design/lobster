"""A mini implementation of the neural network layers used in ``foldseek``."""

from __future__ import annotations

import abc
import functools
import typing
from collections.abc import Iterable

import numpy

from lobster.model.latent_generator.utils.mini3di._utils import relu

if typing.TYPE_CHECKING:
    from ._utils import ArrayN, ArrayNxM


class Layer(abc.ABC):
    @abc.abstractmethod
    def __call__(self, X: ArrayNxM[numpy.floating]) -> ArrayNxM[numpy.floating]:
        raise NotImplementedError


class DenseLayer(Layer):
    def __init__(
        self,
        weights: ArrayNxM[numpy.floating],
        biases: ArrayN[numpy.floating] | None = None,
        activation: bool = True,
    ):
        self.activation = activation
        self.weights = numpy.asarray(weights)
        if biases is None:
            self.biases = numpy.zeros(self.weights.shape[1])
        else:
            self.biases = numpy.asarray(biases)

    def __call__(self, X: ArrayNxM[numpy.floating]) -> ArrayNxM[numpy.floating]:
        _X = numpy.asarray(X)
        out = _X @ self.weights
        out += self.biases

        if self.activation:
            return relu(out, out=out)
        else:
            return out


class CentroidLayer(Layer):
    def __init__(self, centroids: ArrayNxM[numpy.floating]) -> None:
        self.centroids = numpy.asarray(centroids)
        self.r2 = numpy.sum(self.centroids**2, axis=1).reshape(-1, 1).T

    def __call__(self, X: ArrayNxM[numpy.floating]) -> ArrayN[numpy.uint8]:
        # compute pairwise squared distance matrix
        r1 = numpy.sum(X**2, axis=1).reshape(-1, 1)
        D = r1 - 2 * X @ self.centroids.T + self.r2
        # find closest centroid
        states = numpy.empty(D.shape[0], dtype=numpy.uint8)
        D.argmin(axis=1, out=states)
        return states


class Model:
    def __init__(self, layers: Iterable[Layer] = ()):
        self.layers = list(layers)

    def __call__(self, X: ArrayNxM[numpy.floating]) -> ArrayNxM[numpy.floating]:
        return functools.reduce(lambda x, f: f(x), self.layers, X)
