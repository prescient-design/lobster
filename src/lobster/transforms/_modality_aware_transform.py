from collections.abc import Callable, Sequence
from typing import Any

from lobster.constants import Modality


class ModalityAwareTransform:
    """A wrapper that makes any transform function modality-aware.
    This class is necessary because the UMEStreamingDataset requires transform functions
    to specify their input and output modalities when handling multiple sequences. Without
    this information, the dataset cannot properly handle sequences of different modalities
    in the same batch.
    The class wraps any transform function and adds the required modality information
    as attributes, making it compatible with the dataset's requirements.
    Parameters
    ----------
    transform_fn : Callable
        The transform function to wrap
    input_modality : Modality
        The input modality of the transform
    output_modalities : Sequence[Modality] | None, optional
        The output modalities. If None, assumes same as input.
    """

    def __init__(
        self, transform_fn: Callable, input_modality: Modality, output_modalities: Sequence[Modality] | None = None
    ):
        self.transform_fn = transform_fn
        self.input_modality = input_modality
        self.output_modalities = tuple(output_modalities) if output_modalities is not None else (input_modality,)

    def __call__(self, x: Any) -> Any:
        return self.transform_fn(x)


class ComposedModalityAwareTransform(ModalityAwareTransform):
    """A transform that composes multiple modality-aware transforms.
    This class is necessary because simple function composition (e.g., using lambda)
    loses the modality information that UMEStreamingDataset requires. When transforms
    are composed using lambda functions, the resulting function doesn't have the
    required input_modality and output_modalities attributes.
    ComposedModalityAwareTransform properly maintains modality information through the composition
    chain, ensuring that the dataset can still access the necessary modality
    information even when multiple transforms are applied.
    Parameters
    ----------
    *transforms : ModalityAwareTransform
        The transforms to compose, in order of application
    Raises
    ------
    ValueError
        If no transforms are provided
    Examples
    --------
    >>> sanitize = ModalityAwareTransform(
    ...     lambda x: x.replace("|", "."),
    ...     input_modality=Modality.AMINO_ACID
    ... )
    >>> clean = ModalityAwareTransform(
    ...     lambda x: x.strip(),
    ...     input_modality=Modality.AMINO_ACID
    ... )
    >>> composed = ComposedModalityAwareTransform(sanitize, clean)
    >>> composed.input_modality  # Modality.AMINO_ACID
    >>> composed.output_modalities  # (Modality.AMINO_ACID,)
    """

    def __init__(self, *transforms: ModalityAwareTransform):
        if not transforms:
            raise ValueError("At least one transform must be provided")

        # The input modality of the first transform
        self.input_modality = transforms[0].input_modality

        # The output modalities of the last transform
        self.output_modalities = tuple(transforms[-1].output_modalities)

        self.transforms = transforms

    def __call__(self, x: Any) -> Any:
        result = x
        for transform in self.transforms:
            result = transform(result)
        return result
