from enum import Enum
from typing import Any, Callable, Dict, List, Tuple, Type, Union

import torch.utils._pytree
from torch import Tensor
from torch.nn import Module

from lobster.features import Feature


class Transform(Module):
    # Class attribute defining transformed types. Other types are
    # passed-through without any transformation
    #
    # We support both Types and callables that are able to do further checks
    # on the type of the input.
    _transformed_types: Tuple[Union[Type, Callable[[Any], bool]], ...] = (
        Tensor,
        str,
    )

    def __init__(self):
        super().__init__()

    def _check_inputs(self, inputs: List[Any]):
        """
        Parameters
        ----------
        inputs : List[Any]
            The inputs to be checked.
        """
        raise NotImplementedError

    def _get_params(self, inputs: List[Any]) -> Dict[str, Any]:
        """
        Parameters
        ----------
        inputs: List[Any]
            The list of input objects.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the parameters of the method.
        """
        return dict()

    def _transform(self, input: Any, parameters: Dict[str, Any]) -> Any:
        """
        Parameters
        ----------
        input : Any
            The input data to be transformed.

        parameters : Dict[str, Any]
            A dictionary containing the parameters for the transformation.

        Returns
        -------
        Any
            The transformed output data.

        Note
        ----
        This method is expected to be implemented by a subclass of the
        ``Transform`` class. It raises a ``NotImplementedError`` to indicate
        that the subclass must provide its own implementation of the
        ``_transform`` method.
        """
        raise NotImplementedError

    def forward(self, *inputs: Any) -> Any:
        """
        Parameters
        ----------
        inputs : tuple
            The input tensors to be transformed. Can have multiple tensors.

        Returns
        -------
        tensor
            The transformed input tensors.
        """
        if len(inputs) > 1:
            flattened, spec = torch.utils._pytree.tree_flatten(inputs)
        else:
            flattened, spec = torch.utils._pytree.tree_flatten(inputs[0])

        self._check_inputs(flattened)

        transformables = self._transformables(flattened)

        inputs = []

        for x, transformable in zip(flattened, transformables):
            if transformable:
                inputs = [*inputs, x]

        inputs = self._get_params(inputs)

        ys = []

        for x, transformable in zip(flattened, transformables):
            if transformable:
                y = self._transform(x, inputs)
            else:
                y = x

            ys = [*ys, y]

        return torch.utils._pytree.tree_unflatten(ys, spec)

    def _transformables(self, inputs: List[Any]) -> List[bool]:
        """
        Parameters
        ----------
        inputs : List[Any]
            List of input objects to be checked for transformation.

        Returns
        -------
        transformables : List[bool]
            List indicating whether each input object is transformable or not.
            ``True`` if an object can be transformed, ``False`` otherwise.
        """
        # NOTE:
        #   Heuristic for transforming anonymous tensor inputs:
        #
        #       1.  Anonymous tensors, i.e., non-:class:`Feature` tensors,
        #           are passed through if there is an explicit feature in the
        #           sample.
        #
        #       2.  If there is no explicit feature the sample, only the first
        #           encountered anonymous tensor is transformed, while the rest
        #           are passed. The order is defined by the returned
        #           `flat_inputs` of `tree_flatten`, which recurses
        #           depth-first through the input.
        #
        #   The heuristic should work well for most people in practice. The
        #   only case where it doesn't is if someone tries to transform
        #   multiple anonymous features at the same time, expecting them all
        #   to be treated as named features.
        transformables = []

        transform_anonymous_feature = False

        for input in inputs:
            for t in [str]:
                if isinstance(input, t) if isinstance(t, type) else t(input):
                    transform_anonymous_feature = True

        for input in inputs:
            transformable = True

            checked = False

            for t in self._transformed_types:
                if isinstance(input, t) if isinstance(t, type) else t(input):
                    checked = True

            if not checked:
                transformable = False
            elif isinstance(input, Tensor) and not isinstance(input, Feature):
                if transform_anonymous_feature:
                    transform_anonymous_feature = False
                else:
                    transformable = False

            transformables = [*transformables, transformable]

        return transformables

    def extra_repr(self) -> str:
        """
        Returns
        -------
        str
            A string representation of the extra configuration attributes of
            the Transform module.
        """
        extra = []

        for name, value in self.__dict__.items():
            if name.startswith("_") or name == "training":
                continue

            if not isinstance(
                value,
                (Enum, bool, float, int, list, str, tuple),
            ):
                continue

            extra.append(f"{name}={value}")

        return ", ".join(extra)
