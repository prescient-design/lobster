from typing import Any, Callable, Dict, Type

from ._transform import Transform


class Lambda(Transform):
    _transformed_types = (object,)

    def __init__(self, fn: Callable[[Any], Any], *types: Type):
        """
        Parameters
        ----------
        fn : Callable[[Any], Any]
            The function to be used as the transformation function.

        types : Type
            The types of the input arguments that will be passed to the
            function. If no types are provided, the default transformed types
            will be used.

        """
        super().__init__()

        self._fn = fn

        self._types = types or self._transformed_types

    def _transform(self, input: Any, parameters: Dict[str, Any]) -> Any:
        """
        Parameters
        ----------
        input : Any
            The input value to be transformed.

        parameters : Dict[str, Any]
            A dictionary containing any additional parameters required for the
            transformation.

        Returns
        -------
        Any
            The transformed value.

        """
        if isinstance(input, self._types):
            return self._fn(input)
        else:
            return input

    def extra_repr(self) -> str:
        """
        Get a string representation of the Lambda transform.

        Returns
        -------
        str
            A string representation of the Lambda transform, including the
            function name and types.

        """
        extras = []

        name = getattr(self._fn, "__name__", None)

        if name:
            extras.append(name)

        extras.append(f"types={[type.__name__ for type in self._types]}")

        return ", ".join(extras)
