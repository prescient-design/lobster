from __future__ import annotations

from types import ModuleType
from typing import (
    Any,
    Callable,
    Dict,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import torch
from torch import Tensor
from torch._C import DisableTorchFunctionSubclass
from torch.types import _device, _dtype, _size

F = TypeVar("F", bound="Feature")


class Feature(Tensor):
    """
    Feature
    """

    __f: Optional[ModuleType] = None

    @staticmethod
    def _to_tensor(
        data: Any,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[torch.device, str, int]] = None,
        requires_grad: Optional[bool] = None,
    ) -> Tensor:
        if requires_grad is None:
            if isinstance(data, Tensor):
                requires_grad = data.requires_grad
            else:
                requires_grad = False

        tensor = torch.as_tensor(data, dtype=dtype, device=device)

        return tensor.requires_grad_(requires_grad)

    @classmethod
    def wrap_like(cls: Type[F], other: F, tensor: Tensor) -> F:
        raise NotImplementedError

    # NOTE:
    #   We don’t need to wrap the output of ``tensor.requires_grad_``,
    #   because it’s an inplace operation and automatically retains its type.
    _NO_WRAPPING_EXCEPTIONS = {
        Tensor.clone: lambda cls, input, output: cls.wrap_like(input, output),
        Tensor.detach: lambda cls, input, output: cls.wrap_like(input, output),
        Tensor.requires_grad_: lambda cls, input, output: output,
        Tensor.to: lambda cls, input, output: cls.wrap_like(input, output),
    }

    @classmethod
    def __torch_function__(
        cls,
        func: Callable[..., Tensor],
        types: Tuple[Type[Tensor], ...],
        args: Sequence[Any] = (),
        kwargs: Optional[Mapping[str, Any]] = None,
    ) -> Tensor:
        """
        The default behavior of :class:`~Tensor`’s retains the custom tensor
        type. For :class:`Feature`, this creates two problems:

            1.  :class:`Feature` may require metadata and the default wrapping,
                i.e., ``return cls(func(*args, **kwargs))``, will fail.

            2.  For most operations, there is no way of knowing if an input
                type is still valid for the output type.

        To address these two problems, :class:`Feature` disables automatic
        output wrapping for most operators. The exceptions are available from
        :attr:`Feature._NO_WRAPPING_EXCEPTIONS`
        """
        # NOTE:
        #   ``super().__torch_function__`` has no hook to prevent the
        #   coercing of the output type into the input type so this
        #   functionality is reimplemented.
        if not all(issubclass(cls, t) for t in types):
            return NotImplemented

        with DisableTorchFunctionSubclass():
            output = func(*args, **kwargs or dict())

            wrapper = cls._NO_WRAPPING_EXCEPTIONS.get(func)

            # NOTE:
            #   Besides ``func`` being an exception, this class method
            #   requires the input operand, i.e. ``args[0]``, to be an
            #   instance of the class that ``__torch_function__`` invoked.
            #   The ``__torch_function__`` protocol will invoke this method
            #   on each type in the computation by walking the method
            #   resolution  order (MRO). For example,
            #   ``Tensor(...).to(yeji.features.Foo( ...))`` invokes
            #   ``yeji.features.Foo.__torch_function__`` with
            #   ``args = (Tensor(), yeji.featues.Foo())``.
            #   Without this guard, ``Tensor`` would be wrapped into
            #   ``yeji.features.Foo``.
            if wrapper and isinstance(args[0], cls):
                return wrapper(cls, args[0], output)

            # NOTE:
            #   Because inplace ``func``’s, canonically identified with a
            #   trailing underscore in their name, e.g., ``.add_(...)``,
            #   retain their input type, they need to be unwrapped.
            if isinstance(output, cls):
                return output.as_subclass(Tensor)

            return output

    def _make_repr(self, **kwargs: Any) -> str:
        items = []

        for key, value in kwargs.items():
            items.append(f"{key}={value}")

        return f"{super().__repr__()[:-1]}, {', '.join(items)})"

    @property
    def _f(self) -> ModuleType:
        # NOTE:
        #   Lazy import of ``yeji.transforms.functional`` to bypass the
        #   ``ImportError`` raised by the circual import. The
        #   ``yeji.transforms.functional`` import is deferred until the
        #   functional module is referenced and it’s shared across all
        #   instances of the class.
        if Feature.__f is None:
            import lobster.transforms.functional

            Feature.__f = lobster.transforms.functional

        return Feature.__f

    @property
    def device(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> _device:  # type: ignore[override]
        with DisableTorchFunctionSubclass():
            return super().device

    @property
    def ndim(self) -> int:  # type: ignore[override]
        with DisableTorchFunctionSubclass():
            return super().ndim

    @property
    def dtype(self) -> _dtype:  # type: ignore[override]
        with DisableTorchFunctionSubclass():
            return super().dtype

    @property
    def shape(self) -> _size:  # type: ignore[override]
        with DisableTorchFunctionSubclass():
            return super().shape

    def __deepcopy__(self: F, memo: Dict[int, Any]) -> F:
        # NOTE:
        #   Detach, because ``deepcopy(Tensor)``, unlike ``Tensor.clone``,
        #   isn’t be added to the computational graph.

        # NOTE:
        #   Because a side-effect of detaching is clearing
        #   ``Tensor.requires_grad``, it’s refilled before returning.

        # NOTE:
        #   Deep-copying of metadata isn’t explicitly handled.
        return (
            self.detach()
            .clone()
            .requires_grad_(
                self.requires_grad,
            )
        )  # type: ignore[return-value]


_InputType = Union[
    Dict[str, Tensor],
    Feature,
    Sequence[Tensor],
    Tensor,
    str,
]

_InputTypeJIT = Union[Dict[str, Tensor], Sequence[Tensor], Tensor]
