"""Spatial tools general utility functions."""

from __future__ import annotations

import functools
import inspect
import warnings
from collections.abc import Callable, Generator, Hashable, Iterable
from contextlib import contextmanager
from typing import Any, Literal

import numpy as np
import xarray as xr
from spatialdata.models import Image2DModel, Labels2DModel

__all__ = ["singledispatchmethod", "NDArray", "NDArrayA"]


try:
    from functools import singledispatchmethod
except ImportError:
    from functools import singledispatch, update_wrapper

    def singledispatchmethod(func: Callable[..., Any]) -> Callable[..., Any]:  # type: ignore[no-redef]
        """Backport of `singledispatchmethod` for < Python 3.8."""
        dispatcher = singledispatch(func)

        def wrapper(*args: Any, **kw: Any) -> Any:
            return dispatcher.dispatch(args[1].__class__)(*args, **kw)

        wrapper.register = dispatcher.register  # type: ignore[attr-defined]
        update_wrapper(wrapper, func)

        return wrapper


from numpy.typing import NDArray

NDArrayA = NDArray[Any]


def _unique_order_preserving(
    iterable: Iterable[Hashable],
) -> tuple[list[Hashable], set[Hashable]]:
    """Remove items from an iterable while preserving the order."""
    seen: set[Hashable] = set()
    seen_add = seen.add
    return [i for i in iterable if not (i in seen or seen_add(i))], seen


@contextmanager
def verbosity(level: int) -> Generator[None, None, None]:
    """
    Temporarily set the verbosity level of :doc:`scanpy <scanpy:index>`.

    Parameters
    ----------
    level
        The new verbosity level.

    Returns
    -------
    Nothing.
    """
    import scanpy as sc

    verbosity = sc.settings.verbosity
    sc.settings.verbosity = level
    try:
        yield
    finally:
        sc.settings.verbosity = verbosity


string_types = (bytes, str)


def deprecated(reason: str) -> Any:
    """
    This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used.
    """

    if isinstance(reason, string_types):
        # The @deprecated is used with a 'reason'.
        #
        # .. code-block:: python
        #
        #    @deprecated("please, use another function")
        #    def old_function(x, y):
        #      pass

        def decorator(func1: Callable[..., Any]) -> Callable[..., Any]:
            if inspect.isclass(func1):
                fmt1 = "Call to deprecated class {name} ({reason})."
            else:
                fmt1 = "Call to deprecated function {name} ({reason})."

            @functools.wraps(func1)
            def new_func1(*args: Any, **kwargs: Any) -> Any:
                warnings.simplefilter("always", DeprecationWarning)
                warnings.warn(
                    fmt1.format(name=func1.__name__, reason=reason),
                    category=DeprecationWarning,
                    stacklevel=2,
                )
                warnings.simplefilter("default", DeprecationWarning)
                return func1(*args, **kwargs)

            return new_func1

        return decorator

    elif inspect.isclass(reason) or inspect.isfunction(reason):
        # The @deprecated is used without any 'reason'.
        #
        # .. code-block:: python
        #
        #    @deprecated
        #    def old_function(x, y):
        #      pass

        func2 = reason

        if inspect.isclass(func2):
            fmt2 = "Call to deprecated class {name}."
        else:
            fmt2 = "Call to deprecated function {name}."

        @functools.wraps(func2)
        def new_func2(*args: Any, **kwargs: Any) -> Any:
            warnings.simplefilter("always", DeprecationWarning)
            warnings.warn(
                fmt2.format(name=func2.__name__),
                category=DeprecationWarning,
                stacklevel=2,
            )
            warnings.simplefilter("default", DeprecationWarning)
            return func2(*args, **kwargs)

        return new_func2

    else:
        raise TypeError(repr(type(reason)))


def _get_scale_factors(
    element: Image2DModel | Labels2DModel,
) -> list[float]:
    """
    Get the scale factors of an image or labels.
    """
    if not hasattr(element, "keys"):
        return []  # element isn't a datatree -> single scale

    shapes = [_yx_from_shape(element[scale].image.shape) for scale in element.keys()]

    factors: list[float] = [(y0 / y1 + x0 / x1) / 2 for (y0, x0), (y1, x1) in zip(shapes, shapes[1:], strict=False)]
    return [int(f) for f in factors]


def _yx_from_shape(shape: tuple[int, ...]) -> tuple[int, int]:
    if len(shape) == 2:  # (y, x)
        return shape[0], shape[1]
    if len(shape) == 3:  # (c, y, x)
        return shape[1], shape[2]

    raise ValueError(f"Unsupported shape {shape}. Expected (y, x) or (c, y, x).")


def _ensure_dim_order(img_da: xr.DataArray, order: Literal["cyx", "yxc"] = "yxc") -> xr.DataArray:
    """
    Ensure dims are in the requested order and that a 'c' dim exists.
    Only supports images with dims subset of {'y','x','c'}.
    """
    dims = list(img_da.dims)
    if "y" not in dims or "x" not in dims:
        raise ValueError(f'Expected dims to include "y" and "x". Found dims={dims}')
    if "c" not in dims:
        img_da = img_da.expand_dims({"c": [0]})
    # After possible expand, just transpose to target
    return img_da.transpose(*tuple(order))
