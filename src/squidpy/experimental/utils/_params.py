"""Shared internal helper for resolving params-dataclass arguments.

Not part of the public API - symbols here are private and may change
without notice.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import fields
from typing import Any, cast


def resolve_params[T](value: T | Mapping[str, Any] | None, cls: type[T], *, label: str) -> T:
    """Normalise a params argument (``None`` / instance / ``Mapping``) to a ``cls`` instance.

    Parameters
    ----------
    value
        ``None`` (use defaults), an instance of ``cls`` (passed through by
        identity), or a ``Mapping`` of field names to values.
    cls
        The params dataclass to construct.
    label
        The user-facing argument name used verbatim in error messages.  Include
        backticks if the caller's convention uses them (e.g. ``"`tiling_qc_params`"``).
    """
    if value is None:
        return cls()
    if isinstance(value, cls):
        return value
    if isinstance(value, Mapping):
        valid = {f.name for f in fields(cls)}
        unknown = set(value) - valid
        if unknown:
            raise ValueError(f"Unknown {label} field(s): {sorted(unknown)}; expected from {sorted(valid)}.")
        return cls(**value)
    raise TypeError(f"{label} must be {cls.__name__}, Mapping, or None; got {type(value).__name__}.")


def resolve_typed_params[T: Mapping[str, Any]](
    params: T | Mapping[str, Any] | None,
    *,
    defaults: T,
    validate: Callable[[dict[str, Any]], None] | None = None,
    arg_name: str = "method_params",
) -> T:
    """Merge a params mapping over ``defaults`` and validate the result.

    ``T`` is a :class:`~typing.TypedDict`, so callers get static key and value
    checking at the call site; this function is the dynamic half. Unknown keys
    are named rather than silently ignored (a plain ``dict`` would accept them),
    and ``validate`` -- which coerces in place and range-checks -- runs on the
    *merged* mapping, so the defaults are checked on every call rather than
    trusted.

    Returns a new mapping; neither ``params`` nor ``defaults`` is mutated.
    """
    if params is not None and not isinstance(params, Mapping):
        raise TypeError(f"`{arg_name}` must be a Mapping or None; got {type(params).__name__}.")
    if params:
        unknown = set(params) - set(defaults)
        if unknown:
            raise ValueError(
                f"Unknown `{arg_name}` field(s): {sorted(unknown)}; expected from {sorted(defaults)}."
            )
    merged = {**defaults, **(params or {})}
    if validate is not None:
        validate(merged)
    return cast("T", merged)
