"""Validation for the fitted stain reference.

The :class:`~squidpy.experimental.types.StainReference` declaration lives in
:mod:`squidpy.experimental.types` with the other public types; the rules for which
keys each method takes stay here, next to the code that produces and consumes them.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import numpy as np

# Re-exported: the declarations live with the other public types, but every stain
# module reads them from here.
from squidpy.experimental.types import StainMethod, StainReference  # noqa: F401

_DECOMPOSITION_METHODS: frozenset[str] = frozenset({"macenko", "vahadane"})
_VALID_METHODS: frozenset[str] = _DECOMPOSITION_METHODS | {"reinhard"}
_ARRAY_KEYS: tuple[str, ...] = ("stain_matrix", "mu", "sigma", "white_point", "max_concentrations")


def _coerce_finite(arr: Any, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float64)
    if out.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {out.shape}.")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} contains non-finite values.")
    return out


def validate_stain_reference(reference: StainReference | Mapping[str, Any]) -> StainReference:
    """Check a stain reference and return a normalised copy.

    Enforces the per-method key rules, coerces each array to ``float64`` (checking
    shape and finiteness), and fills the keys the method does not use with ``None``
    so consumers can subscript without ``get``. ``reference`` is not modified.
    """
    unknown = set(reference) - {"method", *_ARRAY_KEYS}
    if unknown:
        raise ValueError(f"Unknown reference key(s): {sorted(unknown)}; expected from {sorted(_ARRAY_KEYS)}.")
    method = reference.get("method")
    if method not in _VALID_METHODS:
        raise ValueError(f"Unknown method {method!r}; expected one of {sorted(_VALID_METHODS)}.")

    out: dict[str, Any] = {"method": method, **{key: reference.get(key) for key in _ARRAY_KEYS}}
    if method in _DECOMPOSITION_METHODS:
        if out["stain_matrix"] is None:
            raise ValueError(f"method={method!r} requires stain_matrix.")
        if out["mu"] is not None or out["sigma"] is not None:
            raise ValueError(f"method={method!r} forbids mu/sigma; pass them only for Reinhard.")
        if out["white_point"] is None:
            raise ValueError(f"method={method!r} requires white_point.")
        out["stain_matrix"] = _coerce_finite(out["stain_matrix"], shape=(3, 3), name="stain_matrix")
        bg = _coerce_finite(out["white_point"], shape=(3,), name="white_point")
        if np.any(bg <= 0):
            raise ValueError("white_point must be strictly positive.")
        out["white_point"] = bg
        if out["max_concentrations"] is not None:
            maxc = _coerce_finite(out["max_concentrations"], shape=(2,), name="max_concentrations")
            if np.any(maxc <= 0):
                raise ValueError("max_concentrations must be strictly positive.")
            out["max_concentrations"] = maxc
    else:
        if out["mu"] is None or out["sigma"] is None:
            raise ValueError("method='reinhard' requires both mu and sigma.")
        if out["stain_matrix"] is not None:
            raise ValueError("method='reinhard' forbids stain_matrix.")
        if out["white_point"] is not None:
            raise ValueError(
                "method='reinhard' forbids white_point; Reinhard's color "
                "transfer is in Ruderman Lab and does not use a white point."
            )
        if out["max_concentrations"] is not None:
            raise ValueError("method='reinhard' forbids max_concentrations.")
        out["mu"] = _coerce_finite(out["mu"], shape=(3,), name="mu")
        out["sigma"] = _coerce_finite(out["sigma"], shape=(3,), name="sigma")
        if np.any(out["sigma"] <= 0):
            raise ValueError("sigma must be strictly positive.")
    return cast("StainReference", out)
