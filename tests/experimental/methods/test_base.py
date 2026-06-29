"""Unit tests for the shared aligner contracts in :mod:`squidpy.experimental.methods._base`."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from squidpy.experimental.methods import Aligner, AlignResult, require_optional_deps


class _ShiftResult:
    """Minimal :class:`AlignResult`: a constant offset baked into ``transform``."""

    def __init__(self, delta: np.ndarray) -> None:
        self.delta = delta

    def transform(self, points: npt.ArrayLike, /) -> np.ndarray:
        return np.asarray(points, dtype=float) + self.delta


class _MeanShiftAligner(Aligner):
    """Toy aligner: shift the query centroid onto the reference centroid."""

    def align(self, ref: npt.ArrayLike, query: npt.ArrayLike) -> _ShiftResult:
        delta = np.asarray(ref, dtype=float).mean(0) - np.asarray(query, dtype=float).mean(0)
        return _ShiftResult(delta=delta)


def test_custom_aligner_round_trip() -> None:
    ref = np.array([[1.0, 1.0], [3.0, 3.0]])  # centroid (2, 2)
    query = np.array([[0.0, 0.0], [2.0, 2.0]])  # centroid (1, 1)

    result = _MeanShiftAligner().align(ref, query)

    np.testing.assert_allclose(result.delta, [1.0, 1.0])
    np.testing.assert_allclose(result.transform(query), query + 1.0)
    assert isinstance(result, AlignResult)


def test_aligner_is_abstract() -> None:
    with pytest.raises(TypeError, match="abstract"):
        Aligner()  # type: ignore[abstract]


def test_require_optional_deps_passes_when_present() -> None:
    # `numpy` is always importable in the test env; this should be a no-op.
    require_optional_deps("numpy", feature="demo")


def test_require_optional_deps_raises_for_missing_dependency() -> None:
    with pytest.raises(
        ImportError,
        match=r"'demo' requires 'squidpy_nonexistent_pkg_xyz'.*squidpy\[squidpy_nonexistent_pkg_xyz\]",
    ):
        require_optional_deps("squidpy_nonexistent_pkg_xyz", feature="demo")
