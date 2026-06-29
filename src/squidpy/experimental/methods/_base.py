"""Shared contracts for the in-memory alignment core.

Holds the public result contract (:class:`AlignResult`), the base class every
aligner subclasses (:class:`Aligner`), and the optional-dependency guard
(:func:`require_optional_deps`). The concrete aligners live in the family
subpackages (:mod:`.align_samples`, :mod:`.align_landmarks`); none of them sees a
container -- they operate on plain ``(x, y)`` arrays.

See the :doc:`/extensibility` guide for how to implement a custom aligner.
"""

from __future__ import annotations

import importlib.util
from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

import numpy.typing as npt

from squidpy._utils import NDArrayA

__all__ = [
    "AlignResult",
    "Aligner",
    "require_optional_deps",
]


@runtime_checkable
class AlignResult(Protocol):
    """A fitted alignment that maps ``(N, 2)`` ``(x, y)`` points into the reference frame.

    This is the only thing the public ``align*`` functions require of an
    aligner's result, so ``output_mode="object"`` is agnostic to the method that
    produced it.
    """

    def transform(self, points: npt.ArrayLike, /) -> NDArrayA:
        """Map an ``(N, 2)`` ``(x, y)`` array into the reference frame."""
        ...


class Aligner(ABC):
    """Base class for alignment strategies.

    A custom aligner only has to implement :meth:`align`: take a reference and a
    query (point clouds for sample alignment, paired landmarks for landmark
    alignment), and return an :class:`AlignResult` -- something whose
    ``transform`` maps ``(x, y)`` points into the reference frame. Pass the
    instance to :func:`~squidpy.experimental.tl.align_from_aligner` (or
    :func:`~squidpy.experimental.tl.align_landmarks_from_aligner`) to run it
    against a container.

    See Also
    --------
    StalignAligner : JAX LDDMM point-cloud aligner.
    SimilarityAligner : Closed-form similarity landmark aligner.
    AffineAligner : Closed-form affine landmark aligner.
    """

    @abstractmethod
    def align(self, ref: npt.ArrayLike, query: npt.ArrayLike) -> AlignResult:
        """Fit the alignment mapping ``query`` onto ``ref`` and return the result."""


def require_optional_deps(*packages: str, feature: str) -> None:
    """Raise a helpful :class:`ImportError` if any optional ``packages`` are missing.

    ``feature`` names the thing the caller asked for (e.g. ``"stalign"``) so the
    error points at both the missing package and the ``squidpy`` extra to install.
    """
    missing = [pkg for pkg in packages if importlib.util.find_spec(pkg) is None]
    if not missing:
        return
    verb = "is" if len(missing) == 1 else "are"
    names = ", ".join(repr(p) for p in missing)
    extras = ",".join(missing)
    raise ImportError(
        f'{feature!r} requires {names}, which {verb} not installed. Install with `pip install "squidpy[{extras}]"`.'
    )
