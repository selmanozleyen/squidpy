"""In-memory model-fitting core for experimental methods.

:mod:`._base` holds the shared contracts -- :class:`AlignResult` (the result a
public ``align*`` function consumes) and :class:`Aligner` (the base class every
aligner subclasses). Each family subpackage (e.g. :mod:`.align_samples`,
:mod:`.align_landmarks`) holds the concrete aligners. Each subpackage stays cheap
to import -- heavy or optional dependencies (e.g. JAX) are pulled in lazily, only
when an aligner actually runs.
"""

from __future__ import annotations

from squidpy.experimental.methods._base import Aligner, AlignResult, require_optional_deps
from squidpy.experimental.methods.align_landmarks import (
    AffineAligner,
    AffineFitResult,
    SimilarityAligner,
)
from squidpy.experimental.methods.align_samples import (
    StalignAligner,
    StalignConfig,
    StalignResult,
)

__all__ = [
    "AlignResult",
    "Aligner",
    "require_optional_deps",
    "StalignAligner",
    "StalignConfig",
    "StalignResult",
    "SimilarityAligner",
    "AffineAligner",
    "AffineFitResult",
]
