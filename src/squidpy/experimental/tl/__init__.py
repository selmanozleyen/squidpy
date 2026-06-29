from __future__ import annotations

# `AlignResult` is the only result type on the public surface: it is the aligner
# contract (a `transform` mapping points into the reference frame) and the declared
# return of the `align*` functions. `Aligner` is the base class custom aligners
# subclass. The concrete results (`StalignResult`, `AffineFitResult`) stay in their
# home modules under `squidpy.experimental.methods` for callers that need raw fields
# -- the public API stays method-agnostic.
from squidpy.experimental.methods import Aligner, AlignResult

from ._align import (
    align_from_aligner,
    align_landmarks_affine,
    align_landmarks_from_aligner,
    align_landmarks_similarity,
    align_stalign,
)
from ._tiling_qc import TilingQCParams, calculate_tiling_qc
from ._tiling_stitch import StitchParams, assign_stitch_groups

__all__ = [
    "align_stalign",
    "align_from_aligner",
    "align_landmarks_similarity",
    "align_landmarks_affine",
    "align_landmarks_from_aligner",
    "Aligner",
    "AlignResult",
    "calculate_tiling_qc",
    "TilingQCParams",
    "StitchParams",
    "assign_stitch_groups",
]
