"""``align_samples`` family: align two samples' point clouds (STalign).

Stays cheap to import -- JAX is pulled in lazily, only when an aligner's
:meth:`~squidpy.experimental.methods.StalignAligner.align` runs.
"""

from __future__ import annotations

from squidpy.experimental.methods.align_samples._stalign import (
    StalignAligner,
    StalignConfig,
    StalignResult,
)

__all__ = ["StalignAligner", "StalignConfig", "StalignResult"]
