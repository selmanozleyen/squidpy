from __future__ import annotations

import inspect
from functools import cache
from typing import TYPE_CHECKING, Any

import squidpy as sq

if TYPE_CHECKING:
    from anndata import AnnData

N_JOBS = 6


@cache
def _visium() -> AnnData:
    adata = sq.datasets.visium_hne_adata()
    sq.gr.spatial_neighbors(adata)
    return adata


def visium() -> AnnData:
    return _visium().copy()


def maybe_njobs(fn: Any) -> dict[str, Any]:
    """Return n_jobs/backend/show_progress_bar kwargs if the function accepts them."""
    sig = inspect.signature(fn)
    kwargs: dict[str, Any] = {}
    if "n_jobs" in sig.parameters:
        kwargs["n_jobs"] = N_JOBS
    if "backend" in sig.parameters:
        kwargs["backend"] = "loky"
    if "show_progress_bar" in sig.parameters:
        kwargs["show_progress_bar"] = False
    return kwargs
