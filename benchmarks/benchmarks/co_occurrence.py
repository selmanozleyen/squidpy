"""Benchmark co_occurrence."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._utils import maybe_njobs, visium

if TYPE_CHECKING:
    from anndata import AnnData

from squidpy.gr import co_occurrence

adata: AnnData
CK = "leiden"


class CoOccurrence:
    timeout = 600

    def setup(self) -> None:
        global adata
        adata = visium()

    def time_co_occurrence(self) -> None:
        kw = maybe_njobs(co_occurrence)
        co_occurrence(adata, cluster_key=CK, copy=True, **kw)

    def peakmem_co_occurrence(self) -> None:
        kw = maybe_njobs(co_occurrence)
        co_occurrence(adata, cluster_key=CK, copy=True, **kw)
