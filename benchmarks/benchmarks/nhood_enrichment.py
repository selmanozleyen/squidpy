"""Benchmark nhood_enrichment."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._utils import maybe_njobs, visium

if TYPE_CHECKING:
    from anndata import AnnData

from squidpy.gr import nhood_enrichment

adata: AnnData
CK = "leiden"


class NhoodEnrichment:
    timeout = 600

    def setup(self) -> None:
        global adata
        adata = visium()

    def time_nhood_enrichment(self) -> None:
        kw = maybe_njobs(nhood_enrichment)
        nhood_enrichment(adata, cluster_key=CK, n_perms=500, seed=42, copy=True, **kw)

    def peakmem_nhood_enrichment(self) -> None:
        kw = maybe_njobs(nhood_enrichment)
        nhood_enrichment(adata, cluster_key=CK, n_perms=500, seed=42, copy=True, **kw)
