"""Benchmark centrality_scores."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._utils import maybe_njobs, visium

if TYPE_CHECKING:
    from anndata import AnnData

from squidpy.gr import centrality_scores

adata: AnnData
CK = "leiden"


class CentralityScores:
    timeout = 600

    def setup(self) -> None:
        global adata
        adata = visium()

    def time_centrality_scores(self) -> None:
        kw = maybe_njobs(centrality_scores)
        centrality_scores(adata, cluster_key=CK, copy=True, **kw)

    def peakmem_centrality_scores(self) -> None:
        kw = maybe_njobs(centrality_scores)
        centrality_scores(adata, cluster_key=CK, copy=True, **kw)
