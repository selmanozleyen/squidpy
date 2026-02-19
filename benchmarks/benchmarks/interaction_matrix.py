"""Benchmark interaction_matrix."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._utils import visium

if TYPE_CHECKING:
    from anndata import AnnData

from squidpy.gr import interaction_matrix

adata: AnnData
CK = "leiden"


class InteractionMatrix:
    timeout = 120

    def setup(self) -> None:
        global adata
        adata = visium()

    def time_interaction_matrix(self) -> None:
        interaction_matrix(adata, cluster_key=CK, copy=True)

    def time_interaction_matrix_weighted(self) -> None:
        interaction_matrix(adata, cluster_key=CK, weights=True, copy=True)

    def time_interaction_matrix_normalized(self) -> None:
        interaction_matrix(adata, cluster_key=CK, normalized=True, copy=True)

    def peakmem_interaction_matrix(self) -> None:
        interaction_matrix(adata, cluster_key=CK, copy=True)
