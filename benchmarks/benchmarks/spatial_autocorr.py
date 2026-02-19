"""Benchmark spatial_autocorr."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._utils import maybe_njobs, visium

if TYPE_CHECKING:
    from anndata import AnnData

from squidpy.gr import spatial_autocorr

adata: AnnData


class SpatialAutocorrMoran:
    timeout = 600

    def setup(self) -> None:
        global adata
        adata = visium()
        adata.var["highly_variable"] = False
        adata.var.iloc[:50, adata.var.columns.get_loc("highly_variable")] = True

    def time_moran(self) -> None:
        kw = maybe_njobs(spatial_autocorr)
        spatial_autocorr(adata, mode="moran", n_perms=100, seed=42, copy=True, **kw)

    def peakmem_moran(self) -> None:
        kw = maybe_njobs(spatial_autocorr)
        spatial_autocorr(adata, mode="moran", n_perms=100, seed=42, copy=True, **kw)


class SpatialAutocorrGeary:
    timeout = 600

    def setup(self) -> None:
        global adata
        adata = visium()
        adata.var["highly_variable"] = False
        adata.var.iloc[:50, adata.var.columns.get_loc("highly_variable")] = True

    def time_geary(self) -> None:
        kw = maybe_njobs(spatial_autocorr)
        spatial_autocorr(adata, mode="geary", n_perms=100, seed=42, copy=True, **kw)

    def peakmem_geary(self) -> None:
        kw = maybe_njobs(spatial_autocorr)
        spatial_autocorr(adata, mode="geary", n_perms=100, seed=42, copy=True, **kw)
