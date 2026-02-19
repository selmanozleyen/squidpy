"""Benchmark ligrec.

ligrec's public API uses **kwargs, so n_jobs/show_progress_bar are
silently accepted on both branches.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._utils import N_JOBS, visium

if TYPE_CHECKING:
    from anndata import AnnData

import squidpy as sq

adata: AnnData
CK = "leiden"


class Ligrec:
    timeout = 600

    def setup(self) -> None:
        global adata
        adata = visium()

    def time_ligrec(self) -> None:
        sq.gr.ligrec(
            adata, CK, n_perms=50, seed=42, copy=True,
            n_jobs=N_JOBS, show_progress_bar=False,
        )

    def peakmem_ligrec(self) -> None:
        sq.gr.ligrec(
            adata, CK, n_perms=50, seed=42, copy=True,
            n_jobs=N_JOBS, show_progress_bar=False,
        )
