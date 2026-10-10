from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
from anndata import AnnData
from numba import njit
from scanpy import logging as logg
from scipy.sparse import csc_matrix, csr_matrix, issparse, isspmatrix_csr, spmatrix
from sklearn.metrics import pairwise_distances
from spatialdata import SpatialData

from squidpy._compat import old_positionals
from squidpy._constants._pkg_constants import Key
from squidpy._docs import d, inject_docs
from squidpy._utils import NDArrayA, deprecated_params, get_n_numba_threads, thread_map
from squidpy._validators import assert_non_empty_sequence
from squidpy.gr._utils import (
    _assert_connectivity_key,
    _assert_spatial_basis,
    _extract_expression,
    _save_data,
    extract_adata_if_sdata,
)

__all__ = ["sepal"]


@d.dedent
@inject_docs(key=Key.obsp.spatial_conn())
@old_positionals(
    "max_neighs",
    "genes",
    "n_iter",
    "dt",
    "thresh",
    "connectivity_key",
    "spatial_key",
    "layer",
    "use_raw",
    "copy",
    "n_jobs",
    "show_progress_bar",
)
@deprecated_params({"backend": "1.10.0"})
def sepal(
    adata: AnnData | SpatialData,
    *,
    max_neighs: Literal[4, 6],
    genes: str | Sequence[str] | None = None,
    n_iter: int | None = 30000,
    dt: float = 0.001,
    thresh: float = 1e-8,
    connectivity_key: str = Key.obsp.spatial_conn(),
    spatial_key: str = Key.obsm.spatial,
    layer: str | None = None,
    use_raw: bool = False,
    copy: bool = False,
    n_jobs: int | None = None,
    show_progress_bar: bool = True,
    table_key: str | None = None,
) -> pd.DataFrame | None:
    """
    Identify spatially variable genes with *Sepal*.

    *Sepal* is a method that simulates a diffusion process to quantify spatial structure in tissue.
    See :cite:`andersson2021` for reference.

    Parameters
    ----------
    %(adata)s
    %(table_key)s
    max_neighs
        Maximum number of neighbors of a node in the graph. Valid options are:

            - `4` - for a square-grid (ST, Dbit-seq).
            - `6` - for a hexagonal-grid (Visium).
    genes
        List of gene names, as stored in :attr:`anndata.AnnData.var_names`, used to compute sepal score.

        If `None`, it's computed :attr:`anndata.AnnData.var` ``['highly_variable']``, if present.
        Otherwise, it's computed for all genes.
    n_iter
        Maximum number of iterations for the diffusion simulation.
        If ``n_iter`` iterations are reached, the simulation will terminate
        even though convergence has not been achieved.
    dt
        Time step in diffusion simulation.
    thresh
        Entropy threshold for convergence of diffusion simulation.
    %(conn_key)s
    %(spatial_key)s
    layer
        Layer in :attr:`anndata.AnnData.layers` to use. If `None`, use :attr:`anndata.AnnData.X`.
    use_raw
        Whether to access :attr:`anndata.AnnData.raw`.
    %(copy)s
    %(n_jobs_threads)s
    %(show_progress_bar)s
    Returns
    -------
    If ``copy = True``, returns a :class:`pandas.DataFrame` with the sepal scores.

    Otherwise, modifies the ``adata`` with the following key:

        - :attr:`anndata.AnnData.uns` ``['sepal_score']`` - the sepal scores.

    Notes
    -----
    If some genes in :attr:`anndata.AnnData.uns` ``['sepal_score']`` are `NaN`,
    consider re-running the function with increased ``n_iter``.
    """
    adata = extract_adata_if_sdata(adata, table_key=table_key)
    _assert_connectivity_key(adata, key=connectivity_key)
    _assert_spatial_basis(adata, key=spatial_key)
    if max_neighs not in (4, 6):
        raise ValueError(f"Expected `max_neighs` to be either `4` or `6`, found `{max_neighs}`.")

    spatial = adata.obsm[spatial_key].astype(np.float64)

    if genes is None:
        genes = adata.var_names.values
        if "highly_variable" in adata.var.columns:
            genes = genes[adata.var["highly_variable"].values]
    genes = assert_non_empty_sequence(genes, name="genes")

    n_jobs = get_n_numba_threads(n_jobs)

    g = adata.obsp[connectivity_key]
    if not isspmatrix_csr(g):
        g = csr_matrix(g)
    g.eliminate_zeros()

    max_n = np.diff(g.indptr).max()
    if max_n != max_neighs:
        raise ValueError(f"Expected `max_neighs={max_neighs}`, found node with `{max_n}` neighbors.")

    # get saturated/unsaturated nodes
    sat, sat_idx, unsat, unsat_idx = _compute_idxs(g, spatial, max_neighs, "l1")

    # get counts
    vals, genes = _extract_expression(adata, genes=genes, use_raw=use_raw, layer=layer)
    start = logg.info(f"Calculating sepal score for `{len(genes)}` genes using `{n_jobs}` thread(s)")

    use_hex = max_neighs == 6

    if issparse(vals):
        vals = csc_matrix(vals)
    score = _diffusion_genes(
        vals,
        use_hex=use_hex,
        n_iter=n_iter,
        sat=sat,
        sat_idx=sat_idx,
        unsat=unsat,
        unsat_idx=unsat_idx,
        dt=dt,
        thresh=thresh,
        n_jobs=n_jobs,
        show_progress_bar=show_progress_bar,
    )

    key_added = "sepal_score"
    sepal_score = pd.DataFrame(score, index=genes, columns=[key_added])

    if sepal_score[key_added].isna().any():
        logg.warning("Found `NaN` in sepal scores, consider increasing `n_iter` to a higher value")
    sepal_score = sepal_score.sort_values(by=key_added, ascending=False)

    if copy:
        logg.info("Finish", time=start)
        return sepal_score

    _save_data(adata, attr="uns", key=key_added, data=sepal_score, time=start)


def _diffusion_genes(
    vals: NDArrayA | spmatrix,
    *,
    use_hex: bool,
    n_iter: int,
    sat: NDArrayA,
    sat_idx: NDArrayA,
    unsat: NDArrayA,
    unsat_idx: NDArrayA,
    dt: float,
    thresh: float,
    n_jobs: int,
    show_progress_bar: bool = True,
) -> NDArrayA:
    """Run diffusion for each gene column, parallelised across threads."""

    sparse = issparse(vals)

    def _process_gene(i: int) -> float:
        if sparse:
            conc = np.ascontiguousarray(vals[:, i].toarray().ravel(), dtype=np.float64)
        else:
            conc = np.ascontiguousarray(vals[:, i], dtype=np.float64)
        time_iter = _diffusion(
            conc,
            use_hex,
            n_iter,
            sat,
            sat_idx,
            unsat,
            unsat_idx,
            dt,
            thresh,
        )
        return dt * time_iter

    scores = thread_map(
        _process_gene,
        range(vals.shape[1]),
        n_jobs=n_jobs,
        show_progress_bar=show_progress_bar,
        unit="gene",
    )
    return np.array(scores)


@njit(fastmath=True, nogil=True)
def _diffusion(  # noqa: PLR0917, numba requires positional arguments
    conc: NDArrayA,
    use_hex: bool,
    n_iter: int,
    sat: NDArrayA,
    sat_idx: NDArrayA,
    unsat: NDArrayA,
    unsat_idx: NDArrayA,
    dt: float,
    thresh: float,
) -> float:
    """Simulate diffusion process on a regular graph; return the iteration the entropy settled at.

    Written as plain loops so that an iteration allocates nothing: the kernel runs up to ``n_iter``
    iterations per gene, and per-iteration temporaries made it allocation-bound and kept threads from
    scaling. Each iteration still computes every Laplacian from the previous state before updating.
    """
    n_sat, n_nbrs = sat_idx.shape
    eps = np.finfo(np.float64).eps
    d2 = np.empty(n_sat)
    dcdt = np.zeros(conc.shape[0])  # only saturated entries are ever written or read
    prev_ent = 1.0

    for i in range(n_iter):
        # discrete Laplacian at every saturated node: 7-point stencil (hex) or 5-point (rect)
        for j in range(n_sat):
            nbrs = 0.0
            for m in range(n_nbrs):
                nbrs += conc[sat_idx[j, m]]
            c = conc[sat[j]]
            d2[j] = (2.0 * nbrs - 12.0 * c) / 3.0 if use_hex else nbrs - 4.0 * c
        for j in range(n_sat):
            dcdt[sat[j]] = d2[j]
            conc[sat[j]] += d2[j] * dt
        # unsaturated (border) nodes follow their nearest saturated node
        for u in range(unsat.shape[0]):
            conc[unsat[u]] += dcdt[unsat_idx[u]] * dt
        for k in range(conc.shape[0]):
            if conc[k] < 0:
                conc[k] = 0.0

        # Shannon entropy (nats) of the saturated nodes' normalized concentrations; p = 0 adds 0
        xs = 0.0
        for j in range(n_sat):
            if conc[sat[j]] > 0:
                xs += conc[sat[j]]
        ent = 0.0
        if xs >= eps:
            for j in range(n_sat):
                x = conc[sat[j]]
                if x > 0:
                    xn = x / xs
                    ent -= np.log(max(xn, eps)) * xn
        ent /= n_sat
        if np.abs(ent - prev_ent) <= thresh:
            return float(i)
        prev_ent = ent

    return np.nan


def _compute_idxs(
    g: spmatrix, spatial: NDArrayA, sat_thresh: int, metric: str = "l1"
) -> tuple[NDArrayA, NDArrayA, NDArrayA, NDArrayA]:
    """Get saturated and unsaturated nodes and neighborhood indices."""
    sat, unsat = _get_sat_unsat_idx(g.indptr, g.shape[0], sat_thresh)

    sat_idx, nearest_sat, un_unsat = _get_nhood_idx(sat, unsat, g.indptr, g.indices, sat_thresh)

    # compute dist btwn remaining unsat and all sat
    dist = pairwise_distances(spatial[un_unsat], spatial[sat], metric=metric)
    # assign closest sat to remaining nearest_sat
    nearest_sat[np.isnan(nearest_sat)] = sat[np.argmin(dist, axis=1)]

    return sat, sat_idx, unsat, nearest_sat.astype(np.int32)


@njit
def _get_sat_unsat_idx(g_indptr: NDArrayA, g_shape: int, sat_thresh: int) -> tuple[NDArrayA, NDArrayA]:
    """Get saturated and unsaturated nodes based on thresh."""
    n_indices = np.diff(g_indptr)
    unsat = np.arange(g_shape)[n_indices < sat_thresh]
    sat = np.arange(g_shape)[n_indices == sat_thresh]

    return sat, unsat


@njit
def _get_nhood_idx(
    sat: NDArrayA,
    unsat: NDArrayA,
    g_indptr: NDArrayA,
    g_indices: NDArrayA,
    sat_thresh: int,
) -> tuple[NDArrayA, NDArrayA, NDArrayA]:
    """Get saturated and unsaturated neighborhood indices."""
    # get saturated nhood indices
    sat_idx = np.zeros((sat.shape[0], sat_thresh))
    for idx in range(sat.shape[0]):
        i = sat[idx]
        sat_idx[idx] = g_indices[g_indptr[i] : g_indptr[i + 1]]

    # get closest saturated of unsaturated
    nearest_sat = np.full_like(unsat, fill_value=np.nan, dtype=np.float64)
    for idx in range(unsat.shape[0]):
        i = unsat[idx]
        unsat_neigh = g_indices[g_indptr[i] : g_indptr[i + 1]]
        for u in unsat_neigh:
            if u in sat:  # take the first saturated nhood
                nearest_sat[idx] = u
                break

    # some unsat still don't have a sat nhood
    # return them and compute distances in outer func
    un_unsat = unsat[np.isnan(nearest_sat)]

    return sat_idx.astype(np.int32), nearest_sat, un_unsat
