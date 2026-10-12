from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numba
import numpy as np
import pandas as pd
from anndata import AnnData
from numba import njit, prange
from scanpy import logging as logg
from scipy.sparse import csc_matrix, csr_matrix, issparse, isspmatrix_csr, spmatrix
from scipy.spatial import cKDTree
from sklearn.metrics import pairwise_distances
from spatialdata import SpatialData
from tqdm.auto import tqdm

from squidpy._compat import old_positionals
from squidpy._constants._pkg_constants import Key
from squidpy._docs import d, inject_docs
from squidpy._utils import NDArrayA, deprecated_params, get_n_numba_threads, memory_budget
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
    max_mem: int | str | None = None,
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
    %(max_mem)s
        Here: the genes simulated together, each needing 24 bytes per spot.
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
        max_mem=max_mem,
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


# Working memory per block of genes: the concentrations, their change and a per-spot scratch array,
# 8 bytes each per spot and gene. The default budget's floor (see `memory_budget`); an eighth of
# free memory is used when larger, since the per-gene entropy sums run in parallel over a block.
_BLOCK_BYTES = 4 * 2**30


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
    max_mem: int | str | None = None,
    show_progress_bar: bool = True,
) -> NDArrayA:
    """Run the diffusion for every gene column, a block of genes at a time.

    A block is a ``(spots, genes)`` array advanced together: a spot's neighbour reads fetch a
    contiguous row of genes and share one index lookup, and every step runs in parallel over spots,
    so all threads work even with fewer genes than threads. Each gene still does exactly the
    arithmetic of a run on its own and stops at its own step.
    """
    n_spots, n_genes = vals.shape
    block = int(max(1, min(n_genes, memory_budget(max_mem, floor=_BLOCK_BYTES) // (24 * n_spots))))
    scores = np.empty(n_genes)
    prev_threads = numba.get_num_threads()
    numba.set_num_threads(n_jobs)
    try:
        with tqdm(total=n_genes, unit="gene", disable=not show_progress_bar) as pbar:
            for start in range(0, n_genes, block):
                cols = vals[:, start : start + block]
                conc = np.ascontiguousarray(cols.toarray() if issparse(cols) else cols, dtype=np.float64)
                steps = _diffusion(conc, use_hex, n_iter, sat, sat_idx, unsat, unsat_idx, dt, thresh)
                scores[start : start + block] = dt * steps
                pbar.update(conc.shape[1])
    finally:
        numba.set_num_threads(prev_threads)
    return scores


@njit(parallel=True, fastmath=True, nogil=True)
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
) -> NDArrayA:
    """Simulate diffusion on a regular graph for a ``(spots, genes)`` block, modified in place.

    Returns, per gene, the step at which the entropy settled (NaN if it did not within ``n_iter``).
    Each step computes every Laplacian from the previous state before updating. Everything runs in
    parallel over spots, the two entropy sums too: they add fixed-size chunks of spots in parallel
    and then the chunk partials in chunk order, so the result depends on the grid only, never on the
    thread count or the block a gene is in. On a grid of at most one chunk (any regular Visium slide)
    that is the plain sequential sum. Nothing is allocated per step.

    Stencils and entropy follow https://github.com/almaan/sepal/blob/master/sepal/models.py; the
    hexagonal 7-point stencil is from Benster, Kantorovich & Krylov, *Approximate Methods of Higher
    Analysis* (ISBN 978-0486821603), the rectangular one is the 5-point stencil.
    """
    n_spots, n_genes = conc.shape
    n_sat, n_nbrs = sat_idx.shape
    eps = np.finfo(np.float64).eps
    work = np.empty((n_sat, n_genes))  # Laplacian, later the entropy terms
    dcdt = np.zeros((n_spots, n_genes))  # only saturated rows are ever written or read
    xs = np.empty(n_genes)
    chunk = max(16384, -(-n_sat // 1024))  # at most 1024 partial sums, set by the grid alone
    n_chunks = -(-n_sat // chunk)
    partial = np.empty((n_chunks, n_genes))
    prev_ent = np.ones(n_genes)
    active = np.ones(n_genes, dtype=np.bool_)
    settled = np.full(n_genes, np.nan)

    for i in range(n_iter):
        # discrete Laplacian at every saturated node: 7-point stencil (hex) or 5-point (rect)
        for j in prange(n_sat):
            for g in range(n_genes):
                if active[g]:
                    nbrs = 0.0
                    for m in range(n_nbrs):
                        nbrs += conc[sat_idx[j, m], g]
                    c = conc[sat[j], g]
                    work[j, g] = (2.0 * nbrs - 12.0 * c) / 3.0 if use_hex else nbrs - 4.0 * c
        for j in prange(n_sat):
            for g in range(n_genes):
                if active[g]:
                    dcdt[sat[j], g] = work[j, g]
                    conc[sat[j], g] += work[j, g] * dt
        # unsaturated (border) nodes follow their nearest saturated node
        for u in prange(unsat.shape[0]):
            for g in range(n_genes):
                if active[g]:
                    conc[unsat[u], g] += dcdt[unsat_idx[u], g] * dt
        for k in prange(n_spots):
            for g in range(n_genes):
                if active[g] and conc[k, g] < 0:
                    conc[k, g] = 0.0

        # Shannon entropy (nats) of the saturated nodes' normalized concentrations; p = 0 adds 0.
        # total mass per gene: chunk partials in parallel, then added in chunk order
        for b in prange(n_chunks):
            for g in range(n_genes):
                partial[b, g] = 0.0
            for j in range(b * chunk, min(n_sat, (b + 1) * chunk)):
                for g in range(n_genes):
                    x = conc[sat[j], g]
                    if active[g] and x > 0:
                        partial[b, g] += x
        for g in range(n_genes):
            xs[g] = 0.0
            for b in range(n_chunks):
                xs[g] += partial[b, g]
        # -sum p log p, the same way
        for b in prange(n_chunks):
            for g in range(n_genes):
                partial[b, g] = 0.0
            for j in range(b * chunk, min(n_sat, (b + 1) * chunk)):
                for g in range(n_genes):
                    x = conc[sat[j], g]
                    if active[g] and x > 0 and xs[g] >= eps:
                        xn = x / xs[g]
                        partial[b, g] -= np.log(max(xn, eps)) * xn
        for g in prange(n_genes):
            if active[g]:
                ent = 0.0
                for b in range(n_chunks):
                    ent += partial[b, g]
                ent /= n_sat
                if np.abs(ent - prev_ent[g]) <= thresh:
                    settled[g] = i
                    active[g] = False
                prev_ent[g] = ent
        if not active.any():
            break

    return settled


def _compute_idxs(
    g: spmatrix, spatial: NDArrayA, sat_thresh: int, metric: str = "l1"
) -> tuple[NDArrayA, NDArrayA, NDArrayA, NDArrayA]:
    """Get saturated and unsaturated nodes and neighborhood indices."""
    sat, unsat = _get_sat_unsat_idx(g.indptr, g.shape[0], sat_thresh)

    sat_idx, nearest_sat, un_unsat = _get_nhood_idx(sat, unsat, g.indptr, g.indices, sat_thresh)

    # assign the closest saturated node to the unsaturated ones without a saturated neighbor
    nearest_sat[np.isnan(nearest_sat)] = sat[_closest(spatial[un_unsat], spatial[sat], metric)]

    return sat, sat_idx, unsat, nearest_sat.astype(np.int32)


# Minkowski p of the metrics scipy's cKDTree supports; others take the dense path
_TREE_P = {"l1": 1, "manhattan": 1, "cityblock": 1, "l2": 2, "euclidean": 2}


def _closest(queries: NDArrayA, points: NDArrayA, metric: str) -> NDArrayA:
    """Index of the closest point for every query, the lowest index among equally close ones.

    A KD-tree instead of a dense ``(queries, points)`` distance matrix, which on a Visium HD grid
    (millions of bins, many tissue-border bins) does not fit in memory. Ties resolve like
    ``argmin`` over the dense matrix did: every point at the minimum distance, lowest index wins.
    """
    if len(queries) == 0:
        return np.zeros(0, dtype=np.intp)
    if metric not in _TREE_P:
        return np.argmin(pairwise_distances(queries, points, metric=metric), axis=1)
    p = _TREE_P[metric]
    tree = cKDTree(points)
    dist, _ = tree.query(queries, k=1, p=p)
    return np.array([min(hits) for hits in tree.query_ball_point(queries, r=dist, p=p)], dtype=np.intp)


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

    # get closest saturated of unsaturated: a lookup table, not a scan of `sat` per neighbor
    is_sat = np.zeros(g_indptr.shape[0] - 1, dtype=np.bool_)
    is_sat[sat] = True
    nearest_sat = np.full_like(unsat, fill_value=np.nan, dtype=np.float64)
    for idx in range(unsat.shape[0]):
        i = unsat[idx]
        unsat_neigh = g_indices[g_indptr[i] : g_indptr[i + 1]]
        for u in unsat_neigh:
            if is_sat[u]:  # take the first saturated nhood
                nearest_sat[idx] = u
                break

    # some unsat still don't have a sat nhood
    # return them and compute distances in outer func
    un_unsat = unsat[np.isnan(nearest_sat)]

    return sat_idx.astype(np.int32), nearest_sat, un_unsat
