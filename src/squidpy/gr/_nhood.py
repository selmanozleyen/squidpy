"""Functions for neighborhood enrichment analysis (permutation test, centralities measures etc.)."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, NamedTuple

import fast_array_utils as fau # noqa: F401
import numba.types as nt
import numpy as np
import pandas as pd
from anndata import AnnData
from numba import njit
from numpy.typing import NDArray
from pandas import CategoricalDtype
from scanpy import logging as logg
from scipy.sparse import spmatrix
from spatialdata import SpatialData

from squidpy._constants._constants import Centrality
from squidpy._constants._pkg_constants import Key
from squidpy._docs import d, inject_docs
from squidpy._utils import NDArrayA, Signal, SigQueue, _get_n_cores, deprecated_params, parallelize
from squidpy._validators import assert_positive
from squidpy.gr._utils import (
    _assert_categorical_obs,
    _assert_connectivity_key,
    _save_data,
    _shuffle_group,
)

__all__ = ["nhood_enrichment", "centrality_scores", "interaction_matrix"]


class NhoodEnrichmentResult(NamedTuple):
    """Result of nhood_enrichment function."""

    zscore: NDArray[np.number]
    counts: NDArray[np.number]  # NamedTuple inherits from tuple so cannot use 'count' as attribute name


# data type aliases (both for numpy and numba should match)
dt = nt.uint32
ndt = np.uint32
_template = """
from __future__ import annotations

from numba import njit, prange
import numpy as np

@njit(dt[:, :](dt[:], dt[:], dt[:]), parallel={parallel}, fastmath=True)
def _nenrich_{n_cls}_{parallel}(indices: NDArrayA, indptr: NDArrayA, clustering: NDArrayA) -> np.ndarray:
    '''
    Count how many times clusters :math:`i` and :math:`j` are connected.

    Parameters
    ----------
    indices
        :attr:`scipy.sparse.csr_matrix.indices`.
    indptr
        :attr:`scipy.sparse.csr_matrix.indptr`.
    clustering
        Array of shape ``(n_cells,)`` containig cluster labels ranging from `0` to `n_clusters - 1` inclusive.

    Returns
    -------
    :class:`numpy.ndarray`
        Array of shape ``(n_clusters, n_clusters)`` containing the pairwise counts.
    '''
    res = np.zeros((indptr.shape[0] - 1, {n_cls}), dtype=ndt)

    for i in prange(res.shape[0]):
        xs, xe = indptr[i], indptr[i + 1]
        cols = indices[xs:xe]
        for c in cols:
            res[i, clustering[c]] += 1
    {init}
    {loop}
    {finalize}
"""


def _create_function(n_cls: int, parallel: bool = False) -> Callable[[NDArrayA, NDArrayA, NDArrayA], NDArrayA]:
    """
    Create a :mod:`numba` function which counts the number of connections between clusters.

    Parameters
    ----------
    n_cls
        Number of clusters. We're assuming that cluster labels are `0`, `1`, ..., `n_cls - 1`.
    parallel
        Whether to enable :mod:`numba` parallelization.

    Returns
    -------
    The aforementioned function.
    """
    if n_cls <= 1:
        raise ValueError(f"Expected at least `2` clusters, found `{n_cls}`.")

    rng = range(n_cls)
    init = "".join(
        f"""
    g{i} = np.zeros(({n_cls},), dtype=ndt)"""
        for i in rng
    )

    loop_body = """
        if cl == 0:
            g0 += res[row]"""
    loop_body = loop_body + "".join(
        f"""
        elif cl == {i}:
            g{i} += res[row]"""
        for i in range(1, n_cls)
    )
    loop = f"""
    for row in prange(res.shape[0]):
        cl = clustering[row]
        {loop_body}
        else:
            assert False, "Unhandled case."
    """
    finalize = ", ".join(f"g{i}" for i in rng)
    finalize = f"return np.stack(({finalize}))"  # must really be a tuple

    fn_key = f"_nenrich_{n_cls}_{parallel}"
    if fn_key not in globals():
        template = _template.format(init=init, loop=loop, finalize=finalize, n_cls=n_cls, parallel=parallel)
        exec(compile(template, "", "exec"), globals())

    return globals()[fn_key]  # type: ignore[no-any-return]


@d.get_sections(base="nhood_ench", sections=["Parameters"])
@d.dedent
def nhood_enrichment(
    adata: AnnData | SpatialData,
    cluster_key: str,
    library_key: str | None = None,
    connectivity_key: str | None = None,
    n_perms: int = 1000,
    numba_parallel: bool = False,
    seed: int | None = None,
    copy: bool = False,
    n_jobs: int | None = None,
    backend: str = "loky",
    show_progress_bar: bool = True,
) -> NhoodEnrichmentResult | None:
    """
    Compute neighborhood enrichment by permutation test.

    Parameters
    ----------
    %(adata)s
    %(cluster_key)s
    %(library_key)s
    %(conn_key)s
    %(n_perms)s
    %(numba_parallel)s
    %(seed)s
    %(copy)s
    %(parallelize)s

    Returns
    -------
    If ``copy = True``, returns a :class:`~squidpy.gr.NhoodEnrichmentResult` with the z-score and the enrichment count.

    Otherwise, modifies the ``adata`` with the following keys:

        - :attr:`anndata.AnnData.uns` ``['{cluster_key}_nhood_enrichment']['zscore']`` - the enrichment z-score.
        - :attr:`anndata.AnnData.uns` ``['{cluster_key}_nhood_enrichment']['count']`` - the enrichment count.
    """
    if isinstance(adata, SpatialData):
        adata = adata.table
    connectivity_key = Key.obsp.spatial_conn(connectivity_key)
    _assert_categorical_obs(adata, cluster_key)
    _assert_connectivity_key(adata, connectivity_key)
    assert_positive(n_perms, name="n_perms")

    adj = adata.obsp[connectivity_key]
    original_clust = adata.obs[cluster_key]
    clust_map = {v: i for i, v in enumerate(original_clust.cat.categories.values)}  # map categories
    int_clust = np.array([clust_map[c] for c in original_clust], dtype=ndt)

    if library_key is not None:
        _assert_categorical_obs(adata, key=library_key)
        libraries: pd.Series | None = adata.obs[library_key]
    else:
        libraries = None

    indices, indptr = (adj.indices.astype(ndt), adj.indptr.astype(ndt))
    n_cls = len(clust_map)

    _test = _create_function(n_cls, parallel=numba_parallel)
    count = _test(indices, indptr, int_clust)

    n_jobs = _get_n_cores(n_jobs)
    start = logg.info(f"Calculating neighborhood enrichment using `{n_jobs}` core(s)")

    perms = parallelize(
        _nhood_enrichment_helper,
        collection=np.arange(n_perms).tolist(),
        extractor=np.vstack,
        n_jobs=n_jobs,
        backend=backend,
        show_progress_bar=show_progress_bar,
    )(
        callback=_test,
        indices=indices,
        indptr=indptr,
        int_clust=int_clust,
        libraries=libraries,
        n_cls=n_cls,
        seed=seed,
    )
    zscore = (count - perms.mean(axis=0)) / perms.std(axis=0)

    if copy:
        return NhoodEnrichmentResult(zscore=zscore, counts=count)

    _save_data(
        adata,
        attr="uns",
        key=Key.uns.nhood_enrichment(cluster_key),
        data={"zscore": zscore, "count": count},
        time=start,
    )


@d.dedent
@inject_docs(c=Centrality)
@deprecated_params({"backend": "1.10.0", "show_progress_bar": "1.10.0"})
def centrality_scores(
    adata: AnnData | SpatialData,
    cluster_key: str,
    score: str | Iterable[str] | None = None,
    connectivity_key: str | None = None,
    copy: bool = False,
    n_jobs: int | None = None,
) -> pd.DataFrame | None:
    """
    Compute centrality scores per cluster or cell type.

    Inspired by usage in Gene Regulatory Networks (GRNs) in :cite:`celloracle`.

    Parameters
    ----------
    %(adata)s
    %(cluster_key)s
    score
        Centrality measures.
        If `None`, use all the options below. Valid options are:

            - `{c.CLOSENESS.s!r}` - measure of how close the group is to other nodes.
            - `{c.CLUSTERING.s!r}` - measure of the degree to which nodes cluster together.
            - `{c.DEGREE.s!r}` - fraction of non-group members connected to group members.

    %(conn_key)s
    %(copy)s
    n_jobs
        Number of threads to use. If `None` or ``1``, run sequentially.
        The speedup is most significant for closeness centrality on large graphs.

    Returns
    -------
    If ``copy = True``, returns a :class:`pandas.DataFrame`. Otherwise, modifies the ``adata`` with the following key:

        - :attr:`anndata.AnnData.uns` ``['{{cluster_key}}_centrality_scores']`` - the centrality scores,
          as mentioned above.
    """
    if isinstance(adata, SpatialData):
        adata = adata.table
    connectivity_key = Key.obsp.spatial_conn(connectivity_key)
    _assert_categorical_obs(adata, cluster_key)
    _assert_connectivity_key(adata, connectivity_key)

    if isinstance(score, str | Centrality):
        centrality = [score]
    elif score is None:
        centrality = [c.s for c in Centrality]

    centralities = [Centrality(c) for c in centrality]

    adj = adata.obsp[connectivity_key]
    adj_bin = (adj > 0).astype(np.float32)
    adj_bin.setdiag(0)
    adj_bin.eliminate_zeros()

    cat = adata.obs[cluster_key].cat.categories.values
    clusters = adata.obs[cluster_key].values
    indices = [np.where(clusters == cl)[0] for cl in cat]

    n_workers = _get_n_cores(n_jobs)
    start = logg.info(f"Calculating centralities `{centralities}` using `{n_workers}` thread(s)")

    clustering_coeffs: NDArrayA | None = None
    if any(c == Centrality.CLUSTERING for c in centralities):
        clustering_coeffs = _clustering_coefficients(adj_bin)

    def _score_one(c: Centrality, idx: NDArrayA) -> float:
        if c == Centrality.CLOSENESS:
            return _group_closeness_centrality(adj_bin, idx)
        elif c == Centrality.DEGREE:
            return _group_degree_centrality(adj_bin, idx)
        elif c == Centrality.CLUSTERING:
            assert clustering_coeffs is not None
            return _average_clustering(clustering_coeffs, idx)
        else:
            raise NotImplementedError(f"Centrality `{c}` is not yet implemented.")

    results: dict[str, list[float]] = {}
    if n_workers > 1:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            for c in centralities:
                futures = [pool.submit(_score_one, c, idx) for idx in indices]
                results[c.s] = [f.result() for f in futures]
    else:
        for c in centralities:
            results[c.s] = [_score_one(c, idx) for idx in indices]

    df = pd.DataFrame(results, index=cat)

    if copy:
        return df
    _save_data(
        adata,
        attr="uns",
        key=Key.uns.centrality_scores(cluster_key),
        data=df,
        time=start,
    )


@d.dedent
def interaction_matrix(
    adata: AnnData | SpatialData,
    cluster_key: str,
    connectivity_key: str | None = None,
    normalized: bool = False,
    copy: bool = False,
    weights: bool = False,
) -> NDArrayA | None:
    """
    Compute interaction matrix for clusters.

    Parameters
    ----------
    %(adata)s
    %(cluster_key)s
    %(conn_key)s
    normalized
        If `True`, each row is normalized to sum to 1.
    %(copy)s
    weights
        Whether to use edge weights or binarize.

    Returns
    -------
    If ``copy = True``, returns the interaction matrix.

    Otherwise, modifies the ``adata`` with the following key:

        - :attr:`anndata.AnnData.uns` ``['{cluster_key}_interactions']`` - the interaction matrix.
    """
    if isinstance(adata, SpatialData):
        adata = adata.table
    connectivity_key = Key.obsp.spatial_conn(connectivity_key)
    _assert_categorical_obs(adata, cluster_key)
    _assert_connectivity_key(adata, connectivity_key)

    cats = adata.obs[cluster_key]
    mask = ~pd.isnull(cats).values
    cats = cats.loc[mask]
    if not len(cats):
        raise RuntimeError(f"After removing NaNs in `adata.obs[{cluster_key!r}]`, none remain.")

    g = adata.obsp[connectivity_key]
    g = g[mask, :][:, mask]
    n_cats = len(cats.cat.categories)

    g_data = g.data if weights else np.broadcast_to(1, shape=len(g.data))
    dtype = int if pd.api.types.is_bool_dtype(g.dtype) or pd.api.types.is_integer_dtype(g.dtype) else float
    output: NDArrayA = np.zeros((n_cats, n_cats), dtype=dtype)

    _interaction_matrix(g_data, g.indices, g.indptr, cats.cat.codes.to_numpy(), output)

    if normalized:
        output = output / output.sum(axis=1).reshape((-1, 1))

    if copy:
        return output

    _save_data(adata, attr="uns", key=Key.uns.interaction_matrix(cluster_key), data=output)


@njit
def _interaction_matrix(
    data: NDArrayA,
    indices: NDArrayA,
    indptr: NDArrayA,
    cats: NDArrayA,
    output: NDArrayA,
) -> NDArrayA:
    indices_list = np.split(indices, indptr[1:-1])
    data_list = np.split(data, indptr[1:-1])
    for i in range(len(data_list)):
        cur_row = cats[i]
        cur_indices = indices_list[i]
        cur_data = data_list[i]
        for j, val in zip(cur_indices, cur_data):  # noqa: B905
            cur_col = cats[j]
            output[cur_row, cur_col] += val
    return output

@njit(nogil=True)
def _group_closeness_centrality(g: spmatrix, idx: NDArrayA) -> float:
    """Group closeness centrality via multi-source BFS on a sparse adjacency matrix.

    Uses two pre-allocated buffers swapped each level to avoid allocations
    inside the hot loop, and a bool visited array instead of int32 dist.
    """
    n = g.shape[0]
    n_group = len(idx)
    n_non_group = n - n_group
    if n_non_group == 0:
        return 0.0

    visited = np.zeros(n, dtype=np.bool_)
    visited[idx] = True

    buf_a = np.empty(n, dtype=np.int32)
    buf_b = np.empty(n, dtype=np.int32)
    buf_a[:n_group] = idx
    frontier_size = n_group

    level = 0
    total_dist = np.int64(0)
    reached = 0
    use_a = True
    while frontier_size > 0:
        level += 1
        frontier = buf_a if use_a else buf_b
        next_frontier = buf_b if use_a else buf_a
        next_size = 0
        for i in range(frontier_size):
            v = frontier[i]
            for j in range(g.indptr[v], g.indptr[v + 1]):
                nb = g.indices[j]
                if not visited[nb]:
                    visited[nb] = True
                    total_dist += level
                    reached += 1
                    next_frontier[next_size] = nb
                    next_size += 1
        frontier_size = next_size
        use_a = not use_a
        if reached >= n_non_group:
            break

    if total_dist == 0:
        return 0.0
    return n_non_group / total_dist


def _group_degree_centrality(adj: spmatrix, idx: NDArrayA) -> float:
    """Group degree centrality: fraction of non-group nodes adjacent to any group member."""
    n = adj.shape[0]
    n_group = len(idx)
    if n_group >= n:
        return 0.0

    is_group = np.zeros(n, dtype=np.bool_)
    is_group[idx] = True

    neighbors = adj[idx].sum(axis=0).A1 > 0
    count = int(np.count_nonzero(neighbors & ~is_group))
    return count / (n - n_group)


def _clustering_coefficients(adj: spmatrix) -> NDArrayA:
    """Per-node local clustering coefficients via sparse triangle counting.

    Precomputed once and reused for all clusters.
    """
    degrees = np.asarray(adj.sum(axis=1)).ravel()
    adj_sq = adj @ adj
    triangles = np.asarray(adj.multiply(adj_sq).sum(axis=1)).ravel() / 2.0
    denom = degrees * (degrees - 1)
    cc = np.zeros(len(degrees), dtype=np.float32)
    valid = denom > 0
    cc[valid] = 2.0 * triangles[valid] / denom[valid]
    return cc


def _average_clustering(cc: NDArrayA, idx: NDArrayA) -> float:
    """Average clustering coefficient for a group of nodes."""
    group_cc = cc[idx]
    return float(np.mean(group_cc)) if len(group_cc) > 0 else 0.0


def _nhood_enrichment_helper(
    ixs: NDArrayA,
    callback: Callable[[NDArrayA, NDArrayA, NDArrayA], NDArrayA],
    indices: NDArrayA,
    indptr: NDArrayA,
    int_clust: NDArrayA,
    libraries: pd.Series[CategoricalDtype] | None,
    n_cls: int,
    seed: int | None = None,
    queue: SigQueue | None = None,
) -> NDArrayA:
    perms = np.empty((len(ixs), n_cls, n_cls), dtype=np.float64)
    int_clust = int_clust.copy()  # threading
    rs = np.random.RandomState(seed=None if seed is None else seed + ixs[0])

    for i in range(len(ixs)):
        if libraries is not None:
            int_clust = _shuffle_group(int_clust, libraries, rs)
        else:
            rs.shuffle(int_clust)
        perms[i, ...] = callback(indices, indptr, int_clust)

        if queue is not None:
            queue.put(Signal.UPDATE)

    if queue is not None:
        queue.put(Signal.FINISH)

    return perms
