"""Functions for neighborhood enrichment analysis (permutation test, centralities measures etc.)."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from functools import partial
from typing import Any, NamedTuple

import joblib as jl
import networkx as nx
import numpy as np
import pandas as pd
from anndata import AnnData
from numba import njit
from numpy.typing import NDArray
from pandas import CategoricalDtype
from scanpy import logging as logg
from spatialdata import SpatialData

from squidpy._constants._constants import Centrality
from squidpy._constants._pkg_constants import Key
from squidpy._docs import d, inject_docs
from squidpy._utils import NDArrayA, _get_n_cores
from squidpy.gr._utils import (
    _assert_categorical_obs,
    _assert_connectivity_key,
    _assert_positive,
    _save_data,
    _shuffle_group,
)

__all__ = ["nhood_enrichment", "centrality_scores", "interaction_matrix"]


class NhoodEnrichmentResult(NamedTuple):
    """Result of nhood_enrichment function."""

    zscore: NDArray[np.number]
    counts: NDArray[np.number]  # NamedTuple inherits from tuple so cannot use 'count' as attribute name


ndt = np.uint32


def _count(adj: NDArrayA, clustering: NDArrayA, n_cls: int) -> NDArrayA:
    """
    Count how many times clusters ``i`` and ``j`` are connected.

    Equivalent to ``one_hot.T @ adj @ one_hot`` where ``one_hot`` is the
    indicator matrix for ``clustering``.

    Parameters
    ----------
    adj
        Sparse adjacency matrix of shape ``(n_cells, n_cells)``.
    clustering
        Array of shape ``(n_cells,)`` containing cluster labels
        ranging from ``0`` to ``n_cls - 1`` inclusive.
    n_cls
        Number of clusters.

    Returns
    -------
    :class:`numpy.ndarray`
        Array of shape ``(n_clusters, n_clusters)`` containing the pairwise counts.
    """
    from scipy.sparse import csr_matrix

    if n_cls <= 1:
        raise ValueError(f"Expected at least `2` clusters, found `{n_cls}`.")

    n = len(clustering)
    one_hot = csr_matrix(
        (np.ones(n, dtype=ndt), (np.arange(n), clustering)),
        shape=(n, n_cls),
    )
    return np.asarray((one_hot.T @ adj @ one_hot).todense(), dtype=ndt)


@d.get_sections(base="nhood_ench", sections=["Parameters"])
@d.dedent
def nhood_enrichment(
    adata: AnnData | SpatialData,
    cluster_key: str,
    library_key: str | None = None,
    connectivity_key: str | None = None,
    n_perms: int = 1000,
    seed: int | None = None,
    copy: bool = False,
    n_jobs: int | None = None,
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
    %(seed)s
    %(copy)s
    n_jobs
        Number of parallel threads for the permutation loop.
    show_progress_bar
        Whether to show a progress bar.

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
    _assert_positive(n_perms, name="n_perms")

    adj = adata.obsp[connectivity_key]
    original_clust = adata.obs[cluster_key]
    clust_map = {v: i for i, v in enumerate(original_clust.cat.categories.values)}  # map categories
    int_clust = np.array([clust_map[c] for c in original_clust], dtype=ndt)

    if library_key is not None:
        _assert_categorical_obs(adata, key=library_key)
        libraries: pd.Series | None = adata.obs[library_key]
    else:
        libraries = None

    n_cls = len(clust_map)
    count = _count(adj, int_clust, n_cls)

    n_jobs = _get_n_cores(n_jobs)
    start = logg.info(f"Calculating neighborhood enrichment using `{n_jobs}` core(s)")

    perms = _run_permutations(
        adj=adj,
        int_clust=int_clust,
        n_cls=n_cls,
        n_perms=n_perms,
        n_jobs=n_jobs,
        seed=seed,
        libraries=libraries,
        show_progress_bar=show_progress_bar,
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
def centrality_scores(
    adata: AnnData | SpatialData,
    cluster_key: str,
    score: str | Iterable[str] | None = None,
    connectivity_key: str | None = None,
    copy: bool = False,
    n_jobs: int | None = None,
    backend: str = "loky",
) -> pd.DataFrame | None:
    """
    Compute centrality scores per cluster or cell type.

    Inspired by usage in Gene Regulatory Networks (GRNs) in :cite:`celloracle`.

    Parameters
    ----------
    %(adata)s
    %(cluster_key)s
    score
        Centrality measures as described in :mod:`networkx.algorithms.centrality` :cite:`networkx`.
        If `None`, use all the options below. Valid options are:

            - `{c.CLOSENESS.s!r}` - measure of how close the group is to other nodes.
            - `{c.CLUSTERING.s!r}` - measure of the degree to which nodes cluster together.
            - `{c.DEGREE.s!r}` - fraction of non-group members connected to group members.

    %(conn_key)s
    %(copy)s
    n_jobs
        Number of parallel jobs. See :class:`joblib.Parallel` for details.
    backend
        Parallelization backend. See :class:`joblib.Parallel` for available options.

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

    graph = nx.Graph(adata.obsp[connectivity_key])

    cat = adata.obs[cluster_key].cat.categories.values
    clusters = adata.obs[cluster_key].values

    fun_dict = {}
    for c in centralities:
        if c == Centrality.CLOSENESS:
            fun_dict[c.s] = partial(nx.algorithms.centrality.group_closeness_centrality, graph)
        elif c == Centrality.DEGREE:
            fun_dict[c.s] = partial(nx.algorithms.centrality.group_degree_centrality, graph)
        elif c == Centrality.CLUSTERING:
            fun_dict[c.s] = partial(nx.algorithms.cluster.average_clustering, graph)
        else:
            raise NotImplementedError(f"Centrality `{c}` is not yet implemented.")

    n_jobs = _get_n_cores(n_jobs)
    start = logg.info(f"Calculating centralities `{centralities}` using `{n_jobs}` core(s)")

    res_list = []
    for k, v in fun_dict.items():
        scores = jl.Parallel(n_jobs=n_jobs, backend=backend)(
            jl.delayed(_centrality_scores_helper)(c, clusters=clusters, fun=v)
            for c in cat
        )
        res_list.append(pd.DataFrame(scores, columns=[k], index=cat))

    df = pd.concat(res_list, axis=1)

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


def _centrality_scores_helper(
    cat: Any,
    clusters: Sequence[str],
    fun: Callable[..., float],
) -> float:
    idx = np.where(clusters == cat)[0]
    return fun(idx)


def _run_permutations(
    adj: NDArrayA,
    int_clust: NDArrayA,
    n_cls: int,
    n_perms: int,
    n_jobs: int,
    seed: int | None,
    libraries: pd.Series[CategoricalDtype] | None,
    show_progress_bar: bool,
) -> NDArrayA:
    """Run the permutation loop, optionally across multiple threads."""
    from concurrent.futures import ThreadPoolExecutor

    step = int(np.ceil(n_perms / n_jobs))
    all_ixs = np.arange(n_perms)
    chunks = [all_ixs[i * step : (i + 1) * step] for i in range(n_jobs)]
    chunks = [c for c in chunks if len(c)]

    pbar = _get_pbar(show_progress_bar, n_perms)

    with ThreadPoolExecutor(max_workers=len(chunks)) as pool:
        futures = [
            pool.submit(
                _nhood_enrichment_helper,
                ixs=chunk,
                adj=adj,
                int_clust=int_clust,
                n_cls=n_cls,
                libraries=libraries,
                seed=seed,
                pbar=pbar,
            )
            for chunk in chunks
        ]
        results = [f.result() for f in futures]

    if pbar is not None:
        pbar.close()

    return np.vstack(results)


def _get_pbar(show: bool, total: int) -> Any:
    """Create a tqdm progress bar if available and requested."""
    if not show:
        return None
    try:
        import ipywidgets  # noqa: F401
        from tqdm.auto import tqdm
    except ImportError:
        try:
            from tqdm.std import tqdm
        except ImportError:
            return None
    return tqdm(total=total, unit="perm")


def _nhood_enrichment_helper(
    ixs: NDArrayA,
    adj: NDArrayA,
    int_clust: NDArrayA,
    libraries: pd.Series[CategoricalDtype] | None,
    n_cls: int,
    seed: int | None = None,
    pbar: Any = None,
) -> NDArrayA:
    perms = np.empty((len(ixs), n_cls, n_cls), dtype=np.float64)
    int_clust = int_clust.copy()
    rs = np.random.RandomState(seed=None if seed is None else seed + ixs[0])

    for i in range(len(ixs)):
        if libraries is not None:
            int_clust = _shuffle_group(int_clust, libraries, rs)
        else:
            rs.shuffle(int_clust)
        perms[i, ...] = _count(adj, int_clust, n_cls)

        if pbar is not None:
            pbar.update(1)

    return perms
