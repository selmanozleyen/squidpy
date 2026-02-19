"""Functions for neighborhood enrichment analysis (permutation test, centralities measures etc.)."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import partial
from typing import NamedTuple

import networkx as nx
import numba.types as nt
import numpy as np
import pandas as pd
from anndata import AnnData
from numpy.typing import NDArray
from scanpy import logging as logg
from scipy.sparse import csr_matrix
from spatialdata import SpatialData
from tqdm.auto import tqdm

from squidpy._constants._constants import Centrality
from squidpy._constants._pkg_constants import Key
from squidpy._docs import d, inject_docs
from squidpy._utils import NDArrayA
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

    indices, indptr = (adj.indices.astype(ndt), adj.indptr.astype(ndt))
    n_cls = len(clust_map)

    _test = _create_function(n_cls, parallel=numba_parallel)
    count = _test(indices, indptr, int_clust)

    start = logg.info("Calculating neighborhood enrichment")

    rs = np.random.RandomState(seed=seed)
    perms = np.empty((n_perms, n_cls, n_cls), dtype=np.float64)
    for p in range(n_perms):
        clust_copy = int_clust.copy()
        if libraries is not None:
            clust_copy = _shuffle_group(clust_copy, libraries, rs)
        else:
            rs.shuffle(clust_copy)
        perms[p] = _test(indices, indptr, clust_copy)

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
    show_progress_bar: bool = False,
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
    show_progress_bar
        Whether to show a progress bar.

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

    start = logg.info(f"Calculating centralities `{centralities}`")

    res_list = []
    for k, v in fun_dict.items():
        scores = []
        iterator = tqdm(cat, disable=not show_progress_bar)
        for c in iterator:
            idx = np.where(clusters == c)[0]
            scores.append(v(idx))
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

    if not weights:
        g = g.astype(bool).astype(int)

    codes = cats.cat.codes.to_numpy()
    n_cells = len(codes)
    indicator = csr_matrix((np.ones(n_cells, dtype=g.dtype), (np.arange(n_cells), codes)), shape=(n_cells, n_cats))
    output = (indicator.T @ g @ indicator).toarray()

    if normalized:
        output = output / output.sum(axis=1).reshape((-1, 1))

    if copy:
        return output

    _save_data(adata, attr="uns", key=Key.uns.interaction_matrix(cluster_key), data=output)
