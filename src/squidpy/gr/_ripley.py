"""Functions for point patterns spatial statistics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from anndata import AnnData
from scanpy import logging as logg
from scipy.spatial import ConvexHull, Delaunay, cKDTree
from sklearn.neighbors import KDTree, NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from spatialdata import SpatialData

from squidpy._compat import old_positionals
from squidpy._constants._constants import RipleyStat
from squidpy._constants._pkg_constants import Key
from squidpy._docs import d, inject_docs
from squidpy._utils import (
    NDArrayA,
    RNGLike,
    SeedLike,
    deprecated_randomness_param,
    get_n_numba_threads,
    thread_map,
)
from squidpy.gr._utils import (
    _assert_categorical_obs,
    _assert_spatial_basis,
    _rmax_rule,
    _save_data,
    extract_adata_if_sdata,
)

__all__ = ["ripley"]


@d.dedent
@inject_docs(key=Key.obsm.spatial, rp=RipleyStat)
@old_positionals(
    "cluster_key",
    "mode",
    "spatial_key",
    "metric",
    "n_neigh",
    "n_simulations",
    "n_observations",
    "max_dist",
    "n_steps",
    "seed",
    "copy",
)
@deprecated_randomness_param
def ripley(
    adata: AnnData | SpatialData,
    *,
    cluster_key: str,
    mode: Literal["F", "G", "L"] = "F",
    spatial_key: str = Key.obsm.spatial,
    metric: str = "euclidean",
    n_neigh: int = 2,
    n_simulations: int = 100,
    n_observations: int = 1000,
    max_dist: float | None = None,
    n_steps: int = 50,
    rng: SeedLike | RNGLike | None = None,
    copy: bool = False,
    table_key: str | None = None,
) -> dict[str, pd.DataFrame | NDArrayA]:
    r"""
    Calculate various Ripley's statistics for point processes.

    %(seed_versionchanged)s

    %(rng_versionchanged)s

    According to the `'mode'` argument, it calculates one of the following Ripley's statistics:
    `{rp.F.s!r}`, `{rp.G.s!r}` or `{rp.L.s!r}` statistics.

    `{rp.F.s!r}`, `{rp.G.s!r}` are defined as:

    .. math::

        F(t),G(t)=P( d_{{i,j}} \le t )

    Where :math:`d_{{i,j}}` represents:

        - distances to a random Spatial Poisson Point Process for `{rp.F.s!r}`.
        - distances to any other point of the dataset for `{rp.G.s!r}`.

    `{rp.L.s!r}` we first need to compute :math:`K(t)`, which is defined as:

    .. math::

        K(t) = \frac{{1}}{{\lambda}} \sum_{{i \ne j}} \frac{{I(d_{{i,j}}<t)}}{{n}}


    and then we apply a variance-stabilizing transformation:

    .. math::

        L(t) = (\frac{{K(t)}}{{\pi}})^{{1/2}}


    Parameters
    ----------
    %(adata)s
    %(table_key)s
    %(cluster_key)s
    mode
        Which Ripley's statistic to compute.
    %(spatial_key)s
    metric
        Which metric to use for computing distances.
        For available metrics, check out :class:`sklearn.metrics.DistanceMetric`.
        For Ripley's L specifically, only metrics supported by :class:`sklearn.neighbors.KDTree`
        are valid (see its ``valid_metrics`` attribute).
    n_neigh
        Number of neighbors to consider for the KNN graph.
    n_simulations
        How many simulations to run for computing p-values.
    n_observations
        How many observations to generate for the Spatial Poisson Point Process.
    max_dist
        Maximum distances for the support. If `None`, spatstat's rule (``rmax.rule``) with the area of
        the convex hull: for L, a quarter of the bounding box's shorter side, but at most the radius
        holding 1000 points around a typical point; for F and G, half the bounding box's diagonal, but
        at most the radius by which a Poisson process of the same intensity has :math:`F(r) = 1 - 10^{{-5}}`.
    n_steps
        Number of steps for the support.
    %(rng)s
    %(copy)s

    Returns
    -------
    %(ripley_stat_returns)s

    References
    ----------
    For reference, check out
    `Wikipedia <https://en.wikipedia.org/wiki/Spatial_descriptive_statistics#Ripley's_K_and_L_functions>`_
    or :cite:`Baddeley2015-lm`.
    """
    adata = extract_adata_if_sdata(adata, table_key=table_key)
    _assert_categorical_obs(adata, key=cluster_key)
    _assert_spatial_basis(adata, key=spatial_key)
    coordinates = adata.obsm[spatial_key]
    clusters = adata.obs[cluster_key].values

    mode = RipleyStat(mode)  # type: ignore[assignment]
    if TYPE_CHECKING:
        assert isinstance(mode, RipleyStat)

    # prepare support
    N = coordinates.shape[0]
    hull = ConvexHull(coordinates)
    area = hull.volume
    if max_dist is None:
        max_dist = _rmax_rule(coordinates, area=area, nearest=mode != RipleyStat.L)
    support = np.linspace(0, max_dist, n_steps)

    # prepare labels
    le = LabelEncoder().fit(clusters)
    cluster_idx = le.transform(clusters)
    obs_arr = np.empty((le.classes_.shape[0], n_steps))

    start = logg.info(
        f"Calculating Ripley's {mode} statistic for `{le.classes_.shape[0]}` clusters and `{n_simulations}` simulations"
    )
    obs_rng, *sim_rngs = np.random.default_rng(rng).spawn(n_simulations + 1)

    n_jobs = get_n_numba_threads(None)
    n_clusters = int(np.max(cluster_idx)) + 1
    coords_c = [coordinates[cluster_idx == i, :] for i in range(n_clusters)]
    if mode != RipleyStat.L:
        sizes = np.array([len(c) for c in coords_c])
        if (small := np.flatnonzero(sizes < n_neigh)).size:
            raise ValueError(
                f"Ripley's {mode.s} needs at least `n_neigh={n_neigh}` observations per cluster, but "
                + ", ".join(f"`{le.classes_[i]}` has {sizes[i]}" for i in small)
                + ". Remove these clusters or lower `n_neigh`."
            )
    if mode == RipleyStat.F:
        # drawn in cluster order from the one generator, as before; the last draw is reused below
        randoms = [_ppp(hull, n_simulations=1, n_observations=n_observations, rng=obs_rng) for _ in range(n_clusters)]
        random = randoms[-1]

    def observed(i: int) -> NDArrayA:
        if mode == RipleyStat.F:
            tree_c = NearestNeighbors(metric=metric, n_neighbors=n_neigh).fit(coords_c[i])
            distances, _ = tree_c.kneighbors(randoms[i], n_neighbors=n_neigh)
            return _f_g_function(distances.squeeze(), support)[1]
        if mode == RipleyStat.G:
            tree_c = NearestNeighbors(metric=metric, n_neighbors=n_neigh).fit(coords_c[i])
            distances, _ = tree_c.kneighbors(coordinates[cluster_idx != i, :], n_neighbors=n_neigh)
            return _f_g_function(distances.squeeze(), support)[1]
        raise NotImplementedError(f"Mode `{mode.s!r}` is not yet implemented.")

    if mode == RipleyStat.L:
        obs_arr[:] = _l_estimate(_pair_counts(coords_c, support, metric, n_jobs), N, area)
    else:
        obs_arr[:] = thread_map(observed, range(n_clusters), n_jobs=n_jobs)
    bins = support  # np.histogram returns the given edges unchanged, so F/G bins are the support too

    sims = np.empty((n_simulations, len(bins)))
    pvalues = np.ones((le.classes_.shape[0], len(bins)))

    def simulated(i: int) -> NDArrayA:
        random_i = _ppp(hull, n_simulations=1, n_observations=n_observations, rng=sim_rngs[i])
        if mode == RipleyStat.F:
            tree_i = NearestNeighbors(metric=metric, n_neighbors=n_neigh).fit(random_i)
            distances_i, _ = tree_i.kneighbors(random, n_neighbors=1)
            return _f_g_function(distances_i.squeeze(), support)[1]
        if mode == RipleyStat.G:
            tree_i = NearestNeighbors(metric=metric, n_neighbors=n_neigh).fit(random_i)
            distances_i, _ = tree_i.kneighbors(coordinates, n_neighbors=1)
            return _f_g_function(distances_i.squeeze(), support)[1]
        if mode == RipleyStat.L:
            return _l_estimate(_pair_counts([random_i], support, metric, n_jobs=1), N, area)[0]
        raise NotImplementedError(f"Mode `{mode.s!r}` is not yet implemented.")

    # every simulation has its own generator, so they run in any order and give the same draws
    for i, stats_i in enumerate(thread_map(simulated, range(n_simulations), n_jobs=n_jobs)):
        for j in range(obs_arr.shape[0]):
            pvalues[j] += stats_i >= obs_arr[j]
        sims[i] = stats_i

    pvalues /= n_simulations + 1
    pvalues = np.minimum(pvalues, 1 - pvalues)

    obs_df = _reshape_res(obs_arr.T, columns=le.classes_, index=bins, var_name=cluster_key)
    sims_df = _reshape_res(sims.T, columns=np.arange(n_simulations), index=bins, var_name="simulations")

    res = {f"{mode}_stat": obs_df, "sims_stat": sims_df, "bins": bins, "pvalues": pvalues}

    if TYPE_CHECKING:
        assert isinstance(res, dict)

    if copy:
        logg.info("Finish", time=start)
        return res

    _save_data(adata, attr="uns", key=Key.uns.ripley(cluster_key, mode), data=res, time=start)


def _reshape_res(results: NDArrayA, columns: NDArrayA | list[str], index: NDArrayA, var_name: str) -> pd.DataFrame:
    df = pd.DataFrame(results, columns=columns, index=index)
    df.index.set_names(["bins"], inplace=True)
    df = df.melt(var_name=var_name, value_name="stats", ignore_index=False)
    df[var_name] = df[var_name].astype("category")
    df.reset_index(inplace=True)
    return df


def _f_g_function(distances: NDArrayA, support: NDArrayA) -> tuple[NDArrayA, NDArrayA]:
    counts, bins = np.histogram(distances, bins=support)
    fracs = np.cumsum(counts) / counts.sum()
    return bins, np.concatenate((np.zeros((1,), dtype=float), fracs))


# Minkowski p of the metrics scipy's cKDTree supports; the rest go through sklearn's KDTree
_SCIPY_P = {
    "euclidean": 2.0,
    "l2": 2.0,
    "manhattan": 1.0,
    "cityblock": 1.0,
    "l1": 1.0,
    "chebyshev": np.inf,
    "infinity": np.inf,
}


def _pair_counts(coords: list[NDArrayA], support: NDArrayA, metric: str, n_jobs: int) -> NDArrayA:
    """Ordered pairs ``i != j`` within each ``support`` distance, for every point set in ``coords``.

    Ripley's K(d) counts exactly these. scipy's dual-tree :meth:`~scipy.spatial.cKDTree.count_neighbors`
    computes them cumulatively over ``support`` without materializing pairwise distances and releases
    the GIL, so large point sets are cut into spatial chunks (x-sorted) that are counted in threads
    against the whole set. sklearn's ``two_point_correlation`` holds the GIL; it remains the path for
    metrics scipy does not cover.
    """
    if metric not in _SCIPY_P and metric not in KDTree.valid_metrics:
        raise ValueError(f"Unsupported metric '{metric}'. Ripley's L supports {KDTree.valid_metrics}")
    counts = np.zeros((len(coords), len(support)), dtype=np.int64)
    if metric not in _SCIPY_P:
        for c, pts in enumerate(coords):
            counts[c] = KDTree(pts, metric=metric).two_point_correlation(pts, support, dualtree=True)
        return counts - np.array([len(pts) for pts in coords])[:, None]  # drop the self-pairs at d = 0

    p = _SCIPY_P[metric]
    trees = [cKDTree(pts) for pts in coords]
    work = [
        (c, chunk)
        for c, pts in enumerate(coords)
        for chunk in np.array_split(np.argsort(pts[:, 0], kind="stable"), max(1, min(n_jobs, len(pts) // 20_000)))
    ]

    def count(item: tuple[int, NDArrayA]) -> tuple[int, NDArrayA]:
        c, chunk = item
        return c, cKDTree(coords[c][chunk]).count_neighbors(trees[c], support, p=p, cumulative=True)

    for c, cnt in thread_map(count, work, n_jobs=n_jobs):
        counts[c] += cnt
    return counts - np.array([len(pts) for pts in coords])[:, None]  # drop the self-pairs at d = 0


def _l_estimate(pair_counts: NDArrayA, n: int, area: float) -> NDArrayA:
    """Ripley's L from ordered-pair counts: K(d) = pairs / (n * intensity), L = sqrt(K / pi)."""
    intensity = n / area
    k_estimate = (pair_counts / n) / intensity
    return np.sqrt(k_estimate / np.pi)


def _ppp(
    hull: ConvexHull,
    n_simulations: int,
    n_observations: int,
    rng: np.random.Generator,
) -> NDArrayA:
    """
    Simulate Poisson Point Process on a polygon.

    Parameters
    ----------
    hull
        Convex hull of the area of interest.
    n_simulations
        Number of simulated point processes.
    n_observations
        Number of observations to sample from each simulation.
    rng
        Independent :class:`numpy.random.Generator` used to draw the points.

    Returns
    -------
    An Array with shape ``(n_simulation, n_observations, 2)``.
    """
    vxs = hull.points[hull.vertices]
    deln = Delaunay(vxs)

    bbox = np.array([*vxs.min(0), *vxs.max(0)])
    result = np.empty((n_simulations, n_observations, 2))

    # Rejection sampling in batches with one `find_simplex` call each, instead of a Python loop per
    # point. Candidate k uses the generator's draws 2k and 2k + 1 for x and y, as the point-wise loop
    # did, and the generator ends exactly where that loop left it (later draws see the same stream).
    accept = max(ConvexHull(vxs).volume / ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])), 1e-3)
    for i_sim in range(n_simulations):
        i_obs = 0
        while i_obs < n_observations:
            need = n_observations - i_obs
            state = rng.bit_generator.state
            m = max(64, int(need / accept * 1.2) + 16)
            # `uniform` itself, with x/y bounds interleaved: the same routine (and rounding) as the
            # scalar calls, consuming one draw per value in the same order
            xy = rng.uniform(np.tile(bbox[:2], m), np.tile(bbox[2:], m)).reshape(m, 2)
            hits = np.flatnonzero(deln.find_simplex(xy) >= 0)
            if len(hits) >= need:
                hits = hits[:need]
                rng.bit_generator.state = state  # rewind, then consume only the draws that were used
                rng.random(2 * (int(hits[-1]) + 1))  # `uniform` consumes one double per value too
            result[i_sim, i_obs : i_obs + len(hits)] = xy[hits]
            i_obs += len(hits)

    return result.squeeze()
