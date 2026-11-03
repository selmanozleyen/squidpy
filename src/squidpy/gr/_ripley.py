"""Functions for point patterns spatial statistics."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from anndata import AnnData
from scanpy import logging as logg
from scipy.spatial import ConvexHull, Delaunay, cKDTree
from sklearn.neighbors import KDTree, NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from spatialdata import SpatialData

from squidpy._compat import old_positionals
from squidpy._constants._pkg_constants import Key
from squidpy._docs import d
from squidpy._utils import NDArrayA, RNGLike, SeedLike, deprecated_randomness_param, get_n_numba_threads, thread_map
from squidpy._validators import normalize_choice
from squidpy.gr._utils import (
    _assert_categorical_obs,
    _assert_spatial_basis,
    _count_pairs,
    _save_data,
    extract_adata_if_sdata,
)

__all__ = ["ripley"]


@d.dedent
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
    n_jobs: int | None = None,
    copy: bool = False,
    table_key: str | None = None,
) -> dict[str, pd.DataFrame | NDArrayA]:
    r"""
    Calculate various Ripley's statistics for point processes.

    %(seed_versionchanged)s

    %(rng_versionchanged)s

    According to the `'mode'` argument, it calculates one of the following Ripley's statistics:
    `'F'`, `'G'` or `'L'` statistics.

    `'F'`, `'G'` are defined as:

    .. math::

        F(t),G(t)=P( d_{i,j} \le t )

    Where :math:`d_{i,j}` represents:

        - distances to a random Spatial Poisson Point Process for `'F'`.
        - distances to any other point of the dataset for `'G'`.

    `'L'` we first need to compute :math:`K(t)`, which is defined as:

    .. math::

        K(t) = \frac{1}{\lambda} \sum_{i \ne j} \frac{I(d_{i,j}<t)}{n}


    and then we apply a variance-stabilizing transformation:

    .. math::

        L(t) = (\frac{K(t)}{\pi})^{1/2}


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
        Maximum distances for the support. If `None`, `max_dist=`:math:`\sqrt{area \over 2}`.
    n_steps
        Number of steps for the support.
    %(rng)s
    %(n_jobs_threads)s
        Only metrics scipy's KD-tree supports run in threads; others run serially.
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

    mode = normalize_choice(mode, ("F", "G", "L"), name="mode")

    # prepare support
    N = coordinates.shape[0]
    hull = ConvexHull(coordinates)
    area = hull.volume
    if max_dist is None:
        max_dist = (area / 2) ** 0.5
    support = np.linspace(0, max_dist, n_steps)

    # prepare labels
    le = LabelEncoder().fit(clusters)
    cluster_idx = le.transform(clusters)
    obs_arr = np.empty((le.classes_.shape[0], n_steps))

    start = logg.info(
        f"Calculating Ripley's {mode} statistic for `{le.classes_.shape[0]}` clusters and `{n_simulations}` simulations"
    )
    obs_rng, *sim_rngs = np.random.default_rng(rng).spawn(n_simulations + 1)

    n_clusters = len(le.classes_)
    coords_c = [coordinates[cluster_idx == i, :] for i in range(n_clusters)]
    if mode != "L":  # scipy's kNN returns infinite distances for missing neighbours instead of raising
        sizes = np.array([len(c) for c in coords_c])
        if (small := np.flatnonzero(sizes < n_neigh)).size:
            raise ValueError(
                f"Ripley's {mode} needs at least `n_neigh={n_neigh}` observations per cluster, but "
                + ", ".join(f"`{le.classes_[i]}` has {sizes[i]}" for i in small)
                + ". Remove these clusters or lower `n_neigh`."
            )
    if mode == "F":
        randoms = _ppp(hull, n_simulations=n_clusters, n_observations=n_observations, rng=obs_rng)
        random = randoms[-1]  # the simulations query the last cluster's random points

    def observed(i: int) -> NDArrayA:
        queries = randoms[i] if mode == "F" else coordinates[cluster_idx != i, :]
        return _f_g_function(_knn_distances(coords_c[i], queries, n_neigh, metric), support)

    # threads only through scipy, which releases the GIL; sklearn's trees are run serially
    n_jobs = get_n_numba_threads(n_jobs) if metric in _SCIPY_P else 1
    if mode == "L":
        obs_arr[:] = _l_estimate(_pair_counts(coords_c, support, metric, n_jobs), N, area)
    else:
        obs_arr[:] = thread_map(observed, range(n_clusters), n_jobs=n_jobs)

    sims = np.empty((n_simulations, n_steps))
    pvalues = np.ones((n_clusters, n_steps))

    def simulated(i: int) -> NDArrayA:
        random_i = _ppp(hull, n_simulations=1, n_observations=n_observations, rng=sim_rngs[i])[0]
        if mode == "L":
            return _l_estimate(_pair_counts([random_i], support, metric, n_jobs=1), N, area)[0]
        return _f_g_function(_knn_distances(random_i, random if mode == "F" else coordinates, 1, metric), support)

    # every simulation has its own generator, so they run in any order and give the same draws
    for i, stats_i in enumerate(thread_map(simulated, range(n_simulations), n_jobs=n_jobs)):
        for j in range(obs_arr.shape[0]):
            pvalues[j] += stats_i >= obs_arr[j]
        sims[i] = stats_i

    pvalues /= n_simulations + 1
    pvalues = np.minimum(pvalues, 1 - pvalues)

    obs_df = _reshape_res(obs_arr.T, columns=le.classes_, index=support, var_name=cluster_key)
    sims_df = _reshape_res(sims.T, columns=np.arange(n_simulations), index=support, var_name="simulations")

    res = {f"{mode}_stat": obs_df, "sims_stat": sims_df, "bins": support, "pvalues": pvalues}

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


def _f_g_function(distances: NDArrayA, support: NDArrayA) -> NDArrayA:
    counts, _ = np.histogram(distances, bins=support)
    fracs = np.cumsum(counts) / counts.sum()
    return np.concatenate((np.zeros((1,), dtype=float), fracs))


# Minkowski p of the metrics scipy's cKDTree supports; the rest go through sklearn's KDTree
_SCIPY_P = {
    "euclidean": 2.0,
    "l2": 2.0,
    "manhattan": 1.0,
    "cityblock": 1.0,
    "l1": 1.0,
    "minkowski": 2.0,
    "chebyshev": np.inf,
    "infinity": np.inf,
}


def _knn_distances(points: NDArrayA, queries: NDArrayA, k: int, metric: str) -> NDArrayA:
    """Distances from every query to its ``k`` nearest ``points``.

    scipy's :class:`~scipy.spatial.cKDTree` for the metrics it covers: it releases the GIL and is safe
    to call from many threads. sklearn's ``NearestNeighbors`` otherwise, which changes threadpool
    limits globally and so must not run in concurrent threads.
    """
    if metric in _SCIPY_P:
        return cKDTree(points).query(queries, k=k, p=_SCIPY_P[metric])[0]
    return NearestNeighbors(metric=metric, n_neighbors=k).fit(points).kneighbors(queries, n_neighbors=k)[0]


def _pair_counts(coords: list[NDArrayA], support: NDArrayA, metric: str, n_jobs: int) -> NDArrayA:
    """Ordered pairs ``i != j`` within each ``support`` distance, for every point set in ``coords``.

    Ripley's K(d) counts exactly these. scipy's dual-tree :meth:`~scipy.spatial.cKDTree.count_neighbors`
    computes them in threads (see :func:`~squidpy.gr._utils._count_pairs`). sklearn's
    ``two_point_correlation`` holds the GIL; it remains the path for metrics scipy does not cover.
    """
    if metric not in KDTree.valid_metrics:
        raise ValueError(f"Unsupported metric '{metric}'. Ripley's L supports {KDTree.valid_metrics}")
    if metric not in _SCIPY_P:
        counts = [KDTree(pts, metric=metric).two_point_correlation(pts, support, dualtree=True) for pts in coords]
        return np.array(counts) - np.array([len(pts) for pts in coords])[:, None]  # drop the self-pairs at d = 0
    return _count_pairs(coords, [(c, c) for c in range(len(coords))], support, p=_SCIPY_P[metric], n_jobs=n_jobs)


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
    lo, hi = vxs.min(0), vxs.max(0)

    # Rejection sampling in batches, one `find_simplex` call each. `uniform` with interleaved x/y
    # bounds draws the same values in the same order as one scalar call per coordinate, so the
    # accepted points are those of a point-by-point loop over the same generator.
    accept = hull.volume / np.prod(hi - lo)
    n = n_simulations * n_observations
    batches, n_hits = [], 0
    while n_hits < n:
        m = int((n - n_hits) / accept * 1.2) + 64
        xy = rng.uniform(np.tile(lo, m), np.tile(hi, m)).reshape(m, 2)
        batches.append(xy[deln.find_simplex(xy) >= 0])
        n_hits += len(batches[-1])
    return np.concatenate(batches)[:n].reshape(n_simulations, n_observations, 2)
