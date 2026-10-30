from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from squidpy.gr import ripley

CLUSTER_KEY = "leiden"


def test_ripley_L_unsupported_metric_raises(adata_ripley: AnnData):
    with pytest.raises(ValueError, match="Unsupported metric"):
        ripley(adata_ripley, cluster_key=CLUSTER_KEY, mode="L", metric="cosine")


@pytest.mark.parametrize("mode", ["F", "G", "L"])
def test_ripley_modes(adata_ripley: AnnData, mode: str):
    adata = adata_ripley

    ripley(adata, cluster_key=CLUSTER_KEY, mode=mode)

    UNS_KEY = f"{CLUSTER_KEY}_ripley_{mode}"

    # assert uns
    assert UNS_KEY in adata.uns.keys()
    assert f"{mode}_stat" in adata.uns[UNS_KEY].keys()
    assert "sims_stat" in adata.uns[UNS_KEY].keys()
    assert "bins" in adata.uns[UNS_KEY].keys()
    assert "pvalues" in adata.uns[UNS_KEY].keys()

    obs_df = adata.uns[UNS_KEY][f"{mode}_stat"]
    sims_df = adata.uns[UNS_KEY]["sims_stat"]
    bins = adata.uns[UNS_KEY]["bins"]
    pvalues = adata.uns[UNS_KEY]["pvalues"]

    # assert shapes
    np.testing.assert_array_equal(adata.obs[CLUSTER_KEY].cat.categories, obs_df[CLUSTER_KEY].cat.categories)
    assert obs_df.shape[1] == sims_df.shape[1]
    assert pvalues.shape[0] == adata.obs[CLUSTER_KEY].cat.categories.shape[0]
    assert bins.shape[0] == 50


@pytest.mark.parametrize("mode", ["F", "G", "L"])
@pytest.mark.parametrize(
    "n_simulations",
    [20, 50],
)
@pytest.mark.parametrize(
    "n_observations",
    [10, 100],
)
@pytest.mark.parametrize(
    "max_dist",
    [None, 1000],
)
@pytest.mark.parametrize(
    "n_steps",
    [2, 50, 100],
)
def test_ripley_results(
    adata_ripley: AnnData, mode: str, n_simulations: int, n_observations: int, max_dist: np.float_, n_steps: int
):
    adata = adata_ripley
    n_clusters = adata.obs[CLUSTER_KEY].cat.categories.shape[0]

    res = ripley(
        adata,
        cluster_key=CLUSTER_KEY,
        mode=mode,
        n_simulations=n_simulations,
        n_observations=n_observations,
        max_dist=max_dist,
        n_steps=n_steps,
        copy=True,
    )

    obs_df = res[f"{mode}_stat"]
    sims_df = res["sims_stat"]
    bins = res["bins"]
    pvalues = res["pvalues"]

    # assert shapes
    assert obs_df.shape == (n_steps * n_clusters, 3)
    assert bins.shape == (n_steps,)
    assert sims_df.shape == (n_steps * n_simulations, 3)
    assert pvalues.shape == (n_clusters, n_steps)

    # assert first values is 0
    assert sims_df.bins.values[0] == 0.0
    assert sims_df.bins.values[0] == obs_df.bins.values[0]
    assert sims_df.stats.values[0] == 0.0
    assert sims_df.stats.values[0] == obs_df.stats.values[0]

    # assert n_zeros == n_clusters
    idx = np.nonzero(obs_df.bins.values)[0]
    assert idx.shape[0] == n_steps * n_clusters - n_clusters


@pytest.mark.parametrize("mode", ["F", "G", "L"])
def test_ripley_rng(adata_ripley: AnnData, mode: str):
    """Same seed reproduces simulations, different seeds change them, and simulations are not all identical."""
    adata = adata_ripley
    kw = {"cluster_key": CLUSTER_KEY, "mode": mode, "n_simulations": 20, "copy": True}

    res1 = ripley(adata, rng=np.random.default_rng(42), **kw)
    res2 = ripley(adata, rng=np.random.default_rng(42), **kw)
    res3 = ripley(adata, rng=np.random.default_rng(43), **kw)

    sims1 = res1["sims_stat"].pivot(index="bins", columns="simulations", values="stats").to_numpy()
    sims2 = res2["sims_stat"].pivot(index="bins", columns="simulations", values="stats").to_numpy()
    sims3 = res3["sims_stat"].pivot(index="bins", columns="simulations", values="stats").to_numpy()

    # same seed -> identical simulations
    np.testing.assert_array_equal(sims1, sims2)
    # different seed -> different simulations
    assert not np.array_equal(sims1, sims3)
    # regression: simulations must not all be identical to each other
    # (previously every simulation reused the same seed and produced the same point pattern)
    assert not np.allclose(sims1, sims1[:, [0]])


def test_ppp_batches_match_pointwise_sampling():
    """Batched rejection sampling yields the points of a point-by-point loop over the same generator."""
    from scipy.spatial import ConvexHull, Delaunay

    from squidpy.gr._ripley import _ppp

    pts = np.random.default_rng(1).normal(size=(200, 2)) * [3, 1]  # elongated hull: many rejections
    hull = ConvexHull(pts)
    vxs = hull.points[hull.vertices]
    deln, bbox = Delaunay(vxs), np.array([*vxs.min(0), *vxs.max(0)])

    for n_obs in (1, 37, 1000):
        ref_rng, rng = np.random.default_rng(7), np.random.default_rng(7)
        expected = []
        while len(expected) < n_obs:
            x, y = ref_rng.uniform(bbox[0], bbox[2]), ref_rng.uniform(bbox[1], bbox[3])
            if deln.find_simplex((x, y)) >= 0:
                expected.append((x, y))
        got = _ppp(hull, n_simulations=1, n_observations=n_obs, rng=rng)
        np.testing.assert_array_equal(got, np.reshape(expected, (1, n_obs, 2)))


@pytest.mark.parametrize("mode", ["F", "G"])
def test_ripley_tiny_cluster_raises(adata_ripley: AnnData, mode: str):
    """F and G need n_neigh points in every cluster; a smaller one is named in a clear error."""
    adata = adata_ripley.copy()
    labels = adata.obs[CLUSTER_KEY].astype(str).to_numpy()
    labels[0] = "tiny"
    adata.obs[CLUSTER_KEY] = pd.Categorical(labels)
    with pytest.raises(ValueError, match=r"`tiny` has 1\. Remove these clusters or lower `n_neigh`"):
        ripley(adata, cluster_key=CLUSTER_KEY, mode=mode, n_simulations=2, copy=True)
    ripley(adata, cluster_key=CLUSTER_KEY, mode=mode, n_neigh=1, n_simulations=2, copy=True)  # fits again


@pytest.mark.parametrize("metric", ["euclidean", "manhattan", "chebyshev"])
@pytest.mark.parametrize("k", [1, 2])
def test_knn_distances_match_sklearn(metric: str, k: int):
    """scipy's thread-safe nearest-neighbour path gives sklearn's distances."""
    from sklearn.neighbors import NearestNeighbors

    from squidpy.gr._ripley import _knn_distances

    rng = np.random.default_rng(0)
    points, queries = rng.uniform(0, 100, (500, 2)), rng.uniform(0, 100, (300, 2))
    expected = NearestNeighbors(metric=metric, n_neighbors=k).fit(points).kneighbors(queries, n_neighbors=k)[0]
    np.testing.assert_allclose(_knn_distances(points, queries, k, metric).reshape(300, k), expected, rtol=1e-12)


@pytest.mark.parametrize("mode", ["F", "G", "L"])
def test_ripley_n_jobs_invariant(adata_ripley: AnnData, mode: str):
    """The result does not depend on the number of threads."""
    kw = {"cluster_key": CLUSTER_KEY, "mode": mode, "n_simulations": 5, "rng": 0, "copy": True}
    serial, threaded = ripley(adata_ripley, n_jobs=1, **kw), ripley(adata_ripley, n_jobs=4, **kw)
    for key in serial:
        if isinstance(serial[key], pd.DataFrame):
            pd.testing.assert_frame_equal(serial[key], threaded[key])
        else:
            np.testing.assert_array_equal(serial[key], threaded[key])
