from __future__ import annotations

import numpy as np
import pytest
from anndata import AnnData
from numba import njit
from pandas.testing import assert_frame_equal
from scipy.sparse import csr_matrix

from squidpy.gr import sepal, spatial_neighbors_grid, spatial_neighbors_radius
from squidpy.gr._sepal import _compute_idxs, _diffusion

UNS_KEY = "sepal_score"


def test_sepal_seq_par(adata: AnnData):
    """Check whether sepal results are the same for seq. and parallel computation."""
    spatial_neighbors_grid(adata)
    rng = np.random.default_rng(42)
    adata.var["highly_variable"] = rng.choice([True, False], size=adata.var_names.shape, p=[0.005, 0.995])

    sepal(adata, max_neighs=6)

    df = sepal(adata, max_neighs=6, copy=True)
    df_parallel = sepal(adata, max_neighs=6, copy=True, n_jobs=2)

    idx_df = df.index.values
    idx_adata = adata[:, adata.var.highly_variable.values].var_names.values

    assert UNS_KEY in adata.uns.keys()
    assert df.columns.shape == (1,)
    # test highly variable
    assert adata.uns[UNS_KEY].shape == df.shape
    # assert idx are sorted and contain same elements
    assert not np.array_equal(idx_df, idx_adata)
    np.testing.assert_array_equal(sorted(idx_df), sorted(idx_adata))
    # check parallel gives same results
    assert_frame_equal(df, df_parallel)


def test_sepal_square_seq_par(adata_squaregrid: AnnData):
    """Test sepal for square grid."""
    adata = adata_squaregrid
    spatial_neighbors_radius(adata, radius=1.0)
    rng = np.random.default_rng(42)
    adata.var["highly_variable"] = rng.choice([True, False], size=adata.var_names.shape)

    sepal(adata, max_neighs=4)
    df_parallel = sepal(adata, copy=True, max_neighs=4, n_jobs=2)

    idx_df = df_parallel.index.values
    idx_adata = adata[:, adata.var.highly_variable.values].var_names.values

    assert UNS_KEY in adata.uns.keys()
    assert df_parallel.columns.shape == (1,)
    # test highly variable
    assert adata.uns[UNS_KEY].shape == df_parallel.shape
    # assert idx are sorted and contain same elements
    assert not np.array_equal(idx_df, idx_adata)
    np.testing.assert_array_equal(sorted(idx_df), sorted(idx_adata))
    # check parallel gives same results
    assert_frame_equal(adata.uns[UNS_KEY], df_parallel)


def test_sepal_dense(adata: AnnData):
    """Check whether sepal results are identical for sparse and dense data."""
    spatial_neighbors_grid(adata)
    rng = np.random.default_rng(42)
    adata.var["highly_variable"] = rng.choice([True, False], size=adata.var_names.shape, p=[0.05, 0.95])

    # Compute sepal score for sparse data
    df_sparse = sepal(adata, max_neighs=6, copy=True)

    # Convert to dense and compute sepal score
    adata.X = adata.X.toarray()
    df_dense = sepal(adata, max_neighs=6, copy=True)

    # Assert results are identical
    assert_frame_equal(df_sparse, df_dense)


@njit(fastmath=True)
def _diffusion_vectorized(conc, use_hex, n_iter, sat, sat_idx, unsat, unsat_idx, dt, thresh):  # noqa: PLR0917
    """The array-expression kernel `_diffusion` replaced, kept here as the reference it must match."""
    entropy_arr = np.zeros(n_iter)
    nhood = np.zeros(sat.shape[0])
    dcdt = np.zeros(conc.shape[0])
    eps = np.finfo(np.float64).eps
    prev_ent = 1.0
    for i in range(n_iter):
        for j in range(sat.shape[0]):
            nhood[j] = np.sum(conc[sat_idx[j]])
        centers = conc[sat]
        d2 = (2.0 * nhood - 12.0 * centers) / 3.0 if use_hex else nhood - 4 * centers
        dcdt[:] = 0.0
        dcdt[sat] = d2
        conc[sat] += dcdt[sat] * dt
        conc[unsat] += dcdt[unsat_idx] * dt
        conc[conc < 0] = 0
        xx = conc[sat]
        xnz = xx[xx > 0]
        xs = np.sum(xnz)
        ent = 0.0
        if xs >= eps:
            xn = xnz / xs
            ent = (-np.log(np.maximum(xn, eps)) * xn).sum()
        ent = ent / sat.shape[0]
        entropy_arr[i] = np.abs(ent - prev_ent)
        prev_ent = ent
        if entropy_arr[i] <= thresh:
            break
    tmp = np.nonzero(entropy_arr <= thresh)[0]
    return float(tmp[0] if len(tmp) else np.nan)


@pytest.mark.parametrize(("fixture", "max_neighs"), [("adata", 6), ("adata_squaregrid", 4)])
def test_sepal_kernel_matches_reference(request: pytest.FixtureRequest, fixture: str, max_neighs: int):
    """The block kernel settles at exactly the step the vectorized one did, alone or in a block."""
    adata = request.getfixturevalue(fixture)
    if max_neighs == 6:
        spatial_neighbors_grid(adata)
    else:
        spatial_neighbors_radius(adata, radius=1.0)
    g = csr_matrix(adata.obsp["spatial_connectivities"])
    g.eliminate_zeros()
    sat, sat_idx, unsat, unsat_idx = _compute_idxs(g, adata.obsm["spatial"].astype(np.float64), max_neighs, "l1")
    x = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    genes = np.flatnonzero(x.sum(0) > 0)[:20]
    args = (max_neighs == 6, 30000, sat, sat_idx, unsat, unsat_idx, 0.001, 1e-8)
    want = [_diffusion_vectorized(x[:, g].astype(np.float64), *args) for g in genes]
    together = _diffusion(np.ascontiguousarray(x[:, genes], dtype=np.float64), *args)
    alone = [_diffusion(np.ascontiguousarray(x[:, [g]], dtype=np.float64), *args)[0] for g in genes]
    np.testing.assert_array_equal(together, want)  # NaN (never settled) compares equal here
    np.testing.assert_array_equal(alone, want)


def test_sepal_border_assignment_matches_dense(adata_squaregrid: AnnData):
    """Border nodes get the same saturated node as with the dense distance matrix, ties included."""
    from sklearn.metrics import pairwise_distances

    from squidpy.gr._sepal import _closest

    rng = np.random.default_rng(0)
    xy = adata_squaregrid.obsm["spatial"].astype(np.float64)
    queries = xy[rng.choice(len(xy), 40, replace=False)] + rng.integers(-3, 4, size=(40, 2))  # many ties
    for metric in ("l1", "euclidean", "chebyshev"):
        expected = np.argmin(pairwise_distances(queries, xy, metric=metric), axis=1)
        np.testing.assert_array_equal(_closest(queries, xy, metric), expected)
