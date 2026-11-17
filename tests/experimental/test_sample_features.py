from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy.sparse import csr_matrix

from squidpy.experimental.tl import sample_associations, sample_features


def _toy() -> AnnData:
    """Two samples. s1: a line A-A-B-B. s2: three B cells in a line."""
    obs = pd.DataFrame(
        {
            "sample": pd.Categorical(["s1"] * 4 + ["s2"] * 3),
            "ct": pd.Categorical(["A", "A", "B", "B", "B", "B", "B"], categories=["A", "B"]),
            "condition": ["x"] * 4 + ["y"] * 3,  # constant within samples: carried to the sample table
            "quality": np.arange(7.0),  # varies within samples: dropped
        },
        index=[f"c{i}" for i in range(7)],
    )
    edges = [(0, 1), (1, 2), (2, 3), (4, 5), (5, 6)]
    rows, cols = zip(*(edges + [(j, i) for i, j in edges]), strict=True)
    adata = AnnData(np.zeros((7, 1)), obs=obs)
    adata.obsp["spatial_connectivities"] = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(7, 7))
    return adata


def test_sample_features_by_hand():
    sf = sample_features(_toy(), library_key="sample", cluster_key="ct", min_cells=1)
    assert list(sf.obs_names) == ["s1", "s2"]
    assert list(sf.obs["n_cells"]) == [4, 3]
    assert list(sf.obs["condition"]) == ["x", "y"]
    assert "quality" not in sf.obs
    x = sf.to_df()
    assert x.loc["s1", "prop:A"] == 0.5
    assert x.loc["s2", "prop:A"] == 0.0
    # A cells in s1: cell 0 -> {A}, cell 1 -> {A, B}
    assert x.loc["s1", "nhood:A|A"] == pytest.approx(2 / 3)
    assert x.loc["s1", "nhood:A|B"] == pytest.approx(1 / 3)
    # s2 has no A cells: their neighbourhood is undefined
    assert np.isnan(x.loc["s2", "nhood:A|B"])
    assert np.isnan(x.loc["s2", "log2fe:A|B"])
    assert sf.to_df(layer="support").loc["s1", "nhood:A|B"] == 2


def test_sample_features_ignores_self_loops_and_stored_zeros():
    ref = sample_features(_toy(), library_key="sample", cluster_key="ct", min_cells=1).to_df()
    adata = _toy()
    g = adata.obsp["spatial_connectivities"].tocoo()
    # every cell its own neighbour, plus an explicitly stored zero between cells 0 and 3
    rows = np.r_[g.row, np.arange(7), 0]
    cols = np.r_[g.col, np.arange(7), 3]
    data = np.r_[g.data, np.ones(7), 0.0]
    graph = csr_matrix((data, (rows, cols)), shape=(7, 7))
    assert graph.nnz == len(g.data) + 8  # the zero is stored
    adata.obsp["spatial_connectivities"] = graph
    out = sample_features(adata, library_key="sample", cluster_key="ct", min_cells=1).to_df()
    pd.testing.assert_frame_equal(out, ref)


def test_sample_features_min_cells():
    sf = sample_features(_toy(), library_key="sample", cluster_key="ct", min_cells=3)
    # s1 has 2 A cells, below min_cells; its proportion is still defined
    assert np.isnan(sf.to_df().loc["s1", "nhood:A|B"])
    assert sf.to_df().loc["s1", "prop:A"] == 0.5


def test_sample_features_log2fe_sign():
    # in s1, A cells touch A more than random labelling expects (1 of 3 other cells is A)
    x = sample_features(_toy(), library_key="sample", cluster_key="ct", min_cells=1).to_df()
    assert x.loc["s1", "log2fe:A|A"] > 0
    assert x.loc["s1", "log2fe:A|B"] < 0


def test_sample_features_rejects_cross_sample_edges():
    adata = _toy()
    g = adata.obsp["spatial_connectivities"].tolil()
    g[3, 4] = g[4, 3] = 1
    adata.obsp["spatial_connectivities"] = g.tocsr()
    with pytest.raises(ValueError, match="links cells of different samples"):
        sample_features(adata, library_key="sample", cluster_key="ct")


def _samples(n_patients: int = 30, per_patient: int = 2, effect: float = 1.0, seed: int = 0) -> AnnData:
    rng = np.random.default_rng(seed)
    patient = np.repeat(np.arange(n_patients), per_patient)
    condition = (np.arange(n_patients) % 2)[patient]
    patient_effect = rng.normal(0, 1, n_patients)[patient]
    noise = rng.normal(0, 0.3, len(patient))
    x = np.c_[effect * condition + patient_effect + noise, patient_effect + noise]
    obs = pd.DataFrame(
        {"patient": pd.Categorical(patient.astype(str)), "condition": pd.Categorical(condition.astype(str))},
        index=[f"img{i}" for i in range(len(patient))],
    )
    return AnnData(x, obs=obs, var=pd.DataFrame(index=["signal", "null"]))


@pytest.mark.parametrize("method", ["ols", "mixedlm"])
def test_sample_associations_recovers_effect(method):
    res = sample_associations(_samples(effect=2.0), "~ C(condition)", method=method, groups="patient")
    res = res.set_index("feature")
    assert res.loc["signal", "pval_adj"] < 0.01
    assert res.loc["null", "pval"] > 0.01
    assert res.loc["signal", "coef"] == pytest.approx(2.0, abs=0.8)


def test_sample_associations_pseudobulk_counts_patients():
    res = sample_associations(_samples(n_patients=30, per_patient=3), "~ C(condition)", groups="patient")
    assert (res["n"] == 30).all()
    res = sample_associations(_samples(n_patients=30, per_patient=3), "~ C(condition)")
    assert (res["n"] == 90).all()


def test_sample_associations_cox():
    rng = np.random.default_rng(1)
    sa = _samples(n_patients=80, per_patient=1)
    risk = sa.X[:, 0]
    sa.obs["time"] = rng.exponential(1 / np.exp(risk))
    sa.obs["event"] = 1
    res = sample_associations(sa, "~ 1", method="cox", duration_key="time", event_key="event").set_index("feature")
    assert set(res["term"]) == {"_y"}
    assert res.loc["signal", "coef"] > 0
    assert res.loc["signal", "pval"] < 0.01


def test_sample_associations_drops_missing():
    sa = _samples()
    sa.X[:5, 0] = np.nan
    res = sample_associations(sa, "~ C(condition)").set_index("feature")
    assert res.loc["signal", "n"] == sa.n_obs - 5


def test_sample_associations_validates():
    with pytest.raises(ValueError, match="needs `groups`"):
        sample_associations(_samples(), "~ C(condition)", method="mixedlm")
    with pytest.raises(ValueError, match="duration_key"):
        sample_associations(_samples(), "~ 1", method="cox")
