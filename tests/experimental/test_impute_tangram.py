"""Tests for sq.experimental.tl.impute_tangram."""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy.sparse import csr_matrix

import squidpy as sq

pytest.importorskip("optax")  # needs jax too
pytestmark = pytest.mark.jax


_GENES = [f"g{i}" for i in range(10)]
_N_MEASURED = 7  # the query measures g0-g6, g7-g9 are imputed
_UNMEASURED = _GENES[_N_MEASURED:]


def _data() -> tuple[AnnData, AnnData, np.ndarray]:
    """A reference of 3 cell types, a query of noisy copies of its cells, and the query's unmeasured genes."""
    rng = np.random.default_rng(0)
    profiles = rng.gamma(2.0, size=(3, len(_GENES)))
    ref_x = profiles[rng.integers(3, size=60)] * rng.uniform(0.8, 1.2, size=(60, len(_GENES)))
    query_x = ref_x[rng.integers(60, size=40)] * rng.uniform(0.9, 1.1, size=(40, len(_GENES)))
    ref = AnnData(ref_x, var=pd.DataFrame(index=_GENES))
    query = AnnData(query_x[:, :_N_MEASURED], var=pd.DataFrame(index=_GENES[:_N_MEASURED]))
    return ref, query, query_x[:, _N_MEASURED:]


def _impute(data_ref, data_query, **kwargs) -> pd.DataFrame | None:
    return sq.experimental.tl.impute_tangram(
        data_ref, data_query, **({"genes": _UNMEASURED, "num_epochs": 100, "rng": 0, "inplace": False} | kwargs)
    )


def test_recovers_unmeasured_genes():
    ref, query, truth = _data()

    res = _impute(ref, query, num_epochs=300)

    assert res.columns.tolist() == _UNMEASURED
    for imputed, expected in zip(res.to_numpy().T, truth.T, strict=True):
        assert np.corrcoef(imputed, expected)[0, 1] > 0.8


def test_stores_in_obsm():
    ref, query, _ = _data()
    expected = _impute(ref, query)

    assert _impute(ref, query, inplace=True) is None

    pd.testing.assert_frame_equal(query.obsm["tangram"], expected)


def test_rng():
    ref, query, _ = _data()
    res = _impute(ref, query, rng=1)

    pd.testing.assert_frame_equal(_impute(ref, query, rng=np.random.default_rng(1)), res, check_exact=True)
    assert not np.allclose(_impute(ref, query, rng=2).to_numpy(), res.to_numpy())


def test_sparse_matches_dense():
    ref, query, _ = _data()
    dense = _impute(ref, query)

    ref.X, query.X = csr_matrix(ref.X), csr_matrix(query.X)

    pd.testing.assert_frame_equal(_impute(ref, query), dense, rtol=1e-5)


def test_density_prior():
    ref, query, _ = _data()
    query.X /= query.X.sum(axis=1, keepdims=True)  # equal totals: the count-based density is uniform

    count_based = _impute(ref, query, density_prior="rna_count_based")

    pd.testing.assert_frame_equal(count_based, _impute(ref, query, density_prior="uniform"), rtol=1e-4)
    assert not np.allclose(count_based.to_numpy(), _impute(ref, query, density_prior=None).to_numpy())


def test_count_based_density_with_empty_query_cell():
    ref, query, _ = _data()
    query.X[0] = 0

    res = _impute(ref, query)

    assert np.isfinite(res.to_numpy()).all()


def test_count_based_density_uses_query_layer():
    ref, query, _ = _data()
    expected = _impute(ref, query)
    query.layers["norm"] = query.X
    query.X = np.ones_like(query.X)

    res = _impute(ref, query, layer=(None, "norm"))

    pd.testing.assert_frame_equal(res, expected)


@pytest.mark.parametrize("unexpressed_in", ["ref", "query"])
def test_unexpressed_genes_are_ignored(unexpressed_in):
    ref, query, _ = _data()
    (ref if unexpressed_in == "ref" else query).X[:, 0] = 0
    # without g0 in the query, it is not shared
    expected = _impute(ref, query[:, 1:], density_prior="uniform")

    res = _impute(ref, query, density_prior="uniform")

    pd.testing.assert_frame_equal(res, expected)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        pytest.param({"num_epochs": 0}, "num_epochs", id="num_epochs"),
        pytest.param({"learning_rate": 0}, "learning_rate", id="learning_rate"),
        pytest.param({"density_prior": "foo"}, "density_prior", id="density_prior"),
        pytest.param({"genes": "g7", "layer": (None, "zero")}, "express no gene", id="no-expressed-gene"),
        pytest.param({"layer": (None, "neg")}, "non-negative", id="negative-expression"),
    ],
)
def test_invalid_input(kwargs, match):
    ref, query, _ = _data()
    query.layers["zero"] = np.zeros_like(query.X)
    query.layers["neg"] = -query.X

    with pytest.raises(ValueError, match=match):
        _impute(ref, query, **kwargs)


def test_missing_jax(monkeypatch):
    ref, query, _ = _data()
    monkeypatch.setitem(sys.modules, "jax", None)

    with pytest.raises(ImportError, match=r"squidpy\[jax\]"):
        _impute(ref, query)
