"""Tests for sq.experimental.tl.impute_tangram."""

from __future__ import annotations

import importlib
import os
import sys

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy.sparse import csr_matrix
from scipy.special import softmax, xlogy

import squidpy as sq

# the jax CI env sets JAX_ENABLE_X64, and there a missing jax must fail rather than skip
jax = importlib.import_module("jax") if os.environ.get("JAX_ENABLE_X64") else pytest.importorskip("jax")
pytestmark = pytest.mark.jax

from squidpy.experimental.tl._tangram_jax import loss, map_cells  # noqa: E402

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


def _loss(logits: np.ndarray, ref_x: np.ndarray, query_x: np.ndarray, density: np.ndarray | None) -> float:
    """NumPy port of Tangram's ``Mapper._loss_fn``."""
    probs = softmax(logits, axis=1)
    pred = probs.T @ ref_x
    cosine = (pred * query_x).sum(0) / (np.linalg.norm(pred, axis=0) * np.linalg.norm(query_x, axis=0))
    loss = -cosine.mean()
    if density is not None:
        loss += np.sum(xlogy(density, density) - density * np.log(probs.mean(axis=0)))
    return loss


@pytest.mark.parametrize("with_density", [True, False])
def test_loss_matches_tangram(with_density):
    rng = np.random.default_rng(0)
    logits, ref_x, query_x = rng.normal(size=(5, 4)), rng.uniform(size=(5, 3)), rng.uniform(size=(4, 3))
    density = rng.dirichlet(np.ones(4)) if with_density else None

    res = loss(logits, ref_x, query_x, density)

    np.testing.assert_allclose(res, _loss(logits, ref_x, query_x, density), rtol=1e-5)


def test_map_cells_matches_reference_loop():
    rng = np.random.default_rng(0)
    ref_x, query_x = rng.uniform(size=(5, 3)).astype(np.float32), rng.uniform(size=(4, 3)).astype(np.float32)
    density = rng.dirichlet(np.ones(4)).astype(np.float32)
    # Tangram's training loop, with finite differences instead of autograd
    logits = np.random.default_rng(1).standard_normal((5, 4), dtype=np.float32).astype(np.float64)
    first, second, eps = np.zeros_like(logits), np.zeros_like(logits), 1e-6
    for step in range(1, 4):
        grad = np.zeros_like(logits)
        for idx in np.ndindex(logits.shape):
            shift = np.zeros_like(logits)
            shift[idx] = eps
            grad[idx] = (
                _loss(logits + shift, ref_x, query_x, density) - _loss(logits - shift, ref_x, query_x, density)
            ) / (2 * eps)
        first = 0.9 * first + 0.1 * grad
        second = 0.999 * second + 0.001 * grad**2
        logits -= 0.1 / (1 - 0.9**step) * first / (np.sqrt(second / (1 - 0.999**step)) + 1e-8)

    init = np.random.default_rng(1).standard_normal((5, 4), dtype=np.float32)
    probs = map_cells(init, ref_x, query_x, density, num_epochs=3, learning_rate=0.1)

    np.testing.assert_allclose(probs, softmax(logits, axis=1), rtol=1e-4)


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
    ],
)
def test_invalid_input(kwargs, match):
    ref, query, _ = _data()
    query.layers["zero"] = np.zeros_like(query.X)

    with pytest.raises(ValueError, match=match):
        _impute(ref, query, **kwargs)


def test_rna_count_density_needs_non_negative_expression():
    ref, query, _ = _data()
    query.X[0, 0] = -1e3

    with pytest.raises(ValueError, match="non-negative"):
        _impute(ref, query)


def test_no_shared_genes():
    ref, query, _ = _data()
    query.var_names = [f"other{i}" for i in range(query.n_vars)]

    with pytest.raises(ValueError, match="no genes in common"):
        _impute(ref, query)


def test_missing_jax(monkeypatch):
    ref, query, _ = _data()
    for module in ("jax", "jax.numpy", "squidpy.experimental.tl._tangram_jax"):
        monkeypatch.setitem(sys.modules, module, None)

    with pytest.raises(ImportError, match=r"squidpy\[jax\]"):
        _impute(ref, query)
