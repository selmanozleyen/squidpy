"""Tangram port vs the original, on outputs upstream Tangram computed (squidpy-ports' Tangram bundle)."""

from __future__ import annotations

import importlib
import os

import numpy as np
import pytest

# the jax CI env sets JAX_ENABLE_X64, and there a missing jax must fail rather than skip
jax = importlib.import_module("jax") if os.environ.get("JAX_ENABLE_X64") else pytest.importorskip("jax")
pytestmark = [pytest.mark.jax, pytest.mark.internet]

from squidpy.experimental.tl._impute_tangram import _density  # noqa: E402
from squidpy.experimental.tl._tangram_jax import map_cells  # noqa: E402

from ._reference import Bundle, read_bundle  # noqa: E402
from .test_impute_tangram import _data  # noqa: E402

_BUNDLE_URL = (
    "https://github.com/selmanozleyen/squidpy-ports/releases/download/tangram-reference-v1/tangram_reference.zarr.zip"
)
_BUNDLE_SHA256 = "580e918a96f87954e43f94d91f95060b35611cc208ac908fc5fdd2846ff69a36"
_PRIORS = ["rna_count_based", "uniform", None]


@pytest.fixture(scope="module")
def bundle() -> Bundle:
    return read_bundle(_BUNDLE_URL, _BUNDLE_SHA256)


def test_fixture_is_ours(bundle):
    ref, query, truth = _data()

    np.testing.assert_array_equal(bundle.inputs["ref"], ref.X)
    np.testing.assert_array_equal(bundle.inputs["query"], query.X)
    np.testing.assert_array_equal(bundle.inputs["truth"], truth)


@pytest.mark.parametrize("prior", _PRIORS[:2])
def test_density_matches_upstream(bundle, prior):
    expected = bundle.sections[f"{prior}_epochs1"]["density"]

    np.testing.assert_allclose(_density(bundle.inputs["query"], prior), expected, rtol=1e-6)


@pytest.mark.parametrize("prior", _PRIORS)
# Adam's first step is `lr * g / (|g| + eps)`, so float32 rounding moves it wherever `g` is near eps;
# after that, float32 trajectories from the same start drift apart slowly
@pytest.mark.parametrize(("num_epochs", "atol"), [(1, 1e-5), (10, 1e-5), (100, 1e-4)])
def test_mapping_matches_upstream(bundle, prior, num_epochs, atol):
    section = bundle.sections[f"{prior or 'none'}_epochs{num_epochs}"]
    n_measured = len(bundle.fixtures["measured"])
    ref, query = bundle.inputs["ref"].astype(np.float32), bundle.inputs["query"].astype(np.float32)

    probs = map_cells(
        np.array(bundle.inputs["initial_logits"]),
        ref[:, :n_measured],
        query,
        section.get("density"),
        num_epochs=num_epochs,
        learning_rate=float(bundle.fixtures["LEARNING_RATE"]),
    )

    np.testing.assert_allclose(probs, section["probs"], atol=atol)
    imputed = np.asarray(probs, dtype=np.float64).T @ bundle.inputs["ref"][:, n_measured:]
    np.testing.assert_allclose(imputed, section["imputed"], rtol=1e-3, atol=1e-6)
