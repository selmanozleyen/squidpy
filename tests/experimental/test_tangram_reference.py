"""Tangram port vs the original, on outputs upstream Tangram computed (squidpy-ports' Tangram bundle)."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

# the jax CI env sets JAX_ENABLE_X64, and there a missing jax must fail rather than skip
if os.environ.get("JAX_ENABLE_X64") != "1":
    pytest.importorskip("jax")
    pytest.importorskip("optax")
pytestmark = [pytest.mark.jax, pytest.mark.internet]

import squidpy as sq  # noqa: E402
from squidpy.experimental.tl import _tangram_jax  # noqa: E402
from squidpy.experimental.tl._impute_tangram import _density  # noqa: E402

from ._reference import Bundle, read_bundle  # noqa: E402
from .test_impute_tangram import _data  # noqa: E402

_BUNDLE_URL = (
    "https://github.com/theislab/squidpy-ports/releases/download/tangram-reference-v1/tangram_reference.zarr.zip"
)
_BUNDLE_SHA256 = "3c6ebc596b5affaa3627e2da0873b930de24dcede4291a7f267293b9b30ff155"
_PRIORS = ["rna_count_based", "uniform", None]
# ~10x the largest error measured on Linux, with JAX_ENABLE_X64 on and off; float32 trajectories from the
# same start drift apart, so the tolerance grows with the epochs
_TOLERANCES = {1: (5e-5, 5e-5), 10: (5e-5, 5e-5), 100: (3e-4, 1e-3)}
# by 300 epochs the drift reaches ~1e-2, as large as the differences a tolerance would have to catch
_UNTESTED_EPOCHS = 300


@pytest.fixture(scope="module")
def bundle() -> Bundle:
    return read_bundle(_BUNDLE_URL, _BUNDLE_SHA256)


def _training_data(bundle: Bundle) -> tuple[np.ndarray, np.ndarray]:
    """Reference and query expression of upstream's training genes, in its order and in float32."""
    genes, measured = list(bundle.fixtures["genes"]), list(bundle.fixtures["measured"])
    train = list(bundle.inputs["training_genes"])
    ref = bundle.inputs["ref"][:, [genes.index(g) for g in train]]
    query = bundle.inputs["query"][:, [measured.index(g) for g in train]]
    return ref.astype(np.float32), query.astype(np.float32)


def _section(bundle: Bundle, prior: str | None, num_epochs: int) -> dict:
    return bundle.sections[f"{prior or 'none'}_epochs{num_epochs}"]


def test_sections_are_covered(bundle):
    tested = {f"{prior or 'none'}_epochs{n}" for prior in _PRIORS for n in [*_TOLERANCES, _UNTESTED_EPOCHS]}

    assert set(bundle.sections) == tested


def test_fixture_is_ours(bundle):
    ref, query, truth = _data()

    # the bundle is built on Linux, and gamma draws go through the platform's libm: elsewhere a few
    # differ in the last bit
    np.testing.assert_allclose(ref.X, bundle.inputs["ref"], rtol=1e-14, atol=0)
    np.testing.assert_allclose(query.X, bundle.inputs["query"], rtol=1e-14, atol=0)
    np.testing.assert_allclose(truth, bundle.inputs["truth"], rtol=1e-14, atol=0)


@pytest.mark.parametrize("prior", _PRIORS[:2])
def test_density_matches_upstream(bundle, prior):
    np.testing.assert_array_equal(_density(bundle.inputs["query"], prior), _section(bundle, prior, 1)["density"])


@pytest.mark.parametrize("prior", _PRIORS)
@pytest.mark.parametrize("num_epochs", list(_TOLERANCES))
def test_mapping_matches_upstream(bundle, prior, num_epochs):
    section = _section(bundle, prior, num_epochs)
    # upstream records each loss before its step, so the longest run holds the loss after every shorter one
    losses = _section(bundle, prior, _UNTESTED_EPOCHS)["total_loss"]
    ref, query = _training_data(bundle)
    initial = bundle.inputs["initial_logits"]

    probs = _tangram_jax.map_cells(
        np.array(initial),
        ref,
        query,
        section.get("density"),
        num_epochs=num_epochs,
        learning_rate=float(bundle.fixtures["LEARNING_RATE"]),
    )

    np.testing.assert_allclose(probs, section["probs"], atol=_TOLERANCES[num_epochs][0])
    for logits, expected in ((initial, losses[0]), (np.log(probs), losses[num_epochs])):
        np.testing.assert_allclose(_tangram_jax.loss(logits, ref, query, section.get("density")), expected, atol=2e-6)


@pytest.mark.parametrize("prior", _PRIORS)
@pytest.mark.parametrize("num_epochs", list(_TOLERANCES))
def test_public_call_matches_upstream(bundle, prior, num_epochs, monkeypatch):
    section = _section(bundle, prior, num_epochs)
    genes, measured = list(bundle.fixtures["genes"]), list(bundle.fixtures["measured"])
    map_cells = _tangram_jax.map_cells
    # start from upstream's initial mapping rather than one drawn from `rng`
    initial = np.array(bundle.inputs["initial_logits"])
    monkeypatch.setattr(_tangram_jax, "map_cells", lambda _, *args, **kwargs: map_cells(initial, *args, **kwargs))

    res = sq.experimental.tl.impute_tangram(
        AnnData(bundle.inputs["ref"], var=pd.DataFrame(index=genes)),
        AnnData(bundle.inputs["query"], var=pd.DataFrame(index=measured)),
        genes=list(bundle.fixtures["unmeasured"]),
        density_prior=prior,
        num_epochs=num_epochs,
        learning_rate=float(bundle.fixtures["LEARNING_RATE"]),
        inplace=False,
    )

    expected = section["imputed"]
    np.testing.assert_allclose(res.to_numpy(), expected, atol=_TOLERANCES[num_epochs][1] * np.abs(expected).max())
