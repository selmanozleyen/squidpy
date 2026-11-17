"""Tangram port vs the original, on outputs upstream Tangram computed (squidpy-ports' Tangram bundle)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import squidpy as sq
from squidpy.experimental.tl._impute_tangram import _density

from ._reference import Bundle, read_bundle

pytest.importorskip("optax")  # needs jax too
_tangram_jax = pytest.importorskip("squidpy.experimental.tl._tangram_jax")
pytestmark = [pytest.mark.jax, pytest.mark.internet]

_BUNDLE_URL = (
    "https://github.com/theislab/squidpy-ports/releases/download/tangram-reference-v1/tangram_reference.zarr.zip"
)
_BUNDLE_SHA256 = "3c6ebc596b5affaa3627e2da0873b930de24dcede4291a7f267293b9b30ff155"
_PRIORS = ["rna_count_based", "uniform", None]
# ~10x the largest error measured on Linux; float32 trajectories from the same start drift apart, so the
# tolerances grow with the epochs
_PROBS_ATOL = {1: 5e-5, 10: 5e-5, 100: 3e-4}
_IMPUTED_REL_ATOL = {1: 5e-5, 10: 5e-5, 100: 1e-3}
# by 300 epochs the drift reaches ~1e-2, as large as the differences a tolerance would have to catch, so only
# that run's loss history is compared
_LONGEST_EPOCHS = 300


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


def _section(bundle: Bundle, prior: str | None, n_epochs: int) -> dict:
    return bundle.sections[f"{prior or 'none'}_epochs{n_epochs}"]


def test_sections_are_covered(bundle):
    tested = {f"{prior or 'none'}_epochs{n}" for prior in _PRIORS for n in [*_PROBS_ATOL, _LONGEST_EPOCHS]}

    assert set(bundle.sections) == tested


@pytest.mark.parametrize("prior", ["rna_count_based", "uniform"])
def test_density_matches_upstream(bundle, prior):
    np.testing.assert_array_equal(_density(bundle.inputs["query"], prior), _section(bundle, prior, 1)["density"])


@pytest.mark.parametrize("prior", _PRIORS)
@pytest.mark.parametrize("n_epochs", list(_PROBS_ATOL))
def test_mapping_matches_upstream(bundle, prior, n_epochs):
    section = _section(bundle, prior, n_epochs)
    # upstream records each loss before its step, so the longest run holds the loss after every shorter one
    losses = _section(bundle, prior, _LONGEST_EPOCHS)["total_loss"]
    ref, query = _training_data(bundle)
    initial = bundle.inputs["initial_logits"]

    probs = _tangram_jax.map_cells(
        np.array(initial),
        ref,
        query,
        section.get("density"),
        n_epochs=n_epochs,
        learning_rate=float(bundle.fixtures["LEARNING_RATE"]),
    )

    np.testing.assert_allclose(probs, section["probs"], atol=_PROBS_ATOL[n_epochs])
    for logits, expected in ((initial, losses[0]), (np.log(probs), losses[n_epochs])):
        np.testing.assert_allclose(_tangram_jax.loss(logits, ref, query, section.get("density")), expected, atol=2e-6)


@pytest.mark.parametrize("prior", _PRIORS)
@pytest.mark.parametrize("n_epochs", list(_PROBS_ATOL))
def test_public_call_matches_upstream(bundle, prior, n_epochs, monkeypatch):
    section = _section(bundle, prior, n_epochs)
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
        n_epochs=n_epochs,
        learning_rate=float(bundle.fixtures["LEARNING_RATE"]),
        inplace=False,
    )

    expected = section["imputed"]
    np.testing.assert_allclose(res.to_numpy(), expected, atol=_IMPUTED_REL_ATOL[n_epochs] * np.abs(expected).max())
