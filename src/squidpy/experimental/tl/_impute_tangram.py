from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
from anndata import AnnData
from fast_array_utils import stats
from fast_array_utils.conv import to_dense
from fast_array_utils.types import CSBase
from numba import njit, prange
from scanpy import logging as logg
from spatialdata import SpatialData
from threadpoolctl import threadpool_limits

from squidpy._docs import d
from squidpy._utils import NDArrayA, RNGLike, SeedLike, get_n_numba_threads, numba_threads
from squidpy._validators import assert_positive, normalize_choice, options_of
from squidpy.experimental.tl._utils import resolve_ref_query
from squidpy.gr._utils import _extract_expression, _save_data

__all__ = ["impute_tangram"]

DensityPrior = Literal["rna_count_based", "uniform"]

# Adam's defaults, as in `torch.optim.Adam`
_BETA1, _BETA2, _EPS = 0.9, 0.999, 1e-8


@d.dedent
def impute_tangram(
    data_ref: AnnData | SpatialData,
    data_query: AnnData | SpatialData | None = None,
    *,
    genes: str | Sequence[str],
    table_key: str | tuple[str | None, str | None] | None = None,
    layer: str | tuple[str | None, str | None] | None = None,
    density_prior: DensityPrior | None = "rna_count_based",
    num_epochs: int = 1000,
    learning_rate: float = 0.1,
    key_added: str = "tangram",
    n_jobs: int | None = None,
    rng: SeedLike | RNGLike | None = None,
    inplace: bool = True,
) -> pd.DataFrame | None:
    r"""
    Impute genes in spatial data from a scRNA-seq reference with Tangram :cite:`tangram`.

    Tangram learns a mapping :math:`P` of every reference cell onto the query cells, a probability
    distribution per reference cell given by the softmax of a matrix :math:`M` started at standard normal
    values. With :math:`S` and :math:`G` the reference and query expression of the shared genes, it
    minimizes

    .. math::

        -\frac{1}{n_\text{genes}} \sum_g \cos\left((P^\top S)_{:, g}, G_{:, g}\right)
        + \mathrm{KL}\left(d \,\middle\|\, \tfrac{1}{n_\text{ref}} P^\top \mathbf{1}\right)

    with Adam for ``num_epochs`` steps, where the density term with prior :math:`d` is left out if
    ``density_prior=None``. The imputed expression is :math:`P^\top Y` for the reference expression
    :math:`Y` of ``genes``. This is the ``"cells"`` mode of the `reference implementation
    <https://github.com/broadinstitute/Tangram>`_ with its default weights, training on all shared genes
    that are expressed in both datasets. Unlike the reference implementation, gene names are matched
    case-sensitively.

    :math:`M` holds ``n_obs`` of the reference times ``n_obs`` of the query values, and the optimization keeps
    five arrays of this size in float32, which bounds the size of the datasets.

    Parameters
    ----------
    data_ref, data_query
        The scRNA-seq reference and the spatial query, each an :class:`~anndata.AnnData` or a
        :class:`~spatialdata.SpatialData`. Leave ``data_query=None`` with ``data_ref`` a SpatialData
        holding both samples, distinguished by a ``table_key`` pair. Both should be normalized the same way.
    genes
        Genes of the reference to impute. Genes also measured in the query are among the training genes,
        so their imputation is not a held-out estimate.
        The result is dense, with ``n_obs * len(genes)`` values for the query's ``n_obs``.
    table_key
        For SpatialData input, which table holds the data. A single key applies to both sides;
        a ``(ref, query)`` pair addresses each side separately (use `None` for an AnnData side).
    layer
        Layer holding the expression, or :attr:`anndata.AnnData.X` if `None`. A single value applies to
        both sides; a ``(ref, query)`` pair addresses each side separately.
    density_prior
        Expected share of the reference cells in each query cell:

        - ``"rna_count_based"``: proportional to the query cell's total expression over all its genes.
        - ``"uniform"``: the same for all query cells.
        - `None`: no density term.
    num_epochs
        Number of optimization steps.
    learning_rate
        Learning rate of Adam.
    key_added
        Key in the query's :attr:`anndata.AnnData.obsm` where the result is stored.
    %(n_jobs_threads)s
        A positive value also caps the BLAS and OpenMP threads (via :mod:`threadpoolctl`), which `None` and
        ``-1`` leave as configured.
    %(rng)s
    inplace
        If `True`, store the result in the query, otherwise return it.

    Returns
    -------
    If ``inplace = False``, returns a :class:`pandas.DataFrame` with the query's cells as rows and ``genes``
    as columns.

    Otherwise, stores it in the query's ``obsm[key_added]`` and returns `None`.
    """
    ref, query, genes, shared, ref_layer, query_layer = resolve_ref_query(
        data_ref, data_query, genes=genes, table_key=table_key, layer=layer
    )
    density_prior = normalize_choice(density_prior, (*options_of(DensityPrior), None), name="density_prior")
    assert_positive(num_epochs, name="num_epochs")
    assert_positive(learning_rate, name="learning_rate")
    n_threads = get_n_numba_threads(n_jobs)
    rng = np.random.default_rng(rng)

    ref_x = to_dense(_extract_expression(ref, genes=shared, layer=ref_layer)[0]).astype(np.float32)
    query_x = to_dense(_extract_expression(query, genes=shared, layer=query_layer)[0]).astype(np.float32)
    # a gene without expression in a dataset has no cosine similarity
    expressed = ref_x.any(axis=0) & query_x.any(axis=0)
    if not expressed.any():
        raise ValueError("The reference and the query express no gene in common.")
    if not expressed.all():
        logg.warning(f"Ignoring `{np.sum(~expressed)}` shared genes that are not expressed in either dataset")
    density = _density(query.X if query_layer is None else query.layers[query_layer], density_prior)

    start = logg.info(f"Imputing `{len(genes)}` genes with Tangram using `{n_threads}` thread(s)")
    with numba_threads(n_threads), threadpool_limits(None if n_jobs in (None, -1) else n_threads):
        probs = _map_cells(
            ref_x[:, expressed],
            query_x[:, expressed],
            density,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            rng=rng,
        )
        imputed = probs.T @ to_dense(_extract_expression(ref, genes=genes, layer=ref_layer)[0])

    res = pd.DataFrame(imputed, index=query.obs_names, columns=genes, copy=False)
    if not inplace:
        logg.info("Finish", time=start)
        return res
    _save_data(query, attr="obsm", key=key_added, data=res, time=start)
    return None


def _density(x: NDArrayA | CSBase, prior: DensityPrior | None) -> NDArrayA | None:
    """Tangram's target density over the query cells ``x``, which sums to 1."""
    if prior is None:
        return None
    if prior == "uniform":
        return np.full(x.shape[0], 1 / x.shape[0], dtype=np.float32)
    if stats.min(x) < 0:
        raise ValueError("`density_prior='rna_count_based'` needs non-negative query expression.")
    counts = np.asarray(stats.sum(x, axis=1), dtype=np.float64)
    return (counts / counts.sum()).astype(np.float32)


def _map_cells(
    ref_x: NDArrayA,
    query_x: NDArrayA,
    density: NDArrayA | None,
    *,
    num_epochs: int,
    learning_rate: float,
    rng: np.random.Generator,
) -> NDArrayA:
    """Train Tangram's mapping and return its probabilities, reference cells as rows."""
    logits = rng.standard_normal((ref_x.shape[0], query_x.shape[0]), dtype=np.float32)
    first_moment, second_moment = np.zeros_like(logits), np.zeros_like(logits)
    probs, grad = np.empty_like(logits), np.empty_like(logits)
    query_unit = query_x / np.linalg.norm(query_x, axis=0)
    for step in range(1, num_epochs + 1):
        _softmax(logits, probs)
        _gradient(probs, ref_x, query_unit, density, out=grad)
        _adam_step(logits, grad, first_moment, second_moment, learning_rate, step)
    _softmax(logits, probs)
    return probs


def _gradient(
    probs: NDArrayA, ref_x: NDArrayA, query_unit: NDArrayA, density: NDArrayA | None, *, out: NDArrayA
) -> NDArrayA:
    """Write the gradient of Tangram's loss with respect to the logits of ``probs`` into ``out``.

    ``query_unit`` is the query expression with every gene scaled to unit norm.
    """
    pred = probs.T @ ref_x
    pred_norm = np.linalg.norm(pred, axis=0)
    pred_unit = pred / pred_norm
    cosine = np.einsum("ij,ij->j", pred_unit, query_unit)
    # d(-mean cosine) / d(pred), then through `pred = probs.T @ ref_x`
    np.matmul(ref_x, ((pred_unit * cosine - query_unit) / (pred_norm * ref_x.shape[1])).T, out=out)
    # d KL(density || probs.sum(0) / n_ref) / d(probs) is `-density / probs.sum(0)`, the same for every row
    if density is None:
        density_grad = np.zeros(probs.shape[1], dtype=out.dtype)
    else:
        density_grad = (-density / (np.ones(probs.shape[0], dtype=probs.dtype) @ probs)).astype(out.dtype)
    _softmax_backward(probs, density_grad, out)
    return out


@njit(parallel=True, cache=True)
def _softmax_backward(probs: NDArrayA, row_grad: NDArrayA, grad: NDArrayA) -> None:
    """Turn ``grad + row_grad``, the gradient with respect to the row-wise softmax ``probs``, into the one
    with respect to its logits, in place."""
    for i in prange(probs.shape[0]):
        dot = 0.0
        for j in range(probs.shape[1]):
            dot += probs[i, j] * (grad[i, j] + row_grad[j])
        for j in range(probs.shape[1]):
            grad[i, j] = probs[i, j] * (grad[i, j] + row_grad[j] - dot)


@njit(parallel=True, cache=True)
def _softmax(logits: NDArrayA, out: NDArrayA) -> None:
    for i in prange(logits.shape[0]):
        row_max = logits[i].max()
        total = 0.0
        for j in range(logits.shape[1]):
            out[i, j] = np.exp(logits[i, j] - row_max)
            total += out[i, j]
        for j in range(logits.shape[1]):
            out[i, j] /= total


@njit(parallel=True, cache=True)
def _adam_step(  # noqa: PLR0917, numba requires positional arguments
    params: NDArrayA, grad: NDArrayA, first: NDArrayA, second: NDArrayA, learning_rate: float, step: int
) -> None:
    step_size = learning_rate / (1 - _BETA1**step)
    second_correction = np.sqrt(1 - _BETA2**step)
    for i in prange(params.shape[0]):
        for j in range(params.shape[1]):
            first[i, j] = _BETA1 * first[i, j] + (1 - _BETA1) * grad[i, j]
            second[i, j] = _BETA2 * second[i, j] + (1 - _BETA2) * grad[i, j] ** 2
            params[i, j] -= step_size * first[i, j] / (np.sqrt(second[i, j]) / second_correction + _EPS)
