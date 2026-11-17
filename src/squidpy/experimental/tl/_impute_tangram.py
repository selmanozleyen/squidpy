from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral
from typing import Literal

import numpy as np
import pandas as pd
from anndata import AnnData
from fast_array_utils import stats
from fast_array_utils.conv import to_dense
from fast_array_utils.types import CSBase
from scanpy import logging as logg
from scipy.sparse import csr_matrix
from spatialdata import SpatialData

from squidpy._docs import d
from squidpy._utils import NDArrayA, RNGLike, SeedLike
from squidpy._validators import assert_isinstance, assert_positive, normalize_choice, options_of
from squidpy.experimental.tl._impute_spage import _matmul
from squidpy.experimental.tl._utils import resolve_ref_query
from squidpy.gr._utils import _extract_expression, _save_data

__all__ = ["impute_tangram"]

DensityPrior = Literal["rna_count_based", "uniform"]


@d.dedent
def impute_tangram(
    data_ref: AnnData | SpatialData,
    data_query: AnnData | SpatialData | None = None,
    *,
    genes: str | Sequence[str],
    table_key: str | tuple[str | None, str | None] | None = None,
    layer: str | tuple[str | None, str | None] | None = None,
    density_prior: DensityPrior | None = "rna_count_based",
    n_epochs: int = 1000,
    learning_rate: float = 0.1,
    key_added: str = "tangram",
    rng: SeedLike | RNGLike | None = None,
    inplace: bool = True,
) -> pd.DataFrame | None:
    """
    Impute genes in spatial data from a scRNA-seq reference with Tangram :cite:`tangram`.

    Maps reference cells onto query cells as in Tangram's default ``"cells"`` mode, maximizing the mean per-gene
    cosine similarity between the mapped and the measured expression while penalizing the KL divergence of the
    mapped density from ``density_prior``, and projects the reference expression through the mapping. Unlike
    `Tangram <https://github.com/broadinstitute/Tangram>`_, gene names are case-sensitive.

    Needs ``squidpy[jax]``, which installs JAX for the CPU; install ``jax[cuda12]`` for a GPU, where JAX
    preallocates most of its memory on first use. Memory scales with ``n_ref * n_query``.

    Parameters
    ----------
    %(impute_data)s
    genes
        Reference genes to impute. Genes measured in the query are not held out.
    %(impute_keys)s
    density_prior
        Prior over query cells: ``"rna_count_based"`` (proportional to total counts), ``"uniform"``, or `None`
        (no density term).
    n_epochs
        Number of optimization steps, Tangram's ``num_epochs``.
    learning_rate
        Adam's learning rate.
    key_added
        Key in the query's :attr:`anndata.AnnData.obsm` where the result is stored. With ``inplace = True``, the
        mapping is stored too, in ``obsm[f"{key_added}_mapping"]``: the probability of each reference cell
        (column, in the reference's order) being at each query cell (row), from which, e.g., cell annotations
        can be projected onto the query. It is dense, ``4 * n_query * n_ref`` bytes.
    %(rng)s
    %(impute_inplace)s

    Returns
    -------
    %(impute_returns)s
    """
    ref, query, genes, shared, ref_layer, query_layer = resolve_ref_query(
        data_ref, data_query, genes=genes, table_key=table_key, layer=layer
    )
    density_prior = normalize_choice(density_prior, (*options_of(DensityPrior), None), name="density_prior")
    assert_isinstance(n_epochs, Integral, name="n_epochs")
    assert_positive(n_epochs, name="n_epochs")
    assert_positive(learning_rate, name="learning_rate")
    rng = np.random.default_rng(rng)
    try:
        import jax.numpy as jnp

        from squidpy.experimental.tl._tangram_jax import map_cells
    except ImportError as e:
        raise ImportError('Tangram imputation requires JAX and optax: `pip install "squidpy[jax]"`.') from e

    ref_x = to_dense(_extract_expression(ref, genes=shared, layer=ref_layer)[0]).astype(np.float32)
    query_x = to_dense(_extract_expression(query, genes=shared, layer=query_layer)[0]).astype(np.float32)
    expressed = ref_x.any(axis=0) & query_x.any(axis=0)
    if not expressed.any():
        raise ValueError("The reference and the query express no gene in common.")
    if not expressed.all():
        logg.warning(f"Ignoring `{np.sum(~expressed)}` shared genes unexpressed in the reference or the query")
    density = _density(query.X if query_layer is None else query.layers[query_layer], density_prior)

    start = logg.info(f"Imputing `{len(genes)}` genes with Tangram")
    probs = map_cells(
        rng.standard_normal((ref.n_obs, query.n_obs), dtype=np.float32),
        ref_x[:, expressed],
        query_x[:, expressed],
        density,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
    )
    y = _extract_expression(ref, genes=genes, layer=ref_layer)[0]
    if isinstance(y, CSBase):  # projected on the host, rather than densifying all of `y` for the device
        imputed = _matmul(csr_matrix(y.T, dtype=np.float32), np.asarray(probs)).T
    else:  # on the device, so that only the result is copied back
        imputed = np.asarray(jnp.matmul(probs.T, jnp.asarray(y, dtype=probs.dtype), precision="highest"))

    res = pd.DataFrame(imputed, index=query.obs_names, columns=genes, copy=False)
    if not inplace:
        logg.info("Finish", time=start)
        return res
    _save_data(query, attr="obsm", key=key_added, data=res)
    mapping = np.ascontiguousarray(np.asarray(probs).T)
    _save_data(query, attr="obsm", key=f"{key_added}_mapping", data=mapping, prefix=False, time=start)
    return None


def _density(x: NDArrayA | CSBase, prior: DensityPrior | None) -> NDArrayA | None:
    """Tangram's target density over the query cells ``x``."""
    if prior is None:
        return None
    if prior == "uniform":
        return np.full(x.shape[0], 1 / x.shape[0], dtype=np.float32)
    if stats.min(x) < 0:
        raise ValueError("`density_prior='rna_count_based'` needs non-negative query expression.")
    counts = np.asarray(stats.sum(x, axis=1), dtype=np.float64)
    return (counts / counts.sum()).astype(np.float32)
