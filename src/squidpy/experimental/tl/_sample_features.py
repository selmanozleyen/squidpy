"""Sample-level spatial features and their association with sample-level outcomes.

Spatial statistics are usually computed on one tissue, and comparisons between conditions are then made
with cells (or images) as the unit of replication. Cells of one patient are not independent replicates of
that patient's condition, so such tests are anticonservative. Here every sample (slide, image or patient)
is summarised by one row of features, and associations are tested with samples as observations, with an
optional random effect for samples that share a patient.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import csr_matrix
from spatialdata import SpatialData
from statsmodels.stats.multitest import multipletests

from squidpy._constants._pkg_constants import Key
from squidpy.gr._utils import _assert_categorical_obs, _assert_connectivity_key, extract_adata_if_sdata

__all__ = ["sample_associations", "sample_features"]

_RESERVED = "_y"


def _codes(adata: AnnData, key: str) -> tuple[np.ndarray, pd.Index]:
    """Category codes of an `obs` column, over the categories that have cells."""
    _assert_categorical_obs(adata, key=key)
    col = adata.obs[key]
    codes = col.cat.codes.to_numpy()
    if (codes < 0).any():
        raise ValueError(f"`adata.obs[{key!r}]` contains missing values.")
    used, codes = np.unique(codes, return_inverse=True)
    return codes, col.cat.categories[used]


def _sample_obs(adata: AnnData, library_key: str, lib: np.ndarray, samples: pd.Index) -> pd.DataFrame:
    """Every `obs` column that is constant within each sample, one row per sample."""
    obs = adata.obs
    _, first = np.unique(lib, return_index=True)  # first cell of each sample
    keep = {}
    for col in obs.columns:
        if col == library_key:
            continue
        values = obs[col]
        try:
            n_unique = values.groupby(lib, observed=True).nunique(dropna=False)
        except TypeError:  # unhashable values
            continue
        if (n_unique <= 1).all():
            keep[col] = values.iloc[first].to_numpy()
    out = pd.DataFrame(keep, index=pd.Index(samples.astype(str), name=library_key))
    for col in keep:
        if isinstance(obs[col].dtype, pd.CategoricalDtype):
            out[col] = pd.Categorical(out[col], categories=obs[col].cat.categories)
    return out


def sample_features(
    adata: AnnData | SpatialData,
    *,
    library_key: str,
    cluster_key: str,
    connectivity_key: str = Key.obsp.spatial_conn(),
    min_cells: int = 10,
    table_key: str | None = None,
) -> AnnData:
    """
    Summarise every sample's composition and cell neighbourhoods as one row of features.

    For each sample (a category of ``library_key``), each cluster ``a`` and each cluster ``b``:

        - ``prop:a`` - fraction of the sample's cells in cluster ``a``.
        - ``nhood:a|b`` - fraction of the spatial-graph neighbours of ``a`` cells that are ``b`` cells.
        - ``log2fe:a|b`` - log2 of ``nhood:a|b`` over its expectation under random labelling within the sample,
          with half a pseudo-count on both; ``> 0`` means ``b`` is enriched around ``a``.

    Neighbourhood features of a cluster with fewer than ``min_cells`` cells in a sample are ``NaN``.

    Parameters
    ----------
    adata
        Annotated data, or a :class:`spatialdata.SpatialData` holding it.
    library_key
        Categorical column in :attr:`anndata.AnnData.obs` naming each cell's sample.
    cluster_key
        Categorical column in :attr:`anndata.AnnData.obs` with the cell clusters.
    connectivity_key
        Key in :attr:`anndata.AnnData.obsp` of the spatial graph. It must not link cells of different samples:
        build it with :func:`squidpy.gr.spatial_neighbors` and ``library_key``.
    min_cells
        Minimum number of cells of a cluster in a sample for its neighbourhood features to be defined.
    table_key
        Key of the table in ``adata`` when it is a :class:`spatialdata.SpatialData`.

    Returns
    -------
    Samples x features :class:`anndata.AnnData`. ``.obs`` holds the number of cells and every column of the
    input ``.obs`` that is constant within samples (patient, condition, outcome, ...). ``.var`` holds
    the feature kind and the clusters it refers to. ``.layers['support']`` holds the number of cells each value
    rests on (the source cluster's cells for neighbourhood features, all cells for proportions).
    """
    adata = extract_adata_if_sdata(adata, table_key=table_key)
    _assert_connectivity_key(adata, key=connectivity_key)
    lib, samples = _codes(adata, library_key)
    ct, clusters = _codes(adata, cluster_key)
    n_s, n_k = len(samples), len(clusters)

    graph = csr_matrix(adata.obsp[connectivity_key])
    rows = np.repeat(np.arange(graph.shape[0]), np.diff(graph.indptr))
    cols = graph.indices
    if (lib[rows] != lib[cols]).any():
        raise ValueError(
            f"`adata.obsp[{connectivity_key!r}]` links cells of different samples. Build the graph per sample with "
            f"`squidpy.gr.spatial_neighbors(adata, library_key={library_key!r})`."
        )

    counts = np.bincount(lib * n_k + ct, minlength=n_s * n_k).reshape(n_s, n_k).astype(float)
    n_cells = counts.sum(1, keepdims=True)
    # ordered neighbour pairs (a -> b) per sample; the graph is binarised: a neighbour counts once
    pairs = np.bincount((lib[rows] * n_k + ct[rows]) * n_k + ct[cols], minlength=n_s * n_k * n_k)
    pairs = pairs.reshape(n_s, n_k, n_k).astype(float)
    out_deg = pairs.sum(2, keepdims=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        prop = counts / n_cells
        nhood = pairs / out_deg
        # expected fraction of b among a's neighbours under random labelling: (n_b - [a == b]) / (n - 1)
        exp_num = counts[:, None, :] - np.eye(n_k)[None]
        exp_den = n_cells[:, :, None] - 1
        observed_frac = (pairs + 0.5) / (out_deg + 0.5 * n_k)
        log2fe = np.log2(observed_frac / ((exp_num + 0.5) / (exp_den + 0.5 * n_k)))
    undefined = (counts < min_cells)[:, :, None] | (out_deg == 0)
    nhood[np.broadcast_to(undefined, nhood.shape)] = np.nan
    log2fe[np.broadcast_to(undefined, log2fe.shape)] = np.nan

    names = clusters.astype(str)
    a_idx, b_idx = np.divmod(np.arange(n_k * n_k), n_k)
    var = pd.concat(
        [
            pd.DataFrame({"kind": "prop", "source": names, "target": names}),
            pd.DataFrame({"kind": "nhood", "source": names[a_idx], "target": names[b_idx]}),
            pd.DataFrame({"kind": "log2fe", "source": names[a_idx], "target": names[b_idx]}),
        ],
        ignore_index=True,
    )
    pair = var["kind"] + ":" + var["source"] + "|" + var["target"]
    var.index = np.where(var["kind"] == "prop", var["kind"] + ":" + var["source"], pair)
    x = np.hstack([prop, nhood.reshape(n_s, -1), log2fe.reshape(n_s, -1)])
    support_nhood = np.repeat(counts, n_k, axis=1)
    support = np.hstack([np.repeat(n_cells, n_k, axis=1), support_nhood, support_nhood])

    obs = _sample_obs(adata, library_key, lib, samples)
    obs.insert(0, "n_cells", n_cells[:, 0].astype(int))
    return AnnData(X=x, obs=obs, var=var, layers={"support": support})


def _fit_one(
    df: pd.DataFrame,
    formula: str,
    *,
    method: str,
    groups: str | None,
    duration_key: str | None,
    event_key: str | None,
) -> pd.DataFrame:
    import statsmodels.formula.api as smf
    from statsmodels.duration.hazard_regression import PHReg

    rhs = formula.split("~", 1)[-1].strip()
    if method == "ols":
        res = smf.ols(f"{_RESERVED} ~ {rhs}", df).fit()
        terms = [t for t in res.params.index if t != "Intercept"]
        stat = res.tvalues
    elif method == "mixedlm":
        res = smf.mixedlm(f"{_RESERVED} ~ {rhs}", df, groups=groups).fit(reml=True)
        terms = [t for t in res.fe_params.index if t != "Intercept"]
        stat = res.tvalues
    else:  # cox: the feature is a covariate of survival
        covariates = f"{_RESERVED} + {rhs}" if rhs not in ("", "1") else _RESERVED
        res = PHReg.from_formula(f"{duration_key} ~ {covariates}", df, status=df[event_key].to_numpy()).fit()
        names = res.model.exog_names
        params, bse = pd.Series(res.params, index=names), pd.Series(res.bse, index=names)
        z = params / bse
        pval = pd.Series(res.pvalues, index=names)
        return pd.DataFrame(
            {
                "term": [_RESERVED],
                "coef": [params[_RESERVED]],
                "se": [bse[_RESERVED]],
                "stat": [z[_RESERVED]],
                "pval": [pval[_RESERVED]],
            }
        )
    return pd.DataFrame(
        {
            "term": terms,
            "coef": res.params[terms].to_numpy(),
            "se": res.bse[terms].to_numpy(),
            "stat": stat[terms].to_numpy(),
            "pval": res.pvalues[terms].to_numpy(),
        }
    )


def sample_associations(
    sample_adata: AnnData,
    formula: str,
    *,
    method: Literal["ols", "mixedlm", "cox"] = "ols",
    groups: str | None = None,
    duration_key: str | None = None,
    event_key: str | None = None,
    features: Sequence[str] | None = None,
    min_samples: int = 5,
    corr_method: str | None = "fdr_bh",
) -> pd.DataFrame:
    """
    Test every sample-level feature for association with sample-level covariates.

    The unit of replication is the sample (or the group of samples given by ``groups``), never the cell.

    Parameters
    ----------
    sample_adata
        Samples x features, e.g. from :func:`sample_features`.
    formula
        Right-hand side of a :mod:`statsmodels` formula over columns of ``sample_adata.obs``, e.g.
        ``'~ C(condition) + age'``. For ``'ols'`` and ``'mixedlm'`` each feature is the response. For ``'cox'``
        each feature is a covariate of survival, adjusted for the terms of ``formula`` (``'~ 1'`` for none).
    method
        - ``'ols'`` - linear model. With ``groups``, samples are first averaged within each group (pseudobulk), so
          each group (e.g. patient) counts once; covariates must then be constant within groups.
        - ``'mixedlm'`` - linear mixed model with a random intercept per ``groups``, for several samples per patient.
        - ``'cox'`` - Cox proportional hazards model of ``duration_key`` and ``event_key``. With ``groups``, samples are
          first averaged within each group.
    groups
        Column of ``sample_adata.obs`` with the replication unit shared by several samples (e.g. patient).
    duration_key, event_key
        Columns of ``sample_adata.obs`` with survival time and event indicator (1 = event). Only for ``'cox'``.
    features
        Features to test. Default: all.
    min_samples
        Minimum number of observations (after dropping missing values and pseudobulking) to test a feature.
    corr_method
        Multiple testing correction of :func:`statsmodels.stats.multitest.multipletests`, applied per term.

    Returns
    -------
    One row per feature and term: ``coef``, ``se``, ``stat``, ``pval``, ``n`` (observations used) and
    ``pval_adj`` if ``corr_method`` is given. For ``'cox'`` the term is the feature itself (``'_y'``).
    """
    if method not in ("ols", "mixedlm", "cox"):
        raise ValueError(f"`method` must be 'ols', 'mixedlm' or 'cox', got {method!r}.")
    if method == "mixedlm" and groups is None:
        raise ValueError("`method='mixedlm'` needs `groups`.")
    if method == "cox" and (duration_key is None or event_key is None):
        raise ValueError("`method='cox'` needs `duration_key` and `event_key`.")
    if _RESERVED in sample_adata.obs:
        raise ValueError(f"`sample_adata.obs` must not have a column named {_RESERVED!r}.")

    names = list(sample_adata.var_names if features is None else features)
    x = sample_adata[:, names].X
    x = np.asarray(x.toarray() if hasattr(x, "toarray") else x, dtype=float)
    obs = sample_adata.obs
    rhs = formula.split("~", 1)[-1]
    # rows missing anything the model uses are dropped up front, so every model sees the same rows
    used = [c for c in obs.columns if c in rhs or c in (groups, duration_key, event_key)]
    results = []
    for j, name in enumerate(names):
        df = obs.assign(**{_RESERVED: x[:, j]}).dropna(subset=[_RESERVED, *used])
        if groups is not None and method != "mixedlm":
            agg = {c: "first" for c in df.columns if c != groups}
            agg[_RESERVED] = "mean"
            df = df.groupby(groups, observed=True).agg(agg).reset_index()
        if len(df) < min_samples or df[_RESERVED].nunique() < 2:
            continue
        with warnings.catch_warnings():
            # convergence and boundary warnings of individual fits: the fit is kept, its p-value reported as is
            warnings.simplefilter("ignore")
            try:
                fit = _fit_one(
                    df, formula, method=method, groups=groups, duration_key=duration_key, event_key=event_key
                )
            except (np.linalg.LinAlgError, ValueError):
                continue
        results.append(fit.assign(feature=name, n=len(df)))
    if not results:
        return pd.DataFrame(columns=["feature", "term", "coef", "se", "stat", "pval", "n"])
    out = pd.concat(results, ignore_index=True)[["feature", "term", "coef", "se", "stat", "pval", "n"]]
    if corr_method is not None:
        out["pval_adj"] = np.nan
        for _, idx in out.groupby("term").groups.items():
            p = out.loc[idx, "pval"].to_numpy()
            ok = np.isfinite(p)
            if ok.any():
                adj = np.full(p.shape, np.nan)
                adj[ok] = multipletests(p[ok], method=corr_method)[1]
                out.loc[idx, "pval_adj"] = adj
    return out.sort_values("pval", kind="stable").reset_index(drop=True)
