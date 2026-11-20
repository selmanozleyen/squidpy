"""Sample-level spatial features and their association with sample-level outcomes.

Spatial statistics are usually computed on one tissue, and comparisons between conditions are then made
with cells (or images) as the unit of replication. Cells of one patient are not independent replicates of
that patient's condition, so such tests are anticonservative. Here every sample (slide, image or patient)
is summarised by one row of features, and associations are tested with samples as observations, with an
optional random effect for samples that share a patient.
"""

from __future__ import annotations

import re
import warnings
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
_COX_TERM = "feature"


def _in_formula(column: str, rhs: str) -> bool:
    """Whether ``column`` appears in the formula as a whole name (``age`` is not in ``~ stage``)."""
    return re.search(rf"(?<![\w.]){re.escape(column)}(?![\w.])", rhs) is not None


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
          ``(n_b - [a = b]) / (n - 1)``, computed on counts: ``log2((pairs_ab + 1/2) / (expected_ab + 1/2))`` with
          ``expected_ab`` the number of ``a -> b`` pairs random labelling gives. It is about ``0`` under random
          labelling, finite without contacts, shrunk towards ``0`` when few pairs are expected; ``> 0`` means ``b`` is
          enriched around ``a``. Missing when the sample has no ``b`` cell other than the ``a`` cell itself.
          With few expected pairs the log of a small ratio is biased downward under random labelling (about -0.2 to
          -0.45 below ~6 expected pairs, below 0.05 in magnitude from ~30); because this depends on how rare ``b`` is,
          compare groups on pairs with enough expected contacts (``.layers['support']``, ``min_cells``).

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
        build it per sample, e.g. with :func:`squidpy.gr.spatial_neighbors_delaunay` and ``library_key``.
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

    graph = csr_matrix(adata.obsp[connectivity_key], copy=True)
    graph.eliminate_zeros()  # stored zeros are not edges
    rows = np.repeat(np.arange(graph.shape[0]), np.diff(graph.indptr))
    cols = graph.indices
    not_self = rows != cols  # a cell is not its own neighbour (graphs built with `set_diag=True`)
    rows, cols = rows[not_self], cols[not_self]
    if (lib[rows] != lib[cols]).any():
        raise ValueError(
            f"`adata.obsp[{connectivity_key!r}]` links cells of different samples. Build the graph per sample with "
            f"e.g. `squidpy.gr.spatial_neighbors_delaunay(adata, library_key={library_key!r})`."
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
        expected = (counts[:, None, :] - np.eye(n_k)[None]) / (n_cells[:, :, None] - 1)
        # observed over expected pair counts, half a pseudo-count on both: 0 under random labelling, and an absent
        # pair is never "enriched" (a pseudo-count on the fractions instead pulls them towards 1 / n_clusters)
        log2fe = np.log2((pairs + 0.5) / (out_deg * expected + 0.5))
    undefined = (counts < min_cells)[:, :, None] | (out_deg == 0)
    log2fe[expected <= 0] = np.nan
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


def _require_full_rank(exog: np.ndarray) -> None:
    """Raise if the design is rank-deficient, e.g. a factor left with one level after dropping missing rows.

    Fitting would still return numbers: with a pseudo-inverse the collinear term gets the intercept's test.
    """
    if np.linalg.matrix_rank(exog) < exog.shape[1]:
        raise ValueError("Rank-deficient design.")


def _fit_one(
    df: pd.DataFrame,
    rhs: str,
    *,
    method: str,
    replicate_key: str | None,
    duration_key: str | None,
    event_key: str | None,
) -> pd.DataFrame:
    import statsmodels.formula.api as smf
    from statsmodels.duration.hazard_regression import PHReg

    if method == "ols":
        res = smf.ols(f"{_RESERVED} ~ {rhs}", df).fit()
        _require_full_rank(res.model.exog)
        terms = [t for t in res.params.index if t != "Intercept"]
        stat = res.tvalues
    elif method == "mixedlm":
        res = smf.mixedlm(f"{_RESERVED} ~ {rhs}", df, groups=replicate_key).fit(reml=True)
        _require_full_rank(res.model.exog)
        terms = [t for t in res.fe_params.index if t != "Intercept"]
        stat = res.tvalues
    else:  # cox: the feature is a covariate of survival
        covariates = f"{_RESERVED} + {rhs}" if rhs not in ("", "1") else _RESERVED
        model = PHReg.from_formula(f"{duration_key} ~ {covariates}", df, status=df[event_key].to_numpy())
        _require_full_rank(model.exog)
        res = model.fit()
        i = res.model.exog_names.index(_RESERVED)
        coef, se = res.params[i], res.bse[i]
        return pd.DataFrame(
            {"term": [_COX_TERM], "coef": [coef], "se": [se], "stat": [coef / se], "pval": [res.pvalues[i]]}
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
    adata: AnnData,
    formula: str,
    *,
    method: Literal["ols", "mixedlm", "cox"] = "ols",
    replicate_key: str | None = None,
    duration_key: str | None = None,
    event_key: str | None = None,
    min_samples: int = 5,
    corr_method: str | None = "fdr_bh",
) -> pd.DataFrame:
    """
    Test every sample-level feature for association with sample-level covariates.

    The unit of replication is the sample (or the group of samples given by ``replicate_key``), never the cell.
    To test a subset of features, pass a view, e.g. ``adata[:, adata.var["kind"] == "log2fe"]``; multiple testing
    is corrected over the features passed.

    Parameters
    ----------
    adata
        Samples x features, e.g. from :func:`sample_features`.
    formula
        Right-hand side of a :mod:`statsmodels` formula over columns of ``adata.obs``, e.g.
        ``'~ C(condition) + age'``. For ``'ols'`` and ``'mixedlm'`` each feature is the response. For ``'cox'``
        each feature is a covariate of survival, adjusted for the terms of ``formula`` (``'~ 1'`` for none).
    method
        - ``'ols'`` - linear model. With ``replicate_key``, samples are first averaged within each replicate
          (pseudobulk), so each replicate (e.g. patient) counts once; covariates must then be constant within
          replicates.
        - ``'mixedlm'`` - linear mixed model with a random intercept per ``replicate_key``, for several samples per
          patient.
        - ``'cox'`` - Cox proportional hazards model of ``duration_key`` and ``event_key``. With ``replicate_key``,
          samples are first averaged within each replicate.
    replicate_key
        Column of ``adata.obs`` with the unit of replication shared by several samples (e.g. patient): the unit
        samples are averaged over for ``'ols'`` and ``'cox'``, and the random intercept for ``'mixedlm'``.
        The average is unweighted: every sample of a replicate in which the feature is defined counts the same,
        whatever its number of cells (``.layers['support']`` of :func:`sample_features`), so the samples averaged
        can differ between features.
    duration_key, event_key
        Columns of ``adata.obs`` with survival time and event indicator (1 = event). Only for ``'cox'``.
    min_samples
        Minimum number of observations (after dropping missing values and pseudobulking) to test a feature.
    corr_method
        Multiple testing correction of :func:`statsmodels.stats.multitest.multipletests`, applied per term over the
        features that were tested.

    Returns
    -------
    One row per feature and term: the columns of ``adata.var``, ``coef``, ``se``, ``stat``, ``pval``, ``pval_adj``
    (if ``corr_method`` is given), ``n`` (observations used) and ``status``. Every feature is reported: a feature
    that was not tested has missing statistics and ``status`` says why (too few samples, constant, a failed or
    rank-deficient fit); a fit that raised a convergence warning keeps its statistics, with the warning in
    ``status``. For ``'cox'`` the term is ``'feature'``.
    """
    from statsmodels.tools.sm_exceptions import ConvergenceWarning

    if method not in ("ols", "mixedlm", "cox"):
        raise ValueError(f"`method` must be 'ols', 'mixedlm' or 'cox', got {method!r}.")
    if method == "mixedlm" and replicate_key is None:
        raise ValueError("`method='mixedlm'` needs `replicate_key`.")
    if method == "cox" and (duration_key is None or event_key is None):
        raise ValueError("`method='cox'` needs `duration_key` and `event_key`.")
    lhs, tilde, rhs = formula.partition("~")
    rhs = rhs.strip()
    if not tilde or lhs.strip():
        raise ValueError(f"`formula` must be a right-hand side such as '~ C(condition)', got {formula!r}.")
    if method != "cox" and rhs in ("", "1", "0"):
        raise ValueError(f"`formula` {formula!r} has no term to test.")
    if _RESERVED in adata.obs:
        raise ValueError(f"`adata.obs` must not have a column named {_RESERVED!r}.")

    names = list(adata.var_names)
    x = adata.X
    x = np.asarray(x.toarray() if hasattr(x, "toarray") else x, dtype=float)
    obs = adata.obs
    # rows missing anything the model uses are dropped up front, so every model sees the same rows
    used = [c for c in obs.columns if _in_formula(c, rhs) or c in (replicate_key, duration_key, event_key)]
    if method == "mixedlm" and obs[replicate_key].value_counts().max() < 2:
        raise ValueError(
            f"Every {replicate_key!r} has a single sample, so a random intercept cannot be estimated and the mixed "
            "model's tests are not calibrated. Use `method='ols'` (with one sample per replicate it is the same "
            "pseudobulk)."
        )
    if replicate_key is not None and method != "mixedlm":
        by = obs.groupby(replicate_key, observed=True)
        varying = [c for c in used if c != replicate_key and by[c].nunique().max() > 1]
        if varying:
            raise ValueError(
                f"Pseudobulking by {replicate_key!r} needs covariates constant within replicates; {varying} vary. "
                "Average them per replicate first, or use `method='mixedlm'`."
            )
    fits, skipped = [], []
    for j, name in enumerate(names):
        df = obs.assign(**{_RESERVED: x[:, j]}).dropna(subset=[_RESERVED, *used])
        if replicate_key is not None and method != "mixedlm":
            agg = {c: "first" for c in df.columns if c != replicate_key}
            agg[_RESERVED] = "mean"
            df = df.groupby(replicate_key, observed=True).agg(agg).reset_index()
        if len(df) < min_samples:
            skipped.append((name, len(df), "too few samples"))
            continue
        if df[_RESERVED].nunique() < 2:
            skipped.append((name, len(df), "constant"))
            continue
        if method == "mixedlm" and df[replicate_key].value_counts().max() < 2:
            # only single-sample replicates left for this feature: the random intercept is not identifiable
            skipped.append((name, len(df), "single-sample replicates"))
            continue
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")  # recorded, not shown: one fit per feature
            try:
                fit = _fit_one(
                    df, rhs, method=method, replicate_key=replicate_key, duration_key=duration_key, event_key=event_key
                )
            except (np.linalg.LinAlgError, ValueError) as e:
                skipped.append((name, len(df), f"failed: {e}"))
                continue
        conv = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
        fits.append(fit.assign(feature=name, n=len(df), status=f"warning: {conv[0]}" if conv else "ok"))
    # untested features get a row per term of the tested ones, so they stay in the table
    terms = list(dict.fromkeys(t for f in fits for t in f["term"])) or ([_COX_TERM] if method == "cox" else [None])
    rows = [r for f in fits for r in f.to_dict("records")]
    rows += [{"feature": name, "term": t, "n": n, "status": status} for name, n, status in skipped for t in terms]
    cols = ["feature", "term", "coef", "se", "stat", "pval", "n", "status"]
    out = pd.DataFrame(rows, columns=cols)
    if corr_method is not None:
        out.insert(cols.index("pval") + 1, "pval_adj", np.nan)
        for _, idx in out.groupby("term").groups.items():
            p = out.loc[idx, "pval"].to_numpy(dtype=float)
            ok = np.isfinite(p)
            if ok.any():
                adj = np.full(p.shape, np.nan)
                adj[ok] = multipletests(p[ok], method=corr_method)[1]
                out.loc[idx, "pval_adj"] = adj
    var = adata.var.drop(columns=[c for c in adata.var.columns if c in out.columns])
    out = out.join(var, on="feature")
    out = out[["feature", *var.columns, *(c for c in out.columns if c != "feature" and c not in var.columns)]]
    return out.sort_values("pval", kind="stable", na_position="last").reset_index(drop=True)
