"""Spatial building blocks for perturbation screens read out in tissue.

Perturbed cells in spatial CRISPR screens grow as clones (lesions, colonies). Testing how a perturbation
changes its *surroundings* needs two spatial primitives, provided here:

- :func:`label_components`: split each perturbation label into spatially connected components (clones),
  the unit of replication for neighbour-effect tests.
- :func:`neighbourhood_distances`: for every component, the cells outside all components within a given
  distance, with their distance to that component's nearest cell, from which distance bands are cut.

The statistical test itself (clone-level permutation) lives in :mod:`pertpy`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from spatialdata import SpatialData

from squidpy.gr._utils import _assert_categorical_obs, _assert_spatial_basis, extract_adata_if_sdata

__all__ = ["label_components", "neighbourhood_distances"]


def _libraries(adata: AnnData, library_key: str | None) -> list[np.ndarray]:
    if library_key is None:
        return [np.arange(adata.n_obs)]
    _assert_categorical_obs(adata, key=library_key)
    lib = adata.obs[library_key]
    return [np.flatnonzero((lib == c).to_numpy()) for c in lib.cat.categories]


def label_components(
    adata: AnnData | SpatialData,
    label_key: str,
    *,
    radius: float,
    min_cells: int = 1,
    spatial_key: str = "spatial",
    library_key: str | None = None,
    key_added: str | None = None,
    table_key: str | None = None,
    copy: bool = False,
) -> pd.Series | None:
    """Split each label into spatially connected components.

    Two cells carrying the same label are connected when they are at most ``radius`` apart; components
    are the connected groups. Components are found within each library, so they never span sections.

    Parameters
    ----------
    adata
        Annotated data, or :class:`~spatialdata.SpatialData` with ``table_key``.
    label_key
        Categorical column in :attr:`~anndata.AnnData.obs` with the label to split, e.g. the perturbation
        a cell carries. Cells with a missing value are not part of any component.
    radius
        Linking distance, in the units of ``adata.obsm[spatial_key]``.
    min_cells
        Components with fewer cells are dropped (their cells get a missing value), e.g. stray cells with
        a mis-assigned barcode.
    spatial_key
        Key in :attr:`~anndata.AnnData.obsm` with the coordinates.
    library_key
        Categorical column in :attr:`~anndata.AnnData.obs` with the section; components never cross it.
    key_added
        Column written to :attr:`~anndata.AnnData.obs`; defaults to ``f"{label_key}_component"``.
    table_key
        Table to use when ``adata`` is a :class:`~spatialdata.SpatialData`.
    copy
        Return the component labels instead of writing them.

    Returns
    -------
    If ``copy = True``, a categorical :class:`~pandas.Series` indexed like ``adata.obs``, with values
    ``"{label}:{library}:{k}"`` and missing values for cells outside any component. Otherwise writes it
    to ``adata.obs[key_added]``.
    """
    adata = extract_adata_if_sdata(adata, table_key=table_key)
    _assert_categorical_obs(adata, key=label_key)
    _assert_spatial_basis(adata, key=spatial_key)
    if radius <= 0:
        raise ValueError(f"Expected `radius` to be positive, found `{radius}`.")
    if min_cells < 1:
        raise ValueError(f"Expected `min_cells` to be at least `1`, found `{min_cells}`.")
    coords = np.asarray(adata.obsm[spatial_key])[:, :2].astype(np.float64)
    labels = adata.obs[label_key]
    lib_names = adata.obs[library_key].astype(str).to_numpy() if library_key else np.full(adata.n_obs, "0")
    out = np.full(adata.n_obs, None, dtype=object)
    for rows in _libraries(adata, library_key):
        lab = labels.iloc[rows]
        for value, idx in lab.groupby(lab, observed=True).indices.items():
            cells = rows[idx]
            pairs = cKDTree(coords[cells]).query_pairs(radius, output_type="ndarray")
            n = len(cells)
            graph = sp.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
            _, comp = connected_components(graph, directed=False)
            sizes = np.bincount(comp)
            for k, c in enumerate(np.flatnonzero(sizes >= min_cells)):
                out[cells[comp == c]] = f"{value}:{lib_names[cells[0]]}:{k}"
    res = pd.Series(pd.Categorical(out), index=adata.obs_names, name=key_added or f"{label_key}_component")
    if copy:
        return res
    adata.obs[res.name] = res
    return None


def neighbourhood_distances(
    adata: AnnData | SpatialData,
    component_key: str,
    *,
    max_distance: float,
    spatial_key: str = "spatial",
    library_key: str | None = None,
    table_key: str | None = None,
) -> pd.DataFrame:
    """Distance from each component to the surrounding cells.

    For every component in ``adata.obs[component_key]`` (e.g. from :func:`label_components`), lists the
    cells that belong to no component and lie within ``max_distance`` of it, with their distance to the
    component's nearest cell. A cell near several components appears once per component. Distance bands
    ``(r0, r1]`` are rows with ``r0 < distance <= r1``.

    Parameters
    ----------
    adata
        Annotated data, or :class:`~spatialdata.SpatialData` with ``table_key``.
    component_key
        Categorical column in :attr:`~anndata.AnnData.obs` with component ids; missing = not in one.
    max_distance
        Largest distance to report, in the units of ``adata.obsm[spatial_key]``.
    spatial_key
        Key in :attr:`~anndata.AnnData.obsm` with the coordinates.
    library_key
        Categorical column in :attr:`~anndata.AnnData.obs` with the section; distances never cross it.
    table_key
        Table to use when ``adata`` is a :class:`~spatialdata.SpatialData`.

    Returns
    -------
    :class:`~pandas.DataFrame` with columns ``component``, ``obs_name`` (the neighbouring cell) and
    ``distance``, one row per (component, neighbouring cell) pair.
    """
    adata = extract_adata_if_sdata(adata, table_key=table_key)
    _assert_categorical_obs(adata, key=component_key)
    _assert_spatial_basis(adata, key=spatial_key)
    if max_distance <= 0:
        raise ValueError(f"Expected `max_distance` to be positive, found `{max_distance}`.")
    coords = np.asarray(adata.obsm[spatial_key])[:, :2].astype(np.float64)
    comp = adata.obs[component_key]
    codes = comp.cat.codes.to_numpy()
    frames = []
    for rows in _libraries(adata, library_key):
        c = codes[rows]
        free, member = rows[c < 0], rows[c >= 0]
        if not len(free) or not len(member):
            continue
        m = cKDTree(coords[free]).sparse_distance_matrix(cKDTree(coords[member]), max_distance,
                                                          output_type="coo_matrix")
        df = pd.DataFrame({"cell": free[m.row], "component": codes[member[m.col]], "distance": m.data})
        # coincident points (distance exactly 0) are not stored by the sparse matrix: not neighbours
        frames.append(df.groupby(["component", "cell"], sort=False, as_index=False)["distance"].min())
    if not frames:
        return pd.DataFrame({"component": pd.Categorical([], categories=comp.cat.categories),
                             "obs_name": pd.Series([], dtype=object), "distance": pd.Series([], dtype=float)})
    res = pd.concat(frames, ignore_index=True)
    return pd.DataFrame({
        "component": pd.Categorical.from_codes(res["component"].to_numpy(), categories=comp.cat.categories),
        "obs_name": adata.obs_names.to_numpy()[res["cell"].to_numpy()],
        "distance": res["distance"].to_numpy(),
    })
