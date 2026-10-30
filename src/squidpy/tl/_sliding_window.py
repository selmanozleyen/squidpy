from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd
from anndata import AnnData
from scanpy import logging as logg
from spatialdata import SpatialData

from squidpy._compat import old_positionals
from squidpy._docs import d
from squidpy.gr._utils import _save_data, extract_adata_if_sdata

__all__ = ["sliding_window"]


@d.dedent
@old_positionals(
    "library_key",
    "window_size",
    "overlap",
    "coord_columns",
    "sliding_window_key",
    "spatial_key",
    "drop_partial_windows",
    "copy",
)
def sliding_window(
    adata: AnnData | SpatialData,
    *,
    library_key: str | None = None,
    window_size: int | None = None,
    overlap: int = 0,
    coord_columns: tuple[str, str] = ("globalX", "globalY"),
    sliding_window_key: str = "sliding_window_assignment",
    spatial_key: str = "spatial",
    drop_partial_windows: bool = False,
    copy: bool = False,
    table_key: str | None = None,
) -> pd.DataFrame | None:
    """
    Divide a tissue slice into regulary shaped spatially contiguous regions (windows).

    Parameters
    ----------
    %(adata)s
    %(table_key)s
    window_size: int
        Size of the sliding window.
    %(library_key)s
    coord_columns: Tuple[str, str]
        Tuple of column names in `adata.obs` that specify the coordinates (x, y), e.i. ('globalX', 'globalY')
    sliding_window_key: str
        Base name for sliding window columns.
    overlap: int
        Overlap size between consecutive windows. (0 = no overlap)
    %(spatial_key)s
    drop_partial_windows: bool
        If True, drop windows that are smaller than the window size at the borders.
    copy: bool
        If True, return the result, otherwise save it to the adata object.

    Returns
    -------
    If ``copy = True``, returns the sliding window annotation(s) as pandas dataframe
    Otherwise, stores the sliding window annotation(s) in .obs.
    """
    if overlap < 0:
        raise ValueError("Overlap must be non-negative.")

    adata = extract_adata_if_sdata(adata, table_key=table_key)

    # extract coordinates of observations
    x_col, y_col = coord_columns
    if x_col in adata.obs and y_col in adata.obs:
        coords = adata.obs[[x_col, y_col]].copy()
    elif spatial_key in adata.obsm:
        coords = pd.DataFrame(
            adata.obsm[spatial_key][:, :2],
            index=adata.obs.index,
            columns=[x_col, y_col],
        )
    else:
        raise ValueError(
            f"Coordinates not found. Provide `{coord_columns}` in `adata.obs` or specify a suitable `spatial_key` in `adata.obsm`."
        )

    # infer window size if not provided
    if window_size is None:
        coord_range = max(
            coords[x_col].max() - coords[x_col].min(),
            coords[y_col].max() - coords[y_col].min(),
        )
        # mostly arbitrary choice, except that full integers usually generate windows with 1-2 cells at the borders
        window_size = max(int(np.floor(coord_range // 3.95)), 1)

    if window_size <= 0:
        raise ValueError("Window size must be larger than 0.")

    if library_key is not None and library_key not in adata.obs:
        raise ValueError(f"Library key '{library_key}' not found in adata.obs")

    # per library, cells sorted by x; each window binary-searches its x-range
    if sliding_window_key in adata.obs:
        logg.warning(f"Overwriting existing column '{sliding_window_key}' in adata.obs.")
    x_all, y_all = coords[x_col].to_numpy(), coords[y_col].to_numpy()
    labels = np.full(len(coords), np.nan, dtype=object) if overlap == 0 else None  # one label per cell
    columns: dict[str, np.ndarray] = {}  # overlap > 0: one membership column per window
    cells_of = (
        {None: np.arange(len(coords))}
        if library_key is None
        else adata.obs.groupby(library_key, observed=True, sort=False).indices
    )

    for lib, lib_pos in cells_of.items():
        x, y = x_all[lib_pos], y_all[lib_pos]
        windows = _calculate_window_corners(
            min_x=np.nanmin(x),
            max_x=np.nanmax(x),
            min_y=np.nanmin(y),
            max_y=np.nanmax(y),
            window_size=window_size,
            overlap=overlap,
            drop_partial_windows=drop_partial_windows,
        )
        by_x = np.argsort(x, kind="stable")
        x_sorted = x[by_x]
        lib_key = f"{lib}_" if lib is not None else ""

        for idx, (x_start, x_end, y_start, y_end) in enumerate(windows.itertuples(index=False)):
            cand = by_x[np.searchsorted(x_sorted, x_start, "left") : np.searchsorted(x_sorted, x_end, "right")]
            inside = lib_pos[cand[(y[cand] >= y_start) & (y[cand] <= y_end)]]
            if overlap == 0:
                labels[inside] = f"{lib_key}window_{idx}"  # windows share edges: the later one wins
            else:
                member = np.zeros(len(coords), dtype=bool)
                member[inside] = True
                columns[f"{sliding_window_key}_{lib_key}window_{idx}"] = member

    if overlap == 0:
        # ordered categorical of the windows that received cells; cells outside every window stay NaN
        assigned = pd.unique(labels[pd.notna(labels)])
        sliding_window_df = pd.DataFrame(
            {
                sliding_window_key: pd.Categorical(
                    labels, ordered=True, categories=sorted(assigned, key=lambda x: int(x.split("_")[-1]))
                )
            },
            index=adata.obs.index,
        )
    else:
        sliding_window_df = pd.DataFrame(columns, index=adata.obs.index)

    sliding_window_df[x_col] = coords[x_col]
    sliding_window_df[y_col] = coords[y_col]

    if copy:
        return sliding_window_df
    for col_name, col_data in sliding_window_df.items():
        _save_data(adata, attr="obs", key=col_name, data=col_data)


def _calculate_window_corners(
    min_x: int,
    max_x: int,
    min_y: int,
    max_y: int,
    window_size: int,
    *,
    overlap: int = 0,
    drop_partial_windows: bool = False,
) -> pd.DataFrame:
    """
    Calculate the corner points of all windows covering the area from min_x to max_x and min_y to max_y,
    with specified window_size and overlap.

    Parameters
    ----------
    min_x: float
        minimum X coordinate
    max_x: float
        maximum X coordinate
    min_y: float
        minimum Y coordinate
    max_y: float
        maximum Y coordinate
    window_size: float
        size of each window
    overlap: float
        overlap between consecutive windows (must be less than window_size)
    drop_partial_windows: bool
        if True, drop border windows that are smaller than window_size;
        if False, create smaller windows at the borders to cover the remaining space.

    Returns
    -------
    windows: pandas DataFrame with columns ['x_start', 'x_end', 'y_start', 'y_end']
    """
    if overlap < 0:
        raise ValueError("Overlap must be non-negative.")
    if overlap >= window_size:
        raise ValueError("Overlap must be less than the window size.")

    x_step = window_size - overlap
    y_step = window_size - overlap

    # Generate starting points
    x_starts = np.arange(min_x, max_x, x_step)
    y_starts = np.arange(min_y, max_y, y_step)

    # Create all combinations of x and y starting points
    starts = list(product(x_starts, y_starts))
    windows = pd.DataFrame(starts, columns=["x_start", "y_start"])
    windows["x_end"] = windows["x_start"] + window_size
    windows["y_end"] = windows["y_start"] + window_size

    # Adjust windows that extend beyond the bounds
    if not drop_partial_windows:
        windows["x_end"] = windows["x_end"].clip(upper=max_x)
        windows["y_end"] = windows["y_end"].clip(upper=max_y)
    else:
        valid_windows = (windows["x_end"] <= max_x) & (windows["y_end"] <= max_y)
        windows = windows[valid_windows]

    windows = windows.reset_index(drop=True)
    return windows[["x_start", "x_end", "y_start", "y_end"]]
