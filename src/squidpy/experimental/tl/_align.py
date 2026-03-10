"""Spatial alignment for squidpy.

Dispatches to the appropriate backend based on input types:

- **Point-to-point**: :mod:`ott` optimal transport (optional)
- **Point-to-image** / **Image-to-image**: STalign LDDMM (requires ``torch``)

Install optional dependencies::

    pip install 'squidpy[ott]'       # point-to-point OT alignment
    pip install 'squidpy[torch]'     # image-based LDDMM alignment
    pip install 'squidpy[align]'     # both

Notes
-----
STalign reference: Clifton *et al.*, Nature Communications 14, 8123 (2023).
ott-jax reference: Cuturi *et al.*, (2022).
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from anndata import AnnData
    from numpy.typing import NDArray
    import spatialdata as sd


# ---------------------------------------------------------------------------
# Optional-dependency helpers
# ---------------------------------------------------------------------------

def _check_stalign_deps() -> None:
    """Ensure ``torch`` is available for the STalign/LDDMM backend."""
    try:
        import torch  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Image-based alignment requires PyTorch.\n"
            "Install with:  pip install 'squidpy[torch]'"
        ) from e


def _check_ott() -> None:
    """Ensure ``ott-jax`` is available for point-to-point OT alignment."""
    try:
        import ott  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Point-to-point OT alignment requires ott-jax.\n"
            "Install with:  pip install ott-jax"
        ) from e


def _has_ott() -> bool:
    try:
        import ott  # noqa: F401
        return True
    except ImportError:
        return False


def _has_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Input-type detection
# ---------------------------------------------------------------------------

_InputType = Literal["anndata", "spatialdata", "image", "coords"]


def _detect_input_type(data: Any) -> _InputType:
    """Return ``'anndata'``, ``'spatialdata'``, ``'image'``, or ``'coords'``."""
    # AnnData
    if hasattr(data, "obsm") and hasattr(data, "obs"):
        return "anndata"

    # SpatialData (lazy import)
    try:
        import spatialdata as sd
        if isinstance(data, sd.SpatialData):
            return "spatialdata"
    except ImportError:
        pass

    # numpy array
    if isinstance(data, np.ndarray):
        if data.ndim == 2 and data.shape[1] == 2:
            return "coords"
        return "image"

    # anything convertible to ndarray
    try:
        arr = np.asarray(data)
        if arr.ndim == 2 and arr.shape[1] == 2:
            return "coords"
        return "image"
    except (ValueError, TypeError):
        pass

    raise TypeError(f"Cannot determine input type for {type(data)}")


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _prep_image(img: "NDArray[np.floating]") -> "NDArray[np.floating]":
    """Normalise an image to ``(C=3, H, W)`` float in ``[0, 1]``."""
    img = np.asarray(img, dtype=np.float64)
    if img.ndim == 2:
        img = img[np.newaxis, :, :]
    elif img.ndim == 3 and img.shape[-1] <= 4 and img.shape[0] > 4:
        img = np.moveaxis(img, -1, 0)
    if img.shape[0] == 1:
        img = np.vstack([img] * 3)
    elif img.shape[0] == 4:
        img = img[:3]
    vmin, vmax = img.min(), img.max()
    return (img - vmin) / (vmax - vmin + 1e-8)


def _coords_to_adata(
    coords: "NDArray[np.floating]",
    spatial_key: str = "spatial",
) -> "AnnData":
    """Create a minimal AnnData from an ``(N, 2)`` coordinate array."""
    import anndata as ad

    coords = np.asarray(coords)
    adata = ad.AnnData(X=np.zeros((len(coords), 1)))
    adata.obsm[spatial_key] = coords
    return adata


# ---------------------------------------------------------------------------
# Spatial binning helpers
# ---------------------------------------------------------------------------

def _bin_points(
    coords: "NDArray[np.floating]",
    n_bins: int | tuple[int, int],
    *,
    range_xy: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> tuple["NDArray[np.floating]", "NDArray[np.floating]", "NDArray[np.integer]"]:
    """Bin 2-D coordinates into a regular grid.

    Parameters
    ----------
    coords
        (N, 2) array of (x, y) coordinates.
    n_bins
        Number of bins per axis (single int -> same for both axes).
    range_xy
        ((xmin, xmax), (ymin, ymax)) for the grid.  Computed from
        *coords* if ``None``.

    Returns
    -------
    centroids
        (B, 2) array -- centroid of each non-empty bin.
    weights
        (B,) array -- fraction of points in each bin (sums to 1).
    bin_labels
        (N,) array -- bin index for each original point (indices into
        *centroids*).
    """
    coords = np.asarray(coords, dtype=np.float64)
    if isinstance(n_bins, int):
        n_bins = (n_bins, n_bins)

    if range_xy is None:
        xmin, ymin = coords.min(axis=0)
        xmax, ymax = coords.max(axis=0)
        range_xy = ((xmin, xmax), (ymin, ymax))

    (xmin, xmax), (ymin, ymax) = range_xy
    bx = np.clip(
        ((coords[:, 0] - xmin) / (xmax - xmin) * n_bins[0]).astype(int),
        0, n_bins[0] - 1,
    )
    by = np.clip(
        ((coords[:, 1] - ymin) / (ymax - ymin) * n_bins[1]).astype(int),
        0, n_bins[1] - 1,
    )
    flat = bx * n_bins[1] + by

    unique_bins, inverse, counts = np.unique(flat, return_inverse=True, return_counts=True)

    centroids = np.zeros((len(unique_bins), 2), dtype=np.float64)
    np.add.at(centroids[:, 0], inverse, coords[:, 0])
    np.add.at(centroids[:, 1], inverse, coords[:, 1])
    centroids /= counts[:, np.newaxis]

    weights = counts.astype(np.float64)
    weights /= weights.sum()

    return centroids, weights, inverse



def _choose_n_bins(n_points: int, max_bins: int = 5000) -> int:
    """Pick a grid resolution for one point cloud.

    Target roughly *max_bins* non-empty bins, scaled down for smaller
    datasets.  Returns bins-per-axis (the grid will be n x n).
    """
    target = min(max_bins, max(200, n_points // 20))
    return max(10, int(np.sqrt(target)))


def _warn_scale_mismatch(
    coords_src: "NDArray[np.floating]",
    coords_tgt: "NDArray[np.floating]",
    *,
    threshold: float = 10.0,
) -> None:
    """Warn if the coordinate extents of source and target differ greatly."""
    ext_src = np.ptp(coords_src, axis=0).max()
    ext_tgt = np.ptp(coords_tgt, axis=0).max()
    if ext_src < 1e-12 or ext_tgt < 1e-12:
        return
    ratio = max(ext_src, ext_tgt) / min(ext_src, ext_tgt)
    if ratio > threshold:
        warnings.warn(
            f"Spatial coordinate extents differ by {ratio:.0f}x "
            f"(source extent ~ {ext_src:.1f}, target extent ~ {ext_tgt:.1f}). "
            f"The two datasets may be in different coordinate systems. "
            f"Consider pre-registering them or using LDDMM/STalign instead.",
            stacklevel=4,
        )


# ---------------------------------------------------------------------------
# Backend: ott-jax  (point-to-point)
# ---------------------------------------------------------------------------

_POINTCLOUD_KEYS = frozenset({
    "epsilon", "relative_epsilon", "scale_cost", "cost_fn", "batch_size",
})
_SOLVE_KEYS = frozenset({"tau_a", "tau_b", "rank"})


def _solve_sinkhorn(
    x: "NDArray[np.floating]",
    y: "NDArray[np.floating]",
    a: "NDArray[np.floating] | None" = None,
    b: "NDArray[np.floating] | None" = None,
    verbose: bool = False,
    **kwargs: Any,
) -> Any:
    """Solve a linear OT problem with OTT-JAX.

    Parameters
    ----------
    x, y
        (n, d) and (m, d) point clouds.
    a, b
        Source / target marginals (probability vectors).  Uniform if ``None``.
    verbose
        Print solver diagnostics.
    **kwargs
        Routed automatically:

        - ``epsilon``, ``scale_cost``, ``cost_fn``, ``batch_size`` ->
          ``ott.geometry.pointcloud.PointCloud``
        - ``tau_a``, ``tau_b``, ``rank`` ->
          ``ott.solvers.linear.solve``
        - everything else (``max_iterations``, ``threshold``, ...) ->
          ``Sinkhorn`` / ``LRSinkhorn`` constructor via ``solve(**kwargs)``

    Returns
    -------
    ``SinkhornOutput`` (or ``LRSinkhornOutput``).  Access the transport
    matrix via ``.matrix`` (jax array).
    """
    import jax.numpy as jnp
    from ott.geometry.pointcloud import PointCloud
    from ott.solvers.linear import solve as ot_solve

    geom_kw = {k: kwargs.pop(k) for k in list(kwargs) if k in _POINTCLOUD_KEYS}
    solve_kw = {k: kwargs.pop(k) for k in list(kwargs) if k in _SOLVE_KEYS}
    sinkhorn_kw = kwargs

    # Normalise costs so that epsilon values are interpretable.
    # Without this, spatial coordinates in the thousands produce squared
    # Euclidean costs in the millions and the auto-epsilon is far too large.
    geom_kw.setdefault("scale_cost", "mean")

    # OTT-JAX default epsilon is 0.05 * mean_cost (after scaling ~= 0.05).
    # For spatial alignment this is too diffuse -- source bins spread mass
    # over large neighborhoods instead of matching nearby targets.  A tighter
    # epsilon produces sharper transport plans at the cost of more iterations.
    geom_kw.setdefault("epsilon", 1e-2)

    x_jnp = jnp.asarray(x)
    y_jnp = jnp.asarray(y)
    geom = PointCloud(x_jnp, y_jnp, **geom_kw)

    if verbose:
        print(f"[ott] PointCloud: {geom_kw or '(defaults)'}")
        print(f"[ott] solve:      {solve_kw or '(defaults)'}")
        print(f"[ott] Sinkhorn:   {sinkhorn_kw or '(defaults)'}")
        print(f"[ott] cost matrix: mean={float(geom.mean_cost_matrix):.4f}  "
              f"shape=({len(x)}, {len(y)})  "
              f"epsilon={float(geom.epsilon):.6f}")

    a_jnp = jnp.asarray(a) if a is not None else None
    b_jnp = jnp.asarray(b) if b is not None else None

    out = ot_solve(geom, a=a_jnp, b=b_jnp, **solve_kw, **sinkhorn_kw)

    converged = bool(out.converged)
    n_iters = int(out.n_iters)
    if verbose:
        print(f"[ott] converged={converged}  iterations={n_iters}")
    if not converged:
        warnings.warn(
            f"Sinkhorn did not converge in {n_iters} iterations. "
            "Consider increasing max_iterations or epsilon.",
            stacklevel=3,
        )

    return out


def _align_points_ot(
    adata_source: "AnnData",
    adata_target: "AnnData",
    *,
    spatial_key: str = "spatial",
    key_added: str = "spatial_aligned",
    copy: bool = False,
    verbose: bool = True,
    solve_kwargs: dict[str, Any] | None = None,
    align_kwargs: dict[str, Any] | None = None,
) -> "AnnData | None":
    """Align two point clouds via OTT-JAX Sinkhorn.

    Solves a linear OT problem using spatial coordinates as the cost.

    Parameters
    ----------
    adata_source / adata_target
        AnnData objects with spatial coordinates in ``obsm[spatial_key]``.
    solve_kwargs
        Forwarded to :func:`_solve_sinkhorn` / ``ott.solvers.linear.solve``.
        Useful keys: ``epsilon``, ``scale_cost``, ``max_iterations``,
        ``threshold``, ``tau_a``, ``tau_b``.
    align_kwargs
        ``mode`` (``'warp'`` | ``'affine'``, default ``'warp'``).
    """
    _check_ott()
    import jax.numpy as jnp

    solve_kw: dict[str, Any] = dict(solve_kwargs or {})
    align_kw: dict[str, Any] = dict(align_kwargs or {})

    coords_src = np.asarray(adata_source.obsm[spatial_key], dtype=np.float64)
    coords_tgt = np.asarray(adata_target.obsm[spatial_key], dtype=np.float64)
    n_src, n_tgt = len(coords_src), len(coords_tgt)

    if verbose:
        _warn_scale_mismatch(coords_src, coords_tgt)
        print(f"[ott] {n_src} source + {n_tgt} target = {n_src + n_tgt} cells")

    # Center both clouds so OT operates on shape differences only.
    src_mean = coords_src.mean(axis=0)
    tgt_mean = coords_tgt.mean(axis=0)
    src_c = coords_src - src_mean
    tgt_c = coords_tgt - tgt_mean

    if verbose:
        print("[ott] Solving Sinkhorn ...")

    ot_out = _solve_sinkhorn(src_c, tgt_c, verbose=verbose, **solve_kw)
    T = ot_out.matrix  # jax array (n_src, n_tgt)

    mode = align_kw.pop("mode", "warp")
    if verbose:
        print(f"[ott] Computing aligned coordinates (mode={mode!r}) ...")

    row_sums = T.sum(axis=1)
    row_sums = jnp.where(row_sums > 1e-10, row_sums, 1.0)
    T_norm = T / row_sums[:, None]

    tgt_j = jnp.asarray(tgt_c)

    if mode == "warp":
        aligned_c = np.asarray(T_norm @ tgt_j)
    elif mode == "affine":
        from scipy.linalg import svd as _svd

        out = np.asarray(T_norm @ tgt_j)
        H = src_c.T @ out
        U, _, Vt = _svd(H)
        R = Vt.T @ U.T
        aligned_c = (R @ src_c.T).T
    else:
        raise ValueError(f"Unsupported alignment mode {mode!r}. Use 'warp' or 'affine'.")

    # Shift back to target coordinate space.
    aligned_coords = aligned_c + tgt_mean

    adata = adata_source.copy() if copy else adata_source
    adata.obsm[key_added] = aligned_coords
    adata.uns["spatial_alignment"] = {
        "method": "optimal_transport",
        "backend": "ott-jax",
        "spatial_key": spatial_key,
        "mode": mode,
    }

    if verbose:
        print(f"[ott] Aligned coordinates stored in obsm['{key_added}']")

    return adata if copy else None


# ---------------------------------------------------------------------------
# Backend: ott-jax  (binned point-to-point for large data)
# ---------------------------------------------------------------------------


def prepare_bins(
    adata_source: "AnnData",
    adata_target: "AnnData",
    *,
    spatial_key: str = "spatial",
    n_bins: int | tuple[int, int] | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Compute spatial bins for source and target (without running OT).

    This is the first stage of the binned OT alignment pipeline,
    extracted so you can inspect and plot the bins before committing
    to the full alignment.  The returned dictionary is stored in
    ``adata_source.uns["spatial_alignment_diag"]`` so that
    :func:`plot_ot_alignment` with ``panels=["bins"]`` can display
    them immediately.

    Parameters
    ----------
    adata_source / adata_target
        AnnData objects with spatial coordinates in
        ``obsm[spatial_key]``.
    spatial_key
        Key in ``obsm`` for the spatial coordinates.
    n_bins
        Grid resolution per axis.  ``None`` picks automatically.
        An ``int`` uses the same grid for both.  A tuple
        ``(src_bins, tgt_bins)`` sets each side independently.
    verbose
        Print summary.

    Returns
    -------
    Dictionary with keys ``cen_src``, ``cen_tgt``, ``w_src``,
    ``w_tgt``, ``src_mean``, ``tgt_mean``.  Also stored in
    ``adata_source.uns["spatial_alignment_diag"]``.
    """
    coords_src = np.asarray(adata_source.obsm[spatial_key], dtype=np.float64)
    coords_tgt = np.asarray(adata_target.obsm[spatial_key], dtype=np.float64)
    n_src = len(coords_src)
    n_tgt = len(coords_tgt)

    src_mean = coords_src.mean(axis=0)
    tgt_mean = coords_tgt.mean(axis=0)
    src_c = coords_src - src_mean
    tgt_c = coords_tgt - tgt_mean

    if n_bins is None:
        nb_src = _choose_n_bins(n_src)
        nb_tgt = _choose_n_bins(n_tgt)
    elif isinstance(n_bins, int):
        nb_src = nb_tgt = n_bins
    else:
        nb_src, nb_tgt = n_bins
    grid_src = (nb_src, nb_src)
    grid_tgt = (nb_tgt, nb_tgt)

    cen_src, w_src, _ = _bin_points(src_c, grid_src)
    cen_tgt, w_tgt, _ = _bin_points(tgt_c, grid_tgt)

    if verbose:
        print(
            f"[prepare] {n_src} source + {n_tgt} target cells"
        )
        print(
            f"[prepare] Binned: {len(cen_src)} src (grid {grid_src[0]}x{grid_src[1]}) "
            f"+ {len(cen_tgt)} tgt (grid {grid_tgt[0]}x{grid_tgt[1]})"
        )

    diag = {
        "cen_src": cen_src,
        "cen_tgt": cen_tgt,
        "w_src": w_src,
        "w_tgt": w_tgt,
        "alive": np.ones(len(cen_src), dtype=bool),
        "warped_centroids": cen_src.copy(),
        "src_mean": src_mean,
        "tgt_mean": tgt_mean,
    }
    adata_source.uns["spatial_alignment_diag"] = diag
    return diag


def _align_points_ot_binned(
    adata_source: "AnnData",
    adata_target: "AnnData",
    *,
    spatial_key: str = "spatial",
    key_added: str = "spatial_aligned",
    copy: bool = False,
    verbose: bool = True,
    n_bins: int | tuple[int, int] | None = None,
    solve_kwargs: dict[str, Any] | None = None,
    align_kwargs: dict[str, Any] | None = None,
) -> "AnnData | None":
    """Align two large point clouds by binning, solving OT on centroids,
    then interpolating back to original cells.

    Steps
    -----
    1. Bin source and target coordinates into a regular spatial grid.
    2. Compute the centroid and weight (fraction of points) for each
       non-empty bin.
    3. Solve Sinkhorn OT (via ott-jax) on the (much smaller) centroid
       point clouds, using bin weights as marginals ``a`` and ``b``.
    4. Extract the affine or warp transformation from the coarse
       transport plan and apply it to *all* original source cells.

    Each side is binned at its own resolution proportional to its cell
    count so that denser datasets get finer grids.

    Parameters
    ----------
    n_bins
        Grid resolution per axis.  ``None`` picks automatically and
        independently for source and target based on cell counts.
        An ``int`` uses the same grid for both sides.
        A tuple ``(src_bins, tgt_bins)`` sets each side independently.
    """
    _check_ott()

    solve_kw: dict[str, Any] = dict(solve_kwargs or {})
    align_kw: dict[str, Any] = dict(align_kwargs or {})

    coords_src = np.asarray(adata_source.obsm[spatial_key], dtype=np.float64)
    coords_tgt = np.asarray(adata_target.obsm[spatial_key], dtype=np.float64)
    n_src = len(coords_src)
    n_tgt = len(coords_tgt)

    if verbose:
        _warn_scale_mismatch(coords_src, coords_tgt)

    # Center both clouds so OT operates on shape differences only.
    # The bulk translation is trivially added back at the end.
    src_mean = coords_src.mean(axis=0)
    tgt_mean = coords_tgt.mean(axis=0)
    src_c = coords_src - src_mean
    tgt_c = coords_tgt - tgt_mean

    # Resolve per-side grid resolutions.
    if n_bins is None:
        nb_src = _choose_n_bins(n_src)
        nb_tgt = _choose_n_bins(n_tgt)
    elif isinstance(n_bins, int):
        nb_src = nb_tgt = n_bins
    else:
        nb_src, nb_tgt = n_bins
    grid_src = (nb_src, nb_src)
    grid_tgt = (nb_tgt, nb_tgt)

    # -- Step 1: Bin each centered point cloud --------------------------------
    cen_src, w_src, inv_src = _bin_points(src_c, grid_src)
    cen_tgt, w_tgt, inv_tgt = _bin_points(tgt_c, grid_tgt)

    if verbose:
        print(
            f"[ott-binned] Original: {n_src} source + {n_tgt} target = {n_src + n_tgt} cells"
        )
        print(
            f"[ott-binned] Binned:   {len(cen_src)} source bins (grid {grid_src[0]}x{grid_src[1]}) + "
            f"{len(cen_tgt)} target bins (grid {grid_tgt[0]}x{grid_tgt[1]})"
        )

    # -- Step 2: Solve Sinkhorn with weighted marginals -----------------------
    if verbose:
        print("[ott-binned] Solving Sinkhorn on binned centroids ...")

    import jax.numpy as jnp

    ot_out = _solve_sinkhorn(cen_src, cen_tgt, a=w_src, b=w_tgt, verbose=verbose, **solve_kw)
    T = ot_out.matrix  # jax array

    # -- Step 3: Extract transformation from coarse transport plan ------------
    mode = align_kw.pop("mode", "warp")
    if verbose:
        print(f"[ott-binned] Computing transformation (mode={mode!r}) ...")

    row_sums = T.sum(axis=1)
    alive = np.asarray(row_sums > 1e-10)
    row_sums_safe = jnp.where(row_sums > 1e-10, row_sums, 1.0)
    T_norm = T / row_sums_safe[:, None]

    if verbose:
        n_dead = int((~alive).sum())
        if n_dead:
            print(f"[ott-binned] {n_dead}/{len(alive)} source bins received no mass")

    # Weighted target positions for each source bin (the "OT barycenter").
    warped_centroids = np.asarray(T_norm @ jnp.asarray(cen_tgt))

    if mode == "warp":
        from scipy.interpolate import RBFInterpolator

        displacements = warped_centroids - cen_src

        interp = RBFInterpolator(
            cen_src[alive], displacements[alive], kernel="thin_plate_spline",
        )
        disp_all = interp(src_c)
        aligned_c = src_c + disp_all
    elif mode == "affine":
        from scipy.linalg import svd as _svd

        out = np.asarray(T_norm[alive] @ jnp.asarray(cen_tgt))
        src_centered_alive = cen_src[alive]
        w_alive = w_src[alive]
        H = (src_centered_alive * w_alive[:, np.newaxis]).T @ out
        U, _, Vt = _svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        aligned_c = (R @ src_c.T).T
    else:
        raise ValueError(f"Unsupported alignment mode {mode!r}. Use 'warp' or 'affine'.")

    # Shift back to target coordinate space.
    aligned_coords = aligned_c + tgt_mean

    # -- Store ----------------------------------------------------------------
    adata = adata_source.copy() if copy else adata_source
    adata.obsm[key_added] = aligned_coords
    adata.uns["spatial_alignment"] = {
        "method": "optimal_transport",
        "backend": "ott-jax",
        "binned": True,
        "grid_src": grid_src,
        "grid_tgt": grid_tgt,
        "n_bins_src": len(cen_src),
        "n_bins_tgt": len(cen_tgt),
        "spatial_key": spatial_key,
        "mode": mode,
    }
    adata.uns["spatial_alignment_diag"] = {
        "cen_src": cen_src,
        "cen_tgt": cen_tgt,
        "warped_centroids": warped_centroids,
        "T_norm": np.asarray(T_norm),
        "w_src": w_src,
        "w_tgt": w_tgt,
        "alive": alive,
        "src_mean": src_mean,
        "tgt_mean": tgt_mean,
    }

    if verbose:
        print(f"[ott-binned] Aligned coordinates stored in obsm['{key_added}']")

    return adata if copy else None


# ---------------------------------------------------------------------------
# Backend: STalign / LDDMM (point-to-image)
# ---------------------------------------------------------------------------

def _align_points_to_image_stalign(
    adata_source: "AnnData",
    target_image: "NDArray[np.floating]",
    *,
    spatial_key: str = "spatial",
    method: Literal["affine", "lddmm"] = "lddmm",
    resolution: float = 30.0,
    blur: float = 1.5,
    niter: int = 2000,
    diffeo_start: int = 100,
    a: float = 500.0,
    p: float = 2.0,
    sigmaM: float = 1.0,
    sigmaR: float = 5e5,
    initial_rotation_deg: float = 0.0,
    landmark_source: "NDArray[np.floating] | None" = None,
    landmark_target: "NDArray[np.floating] | None" = None,
    device: str = "cpu",
    verbose: bool = True,
    key_added: str = "spatial_aligned",
    copy: bool = False,
    **lddmm_kwargs: Any,
) -> "AnnData | None":
    """Align coordinates to a reference image via LDDMM.

    Source coordinates are rasterised into a Gaussian-kernel density image
    and then aligned to *target_image* using LDDMM.  The resulting
    transformation is applied back to the original point coordinates.

    Parameters
    ----------
    adata_source
        AnnData with spatial coordinates in ``obsm[spatial_key]``.
    target_image
        Target image ``(H, W)`` or ``(H, W, C)`` or ``(C, H, W)``.
    spatial_key
        Key in ``obsm`` for spatial coordinates.
    method
        ``"lddmm"`` or ``"affine"``.
    resolution
        Pixel size for rasterisation of source coordinates.
    blur
        Gaussian sigma (in pixels) for rasterisation.
    niter / diffeo_start / a / p / sigmaM / sigmaR
        LDDMM hyper-parameters.
    initial_rotation_deg
        Initial clockwise rotation in degrees.
    landmark_source / landmark_target
        Optional N×2 landmark arrays in ``(x, y)`` order.
    device
        PyTorch device (``"cpu"`` or ``"cuda:0"``).
    verbose
        Print progress.
    key_added
        obsm key for aligned coordinates.
    copy
        Return a modified copy of *adata_source*.
    **lddmm_kwargs
        Extra keyword arguments forwarded to ``LDDMM()``.

    Returns
    -------
    Modified *adata_source* (if *copy*) or ``None``.
    """
    _check_stalign_deps()

    from squidpy.experimental._lddmm import LDDMM, rasterize, transform_points_source_to_target
    from squidpy.experimental._lddmm._transforms import L_T_from_points

    coords = np.asarray(adata_source.obsm[spatial_key])
    xI, yI = coords[:, 0], coords[:, 1]

    # Prepare target image
    J = _prep_image(target_image)
    _, h_J, w_J = J.shape
    XJ = np.arange(w_J, dtype=np.float64)
    YJ = np.arange(h_J, dtype=np.float64)

    # Initial affine -------------------------------------------------------
    L_init, T_init = None, None
    if landmark_source is not None and landmark_target is not None:
        pts_I_rc = np.column_stack([landmark_source[:, 1], landmark_source[:, 0]])
        pts_J_rc = np.column_stack([landmark_target[:, 1], landmark_target[:, 0]])
        L_init, T_init = L_T_from_points(pts_I_rc, pts_J_rc)
        if verbose:
            print(f"[stalign] Initial affine from {len(pts_I_rc)} landmarks")
    elif initial_rotation_deg != 0.0:
        from squidpy.experimental._lddmm._transforms import compute_initial_affine
        L_init, T_init = compute_initial_affine(xI, yI, XJ, YJ, initial_rotation_deg)
        if verbose:
            print(f"[stalign] Initial rotation: {initial_rotation_deg}°")

    # Pre-transform source for rasterisation --------------------------------
    if L_init is not None:
        rc = np.column_stack([yI, xI])
        rc_t = (L_init @ rc.T).T + T_init
        yI_t, xI_t = rc_t[:, 0], rc_t[:, 1]
    else:
        xI_t, yI_t = xI, yI

    # Rasterise source coordinates ------------------------------------------
    XI, YI, I = rasterize(xI_t, yI_t, dx=resolution, blur=blur)
    I_rgb = np.vstack([I] * 3)

    if method == "affine":
        diffeo_start = niter + 1

    if verbose:
        print(f"[stalign] Source rasterised: {I_rgb.shape}")
        print(f"[stalign] Target image:      {J.shape}")
        print(f"[stalign] Running {'LDDMM' if method == 'lddmm' else 'affine'} for {niter} iters ...")

    # Run LDDMM ------------------------------------------------------------
    result = LDDMM(
        [YI, XI], I_rgb,
        [YJ, XJ], J,
        L=L_init, T=T_init,
        niter=niter, diffeo_start=diffeo_start,
        a=a, p=p, sigmaM=sigmaM, sigmaR=sigmaR,
        device=device, verbose=verbose,
        **lddmm_kwargs,
    )

    # Transform original source coordinates ---------------------------------
    pts_rc = np.column_stack([yI, xI])
    if L_init is not None:
        pts_rc = (L_init @ pts_rc.T).T + T_init

    aligned_rc = transform_points_source_to_target(
        result["xv"], result["v"], result["A"], pts_rc,
    ).cpu().numpy()
    aligned_xy = np.column_stack([aligned_rc[:, 1], aligned_rc[:, 0]])

    # Store results ---------------------------------------------------------
    adata = adata_source.copy() if copy else adata_source
    adata.obsm[key_added] = aligned_xy
    adata.uns["spatial_alignment"] = {
        "method": method,
        "backend": "stalign",
        "A": result["A"].cpu().numpy(),
        "v": result["v"].cpu().numpy(),
        "xv": [x.cpu().numpy() for x in result["xv"]],
        "loss_history": result["Esave"],
    }

    if verbose:
        final = result["Esave"][-1][0] if result["Esave"] else float("nan")
        print(f"[stalign] Done — final loss {final:.4f}")
        print(f"[stalign] Stored in obsm['{key_added}']")

    return adata if copy else None


# ---------------------------------------------------------------------------
# Backend: STalign / LDDMM (point-to-point via rasterisation)
# ---------------------------------------------------------------------------

def _align_points_to_points_stalign(
    adata_source: "AnnData",
    adata_target: "AnnData",
    *,
    spatial_key: str = "spatial",
    method: Literal["affine", "lddmm"] = "lddmm",
    resolution: float = 30.0,
    blur: float = 1.5,
    niter: int = 2000,
    diffeo_start: int = 100,
    a: float = 500.0,
    p: float = 2.0,
    sigmaM: float = 1.0,
    sigmaR: float = 5e5,
    initial_rotation_deg: float = 0.0,
    landmark_source: "NDArray[np.floating] | None" = None,
    landmark_target: "NDArray[np.floating] | None" = None,
    device: str = "cpu",
    verbose: bool = True,
    key_added: str = "spatial_aligned",
    copy: bool = False,
    **lddmm_kwargs: Any,
) -> "AnnData | None":
    """Align two point clouds by rasterising both and running LDDMM.

    This is the STalign fallback for point-to-point alignment when
    ott-jax is not installed.
    """
    _check_stalign_deps()

    from squidpy.experimental._lddmm import LDDMM, rasterize, transform_points_source_to_target
    from squidpy.experimental._lddmm._transforms import L_T_from_points

    coords_src = np.asarray(adata_source.obsm[spatial_key])
    coords_tgt = np.asarray(adata_target.obsm[spatial_key])
    xI, yI = coords_src[:, 0], coords_src[:, 1]
    xJ, yJ = coords_tgt[:, 0], coords_tgt[:, 1]

    # Initial affine -------------------------------------------------------
    L_init, T_init = None, None
    if landmark_source is not None and landmark_target is not None:
        pts_I_rc = np.column_stack([landmark_source[:, 1], landmark_source[:, 0]])
        pts_J_rc = np.column_stack([landmark_target[:, 1], landmark_target[:, 0]])
        L_init, T_init = L_T_from_points(pts_I_rc, pts_J_rc)
    elif initial_rotation_deg != 0.0:
        from squidpy.experimental._lddmm._transforms import compute_initial_affine
        L_init, T_init = compute_initial_affine(xI, yI, xJ, yJ, initial_rotation_deg)

    # Pre-transform source for rasterisation --------------------------------
    if L_init is not None:
        rc = np.column_stack([yI, xI])
        rc_t = (L_init @ rc.T).T + T_init
        yI_t, xI_t = rc_t[:, 0], rc_t[:, 1]
        L_pre, T_pre = L_init, T_init
        L_lddmm, T_lddmm = None, None
    else:
        xI_t, yI_t = xI, yI
        L_pre, T_pre = None, None
        L_lddmm, T_lddmm = None, None

    # Rasterise both point clouds -------------------------------------------
    XI, YI, I = rasterize(xI_t, yI_t, dx=resolution, blur=blur)
    XJ, YJ, J = rasterize(xJ, yJ, dx=resolution, blur=blur)
    I_rgb = np.vstack([I] * 3)
    J_rgb = np.vstack([J] * 3)

    if method == "affine":
        diffeo_start = niter + 1

    if verbose:
        print(f"[stalign] Source rasterised: {I_rgb.shape}")
        print(f"[stalign] Target rasterised: {J_rgb.shape}")
        print(f"[stalign] Running {'LDDMM' if method == 'lddmm' else 'affine'} for {niter} iters ...")

    # Run LDDMM ------------------------------------------------------------
    result = LDDMM(
        [YI, XI], I_rgb,
        [YJ, XJ], J_rgb,
        L=L_lddmm, T=T_lddmm,
        niter=niter, diffeo_start=diffeo_start,
        a=a, p=p, sigmaM=sigmaM, sigmaR=sigmaR,
        device=device, verbose=verbose,
        **lddmm_kwargs,
    )

    # Transform original source coordinates ---------------------------------
    pts_rc = np.column_stack([yI, xI])
    if L_pre is not None:
        pts_rc = (L_pre @ pts_rc.T).T + T_pre

    aligned_rc = transform_points_source_to_target(
        result["xv"], result["v"], result["A"], pts_rc,
    ).cpu().numpy()
    aligned_xy = np.column_stack([aligned_rc[:, 1], aligned_rc[:, 0]])

    # Store -----------------------------------------------------------------
    adata = adata_source.copy() if copy else adata_source
    adata.obsm[key_added] = aligned_xy
    adata.uns["spatial_alignment"] = {
        "method": method,
        "backend": "stalign",
        "A": result["A"].cpu().numpy(),
        "v": result["v"].cpu().numpy(),
        "xv": [x.cpu().numpy() for x in result["xv"]],
        "loss_history": result["Esave"],
    }

    if verbose:
        final = result["Esave"][-1][0] if result["Esave"] else float("nan")
        print(f"[stalign] Done — final loss {final:.4f}")
        print(f"[stalign] Stored in obsm['{key_added}']")

    return adata if copy else None


# ---------------------------------------------------------------------------
# Backend: STalign / LDDMM  (image-to-image)
# ---------------------------------------------------------------------------

def _align_images_stalign(
    source_image: "NDArray[np.floating]",
    target_image: "NDArray[np.floating]",
    *,
    method: Literal["affine", "lddmm"] = "lddmm",
    niter: int = 2000,
    diffeo_start: int = 100,
    a: float = 500.0,
    p: float = 2.0,
    sigmaM: float = 1.0,
    sigmaR: float = 5e5,
    initial_rotation_deg: float = 0.0,
    landmark_source: "NDArray[np.floating] | None" = None,
    landmark_target: "NDArray[np.floating] | None" = None,
    device: str = "cpu",
    verbose: bool = True,
    **lddmm_kwargs: Any,
) -> dict[str, Any]:
    """Align two images using LDDMM diffeomorphic registration.

    Returns a transformation dictionary.
    """
    _check_stalign_deps()

    from squidpy.experimental._lddmm import LDDMM
    from squidpy.experimental._lddmm._transforms import L_T_from_points

    I = _prep_image(source_image)
    J = _prep_image(target_image)

    _, h_I, w_I = I.shape
    _, h_J, w_J = J.shape
    XI = np.arange(w_I, dtype=np.float64)
    YI = np.arange(h_I, dtype=np.float64)
    XJ = np.arange(w_J, dtype=np.float64)
    YJ = np.arange(h_J, dtype=np.float64)

    # Auto-adjust smoothness for small images
    extent = max(h_I, w_I, h_J, w_J)
    if a == 500.0 and extent < 200:
        a = max(5.0, extent / 4)

    # Initial affine
    L_init, T_init = None, None
    if landmark_source is not None and landmark_target is not None:
        L_init, T_init = L_T_from_points(landmark_source, landmark_target)
    elif initial_rotation_deg != 0.0:
        theta = np.radians(-initial_rotation_deg)
        L_init = np.array([[np.cos(theta), -np.sin(theta)],
                           [np.sin(theta),  np.cos(theta)]])
        center = np.array([YI.mean(), XI.mean()])
        T_init = center - L_init @ center

    if method == "affine":
        diffeo_start = niter + 1

    if verbose:
        print(f"[stalign] Source: {I.shape}  Target: {J.shape}")
        print(f"[stalign] Running {'LDDMM' if method == 'lddmm' else 'affine'} for {niter} iters ...")

    result = LDDMM(
        [YI, XI], I,
        [YJ, XJ], J,
        L=L_init, T=T_init,
        niter=niter, diffeo_start=diffeo_start,
        a=a, p=p, sigmaM=sigmaM, sigmaR=sigmaR,
        device=device, verbose=verbose,
        **lddmm_kwargs,
    )

    transform_dict: dict[str, Any] = {
        "A": result["A"].cpu().numpy(),
        "v": result["v"].cpu().numpy(),
        "xv": [x.cpu().numpy() for x in result["xv"]],
        "method": method,
        "backend": "stalign",
        "loss_history": result["Esave"],
    }

    if verbose:
        final = result["Esave"][-1][0] if result["Esave"] else float("nan")
        print(f"[stalign] Done — final loss {final:.4f}")

    return transform_dict


# ---------------------------------------------------------------------------
# SpatialData helpers
# ---------------------------------------------------------------------------

def _extract_image_from_sdata(
    sdata: "sd.SpatialData",
    image_key: str,
    scale: str | Literal["auto"] = "auto",
) -> "NDArray[np.floating]":
    """Pull a single-scale image from *sdata.images[image_key]* as ``(C, H, W)``."""
    from squidpy.experimental.im._utils import _get_element_data

    if image_key not in sdata.images:
        raise KeyError(
            f"Image '{image_key}' not found.  Available: {list(sdata.images.keys())}"
        )

    node = sdata.images[image_key]
    data = _get_element_data(node, scale, "image", image_key)
    img = np.asarray(data.values if hasattr(data, "values") else data)
    return _prep_image(img)


# ---------------------------------------------------------------------------
# apply_affine  (kept from previous version)
# ---------------------------------------------------------------------------

def apply_affine(
    image: "NDArray[np.floating]",
    affine_matrix: "NDArray[np.floating]",
    output_shape: tuple[int, int] | None = None,
) -> "NDArray[np.floating]":
    """Apply an affine transformation to an image.

    Parameters
    ----------
    image
        Input image ``(C, H, W)`` / ``(H, W)`` / ``(H, W, C)``.
    affine_matrix
        3×3 affine matrix mapping source → target coordinates.
    output_shape
        ``(H, W)`` of the output.  Defaults to input shape.

    Returns
    -------
    Transformed image in the same channel layout as *image*.
    """
    _check_stalign_deps()
    import torch

    image = np.asarray(image)
    A = np.asarray(affine_matrix)

    original_ndim = image.ndim
    hwc_format = False

    if image.ndim == 2:
        image = image[np.newaxis, :, :]
    elif image.ndim == 3 and image.shape[-1] <= 4 and image.shape[0] > 4:
        hwc_format = True
        image = np.moveaxis(image, -1, 0)

    _, h_I, w_I = image.shape
    h_J, w_J = output_shape or (h_I, w_I)

    src = torch.tensor(image, dtype=torch.float32).unsqueeze(0)

    gy, gx = torch.meshgrid(
        torch.linspace(-1, 1, h_J),
        torch.linspace(-1, 1, w_J),
        indexing="ij",
    )
    px = (gx + 1) / 2 * (w_J - 1)
    py = (gy + 1) / 2 * (h_J - 1)
    ones = torch.ones_like(px)
    grid_h = torch.stack([py, px, ones], dim=-1)

    A_inv = torch.tensor(np.linalg.inv(A), dtype=torch.float32)
    src_coords = (A_inv @ grid_h.reshape(-1, 3).T).T.reshape(h_J, w_J, 3)

    sy_norm = src_coords[..., 0] / (h_I - 1) * 2 - 1
    sx_norm = src_coords[..., 1] / (w_I - 1) * 2 - 1
    sample_grid = torch.stack([sx_norm, sy_norm], dim=-1).unsqueeze(0)

    out = torch.nn.functional.grid_sample(
        src, sample_grid, mode="bilinear", padding_mode="zeros", align_corners=True,
    )
    result = out.squeeze(0).numpy()

    if hwc_format:
        result = np.moveaxis(result, 0, -1)
    elif original_ndim == 2:
        result = result[0]
    return result


# ---------------------------------------------------------------------------
# apply_transform  (for STalign transform dicts)
# ---------------------------------------------------------------------------

def apply_transform(
    coords: "NDArray[np.floating]",
    transform: dict[str, Any],
    direction: Literal["source_to_target", "target_to_source"] = "source_to_target",
) -> "NDArray[np.floating]":
    """Apply a saved STalign transformation to new coordinates.

    Parameters
    ----------
    coords
        N×2 array in ``(x, y)`` order.
    transform
        Dictionary returned by :func:`align` (must have ``backend='stalign'``).
    direction
        ``"source_to_target"`` or ``"target_to_source"``.

    Returns
    -------
    Transformed N×2 array in ``(x, y)`` order.
    """
    _check_stalign_deps()
    import torch

    from squidpy.experimental._lddmm import (
        transform_points_source_to_target,
        transform_points_target_to_source,
    )

    A = torch.tensor(transform["A"])
    v = torch.tensor(transform["v"])
    xv = [torch.tensor(x) for x in transform["xv"]]

    coords_rc = np.column_stack([coords[:, 1], coords[:, 0]])

    if direction == "source_to_target":
        out_rc = transform_points_source_to_target(xv, v, A, coords_rc)
    else:
        out_rc = transform_points_target_to_source(xv, v, A, coords_rc)

    out_rc = out_rc.cpu().numpy()
    return np.column_stack([out_rc[:, 1], out_rc[:, 0]])


# ---------------------------------------------------------------------------
# Unified public API
# ---------------------------------------------------------------------------

def align(
    source: Any,
    target: Any,
    *,
    # Common -----------------------------------------------------------------
    spatial_key: str = "spatial",
    method: Literal["affine", "lddmm", "optimal_transport"] | None = None,
    n_bins: int | tuple[int, int] | Literal["auto"] | None = None,
    device: str = "cpu",
    verbose: bool = True,
    key_added: str = "spatial_aligned",
    copy: bool = False,
    # Backend-specific -------------------------------------------------------
    solve_kwargs: dict[str, Any] | None = None,
    align_kwargs: dict[str, Any] | None = None,
    stalign_kwargs: dict[str, Any] | None = None,
    # SpatialData ------------------------------------------------------------
    source_image_key: str | None = None,
    target_image_key: str | None = None,
    scale: str | Literal["auto"] = "auto",
) -> "AnnData | dict[str, Any] | None":
    """Align spatial transcriptomics data.

    Automatically dispatches to the right backend depending on what
    *source* and *target* are:

    ==============================  ==================  ====================
    Source -> Target                 Backend             Extra dependency
    ==============================  ==================  ====================
    AnnData -> AnnData (points)     ott-jax (Sinkhorn)  ``pip install ott-jax``
    AnnData -> image                STalign (LDDMM)     ``pip install torch``
    image   -> image                STalign (LDDMM)     ``pip install torch``
    SpatialData (image keys)        STalign (LDDMM)     ``pip install torch``
    ==============================  ==================  ====================

    Parameters
    ----------
    source
        Source data — :class:`~anndata.AnnData`, :class:`~spatialdata.SpatialData`,
        numpy image, or N×2 coordinate array.
    target
        Target data (same possible types).
    spatial_key
        ``obsm`` key for spatial coordinates (AnnData inputs).
    method
        ``None`` -- auto-select (ott-jax for point-to-point, LDDMM for images).
        ``"optimal_transport"`` -- force ott-jax Sinkhorn.
        ``"lddmm"`` — force STalign LDDMM.
        ``"affine"`` — affine-only via STalign.
    n_bins
        Spatial binning resolution for large point-to-point alignment
        (OT backend only).  Both point clouds are binned into a
        regular grid and OT is solved on the (much smaller) weighted
        centroids, then the transformation is applied to all cells.

        - ``None`` (default) -- no binning; use the full point clouds.
        - ``"auto"`` -- enable binning when the total number of cells
          exceeds 50 000 (grid resolution chosen automatically per
          side based on cell count).
        - ``int`` -- number of bins per axis, same for both sides
          (e.g. 140 -> 140x140 grid for source and target).
        - ``(int, int)`` -- ``(source_bins, target_bins)`` per axis,
          allowing finer grids for denser datasets.
    device
        PyTorch device for STalign (``"cpu"`` or ``"cuda:0"``).
    verbose
        Print progress.
    key_added
        ``obsm`` key for aligned coordinates (AnnData inputs).
    copy
        Return a modified copy of AnnData instead of in-place.
    solve_kwargs
        Keyword arguments forwarded to ``ott.solvers.linear.solve``
        (point-to-point OT backend).  Useful keys: ``epsilon``,
        ``scale_cost``, ``max_iterations``, ``threshold``,
        ``tau_a``, ``tau_b``.
    align_kwargs
        Options for the alignment step after OT is solved.
        Useful keys: ``mode`` (``'warp'`` or ``'affine'``,
        default ``'warp'``).
    stalign_kwargs
        Backend-specific options for STalign/LDDMM:

        - ``resolution`` — pixel size for rasterising coordinates (default 30).
        - ``blur`` — Gaussian σ for rasterisation (default 1.5).
        - ``niter`` — number of iterations (default 2000).
        - ``diffeo_start`` — iteration to start diffeomorphism (default 100).
        - ``a`` — smoothness lengthscale (default 500).
        - ``p`` — smoothness exponent (default 2).
        - ``sigmaM`` — matching weight (default 1).
        - ``sigmaR`` — regularisation weight (default 5e5).
        - ``initial_rotation_deg`` — initial rotation in degrees (default 0).
        - ``landmark_source`` / ``landmark_target`` — N×2 arrays ``(x, y)``.
    source_image_key
        Image key in SpatialData for the source.
    target_image_key
        Image key in SpatialData for the target.
    scale
        Scale level for SpatialData images (default ``"auto"``).

    Returns
    -------
    - AnnData source → modified AnnData (if *copy*) or ``None``
    - Image source → transformation ``dict``

    Examples
    --------
    **Point-to-point** via ott-jax (warp mode):

    >>> import squidpy as sq
    >>> sq.experimental.tl.align(adata_src, adata_tgt)

    **Point-to-point** with custom params:

    >>> sq.experimental.tl.align(
    ...     adata_src, adata_tgt,
    ...     method="optimal_transport",
    ...     solve_kwargs={'epsilon': 1e-2},
    ...     align_kwargs={'mode': 'affine'},
    ... )

    **Point-to-point** with spatial binning (fast, large data):

    >>> sq.experimental.tl.align(
    ...     adata_src, adata_tgt,
    ...     n_bins="auto",
    ... )

    **Point-to-point** via STalign (rasterise + LDDMM):

    >>> sq.experimental.tl.align(
    ...     adata_src, adata_tgt,
    ...     method="lddmm",
    ...     stalign_kwargs={'niter': 3000, 'a': 200},
    ... )

    **Point-to-image** via STalign:

    >>> sq.experimental.tl.align(adata, histology_image)

    **Image-to-image** via STalign:

    >>> transform = sq.experimental.tl.align(img_src, img_tgt)

    Notes
    -----
    **Choosing a backend.**  The OT and LDDMM backends solve different
    problems and each has a natural domain of applicability.

    *Optimal transport (ott-jax)* works well when:

    - Source and target are in **compatible coordinate systems**
      (similar spatial extent and units).  If they differ (e.g.
      mm vs microns), rescale one before calling ``align()``.
    - Tissue morphology is similar enough that spatial proximity
      is meaningful.  Works across technologies (e.g. MERFISH to
      Xenium) as long as coordinates are compatible.
    - Cell counts are of similar magnitude.  Highly unbalanced
      counts can make balanced OT degenerate; use ``n_bins`` to
      mitigate.

    OT alignment gives misleading results when:

    - The two datasets come from unrelated tissues.  OT always
      returns *some* transport plan -- it never reports "these
      datasets cannot be aligned."
    - Coordinate scales differ by orders of magnitude and have
      not been rescaled.  Our Sinkhorn backend uses raw
      coordinates, so the cost is dominated by the offset.

    *LDDMM / STalign* is preferable when:

    - Aligning points to a **reference image** (e.g. H&E
      histology, atlas image).
    - Tissue sections contain significant **non-linear
      distortions** (tears, folds).
    - Coordinate systems are incompatible and cannot easily be
      pre-registered.
    - You need a **smooth diffeomorphic transformation** rather
      than a transport plan.

    See [3]_ and [4]_ for benchmarking of alignment methods.

    References
    ----------
    .. [1] Clifton *et al.*, "STalign: Alignment of spatial
       transcriptomics data using diffeomorphic metric mapping",
       *Nat. Commun.* 14, 8123 (2023).
    .. [2] Cuturi *et al.*, "Optimal Transport Tools (OTT): A JAX
       Toolbox for all things Wasserstein" (2022).
    .. [3] Zeira *et al.*, "Alignment and integration of spatial
       transcriptomics data", *Nat. Methods* 19, 567--575 (2022).
    .. [4] Li *et al.*, "Benchmarking clustering, alignment, and
       integration methods for spatial transcriptomics",
       *Genome Biol.* 25, 212 (2024).
    """
    solve_kw: dict[str, Any] = dict(solve_kwargs or {})
    align_kw: dict[str, Any] = dict(align_kwargs or {})
    stalign_kw: dict[str, Any] = dict(stalign_kwargs or {})

    # Inject top-level common params into stalign_kw (all stalign funcs
    # accept these).  User overrides in stalign_kwargs take precedence.
    stalign_kw.setdefault("device", device)
    stalign_kw.setdefault("verbose", verbose)

    src_type = _detect_input_type(source)
    tgt_type = _detect_input_type(target)

    # ----- SpatialData shortcut (image keys) -------------------------------
    if src_type == "spatialdata":
        if source_image_key and target_image_key:
            src_img = _extract_image_from_sdata(source, source_image_key, scale)
            tgt_img = _extract_image_from_sdata(source, target_image_key, scale)
            stalign_kw.setdefault("method", method or "lddmm")
            return _align_images_stalign(src_img, tgt_img, **stalign_kw)

    # ----- Point-to-point -------------------------------------------------
    if src_type in ("anndata", "coords") and tgt_type in ("anndata", "coords"):
        # Wrap raw coords in AnnData
        if src_type == "coords":
            source = _coords_to_adata(np.asarray(source), spatial_key)
        if tgt_type == "coords":
            target = _coords_to_adata(np.asarray(target), spatial_key)

        # Auto-select backend
        use_ot = False
        if method == "optimal_transport":
            use_ot = True
        elif method is None:
            if _has_ott():
                use_ot = True
            elif _has_torch():
                use_ot = False
            else:
                raise ImportError(
                    "Point-to-point alignment requires either ott-jax or torch.\n"
                    "Install with:  pip install ott-jax   OR   pip install torch"
                )

        if use_ot:
            _AUTO_BIN_THRESHOLD = 50_000
            use_binning = False
            _n_bins = n_bins
            if _n_bins == "auto":
                n_total = source.n_obs + target.n_obs
                if n_total > _AUTO_BIN_THRESHOLD:
                    use_binning = True
                    _n_bins = None  # per-side auto
                    if verbose:
                        print(
                            f"[align] Auto-binning enabled: {n_total} cells "
                            f"> {_AUTO_BIN_THRESHOLD} threshold"
                        )
            elif _n_bins is not None:
                use_binning = True

            if use_binning:
                return _align_points_ot_binned(
                    source, target,
                    spatial_key=spatial_key,
                    key_added=key_added,
                    copy=copy,
                    verbose=verbose,
                    n_bins=_n_bins,
                    solve_kwargs=solve_kw,
                    align_kwargs=align_kw,
                )

            return _align_points_ot(
                source, target,
                spatial_key=spatial_key,
                key_added=key_added,
                copy=copy,
                verbose=verbose,
                solve_kwargs=solve_kw,
                align_kwargs=align_kw,
            )
        else:
            stalign_kw.setdefault("method", method or "lddmm")
            stalign_kw.setdefault("spatial_key", spatial_key)
            stalign_kw.setdefault("key_added", key_added)
            stalign_kw.setdefault("copy", copy)
            if verbose:
                print("[align] Using STalign (rasterise) for point-to-point alignment")
            return _align_points_to_points_stalign(source, target, **stalign_kw)

    # ----- Point-to-image -------------------------------------------------
    if src_type in ("anndata", "coords") and tgt_type == "image":
        if src_type == "coords":
            source = _coords_to_adata(np.asarray(source), spatial_key)
        stalign_kw.setdefault("method", method if method in ("affine", "lddmm") else "lddmm")
        stalign_kw.setdefault("spatial_key", spatial_key)
        stalign_kw.setdefault("key_added", key_added)
        stalign_kw.setdefault("copy", copy)
        return _align_points_to_image_stalign(source, np.asarray(target), **stalign_kw)

    # ----- Image-to-image -------------------------------------------------
    if src_type == "image" and tgt_type == "image":
        stalign_kw.setdefault("method", method if method in ("affine", "lddmm") else "lddmm")
        return _align_images_stalign(np.asarray(source), np.asarray(target), **stalign_kw)

    raise TypeError(
        f"Unsupported alignment combination: {src_type} -> {tgt_type}.\n"
        f"Supported: points->points, points->image, image->image."
    )


# ---------------------------------------------------------------------------
# Diagnostic plots for binned OT alignment
# ---------------------------------------------------------------------------

def _get_diag(adata_source: "AnnData") -> dict:
    """Retrieve spatial_alignment_diag from adata, raising if absent."""
    diag = adata_source.uns.get("spatial_alignment_diag")
    if diag is None:
        raise ValueError(
            "No diagnostic data found. Re-run align() with the latest code "
            "so that adata.uns['spatial_alignment_diag'] is populated."
        )
    return diag


def _build_rbf_field(
    diag: dict,
    coords_src: "NDArray[np.floating]",
    grid_res: int = 40,
) -> tuple["NDArray", "NDArray", "NDArray", "NDArray"]:
    """Evaluate the RBF displacement field on a regular grid.

    Returns ``(X, Y, U, V)`` arrays for quiver / streamline plots.
    """
    from scipy.interpolate import RBFInterpolator

    cen_src = diag["cen_src"]
    warped = diag["warped_centroids"]
    alive = diag["alive"]

    displacements = (warped - cen_src)[alive]

    interp = RBFInterpolator(
        cen_src[alive], displacements, kernel="thin_plate_spline",
    )

    pts_alive = cen_src[alive]
    xmin, ymin = pts_alive.min(axis=0)
    xmax, ymax = pts_alive.max(axis=0)
    buf = 0.05
    rx = (xmax - xmin) * buf
    ry = (ymax - ymin) * buf
    xs = np.linspace(xmin - rx, xmax + rx, grid_res)
    ys = np.linspace(ymin - ry, ymax + ry, grid_res)
    X, Y = np.meshgrid(xs, ys)
    pts = np.column_stack([X.ravel(), Y.ravel()])

    D = interp(pts)
    U = D[:, 0].reshape(X.shape)
    V = D[:, 1].reshape(X.shape)

    return X, Y, U, V


def plot_ot_alignment(
    adata_source: "AnnData",
    adata_target: "AnnData",
    *,
    spatial_key: str = "spatial",
    key_aligned: str = "spatial_aligned",
    panels: tuple[str, ...] | list[str] | None = None,
    figsize: tuple[float, float] | None = None,
    arrow_subsample: int | None = None,
    arrow_scale: float = 1.0,
    arrow_width: float = 0.002,
    s_points: float = 0.3,
    s_bins: float = 8,
    alpha_points: float = 0.15,
    grid_res: int = 40,
    stream_density: float = 1.5,
    stream_linewidth: float | None = None,
    cmap: str = "coolwarm",
    matching_top_k: int = 3,
    show: bool = True,
) -> Any:
    """Diagnostic visualisation for binned OT alignment.

    By default all six panels are shown.  Pass ``panels`` to select a
    subset or reorder them.

    Available panels
    ----------------
    ``"bins"``
        **Bin / sample layout.**  Source points (blue) and target
        points (orange) shown in separate subpanels.  Marker size is
        proportional to bin mass (or uniform for subsampling).
    ``"grid"``
        **Grid-level OT transport plan.**  Source bin centroids (blue)
        with arrows to their OT-warped positions, overlaid on target
        centroids (orange).  Arrow colour encodes displacement
        magnitude.  Centroid size encodes bin mass.
    ``"matching"``
        **OT matching candidates.**  For each alive source bin, lines
        are drawn to its top-k target bins (by transport weight).
        Line opacity/width encodes coupling strength.  Useful for
        inspecting whether the OT plan makes spatial sense.
    ``"quiver"``
        **Displacement vector field on a regular grid.**  The RBF
        interpolant learned from the bin-level OT displacements is
        evaluated on a ``grid_res x grid_res`` lattice and displayed
        as a quiver plot, analogous to RNA velocity fields.
    ``"stream"``
        **Streamlines of the displacement field.**  Same RBF
        interpolant as ``"quiver"`` but rendered as streamlines with
        line width proportional to displacement magnitude.
    ``"magnitude"``
        **Displacement magnitude heatmap.**  Filled contour of the
        RBF displacement magnitude over the source domain.
    ``"overlay"``
        **Final overlay.**  Aligned source cells (blue) on top of
        target cells (orange).

    Parameters
    ----------
    adata_source
        Source AnnData *after* calling :func:`align` (must contain
        ``uns['spatial_alignment_diag']`` and ``obsm[key_aligned]``).
    adata_target
        Target AnnData.
    panels
        Which panels to show (default: all six).  Accepts any
        combination of ``"grid"``, ``"matching"``, ``"quiver"``,
        ``"stream"``, ``"magnitude"``, ``"overlay"``.
    figsize
        Overall figure size.  Defaults to ``(6*n_panels, 5)``.
    arrow_subsample
        Show every *n*-th arrow to reduce clutter (``"grid"`` panel).
        ``None`` shows all alive bins.
    arrow_scale, arrow_width
        Passed to :func:`matplotlib.pyplot.quiver` for the
        ``"grid"`` and ``"quiver"`` panels.
    s_points
        Marker size for individual cells.
    s_bins
        Base marker size for bin centroids (scaled by mass in the
        ``"grid"`` panel).
    alpha_points
        Alpha for individual cell scatter.
    grid_res
        Number of grid points per axis for ``"quiver"``,
        ``"stream"``, and ``"magnitude"`` panels.
    stream_density
        Streamline density for the ``"stream"`` panel.
    stream_linewidth
        If ``None`` (default), line width scales with displacement
        magnitude.  Otherwise a fixed width.
    cmap
        Colour map for displacement magnitude (used in ``"grid"``,
        ``"quiver"``, ``"magnitude"``).
    matching_top_k
        Number of top target matches to draw per source bin in the
        ``"matching"`` panel.  Default 3.
    show
        Call ``plt.show()`` at the end.  Set ``False`` for further
        customisation.

    Returns
    -------
    ``matplotlib.figure.Figure``
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    ALL_PANELS = ("quiver", "stream", "bins", "overlay")
    if panels is None:
        panels = ALL_PANELS
    else:
        panels = tuple(panels)
    unknown = set(panels) - set(ALL_PANELS)
    if unknown:
        raise ValueError(
            f"Unknown panel(s): {unknown}. Choose from {ALL_PANELS}."
        )
    n = len(panels)
    if n == 0:
        raise ValueError("At least one panel must be specified.")

    diag = _get_diag(adata_source)
    cen_src = diag["cen_src"]
    cen_tgt = diag["cen_tgt"]
    warped = diag["warped_centroids"]
    alive = diag["alive"]
    w_src = diag["w_src"]
    w_tgt = diag["w_tgt"]
    T_norm = diag.get("T_norm")

    coords_src_raw = np.asarray(adata_source.obsm[spatial_key])
    coords_tgt_raw = np.asarray(adata_target.obsm[spatial_key])
    coords_aligned = np.asarray(adata_source.obsm[key_aligned])

    src_mean = diag.get("src_mean", coords_src_raw.mean(axis=0))
    src_c = coords_src_raw - src_mean

    disp_bin = warped - cen_src
    disp_bin_mag = np.linalg.norm(disp_bin, axis=1)

    need_field = bool({"quiver", "stream", "magnitude"} & set(panels))
    if need_field:
        X, Y, U, V = _build_rbf_field(
            diag, src_c, grid_res=grid_res,
        )
        mag_grid = np.sqrt(U**2 + V**2)

    # "bins" panel gets 2 columns; everything else gets 1.
    col_widths = [2 if p == "bins" else 1 for p in panels]
    total_cols = sum(col_widths)
    if figsize is None:
        figsize = (6 * total_cols, 5)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, total_cols)
    axes = []
    col = 0
    for w in col_widths:
        axes.append(fig.add_subplot(gs[0, col:col + w]))
        col += w

    alive_idx = np.where(alive)[0]
    if arrow_subsample is not None and arrow_subsample > 1:
        sub_idx = alive_idx[::arrow_subsample]
    else:
        sub_idx = alive_idx

    norm = Normalize(vmin=disp_bin_mag[alive].min(),
                     vmax=disp_bin_mag[alive].max())

    for ax, panel in zip(axes, panels, strict=True):

        if panel == "bins":
            mass_src = w_src / w_src.max() * s_bins * 5
            mass_tgt = w_tgt / w_tgt.max() * s_bins * 5

            pos = ax.get_position()
            ax.set_visible(False)
            mid = (pos.x0 + pos.x1) / 2
            gap = (pos.x1 - pos.x0) * 0.04
            ax_s = fig.add_axes([pos.x0, pos.y0,
                                 mid - pos.x0 - gap, pos.y1 - pos.y0])
            ax_t = fig.add_axes([mid + gap, pos.y0,
                                 pos.x1 - mid - gap, pos.y1 - pos.y0])

            ax_s.scatter(
                src_c[:, 0], src_c[:, 1],
                s=s_points * 0.3, c="grey", alpha=0.05,
            )
            ax_s.scatter(
                cen_src[:, 0], cen_src[:, 1],
                s=mass_src, c="steelblue", alpha=0.5,
            )
            ax_s.set_aspect("equal")
            ax_s.set_title(f"Source bins ({len(cen_src)})")

            ax_t.scatter(
                cen_tgt[:, 0], cen_tgt[:, 1],
                s=mass_tgt, c="orange", alpha=0.5,
            )
            ax_t.set_aspect("equal")
            ax_t.set_title(f"Target bins ({len(cen_tgt)})")
            continue

        elif panel == "grid":
            mass_scale = w_src / w_src.max() * s_bins * 5
            ax.scatter(
                cen_tgt[:, 0], cen_tgt[:, 1],
                s=s_bins, c="orange", alpha=0.35, label="target bins",
            )
            ax.scatter(
                cen_src[alive, 0], cen_src[alive, 1],
                s=mass_scale[alive], c="steelblue", alpha=0.4,
                label="source bins",
            )
            q = ax.quiver(
                cen_src[sub_idx, 0], cen_src[sub_idx, 1],
                disp_bin[sub_idx, 0], disp_bin[sub_idx, 1],
                disp_bin_mag[sub_idx],
                angles="xy", scale_units="xy", scale=arrow_scale,
                width=arrow_width, cmap=cmap, norm=norm, alpha=0.8,
            )
            plt.colorbar(q, ax=ax, label="displacement", shrink=0.7)
            ax.legend(markerscale=3, fontsize=7, loc="best")
            ax.set_title("Grid-level OT transport")

        elif panel == "matching":
            if T_norm is None:
                ax.text(0.5, 0.5, "T_norm not stored.\nRe-run align().",
                        ha="center", va="center", transform=ax.transAxes)
                ax.set_title("OT matching (no data)")
            else:
                from matplotlib.collections import LineCollection
                lines = []
                weights = []
                for i in alive_idx[::max(1, len(alive_idx) // 200)]:
                    row = T_norm[i]
                    top_j = np.argsort(row)[-matching_top_k:]
                    for j in top_j:
                        if row[j] > 1e-6:
                            lines.append([cen_src[i], cen_tgt[j]])
                            weights.append(float(row[j]))
                if lines:
                    weights = np.array(weights)
                    weights = weights / (weights.max() + 1e-12)
                    lc = LineCollection(
                        lines, linewidths=0.3 + 1.5 * weights,
                        colors=plt.cm.viridis(weights), alpha=0.6,
                    )
                    ax.add_collection(lc)
                ax.scatter(
                    cen_src[alive, 0], cen_src[alive, 1],
                    s=s_bins, c="steelblue", alpha=0.5, label="source",
                )
                ax.scatter(
                    cen_tgt[:, 0], cen_tgt[:, 1],
                    s=s_bins, c="orange", alpha=0.5, label="target",
                )
                ax.autoscale_view()
                ax.legend(markerscale=3, fontsize=7, loc="best")
                ax.set_title(f"OT matching (top-{matching_top_k})")

        elif panel == "quiver":
            Xf, Yf = X.ravel(), Y.ravel()
            Uf, Vf = U.ravel(), V.ravel()
            Mf = np.sqrt(Uf**2 + Vf**2)
            q = ax.quiver(
                Xf, Yf, Uf, Vf, Mf,
                angles="xy", scale_units="xy", scale=arrow_scale,
                width=arrow_width * 1.5, cmap=cmap, alpha=0.85,
            )
            ax.scatter(
                src_c[:, 0], src_c[:, 1],
                s=s_points * 0.3, c="grey", alpha=0.08,
            )
            plt.colorbar(q, ax=ax, label="displacement", shrink=0.7)
            ax.set_title("Displacement field (quiver)")

        elif panel == "stream":
            speed = mag_grid.copy()
            if stream_linewidth is None:
                maxs = speed.max() + 1e-12
                lw = 0.5 + 2.5 * speed / maxs
            else:
                lw = stream_linewidth
            strm = ax.streamplot(
                X[0, :], Y[:, 0], U, V,
                color=speed, cmap=cmap, density=stream_density,
                linewidth=lw, arrowsize=1.2,
            )
            ax.scatter(
                src_c[:, 0], src_c[:, 1],
                s=s_points * 0.3, c="grey", alpha=0.08,
            )
            plt.colorbar(strm.lines, ax=ax, label="displacement", shrink=0.7)
            ax.set_title("Displacement streamlines")

        elif panel == "magnitude":
            vmin = mag_grid.min()
            vmax = mag_grid.max()
            levels = np.linspace(vmin, vmax, 30)
            cf = ax.contourf(X, Y, mag_grid, levels=levels, cmap=cmap,
                             alpha=0.85, extend="both")
            ax.scatter(
                src_c[:, 0], src_c[:, 1],
                s=s_points * 0.3, c="k", alpha=0.06,
            )
            plt.colorbar(cf, ax=ax, label="displacement magnitude",
                         shrink=0.7)
            ax.set_title("Displacement magnitude")

        elif panel == "overlay":
            ax.scatter(
                coords_tgt_raw[:, 0], coords_tgt_raw[:, 1],
                s=s_points, c="orange", alpha=alpha_points,
                label="target",
            )
            ax.scatter(
                coords_aligned[:, 0], coords_aligned[:, 1],
                s=s_points, c="steelblue", alpha=alpha_points,
                label="aligned source",
            )
            ax.legend(markerscale=5, fontsize=7, loc="best")
            ax.set_title("Final overlay")

        ax.set_aspect("equal")

    fig.tight_layout()
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# Alignment quality scoring
# ---------------------------------------------------------------------------

def score_alignment(
    adata_source: "AnnData",
    adata_target: "AnnData",
    *,
    spatial_key_source: str = "spatial_aligned",
    spatial_key_target: str = "spatial",
    genes: list[str] | None = None,
    k: int = 10,
    radius: float | Literal["auto"] = "auto",
    verbose: bool = True,
) -> dict[str, Any]:
    """Score the quality of a spatial alignment.

    Given aligned source coordinates and target coordinates, compute a
    suite of metrics that quantify how well the two datasets overlap
    spatially and transcriptomically.

    Parameters
    ----------
    adata_source
        Source AnnData *after* alignment (must contain aligned
        coordinates in ``obsm[spatial_key_source]``).
    adata_target
        Target AnnData with spatial coordinates in
        ``obsm[spatial_key_target]``.
    spatial_key_source
        ``obsm`` key for aligned source coordinates.
    spatial_key_target
        ``obsm`` key for target coordinates.
    genes
        Genes (column names in ``.var_names``) to use for gene
        expression metrics.  ``None`` auto-detects the intersection
        of source and target ``var_names``.  If there are no shared
        genes the expression-based metrics are skipped.
    k
        Number of nearest neighbors for kNN-based metrics.
    radius
        Distance threshold for coverage.  Each target cell is
        "covered" if at least one aligned source cell falls within
        *radius*.

        - ``"auto"`` (default) -- set to the median nearest-neighbor
          distance among target cells (a natural length-scale of the
          target tissue).
        - ``float`` -- explicit distance in coordinate units.
    verbose
        Print a summary table.

    Returns
    -------
    A dictionary with the following keys (not all keys are always
    present; expression metrics require shared genes):

    - **coverage** (``float``, 0--1): fraction of target cells that
      have at least one aligned source neighbor within *radius*.
    - **mean_nn_dist** (``float``): mean nearest-neighbor distance
      from each target cell to its closest aligned source cell.
    - **median_nn_dist** (``float``): median of the same.
    - **expr_knn_corr** (``float``, -1--1): for each aligned source
      cell, find its *k* nearest target cells; average the target
      expression over those neighbours; then compute the Pearson
      correlation with the source cell's own expression.  Reported
      as the median across source cells.  Expression is
      library-size-normalized and log1p-transformed before
      comparison.
    - **expr_knn_cosine** (``float``, 0--1): same procedure as above
      but using cosine similarity instead of Pearson correlation.
    - **radius_used** (``float``): the *radius* that was used
      (useful when ``radius="auto"``).
    - **n_source** / **n_target** (``int``): cell counts.
    - **n_genes_used** (``int``): number of shared genes used for
      expression metrics.

    Notes
    -----
    The **coverage** metric is inspired by the point-cloud
    registration literature where "inlier ratio" measures the
    fraction of points that find a match within a tolerance
    [Pomerleau *et al.*, 2015].

    Examples
    --------
    >>> scores = sq.experimental.tl.score_alignment(
    ...     adata_source, adata_target,
    ... )
    >>> scores["coverage"]
    0.94
    """
    from scipy.spatial import KDTree
    from scipy.sparse import issparse

    coords_src = np.asarray(adata_source.obsm[spatial_key_source], dtype=np.float64)
    coords_tgt = np.asarray(adata_target.obsm[spatial_key_target], dtype=np.float64)

    n_src = len(coords_src)
    n_tgt = len(coords_tgt)

    tree_src = KDTree(coords_src)
    tree_tgt = KDTree(coords_tgt)

    result: dict[str, Any] = {"n_source": n_src, "n_target": n_tgt}

    # -- 1. Nearest-neighbor distances (target -> source) ---------------------
    nn_dists, _ = tree_src.query(coords_tgt, k=1)
    result["mean_nn_dist"] = float(np.mean(nn_dists))
    result["median_nn_dist"] = float(np.median(nn_dists))

    # -- 2. Coverage ----------------------------------------------------------
    if radius == "auto":
        tgt_self_dists, _ = tree_tgt.query(coords_tgt, k=2)
        radius_val = float(np.median(tgt_self_dists[:, 1]))
    else:
        radius_val = float(radius)
    result["radius_used"] = radius_val

    covered = nn_dists <= radius_val
    result["coverage"] = float(covered.mean())

    # -- 3. Gene-expression-based metrics -------------------------------------
    if genes is not None:
        shared_genes = [g for g in genes if g in adata_source.var_names and g in adata_target.var_names]
    else:
        shared_genes = sorted(set(adata_source.var_names) & set(adata_target.var_names))

    result["n_genes_used"] = len(shared_genes)

    if len(shared_genes) > 1:
        X_src = adata_source[:, shared_genes].X
        X_tgt = adata_target[:, shared_genes].X
        if issparse(X_src):
            X_src = X_src.toarray()
        if issparse(X_tgt):
            X_tgt = X_tgt.toarray()
        X_src = np.asarray(X_src, dtype=np.float64)
        X_tgt = np.asarray(X_tgt, dtype=np.float64)

        # Library-size normalize + log1p so correlation/cosine are
        # not dominated by a few highly expressed genes.
        def _lognorm(X: np.ndarray) -> np.ndarray:
            totals = X.sum(axis=1, keepdims=True)
            totals = np.where(totals > 0, totals, 1.0)
            median_total = np.median(totals[totals > 1.0]) if (totals > 1.0).any() else 1.0
            return np.log1p(X / totals * median_total)

        X_src = _lognorm(X_src)
        X_tgt = _lognorm(X_tgt)

        _, nn_tgt_idx = tree_tgt.query(coords_src, k=k)
        avg_tgt_expr = np.zeros_like(X_src)
        for i in range(n_src):
            avg_tgt_expr[i] = X_tgt[nn_tgt_idx[i]].mean(axis=0)

        # Pearson correlation per source cell
        def _rowwise_pearson(A: np.ndarray, B: np.ndarray) -> np.ndarray:
            A_c = A - A.mean(axis=1, keepdims=True)
            B_c = B - B.mean(axis=1, keepdims=True)
            num = (A_c * B_c).sum(axis=1)
            denom = np.sqrt((A_c ** 2).sum(axis=1) * (B_c ** 2).sum(axis=1))
            denom = np.where(denom > 1e-12, denom, 1.0)
            return num / denom

        pearson = _rowwise_pearson(X_src, avg_tgt_expr)
        result["expr_knn_corr"] = float(np.nanmedian(pearson))

        # Cosine similarity per source cell
        def _rowwise_cosine(A: np.ndarray, B: np.ndarray) -> np.ndarray:
            num = (A * B).sum(axis=1)
            denom = np.sqrt((A ** 2).sum(axis=1) * (B ** 2).sum(axis=1))
            denom = np.where(denom > 1e-12, denom, 1.0)
            return num / denom

        cosine = _rowwise_cosine(X_src, avg_tgt_expr)
        result["expr_knn_cosine"] = float(np.nanmedian(cosine))

    # -- Summary --------------------------------------------------------------
    if verbose:
        print("[score_alignment] Alignment quality metrics:")
        print(f"  n_source:            {n_src}")
        print(f"  n_target:            {n_tgt}")
        print(f"  radius_used:         {result['radius_used']:.4f}")
        print(f"  coverage:            {result['coverage']:.4f}")
        print(f"  mean_nn_dist:        {result['mean_nn_dist']:.4f}")
        print(f"  median_nn_dist:      {result['median_nn_dist']:.4f}")
        if "expr_knn_corr" in result:
            print(f"  expr_knn_corr:       {result['expr_knn_corr']:.4f}")
            print(f"  expr_knn_cosine:     {result['expr_knn_cosine']:.4f}")
        else:
            g = result["n_genes_used"]
            print(f"  (expression metrics skipped: {g} shared genes)")

    return result
