"""High-level spatial alignment functions.

This module provides the main entry points for aligning spatial transcriptomics
data using LDDMM (Large Deformation Diffeomorphic Metric Mapping).
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    import spatialdata as sd
    from anndata import AnnData
    from numpy.typing import NDArray


def _check_torch() -> None:
    """Check if PyTorch is available."""
    try:
        import torch  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Spatial alignment requires PyTorch. " "Install with: pip install torch"
        ) from e


def _get_torch():
    """Import and return torch module."""
    _check_torch()
    import torch

    return torch


def rasterize_coordinates(
    coords: NDArray[np.floating],
    resolution: float = 30.0,
    blur: float = 1.5,
    expand: float = 1.1,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Rasterize point coordinates into a density image.

    Parameters
    ----------
    coords
        Nx2 array of (x, y) coordinates.
    resolution
        Pixel size in coordinate units. Default: 30.0.
    blur
        Gaussian kernel sigma in pixels. Default: 1.5.
    expand
        Factor to expand image bounds. Default: 1.1.

    Returns
    -------
    A tuple containing:

    - X: 1D array of x pixel locations.
    - Y: 1D array of y pixel locations.
    - image: Density image with shape (1, H, W).

    Examples
    --------
    >>> import squidpy as sq
    >>> coords = np.random.rand(1000, 2) * 1000
    >>> X, Y, image = sq.experimental.rasterize_coordinates(coords, resolution=30.0)
    """
    from squidpy.experimental._lddmm import rasterize

    x, y = coords[:, 0], coords[:, 1]
    X, Y, image = rasterize(x, y, dx=resolution, blur=blur, expand=expand)
    return X, Y, image


def apply_transform(
    coords: NDArray[np.floating],
    transform: dict[str, Any],
    direction: Literal["source_to_target", "target_to_source"] = "source_to_target",
) -> NDArray[np.floating]:
    """Apply a transformation to coordinates.

    Parameters
    ----------
    coords
        Nx2 array of coordinates in (x, y) order.
    transform
        Transformation dict from align_spatial (contains 'A', 'v', 'xv').
    direction
        Direction of transformation. Default: 'source_to_target'.

    Returns
    -------
    Transformed coordinates as Nx2 array in (x, y) order.

    Examples
    --------
    >>> import squidpy as sq
    >>> # After running align_spatial
    >>> new_coords = np.random.rand(50, 2) * 1000
    >>> transformed = sq.experimental.apply_transform(
    ...     new_coords,
    ...     adata_source.uns['spatial_alignment'],
    ...     direction='source_to_target'
    ... )
    """
    _check_torch()
    torch = _get_torch()

    from squidpy.experimental._lddmm import (
        transform_points_source_to_target,
        transform_points_target_to_source,
    )

    # Convert to torch
    A = torch.tensor(transform["A"])
    v = torch.tensor(transform["v"])
    xv = [torch.tensor(x) for x in transform["xv"]]

    # Convert coordinates to (row, col) = (y, x) for STalign convention
    coords_rc = np.column_stack([coords[:, 1], coords[:, 0]])

    if direction == "source_to_target":
        transformed_rc = transform_points_source_to_target(xv, v, A, coords_rc)
    else:
        transformed_rc = transform_points_target_to_source(xv, v, A, coords_rc)

    # Convert back to numpy and (x, y) order
    transformed_rc = transformed_rc.cpu().numpy()
    return np.column_stack([transformed_rc[:, 1], transformed_rc[:, 0]])


def align_spatial(
    adata_source: "AnnData",
    adata_target: "AnnData",
    *,
    spatial_key: str = "spatial",
    method: Literal["affine", "lddmm"] = "lddmm",
    # Rasterization parameters
    resolution: float = 30.0,
    blur: float = 1.5,
    # LDDMM parameters
    niter: int = 2000,
    diffeo_start: int = 100,
    a: float = 500.0,
    p: float = 2.0,
    sigmaM: float = 1.0,
    sigmaR: float = 5e5,
    # Initial transform (optional)
    initial_rotation_deg: float = 0.0,
    landmark_points_source: NDArray[np.floating] | None = None,
    landmark_points_target: NDArray[np.floating] | None = None,
    # Computation
    device: str = "cpu",
    verbose: bool = True,
    # Output
    key_added: str = "spatial_aligned",
    copy: bool = False,
) -> "AnnData | None":
    """Align spatial coordinates from source AnnData to target AnnData.

    This function performs diffeomorphic registration (LDDMM) to align
    spatial transcriptomics data. It rasterizes cell coordinates into
    density images, computes a smooth nonlinear transformation, and
    applies it to the original coordinates.

    Parameters
    ----------
    adata_source
        Source AnnData with spatial coordinates in ``obsm[spatial_key]``.
    adata_target
        Target AnnData to align to.
    spatial_key
        Key in ``obsm`` containing spatial coordinates. Default: 'spatial'.
    method
        Alignment method. 'lddmm' for full diffeomorphic, 'affine' for
        affine-only. Default: 'lddmm'.
    resolution
        Pixel size for rasterization in coordinate units. Default: 30.0.
    blur
        Gaussian blur sigma in pixels for rasterization. Default: 1.5.
    niter
        Number of optimization iterations. Default: 2000.
    diffeo_start
        Iteration to start nonlinear deformation (affine-only before).
        Default: 100.
    a
        Smoothness scale of velocity field. Larger = smoother. Default: 500.0.
    p
        Power of Laplacian regularization. Default: 2.0.
    sigmaM
        Image matching weight. Smaller = more accurate. Default: 1.0.
    sigmaR
        Regularization weight. Smaller = smoother. Default: 5e5.
    initial_rotation_deg
        Initial rotation in degrees (clockwise). Default: 0.0.
    landmark_points_source
        Optional Nx2 array of landmark points in source (x, y order).
    landmark_points_target
        Optional Nx2 array of corresponding landmark points in target.
    device
        PyTorch device ('cpu' or 'cuda:0'). Default: 'cpu'.
    verbose
        Print progress. Default: True.
    key_added
        Key for aligned coordinates in ``obsm``. Default: 'spatial_aligned'.
    copy
        Return a copy. Default: False (modify in place).

    Returns
    -------
    If ``copy=True``, returns modified AnnData. Otherwise modifies in place
    and stores:

    - ``adata_source.obsm[key_added]``: aligned coordinates
    - ``adata_source.uns['spatial_alignment']``: transformation dict

    Examples
    --------
    >>> import squidpy as sq
    >>> # Simple alignment
    >>> sq.experimental.align_spatial(adata_source, adata_target)
    >>> # Access aligned coordinates
    >>> aligned_coords = adata_source.obsm['spatial_aligned']

    >>> # With initial rotation and landmarks
    >>> sq.experimental.align_spatial(
    ...     adata_source, adata_target,
    ...     initial_rotation_deg=45,
    ...     landmark_points_source=source_pts,
    ...     landmark_points_target=target_pts
    ... )

    Notes
    -----
    This function is based on STalign (https://github.com/JEFworks-Lab/STalign).
    The algorithm uses Large Deformation Diffeomorphic Metric Mapping (LDDMM)
    to compute a smooth, invertible transformation between coordinate systems.

    For large datasets (>100k cells), consider:

    - Increasing ``resolution`` to reduce image size
    - Using ``device='cuda:0'`` if GPU available
    """
    _check_torch()
    torch = _get_torch()

    from squidpy.experimental._lddmm import (
        LDDMM,
        rasterize,
        transform_points_source_to_target,
    )
    from squidpy.experimental._lddmm._transforms import (
        L_T_from_points,
        compute_initial_affine,
    )

    # Validate inputs
    if spatial_key not in adata_source.obsm:
        raise KeyError(
            f"Spatial key '{spatial_key}' not found in adata_source.obsm. "
            f"Available keys: {list(adata_source.obsm.keys())}"
        )
    if spatial_key not in adata_target.obsm:
        raise KeyError(
            f"Spatial key '{spatial_key}' not found in adata_target.obsm. "
            f"Available keys: {list(adata_target.obsm.keys())}"
        )

    coords_source = adata_source.obsm[spatial_key]
    coords_target = adata_target.obsm[spatial_key]

    if coords_source.shape[1] != 2:
        raise ValueError(f"Expected 2D coordinates, got shape {coords_source.shape}")
    if coords_target.shape[1] != 2:
        raise ValueError(f"Expected 2D coordinates, got shape {coords_target.shape}")

    if coords_source.shape[0] < 100:
        warnings.warn("Very few cells (<100). Alignment may be unreliable.", stacklevel=2)

    # Extract x, y from (x, y) format
    xI, yI = coords_source[:, 0], coords_source[:, 1]
    xJ, yJ = coords_target[:, 0], coords_target[:, 1]

    # Compute initial affine transform
    L_init, T_init = None, None
    pointsI_lddmm, pointsJ_lddmm = None, None

    if initial_rotation_deg != 0.0:
        L_init, T_init = compute_initial_affine(xI, yI, xJ, yJ, initial_rotation_deg)
        if verbose:
            print(f"Applied initial rotation of {initial_rotation_deg} degrees")

    if landmark_points_source is not None and landmark_points_target is not None:
        # Convert landmarks from (x, y) to (row, col) = (y, x)
        pts_I_rc = np.column_stack([landmark_points_source[:, 1], landmark_points_source[:, 0]])
        pts_J_rc = np.column_stack([landmark_points_target[:, 1], landmark_points_target[:, 0]])

        if L_init is None:
            L_init, T_init = L_T_from_points(pts_I_rc, pts_J_rc)
            if verbose:
                print(f"Computed initial transform from {len(pts_I_rc)} landmark points")

        # Also use landmarks for point matching in LDDMM
        pointsI_lddmm = pts_I_rc
        pointsJ_lddmm = pts_J_rc

    # Apply initial transform to source coordinates for rasterization
    if L_init is not None:
        coords_I_rc = np.column_stack([yI, xI])  # (row, col)
        coords_I_transformed = (L_init @ coords_I_rc.T).T + T_init
        yI_t, xI_t = coords_I_transformed[:, 0], coords_I_transformed[:, 1]
    else:
        xI_t, yI_t = xI, yI

    # Rasterize both coordinate sets
    XI, YI, I = rasterize(xI_t, yI_t, dx=resolution, blur=blur)
    XJ, YJ, J = rasterize(xJ, yJ, dx=resolution, blur=blur)

    # Make 3-channel for LDDMM
    I_rgb = np.vstack([I, I, I])
    J_rgb = np.vstack([J, J, J])

    if verbose:
        print(f"Source image shape: {I_rgb.shape}")
        print(f"Target image shape: {J_rgb.shape}")
        print(f"Running {'LDDMM' if method == 'lddmm' else 'affine'} for {niter} iterations...")

    # Adjust diffeo_start for affine-only method
    if method == "affine":
        diffeo_start = niter + 1  # Never start diffeomorphic

    # Run LDDMM
    result = LDDMM(
        [YI, XI],
        I_rgb,
        [YJ, XJ],
        J_rgb,
        pointsI=pointsI_lddmm,
        pointsJ=pointsJ_lddmm,
        L=L_init,
        T=T_init,
        niter=niter,
        diffeo_start=diffeo_start,
        a=a,
        p=p,
        sigmaM=sigmaM,
        sigmaR=sigmaR,
        device=device,
        verbose=verbose,
    )

    # Transform source coordinates
    # Convert to (row, col) = (y, x) for transformation
    points_source_rc = np.column_stack([yI, xI])

    points_aligned_rc = transform_points_source_to_target(
        result["xv"],
        result["v"],
        result["A"],
        points_source_rc,
    )

    # Convert back to numpy and (x, y) order
    points_aligned_rc = points_aligned_rc.cpu().numpy()
    aligned_coords = np.column_stack([points_aligned_rc[:, 1], points_aligned_rc[:, 0]])

    # Build transform dictionary for later use
    transform_dict = {
        "A": result["A"].cpu().numpy(),
        "v": result["v"].cpu().numpy(),
        "xv": [x.cpu().numpy() for x in result["xv"]],
        "source_spatial_key": spatial_key,
        "target_shape": coords_target.shape,
        "resolution": resolution,
        "blur": blur,
        "method": method,
        "loss_history": result["Esave"],
    }

    # Store results
    adata = adata_source.copy() if copy else adata_source
    adata.obsm[key_added] = aligned_coords
    adata.uns["spatial_alignment"] = transform_dict

    if verbose:
        final_loss = result["Esave"][-1][0] if result["Esave"] else float("nan")
        print(f"Alignment complete. Final loss: {final_loss:.4f}")
        print(f"Aligned coordinates stored in adata.obsm['{key_added}']")

    if copy:
        return adata
    return None
