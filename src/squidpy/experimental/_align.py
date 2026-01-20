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

    # For initial transform handling:
    # - If we have an initial L/T, we apply it to coords BEFORE rasterization
    #   (matching STalign workflow), then pass L=None, T=None to LDDMM
    # - LDDMM will optimize from identity, finding residual transformation
    if L_init is not None:
        coords_I_rc = np.column_stack([yI, xI])  # (row, col)
        coords_I_transformed = (L_init @ coords_I_rc.T).T + T_init
        yI_t, xI_t = coords_I_transformed[:, 0], coords_I_transformed[:, 1]
        # Store initial transform for later composition
        L_initial, T_initial = L_init, T_init
        # Don't pass to LDDMM - data is already pre-transformed
        L_init_lddmm, T_init_lddmm = None, None
    else:
        xI_t, yI_t = xI, yI
        L_initial, T_initial = None, None
        L_init_lddmm, T_init_lddmm = None, None

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
        L=L_init_lddmm,
        T=T_init_lddmm,
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
    # If we pre-transformed data, we need to apply the same initial transform first
    # Convert to (row, col) = (y, x) for transformation
    points_source_rc = np.column_stack([yI, xI])

    # Step 1: Apply initial transform if it was used
    if L_initial is not None:
        points_source_rc = (L_initial @ points_source_rc.T).T + T_initial

    # Step 2: Apply LDDMM transform
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
    # Store initial transform if used (for debugging/analysis)
    if L_initial is not None:
        transform_dict["L_initial"] = L_initial
        transform_dict["T_initial"] = T_initial
        transform_dict["initial_rotation_deg"] = initial_rotation_deg

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


def align_images(
    source_image: "NDArray[np.floating]",
    target_image: "NDArray[np.floating]",
    source_extent: tuple[float, float, float, float] | None = None,
    target_extent: tuple[float, float, float, float] | None = None,
    *,
    method: Literal["affine", "lddmm"] = "lddmm",
    niter: int = 2000,
    diffeo_start: int = 100,
    a: float = 500.0,
    p: float = 2.0,
    sigmaM: float = 1.0,
    sigmaR: float = 5e5,
    initial_rotation_deg: float = 0.0,
    landmark_points_source: "NDArray[np.floating] | None" = None,
    landmark_points_target: "NDArray[np.floating] | None" = None,
    device: str = "cpu",
    verbose: bool = True,
) -> dict[str, Any]:
    """Align two images using LDDMM diffeomorphic registration.

    This function performs diffeomorphic registration between two images,
    useful for aligning histology images or pre-rasterized density maps.

    Parameters
    ----------
    source_image
        Source image array with shape (H, W) or (H, W, C) or (C, H, W).
    target_image
        Target image array with same format as source.
    source_extent
        Optional (xmin, xmax, ymin, ymax) extent for source image.
        If None, uses pixel coordinates.
    target_extent
        Optional (xmin, xmax, ymin, ymax) extent for target image.
    method
        Alignment method. 'lddmm' for full diffeomorphic, 'affine' for
        affine-only. Default: 'lddmm'.
    niter
        Number of optimization iterations. Default: 2000.
    diffeo_start
        Iteration to start nonlinear deformation. Default: 100.
    a
        Smoothness scale of velocity field. Default: 500.0.
    p
        Power of Laplacian regularization. Default: 2.0.
    sigmaM
        Image matching weight. Default: 1.0.
    sigmaR
        Regularization weight. Default: 5e5.
    initial_rotation_deg
        Initial rotation in degrees. Default: 0.0.
    landmark_points_source
        Optional Nx2 array of landmark points in source (row, col order).
    landmark_points_target
        Optional Nx2 array of corresponding landmark points in target.
    device
        PyTorch device. Default: 'cpu'.
    verbose
        Print progress. Default: True.

    Returns
    -------
    Dictionary containing transformation parameters:

    - 'A': 3x3 affine matrix
    - 'v': velocity field
    - 'xv': velocity field coordinates
    - 'source_extent': source image extent
    - 'target_extent': target image extent
    - 'loss_history': optimization loss history

    Examples
    --------
    >>> import squidpy as sq
    >>> import numpy as np
    >>> # Align two histology images
    >>> transform = sq.experimental.align_images(source_img, target_img)
    >>> # Transform coordinates using the result
    >>> aligned_coords = sq.experimental.apply_transform(
    ...     coords, transform, direction='source_to_target'
    ... )
    """
    _check_torch()

    from squidpy.experimental._lddmm import LDDMM
    from squidpy.experimental._lddmm._transforms import L_T_from_points

    # Normalize image shapes to (C, H, W)
    source_image = np.asarray(source_image)
    target_image = np.asarray(target_image)

    if source_image.ndim == 2:
        source_image = source_image[np.newaxis, :, :]
    elif source_image.ndim == 3 and source_image.shape[-1] <= 4:
        # (H, W, C) -> (C, H, W)
        source_image = np.moveaxis(source_image, -1, 0)

    if target_image.ndim == 2:
        target_image = target_image[np.newaxis, :, :]
    elif target_image.ndim == 3 and target_image.shape[-1] <= 4:
        target_image = np.moveaxis(target_image, -1, 0)

    # Ensure 3 channels for LDDMM
    if source_image.shape[0] == 1:
        source_image = np.vstack([source_image] * 3)
    if target_image.shape[0] == 1:
        target_image = np.vstack([target_image] * 3)

    # Normalize to [0, 1]
    source_image = (source_image - source_image.min()) / (source_image.max() - source_image.min() + 1e-8)
    target_image = (target_image - target_image.min()) / (target_image.max() - target_image.min() + 1e-8)

    # Create coordinate arrays
    _, h_I, w_I = source_image.shape
    _, h_J, w_J = target_image.shape

    if source_extent is None:
        XI = np.arange(w_I, dtype=np.float64)
        YI = np.arange(h_I, dtype=np.float64)
    else:
        xmin, xmax, ymin, ymax = source_extent
        XI = np.linspace(xmin, xmax, w_I)
        YI = np.linspace(ymin, ymax, h_I)

    if target_extent is None:
        XJ = np.arange(w_J, dtype=np.float64)
        YJ = np.arange(h_J, dtype=np.float64)
    else:
        xmin, xmax, ymin, ymax = target_extent
        XJ = np.linspace(xmin, xmax, w_J)
        YJ = np.linspace(ymin, ymax, h_J)

    # Auto-adjust smoothness scale 'a' for small images
    # Default a=500 is for large images; scale down for smaller images
    image_extent = max(h_I, w_I, h_J, w_J)
    if a == 500.0 and image_extent < 200:
        # Scale a proportionally to image size, minimum 5
        a = max(5.0, image_extent / 4)

    # Initial transform
    L_init, T_init = None, None
    if initial_rotation_deg != 0.0:
        theta = np.radians(-initial_rotation_deg)
        L_init = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        center = np.array([YI.mean(), XI.mean()])
        T_init = center - L_init @ center

    if landmark_points_source is not None and landmark_points_target is not None:
        if L_init is None:
            L_init, T_init = L_T_from_points(landmark_points_source, landmark_points_target)

    if method == "affine":
        diffeo_start = niter + 1

    if verbose:
        print(f"Source image shape: {source_image.shape}")
        print(f"Target image shape: {target_image.shape}")
        print(f"Running {'LDDMM' if method == 'lddmm' else 'affine'} for {niter} iterations...")

    result = LDDMM(
        [YI, XI],
        source_image,
        [YJ, XJ],
        target_image,
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

    transform_dict = {
        "A": result["A"].cpu().numpy(),
        "v": result["v"].cpu().numpy(),
        "xv": [x.cpu().numpy() for x in result["xv"]],
        "source_extent": source_extent or (0, w_I - 1, 0, h_I - 1),
        "target_extent": target_extent or (0, w_J - 1, 0, h_J - 1),
        "method": method,
        "loss_history": result["Esave"],
    }

    if verbose:
        final_loss = result["Esave"][-1][0] if result["Esave"] else float("nan")
        print(f"Alignment complete. Final loss: {final_loss:.4f}")

    return transform_dict


def transform_image(
    image: "NDArray[np.floating]",
    transform: dict[str, Any],
    output_shape: tuple[int, int] | None = None,
) -> "NDArray[np.floating]":
    """Transform an image using a saved alignment transformation.

    Parameters
    ----------
    image
        Image array with shape (H, W) or (H, W, C) or (C, H, W).
    transform
        Transformation dict from align_spatial or align_images.
    output_shape
        Optional (H, W) shape for output. If None, uses input shape.

    Returns
    -------
    Transformed image array with same format as input.

    Examples
    --------
    >>> import squidpy as sq
    >>> # After running alignment
    >>> transformed_img = sq.experimental.transform_image(
    ...     source_image,
    ...     adata.uns['spatial_alignment']
    ... )
    """
    _check_torch()
    torch = _get_torch()

    from squidpy.experimental._lddmm import transform_image_source_to_target

    image = np.asarray(image)
    original_ndim = image.ndim
    hwc_format = False

    # Normalize to (C, H, W)
    if image.ndim == 2:
        image = image[np.newaxis, :, :]
    elif image.ndim == 3 and image.shape[-1] <= 4:
        hwc_format = True
        image = np.moveaxis(image, -1, 0)

    _, h, w = image.shape

    # Build coordinate arrays
    xI = [np.arange(h, dtype=np.float64), np.arange(w, dtype=np.float64)]

    if output_shape is None:
        output_shape = (h, w)

    XJ = [
        np.arange(output_shape[0], dtype=np.float64),
        np.arange(output_shape[1], dtype=np.float64),
    ]

    A = torch.tensor(transform["A"])
    v = torch.tensor(transform["v"])
    xv = [torch.tensor(x) for x in transform["xv"]]

    transformed = transform_image_source_to_target(xv, v, A, xI, image, XJ)
    transformed = transformed.cpu().numpy()

    # Convert back to original format
    if hwc_format:
        transformed = np.moveaxis(transformed, 0, -1)
    elif original_ndim == 2:
        transformed = transformed[0]

    return transformed


def _extract_image_from_sdata(
    sdata: "sd.SpatialData",
    image_key: str,
    scale: str = "auto",
) -> tuple["NDArray[np.floating]", tuple[float, float, float, float] | None]:
    """Extract image and its extent from a SpatialData object.

    Parameters
    ----------
    sdata
        SpatialData object.
    image_key
        Key of the image in sdata.images.
    scale
        Scale level to use.

    Returns
    -------
    Tuple of (image array in (C, H, W) format, extent or None).
    """
    from squidpy.experimental.im._utils import _get_element_data

    image_node = sdata.images[image_key]
    image_data = _get_element_data(image_node, scale, "image", image_key)

    # Convert to numpy
    img = np.asarray(image_data.values if hasattr(image_data, "values") else image_data)

    # Try to extract coordinate extent from xarray coords
    extent = None
    if hasattr(image_data, "coords"):
        try:
            if "x" in image_data.coords and "y" in image_data.coords:
                x_coords = image_data.coords["x"].values
                y_coords = image_data.coords["y"].values
                extent = (
                    float(x_coords.min()),
                    float(x_coords.max()),
                    float(y_coords.min()),
                    float(y_coords.max()),
                )
        except (KeyError, AttributeError):
            pass

    # Normalize to (C, H, W) format
    if img.ndim == 2:
        img = img[np.newaxis, :, :]
    elif img.ndim == 3:
        # Check if it's (H, W, C) or (C, H, W)
        if img.shape[-1] <= 4 and img.shape[0] > 4:
            # Likely (H, W, C)
            img = np.moveaxis(img, -1, 0)

    return img, extent


def align_images_sdata(
    sdata: "sd.SpatialData",
    source_image_key: str,
    target_image_key: str,
    *,
    scale: str = "auto",
    **kwargs: Any,
) -> dict[str, Any]:
    """Align two images from a SpatialData object.

    This is a convenience function that extracts images from a SpatialData
    object and calls `align_images`.

    Parameters
    ----------
    sdata
        SpatialData object containing the images.
    source_image_key
        Key of the source image in ``sdata.images``.
    target_image_key
        Key of the target image in ``sdata.images``.
    scale
        Scale level to use. 'auto' uses the coarsest available scale.
    **kwargs
        Additional arguments passed to :func:`align_images`.

    Returns
    -------
    Transformation dictionary from :func:`align_images`.

    Examples
    --------
    >>> import squidpy as sq
    >>> transform = sq.experimental.align_images_sdata(
    ...     sdata, 'image_source', 'image_target'
    ... )
    """
    source_img, source_extent = _extract_image_from_sdata(sdata, source_image_key, scale)
    target_img, target_extent = _extract_image_from_sdata(sdata, target_image_key, scale)

    return align_images(
        source_img,
        target_img,
        source_extent=source_extent,
        target_extent=target_extent,
        **kwargs,
    )


def align_to_image(
    adata_source: "AnnData",
    target_image: "NDArray[np.floating] | sd.SpatialData",
    target_image_key: str | None = None,
    *,
    spatial_key: str = "spatial",
    scale: str = "auto",
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
    landmark_points_source: "NDArray[np.floating] | None" = None,
    landmark_points_target: "NDArray[np.floating] | None" = None,
    device: str = "cpu",
    verbose: bool = True,
    key_added: str = "spatial_aligned",
    copy: bool = False,
) -> "AnnData | None":
    """Align spatial coordinates to a target image.

    This function aligns source cell coordinates to a target histology image
    (e.g., H&E) or other reference image. This is useful for:

    - Aligning single-cell ST data to corresponding H&E images
    - Aligning to reference atlas images
    - Cross-modal alignment (ST to histology)

    The source coordinates are rasterized into a density image, which is then
    aligned to the target image using LDDMM.

    Parameters
    ----------
    adata_source
        Source AnnData with spatial coordinates in ``obsm[spatial_key]``.
    target_image
        Target image as numpy array (H, W) or (H, W, C) or (C, H, W),
        or a SpatialData object (requires ``target_image_key``).
    target_image_key
        If ``target_image`` is a SpatialData object, the key of the image
        in ``sdata.images`` to align to.
    spatial_key
        Key in ``obsm`` containing spatial coordinates. Default: 'spatial'.
    scale
        Scale level for SpatialData images. Default: 'auto'.
    method
        Alignment method. 'lddmm' for full diffeomorphic, 'affine' for
        affine-only. Default: 'lddmm'.
    resolution
        Pixel size for source coordinate rasterization. Default: 30.0.
    blur
        Gaussian blur sigma for rasterization. Default: 1.5.
    niter
        Number of optimization iterations. Default: 2000.
    diffeo_start
        Iteration to start nonlinear deformation. Default: 100.
    a
        Smoothness scale of velocity field. Default: 500.0.
    p
        Power of Laplacian regularization. Default: 2.0.
    sigmaM
        Image matching weight. Default: 1.0.
    sigmaR
        Regularization weight. Default: 5e5.
    initial_rotation_deg
        Initial rotation in degrees. Default: 0.0.
    landmark_points_source
        Optional Nx2 array of landmark points in source (x, y order).
    landmark_points_target
        Optional Nx2 array of corresponding landmark points in target (x, y order).
    device
        PyTorch device. Default: 'cpu'.
    verbose
        Print progress. Default: True.
    key_added
        Key for aligned coordinates in ``obsm``. Default: 'spatial_aligned'.
    copy
        Return a copy. Default: False.

    Returns
    -------
    If ``copy=True``, returns modified AnnData. Otherwise modifies in place.

    Examples
    --------
    >>> import squidpy as sq
    >>> # Align cells to H&E histology image
    >>> sq.experimental.align_to_image(
    ...     adata_source,
    ...     histology_image,
    ...     resolution=10.0,  # Match histology resolution
    ... )

    >>> # Align to image from SpatialData
    >>> sq.experimental.align_to_image(
    ...     adata_source,
    ...     sdata,
    ...     target_image_key='he_image',
    ... )

    Notes
    -----
    This function is inspired by STalign's ability to align single-cell
    resolution ST data to corresponding histology images. The source
    coordinates are converted to a density image that represents cell
    distribution, which is then aligned to the target histology image.

    For best results:

    - Adjust ``resolution`` to match the scale of the target image
    - Use landmarks if automatic alignment fails
    - Consider pre-rotating with ``initial_rotation_deg`` if images
      have different orientations
    """
    _check_torch()
    torch = _get_torch()

    import spatialdata as sd

    from squidpy.experimental._lddmm import (
        LDDMM,
        rasterize,
        transform_points_source_to_target,
    )
    from squidpy.experimental._lddmm._transforms import (
        L_T_from_points,
        compute_initial_affine,
    )

    # Validate source coordinates
    if spatial_key not in adata_source.obsm:
        raise KeyError(
            f"Spatial key '{spatial_key}' not found in adata_source.obsm. "
            f"Available keys: {list(adata_source.obsm.keys())}"
        )

    coords_source = adata_source.obsm[spatial_key]
    if coords_source.shape[1] != 2:
        raise ValueError(f"Expected 2D coordinates, got shape {coords_source.shape}")

    if coords_source.shape[0] < 100:
        warnings.warn("Very few cells (<100). Alignment may be unreliable.", stacklevel=2)

    # Extract target image
    target_extent = None
    if isinstance(target_image, sd.SpatialData):
        if target_image_key is None:
            raise ValueError("target_image_key must be provided when target_image is a SpatialData object")
        J, target_extent = _extract_image_from_sdata(target_image, target_image_key, scale)
        if verbose:
            print(f"Loaded target image '{target_image_key}' from SpatialData")
    else:
        J = np.asarray(target_image)
        # Normalize to (C, H, W)
        if J.ndim == 2:
            J = J[np.newaxis, :, :]
        elif J.ndim == 3 and J.shape[-1] <= 4 and J.shape[0] > 4:
            J = np.moveaxis(J, -1, 0)

    # Ensure 3 channels
    if J.shape[0] == 1:
        J = np.vstack([J, J, J])
    elif J.shape[0] == 4:
        J = J[:3]  # Drop alpha channel

    # Normalize to [0, 1]
    J = (J - J.min()) / (J.max() - J.min() + 1e-8)

    _, h_J, w_J = J.shape

    # Create target coordinate arrays
    if target_extent is not None:
        xmin, xmax, ymin, ymax = target_extent
        XJ = np.linspace(xmin, xmax, w_J)
        YJ = np.linspace(ymin, ymax, h_J)
    else:
        XJ = np.arange(w_J, dtype=np.float64)
        YJ = np.arange(h_J, dtype=np.float64)

    # Extract source coordinates
    xI, yI = coords_source[:, 0], coords_source[:, 1]

    # Compute initial affine transform
    L_init, T_init = None, None
    pointsI_lddmm, pointsJ_lddmm = None, None

    if initial_rotation_deg != 0.0:
        L_init, T_init = compute_initial_affine(xI, yI, XJ, YJ, initial_rotation_deg)
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

        pointsI_lddmm = pts_I_rc
        pointsJ_lddmm = pts_J_rc

    # Apply initial transform to source coordinates for rasterization
    if L_init is not None:
        coords_I_rc = np.column_stack([yI, xI])
        coords_I_transformed = (L_init @ coords_I_rc.T).T + T_init
        yI_t, xI_t = coords_I_transformed[:, 0], coords_I_transformed[:, 1]
    else:
        xI_t, yI_t = xI, yI

    # Rasterize source coordinates
    XI, YI, I = rasterize(xI_t, yI_t, dx=resolution, blur=blur)

    # Make 3-channel
    I_rgb = np.vstack([I, I, I])

    if verbose:
        print(f"Source (rasterized) image shape: {I_rgb.shape}")
        print(f"Target image shape: {J.shape}")
        print(f"Running {'LDDMM' if method == 'lddmm' else 'affine'} for {niter} iterations...")

    # Adjust diffeo_start for affine-only method
    if method == "affine":
        diffeo_start = niter + 1

    # Run LDDMM
    result = LDDMM(
        [YI, XI],
        I_rgb,
        [YJ, XJ],
        J,
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

    # Build transform dictionary
    transform_dict = {
        "A": result["A"].cpu().numpy(),
        "v": result["v"].cpu().numpy(),
        "xv": [x.cpu().numpy() for x in result["xv"]],
        "source_spatial_key": spatial_key,
        "target_image_shape": J.shape,
        "target_extent": target_extent,
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
