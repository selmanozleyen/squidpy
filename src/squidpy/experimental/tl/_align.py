"""High-level spatial alignment function for SpatialData.

This module provides the main entry point for aligning spatial transcriptomics
data using LDDMM (Large Deformation Diffeomorphic Metric Mapping).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    import spatialdata as sd
    from numpy.typing import NDArray


def _check_torch() -> None:
    """Check if PyTorch is available."""
    try:
        import torch  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Spatial alignment requires PyTorch. Install with: pip install torch"
        ) from e


def _get_torch():
    """Import and return torch module."""
    _check_torch()
    import torch

    return torch


def _get_element_data_for_align(
    sdata: "sd.SpatialData",
    image_key: str,
    scale: str | Literal["auto"] = "auto",
) -> tuple["NDArray[np.floating]", tuple[float, float, float, float] | None]:
    """Extract image data from SpatialData for alignment.

    Parameters
    ----------
    sdata
        SpatialData object.
    image_key
        Key in sdata.images.
    scale
        Scale level. "auto" picks the coarsest (smallest) scale for efficiency.

    Returns
    -------
    Tuple of (image array in (C, H, W) format, extent or None).
    """
    from squidpy.experimental.im._utils import _get_element_data

    if image_key not in sdata.images:
        raise KeyError(
            f"Image '{image_key}' not found in sdata.images. "
            f"Available: {list(sdata.images.keys())}"
        )

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


def _ensure_3_channels(image: "NDArray[np.floating]") -> "NDArray[np.floating]":
    """Ensure image has 3 channels for LDDMM."""
    if image.shape[0] == 1:
        return np.vstack([image] * 3)
    elif image.shape[0] == 4:
        return image[:3]  # Drop alpha channel
    return image


def _normalize_image_range(image: "NDArray[np.floating]") -> "NDArray[np.floating]":
    """Normalize image to [0, 1] range."""
    return (image - image.min()) / (image.max() - image.min() + 1e-8)


def _get_available_scales(sdata: "sd.SpatialData", image_key: str) -> list[str]:
    """Get available scale levels for an image."""
    image_node = sdata.images[image_key]
    if hasattr(image_node, "keys"):
        return list(image_node.keys())
    return ["scale0"]


def apply_affine(
    image: "NDArray[np.floating]",
    affine_matrix: "NDArray[np.floating]",
    output_shape: tuple[int, int] | None = None,
) -> "NDArray[np.floating]":
    """Apply an affine transformation to an image.

    Parameters
    ----------
    image
        Input image with shape (C, H, W) or (H, W) or (H, W, C).
    affine_matrix
        3x3 affine transformation matrix that maps source to target coordinates.
    output_shape
        Output image shape as (H, W). If None, uses input image shape.

    Returns
    -------
    Transformed image with same channel format as input.

    Examples
    --------
    >>> import squidpy as sq
    >>> import numpy as np

    >>> # After running alignment
    >>> transform = sq.experimental.tl.align(sdata, "img1", "img2")

    >>> # Apply the affine to another image
    >>> transformed = sq.experimental.tl.apply_affine(
    ...     my_image,
    ...     transform["A"],
    ...     output_shape=(target_h, target_w),
    ... )
    """
    _check_torch()
    torch = _get_torch()

    image = np.asarray(image)
    A = np.asarray(affine_matrix)

    # Track original format
    original_ndim = image.ndim
    hwc_format = False

    # Normalize to (C, H, W)
    if image.ndim == 2:
        image = image[np.newaxis, :, :]
    elif image.ndim == 3 and image.shape[-1] <= 4 and image.shape[0] > 4:
        # Likely (H, W, C)
        hwc_format = True
        image = np.moveaxis(image, -1, 0)

    _, h_I, w_I = image.shape

    if output_shape is None:
        h_J, w_J = h_I, w_I
    else:
        h_J, w_J = output_shape

    # Convert to torch tensor
    source_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # (1, C, H, W)

    # Create grid for target coordinates
    y_coords = torch.linspace(-1, 1, h_J)
    x_coords = torch.linspace(-1, 1, w_J)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')

    # Convert grid to pixel coordinates
    grid_pixels_x = (grid_x + 1) / 2 * (w_J - 1)
    grid_pixels_y = (grid_y + 1) / 2 * (h_J - 1)

    # Stack as homogeneous coordinates (row, col, 1)
    ones = torch.ones_like(grid_pixels_x)
    grid_homogeneous = torch.stack([grid_pixels_y, grid_pixels_x, ones], dim=-1)  # (H, W, 3)

    # Apply INVERSE of affine to get source coordinates
    # A maps source -> target, so A_inv maps target -> source
    A_inv = np.linalg.inv(A)
    A_inv_tensor = torch.tensor(A_inv, dtype=torch.float32)

    # Reshape for matrix multiplication
    grid_flat = grid_homogeneous.reshape(-1, 3)  # (H*W, 3)
    source_coords = (A_inv_tensor @ grid_flat.T).T  # (H*W, 3)
    source_coords = source_coords.reshape(h_J, w_J, 3)

    # Convert back to normalized [-1, 1] for grid_sample
    source_y_norm = source_coords[..., 0] / (h_I - 1) * 2 - 1
    source_x_norm = source_coords[..., 1] / (w_I - 1) * 2 - 1

    # grid_sample expects (x, y) order
    sample_grid = torch.stack([source_x_norm, source_y_norm], dim=-1).unsqueeze(0)  # (1, H, W, 2)

    # Apply transform
    transformed = torch.nn.functional.grid_sample(
        source_tensor,
        sample_grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    )
    result = transformed.squeeze(0).numpy()  # (C, H, W)

    # Convert back to original format
    if hwc_format:
        result = np.moveaxis(result, 0, -1)
    elif original_ndim == 2:
        result = result[0]

    return result


def align(
    sdata: "sd.SpatialData",
    source_image_key: str,
    target_image_key: str,
    *,
    scale: str | Literal["auto"] = "auto",
    method: Literal["affine", "lddmm"] = "lddmm",
    # LDDMM parameters
    niter: int = 2000,
    diffeo_start: int = 100,
    a: float = 500.0,
    p: float = 2.0,
    sigmaM: float = 1.0,
    sigmaR: float = 5e5,
    # Initial transform
    initial_rotation_deg: float = 0.0,
    landmark_points_source: "NDArray[np.floating] | None" = None,
    landmark_points_target: "NDArray[np.floating] | None" = None,
    # Computation
    device: str = "cpu",
    verbose: bool = True,
) -> dict[str, Any]:
    """Align two images from a SpatialData object using LDDMM.

    This function performs diffeomorphic registration (LDDMM) to align
    two images within a SpatialData object. By default, it uses the
    coarsest available scale (smallest image) for efficient computation.

    Parameters
    ----------
    sdata
        SpatialData object containing the images to align.
    source_image_key
        Key of the source image in ``sdata.images``.
    target_image_key
        Key of the target image in ``sdata.images``.
    scale
        Scale level to use for alignment. Default: "auto" (uses the coarsest
        available scale, i.e., the smallest image with the highest scale number).
        This is recommended for efficiency. Use "scale0" for full resolution.
    method
        Alignment method. 'lddmm' for full diffeomorphic registration,
        'affine' for affine-only. Default: 'lddmm'.
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
        Optional Nx2 array of landmark points in source (row, col order).
    landmark_points_target
        Optional Nx2 array of corresponding landmark points in target.
    device
        PyTorch device ('cpu' or 'cuda:0'). Default: 'cpu'.
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
    - 'source_image_key': source image key
    - 'target_image_key': target image key
    - 'scale_used': scale level used for alignment
    - 'loss_history': optimization loss history
    - 'method': alignment method used

    Examples
    --------
    >>> import squidpy as sq
    >>> import spatialdata as sd

    >>> # Load SpatialData
    >>> sdata = sd.read_zarr("path/to/data.zarr")

    >>> # Align two images (uses coarsest scale by default)
    >>> transform = sq.experimental.tl.align(
    ...     sdata,
    ...     source_image_key="image1",
    ...     target_image_key="image2",
    ... )

    >>> # Use full resolution
    >>> transform = sq.experimental.tl.align(
    ...     sdata,
    ...     source_image_key="image1",
    ...     target_image_key="image2",
    ...     scale="scale0",
    ... )

    >>> # Affine-only alignment (faster)
    >>> transform = sq.experimental.tl.align(
    ...     sdata,
    ...     source_image_key="image1",
    ...     target_image_key="image2",
    ...     method="affine",
    ... )

    Notes
    -----
    This function is based on STalign (https://github.com/JEFworks-Lab/STalign).
    The algorithm uses Large Deformation Diffeomorphic Metric Mapping (LDDMM)
    to compute a smooth, invertible transformation between images.

    The "auto" scale selection picks the coarsest available scale (e.g., "scale4"
    if available) which provides the smallest image. This is recommended for
    initial alignment as it's much faster. You can then refine with finer scales
    if needed.

    See Also
    --------
    squidpy.experimental.apply_transform : Apply transformation to coordinates
    squidpy.experimental.transform_image : Apply transformation to images
    """
    _check_torch()

    from squidpy.experimental._lddmm import LDDMM
    from squidpy.experimental._lddmm._transforms import L_T_from_points

    # Extract images
    if verbose:
        source_scales = _get_available_scales(sdata, source_image_key)
        target_scales = _get_available_scales(sdata, target_image_key)
        print(f"Source image '{source_image_key}' scales: {source_scales}")
        print(f"Target image '{target_image_key}' scales: {target_scales}")

    source_img, source_extent = _get_element_data_for_align(sdata, source_image_key, scale)
    target_img, target_extent = _get_element_data_for_align(sdata, target_image_key, scale)

    # Ensure 3 channels
    source_img = _ensure_3_channels(source_img)
    target_img = _ensure_3_channels(target_img)

    # Normalize to [0, 1]
    source_img = _normalize_image_range(source_img)
    target_img = _normalize_image_range(target_img)

    # Create coordinate arrays
    _, h_I, w_I = source_img.shape
    _, h_J, w_J = target_img.shape

    if source_extent is not None:
        xmin, xmax, ymin, ymax = source_extent
        XI = np.linspace(xmin, xmax, w_I)
        YI = np.linspace(ymin, ymax, h_I)
    else:
        XI = np.arange(w_I, dtype=np.float64)
        YI = np.arange(h_I, dtype=np.float64)

    if target_extent is not None:
        xmin, xmax, ymin, ymax = target_extent
        XJ = np.linspace(xmin, xmax, w_J)
        YJ = np.linspace(ymin, ymax, h_J)
    else:
        XJ = np.arange(w_J, dtype=np.float64)
        YJ = np.arange(h_J, dtype=np.float64)

    # Auto-adjust smoothness scale 'a' for small images
    image_extent = max(h_I, w_I, h_J, w_J)
    a_adjusted = a
    if a == 500.0 and image_extent < 200:
        a_adjusted = max(5.0, image_extent / 4)
        if verbose:
            print(f"Auto-adjusted smoothness 'a' from {a} to {a_adjusted} for small images")

    # Initial transform
    L_init, T_init = None, None
    if initial_rotation_deg != 0.0:
        theta = np.radians(-initial_rotation_deg)
        L_init = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        center = np.array([YI.mean(), XI.mean()])
        T_init = center - L_init @ center
        if verbose:
            print(f"Applied initial rotation of {initial_rotation_deg} degrees")

    if landmark_points_source is not None and landmark_points_target is not None:
        if L_init is None:
            L_init, T_init = L_T_from_points(landmark_points_source, landmark_points_target)
            if verbose:
                print(f"Computed initial transform from {len(landmark_points_source)} landmark points")

    if method == "affine":
        diffeo_start = niter + 1

    if verbose:
        print(f"Source image shape: {source_img.shape}")
        print(f"Target image shape: {target_img.shape}")
        print(f"Running {'LDDMM' if method == 'lddmm' else 'affine'} for {niter} iterations...")

    # Run LDDMM
    result = LDDMM(
        [YI, XI],
        source_img,
        [YJ, XJ],
        target_img,
        L=L_init,
        T=T_init,
        niter=niter,
        diffeo_start=diffeo_start,
        a=a_adjusted,
        p=p,
        sigmaM=sigmaM,
        sigmaR=sigmaR,
        device=device,
        verbose=verbose,
    )

    # Determine which scale was actually used
    scale_used = scale
    if scale == "auto":
        # Get the actual scale that was selected
        source_node = sdata.images[source_image_key]
        if hasattr(source_node, "keys"):
            available = list(source_node.keys())

            def _idx(k: str) -> int:
                num = "".join(ch for ch in k if ch.isdigit())
                return int(num) if num else -1

            scale_used = max(available, key=_idx)

    transform_dict = {
        "A": result["A"].cpu().numpy(),
        "v": result["v"].cpu().numpy(),
        "xv": [x.cpu().numpy() for x in result["xv"]],
        "source_extent": source_extent or (0, w_I - 1, 0, h_I - 1),
        "target_extent": target_extent or (0, w_J - 1, 0, h_J - 1),
        "source_image_key": source_image_key,
        "target_image_key": target_image_key,
        "scale_used": scale_used,
        "method": method,
        "loss_history": result["Esave"],
    }

    if verbose:
        final_loss = result["Esave"][-1][0] if result["Esave"] else float("nan")
        print(f"Alignment complete. Final loss: {final_loss:.4f}")
        print(f"Scale used: {scale_used}")

    return transform_dict
