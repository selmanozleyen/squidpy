"""Spatial alignment for squidpy.

Dispatches to the appropriate backend based on input types:

- **Point-to-point**: :mod:`moscot` optimal transport (optional)
- **Point-to-image** / **Image-to-image**: STalign LDDMM (requires ``torch``)

Install optional dependencies::

    pip install 'squidpy[moscot]'    # point-to-point OT alignment
    pip install 'squidpy[torch]'     # image-based LDDMM alignment
    pip install 'squidpy[align]'     # both

Notes
-----
STalign reference: Clifton *et al.*, Nature Communications 14, 8123 (2023).
moscot reference: Klein *et al.*, (2023).
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

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


def _check_moscot() -> None:
    """Ensure ``moscot`` is available for point-to-point alignment."""
    try:
        import moscot  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Point-to-point alignment requires moscot.\n"
            "Install with:  pip install 'squidpy[moscot]'"
        ) from e


def _has_moscot() -> bool:
    try:
        import moscot  # noqa: F401
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
# Backend: moscot  (point-to-point)
# ---------------------------------------------------------------------------

def _align_points_moscot(
    adata_source: "AnnData",
    adata_target: "AnnData",
    *,
    spatial_key: str = "spatial",
    batch_key: str = "_align_batch",
    joint_attr: str | dict[str, Any] | None = None,
    GW_kwargs: dict[str, Any] | None = None,
    key_added: str = "spatial_aligned",
    copy: bool = False,
    verbose: bool = True,
    **kwargs: Any,
) -> "AnnData | None":
    """Align two point clouds via :class:`moscot.problems.space.AlignmentProblem`.

    Parameters
    ----------
    adata_source / adata_target
        AnnData objects with spatial coordinates in ``obsm[spatial_key]``.
    spatial_key
        Key in ``obsm`` for spatial coordinates.
    batch_key
        Temporary obs column used to distinguish source/target after concat.
    joint_attr
        Joint attribute for moscot.  ``None`` → spatial-only,
        ``"X_pca"`` → gene-expression guided.
    GW_kwargs
        Extra keyword arguments forwarded to ``AlignmentProblem.solve()``.
    key_added
        obsm key where aligned coordinates are stored.
    copy
        Return a modified copy instead of modifying *adata_source* in place.
    verbose
        Print progress.
    **kwargs
        Forwarded to ``AlignmentProblem.prepare()``.

    Returns
    -------
    Modified *adata_source* (if *copy*) or ``None``.
    """
    _check_moscot()
    import anndata as ad
    from moscot.problems.space import AlignmentProblem

    src_label, tgt_label = "source", "target"

    adata_src = adata_source.copy()
    adata_tgt = adata_target.copy()
    adata_src.obs[batch_key] = src_label
    adata_tgt.obs[batch_key] = tgt_label

    adata_combined = ad.concat(
        [adata_src, adata_tgt],
        label=batch_key,
        keys=[src_label, tgt_label],
    )
    adata_combined.obs[batch_key] = adata_combined.obs[batch_key].astype("category")

    if verbose:
        n_src, n_tgt = adata_src.n_obs, adata_tgt.n_obs
        print(
            f"[moscot] Combined AnnData: {adata_combined.n_obs} cells "
            f"({n_src} source + {n_tgt} target)"
        )

    # Prepare ---
    if joint_attr is None:
        joint_attr = {"attr": "obsm", "key": spatial_key}

    prepare_kw: dict[str, Any] = dict(batch_key=batch_key, joint_attr=joint_attr, **kwargs)
    ap = AlignmentProblem(adata_combined)
    ap = ap.prepare(**prepare_kw)

    # Solve ---
    solve_kw = dict(GW_kwargs or {})
    solve_kw.setdefault("max_iterations", 100)
    if verbose:
        print("[moscot] Solving optimal-transport alignment ...")
    ap = ap.solve(**solve_kw)

    # Align ---
    if verbose:
        print(f"[moscot] Projecting source onto target reference ...")
    ap.align(reference=tgt_label, key_added=key_added)

    # Extract aligned coordinates for source cells
    source_mask = adata_combined.obs[batch_key] == src_label
    aligned_coords = np.asarray(adata_combined[source_mask].obsm[key_added])

    # Store ---
    adata = adata_source.copy() if copy else adata_source
    adata.obsm[key_added] = aligned_coords
    adata.uns["spatial_alignment"] = {
        "method": "optimal_transport",
        "backend": "moscot",
        "spatial_key": spatial_key,
    }

    if verbose:
        print(f"[moscot] Aligned coordinates stored in obsm['{key_added}']")

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
    moscot is not installed.
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
    device: str = "cpu",
    verbose: bool = True,
    key_added: str = "spatial_aligned",
    copy: bool = False,
    # STalign / LDDMM -------------------------------------------------------
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
    # moscot -----------------------------------------------------------------
    joint_attr: str | dict[str, Any] | None = None,
    **kwargs: Any,
) -> "AnnData | dict[str, Any] | None":
    """Align spatial transcriptomics data.

    Automatically dispatches to the right backend depending on what
    *source* and *target* are:

    ==============================  ================  ====================
    Source → Target                 Backend           Extra dependency
    ==============================  ================  ====================
    AnnData → AnnData (points)      moscot (OT)       ``pip install moscot``
    AnnData → image                 STalign (LDDMM)   ``pip install torch``
    image   → image                 STalign (LDDMM)   ``pip install torch``
    SpatialData (image keys)        STalign (LDDMM)   ``pip install torch``
    ==============================  ================  ====================

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
        ``None`` — auto-select (moscot for point-to-point, LDDMM for images).
        ``"optimal_transport"`` — force moscot.
        ``"lddmm"`` — force STalign LDDMM.
        ``"affine"`` — affine-only via STalign.
    device
        PyTorch device for STalign (``"cpu"`` or ``"cuda:0"``).
    verbose
        Print progress.
    key_added
        ``obsm`` key for aligned coordinates (AnnData inputs).
    copy
        Return a modified copy of AnnData instead of in-place.
    resolution
        Pixel size for rasterising coordinates (STalign).
    blur
        Gaussian σ for rasterisation (STalign).
    niter / diffeo_start / a / p / sigmaM / sigmaR
        LDDMM hyper-parameters (STalign).
    initial_rotation_deg
        Initial clockwise rotation in degrees (STalign).
    landmark_source / landmark_target
        N×2 landmark arrays in ``(x, y)`` order (STalign).
    joint_attr
        Joint attribute for moscot (``None`` → spatial-only,
        ``"X_pca"`` → gene-expression guided).
    **kwargs
        Forwarded to the backend.

    Returns
    -------
    - AnnData source → modified AnnData (if *copy*) or ``None``
    - Image source → transformation ``dict``

    Examples
    --------
    **Point-to-point** via moscot:

    >>> import squidpy as sq
    >>> sq.experimental.tl.align(adata_src, adata_tgt)              # auto → moscot
    >>> sq.experimental.tl.align(adata_src, adata_tgt, method="optimal_transport")

    **Point-to-image** via STalign:

    >>> sq.experimental.tl.align(adata, histology_image)
    >>> aligned = adata.obsm["spatial_aligned"]

    **Image-to-image** via STalign:

    >>> transform = sq.experimental.tl.align(img_src, img_tgt)

    Notes
    -----
    Based on STalign [1]_ for image alignment and moscot [2]_ for
    optimal-transport alignment.

    References
    ----------
    .. [1] Clifton *et al.*, "STalign: Alignment of spatial
       transcriptomics data using diffeomorphic metric mapping",
       *Nat. Commun.* 14, 8123 (2023).
    .. [2] Klein *et al.*, "moscot: Multi-omic single-cell optimal
       transport tools" (2023).
    """
    src_type = _detect_input_type(source)
    tgt_type = _detect_input_type(target)

    # ----- SpatialData shortcut (image keys in kwargs) ---------------------
    if src_type == "spatialdata":
        source_key = kwargs.pop("source_image_key", None)
        target_key = kwargs.pop("target_image_key", None)
        if source_key and target_key:
            scale = kwargs.pop("scale", "auto")
            src_img = _extract_image_from_sdata(source, source_key, scale)
            tgt_img = _extract_image_from_sdata(source, target_key, scale)
            return _align_images_stalign(
                src_img, tgt_img,
                method=method or "lddmm",
                niter=niter, diffeo_start=diffeo_start,
                a=a, p=p, sigmaM=sigmaM, sigmaR=sigmaR,
                initial_rotation_deg=initial_rotation_deg,
                landmark_source=landmark_source,
                landmark_target=landmark_target,
                device=device, verbose=verbose,
                **kwargs,
            )

    # ----- Point-to-point -------------------------------------------------
    if src_type in ("anndata", "coords") and tgt_type in ("anndata", "coords"):
        # Wrap raw coords in AnnData
        if src_type == "coords":
            source = _coords_to_adata(np.asarray(source), spatial_key)
        if tgt_type == "coords":
            target = _coords_to_adata(np.asarray(target), spatial_key)

        # Auto-select backend
        use_moscot = False
        if method == "optimal_transport":
            use_moscot = True
        elif method is None:
            # Prefer moscot for point-to-point when available
            if _has_moscot():
                use_moscot = True
            elif _has_torch():
                use_moscot = False
            else:
                raise ImportError(
                    "Point-to-point alignment requires either moscot or torch.\n"
                    "Install with:  pip install moscot   OR   pip install torch"
                )

        if use_moscot:
            return _align_points_moscot(
                source, target,
                spatial_key=spatial_key,
                joint_attr=joint_attr,
                key_added=key_added,
                copy=copy,
                verbose=verbose,
                **kwargs,
            )
        else:
            if method is None:
                method = "lddmm"
            if verbose:
                print("[align] Using STalign (rasterise) for point-to-point alignment")
            return _align_points_to_points_stalign(
                source, target,
                spatial_key=spatial_key,
                method=method,
                resolution=resolution, blur=blur,
                niter=niter, diffeo_start=diffeo_start,
                a=a, p=p, sigmaM=sigmaM, sigmaR=sigmaR,
                initial_rotation_deg=initial_rotation_deg,
                landmark_source=landmark_source,
                landmark_target=landmark_target,
                device=device, verbose=verbose,
                key_added=key_added, copy=copy,
                **kwargs,
            )

    # ----- Point-to-image -------------------------------------------------
    if src_type in ("anndata", "coords") and tgt_type == "image":
        if src_type == "coords":
            source = _coords_to_adata(np.asarray(source), spatial_key)
        return _align_points_to_image_stalign(
            source, np.asarray(target),
            spatial_key=spatial_key,
            method=method if method in ("affine", "lddmm") else "lddmm",
            resolution=resolution, blur=blur,
            niter=niter, diffeo_start=diffeo_start,
            a=a, p=p, sigmaM=sigmaM, sigmaR=sigmaR,
            initial_rotation_deg=initial_rotation_deg,
            landmark_source=landmark_source,
            landmark_target=landmark_target,
            device=device, verbose=verbose,
            key_added=key_added, copy=copy,
            **kwargs,
        )

    # ----- Image-to-image -------------------------------------------------
    if src_type == "image" and tgt_type == "image":
        return _align_images_stalign(
            np.asarray(source), np.asarray(target),
            method=method if method in ("affine", "lddmm") else "lddmm",
            niter=niter, diffeo_start=diffeo_start,
            a=a, p=p, sigmaM=sigmaM, sigmaR=sigmaR,
            initial_rotation_deg=initial_rotation_deg,
            landmark_source=landmark_source,
            landmark_target=landmark_target,
            device=device, verbose=verbose,
            **kwargs,
        )

    raise TypeError(
        f"Unsupported alignment combination: {src_type} → {tgt_type}.\n"
        f"Supported: points→points, points→image, image→image."
    )
