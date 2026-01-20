"""Transform functions for applying LDDMM transformations.

Functions ported from STalign (https://github.com/JEFworks-Lab/STalign).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    import torch
    from numpy.typing import NDArray


def _get_torch():
    """Import and return torch module."""
    try:
        import torch

        return torch
    except ImportError as e:
        raise ImportError(
            "Spatial alignment requires PyTorch. " "Install with: pip install torch"
        ) from e


def extent_from_x(xJ: list["torch.Tensor"]) -> tuple[float, float, float, float]:
    """Get matplotlib imshow extent from pixel locations.

    Parameters
    ----------
    xJ
        Location of pixels along each axis (row, col order).

    Returns
    -------
    (xmin, xmax, ymin, ymax) extent for imshow.

    Examples
    --------
    >>> extent = extent_from_x(xJ)
    >>> plt.imshow(image, extent=extent)
    """
    dJ = [x[1] - x[0] for x in xJ]
    extentJ = (
        (xJ[1][0] - dJ[1] / 2.0).item(),
        (xJ[1][-1] + dJ[1] / 2.0).item(),
        (xJ[0][-1] + dJ[0] / 2.0).item(),
        (xJ[0][0] - dJ[0] / 2.0).item(),
    )
    return extentJ


def L_T_from_points(
    pointsI: NDArray[np.floating],
    pointsJ: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Compute affine transformation from corresponding points.

    Estimates the best-fit affine transform that maps pointsI to pointsJ.
    For 2 points, computes translation only. For 3+ points, computes full affine.

    Parameters
    ----------
    pointsI
        Source points with shape (N, 2) in (row, col) order.
    pointsJ
        Target points with shape (N, 2) in (row, col) order.

    Returns
    -------
    A tuple containing:

    - L: 2x2 linear transform matrix.
    - T: 2-element translation vector.

    Raises
    ------
    ValueError
        If inputs have inconsistent shapes.
    """
    if pointsI is None or pointsJ is None:
        raise ValueError("Points cannot be None")

    nI, nJ = pointsI.shape[0], pointsJ.shape[0]

    if nI != nJ:
        raise ValueError(f"Number of source points ({nI}) != target points ({nJ})")
    if pointsI.shape[1] != 2 or pointsJ.shape[1] != 2:
        raise ValueError("Points must have shape (N, 2)")

    if nI < 3:
        # Translation only
        L = np.eye(2)
        T = np.mean(pointsJ, 0) - np.mean(pointsI, 0)
    else:
        # Full affine transform
        pointsI_ = np.concatenate((pointsI, np.ones((nI, 1))), 1)
        pointsJ_ = np.concatenate((pointsJ, np.ones((nI, 1))), 1)
        II = pointsI_.T @ pointsI_
        IJ = pointsI_.T @ pointsJ_
        A = (np.linalg.inv(II) @ IJ).T
        L = A[:2, :2]
        T = A[:2, -1]

    return L, T


def _interp(
    x: list["torch.Tensor"],
    I: "torch.Tensor",
    phii: "torch.Tensor",
    **kwargs,
) -> "torch.Tensor":
    """Interpolate 2D image at specified coordinates (internal function)."""
    torch = _get_torch()
    from torch.nn.functional import grid_sample

    I = torch.as_tensor(I)
    phii = torch.as_tensor(phii)
    phii = torch.clone(phii)

    # Normalize coordinates to [-1, 1] range
    for i in range(2):
        phii[i] -= x[i][0]
        phii[i] /= x[i][-1] - x[i][0]
    phii = phii * 2.0 - 1.0

    # grid_sample expects (N, H, W, 2) with (x, y) order
    out = grid_sample(
        I[None],
        phii.flip(0).permute((1, 2, 0))[None],
        align_corners=True,
        **kwargs,
    )

    return out[0]


def build_transform(
    xv: list["torch.Tensor"],
    v: "torch.Tensor",
    A: "torch.Tensor",
    direction: Literal["f", "b"] = "b",
    XJ: list[np.ndarray] | "torch.Tensor" | None = None,
) -> "torch.Tensor":
    """Build transformation from LDDMM output.

    Parameters
    ----------
    xv
        Velocity field sample coordinates.
    v
        Velocity field with shape (nt, H_v, W_v, 2).
    A
        3x3 affine matrix.
    direction
        'f' for forward (source->target), 'b' for backward (target->source).
        Use 'b' for transforming images, 'f' for transforming points.
    XJ
        Target grid coordinates. If None, uses xv grid.

    Returns
    -------
    Sample points with shape (H, W, 2).
    """
    torch = _get_torch()
    A = torch.as_tensor(A)
    if v is not None:
        v = torch.as_tensor(v)

    if XJ is not None:
        if isinstance(XJ, list):
            if XJ[0].ndim == 1:  # Need meshgrid
                XJ = torch.stack(torch.meshgrid([torch.as_tensor(x) for x in XJ], indexing="ij"), -1)
            elif XJ[0].ndim == 2:  # Already meshgrid
                XJ = torch.stack([torch.as_tensor(x) for x in XJ], -1)
            else:
                raise ValueError("Could not understand XJ type")
        XJ = torch.as_tensor(XJ)
    else:
        XJ = torch.stack(torch.meshgrid([torch.as_tensor(x) for x in xv], indexing="ij"), -1)

    if direction == "b":
        Ai = torch.linalg.inv(A)
        Xs = (Ai[:-1, :-1] @ XJ[..., None])[..., 0] + Ai[:-1, -1]
        if v is not None:
            nt = v.shape[0]
            for t in range(nt - 1, -1, -1):
                Xs = Xs + _interp(xv, -v[t].permute(2, 0, 1), Xs.permute(2, 0, 1)).permute(1, 2, 0) / nt
    elif direction == "f":
        Xs = torch.clone(XJ)
        if v is not None:
            nt = v.shape[0]
            for t in range(nt):
                Xs = Xs + _interp(xv, v[t].permute(2, 0, 1), Xs.permute(2, 0, 1)).permute(1, 2, 0) / nt
        Xs = (A[:2, :2] @ Xs[..., None])[..., 0] + A[:2, -1]
    else:
        raise ValueError(f'Direction must be "f" or "b", got {direction}')

    return Xs


def transform_points_source_to_target(
    xv: list["torch.Tensor"],
    v: "torch.Tensor",
    A: "torch.Tensor",
    pointsI: NDArray[np.floating] | "torch.Tensor",
) -> "torch.Tensor":
    """Transform points from source to target space.

    Parameters
    ----------
    xv
        Velocity field sample coordinates.
    v
        Velocity field with shape (nt, H_v, W_v, 2).
    A
        3x3 affine matrix.
    pointsI
        Points to transform with shape (N, 2) in (row, col) order.

    Returns
    -------
    Transformed points with shape (N, 2).
    """
    torch = _get_torch()
    if isinstance(pointsI, torch.Tensor):
        pointsIt = torch.clone(pointsI)
    else:
        pointsIt = torch.tensor(pointsI, dtype=v.dtype, device=v.device)

    A = torch.as_tensor(A, dtype=v.dtype, device=v.device)

    nt = v.shape[0]
    for t in range(nt):
        pointsIt += _interp(xv, v[t].permute(2, 0, 1), pointsIt.T[..., None])[..., 0].T / nt

    pointsIt = (A[:2, :2] @ pointsIt.T + A[:2, -1][..., None]).T
    return pointsIt


def transform_points_target_to_source(
    xv: list["torch.Tensor"],
    v: "torch.Tensor",
    A: "torch.Tensor",
    pointsJ: NDArray[np.floating] | "torch.Tensor",
) -> "torch.Tensor":
    """Transform points from target to source space.

    Parameters
    ----------
    xv
        Velocity field sample coordinates.
    v
        Velocity field with shape (nt, H_v, W_v, 2).
    A
        3x3 affine matrix.
    pointsJ
        Points to transform with shape (N, 2) in (row, col) order.

    Returns
    -------
    Transformed points with shape (N, 2).
    """
    torch = _get_torch()
    if isinstance(pointsJ, torch.Tensor):
        pointsIt = torch.clone(pointsJ)
    else:
        pointsIt = torch.tensor(pointsJ, dtype=v.dtype, device=v.device)

    A = torch.as_tensor(A, dtype=v.dtype, device=v.device)
    Ai = torch.linalg.inv(A)

    pointsIt = (Ai[:2, :2] @ pointsIt.T + Ai[:2, -1][..., None]).T

    nt = v.shape[0]
    for t in range(nt):
        pointsIt += _interp(xv, -v[t].permute(2, 0, 1), pointsIt.T[..., None])[..., 0].T / nt

    return pointsIt


def transform_image_source_to_target(
    xv: list["torch.Tensor"],
    v: "torch.Tensor",
    A: "torch.Tensor",
    xI: list[np.ndarray],
    I: np.ndarray,
    XJ: list[np.ndarray] | None = None,
) -> "torch.Tensor":
    """Transform image from source to target space.

    Parameters
    ----------
    xv
        Velocity field sample coordinates.
    v
        Velocity field.
    A
        Affine matrix.
    xI
        Source image pixel coordinates.
    I
        Source image with shape (C, H, W).
    XJ
        Target grid coordinates.

    Returns
    -------
    Transformed image.
    """
    torch = _get_torch()
    xI = [torch.as_tensor(x) for x in xI]
    I = torch.as_tensor(I)

    phii = build_transform(xv, v, A, direction="b", XJ=XJ)
    phiI = _interp(xI, I, phii.permute(2, 0, 1), padding_mode="border")
    return phiI


def compute_initial_affine(
    xI: NDArray[np.floating],
    yI: NDArray[np.floating],
    xJ: NDArray[np.floating],
    yJ: NDArray[np.floating],
    rotation_deg: float = 0.0,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Compute initial affine transform with rotation.

    Computes an affine transform that:
    1. Rotates source by rotation_deg degrees (clockwise)
    2. Centers rotation about source centroid
    3. Translates to align centroids

    Parameters
    ----------
    xI, yI
        Source coordinates.
    xJ, yJ
        Target coordinates.
    rotation_deg
        Rotation angle in degrees (clockwise). Default: 0.

    Returns
    -------
    A tuple containing:

    - L: 2x2 rotation matrix.
    - T: 2-element translation vector.
    """
    theta = np.radians(-rotation_deg)

    # Rotation matrix
    L = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    # Translation: rotate about source centroid, then translate to target centroid
    mean_I = np.array([np.mean(yI), np.mean(xI)])  # (row, col) = (y, x)
    mean_J = np.array([np.mean(yJ), np.mean(xJ)])

    T = mean_I - L @ mean_I + (mean_J - mean_I)

    return L, T
