"""Rasterization functions for converting point coordinates to density images.

Functions ported from STalign (https://github.com/JEFworks-Lab/STalign).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def normalize(arr: NDArray[np.floating], t_min: float = 0, t_max: float = 1) -> NDArray[np.floating]:
    """Linearly normalize an array to a specified range.

    Parameters
    ----------
    arr
        Array to normalize.
    t_min
        Lower bound of normalization range. Default: 0.
    t_max
        Upper bound of normalization range. Default: 1.

    Returns
    -------
    Normalized array with values in [t_min, t_max].
    """
    diff = t_max - t_min
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    diff_arr = arr_max - arr_min

    if diff_arr == 0:
        return np.full_like(arr, (t_min + t_max) / 2)

    return ((arr - arr_min) / diff_arr * diff) + t_min


def rasterize(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    g: NDArray[np.floating] | None = None,
    dx: float = 30.0,
    blur: float = 1.0,
    expand: float = 1.1,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Rasterize spatial coordinates into a density image.

    Converts point cloud data (cell coordinates) into a regular grid image
    by applying Gaussian kernels at each point location.

    Parameters
    ----------
    x
        X coordinates of points (length N).
    y
        Y coordinates of points (length N).
    g
        Weights for each point. If None, uniform weights are used.
    dx
        Pixel size in coordinate units. Default: 30.0.
    blur
        Gaussian kernel sigma in pixels. Default: 1.0.
    expand
        Factor to expand image bounds beyond data extent. Default: 1.1.

    Returns
    -------
    A tuple containing:

    - X: 1D array of pixel x-coordinates.
    - Y: 1D array of pixel y-coordinates.
    - W: Density image with shape (1, H, W), channels first.

    Examples
    --------
    >>> x = np.random.rand(1000) * 1000
    >>> y = np.random.rand(1000) * 1000
    >>> X, Y, image = rasterize(x, y, dx=30.0, blur=1.5)
    """
    if g is None:
        g = np.ones(len(x))
    else:
        g = np.resize(g, x.size)
        if not (g == 1.0).all():
            g = normalize(g)

    # Compute image bounds
    minx, maxx = np.min(x), np.max(x)
    miny, maxy = np.min(y), np.max(y)

    # Expand bounds
    cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
    wx, wy = (maxx - minx) / 2.0 * expand, (maxy - miny) / 2.0 * expand
    minx, maxx = cx - wx, cx + wx
    miny, maxy = cy - wy, cy + wy

    # Create pixel coordinate arrays
    X_ = np.arange(minx, maxx, dx)
    Y_ = np.arange(miny, maxy, dx)

    # Create meshgrid
    X_mesh = np.stack(np.meshgrid(X_, Y_))  # (2, H, W), xy order

    # Initialize output
    W = np.zeros((X_mesh.shape[1], X_mesh.shape[2], 1))

    # Windowed rasterization (faster)
    maxblur = blur
    r = int(np.ceil(maxblur * 4))
    blur_arr = np.array([blur])

    for x_, y_, g_ in zip(x, y, g):
        # Find pixel indices
        col = int(np.round((x_ - X_[0]) / dx))
        row = int(np.round((y_ - Y_[0]) / dx))

        # Window bounds with boundary conditions
        row0 = max(0, min(row - r, W.shape[0] - 1))
        row1 = max(0, min(row + r, W.shape[0] - 1))
        col0 = max(0, min(col - r, W.shape[1] - 1))
        col1 = max(0, min(col + r, W.shape[1] - 1))

        if row1 <= row0 or col1 <= col0:
            continue

        # Compute Gaussian kernel in window
        window_X = X_mesh[0][row0 : row1 + 1, col0 : col1 + 1, None]
        window_Y = X_mesh[1][row0 : row1 + 1, col0 : col1 + 1, None]

        k = np.exp(-((window_X - x_) ** 2 + (window_Y - y_) ** 2) / (2.0 * (dx * blur_arr * 2) ** 2))
        k_sum = np.sum(k, axis=(0, 1), keepdims=True)
        if k_sum.any():
            k /= k_sum
        k *= g_

        W[row0 : row1 + 1, col0 : col1 + 1, :] += k

    # Transpose to channels-first format
    W = np.abs(W)
    W = W.transpose((-1, 0, 1))  # (1, H, W)

    return X_, Y_, W
