"""
STalign Port for Squidpy Integration
=====================================

This file contains all necessary functions from STalign for spatial transcriptomics
alignment using LDDMM (Large Deformation Diffeomorphic Metric Mapping).

Target namespace: sq.experimental.align_spatial() 
(experimental first, promote to sq.tl when stable)

Dependencies:
- numpy
- torch>=2.0.0
- tqdm (for progress bars)

Usage:
    Copy this file to squidpy/experimental/ and split into appropriate modules 
    as described in implementation_plan.md.

    For progress/logging in squidpy, replace:
    - tqdm with squidpy's tqdm wrapper (if available)
    - print statements with spatialdata._logging.logger

Original Authors: JEFworks Lab (https://github.com/JEFworks-Lab/STalign)
Ported for Squidpy integration.
"""

import numpy as np
from typing import List, Tuple, Optional, Union, Dict, Any, Literal

# PyTorch imports
import torch
from torch.nn.functional import grid_sample

# Progress bar - in squidpy, replace with spatialdata's tqdm if available
try:
    from tqdm.auto import tqdm
except ImportError:
    # Fallback if tqdm not installed
    def tqdm(iterable, **kwargs):
        return iterable


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def normalize(arr: np.ndarray, t_min: float = 0, t_max: float = 1) -> np.ndarray:
    """Linearly normalize an array to a specified range.
    
    Parameters
    ----------
    arr : np.ndarray
        Array to normalize.
    t_min : float
        Lower bound of normalization range. Default: 0.
    t_max : float
        Upper bound of normalization range. Default: 1.
    
    Returns
    -------
    np.ndarray
        Normalized array with values in [t_min, t_max].
    """
    diff = t_max - t_min
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    diff_arr = arr_max - arr_min
    
    if diff_arr == 0:
        return np.full_like(arr, (t_min + t_max) / 2)
    
    return ((arr - arr_min) / diff_arr * diff) + t_min


def clip(I: torch.Tensor) -> torch.Tensor:
    """Clip tensor values to [0, 1] range.
    
    Parameters
    ----------
    I : torch.Tensor
        Input tensor.
    
    Returns
    -------
    torch.Tensor
        Clipped tensor.
    """
    Ic = torch.clone(I)
    Ic[Ic < 0] = 0
    Ic[Ic > 1] = 1
    return Ic


def extent_from_x(xJ: List[torch.Tensor]) -> Tuple[float, float, float, float]:
    """Get matplotlib imshow extent from pixel locations.
    
    Parameters
    ----------
    xJ : list of torch.Tensor
        Location of pixels along each axis (row, col order).
    
    Returns
    -------
    tuple
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
        (xJ[0][0] - dJ[0] / 2.0).item()
    )
    return extentJ


# =============================================================================
# RASTERIZATION FUNCTIONS
# =============================================================================

def rasterize(
    x: np.ndarray,
    y: np.ndarray,
    g: Optional[np.ndarray] = None,
    dx: float = 30.0,
    blur: float = 1.0,
    expand: float = 1.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rasterize spatial coordinates into a density image.
    
    Converts point cloud data (cell coordinates) into a regular grid image
    by applying Gaussian kernels at each point location.
    
    Parameters
    ----------
    x : np.ndarray
        X coordinates of points (length N).
    y : np.ndarray
        Y coordinates of points (length N).
    g : np.ndarray, optional
        Weights for each point. If None, uniform weights are used.
    dx : float
        Pixel size in coordinate units. Default: 30.0.
    blur : float
        Gaussian kernel sigma in pixels. Default: 1.0.
    expand : float
        Factor to expand image bounds beyond data extent. Default: 1.1.
    
    Returns
    -------
    X : np.ndarray
        1D array of pixel x-coordinates.
    Y : np.ndarray
        1D array of pixel y-coordinates.
    W : np.ndarray
        Density image with shape (1, H, W), channels first.
    
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
        window_X = X_mesh[0][row0:row1+1, col0:col1+1, None]
        window_Y = X_mesh[1][row0:row1+1, col0:col1+1, None]
        
        k = np.exp(-((window_X - x_)**2 + (window_Y - y_)**2) / (2.0 * (dx * blur_arr * 2)**2))
        k_sum = np.sum(k, axis=(0, 1), keepdims=True)
        if k_sum.any():
            k /= k_sum
        k *= g_
        
        W[row0:row1+1, col0:col1+1, :] += k
    
    # Transpose to channels-first format
    W = np.abs(W)
    W = W.transpose((-1, 0, 1))  # (1, H, W)
    
    return X_, Y_, W


# =============================================================================
# INTERPOLATION FUNCTIONS
# =============================================================================

def interp(
    x: List[torch.Tensor],
    I: torch.Tensor,
    phii: torch.Tensor,
    **kwargs
) -> torch.Tensor:
    """Interpolate 2D image at specified coordinates.
    
    Uses PyTorch's grid_sample for bilinear interpolation.
    
    Parameters
    ----------
    x : list of torch.Tensor
        List of 1D tensors with pixel locations along each axis (row, col order).
    I : torch.Tensor
        Image tensor with shape (C, H, W).
    phii : torch.Tensor
        Sample coordinates with shape (2, H_out, W_out).
    **kwargs
        Additional arguments passed to torch.nn.functional.grid_sample.
    
    Returns
    -------
    torch.Tensor
        Interpolated image with shape (C, H_out, W_out).
    """
    I = torch.as_tensor(I)
    phii = torch.as_tensor(phii)
    phii = torch.clone(phii)
    
    # Normalize coordinates to [-1, 1] range
    for i in range(2):
        phii[i] -= x[i][0]
        phii[i] /= x[i][-1] - x[i][0]
    phii = phii * 2.0 - 1.0
    
    # grid_sample expects (N, H, W, 2) with (x, y) order
    # Our phii is (2, H, W) with (row, col) order, so flip and permute
    out = grid_sample(
        I[None],  # Add batch dimension
        phii.flip(0).permute((1, 2, 0))[None],  # (1, H, W, 2)
        align_corners=True,
        **kwargs
    )
    
    return out[0]  # Remove batch dimension


# =============================================================================
# DIFFEOMORPHISM FUNCTIONS
# =============================================================================

def v_to_phii(xv: List[torch.Tensor], v: torch.Tensor) -> torch.Tensor:
    """Integrate velocity field to get diffeomorphism (inverse map).
    
    Parameters
    ----------
    xv : list of torch.Tensor
        List of 1D tensors with sample point locations.
    v : torch.Tensor
        Velocity field with shape (nt, 2, H, W).
    
    Returns
    -------
    torch.Tensor
        Inverse map (diffeomorphism) with shape (2, H, W).
    """
    XV = torch.stack(torch.meshgrid(xv, indexing='ij'))
    phii = torch.clone(XV)
    dt = 1.0 / v.shape[0]
    
    for t in range(v.shape[0]):
        Xs = XV - v[t] * dt
        phii = interp(xv, phii - XV, Xs) + Xs
    
    return phii


def to_A(L: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """Convert linear transform and translation to affine matrix.
    
    Parameters
    ----------
    L : torch.Tensor
        2x2 linear transform matrix.
    T : torch.Tensor
        2-element translation vector.
    
    Returns
    -------
    torch.Tensor
        3x3 affine transformation matrix.
    """
    O = torch.tensor([0., 0., 1.], device=L.device, dtype=L.dtype)
    A = torch.cat((torch.cat((L, T[:, None]), 1), O[None]))
    return A


# =============================================================================
# AFFINE FROM POINTS
# =============================================================================

def L_T_from_points(
    pointsI: np.ndarray,
    pointsJ: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute affine transformation from corresponding points.
    
    Estimates the best-fit affine transform that maps pointsI to pointsJ.
    For 2 points, computes translation only. For 3+ points, computes full affine.
    
    Parameters
    ----------
    pointsI : np.ndarray
        Source points with shape (N, 2) in (row, col) order.
    pointsJ : np.ndarray
        Target points with shape (N, 2) in (row, col) order.
    
    Returns
    -------
    L : np.ndarray
        2x2 linear transform matrix.
    T : np.ndarray
        2-element translation vector.
    
    Raises
    ------
    ValueError
        If inputs have inconsistent shapes.
    """
    if pointsI is None or pointsJ is None:
        raise ValueError('Points cannot be None')
    
    nI, nJ = pointsI.shape[0], pointsJ.shape[0]
    
    if nI != nJ:
        raise ValueError(f'Number of source points ({nI}) != target points ({nJ})')
    if pointsI.shape[1] != 2 or pointsJ.shape[1] != 2:
        raise ValueError('Points must have shape (N, 2)')
    
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


# =============================================================================
# MAIN LDDMM ALGORITHM
# =============================================================================

def LDDMM(
    xI: List[np.ndarray],
    I: np.ndarray,
    xJ: List[np.ndarray],
    J: np.ndarray,
    pointsI: Optional[np.ndarray] = None,
    pointsJ: Optional[np.ndarray] = None,
    L: Optional[np.ndarray] = None,
    T: Optional[np.ndarray] = None,
    A: Optional[np.ndarray] = None,
    v: Optional[np.ndarray] = None,
    xv: Optional[List[np.ndarray]] = None,
    a: float = 500.0,
    p: float = 2.0,
    expand: float = 2.0,
    nt: int = 3,
    niter: int = 5000,
    diffeo_start: int = 0,
    epL: float = 2e-8,
    epT: float = 2e-1,
    epV: float = 2e3,
    sigmaM: float = 1.0,
    sigmaB: float = 2.0,
    sigmaA: float = 5.0,
    sigmaR: float = 5e5,
    sigmaP: float = 2e1,
    device: str = 'cpu',
    dtype: torch.dtype = torch.float64,
    muB: Optional[torch.Tensor] = None,
    muA: Optional[torch.Tensor] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run LDDMM diffeomorphic registration between two images.
    
    Jointly estimates an affine transform A and a diffeomorphism phi.
    The full map is: x -> A(phi(x))
    
    Parameters
    ----------
    xI : list of np.ndarray
        Pixel locations in source image I (row, col arrays).
    I : np.ndarray
        Source image with shape (C, H_I, W_I).
    xJ : list of np.ndarray
        Pixel locations in target image J.
    J : np.ndarray
        Target image with shape (C, H_J, W_J).
    pointsI : np.ndarray, optional
        Source landmark points with shape (N, 2) in (row, col) order.
    pointsJ : np.ndarray, optional
        Target landmark points with shape (N, 2).
    L : np.ndarray, optional
        Initial 2x2 linear transform. Default: identity.
    T : np.ndarray, optional
        Initial 2-element translation. Default: zero.
    A : np.ndarray, optional
        Initial 3x3 affine matrix. Overrides L and T if provided.
    v : np.ndarray, optional
        Initial velocity field.
    xv : list of np.ndarray, optional
        Velocity field sample locations.
    a : float
        Smoothness scale of velocity field. Default: 500.0.
    p : float
        Power of Laplacian regularization. Default: 2.0.
    expand : float
        Factor to expand velocity field domain. Default: 2.0.
    nt : int
        Number of time steps for velocity integration. Default: 3.
    niter : int
        Number of optimization iterations. Default: 5000.
    diffeo_start : int
        Iteration to begin optimizing velocity field. Default: 0.
    epL : float
        Learning rate for linear transform. Default: 2e-8.
    epT : float
        Learning rate for translation. Default: 2e-1.
    epV : float
        Learning rate for velocity field. Default: 2e3.
    sigmaM : float
        Matching term weight (smaller = more accurate). Default: 1.0.
    sigmaB : float
        Background term weight. Default: 2.0.
    sigmaA : float
        Artifact term weight. Default: 5.0.
    sigmaR : float
        Regularization weight (smaller = smoother). Default: 5e5.
    sigmaP : float
        Point matching weight. Default: 2e1.
    device : str
        PyTorch device. Default: 'cpu'.
    dtype : torch.dtype
        Data type for computation. Default: torch.float64.
    muB : torch.Tensor, optional
        Background mean. Estimated if not provided.
    muA : torch.Tensor, optional
        Artifact mean. Estimated if not provided.
    verbose : bool
        Show tqdm progress bar. Default: True.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'A': Affine transform (3x3 tensor)
        - 'v': Velocity field (nt, H_v, W_v, 2 tensor)
        - 'xv': Velocity field coordinates (list of tensors)
        - 'WM': Matching weights (H, W tensor)
        - 'WB': Background weights (H, W tensor)
        - 'WA': Artifact weights (H, W tensor)
        - 'Esave': Loss history (list)
    """
    # Initialize affine transform
    if A is not None:
        if L is not None or T is not None:
            raise ValueError('Cannot specify both A and L/T')
        L = torch.tensor(A[:2, :2], device=device, dtype=dtype, requires_grad=True)
        T = torch.tensor(A[:2, -1], device=device, dtype=dtype, requires_grad=True)
    else:
        if L is None:
            L = torch.eye(2, device=device, dtype=dtype, requires_grad=True)
        else:
            L = torch.tensor(L, device=device, dtype=dtype, requires_grad=True)
        if T is None:
            T = torch.zeros(2, device=device, dtype=dtype, requires_grad=True)
        else:
            T = torch.tensor(T, device=device, dtype=dtype, requires_grad=True)
    
    # Convert images to torch
    I = torch.tensor(I, device=device, dtype=dtype)
    J = torch.tensor(J, device=device, dtype=dtype)
    
    # Initialize velocity field
    if v is not None and xv is not None:
        v = torch.tensor(v, device=device, dtype=dtype, requires_grad=True)
        xv = [torch.tensor(x, device=device, dtype=dtype) for x in xv]
        XV = torch.stack(torch.meshgrid(xv, indexing='ij'), -1)
        nt = v.shape[0]
    elif v is None and xv is None:
        minv = torch.as_tensor([x[0] for x in xI], device=device, dtype=dtype)
        maxv = torch.as_tensor([x[-1] for x in xI], device=device, dtype=dtype)
        minv, maxv = (minv + maxv) * 0.5 + 0.5 * torch.tensor([-1.0, 1.0], device=device, dtype=dtype)[..., None] * (maxv - minv) * expand
        xv = [torch.arange(m, M, a * 0.5, device=device, dtype=dtype) for m, M in zip(minv, maxv)]
        XV = torch.stack(torch.meshgrid(xv, indexing='ij'), -1)
        v = torch.zeros((nt, XV.shape[0], XV.shape[1], XV.shape[2]), device=device, dtype=dtype, requires_grad=True)
    else:
        raise ValueError('Must provide both xv and v, or neither')
    
    dv = torch.as_tensor([x[1] - x[0] for x in xv], device=device, dtype=dtype)
    
    # Build smoothing kernel in frequency domain
    fv = [torch.arange(n, device=device, dtype=dtype) / n / d for n, d in zip(XV.shape, dv)]
    FV = torch.stack(torch.meshgrid(fv, indexing='ij'), -1)
    LL = (1.0 + 2.0 * a**2 * torch.sum((1.0 - torch.cos(2.0 * np.pi * FV * dv)) / dv**2, -1))**(p * 2.0)
    K = 1.0 / LL
    DV = torch.prod(dv)
    
    # Initialize mixture weights
    WM = torch.ones(J[0].shape, dtype=J.dtype, device=J.device) * 0.5
    WB = torch.ones(J[0].shape, dtype=J.dtype, device=J.device) * 0.4
    WA = torch.ones(J[0].shape, dtype=J.dtype, device=J.device) * 0.1
    
    # Initialize landmark points
    if pointsI is None and pointsJ is None:
        pointsI = torch.zeros((0, 2), device=J.device, dtype=J.dtype)
        pointsJ = torch.zeros((0, 2), device=J.device, dtype=J.dtype)
    elif (pointsI is None) != (pointsJ is None):
        raise ValueError('Must specify both pointsI and pointsJ, or neither')
    else:
        pointsI = torch.tensor(pointsI, device=J.device, dtype=J.dtype)
        pointsJ = torch.tensor(pointsJ, device=J.device, dtype=J.dtype)
    
    # Convert coordinate arrays
    xI = [torch.tensor(x, device=device, dtype=dtype) for x in xI]
    xJ = [torch.tensor(x, device=device, dtype=dtype) for x in xJ]
    XI = torch.stack(torch.meshgrid(*xI, indexing='ij'), -1)
    XJ = torch.stack(torch.meshgrid(*xJ, indexing='ij'), -1)
    
    # Mixture model estimation flags
    estimate_muA = muA is None
    estimate_muB = muB is None
    
    Esave = []
    
    # Zero gradients if they exist
    if L.grad is not None:
        L.grad.zero_()
    if T.grad is not None:
        T.grad.zero_()
    
    # Progress bar
    iterator = tqdm(range(niter), desc="LDDMM", disable=not verbose)
    
    for it in iterator:
        # Build affine matrix
        A_mat = to_A(L, T)
        Ai = torch.linalg.inv(A_mat)
        
        # Transform sample points (inverse affine)
        Xs = (Ai[:2, :2] @ XJ[..., None])[..., 0] + Ai[:2, -1]
        
        # Apply diffeomorphism (backward integration)
        for t in range(nt - 1, -1, -1):
            Xs = Xs + interp(xv, -v[t].permute(2, 0, 1), Xs.permute(2, 0, 1)).permute(1, 2, 0) / nt
        
        # Transform landmark points (forward)
        pointsIt = torch.clone(pointsI)
        if pointsIt.shape[0] > 0:
            for t in range(nt):
                pointsIt += interp(xv, v[t].permute(2, 0, 1), pointsIt.T[..., None])[..., 0].T / nt
            pointsIt = (A_mat[:2, :2] @ pointsIt.T + A_mat[:2, -1][..., None]).T
        
        # Transform source image
        AI = interp(xI, I, Xs.permute(2, 0, 1), padding_mode="border")
        
        # Contrast correction
        B = torch.ones(1 + AI.shape[0], AI.shape[1] * AI.shape[2], device=AI.device, dtype=AI.dtype)
        B[1:AI.shape[0] + 1] = AI.reshape(AI.shape[0], -1)
        
        with torch.no_grad():
            BB = B @ (B * WM.ravel()).T
            BJ = B @ ((J * WM).reshape(J.shape[0], J.shape[1] * J.shape[2])).T
            small = 0.1
            coeffs = torch.linalg.solve(BB + small * torch.eye(BB.shape[0], device=BB.device, dtype=BB.dtype), BJ)
        fAI = ((B.T @ coeffs).T).reshape(J.shape)
        
        # Compute loss
        EM = torch.sum((fAI - J)**2 * WM) / 2.0 / sigmaM**2
        ER = torch.sum(torch.sum(torch.abs(torch.fft.fftn(v, dim=(1, 2)))**2, dim=(0, -1)) * LL) * DV / 2.0 / v.shape[1] / v.shape[2] / sigmaR**2
        E = EM + ER
        tosave = [E.item(), EM.item(), ER.item()]
        
        if pointsIt.shape[0] > 0:
            EP = torch.sum((pointsIt - pointsJ)**2) / 2.0 / sigmaP**2
            E += EP
            tosave.append(EP.item())
        
        Esave.append(tosave)
        
        # Update progress bar
        if verbose:
            loss_info = {"E": f"{E.item():.3f}", "EM": f"{EM.item():.3f}"}
            if pointsIt.shape[0] > 0:
                loss_info["EP"] = f"{tosave[-1]:.3f}"
            iterator.set_postfix(loss_info)
        
        # Gradient descent step
        E.backward()
        
        with torch.no_grad():
            # Update affine (reduce learning rate after diffeo_start)
            lr_scale = 1.0 / (1.0 + (it >= diffeo_start) * 9)
            L -= epL * lr_scale * L.grad
            T -= epT * lr_scale * T.grad
            
            L.grad.zero_()
            T.grad.zero_()
            
            # Update velocity field
            vgrad = v.grad
            vgrad = torch.fft.ifftn(torch.fft.fftn(vgrad, dim=(1, 2)) * K[..., None], dim=(1, 2)).real
            if it >= diffeo_start:
                v -= vgrad * epV
            v.grad.zero_()
        
        # Update mixture weights (every 5 iterations, starting at iteration 50)
        if it % 5 == 0:
            with torch.no_grad():
                if estimate_muA:
                    muA = torch.sum(WA * J, dim=(-1, -2)) / torch.sum(WA)
                if estimate_muB:
                    muB = torch.sum(WB * J, dim=(-1, -2)) / torch.sum(WB)
                
                if it >= 50:
                    W = torch.stack((WM, WA, WB))
                    pi = torch.sum(W, dim=(1, 2))
                    pi += torch.max(pi) * 1e-6
                    pi /= torch.sum(pi)
                    
                    # E-step: update weights
                    WM = pi[0] * torch.exp(-torch.sum((fAI - J)**2, 0) / 2.0 / sigmaM**2) / np.sqrt(2.0 * np.pi * sigmaM**2)**J.shape[0]
                    WA = pi[1] * torch.exp(-torch.sum((muA[..., None, None] - J)**2, 0) / 2.0 / sigmaA**2) / np.sqrt(2.0 * np.pi * sigmaA**2)**J.shape[0]
                    WB = pi[2] * torch.exp(-torch.sum((muB[..., None, None] - J)**2, 0) / 2.0 / sigmaB**2) / np.sqrt(2.0 * np.pi * sigmaB**2)**J.shape[0]
                    WS = WM + WB + WA
                    WS += torch.max(WS) * 1e-6
                    WM /= WS
                    WB /= WS
                    WA /= WS
    
    # Note: In squidpy, replace print with spatialdata._logging.logger.info()
    if verbose:
        print(f"LDDMM completed. Final loss: {Esave[-1][0]:.4f}")
    
    return {
        'A': A_mat.clone().detach(),
        'v': v.clone().detach(),
        'xv': xv,
        'WM': WM.clone().detach(),
        'WB': WB.clone().detach(),
        'WA': WA.clone().detach(),
        'Esave': Esave,
    }


# =============================================================================
# TRANSFORMATION FUNCTIONS
# =============================================================================

def build_transform(
    xv: List[torch.Tensor],
    v: torch.Tensor,
    A: torch.Tensor,
    direction: Literal['f', 'b'] = 'b',
    XJ: Optional[Union[List[np.ndarray], torch.Tensor]] = None
) -> torch.Tensor:
    """Build transformation from LDDMM output.
    
    Parameters
    ----------
    xv : list of torch.Tensor
        Velocity field sample coordinates.
    v : torch.Tensor
        Velocity field with shape (nt, H_v, W_v, 2).
    A : torch.Tensor
        3x3 affine matrix.
    direction : str
        'f' for forward (source→target), 'b' for backward (target→source).
        Use 'b' for transforming images, 'f' for transforming points.
    XJ : optional
        Target grid coordinates. If None, uses xv grid.
    
    Returns
    -------
    torch.Tensor
        Sample points with shape (H, W, 2).
    """
    A = torch.as_tensor(A)
    if v is not None:
        v = torch.as_tensor(v)
    
    if XJ is not None:
        if isinstance(XJ, list):
            if XJ[0].ndim == 1:  # Need meshgrid
                XJ = torch.stack(torch.meshgrid([torch.as_tensor(x) for x in XJ], indexing='ij'), -1)
            elif XJ[0].ndim == 2:  # Already meshgrid
                XJ = torch.stack([torch.as_tensor(x) for x in XJ], -1)
            else:
                raise ValueError('Could not understand XJ type')
        XJ = torch.as_tensor(XJ)
    else:
        XJ = torch.stack(torch.meshgrid([torch.as_tensor(x) for x in xv], indexing='ij'), -1)
    
    if direction == 'b':
        Ai = torch.linalg.inv(A)
        Xs = (Ai[:-1, :-1] @ XJ[..., None])[..., 0] + Ai[:-1, -1]
        if v is not None:
            nt = v.shape[0]
            for t in range(nt - 1, -1, -1):
                Xs = Xs + interp(xv, -v[t].permute(2, 0, 1), Xs.permute(2, 0, 1)).permute(1, 2, 0) / nt
    elif direction == 'f':
        Xs = torch.clone(XJ)
        if v is not None:
            nt = v.shape[0]
            for t in range(nt):
                Xs = Xs + interp(xv, v[t].permute(2, 0, 1), Xs.permute(2, 0, 1)).permute(1, 2, 0) / nt
        Xs = (A[:2, :2] @ Xs[..., None])[..., 0] + A[:2, -1]
    else:
        raise ValueError(f'Direction must be "f" or "b", got {direction}')
    
    return Xs


def transform_points_source_to_target(
    xv: List[torch.Tensor],
    v: torch.Tensor,
    A: torch.Tensor,
    pointsI: Union[np.ndarray, torch.Tensor]
) -> torch.Tensor:
    """Transform points from source to target space.
    
    Parameters
    ----------
    xv : list of torch.Tensor
        Velocity field sample coordinates.
    v : torch.Tensor
        Velocity field with shape (nt, H_v, W_v, 2).
    A : torch.Tensor
        3x3 affine matrix.
    pointsI : np.ndarray or torch.Tensor
        Points to transform with shape (N, 2) in (row, col) order.
    
    Returns
    -------
    torch.Tensor
        Transformed points with shape (N, 2).
    """
    if isinstance(pointsI, torch.Tensor):
        pointsIt = torch.clone(pointsI)
    else:
        pointsIt = torch.tensor(pointsI, dtype=v.dtype, device=v.device)
    
    A = torch.as_tensor(A, dtype=v.dtype, device=v.device)
    
    nt = v.shape[0]
    for t in range(nt):
        pointsIt += interp(xv, v[t].permute(2, 0, 1), pointsIt.T[..., None])[..., 0].T / nt
    
    pointsIt = (A[:2, :2] @ pointsIt.T + A[:2, -1][..., None]).T
    return pointsIt


def transform_points_target_to_source(
    xv: List[torch.Tensor],
    v: torch.Tensor,
    A: torch.Tensor,
    pointsJ: Union[np.ndarray, torch.Tensor]
) -> torch.Tensor:
    """Transform points from target to source space.
    
    Parameters
    ----------
    xv : list of torch.Tensor
        Velocity field sample coordinates.
    v : torch.Tensor
        Velocity field with shape (nt, H_v, W_v, 2).
    A : torch.Tensor
        3x3 affine matrix.
    pointsJ : np.ndarray or torch.Tensor
        Points to transform with shape (N, 2) in (row, col) order.
    
    Returns
    -------
    torch.Tensor
        Transformed points with shape (N, 2).
    """
    if isinstance(pointsJ, torch.Tensor):
        pointsIt = torch.clone(pointsJ)
    else:
        pointsIt = torch.tensor(pointsJ, dtype=v.dtype, device=v.device)
    
    A = torch.as_tensor(A, dtype=v.dtype, device=v.device)
    Ai = torch.linalg.inv(A)
    
    pointsIt = (Ai[:2, :2] @ pointsIt.T + Ai[:2, -1][..., None]).T
    
    nt = v.shape[0]
    for t in range(nt):
        pointsIt += interp(xv, -v[t].permute(2, 0, 1), pointsIt.T[..., None])[..., 0].T / nt
    
    return pointsIt


def transform_image_source_to_target(
    xv: List[torch.Tensor],
    v: torch.Tensor,
    A: torch.Tensor,
    xI: List[np.ndarray],
    I: np.ndarray,
    XJ: Optional[List[np.ndarray]] = None
) -> torch.Tensor:
    """Transform image from source to target space.
    
    Parameters
    ----------
    xv : list of torch.Tensor
        Velocity field sample coordinates.
    v : torch.Tensor
        Velocity field.
    A : torch.Tensor
        Affine matrix.
    xI : list of np.ndarray
        Source image pixel coordinates.
    I : np.ndarray
        Source image with shape (C, H, W).
    XJ : list of np.ndarray, optional
        Target grid coordinates.
    
    Returns
    -------
    torch.Tensor
        Transformed image.
    """
    phii = build_transform(xv, v, A, direction='b', XJ=XJ)
    phiI = interp(xI, I, phii.permute(2, 0, 1), padding_mode="border")
    return phiI


# =============================================================================
# HELPER FUNCTIONS FOR INITIAL ALIGNMENT
# =============================================================================

def compute_initial_affine(
    xI: np.ndarray,
    yI: np.ndarray,
    xJ: np.ndarray,
    yJ: np.ndarray,
    rotation_deg: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute initial affine transform with rotation.
    
    Computes an affine transform that:
    1. Rotates source by rotation_deg degrees (clockwise)
    2. Centers rotation about source centroid
    3. Translates to align centroids
    
    Parameters
    ----------
    xI, yI : np.ndarray
        Source coordinates.
    xJ, yJ : np.ndarray
        Target coordinates.
    rotation_deg : float
        Rotation angle in degrees (clockwise). Default: 0.
    
    Returns
    -------
    L : np.ndarray
        2x2 rotation matrix.
    T : np.ndarray
        2-element translation vector.
    """
    theta = np.radians(-rotation_deg)
    
    # Rotation matrix
    L = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    
    # Translation: rotate about source centroid, then translate to target centroid
    mean_I = np.array([np.mean(yI), np.mean(xI)])  # (row, col) = (y, x)
    mean_J = np.array([np.mean(yJ), np.mean(xJ)])
    
    T = mean_I - L @ mean_I + (mean_J - mean_I)
    
    return L, T


def calculate_tre(
    pointsI: np.ndarray,
    pointsJ: np.ndarray
) -> Tuple[float, float]:
    """Calculate Target Registration Error between point sets.
    
    Parameters
    ----------
    pointsI : np.ndarray
        First point set with shape (N, 2).
    pointsJ : np.ndarray
        Second point set with shape (N, 2).
    
    Returns
    -------
    mean_tre : float
        Mean TRE across all points.
    std_tre : float
        Standard deviation of TRE.
    """
    tre_per_point = np.sqrt(np.sum((pointsI - pointsJ)**2, axis=1))
    return np.mean(tre_per_point), np.std(tre_per_point)


# =============================================================================
# HIGH-LEVEL API (for Squidpy integration)
# =============================================================================

def align_coordinates(
    coords_source: np.ndarray,
    coords_target: np.ndarray,
    resolution: float = 30.0,
    blur: float = 1.5,
    niter: int = 2000,
    diffeo_start: int = 100,
    initial_rotation_deg: float = 0.0,
    landmark_source: Optional[np.ndarray] = None,
    landmark_target: Optional[np.ndarray] = None,
    device: str = 'cpu',
    verbose: bool = True,
    **lddmm_kwargs
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """High-level function to align spatial coordinates.
    
    This is the main entry point for aligning spatial transcriptomics data.
    
    Parameters
    ----------
    coords_source : np.ndarray
        Source coordinates with shape (N, 2) in (x, y) order.
    coords_target : np.ndarray
        Target coordinates with shape (M, 2) in (x, y) order.
    resolution : float
        Pixel size for rasterization. Default: 30.0.
    blur : float
        Gaussian blur for rasterization. Default: 1.5.
    niter : int
        Number of LDDMM iterations. Default: 2000.
    diffeo_start : int
        Iteration to start nonlinear optimization. Default: 100.
    initial_rotation_deg : float
        Initial rotation in degrees. Default: 0.
    landmark_source : np.ndarray, optional
        Landmark points in source (N, 2) in (x, y) order.
    landmark_target : np.ndarray, optional
        Corresponding landmark points in target.
    device : str
        PyTorch device. Default: 'cpu'.
    verbose : bool
        Print progress. Default: True.
    **lddmm_kwargs
        Additional arguments passed to LDDMM().
    
    Returns
    -------
    aligned_coords : np.ndarray
        Aligned source coordinates with shape (N, 2) in (x, y) order.
    transform_dict : dict
        Dictionary with transformation parameters for later use.
    
    Examples
    --------
    >>> source_coords = adata_source.obsm['spatial']
    >>> target_coords = adata_target.obsm['spatial']
    >>> aligned, transform = align_coordinates(source_coords, target_coords)
    >>> adata_source.obsm['spatial_aligned'] = aligned
    """
    # Extract x, y from (x, y) format
    xI, yI = coords_source[:, 0], coords_source[:, 1]
    xJ, yJ = coords_target[:, 0], coords_target[:, 1]
    
    # Compute initial affine transform
    if initial_rotation_deg != 0.0:
        L_init, T_init = compute_initial_affine(xI, yI, xJ, yJ, initial_rotation_deg)
    elif landmark_source is not None and landmark_target is not None:
        # Convert landmarks from (x, y) to (row, col) = (y, x)
        pts_I = np.column_stack([landmark_source[:, 1], landmark_source[:, 0]])
        pts_J = np.column_stack([landmark_target[:, 1], landmark_target[:, 0]])
        L_init, T_init = L_T_from_points(pts_I, pts_J)
    else:
        L_init, T_init = None, None
    
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
        print(f"Running LDDMM for {niter} iterations...")
    
    # Run LDDMM
    result = LDDMM(
        [YI, XI], I_rgb,
        [YJ, XJ], J_rgb,
        L=L_init,
        T=T_init,
        niter=niter,
        diffeo_start=diffeo_start,
        device=device,
        verbose=verbose,
        **lddmm_kwargs
    )
    
    # Transform source coordinates
    # Convert to (row, col) = (y, x) for transformation
    points_source_rc = np.column_stack([yI, xI])
    
    points_aligned_rc = transform_points_source_to_target(
        result['xv'],
        result['v'],
        result['A'],
        points_source_rc
    )
    
    # Convert back to numpy and (x, y) order
    points_aligned_rc = points_aligned_rc.cpu().numpy()
    aligned_coords = np.column_stack([points_aligned_rc[:, 1], points_aligned_rc[:, 0]])
    
    # Build transform dictionary for later use
    transform_dict = {
        'A': result['A'].cpu().numpy(),
        'v': result['v'].cpu().numpy(),
        'xv': [x.cpu().numpy() for x in result['xv']],
        'source_resolution': resolution,
        'source_blur': blur,
        'loss_history': result['Esave'],
    }
    
    return aligned_coords, transform_dict


def apply_saved_transform(
    coords: np.ndarray,
    transform_dict: Dict[str, Any],
    direction: Literal['source_to_target', 'target_to_source'] = 'source_to_target'
) -> np.ndarray:
    """Apply a saved transformation to new coordinates.
    
    Parameters
    ----------
    coords : np.ndarray
        Coordinates with shape (N, 2) in (x, y) order.
    transform_dict : dict
        Transformation dictionary from align_coordinates().
    direction : str
        Direction of transformation.
    
    Returns
    -------
    np.ndarray
        Transformed coordinates with shape (N, 2) in (x, y) order.
    """
    # Convert to torch
    A = torch.tensor(transform_dict['A'])
    v = torch.tensor(transform_dict['v'])
    xv = [torch.tensor(x) for x in transform_dict['xv']]
    
    # Convert coordinates to (row, col) = (y, x)
    coords_rc = np.column_stack([coords[:, 1], coords[:, 0]])
    
    if direction == 'source_to_target':
        transformed_rc = transform_points_source_to_target(xv, v, A, coords_rc)
    else:
        transformed_rc = transform_points_target_to_source(xv, v, A, coords_rc)
    
    # Convert back to (x, y)
    transformed_rc = transformed_rc.cpu().numpy()
    return np.column_stack([transformed_rc[:, 1], transformed_rc[:, 0]])


# =============================================================================
# MODULE TEST
# =============================================================================

if __name__ == '__main__':
    # Quick test with synthetic data
    print("Testing STalign port...")
    
    # Create synthetic circular point clouds
    np.random.seed(42)
    n_points = 500
    
    # Source: circle
    theta = np.random.uniform(0, 2*np.pi, n_points)
    r = np.random.uniform(0, 100, n_points)
    source = np.column_stack([
        r * np.cos(theta) + 500,
        r * np.sin(theta) + 500
    ])
    
    # Target: rotated and translated circle
    theta_rot = np.radians(30)
    rotation = np.array([
        [np.cos(theta_rot), -np.sin(theta_rot)],
        [np.sin(theta_rot), np.cos(theta_rot)]
    ])
    target = (rotation @ (source - 500).T).T + 520
    
    print(f"Source shape: {source.shape}")
    print(f"Target shape: {target.shape}")
    
    # Test rasterization
    X, Y, img = rasterize(source[:, 0], source[:, 1], dx=10, blur=1.0)
    print(f"Rasterized image shape: {img.shape}")
    
    # Test alignment (small number of iterations for quick test)
    print("\nRunning alignment test (100 iterations)...")
    aligned, transform = align_coordinates(
        source, target,
        resolution=10.0,
        blur=1.0,
        niter=100,
        diffeo_start=50,
        verbose=True
    )
    
    print(f"\nAligned coords shape: {aligned.shape}")
    print(f"Transform keys: {list(transform.keys())}")
    
    # Calculate alignment error
    mean_error = np.mean(np.sqrt(np.sum((aligned - target)**2, axis=1)))
    print(f"Mean alignment error: {mean_error:.2f} units")
    
    print("\nTest completed successfully!")
