"""Core LDDMM algorithm for diffeomorphic registration.

Functions ported from STalign (https://github.com/JEFworks-Lab/STalign).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from tqdm.auto import tqdm

if TYPE_CHECKING:
    import torch


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


def clip(I: "torch.Tensor") -> "torch.Tensor":
    """Clip tensor values to [0, 1] range.

    Parameters
    ----------
    I
        Input tensor.

    Returns
    -------
    Clipped tensor.
    """
    torch = _get_torch()
    Ic = torch.clone(I)
    Ic[Ic < 0] = 0
    Ic[Ic > 1] = 1
    return Ic


def interp(
    x: list["torch.Tensor"],
    I: "torch.Tensor",
    phii: "torch.Tensor",
    **kwargs: Any,
) -> "torch.Tensor":
    """Interpolate 2D image at specified coordinates.

    Uses PyTorch's grid_sample for bilinear interpolation.

    Parameters
    ----------
    x
        List of 1D tensors with pixel locations along each axis (row, col order).
    I
        Image tensor with shape (C, H, W).
    phii
        Sample coordinates with shape (2, H_out, W_out).
    **kwargs
        Additional arguments passed to torch.nn.functional.grid_sample.

    Returns
    -------
    Interpolated image with shape (C, H_out, W_out).
    """
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
    # Our phii is (2, H, W) with (row, col) order, so flip and permute
    out = grid_sample(
        I[None],  # Add batch dimension
        phii.flip(0).permute((1, 2, 0))[None],  # (1, H, W, 2)
        align_corners=True,
        **kwargs,
    )

    return out[0]  # Remove batch dimension


def v_to_phii(xv: list["torch.Tensor"], v: "torch.Tensor") -> "torch.Tensor":
    """Integrate velocity field to get diffeomorphism (inverse map).

    Parameters
    ----------
    xv
        List of 1D tensors with sample point locations.
    v
        Velocity field with shape (nt, 2, H, W).

    Returns
    -------
    Inverse map (diffeomorphism) with shape (2, H, W).
    """
    torch = _get_torch()
    XV = torch.stack(torch.meshgrid(xv, indexing="ij"))
    phii = torch.clone(XV)
    dt = 1.0 / v.shape[0]

    for t in range(v.shape[0]):
        Xs = XV - v[t] * dt
        phii = interp(xv, phii - XV, Xs) + Xs

    return phii


def to_A(L: "torch.Tensor", T: "torch.Tensor") -> "torch.Tensor":
    """Convert linear transform and translation to affine matrix.

    Parameters
    ----------
    L
        2x2 linear transform matrix.
    T
        2-element translation vector.

    Returns
    -------
    3x3 affine transformation matrix.
    """
    torch = _get_torch()
    O = torch.tensor([0.0, 0.0, 1.0], device=L.device, dtype=L.dtype)
    A = torch.cat((torch.cat((L, T[:, None]), 1), O[None]))
    return A


def LDDMM(
    xI: list[np.ndarray],
    I: np.ndarray,
    xJ: list[np.ndarray],
    J: np.ndarray,
    pointsI: np.ndarray | None = None,
    pointsJ: np.ndarray | None = None,
    L: np.ndarray | None = None,
    T: np.ndarray | None = None,
    A: np.ndarray | None = None,
    v: np.ndarray | None = None,
    xv: list[np.ndarray] | None = None,
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
    device: str = "cpu",
    dtype: "torch.dtype | None" = None,
    muB: "torch.Tensor | None" = None,
    muA: "torch.Tensor | None" = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run LDDMM diffeomorphic registration between two images.

    Jointly estimates an affine transform A and a diffeomorphism phi.
    The full map is: x -> A(phi(x))

    Parameters
    ----------
    xI
        Pixel locations in source image I (row, col arrays).
    I
        Source image with shape (C, H_I, W_I).
    xJ
        Pixel locations in target image J.
    J
        Target image with shape (C, H_J, W_J).
    pointsI
        Source landmark points with shape (N, 2) in (row, col) order.
    pointsJ
        Target landmark points with shape (N, 2).
    L
        Initial 2x2 linear transform. Default: identity.
    T
        Initial 2-element translation. Default: zero.
    A
        Initial 3x3 affine matrix. Overrides L and T if provided.
    v
        Initial velocity field.
    xv
        Velocity field sample locations.
    a
        Smoothness scale of velocity field. Default: 500.0.
    p
        Power of Laplacian regularization. Default: 2.0.
    expand
        Factor to expand velocity field domain. Default: 2.0.
    nt
        Number of time steps for velocity integration. Default: 3.
    niter
        Number of optimization iterations. Default: 5000.
    diffeo_start
        Iteration to begin optimizing velocity field. Default: 0.
    epL
        Learning rate for linear transform. Default: 2e-8.
    epT
        Learning rate for translation. Default: 2e-1.
    epV
        Learning rate for velocity field. Default: 2e3.
    sigmaM
        Matching term weight (smaller = more accurate). Default: 1.0.
    sigmaB
        Background term weight. Default: 2.0.
    sigmaA
        Artifact term weight. Default: 5.0.
    sigmaR
        Regularization weight (smaller = smoother). Default: 5e5.
    sigmaP
        Point matching weight. Default: 2e1.
    device
        PyTorch device. Default: 'cpu'.
    dtype
        Data type for computation. Default: torch.float64.
    muB
        Background mean. Estimated if not provided.
    muA
        Artifact mean. Estimated if not provided.
    verbose
        Show tqdm progress bar. Default: True.

    Returns
    -------
    Dictionary containing:

    - 'A': Affine transform (3x3 tensor)
    - 'v': Velocity field (nt, H_v, W_v, 2 tensor)
    - 'xv': Velocity field coordinates (list of tensors)
    - 'WM': Matching weights (H, W tensor)
    - 'WB': Background weights (H, W tensor)
    - 'WA': Artifact weights (H, W tensor)
    - 'Esave': Loss history (list)
    """
    torch = _get_torch()
    if dtype is None:
        dtype = torch.float64

    # Initialize affine transform
    if A is not None:
        if L is not None or T is not None:
            raise ValueError("Cannot specify both A and L/T")
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
        XV = torch.stack(torch.meshgrid(xv, indexing="ij"), -1)
        nt = v.shape[0]
    elif v is None and xv is None:
        minv = torch.as_tensor([x[0] for x in xI], device=device, dtype=dtype)
        maxv = torch.as_tensor([x[-1] for x in xI], device=device, dtype=dtype)
        minv, maxv = (minv + maxv) * 0.5 + 0.5 * torch.tensor([-1.0, 1.0], device=device, dtype=dtype)[
            ..., None
        ] * (maxv - minv) * expand

        # Compute step size, ensuring at least 2 grid points per dimension
        step = a * 0.5
        image_extent = maxv - minv
        # If step is too large, reduce it to ensure minimum grid size
        for i in range(len(minv)):
            extent_i = (image_extent[i]).item()
            if extent_i > 0 and extent_i / step < 2:
                step = min(step, extent_i / 3)  # Ensure at least 3 points

        xv = [torch.arange(m, M, step, device=device, dtype=dtype) for m, M in zip(minv, maxv)]

        # Ensure at least 2 points in each dimension
        for i in range(len(xv)):
            if len(xv[i]) < 2:
                xv[i] = torch.linspace(minv[i].item(), maxv[i].item(), 3, device=device, dtype=dtype)

        XV = torch.stack(torch.meshgrid(xv, indexing="ij"), -1)
        v = torch.zeros((nt, XV.shape[0], XV.shape[1], XV.shape[2]), device=device, dtype=dtype, requires_grad=True)
    else:
        raise ValueError("Must provide both xv and v, or neither")

    # Compute grid spacing - handle edge case of 1-element grids
    dv = torch.as_tensor(
        [x[1] - x[0] if len(x) > 1 else torch.tensor(1.0, device=device, dtype=dtype) for x in xv],
        device=device,
        dtype=dtype,
    )

    # Build smoothing kernel in frequency domain
    fv = [torch.arange(n, device=device, dtype=dtype) / n / d for n, d in zip(XV.shape, dv)]
    FV = torch.stack(torch.meshgrid(fv, indexing="ij"), -1)
    LL = (1.0 + 2.0 * a**2 * torch.sum((1.0 - torch.cos(2.0 * np.pi * FV * dv)) / dv**2, -1)) ** (p * 2.0)
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
        raise ValueError("Must specify both pointsI and pointsJ, or neither")
    else:
        pointsI = torch.tensor(pointsI, device=J.device, dtype=J.dtype)
        pointsJ = torch.tensor(pointsJ, device=J.device, dtype=J.dtype)

    # Convert coordinate arrays
    xI = [torch.tensor(x, device=device, dtype=dtype) for x in xI]
    xJ = [torch.tensor(x, device=device, dtype=dtype) for x in xJ]
    XI = torch.stack(torch.meshgrid(*xI, indexing="ij"), -1)
    XJ = torch.stack(torch.meshgrid(*xJ, indexing="ij"), -1)

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
        B[1 : AI.shape[0] + 1] = AI.reshape(AI.shape[0], -1)

        with torch.no_grad():
            BB = B @ (B * WM.ravel()).T
            BJ = B @ ((J * WM).reshape(J.shape[0], J.shape[1] * J.shape[2])).T
            small = 0.1
            coeffs = torch.linalg.solve(BB + small * torch.eye(BB.shape[0], device=BB.device, dtype=BB.dtype), BJ)
        fAI = ((B.T @ coeffs).T).reshape(J.shape)

        # Compute loss
        EM = torch.sum((fAI - J) ** 2 * WM) / 2.0 / sigmaM**2
        ER = (
            torch.sum(torch.sum(torch.abs(torch.fft.fftn(v, dim=(1, 2))) ** 2, dim=(0, -1)) * LL)
            * DV
            / 2.0
            / v.shape[1]
            / v.shape[2]
            / sigmaR**2
        )
        E = EM + ER
        tosave = [E.item(), EM.item(), ER.item()]

        if pointsIt.shape[0] > 0:
            EP = torch.sum((pointsIt - pointsJ) ** 2) / 2.0 / sigmaP**2
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
                    WM = (
                        pi[0]
                        * torch.exp(-torch.sum((fAI - J) ** 2, 0) / 2.0 / sigmaM**2)
                        / np.sqrt(2.0 * np.pi * sigmaM**2) ** J.shape[0]
                    )
                    WA = (
                        pi[1]
                        * torch.exp(-torch.sum((muA[..., None, None] - J) ** 2, 0) / 2.0 / sigmaA**2)
                        / np.sqrt(2.0 * np.pi * sigmaA**2) ** J.shape[0]
                    )
                    WB = (
                        pi[2]
                        * torch.exp(-torch.sum((muB[..., None, None] - J) ** 2, 0) / 2.0 / sigmaB**2)
                        / np.sqrt(2.0 * np.pi * sigmaB**2) ** J.shape[0]
                    )
                    WS = WM + WB + WA
                    WS += torch.max(WS) * 1e-6
                    WM /= WS
                    WB /= WS
                    WA /= WS

    return {
        "A": A_mat.clone().detach(),
        "v": v.clone().detach(),
        "xv": xv,
        "WM": WM.clone().detach(),
        "WB": WB.clone().detach(),
        "WA": WA.clone().detach(),
        "Esave": Esave,
    }
