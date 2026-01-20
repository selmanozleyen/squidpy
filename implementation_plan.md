# STalign → Squidpy Integration: Implementation Plan

## Overview

This document provides a complete implementation plan for integrating STalign (spatial transcriptomics alignment) into Squidpy. An AI agent with access to Squidpy's codebase and the provided `stalign_port.py` file should be able to complete this integration.

## What is STalign?

STalign performs **diffeomorphic registration** of spatial transcriptomics data using Large Deformation Diffeomorphic Metric Mapping (LDDMM). It allows alignment of:
- Single-cell spatial data (MERFISH, Xenium, etc.) to each other
- Spatial data to histology images
- Spatial data to reference atlases

The core workflow:
1. **Rasterize** cell coordinates into density images
2. **Run LDDMM** to compute affine + diffeomorphic transformation
3. **Apply transformation** to original cell coordinates

---

## Files Provided

### `stalign_port.py`
A self-contained Python file with all necessary functions from STalign, cleaned up and ready for integration. This file contains:

| Function | Purpose | Category |
|----------|---------|----------|
| `rasterize()` | Convert cell coordinates → density image | Core |
| `LDDMM()` | Main diffeomorphic registration algorithm | Core |
| `interp()` | 2D bilinear interpolation using torch | Internal |
| `v_to_phii()` | Integrate velocity → diffeomorphism | Internal |
| `to_A()` | Convert L, T matrices to affine | Internal |
| `extent_from_x()` | Get imshow extent from coordinates | Utility |
| `L_T_from_points()` | Compute affine from landmark points | Utility |
| `build_transform()` | Build transformation from LDDMM output | Core |
| `transform_points_source_to_target()` | Transform coordinates (forward) | Core |
| `transform_points_target_to_source()` | Transform coordinates (backward) | Core |
| `transform_image_source_to_target()` | Transform images | Core |

---

## Target API

### High-Level Function (Main Entry Point)

```python
def align_spatial(
    adata_source: AnnData,
    adata_target: AnnData,
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
    landmark_points_source: Optional[np.ndarray] = None,
    landmark_points_target: Optional[np.ndarray] = None,
    # Computation
    device: str = "cpu",
    verbose: bool = True,
    # Output
    key_added: str = "spatial_aligned",
    copy: bool = False,
) -> Optional[AnnData]:
    """
    Align spatial coordinates from source AnnData to target AnnData.

    This function performs diffeomorphic registration (LDDMM) to align 
    spatial transcriptomics data. It rasterizes cell coordinates into 
    density images, computes a smooth nonlinear transformation, and 
    applies it to the original coordinates.

    Parameters
    ----------
    adata_source
        Source AnnData with spatial coordinates in `obsm[spatial_key]`.
    adata_target
        Target AnnData to align to.
    spatial_key
        Key in `obsm` containing spatial coordinates. Default: "spatial".
    method
        Alignment method. "lddmm" for full diffeomorphic, "affine" for 
        affine-only. Default: "lddmm".
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
        Optional Nx2 array of landmark points in source (row, col order).
    landmark_points_target
        Optional Nx2 array of corresponding landmark points in target.
    device
        PyTorch device ('cpu' or 'cuda:0'). Default: 'cpu'.
    verbose
        Print progress. Default: True.
    key_added
        Key for aligned coordinates in `obsm`. Default: "spatial_aligned".
    copy
        Return a copy. Default: False (modify in place).

    Returns
    -------
    If `copy=True`, returns modified AnnData. Otherwise modifies in place 
    and stores:
    - `adata_source.obsm[key_added]`: aligned coordinates
    - `adata_source.uns['spatial_alignment']`: transformation dict

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
    """
```

### Lower-Level Functions to Expose

```python
def rasterize_coordinates(
    coords: np.ndarray,
    resolution: float = 30.0,
    blur: float = 1.5,
    expand: float = 1.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rasterize point coordinates into a density image.
    
    Parameters
    ----------
    coords
        Nx2 array of (x, y) coordinates.
    resolution
        Pixel size in coordinate units.
    blur
        Gaussian kernel sigma in pixels.
    expand
        Factor to expand image bounds.
        
    Returns
    -------
    X
        1D array of x pixel locations.
    Y
        1D array of y pixel locations.
    image
        Density image with shape (1, H, W).
    """

def apply_transform(
    coords: np.ndarray,
    transform: dict,
    direction: Literal["source_to_target", "target_to_source"] = "source_to_target",
) -> np.ndarray:
    """
    Apply a transformation to coordinates.
    
    Parameters
    ----------
    coords
        Nx2 array of coordinates (row, col order).
    transform
        Transformation dict from align_spatial (contains 'A', 'v', 'xv').
    direction
        Direction of transformation.
        
    Returns
    -------
    Transformed coordinates as Nx2 array.
    """
```

---

## Implementation Steps

### Step 1: Create Module Structure

Create the following files in Squidpy under the `experimental` namespace:

```
squidpy/
├── experimental/
│   ├── __init__.py              # Add exports
│   ├── _align.py                # High-level align_spatial()
│   └── _lddmm/
│       ├── __init__.py          # Package exports
│       ├── _core.py             # LDDMM algorithm (from stalign_port.py)
│       ├── _rasterize.py        # Rasterization (from stalign_port.py)
│       └── _transforms.py       # Transform functions (from stalign_port.py)
```

**Note**: This is placed in `sq.experimental` as the feature is new and API may evolve. 
Once stable, it can be promoted to `sq.tl`.

### Step 2: Set Up Optional Torch Dependency

In `squidpy/experimental/_align.py`:

```python
def _check_torch():
    """Check if torch is available."""
    try:
        import torch
        return True
    except ImportError:
        return False

def align_spatial(...):
    if not _check_torch():
        raise ImportError(
            "Spatial alignment requires PyTorch. "
            "Install with: pip install torch"
        )
    ...
```

In `pyproject.toml` or `setup.py`, add optional dependency:

```toml
[project.optional-dependencies]
torch = ["torch>=2.0.0"]
```

### Step 3: Port Core Functions

Copy functions from `stalign_port.py` into the appropriate module files under `squidpy/experimental/`:

**`_lddmm/_rasterize.py`**:
- `normalize()`
- `rasterize()`

**`_lddmm/_core.py`**:
- `interp()`
- `v_to_phii()`
- `to_A()`
- `LDDMM()` (modify to use tqdm for progress, spatialdata logging)

**`_lddmm/_transforms.py`**:
- `extent_from_x()`
- `L_T_from_points()`
- `build_transform()`
- `transform_points_source_to_target()`
- `transform_points_target_to_source()`
- `transform_image_source_to_target()`

### Step 4: Implement High-Level Wrapper

In `squidpy/experimental/_align.py`, implement `align_spatial()` that:

1. **Extracts coordinates** from AnnData objects
2. **Computes initial affine** from rotation/landmarks if provided
3. **Rasterizes** both source and target coordinates
4. **Runs LDDMM** with proper parameters
5. **Transforms coordinates** using the result
6. **Stores results** in AnnData

```python
# Pseudocode for align_spatial implementation

def align_spatial(adata_source, adata_target, ...):
    # 1. Extract coordinates (note: spatial is usually (x, y), need to convert to (row, col))
    coords_source = adata_source.obsm[spatial_key]  # shape (n_cells, 2)
    coords_target = adata_target.obsm[spatial_key]
    
    # Convert from (x, y) to (row, col) = (y, x) for STalign convention
    xI, yI = coords_source[:, 0], coords_source[:, 1]
    xJ, yJ = coords_target[:, 0], coords_target[:, 1]
    
    # 2. Compute initial affine transform
    if initial_rotation_deg != 0 or landmark_points_source is not None:
        L, T = _compute_initial_transform(
            xI, yI, xJ, yJ,
            initial_rotation_deg,
            landmark_points_source,
            landmark_points_target
        )
    else:
        L, T = None, None
    
    # 3. Rasterize to density images
    XI, YI, I = rasterize(xI, yI, dx=resolution, blur=blur)
    XJ, YJ, J = rasterize(xJ, yJ, dx=resolution, blur=blur)
    
    # Make 3-channel (RGB-like) for LDDMM
    I_rgb = np.vstack([I, I, I])
    J_rgb = np.vstack([J, J, J])
    
    # 4. Run LDDMM
    result = LDDMM(
        [YI, XI], I_rgb,
        [YJ, XJ], J_rgb,
        L=L, T=T,
        niter=niter,
        diffeo_start=diffeo_start,
        a=a, p=p,
        sigmaM=sigmaM,
        sigmaR=sigmaR,
        device=device,
        verbose=verbose
    )
    
    # 5. Transform source coordinates
    # Points need to be in (row, col) order = (y, x)
    points_source = np.column_stack([yI, xI])
    points_aligned = transform_points_source_to_target(
        result['xv'],
        result['v'],
        result['A'],
        points_source
    )
    
    # Convert back to (x, y)
    aligned_coords = np.column_stack([
        points_aligned[:, 1],  # x
        points_aligned[:, 0]   # y
    ])
    
    # 6. Store results
    adata = adata_source.copy() if copy else adata_source
    adata.obsm[key_added] = aligned_coords
    adata.uns['spatial_alignment'] = {
        'A': result['A'].numpy(),
        'v': result['v'].numpy(),
        'xv': [x.numpy() for x in result['xv']],
        'source_spatial_key': spatial_key,
        'target_shape': coords_target.shape,
    }
    
    if copy:
        return adata
```

### Step 5: Modify LDDMM for Squidpy

The LDDMM function in `stalign_port.py` includes matplotlib plotting. Modify it to:

1. **Remove all plotting code** (or make it optional via a `plot` parameter)
2. **Add progress reporting** via `tqdm` progress bar
3. **Use spatialdata logging module** for messages

```python
from tqdm import tqdm
from spatialdata._logging import logger

def LDDMM(..., verbose=True, progress_callback=None):
    """
    ...
    verbose : bool
        Show tqdm progress bar.
    progress_callback : callable, optional
        Function called with (iteration, loss) for custom progress tracking.
    """
    # Remove: fig,ax = plt.subplots(...)
    # Remove: all ax.imshow(...) and fig.canvas.draw() calls
    
    iterator = tqdm(range(niter), desc="LDDMM", disable=not verbose)
    for it in iterator:
        # ... existing computation ...
        
        # Update progress bar
        if verbose:
            iterator.set_postfix(loss=f"{E.item():.4f}")
        
        if progress_callback is not None:
            progress_callback(it, E.item())
    
    logger.info(f"LDDMM completed. Final loss: {Esave[-1][0]:.4f}")
    return {...}
```

### Step 6: Add to Squidpy Exports

In `squidpy/experimental/__init__.py`:

```python
from ._align import align_spatial, rasterize_coordinates, apply_transform

__all__ = [
    # ... existing exports ...
    "align_spatial",
    "rasterize_coordinates", 
    "apply_transform",
]
```

Also ensure `squidpy/__init__.py` imports the experimental module:

```python
from . import experimental
```

### Step 7: Write Tests

Create test file `tests/experimental/test_align.py`:

```python
import numpy as np
import pytest
import anndata as ad

# Skip all tests if torch not installed
torch = pytest.importorskip("torch")

import squidpy as sq


def _create_test_adata(n_cells=500, seed=42):
    """Create a simple test AnnData with spatial coordinates."""
    rng = np.random.default_rng(seed)
    
    # Create circular pattern
    theta = rng.uniform(0, 2*np.pi, n_cells)
    r = rng.uniform(0, 100, n_cells)
    x = r * np.cos(theta) + 500
    y = r * np.sin(theta) + 500
    
    adata = ad.AnnData(
        X=rng.random((n_cells, 10)),
        obsm={"spatial": np.column_stack([x, y])}
    )
    return adata


def test_align_spatial_basic():
    """Test basic alignment runs without error."""
    adata_source = _create_test_adata(seed=42)
    adata_target = _create_test_adata(seed=43)
    
    # Should run without error
    sq.experimental.align_spatial(
        adata_source, 
        adata_target,
        niter=100,  # Small for fast test
        diffeo_start=50,
    )
    
    # Check outputs exist
    assert "spatial_aligned" in adata_source.obsm
    assert "spatial_alignment" in adata_source.uns
    assert adata_source.obsm["spatial_aligned"].shape == adata_source.obsm["spatial"].shape


def test_align_spatial_affine_only():
    """Test affine-only alignment."""
    adata_source = _create_test_adata(seed=42)
    adata_target = _create_test_adata(seed=43)
    
    sq.experimental.align_spatial(
        adata_source,
        adata_target,
        method="affine",
        niter=100,
    )
    
    assert "spatial_aligned" in adata_source.obsm


def test_align_spatial_with_rotation():
    """Test alignment with initial rotation."""
    adata_source = _create_test_adata(seed=42)
    adata_target = _create_test_adata(seed=43)
    
    sq.experimental.align_spatial(
        adata_source,
        adata_target,
        initial_rotation_deg=45,
        niter=100,
    )
    
    assert "spatial_aligned" in adata_source.obsm


def test_align_spatial_copy():
    """Test copy=True returns new AnnData."""
    adata_source = _create_test_adata(seed=42)
    adata_target = _create_test_adata(seed=43)
    
    result = sq.experimental.align_spatial(
        adata_source,
        adata_target,
        niter=100,
        copy=True,
    )
    
    assert result is not adata_source
    assert "spatial_aligned" in result.obsm
    assert "spatial_aligned" not in adata_source.obsm


def test_rasterize_coordinates():
    """Test rasterization function."""
    coords = np.random.rand(100, 2) * 1000
    
    X, Y, image = sq.experimental.rasterize_coordinates(coords, resolution=30.0)
    
    assert X.ndim == 1
    assert Y.ndim == 1
    assert image.ndim == 3
    assert image.shape[0] == 1  # Single channel


def test_apply_transform():
    """Test applying saved transform to new coordinates."""
    adata_source = _create_test_adata(seed=42)
    adata_target = _create_test_adata(seed=43)
    
    sq.experimental.align_spatial(adata_source, adata_target, niter=100)
    
    # Apply transform to new coordinates
    new_coords = np.random.rand(50, 2) * 1000
    transformed = sq.experimental.apply_transform(
        new_coords,
        adata_source.uns["spatial_alignment"],
        direction="source_to_target"
    )
    
    assert transformed.shape == new_coords.shape
```

---

## Important Implementation Notes

### Coordinate Conventions

**CRITICAL**: STalign uses **(row, col)** order internally, but most spatial data uses **(x, y)** order.

```python
# AnnData spatial coordinates: (x, y)
coords_xy = adata.obsm['spatial']  # shape (n, 2), columns are [x, y]

# STalign internal: (row, col) = (y, x)
coords_rc = np.column_stack([coords_xy[:, 1], coords_xy[:, 0]])  # [y, x]

# After alignment, convert back
aligned_xy = np.column_stack([aligned_rc[:, 1], aligned_rc[:, 0]])  # [x, y]
```

### Memory Considerations

For large datasets (>100k cells):
- Increase `resolution` parameter to reduce image size
- Use `device='cuda:0'` if GPU available
- Consider chunking for very large datasets

### Default Parameters Explanation

| Parameter | Default | Effect |
|-----------|---------|--------|
| `resolution=30.0` | Typical for MERFISH/Xenium in microns |
| `blur=1.5` | Smooth kernel, good for cell density |
| `niter=2000` | Usually sufficient for convergence |
| `diffeo_start=100` | Affine-first helps convergence |
| `a=500.0` | Velocity field smoothness scale |
| `sigmaR=5e5` | Regularization (larger = allow more deformation) |

---

## Error Handling

Add proper error messages:

```python
def align_spatial(...):
    # Check inputs
    if spatial_key not in adata_source.obsm:
        raise KeyError(
            f"Spatial key '{spatial_key}' not found in adata_source.obsm. "
            f"Available keys: {list(adata_source.obsm.keys())}"
        )
    
    coords_source = adata_source.obsm[spatial_key]
    if coords_source.shape[1] != 2:
        raise ValueError(
            f"Expected 2D coordinates, got shape {coords_source.shape}"
        )
    
    if coords_source.shape[0] < 100:
        import warnings
        warnings.warn(
            "Very few cells (<100). Alignment may be unreliable."
        )
```

---

## Documentation

Add docstrings following NumPy style (already shown above). Also create a tutorial notebook:

```python
# docs/notebooks/spatial_alignment.ipynb

"""
# Spatial Alignment with Squidpy

This tutorial shows how to align spatial transcriptomics datasets.

## Basic Usage

```python
import squidpy as sq
import scanpy as sc

# Load two spatial datasets
adata1 = sc.read_h5ad("slice1.h5ad")
adata2 = sc.read_h5ad("slice2.h5ad")

# Align slice1 to slice2
sq.experimental.align_spatial(adata1, adata2)

# Visualize
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original source
axes[0].scatter(*adata1.obsm['spatial'].T, s=1, alpha=0.3)
axes[0].set_title('Source (original)')

# Target
axes[1].scatter(*adata2.obsm['spatial'].T, s=1, alpha=0.3)
axes[1].set_title('Target')

# Aligned source
axes[2].scatter(*adata1.obsm['spatial_aligned'].T, s=1, alpha=0.3)
axes[2].scatter(*adata2.obsm['spatial'].T, s=1, alpha=0.1)
axes[2].set_title('Source aligned to target')
```
"""
```

---

## Checklist for Implementation

- [ ] Create `squidpy/experimental/_lddmm/` directory structure
- [ ] Port `_rasterize.py` from `stalign_port.py`
- [ ] Port `_core.py` from `stalign_port.py` (use tqdm, spatialdata logging)
- [ ] Port `_transforms.py` from `stalign_port.py`
- [ ] Implement `align_spatial()` wrapper in `_align.py`
- [ ] Implement `rasterize_coordinates()` wrapper
- [ ] Implement `apply_transform()` wrapper
- [ ] Add torch as optional dependency
- [ ] Add exports to `squidpy/experimental/__init__.py`
- [ ] Write unit tests in `tests/experimental/`
- [ ] Write documentation
- [ ] Create tutorial notebook

**Note**: Functions are in `sq.experimental` initially. Once API is stable, they can be 
promoted to `sq.tl` in a future release.

---

## Design Decisions (Confirmed)

1. **Namespace**: `sq.experimental.align_spatial()` (experimental first, promote to `sq.tl` when stable)
2. **Progress bars**: Use `tqdm` for progress (matches Scanpy style)
3. **Logging**: Use `spatialdata._logging.logger` module
4. **GPU default**: CPU only for now, GPU support can be added later
5. **Image support**: Use spatialdata images only (no ImageContainer from squidpy)
