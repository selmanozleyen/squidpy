#!/usr/bin/env python
"""Demo: squidpy.experimental.tl.align() — unified spatial alignment.

This script demonstrates the three alignment paths:

1. **Point-to-point** (moscot backend — optimal transport)
2. **Point-to-image** (STalign backend — LDDMM)
3. **Image-to-image** (STalign backend — LDDMM)

Install the optional backends you need::

    pip install 'squidpy[align]'     # both moscot + torch
    pip install 'squidpy[moscot]'    # point-to-point only
    pip install 'squidpy[torch]'     # image-based only
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Helpers to generate synthetic data
# ---------------------------------------------------------------------------

def _make_tissue_coords(
    n: int = 800,
    center: tuple[float, float] = (500.0, 500.0),
    radius: float = 200.0,
    seed: int = 42,
) -> np.ndarray:
    """Circular point cloud mimicking a tissue section."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, n)
    r = radius * np.sqrt(rng.uniform(0, 1, n))
    x = r * np.cos(theta) + center[0]
    y = r * np.sin(theta) + center[1]
    return np.column_stack([x, y])


def _rotate_coords(
    coords: np.ndarray,
    angle_deg: float,
    center: tuple[float, float] | None = None,
) -> np.ndarray:
    """Rotate coordinates clockwise by *angle_deg* around *center*."""
    if center is None:
        center = coords.mean(axis=0)
    theta = np.radians(angle_deg)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    c = np.asarray(center)
    return (R @ (coords - c).T).T + c


def _make_density_image(coords: np.ndarray, shape: tuple[int, int] = (128, 128)) -> np.ndarray:
    """Rasterise coordinates into a simple Gaussian-blurred density image."""
    from scipy.ndimage import gaussian_filter

    img = np.zeros(shape, dtype=np.float64)
    xmin, xmax = coords[:, 0].min(), coords[:, 0].max()
    ymin, ymax = coords[:, 1].min(), coords[:, 1].max()
    for x, y in coords:
        col = int((x - xmin) / (xmax - xmin + 1e-8) * (shape[1] - 1))
        row = int((y - ymin) / (ymax - ymin + 1e-8) * (shape[0] - 1))
        col = np.clip(col, 0, shape[1] - 1)
        row = np.clip(row, 0, shape[0] - 1)
        img[row, col] += 1.0
    return gaussian_filter(img, sigma=3)


# ---------------------------------------------------------------------------
# Demo 1: Point-to-point alignment  (moscot)
# ---------------------------------------------------------------------------

def demo_point_to_point():
    """Align two spatial point clouds using moscot optimal transport."""
    print("\n" + "=" * 60)
    print("  DEMO 1: Point-to-point alignment (moscot)")
    print("=" * 60 + "\n")

    try:
        import moscot  # noqa: F401
    except ImportError:
        print("  [SKIP] moscot not installed — pip install moscot")
        return

    import anndata as ad
    import squidpy as sq

    # Synthetic tissue: source is rotated + shifted copy of target
    target_coords = _make_tissue_coords(n=600, center=(500, 500), seed=0)
    source_coords = _rotate_coords(target_coords, angle_deg=25) + np.array([30.0, -20.0])

    # Create minimal AnnData objects (moscot also supports gene features)
    rng = np.random.default_rng(1)
    adata_src = ad.AnnData(X=rng.normal(size=(len(source_coords), 50)).astype(np.float32))
    adata_tgt = ad.AnnData(X=rng.normal(size=(len(target_coords), 50)).astype(np.float32))
    adata_src.obsm["spatial"] = source_coords
    adata_tgt.obsm["spatial"] = target_coords

    # Align!
    sq.experimental.tl.align(
        adata_src,
        adata_tgt,
        method="optimal_transport",
        verbose=True,
    )

    aligned = adata_src.obsm["spatial_aligned"]

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(*source_coords.T, s=4, alpha=0.5, label="source")
    axes[0].scatter(*target_coords.T, s=4, alpha=0.5, label="target")
    axes[0].set_title("Before alignment")
    axes[0].legend()
    axes[0].set_aspect("equal")

    axes[1].scatter(*aligned.T, s=4, alpha=0.5, label="aligned source")
    axes[1].scatter(*target_coords.T, s=4, alpha=0.5, label="target")
    axes[1].set_title("After alignment (moscot OT)")
    axes[1].legend()
    axes[1].set_aspect("equal")

    plt.tight_layout()
    plt.savefig("demo_point_to_point.png", dpi=150)
    print("  Saved demo_point_to_point.png")
    plt.close()


# ---------------------------------------------------------------------------
# Demo 2: Point-to-image alignment  (STalign)
# ---------------------------------------------------------------------------

def demo_point_to_image():
    """Align cell coordinates to a reference density image via LDDMM."""
    print("\n" + "=" * 60)
    print("  DEMO 2: Point-to-image alignment (STalign/LDDMM)")
    print("=" * 60 + "\n")

    try:
        import torch  # noqa: F401
    except ImportError:
        print("  [SKIP] torch not installed — pip install torch")
        return

    import anndata as ad
    import squidpy as sq

    # Create a reference image from one set of coordinates
    target_coords = _make_tissue_coords(n=800, center=(500, 500), seed=10)
    target_image = _make_density_image(target_coords, shape=(128, 128))

    # Source: rotated version of the same tissue
    source_coords = _rotate_coords(target_coords, angle_deg=15) + np.array([20.0, -10.0])

    adata = ad.AnnData(X=np.zeros((len(source_coords), 1)))
    adata.obsm["spatial"] = source_coords

    # Align coordinates to the image
    sq.experimental.tl.align(
        adata,
        target_image,
        resolution=15.0,
        blur=1.5,
        niter=200,        # low for demo speed
        diffeo_start=50,
        verbose=True,
    )

    aligned = adata.obsm["spatial_aligned"]

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].imshow(target_image, cmap="hot", origin="lower")
    axes[0].set_title("Target image")

    axes[1].scatter(*source_coords.T, s=3, alpha=0.5, c="blue")
    axes[1].imshow(target_image, cmap="hot", alpha=0.3, origin="lower",
                   extent=[source_coords[:, 0].min(), source_coords[:, 0].max(),
                           source_coords[:, 1].min(), source_coords[:, 1].max()])
    axes[1].set_title("Source coords (before)")
    axes[1].set_aspect("equal")

    axes[2].scatter(*aligned.T, s=3, alpha=0.5, c="green")
    axes[2].set_title("Aligned coords (after)")
    axes[2].set_aspect("equal")

    plt.tight_layout()
    plt.savefig("demo_point_to_image.png", dpi=150)
    print("  Saved demo_point_to_image.png")
    plt.close()


# ---------------------------------------------------------------------------
# Demo 3: Image-to-image alignment  (STalign)
# ---------------------------------------------------------------------------

def demo_image_to_image():
    """Align two density images using LDDMM."""
    print("\n" + "=" * 60)
    print("  DEMO 3: Image-to-image alignment (STalign/LDDMM)")
    print("=" * 60 + "\n")

    try:
        import torch  # noqa: F401
    except ImportError:
        print("  [SKIP] torch not installed — pip install torch")
        return

    import squidpy as sq

    # Two tissue images from different coordinate sets
    coords_a = _make_tissue_coords(n=800, center=(500, 500), seed=20)
    coords_b = _rotate_coords(coords_a, angle_deg=20) + np.array([15.0, -15.0])

    img_source = _make_density_image(coords_a, shape=(100, 100))
    img_target = _make_density_image(coords_b, shape=(100, 100))

    # Align
    transform = sq.experimental.tl.align(
        img_source,
        img_target,
        niter=200,       # low for demo speed
        diffeo_start=50,
        verbose=True,
    )

    print(f"\n  Transform keys: {list(transform.keys())}")
    print(f"  Affine matrix:\n{transform['A']}")

    # Apply affine to source image for visualisation
    aligned_img = sq.experimental.tl.apply_affine(
        img_source,
        transform["A"],
        output_shape=img_target.shape,
    )

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].imshow(img_source, cmap="hot")
    axes[0].set_title("Source image")
    axes[1].imshow(img_target, cmap="hot")
    axes[1].set_title("Target image")
    axes[2].imshow(aligned_img, cmap="hot")
    axes[2].set_title("Aligned source (affine)")
    plt.tight_layout()
    plt.savefig("demo_image_to_image.png", dpi=150)
    print("  Saved demo_image_to_image.png")
    plt.close()


# ---------------------------------------------------------------------------
# Demo 4: Auto-dispatch — show how method is auto-selected
# ---------------------------------------------------------------------------

def demo_auto_dispatch():
    """Show the automatic backend selection."""
    print("\n" + "=" * 60)
    print("  DEMO 4: Auto-dispatch")
    print("=" * 60 + "\n")

    from squidpy.experimental.tl._align import _has_moscot, _has_torch

    print(f"  moscot installed: {_has_moscot()}")
    print(f"  torch  installed: {_has_torch()}")
    print()
    print("  align(AnnData, AnnData)")
    if _has_moscot():
        print("    → auto-selects moscot (optimal transport)")
    elif _has_torch():
        print("    → auto-selects STalign (rasterise + LDDMM)")
    else:
        print("    → ERROR: needs moscot or torch")

    print("  align(AnnData, image)")
    if _has_torch():
        print("    → auto-selects STalign (LDDMM)")
    else:
        print("    → ERROR: needs torch")

    print("  align(image, image)")
    if _has_torch():
        print("    → auto-selects STalign (LDDMM)")
    else:
        print("    → ERROR: needs torch")

    print()
    print("  You can always force a backend:")
    print('    align(..., method="optimal_transport")  # moscot')
    print('    align(..., method="lddmm")              # STalign LDDMM')
    print('    align(..., method="affine")              # STalign affine-only')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  squidpy.experimental.tl.align() — Alignment Demo      ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print("║  Backends:                                              ║")
    print("║    point → point  :  moscot  (optimal transport)        ║")
    print("║    point → image  :  STalign (LDDMM)                   ║")
    print("║    image → image  :  STalign (LDDMM)                   ║")
    print("╚══════════════════════════════════════════════════════════╝")

    demo_auto_dispatch()
    demo_point_to_point()
    demo_point_to_image()
    demo_image_to_image()

    print("\n\nAll demos complete!")
