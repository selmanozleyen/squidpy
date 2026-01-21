#!/usr/bin/env python
"""
Example: MERFISH-to-Visium H&E image alignment using Squidpy

This example demonstrates aligning single-cell MERFISH coordinates to a
Visium H&E staining image. This is a cross-modal alignment where we align
point coordinates to a tissue image.

This is adapted from the original STalign merfish-visium-alignment notebook.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import anndata as ad
import squidpy as sq
from pathlib import Path

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR.parent  # stalign-docs directory
OUTPUT_DIR = DATA_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Use non-interactive backend for saving figures
import matplotlib
matplotlib.use('Agg')

# Optional: for spatialdata integration
try:
    import spatialdata as sd
    from spatialdata.models import TableModel, PointsModel, Image2DModel
    import xarray as xr
    HAS_SPATIALDATA = True
except ImportError:
    HAS_SPATIALDATA = False
    print("spatialdata not installed - running without spatialdata integration")

plt.rcParams["figure.figsize"] = (12, 10)

# =============================================================================
# Load MERFISH data (source - coordinates)
# =============================================================================

print("Loading MERFISH dataset...")

fname_source = DATA_DIR / 'merfish_data/datasets_mouse_brain_map_BrainReceptorShowcase_Slice2_Replicate3_cell_metadata_S2R3.csv.gz'
df_source = pd.read_csv(fname_source)

# Create AnnData for source
coords = df_source[['center_x', 'center_y']].values
adata_source = ad.AnnData(X=np.ones((len(df_source), 1)))
adata_source.obs_names = [f"cell_{i}" for i in range(len(df_source))]
adata_source.var_names = ['placeholder']
adata_source.obsm['spatial'] = coords

print(f"Source: {len(df_source)} cells")

# =============================================================================
# Load Visium H&E image (target - image)
# =============================================================================

print("Loading Visium H&E image...")

image_file = DATA_DIR / 'visium_data/tissue_hires_image.png'
V = plt.imread(image_file)

print(f"Target image shape: {V.shape}")
print(f"Target image range: [{V.min():.3f}, {V.max():.3f}]")

# =============================================================================
# Optional: Create SpatialData with image
# =============================================================================

if HAS_SPATIALDATA:
    print("\nCreating SpatialData objects...")

    # Create points for MERFISH
    source_points = PointsModel.parse(
        df_source[['center_x', 'center_y']].rename(columns={'center_x': 'x', 'center_y': 'y'}),
        transformations={"global": sd.transformations.Identity()}
    )

    # Create image element for Visium H&E
    # Image is HWC format, convert to CYX for spatialdata
    if V.ndim == 3 and V.shape[2] in [3, 4]:
        img_data = V.transpose(2, 0, 1)  # HWC -> CHW
    else:
        img_data = V[np.newaxis, ...]  # Add channel dim

    # Create xarray DataArray
    c_coords = ['r', 'g', 'b'] if img_data.shape[0] == 3 else ['r', 'g', 'b', 'a'][:img_data.shape[0]]
    img_xr = xr.DataArray(
        img_data,
        dims=['c', 'y', 'x'],
        coords={'c': c_coords[:img_data.shape[0]]}
    )

    target_image = Image2DModel.parse(
        img_xr,
        transformations={"global": sd.transformations.Identity()}
    )

    sdata_source = sd.SpatialData(points={"cells": source_points})
    sdata_target = sd.SpatialData(images={"he_image": target_image})

    print(f"Source SpatialData: {sdata_source}")
    print(f"Target SpatialData (with image): {sdata_target}")

# =============================================================================
# Load landmark points for initial alignment
# =============================================================================

print("\nLoading landmark points...")

# Pre-defined corresponding landmarks between MERFISH and Visium
data = np.load(DATA_DIR / 'visium_data/visium2_points.npz')
pointsI = np.array(data['pointsI'][..., ::-1])  # Source landmarks (row, col)
pointsJ = np.array(data['pointsJ'][..., ::-1])  # Target landmarks (row, col)

print(f"Landmark points: {len(pointsI)} pairs")
print(f"Source landmarks:\n{pointsI}")
print(f"Target landmarks:\n{pointsJ}")

# Convert to (x, y) format for squidpy
# Note: landmarks are in (row, col) = (y, x) format, need to swap
landmark_source = pointsI[:, ::-1]  # (row, col) -> (col, row) = (x, y)
landmark_target = pointsJ[:, ::-1]

# =============================================================================
# Visualize source and target
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Rasterize source for visualization
X, Y, source_img = sq.experimental.rasterize_coordinates(
    adata_source.obsm['spatial'],
    resolution=30.0,
    blur=1.0
)

axes[0].imshow(source_img[0], extent=[X.min(), X.max(), Y.max(), Y.min()], cmap='viridis')
axes[0].scatter(landmark_source[:, 0], landmark_source[:, 1], c='red', s=50, marker='x')
axes[0].set_title('Source: MERFISH (rasterized)')

axes[1].imshow(V)
axes[1].scatter(landmark_target[:, 0], landmark_target[:, 1], c='red', s=50, marker='x')
axes[1].set_title('Target: Visium H&E')

# Add landmark labels
for i in range(len(landmark_source)):
    axes[0].annotate(str(i), landmark_source[i], color='white', fontsize=10)
    axes[1].annotate(str(i), landmark_target[i], color='red', fontsize=10)

plt.tight_layout()
output_fig = OUTPUT_DIR / 'merfish_visium_before_alignment.png'
plt.savefig(output_fig, dpi=150)
plt.close()
print(f"Figure saved to {output_fig}")

# =============================================================================
# Run coordinate-to-image alignment (using unified align API)
# =============================================================================

print("\nRunning coordinate-to-image alignment...")
print("This aligns MERFISH cell coordinates to the Visium H&E image space")

# The unified align() function auto-detects:
# - Source: AnnData (coordinates)
# - Target: numpy array (image)
# And automatically uses coordinate-to-image alignment
sq.experimental.align(
    adata_source,
    V,  # Target image (numpy array)
    source_key='spatial',
    key_added='spatial_aligned',
    # Rasterization parameters for source
    resolution=30.0,
    blur=1.0,
    # Landmark points for initial alignment
    landmark_points_source=landmark_source,
    landmark_points_target=landmark_target,
    # LDDMM parameters
    niter=200,
    diffeo_start=100,
    sigmaM=0.2,
    # Other options
    verbose=True,
)

# Note: sq.experimental.align_to_image() also works for explicit usage

print("\nAlignment complete!")

# =============================================================================
# Visualize results
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Before
axes[0].imshow(V)
axes[0].scatter(
    adata_source.obsm['spatial'][:, 0] / 30,  # Rough scaling for visualization
    adata_source.obsm['spatial'][:, 1] / 30,
    s=1, alpha=0.1, c='cyan'
)
axes[0].set_title('Before Alignment (approximate overlay)')

# After
axes[1].imshow(V)
axes[1].scatter(
    adata_source.obsm['spatial_aligned'][:, 0],
    adata_source.obsm['spatial_aligned'][:, 1],
    s=1, alpha=0.1, c='cyan', label='aligned cells'
)
axes[1].scatter(landmark_target[:, 0], landmark_target[:, 1], c='red', s=50, marker='x', label='landmarks')
axes[1].set_title('After Alignment')
axes[1].legend(loc='upper right')

plt.tight_layout()
output_fig = OUTPUT_DIR / 'merfish_visium_after_alignment.png'
plt.savefig(output_fig, dpi=150)
plt.close()
print(f"Figure saved to {output_fig}")

# =============================================================================
# Save results
# =============================================================================

df_results = df_source.copy()
df_results['aligned_x'] = adata_source.obsm['spatial_aligned'][:, 0]
df_results['aligned_y'] = adata_source.obsm['spatial_aligned'][:, 1]

output_file = OUTPUT_DIR / 'squidpy_merfish_visium_results.csv.gz'
df_results.to_csv(output_file, compression='gzip', index=False)
print(f"\nResults saved to {output_file}")

# =============================================================================
# Optional: Create aligned points in SpatialData
# =============================================================================

if HAS_SPATIALDATA:
    print("\nUpdating SpatialData with aligned coordinates...")

    aligned_df = pd.DataFrame({
        'x': adata_source.obsm['spatial_aligned'][:, 0],
        'y': adata_source.obsm['spatial_aligned'][:, 1]
    })
    aligned_points = PointsModel.parse(
        aligned_df,
        transformations={"global": sd.transformations.Identity()}
    )

    # Add aligned points to the target SpatialData (since they're now in image space)
    sdata_target.points["merfish_aligned"] = aligned_points
    print(f"Updated target SpatialData: {sdata_target}")

print("\nDone!")
