#!/usr/bin/env python
"""
Example: Visium-to-Visium affine alignment using Squidpy

This example demonstrates aligning two Visium spot-based datasets
using affine transformation. Visium data has lower spatial resolution
than single-cell methods, so affine alignment is often sufficient.

This is adapted from the original STalign visium-visium-alignment-affine-only notebook.
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
    from spatialdata.models import TableModel, PointsModel, ShapesModel
    from shapely.geometry import Point
    import geopandas as gpd
    HAS_SPATIALDATA = True
except ImportError:
    HAS_SPATIALDATA = False
    print("spatialdata not installed - running without spatialdata integration")

plt.rcParams["figure.figsize"] = (12, 10)

# =============================================================================
# Load Visium datasets
# =============================================================================

print("Loading Visium datasets...")

# Source dataset (slice 1)
fname_source = DATA_DIR / 'visium_data/slice1_coor.csv'
df_source = pd.read_csv(fname_source)

# Target dataset (slice 2)
fname_target = DATA_DIR / 'visium_data/slice2_coor.csv'
df_target = pd.read_csv(fname_target)

print(f"Source (slice 1): {len(df_source)} spots")
print(f"Target (slice 2): {len(df_target)} spots")

# =============================================================================
# Create AnnData objects
# =============================================================================

def create_adata_from_visium(df: pd.DataFrame, name: str) -> ad.AnnData:
    """Create AnnData object from Visium coordinates."""
    # Get x, y columns (first two columns)
    x_col = df.columns[0]
    y_col = df.columns[1]
    coords = df[[x_col, y_col]].values

    # Create minimal AnnData
    X = np.ones((len(df), 1))

    adata = ad.AnnData(X=X)
    adata.obs_names = [f"{name}_spot_{i}" for i in range(len(df))]
    adata.var_names = ['placeholder']
    adata.obsm['spatial'] = coords

    return adata

adata_source = create_adata_from_visium(df_source, "source")
adata_target = create_adata_from_visium(df_target, "target")

print(f"\nSource AnnData: {adata_source}")
print(f"Target AnnData: {adata_target}")

# =============================================================================
# Optional: Create SpatialData with spots as shapes
# =============================================================================

if HAS_SPATIALDATA:
    print("\nCreating SpatialData objects with spot shapes...")

    def coords_to_circles(coords, radius=0.5):
        """Convert coordinates to circle geometries for spots."""
        geometries = [Point(x, y).buffer(radius) for x, y in coords]
        return gpd.GeoDataFrame(geometry=geometries)

    # Create spot shapes
    source_spots = ShapesModel.parse(
        coords_to_circles(adata_source.obsm['spatial']),
        transformations={"global": sd.transformations.Identity()}
    )
    target_spots = ShapesModel.parse(
        coords_to_circles(adata_target.obsm['spatial']),
        transformations={"global": sd.transformations.Identity()}
    )

    sdata_source = sd.SpatialData(
        shapes={"spots": source_spots},
        tables={"table": TableModel.parse(adata_source)}
    )
    sdata_target = sd.SpatialData(
        shapes={"spots": target_spots},
        tables={"table": TableModel.parse(adata_target)}
    )

    print(f"Source SpatialData: {sdata_source}")
    print(f"Target SpatialData: {sdata_target}")

# =============================================================================
# Visualize original data
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Source
axes[0].scatter(
    adata_source.obsm['spatial'][:, 0],
    adata_source.obsm['spatial'][:, 1],
    s=20, alpha=0.5, label='source'
)
axes[0].set_title('Source (Visium slice 1)')
axes[0].legend()
axes[0].set_aspect('equal')

# Target
axes[1].scatter(
    adata_target.obsm['spatial'][:, 0],
    adata_target.obsm['spatial'][:, 1],
    s=20, alpha=0.5, c='orange', label='target'
)
axes[1].set_title('Target (Visium slice 2)')
axes[1].legend()
axes[1].set_aspect('equal')

# Overlay
axes[2].scatter(
    adata_source.obsm['spatial'][:, 0],
    adata_source.obsm['spatial'][:, 1],
    s=20, alpha=0.5, label='source'
)
axes[2].scatter(
    adata_target.obsm['spatial'][:, 0],
    adata_target.obsm['spatial'][:, 1],
    s=20, alpha=0.3, label='target'
)
axes[2].set_title('Overlay (before alignment)')
axes[2].legend()
axes[2].set_aspect('equal')

plt.tight_layout()
output_fig = OUTPUT_DIR / 'visium_visium_before_alignment.png'
plt.savefig(output_fig, dpi=150)
plt.close()
print(f"Figure saved to {output_fig}")

# =============================================================================
# Run affine-only alignment
# =============================================================================

print("\nRunning affine-only alignment...")
print("Visium spots are already well-registered, affine is sufficient")

# For Visium data with relatively uniform spot patterns,
# we use finer rasterization and affine-only method
sq.experimental.align_spatial(
    adata_source,
    adata_target,
    spatial_key='spatial',
    key_added='spatial_aligned',
    # Rasterization - use dx=1 for spot data
    resolution=1.0,
    blur=0.5,
    # No initial rotation needed (already aligned)
    initial_rotation_deg=0.0,
    # Affine-only method
    method='affine',
    niter=1000,
    # Smoothness scale - smaller for spot data
    a=5.0,
    # Other options
    copy=False,
    verbose=True,
)

print("\nAlignment complete!")

# =============================================================================
# Visualize results (matching STalign style for comparison)
# =============================================================================

# Get coordinates for plotting
xI = adata_source.obsm['spatial'][:, 0]
yI = adata_source.obsm['spatial'][:, 1]
xJ = adata_target.obsm['spatial'][:, 0]
yJ = adata_target.obsm['spatial'][:, 1]
xI_aligned = adata_source.obsm['spatial_aligned'][:, 0]
yI_aligned = adata_source.obsm['spatial_aligned'][:, 1]

# Plot: Before/After side by side (matching STalign format)
fig, ax = plt.subplots(1, 2)
ax[0].scatter(xI, yI, s=20, alpha=0.1, label='source')
ax[0].scatter(xJ, yJ, s=20, alpha=0.1, label='target')
ax[1].scatter(xI_aligned, yI_aligned, s=20, alpha=0.1, label='source squidpy-aligned')
ax[1].scatter(xJ, yJ, s=20, alpha=0.1, label='target')

lgnd = ax[0].legend(scatterpoints=1, fontsize=10)
for handle in lgnd.legend_handles:
    handle.set_sizes([20.0])

lgnd = ax[1].legend(scatterpoints=1, fontsize=10)
for handle in lgnd.legend_handles:
    handle.set_sizes([20.0])

ax[0].set_aspect('equal')
ax[1].set_aspect('equal')
ax[0].set_title('Before Alignment')
ax[1].set_title('After Squidpy Affine Alignment')

plt.tight_layout()
output_fig = OUTPUT_DIR / 'squidpy_visium_visium_affine_comparison.png'
plt.savefig(output_fig, dpi=150)
plt.close()
print(f"Figure saved to {output_fig}")

# =============================================================================
# Display transformation
# =============================================================================

transform = adata_source.uns['spatial_alignment']
A = transform['A']

print("\n" + "="*50)
print("Affine Transformation Matrix:")
print("="*50)
print(A)

# =============================================================================
# Save results
# =============================================================================

df_results = df_source.copy()
df_results['aligned_x'] = adata_source.obsm['spatial_aligned'][:, 0]
df_results['aligned_y'] = adata_source.obsm['spatial_aligned'][:, 1]

output_file = OUTPUT_DIR / 'squidpy_visium_visium_affine_results.csv.gz'
df_results.to_csv(output_file, compression='gzip', index=False)
print(f"\nResults saved to {output_file}")

# =============================================================================
# Optional: Update SpatialData
# =============================================================================

if HAS_SPATIALDATA:
    print("\nUpdating SpatialData with aligned spots...")

    def coords_to_circles(coords, radius=0.5):
        geometries = [Point(x, y).buffer(radius) for x, y in coords]
        return gpd.GeoDataFrame(geometry=geometries)

    aligned_spots = ShapesModel.parse(
        coords_to_circles(adata_source.obsm['spatial_aligned']),
        transformations={"global": sd.transformations.Identity()}
    )

    sdata_source.shapes["spots_aligned"] = aligned_spots
    print(f"Updated SpatialData: {sdata_source}")

print("\nDone!")
