#!/usr/bin/env python
"""
Example: MERFISH-to-MERFISH alignment using Squidpy

This example demonstrates aligning two MERFISH spatial transcriptomics datasets
of mouse brain coronal sections using squidpy.experimental alignment functions.

This is adapted from the original STalign merfish-merfish-alignment notebook.
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
    from spatialdata.models import TableModel, PointsModel
    HAS_SPATIALDATA = True
except ImportError:
    HAS_SPATIALDATA = False
    print("spatialdata not installed - running without spatialdata integration")

plt.rcParams["figure.figsize"] = (12, 10)

# =============================================================================
# Load MERFISH data
# =============================================================================

print("Loading MERFISH datasets...")

# Source dataset (Slice 2, Replicate 3)
fname_source = DATA_DIR / 'merfish_data/datasets_mouse_brain_map_BrainReceptorShowcase_Slice2_Replicate3_cell_metadata_S2R3.csv.gz'
df_source = pd.read_csv(fname_source)

# Target dataset (Slice 2, Replicate 2)
fname_target = DATA_DIR / 'merfish_data/datasets_mouse_brain_map_BrainReceptorShowcase_Slice2_Replicate2_cell_metadata_S2R2.csv.gz'
df_target = pd.read_csv(fname_target)

print(f"Source: {len(df_source)} cells")
print(f"Target: {len(df_target)} cells")

# =============================================================================
# Create AnnData objects with spatial coordinates
# =============================================================================

def create_adata_from_merfish(df: pd.DataFrame, name: str) -> ad.AnnData:
    """Create AnnData object from MERFISH cell metadata."""
    # Extract coordinates (x, y format for squidpy)
    coords = df[['center_x', 'center_y']].values

    # Create minimal expression matrix (just cell count placeholder)
    X = np.ones((len(df), 1))

    # Create AnnData
    adata = ad.AnnData(X=X)
    adata.obs_names = [f"{name}_cell_{i}" for i in range(len(df))]
    adata.var_names = ['placeholder']

    # Store spatial coordinates in obsm (squidpy convention)
    adata.obsm['spatial'] = coords

    # Store original metadata (exclude coordinate columns and index-like columns)
    skip_cols = {'center_x', 'center_y', 'Unnamed: 0'}
    for col in df.columns:
        if col not in skip_cols:
            adata.obs[col] = df[col].values

    return adata

adata_source = create_adata_from_merfish(df_source, "source")
adata_target = create_adata_from_merfish(df_target, "target")

print(f"\nSource AnnData: {adata_source}")
print(f"Target AnnData: {adata_target}")

# =============================================================================
# Optional: Create SpatialData objects for richer integration
# =============================================================================

if HAS_SPATIALDATA:
    print("\nCreating SpatialData objects...")

    # Create points elements from coordinates
    source_points = PointsModel.parse(
        df_source[['center_x', 'center_y']].rename(columns={'center_x': 'x', 'center_y': 'y'}),
        transformations={"global": sd.transformations.Identity()}
    )
    target_points = PointsModel.parse(
        df_target[['center_x', 'center_y']].rename(columns={'center_x': 'x', 'center_y': 'y'}),
        transformations={"global": sd.transformations.Identity()}
    )

    # Create SpatialData containers
    sdata_source = sd.SpatialData(
        points={"cells": source_points},
        tables={"table": TableModel.parse(adata_source)}
    )
    sdata_target = sd.SpatialData(
        points={"cells": target_points},
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
    s=1, alpha=0.2, label='source'
)
axes[0].set_title('Source (MERFISH S2R3)')
axes[0].legend(markerscale=10)
axes[0].set_aspect('equal')

# Target
axes[1].scatter(
    adata_target.obsm['spatial'][:, 0],
    adata_target.obsm['spatial'][:, 1],
    s=1, alpha=0.2, c='orange', label='target'
)
axes[1].set_title('Target (MERFISH S2R2)')
axes[1].legend(markerscale=10)
axes[1].set_aspect('equal')

# Overlay
axes[2].scatter(
    adata_source.obsm['spatial'][:, 0],
    adata_source.obsm['spatial'][:, 1],
    s=1, alpha=0.2, label='source'
)
axes[2].scatter(
    adata_target.obsm['spatial'][:, 0],
    adata_target.obsm['spatial'][:, 1],
    s=1, alpha=0.1, label='target'
)
axes[2].set_title('Overlay (before alignment)')
axes[2].legend(markerscale=10)
axes[2].set_aspect('equal')

plt.tight_layout()
output_fig = OUTPUT_DIR / 'merfish_merfish_before_alignment.png'
plt.savefig(output_fig, dpi=150)
plt.close()
print(f"Figure saved to {output_fig}")

# =============================================================================
# Run alignment with Squidpy (using unified align API)
# =============================================================================

print("\nRunning LDDMM alignment...")
print("This uses rasterization at 30µm resolution and full diffeomorphic registration")

# The unified sq.experimental.align() function auto-detects input types:
# - AnnData → AnnData: coordinate-to-coordinate alignment (like align_spatial)
# - AnnData → Image: coordinate-to-image alignment (like align_to_image)
# - Image → Image: image-to-image alignment (like align_images)
#
# It handles:
# 1. Rasterization of point clouds to images
# 2. Initial affine alignment (with optional rotation hint)
# 3. LDDMM diffeomorphic registration
# 4. Storing results back in adata_source

sq.experimental.align(
    adata_source,
    adata_target,
    source_key='spatial',
    key_added='spatial_aligned',
    # Rasterization parameters
    resolution=30.0,  # 30µm resolution
    blur=1.5,         # Gaussian blur for smoothing
    # Initial alignment - the datasets need ~45° rotation
    initial_rotation_deg=45.0,
    # LDDMM parameters
    niter=2000,       # Number of iterations (reduce for faster testing)
    diffeo_start=100, # Start diffeomorphic registration after 100 iterations
    a=500.0,          # Smoothness scale
    # Other options
    copy=False,       # Modify adata_source in place
    verbose=True,     # Show progress
)

# Note: You can also use the explicit sq.experimental.align_spatial() function
# for the same result. The unified align() is recommended for new code.

print("\nAlignment complete!")
print(f"Aligned coordinates stored in adata_source.obsm['spatial_aligned']")
print(f"Transform stored in adata_source.uns['spatial_alignment']")

# =============================================================================
# Visualize results
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Before alignment
axes[0].scatter(
    adata_source.obsm['spatial'][:, 0],
    adata_source.obsm['spatial'][:, 1],
    s=1, alpha=0.1, label='source (original)'
)
axes[0].scatter(
    adata_target.obsm['spatial'][:, 0],
    adata_target.obsm['spatial'][:, 1],
    s=1, alpha=0.1, label='target'
)
axes[0].set_title('Before Alignment')
axes[0].legend(markerscale=10, loc='lower left')
axes[0].set_aspect('equal')

# After alignment
axes[1].scatter(
    adata_source.obsm['spatial_aligned'][:, 0],
    adata_source.obsm['spatial_aligned'][:, 1],
    s=1, alpha=0.1, label='source (aligned)'
)
axes[1].scatter(
    adata_target.obsm['spatial'][:, 0],
    adata_target.obsm['spatial'][:, 1],
    s=1, alpha=0.1, label='target'
)
axes[1].set_title('After LDDMM Alignment')
axes[1].legend(markerscale=10, loc='lower left')
axes[1].set_aspect('equal')

# Show all three
axes[2].scatter(
    adata_source.obsm['spatial'][:, 0],
    adata_source.obsm['spatial'][:, 1],
    s=1, alpha=0.1, label='source (original)'
)
axes[2].scatter(
    adata_source.obsm['spatial_aligned'][:, 0],
    adata_source.obsm['spatial_aligned'][:, 1],
    s=1, alpha=0.1, label='source (aligned)'
)
axes[2].scatter(
    adata_target.obsm['spatial'][:, 0],
    adata_target.obsm['spatial'][:, 1],
    s=1, alpha=0.1, label='target'
)
axes[2].set_title('All Three')
axes[2].legend(markerscale=10, loc='lower left')
axes[2].set_aspect('equal')

plt.tight_layout()
output_fig = OUTPUT_DIR / 'merfish_merfish_after_alignment.png'
plt.savefig(output_fig, dpi=150)
plt.close()
print(f"Figure saved to {output_fig}")

# =============================================================================
# Save results
# =============================================================================

# Add aligned coordinates to the original dataframe
df_results = df_source.copy()
df_results['aligned_x'] = adata_source.obsm['spatial_aligned'][:, 0]
df_results['aligned_y'] = adata_source.obsm['spatial_aligned'][:, 1]

# Save to CSV
output_file = OUTPUT_DIR / 'squidpy_merfish_merfish_lddmm_results.csv.gz'
df_results.to_csv(output_file, compression='gzip', index=False)
print(f"\nResults saved to {output_file}")

# =============================================================================
# Optional: Update SpatialData with aligned coordinates
# =============================================================================

if HAS_SPATIALDATA:
    print("\nUpdating SpatialData with aligned coordinates...")

    # Create new points element with aligned coordinates
    aligned_df = pd.DataFrame({
        'x': adata_source.obsm['spatial_aligned'][:, 0],
        'y': adata_source.obsm['spatial_aligned'][:, 1]
    })
    aligned_points = PointsModel.parse(
        aligned_df,
        transformations={"global": sd.transformations.Identity()}
    )

    # Add to SpatialData
    sdata_source.points["cells_aligned"] = aligned_points
    print(f"Updated SpatialData: {sdata_source}")

print("\nDone!")
