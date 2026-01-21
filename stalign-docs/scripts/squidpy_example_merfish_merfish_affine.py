#!/usr/bin/env python
"""
Example: MERFISH-to-MERFISH affine-only alignment using Squidpy

This example demonstrates aligning two MERFISH spatial transcriptomics datasets
using only affine transformation (no diffeomorphic registration).

Affine-only alignment is faster and suitable when datasets are already
well-aligned or only need global rotation/translation/scaling.

This is adapted from the original STalign merfish-merfish-alignment-affine-only notebook.
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

# Source dataset
fname_source = DATA_DIR / 'merfish_data/datasets_mouse_brain_map_BrainReceptorShowcase_Slice2_Replicate3_cell_metadata_S2R3.csv.gz'
df_source = pd.read_csv(fname_source)

# Target dataset
fname_target = DATA_DIR / 'merfish_data/datasets_mouse_brain_map_BrainReceptorShowcase_Slice2_Replicate2_cell_metadata_S2R2.csv.gz'
df_target = pd.read_csv(fname_target)

print(f"Source: {len(df_source)} cells")
print(f"Target: {len(df_target)} cells")

# =============================================================================
# Create AnnData objects
# =============================================================================

def create_adata_from_merfish(df: pd.DataFrame, name: str) -> ad.AnnData:
    """Create AnnData object from MERFISH cell metadata."""
    coords = df[['center_x', 'center_y']].values
    X = np.ones((len(df), 1))

    adata = ad.AnnData(X=X)
    adata.obs_names = [f"{name}_cell_{i}" for i in range(len(df))]
    adata.var_names = ['placeholder']
    adata.obsm['spatial'] = coords

    # Skip coordinate and index-like columns
    skip_cols = {'center_x', 'center_y', 'Unnamed: 0'}
    for col in df.columns:
        if col not in skip_cols:
            adata.obs[col] = df[col].values

    return adata

adata_source = create_adata_from_merfish(df_source, "source")
adata_target = create_adata_from_merfish(df_target, "target")

# =============================================================================
# Optional: Create SpatialData objects
# =============================================================================

if HAS_SPATIALDATA:
    print("\nCreating SpatialData objects...")

    source_points = PointsModel.parse(
        df_source[['center_x', 'center_y']].rename(columns={'center_x': 'x', 'center_y': 'y'}),
        transformations={"global": sd.transformations.Identity()}
    )
    target_points = PointsModel.parse(
        df_target[['center_x', 'center_y']].rename(columns={'center_x': 'x', 'center_y': 'y'}),
        transformations={"global": sd.transformations.Identity()}
    )

    sdata_source = sd.SpatialData(points={"cells": source_points})
    sdata_target = sd.SpatialData(points={"cells": target_points})

# =============================================================================
# Run AFFINE-ONLY alignment (using unified align API)
# =============================================================================

print("\nRunning affine-only alignment...")
print("This is faster than full LDDMM and suitable for global transformations")

# Use the unified align() with method='affine'
# The function auto-detects that both inputs are AnnData objects
# and performs coordinate-to-coordinate alignment
sq.experimental.align(
    adata_source,
    adata_target,
    source_key='spatial',
    key_added='spatial_aligned',
    # Rasterization parameters
    resolution=15.0,  # Can use finer resolution for affine
    blur=1.5,
    # Initial rotation hint
    initial_rotation_deg=45.0,
    # Use affine-only method
    method='affine',
    niter=1000,       # Iterations for affine optimization
    # Other options
    copy=False,
    verbose=True,
)

# Note: sq.experimental.align_spatial(..., method='affine') also works

print("\nAffine alignment complete!")

# =============================================================================
# Visualize results
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Before
axes[0].scatter(
    adata_source.obsm['spatial'][:, 0],
    adata_source.obsm['spatial'][:, 1],
    s=1, alpha=0.1, label='source'
)
axes[0].scatter(
    adata_target.obsm['spatial'][:, 0],
    adata_target.obsm['spatial'][:, 1],
    s=1, alpha=0.1, label='target'
)
axes[0].set_title('Before Alignment')
axes[0].legend(markerscale=10, loc='lower left')
axes[0].set_aspect('equal')

# After
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
axes[1].set_title('After Affine Alignment')
axes[1].legend(markerscale=10, loc='lower left')
axes[1].set_aspect('equal')

plt.tight_layout()
output_fig = OUTPUT_DIR / 'merfish_merfish_affine_alignment.png'
plt.savefig(output_fig, dpi=150)
plt.close()
print(f"Figure saved to {output_fig}")

# =============================================================================
# Extract and display the affine transformation matrix
# =============================================================================

transform = adata_source.uns['spatial_alignment']
A = transform['A']

print("\n" + "="*50)
print("Affine Transformation Matrix (3x3):")
print("="*50)
print(A)
print("\nThis matrix encodes:")
print(f"  - Linear transform (rotation + scale): A[:2,:2]")
print(f"  - Translation: A[:2,2]")

# =============================================================================
# Save results
# =============================================================================

df_results = df_source.copy()
df_results['aligned_x'] = adata_source.obsm['spatial_aligned'][:, 0]
df_results['aligned_y'] = adata_source.obsm['spatial_aligned'][:, 1]

output_file = OUTPUT_DIR / 'squidpy_merfish_merfish_affine_results.csv.gz'
df_results.to_csv(output_file, compression='gzip', index=False)
print(f"\nResults saved to {output_file}")

# =============================================================================
# Optional: Update SpatialData
# =============================================================================

if HAS_SPATIALDATA:
    aligned_df = pd.DataFrame({
        'x': adata_source.obsm['spatial_aligned'][:, 0],
        'y': adata_source.obsm['spatial_aligned'][:, 1]
    })
    aligned_points = PointsModel.parse(
        aligned_df,
        transformations={"global": sd.transformations.Identity()}
    )
    sdata_source.points["cells_aligned"] = aligned_points
    print(f"Updated SpatialData: {sdata_source}")

print("\nDone!")
