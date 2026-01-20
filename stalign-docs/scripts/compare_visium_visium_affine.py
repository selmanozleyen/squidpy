#!/usr/bin/env python
# coding: utf-8
"""
Compare Visium-Visium affine alignment results between STalign and Squidpy.

This script loads the results from both methods and creates a comparison plot.
Run after executing both:
  - visium-visium-alignment-affine-only.py (with STalign installed)
  - squidpy_example_visium_visium_affine.py (with squidpy installed)
"""

from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = DATA_DIR / "comparison"
OUTPUT_DIR.mkdir(exist_ok=True)

# Use non-interactive backend
import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["figure.figsize"] = (16, 12)

print("Loading data...")

# Load original data
fname_source = DATA_DIR / 'visium_data/slice1_coor.csv'
fname_target = DATA_DIR / 'visium_data/slice2_coor.csv'
df_source = pd.read_csv(fname_source)
df_target = pd.read_csv(fname_target)

# Get coordinates
xI = np.array(df_source[df_source.columns[0]])
yI = np.array(df_source[df_source.columns[1]])
xJ = np.array(df_target[df_target.columns[0]])
yJ = np.array(df_target[df_target.columns[1]])

# Load STalign results
stalign_results_file = DATA_DIR / 'output_stalign/stalign_visium_visium_affine_results.csv.gz'
if stalign_results_file.exists():
    df_stalign = pd.read_csv(stalign_results_file)
    xI_stalign = np.array(df_stalign['aligned_x'])
    yI_stalign = np.array(df_stalign['aligned_y'])
    has_stalign = True
    print(f"Loaded STalign results: {len(df_stalign)} spots")
else:
    has_stalign = False
    print(f"STalign results not found at {stalign_results_file}")
    print("Run: python stalign-docs/scripts/visium-visium-alignment-affine-only.py")

# Load Squidpy results
squidpy_results_file = DATA_DIR / 'output/squidpy_visium_visium_affine_results.csv.gz'
if squidpy_results_file.exists():
    df_squidpy = pd.read_csv(squidpy_results_file)
    xI_squidpy = np.array(df_squidpy['aligned_x'])
    yI_squidpy = np.array(df_squidpy['aligned_y'])
    has_squidpy = True
    print(f"Loaded Squidpy results: {len(df_squidpy)} spots")
else:
    has_squidpy = False
    print(f"Squidpy results not found at {squidpy_results_file}")
    print("Run: python stalign-docs/scripts/squidpy_example_visium_visium_affine.py")

if not has_stalign and not has_squidpy:
    print("\nNo results found. Please run both alignment scripts first.")
    exit(1)

# =============================================================================
# Create comparison plot
# =============================================================================

print("\nGenerating comparison plot...")

n_plots = 1 + int(has_stalign) + int(has_squidpy)
fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 6))

if n_plots == 1:
    axes = [axes]

plot_idx = 0

# Original (before alignment)
axes[plot_idx].scatter(xI, yI, s=20, alpha=0.1, label='source')
axes[plot_idx].scatter(xJ, yJ, s=20, alpha=0.1, label='target')
axes[plot_idx].set_title('Before Alignment', fontsize=14)
lgnd = axes[plot_idx].legend(scatterpoints=1, fontsize=10)
for handle in lgnd.legend_handles:
    handle.set_sizes([20.0])
axes[plot_idx].set_aspect('equal')
plot_idx += 1

# STalign results
if has_stalign:
    axes[plot_idx].scatter(xI_stalign, yI_stalign, s=20, alpha=0.1, label='source STaligned')
    axes[plot_idx].scatter(xJ, yJ, s=20, alpha=0.1, label='target')
    axes[plot_idx].set_title('After STalign Affine', fontsize=14)
    lgnd = axes[plot_idx].legend(scatterpoints=1, fontsize=10)
    for handle in lgnd.legend_handles:
        handle.set_sizes([20.0])
    axes[plot_idx].set_aspect('equal')
    plot_idx += 1

# Squidpy results
if has_squidpy:
    axes[plot_idx].scatter(xI_squidpy, yI_squidpy, s=20, alpha=0.1, label='source Squidpy-aligned')
    axes[plot_idx].scatter(xJ, yJ, s=20, alpha=0.1, label='target')
    axes[plot_idx].set_title('After Squidpy Affine', fontsize=14)
    lgnd = axes[plot_idx].legend(scatterpoints=1, fontsize=10)
    for handle in lgnd.legend_handles:
        handle.set_sizes([20.0])
    axes[plot_idx].set_aspect('equal')

plt.tight_layout()
output_fig = OUTPUT_DIR / 'visium_visium_affine_comparison.png'
plt.savefig(output_fig, dpi=150)
plt.close()
print(f"Comparison figure saved to {output_fig}")

# =============================================================================
# Quantitative comparison
# =============================================================================

if has_stalign and has_squidpy:
    print("\n" + "=" * 60)
    print("Quantitative Comparison")
    print("=" * 60)

    # Compute distances to target centroid
    target_centroid = np.array([np.mean(xJ), np.mean(yJ)])

    source_centroid_before = np.array([np.mean(xI), np.mean(yI)])
    source_centroid_stalign = np.array([np.mean(xI_stalign), np.mean(yI_stalign)])
    source_centroid_squidpy = np.array([np.mean(xI_squidpy), np.mean(yI_squidpy)])

    dist_before = np.linalg.norm(source_centroid_before - target_centroid)
    dist_stalign = np.linalg.norm(source_centroid_stalign - target_centroid)
    dist_squidpy = np.linalg.norm(source_centroid_squidpy - target_centroid)

    print(f"\nCentroid distance to target:")
    print(f"  Before alignment:  {dist_before:.4f}")
    print(f"  After STalign:     {dist_stalign:.4f}")
    print(f"  After Squidpy:     {dist_squidpy:.4f}")

    # Compute mean pairwise distances
    # Find nearest neighbor in target for each source point
    from scipy.spatial import cKDTree

    tree_target = cKDTree(np.column_stack([xJ, yJ]))

    source_coords_before = np.column_stack([xI, yI])
    source_coords_stalign = np.column_stack([xI_stalign, yI_stalign])
    source_coords_squidpy = np.column_stack([xI_squidpy, yI_squidpy])

    dists_before, _ = tree_target.query(source_coords_before)
    dists_stalign, _ = tree_target.query(source_coords_stalign)
    dists_squidpy, _ = tree_target.query(source_coords_squidpy)

    print(f"\nMean nearest-neighbor distance to target:")
    print(f"  Before alignment:  {np.mean(dists_before):.4f}")
    print(f"  After STalign:     {np.mean(dists_stalign):.4f}")
    print(f"  After Squidpy:     {np.mean(dists_squidpy):.4f}")

    # Difference between methods
    diff = np.sqrt((xI_stalign - xI_squidpy)**2 + (yI_stalign - yI_squidpy)**2)
    print(f"\nDifference between STalign and Squidpy:")
    print(f"  Mean distance:     {np.mean(diff):.4f}")
    print(f"  Max distance:      {np.max(diff):.4f}")
    print(f"  Std distance:      {np.std(diff):.4f}")

print("\nDone!")
