#!/usr/bin/env python
# coding: utf-8
"""
Compare MERFISH-MERFISH affine alignment results between STalign and Squidpy.
"""

from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = DATA_DIR / "comparison"
OUTPUT_DIR.mkdir(exist_ok=True)

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["figure.figsize"] = (16, 12)

print("Loading MERFISH data...")

# Load original data
fname_source = DATA_DIR / 'merfish_data/datasets_mouse_brain_map_BrainReceptorShowcase_Slice2_Replicate3_cell_metadata_S2R3.csv.gz'
fname_target = DATA_DIR / 'merfish_data/datasets_mouse_brain_map_BrainReceptorShowcase_Slice2_Replicate2_cell_metadata_S2R2.csv.gz'
df_source = pd.read_csv(fname_source)
df_target = pd.read_csv(fname_target)

xI = np.array(df_source['center_x'])
yI = np.array(df_source['center_y'])
xJ = np.array(df_target['center_x'])
yJ = np.array(df_target['center_y'])

# Load STalign results
stalign_file = DATA_DIR / 'output_stalign/stalign_merfish_merfish_affine_results.csv.gz'
has_stalign = stalign_file.exists()
if has_stalign:
    df_stalign = pd.read_csv(stalign_file)
    xI_stalign = np.array(df_stalign['aligned_x'])
    yI_stalign = np.array(df_stalign['aligned_y'])
    print(f"Loaded STalign results: {len(df_stalign)} cells")
else:
    print(f"STalign results not found: {stalign_file}")

# Load Squidpy results
squidpy_file = DATA_DIR / 'output/squidpy_merfish_merfish_affine_results.csv.gz'
has_squidpy = squidpy_file.exists()
if has_squidpy:
    df_squidpy = pd.read_csv(squidpy_file)
    xI_squidpy = np.array(df_squidpy['aligned_x'])
    yI_squidpy = np.array(df_squidpy['aligned_y'])
    print(f"Loaded Squidpy results: {len(df_squidpy)} cells")
else:
    print(f"Squidpy results not found: {squidpy_file}")

if not has_stalign and not has_squidpy:
    print("\nNo results found. Run alignment scripts first.")
    exit(1)

# Create comparison plot
print("\nGenerating comparison plot...")

n_plots = 1 + int(has_stalign) + int(has_squidpy)
fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 6))
if n_plots == 1:
    axes = [axes]

plot_idx = 0

# Before
axes[plot_idx].scatter(xI, yI, s=1, alpha=0.1, label='source')
axes[plot_idx].scatter(xJ, yJ, s=1, alpha=0.1, label='target')
axes[plot_idx].set_title('Before Alignment', fontsize=14)
axes[plot_idx].legend(markerscale=5)
axes[plot_idx].set_aspect('equal')
plot_idx += 1

if has_stalign:
    axes[plot_idx].scatter(xI_stalign, yI_stalign, s=1, alpha=0.1, label='source STaligned')
    axes[plot_idx].scatter(xJ, yJ, s=1, alpha=0.1, label='target')
    axes[plot_idx].set_title('After STalign Affine', fontsize=14)
    axes[plot_idx].legend(markerscale=5)
    axes[plot_idx].set_aspect('equal')
    plot_idx += 1

if has_squidpy:
    axes[plot_idx].scatter(xI_squidpy, yI_squidpy, s=1, alpha=0.1, label='source Squidpy-aligned')
    axes[plot_idx].scatter(xJ, yJ, s=1, alpha=0.1, label='target')
    axes[plot_idx].set_title('After Squidpy Affine', fontsize=14)
    axes[plot_idx].legend(markerscale=5)
    axes[plot_idx].set_aspect('equal')

plt.tight_layout()
output_fig = OUTPUT_DIR / 'merfish_merfish_affine_comparison.png'
plt.savefig(output_fig, dpi=150)
plt.close()
print(f"Figure saved to {output_fig}")

# Quantitative comparison
if has_stalign and has_squidpy:
    print("\n" + "=" * 60)
    print("Quantitative Comparison")
    print("=" * 60)

    target_centroid = np.array([np.mean(xJ), np.mean(yJ)])
    dist_before = np.linalg.norm([np.mean(xI), np.mean(yI)] - target_centroid)
    dist_stalign = np.linalg.norm([np.mean(xI_stalign), np.mean(yI_stalign)] - target_centroid)
    dist_squidpy = np.linalg.norm([np.mean(xI_squidpy), np.mean(yI_squidpy)] - target_centroid)

    print(f"\nCentroid distance to target:")
    print(f"  Before:   {dist_before:.4f}")
    print(f"  STalign:  {dist_stalign:.4f}")
    print(f"  Squidpy:  {dist_squidpy:.4f}")

    diff = np.sqrt((xI_stalign - xI_squidpy)**2 + (yI_stalign - yI_squidpy)**2)
    print(f"\nDifference between methods:")
    print(f"  Mean: {np.mean(diff):.4f}, Max: {np.max(diff):.4f}, Std: {np.std(diff):.4f}")

print("\nDone!")
