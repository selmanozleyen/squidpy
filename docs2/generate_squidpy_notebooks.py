#!/usr/bin/env python3
"""Generate squidpy-compatible versions of all STalign tutorial notebooks.

For point-to-point alignments, uses moscot via sq.experimental.tl.align().
For point-to-image alignments, uses STalign via sq.experimental.tl.align().
For 3D atlas alignments, keeps STalign-native calls (not yet in squidpy API).
"""

import json
import os

NOTEBOOK_DIR = os.path.join(os.path.dirname(__file__), "notebooks")


def make_nb(cells):
    """Create a notebook dict from a list of (cell_type, source_lines) tuples."""
    nb_cells = []
    for cell_type, source in cells:
        lines = []
        for i, line in enumerate(source.split("\n")):
            if i < len(source.split("\n")) - 1:
                lines.append(line + "\n")
            else:
                lines.append(line)
        cell = {
            "cell_type": cell_type,
            "metadata": {},
            "source": lines,
        }
        if cell_type == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        nb_cells.append(cell)
    return {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11.0",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_nb(name, cells):
    """Write notebook JSON to disk."""
    path = os.path.join(NOTEBOOK_DIR, name)
    nb = make_nb(cells)
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"  wrote {path}")


# ============================================================================
# 1. heart-alignment-varying-thickness  (point-to-point → moscot)
# ============================================================================
def heart_alignment_varying_thickness():
    return [
        ("markdown", "# Aligning heart ST data from ISS (squidpy + moscot)\n\nSerial sections of 6.5 PCW human heart from the Human Cell Atlas https://doi.org/10.1016/j.cell.2019.11.025\n\nThis notebook uses `squidpy` with the `moscot` backend for point-to-point alignment via optimal transport."),
        ("code", "import numpy as np\nimport matplotlib.pyplot as plt\nimport pandas as pd\nimport anndata as ad\nimport squidpy as sq\n\nplt.rcParams[\"figure.figsize\"] = (12, 10)"),
        ("markdown", "## Load source data\n\nWe read in the cell information for the first dataset."),
        ("code", "# Single cell data 1\nfname = '../heart_data/3_CN73_D2.csv.gz'\ndf1 = pd.read_csv(fname)\nprint(df1.head())"),
        ("code", "# Create AnnData for source\ncoords_source = np.column_stack([df1['x'].values, df1['y'].values])\nadata_source = ad.AnnData(\n    X=np.zeros((len(coords_source), 1)),\n    obs=df1,\n)\nadata_source.obsm['spatial'] = coords_source\nprint(f\"Source: {adata_source.n_obs} cells\")"),
        ("code", "# Plot source\nfig, ax = plt.subplots()\nax.scatter(adata_source.obsm['spatial'][:, 0],\n           adata_source.obsm['spatial'][:, 1],\n           s=1, alpha=0.2)\nax.set_title('Source')\nax.set_aspect('equal')"),
        ("markdown", "## Load target data"),
        ("code", "# Single cell data 2\nfname = '../heart_data/4_CN73_C2.csv.gz'\ndf2 = pd.read_csv(fname, skiprows=[1])\nprint(df2.head())"),
        ("code", "# Create AnnData for target\ncoords_target = np.column_stack([df2['x'].values, df2['y'].values])\nadata_target = ad.AnnData(\n    X=np.zeros((len(coords_target), 1)),\n    obs=df2,\n)\nadata_target.obsm['spatial'] = coords_target\nprint(f\"Target: {adata_target.n_obs} cells\")"),
        ("code", "# Plot both before alignment\nfig, ax = plt.subplots()\nax.scatter(adata_source.obsm['spatial'][:, 0],\n           adata_source.obsm['spatial'][:, 1],\n           s=1, alpha=0.1, label='source')\nax.scatter(adata_target.obsm['spatial'][:, 0],\n           adata_target.obsm['spatial'][:, 1],\n           s=1, alpha=0.2, label='target')\nax.legend(markerscale=10)\nax.set_aspect('equal')"),
        ("markdown", "## Align using moscot (optimal transport)\n\n`sq.experimental.tl.align` automatically selects the moscot backend for point-to-point alignment. Moscot uses optimal transport to find a soft correspondence between the two point clouds."),
        ("code", "# Align source to target using moscot optimal transport\nsq.experimental.tl.align(\n    adata_source,\n    adata_target,\n    method='optimal_transport',\n    verbose=True,\n)"),
        ("markdown", "## Visualize results"),
        ("code", "# Plot results\naligned = adata_source.obsm['spatial_aligned']\n\nfig, ax = plt.subplots()\nax.scatter(adata_source.obsm['spatial'][:, 0],\n           adata_source.obsm['spatial'][:, 1],\n           s=1, alpha=0.1, label='source')\nax.scatter(aligned[:, 0], aligned[:, 1],\n           s=1, alpha=0.2, label='source aligned')\nax.scatter(adata_target.obsm['spatial'][:, 0],\n           adata_target.obsm['spatial'][:, 1],\n           s=1, alpha=0.2, label='target')\n\nlgnd = plt.legend(loc='upper right', scatterpoints=1, fontsize=10)\nfor handle in lgnd.legend_handles:\n    handle.set_sizes([10.0])"),
        ("code", "# Side-by-side comparison\nfig, axes = plt.subplots(1, 2, figsize=(20, 8))\n\naxes[0].scatter(adata_source.obsm['spatial'][:, 0],\n                adata_source.obsm['spatial'][:, 1],\n                s=1, alpha=0.1, label='source')\naxes[0].scatter(adata_target.obsm['spatial'][:, 0],\n                adata_target.obsm['spatial'][:, 1],\n                s=1, alpha=0.1, label='target')\naxes[0].set_title('Before alignment')\naxes[0].legend(markerscale=10)\naxes[0].set_aspect('equal')\n\naxes[1].scatter(aligned[:, 0], aligned[:, 1],\n                s=1, alpha=0.1, label='source aligned')\naxes[1].scatter(adata_target.obsm['spatial'][:, 0],\n                adata_target.obsm['spatial'][:, 1],\n                s=1, alpha=0.1, label='target')\naxes[1].set_title('After alignment')\naxes[1].legend(markerscale=10)\naxes[1].set_aspect('equal')"),
        ("code", "# Check alignment metadata\nprint(adata_source.uns['spatial_alignment'])"),
    ]


# ============================================================================
# 2. heart-alignment  (point-to-point → moscot)
# ============================================================================
def heart_alignment():
    return [
        ("markdown", "# Aligning heart ST data from ISS (squidpy + moscot)\n\nSerial sections of 6.5 PCW human heart from the Human Cell Atlas https://doi.org/10.1016/j.cell.2019.11.025\n\nThis notebook uses `squidpy` with the `moscot` backend for point-to-point alignment via optimal transport."),
        ("code", "import numpy as np\nimport matplotlib.pyplot as plt\nimport pandas as pd\nimport anndata as ad\nimport squidpy as sq\n\nplt.rcParams[\"figure.figsize\"] = (12, 10)"),
        ("markdown", "## Load source data"),
        ("code", "# Single cell data 1\nfname = '../heart_data/CN73_E1.csv.gz'\ndf1 = pd.read_csv(fname)\nprint(df1.head())"),
        ("code", "# Create AnnData for source\ncoords_source = np.column_stack([df1['x'].values, df1['y'].values])\nadata_source = ad.AnnData(\n    X=np.zeros((len(coords_source), 1)),\n    obs=df1,\n)\nadata_source.obsm['spatial'] = coords_source\nprint(f\"Source: {adata_source.n_obs} cells\")"),
        ("code", "# Plot source\nfig, ax = plt.subplots()\nax.scatter(adata_source.obsm['spatial'][:, 0],\n           adata_source.obsm['spatial'][:, 1],\n           s=1, alpha=0.2)\nax.set_title('Source')\nax.set_aspect('equal')"),
        ("markdown", "## Load target data"),
        ("code", "# Single cell data 2\nfname = '../heart_data/CN73_E2.csv.gz'\ndf2 = pd.read_csv(fname, skiprows=[1])\nprint(df2.head())"),
        ("code", "# Create AnnData for target\ncoords_target = np.column_stack([df2['x'].values, df2['y'].values])\nadata_target = ad.AnnData(\n    X=np.zeros((len(coords_target), 1)),\n    obs=df2,\n)\nadata_target.obsm['spatial'] = coords_target\nprint(f\"Target: {adata_target.n_obs} cells\")"),
        ("code", "# Plot both before alignment\nfig, ax = plt.subplots()\nax.scatter(adata_source.obsm['spatial'][:, 0],\n           adata_source.obsm['spatial'][:, 1],\n           s=1, alpha=0.1, label='source')\nax.scatter(adata_target.obsm['spatial'][:, 0],\n           adata_target.obsm['spatial'][:, 1],\n           s=1, alpha=0.2, label='target')\nax.legend(markerscale=10)\nax.set_aspect('equal')"),
        ("markdown", "## Align using moscot (optimal transport)\n\n`sq.experimental.tl.align` automatically selects the moscot backend for point-to-point alignment."),
        ("code", "# Align source to target using moscot optimal transport\nsq.experimental.tl.align(\n    adata_source,\n    adata_target,\n    method='optimal_transport',\n    verbose=True,\n)"),
        ("markdown", "## Visualize results"),
        ("code", "# Plot results\naligned = adata_source.obsm['spatial_aligned']\n\nfig, ax = plt.subplots()\nax.scatter(adata_source.obsm['spatial'][:, 0],\n           adata_source.obsm['spatial'][:, 1],\n           s=1, alpha=0.1, label='source')\nax.scatter(aligned[:, 0], aligned[:, 1],\n           s=1, alpha=0.2, label='source aligned')\nax.scatter(adata_target.obsm['spatial'][:, 0],\n           adata_target.obsm['spatial'][:, 1],\n           s=1, alpha=0.2, label='target')\n\nlgnd = plt.legend(loc='upper right', scatterpoints=1, fontsize=10)\nfor handle in lgnd.legend_handles:\n    handle.set_sizes([10.0])"),
        ("code", "# Side-by-side comparison\nfig, axes = plt.subplots(1, 2, figsize=(20, 8))\n\naxes[0].scatter(adata_source.obsm['spatial'][:, 0],\n                adata_source.obsm['spatial'][:, 1],\n                s=1, alpha=0.1, label='source')\naxes[0].scatter(adata_target.obsm['spatial'][:, 0],\n                adata_target.obsm['spatial'][:, 1],\n                s=1, alpha=0.1, label='target')\naxes[0].set_title('Before alignment')\naxes[0].legend(markerscale=10)\naxes[0].set_aspect('equal')\n\naxes[1].scatter(aligned[:, 0], aligned[:, 1],\n                s=1, alpha=0.1, label='source aligned')\naxes[1].scatter(adata_target.obsm['spatial'][:, 0],\n                adata_target.obsm['spatial'][:, 1],\n                s=1, alpha=0.1, label='target')\naxes[1].set_title('After alignment')\naxes[1].legend(markerscale=10)\naxes[1].set_aspect('equal')"),
        ("code", "print(adata_source.uns['spatial_alignment'])"),
    ]


# ============================================================================
# 3. merfish-merfish-alignment-affine-only-with-points (point-to-point → moscot)
# ============================================================================
def merfish_merfish_alignment_affine_only_with_points():
    return [
        ("markdown", "# Aligning two coronal sections of adult mouse brain from MERFISH (squidpy + moscot)\n\nIn this notebook, we align two single cell resolution spatial transcriptomics datasets of coronal sections of the adult mouse brain from matched locations with respect to bregma assayed by MERFISH.\n\nThe original notebook used STalign affine-only alignment with manually picked landmark points. Here we use `squidpy` with the `moscot` backend for point-to-point alignment via optimal transport, which does not require manual landmarks."),
        ("code", "import numpy as np\nimport matplotlib.pyplot as plt\nimport pandas as pd\nimport anndata as ad\nimport squidpy as sq\n\nplt.rcParams[\"figure.figsize\"] = (12, 10)"),
        ("markdown", "## Load source data"),
        ("code", "# Single cell data 1\nfname = '../merfish_data/datasets_mouse_brain_map_BrainReceptorShowcase_Slice2_Replicate3_cell_metadata_S2R3.csv.gz'\ndf1 = pd.read_csv(fname)\nprint(df1.head())"),
        ("code", "# Create AnnData for source\ncoords_source = np.column_stack([df1['center_x'].values, df1['center_y'].values])\nadata_source = ad.AnnData(\n    X=np.zeros((len(coords_source), 1)),\n    obs=df1,\n)\nadata_source.obsm['spatial'] = coords_source\n\n# Plot\nfig, ax = plt.subplots()\nax.scatter(coords_source[:, 0], coords_source[:, 1], s=1, alpha=0.2, label='source')\nax.legend(markerscale=10)\nax.set_aspect('equal')"),
        ("markdown", "## Load target data"),
        ("code", "# Single cell data 2\nfname = '../merfish_data/datasets_mouse_brain_map_BrainReceptorShowcase_Slice2_Replicate2_cell_metadata_S2R2.csv.gz'\ndf2 = pd.read_csv(fname)\n\ncoords_target = np.column_stack([df2['center_x'].values, df2['center_y'].values])\nadata_target = ad.AnnData(\n    X=np.zeros((len(coords_target), 1)),\n    obs=df2,\n)\nadata_target.obsm['spatial'] = coords_target\n\n# Plot\nfig, ax = plt.subplots()\nax.scatter(coords_target[:, 0], coords_target[:, 1], s=1, alpha=0.2, c='#ff7f0e', label='target')\nax.legend(markerscale=10)\nax.set_aspect('equal')"),
        ("code", "# Plot both before alignment\nfig, ax = plt.subplots()\nax.scatter(coords_source[:, 0], coords_source[:, 1], s=1, alpha=0.2, label='source')\nax.scatter(coords_target[:, 0], coords_target[:, 1], s=1, alpha=0.1, label='target')\nax.legend(markerscale=10)\nax.set_aspect('equal')"),
        ("markdown", "## Align using moscot (optimal transport)\n\nUnlike the STalign affine-only approach which requires manual landmark selection, moscot finds an optimal transport plan between the two point clouds automatically."),
        ("code", "# Align source to target using moscot\nsq.experimental.tl.align(\n    adata_source,\n    adata_target,\n    method='optimal_transport',\n    verbose=True,\n)"),
        ("markdown", "## Visualize results"),
        ("code", "# Plot results\naligned = adata_source.obsm['spatial_aligned']\n\nfig, ax = plt.subplots()\nax.scatter(coords_source[:, 0], coords_source[:, 1],\n           s=1, alpha=0.1, label='source')\nax.scatter(aligned[:, 0], aligned[:, 1],\n           s=1, alpha=0.1, label='source aligned')\nax.scatter(coords_target[:, 0], coords_target[:, 1],\n           s=1, alpha=0.1, label='target')\nax.legend(markerscale=10)\nax.set_aspect('equal')"),
        ("code", "# 2x2 panel summary\nfig, ax = plt.subplots(2, 2, figsize=(16, 14))\n\nax[0][0].scatter(coords_source[:, 0], coords_source[:, 1], s=1, alpha=0.1, label='source cells')\nax[0][0].set_title('Source')\nax[0][0].set_aspect('equal')\nax[0][0].legend(markerscale=10)\n\nax[0][1].scatter(coords_target[:, 0], coords_target[:, 1], s=1, alpha=0.1, c='orange', label='target cells')\nax[0][1].set_title('Target')\nax[0][1].set_aspect('equal')\nax[0][1].legend(markerscale=10)\n\nax[1][0].scatter(aligned[:, 0], aligned[:, 1], s=1, alpha=0.1, label='aligned source cells')\nax[1][0].set_title('Aligned Source')\nax[1][0].set_aspect('equal')\nax[1][0].legend(markerscale=10)\n\nax[1][1].scatter(coords_target[:, 0], coords_target[:, 1], s=1, alpha=0.1, c='orange', label='target')\nax[1][1].scatter(aligned[:, 0], aligned[:, 1], s=1, alpha=0.1, label='aligned source')\nax[1][1].set_title('Overlay')\nax[1][1].set_aspect('equal')\nax[1][1].legend(markerscale=10)"),
        ("markdown", "## Save results"),
        ("code", "# Save aligned coordinates\ndf_aligned = pd.DataFrame({\n    'aligned_x': aligned[:, 0],\n    'aligned_y': aligned[:, 1],\n})\nresults = pd.concat([df1, df_aligned], axis=1)\nresults.head()"),
    ]


# ============================================================================
# 4. merfish-merfish-alignment-affine-only (point-to-point → moscot)
# ============================================================================
def merfish_merfish_alignment_affine_only():
    return [
        ("markdown", "# Aligning two coronal sections of adult mouse brain from MERFISH (squidpy + moscot)\n\nIn this notebook, we align two single cell resolution spatial transcriptomics datasets of coronal sections of the adult mouse brain from matched locations with respect to bregma assayed by MERFISH.\n\nThe original notebook used STalign with affine-only + manual rotation initialization. Here we use `squidpy` with the `moscot` backend which handles alignment automatically."),
        ("code", "import numpy as np\nimport matplotlib.pyplot as plt\nimport pandas as pd\nimport anndata as ad\nimport squidpy as sq\n\nplt.rcParams[\"figure.figsize\"] = (12, 10)"),
        ("markdown", "## Load source data"),
        ("code", "# Single cell data 1\nfname = '../merfish_data/datasets_mouse_brain_map_BrainReceptorShowcase_Slice2_Replicate3_cell_metadata_S2R3.csv.gz'\ndf1 = pd.read_csv(fname)\n\ncoords_source = np.column_stack([df1['center_x'].values, df1['center_y'].values])\nadata_source = ad.AnnData(\n    X=np.zeros((len(coords_source), 1)),\n    obs=df1,\n)\nadata_source.obsm['spatial'] = coords_source\n\nfig, ax = plt.subplots()\nax.scatter(coords_source[:, 0], coords_source[:, 1], s=1, alpha=0.2, label='source')\nax.legend(markerscale=10)"),
        ("markdown", "## Load target data"),
        ("code", "# Single cell data 2\nfname = '../merfish_data/datasets_mouse_brain_map_BrainReceptorShowcase_Slice2_Replicate2_cell_metadata_S2R2.csv.gz'\ndf2 = pd.read_csv(fname)\n\ncoords_target = np.column_stack([df2['center_x'].values, df2['center_y'].values])\nadata_target = ad.AnnData(\n    X=np.zeros((len(coords_target), 1)),\n    obs=df2,\n)\nadata_target.obsm['spatial'] = coords_target\n\nfig, ax = plt.subplots()\nax.scatter(coords_target[:, 0], coords_target[:, 1], s=1, alpha=0.2, c='#ff7f0e', label='target')\nax.legend(markerscale=10)"),
        ("code", "# Plot overlay before alignment\nfig, ax = plt.subplots()\nax.scatter(coords_source[:, 0], coords_source[:, 1], s=1, alpha=0.2, label='source')\nax.scatter(coords_target[:, 0], coords_target[:, 1], s=1, alpha=0.1, label='target')\nax.legend(markerscale=10)"),
        ("markdown", "## Align using moscot (optimal transport)\n\nMoscot handles alignment without needing manual rotation angles or pre-transforms."),
        ("code", "# Align\nsq.experimental.tl.align(\n    adata_source,\n    adata_target,\n    method='optimal_transport',\n    verbose=True,\n)"),
        ("markdown", "## Visualize results"),
        ("code", "aligned = adata_source.obsm['spatial_aligned']\n\nfig, ax = plt.subplots()\nax.scatter(coords_source[:, 0], coords_source[:, 1], s=1, alpha=0.1, label='source')\nax.scatter(aligned[:, 0], aligned[:, 1], s=1, alpha=0.1, label='source aligned')\nax.scatter(coords_target[:, 0], coords_target[:, 1], s=1, alpha=0.1, label='target')\nax.legend(markerscale=10)"),
        ("code", "fig, ax = plt.subplots(1, 2, figsize=(20, 8))\nax[0].scatter(coords_source[:, 0], coords_source[:, 1], s=1, alpha=0.1, label='source')\nax[0].scatter(coords_target[:, 0], coords_target[:, 1], s=1, alpha=0.1, label='target')\nax[0].legend(markerscale=10, loc='lower left')\nax[0].set_title('Before alignment')\n\nax[1].scatter(aligned[:, 0], aligned[:, 1], s=1, alpha=0.1, label='source aligned')\nax[1].scatter(coords_target[:, 0], coords_target[:, 1], s=1, alpha=0.1, label='target')\nax[1].legend(markerscale=10, loc='lower left')\nax[1].set_title('After alignment')"),
        ("markdown", "## Save results"),
        ("code", "df_aligned = pd.DataFrame({'aligned_x': aligned[:, 0], 'aligned_y': aligned[:, 1]})\nresults = pd.concat([df1, df_aligned], axis=1)\nresults.head()"),
    ]


# ============================================================================
# 5. merfish-merfish-alignment-simulation (point-to-point → moscot)
# ============================================================================
def merfish_merfish_alignment_simulation():
    return [
        ("markdown", "## Aligning single-cell spatial transcriptomics datasets simulated with non-linear distortions (squidpy + moscot)\n\nIn this notebook, we simulate a warped ST dataset and evaluate how well moscot (via squidpy) can align it to the original.\n\nThe original notebook used STalign LDDMM. Here we use `moscot` optimal transport, which provides a different approach to spatial alignment."),
        ("code", "import numpy as np\nimport matplotlib.pyplot as plt\nimport pandas as pd\nimport anndata as ad\nimport squidpy as sq\n\nplt.rcParams[\"figure.figsize\"] = (12, 10)"),
        ("markdown", "### Load and warp data"),
        ("code", "# Load MERFISH data\nfname = '../merfish_data/datasets_mouse_brain_map_BrainReceptorShowcase_Slice2_Replicate3_cell_metadata_S2R3.csv.gz'\ndf1 = pd.read_csv(fname)\n\nxI0 = np.array(df1['center_x'])\nyI0 = np.array(df1['center_y'])\n\nfig, ax = plt.subplots()\nax.scatter(xI0, yI0, s=1, alpha=0.2, label='source init')\nax.legend(markerscale=10)"),
        ("code", "# Warp the coordinates (simulating experimental distortion)\nxI = pow(xI0, 1.25) / 10 + 500\nyI = pow(yI0, 1.25) / 10 + 500\n\nfig, ax = plt.subplots()\nax.scatter(xI, yI, s=1, alpha=0.2, label='source warped')\nax.legend(markerscale=10)"),
        ("code", "# Target is the original unwarped coordinates\nxJ = xI0\nyJ = yI0\n\nfig, ax = plt.subplots()\nax.scatter(xJ, yJ, s=1, alpha=0.2, c='#ff7f0e', label='target')\nax.legend(markerscale=10)"),
        ("code", "# Plot overlay\nfig, ax = plt.subplots()\nax.scatter(xI, yI, s=1, alpha=0.2, label='source warped')\nax.scatter(xJ, yJ, s=1, alpha=0.1, label='target')\nax.legend(markerscale=10)"),
        ("markdown", "### Create AnnData objects"),
        ("code", "# Source: warped coordinates\nadata_source = ad.AnnData(X=np.zeros((len(xI), 1)))\nadata_source.obsm['spatial'] = np.column_stack([xI, yI])\n\n# Target: original coordinates\nadata_target = ad.AnnData(X=np.zeros((len(xJ), 1)))\nadata_target.obsm['spatial'] = np.column_stack([xJ, yJ])"),
        ("markdown", "### Perform alignment with moscot"),
        ("code", "sq.experimental.tl.align(\n    adata_source,\n    adata_target,\n    method='optimal_transport',\n    verbose=True,\n)"),
        ("markdown", "### Visualize results"),
        ("code", "aligned = adata_source.obsm['spatial_aligned']\n\nfig, ax = plt.subplots()\nax.scatter(aligned[:, 0], aligned[:, 1], s=1, alpha=0.1, label='source warped aligned')\nax.scatter(xJ, yJ, s=1, alpha=0.1, label='target')\nax.legend(markerscale=10)"),
        ("code", "plt.rcParams[\"figure.figsize\"] = (12, 5)\nfig, ax = plt.subplots(1, 2)\nax[0].scatter(xI, yI, s=0.5, alpha=0.1, label='source warped')\nax[0].scatter(xJ, yJ, s=0.5, alpha=0.1, label='target')\nax[1].scatter(aligned[:, 0], aligned[:, 1], s=0.5, alpha=0.1, label='source warped aligned')\nax[1].scatter(xJ, yJ, s=0.5, alpha=0.1, label='target')\nax[0].legend(markerscale=10, loc='lower left')\nax[1].legend(markerscale=10, loc='lower left')"),
        ("markdown", "### Evaluate performance"),
        ("code", "from sklearn.metrics import mean_squared_error\n\nerr_init = mean_squared_error([xI0, yI0], [xI, yI], squared=False)\nprint(f\"RMSE before alignment: {err_init:.2f}\")\n\nerr_aligned = mean_squared_error([xI0, yI0], [aligned[:, 0], aligned[:, 1]], squared=False)\nprint(f\"RMSE after alignment: {err_aligned:.2f}\")"),
    ]


# ============================================================================
# 6. merfish-merfish-alignment-using-L-T (point-to-point → moscot)
# ============================================================================
def merfish_merfish_alignment_using_LT():
    return [
        ("markdown", "# Aligning two coronal sections of adult mouse brain from MERFISH (squidpy + moscot)\n\nIn this notebook, we align two single cell resolution spatial transcriptomics datasets of coronal sections of the adult mouse brain from matched locations with respect to bregma from different individuals assayed by MERFISH.\n\nThe original notebook used STalign LDDMM with manual L and T initialization. Here we use `squidpy` with `moscot` optimal transport."),
        ("code", "import numpy as np\nimport matplotlib.pyplot as plt\nimport pandas as pd\nimport anndata as ad\nimport squidpy as sq\n\nplt.rcParams[\"figure.figsize\"] = (12, 10)"),
        ("markdown", "## Load data"),
        ("code", "# Source\nfname = '../merfish_data/datasets_mouse_brain_map_BrainReceptorShowcase_Slice2_Replicate3_cell_metadata_S2R3.csv.gz'\ndf1 = pd.read_csv(fname)\n\ncoords_source = np.column_stack([df1['center_x'].values, df1['center_y'].values])\nadata_source = ad.AnnData(X=np.zeros((len(coords_source), 1)), obs=df1)\nadata_source.obsm['spatial'] = coords_source\n\nfig, ax = plt.subplots()\nax.scatter(coords_source[:, 0], coords_source[:, 1], s=1, alpha=0.2, label='source')\nax.legend(markerscale=10)"),
        ("code", "# Target\nfname = '../merfish_data/datasets_mouse_brain_map_BrainReceptorShowcase_Slice2_Replicate2_cell_metadata_S2R2.csv.gz'\ndf2 = pd.read_csv(fname)\n\ncoords_target = np.column_stack([df2['center_x'].values, df2['center_y'].values])\nadata_target = ad.AnnData(X=np.zeros((len(coords_target), 1)), obs=df2)\nadata_target.obsm['spatial'] = coords_target\n\nfig, ax = plt.subplots()\nax.scatter(coords_target[:, 0], coords_target[:, 1], s=1, alpha=0.2, c='#ff7f0e', label='target')\nax.legend(markerscale=10)"),
        ("code", "# Plot overlay\nfig, ax = plt.subplots()\nax.scatter(coords_source[:, 0], coords_source[:, 1], s=1, alpha=0.2, label='source')\nax.scatter(coords_target[:, 0], coords_target[:, 1], s=1, alpha=0.1, label='target')\nax.legend(markerscale=10)"),
        ("markdown", "## Align using moscot"),
        ("code", "sq.experimental.tl.align(\n    adata_source,\n    adata_target,\n    method='optimal_transport',\n    verbose=True,\n)"),
        ("markdown", "## Visualize results"),
        ("code", "aligned = adata_source.obsm['spatial_aligned']\n\nfig, ax = plt.subplots()\nax.scatter(coords_source[:, 0], coords_source[:, 1], s=1, alpha=0.1, label='source')\nax.scatter(aligned[:, 0], aligned[:, 1], s=1, alpha=0.1, label='source aligned')\nax.scatter(coords_target[:, 0], coords_target[:, 1], s=1, alpha=0.1, label='target')\nax.legend(markerscale=10)"),
        ("code", "fig, ax = plt.subplots(1, 2, figsize=(20, 8))\nax[0].scatter(coords_source[:, 0], coords_source[:, 1], s=1, alpha=0.1, label='source')\nax[0].scatter(coords_target[:, 0], coords_target[:, 1], s=1, alpha=0.1, label='target')\nax[0].legend(markerscale=10, loc='lower left')\nax[0].set_title('Before alignment')\n\nax[1].scatter(aligned[:, 0], aligned[:, 1], s=1, alpha=0.1, label='source aligned')\nax[1].scatter(coords_target[:, 0], coords_target[:, 1], s=1, alpha=0.1, label='target')\nax[1].legend(markerscale=10, loc='lower left')\nax[1].set_title('After alignment')"),
        ("markdown", "## Save results"),
        ("code", "df_aligned = pd.DataFrame({'aligned_x': aligned[:, 0], 'aligned_y': aligned[:, 1]})\nresults = pd.concat([df1, df_aligned], axis=1)\nresults.head()"),
    ]


# ============================================================================
# 7. merfish-merfish-alignment (point-to-point → moscot)
# ============================================================================
def merfish_merfish_alignment():
    return [
        ("markdown", "# Aligning two coronal sections of adult mouse brain from MERFISH (squidpy + moscot)\n\nIn this notebook, we align two single cell resolution spatial transcriptomics datasets of coronal sections of the adult mouse brain from matched locations with respect to bregma assayed by MERFISH.\n\nThe original notebook used STalign LDDMM with 10,000 iterations. Here we use `squidpy` with `moscot` optimal transport for a simpler workflow."),
        ("code", "import numpy as np\nimport matplotlib.pyplot as plt\nimport pandas as pd\nimport anndata as ad\nimport squidpy as sq\n\nplt.rcParams[\"figure.figsize\"] = (12, 10)"),
        ("markdown", "## Load data"),
        ("code", "# Source\nfname = '../merfish_data/datasets_mouse_brain_map_BrainReceptorShowcase_Slice2_Replicate3_cell_metadata_S2R3.csv.gz'\ndf1 = pd.read_csv(fname)\n\ncoords_source = np.column_stack([df1['center_x'].values, df1['center_y'].values])\nadata_source = ad.AnnData(X=np.zeros((len(coords_source), 1)), obs=df1)\nadata_source.obsm['spatial'] = coords_source\n\nfig, ax = plt.subplots()\nax.scatter(coords_source[:, 0], coords_source[:, 1], s=1, alpha=0.2, label='source')\nax.legend(markerscale=10)"),
        ("code", "# Target\nfname = '../merfish_data/datasets_mouse_brain_map_BrainReceptorShowcase_Slice2_Replicate2_cell_metadata_S2R2.csv.gz'\ndf2 = pd.read_csv(fname)\n\ncoords_target = np.column_stack([df2['center_x'].values, df2['center_y'].values])\nadata_target = ad.AnnData(X=np.zeros((len(coords_target), 1)), obs=df2)\nadata_target.obsm['spatial'] = coords_target\n\nfig, ax = plt.subplots()\nax.scatter(coords_target[:, 0], coords_target[:, 1], s=1, alpha=0.2, c='#ff7f0e', label='target')\nax.legend(markerscale=10)"),
        ("code", "# Overlay\nfig, ax = plt.subplots()\nax.scatter(coords_source[:, 0], coords_source[:, 1], s=1, alpha=0.2, label='source')\nax.scatter(coords_target[:, 0], coords_target[:, 1], s=1, alpha=0.1, label='target')\nax.legend(markerscale=10)"),
        ("markdown", "## Align using moscot"),
        ("code", "sq.experimental.tl.align(\n    adata_source,\n    adata_target,\n    method='optimal_transport',\n    verbose=True,\n)"),
        ("markdown", "## Visualize"),
        ("code", "aligned = adata_source.obsm['spatial_aligned']\n\nfig, ax = plt.subplots()\nax.scatter(coords_source[:, 0], coords_source[:, 1], s=1, alpha=0.1, label='source')\nax.scatter(aligned[:, 0], aligned[:, 1], s=1, alpha=0.1, label='source aligned')\nax.scatter(coords_target[:, 0], coords_target[:, 1], s=1, alpha=0.1, label='target')\nax.legend(markerscale=10)"),
        ("code", "fig, ax = plt.subplots(1, 2, figsize=(20, 8))\nax[0].scatter(coords_source[:, 0], coords_source[:, 1], s=1, alpha=0.1, label='source')\nax[0].scatter(coords_target[:, 0], coords_target[:, 1], s=1, alpha=0.1, label='target')\nax[0].legend(markerscale=10, loc='lower left')\nax[0].set_title('Before alignment')\nax[1].scatter(aligned[:, 0], aligned[:, 1], s=1, alpha=0.1, label='source aligned')\nax[1].scatter(coords_target[:, 0], coords_target[:, 1], s=1, alpha=0.1, label='target')\nax[1].legend(markerscale=10, loc='lower left')\nax[1].set_title('After alignment')"),
        ("markdown", "## Save results"),
        ("code", "df_aligned = pd.DataFrame({'aligned_x': aligned[:, 0], 'aligned_y': aligned[:, 1]})\nresults = pd.concat([df1, df_aligned], axis=1)\nresults.head()"),
    ]


# ============================================================================
# 8. merfish-xenium-alignment (point-to-point → moscot)
# ============================================================================
def merfish_xenium_alignment():
    return [
        ("markdown", "# Aligning full coronal sections of adult mouse brain from MERFISH and Xenium (squidpy + moscot)\n\nIn this notebook, we align two single cell resolution spatial transcriptomics datasets of full coronal sections of the adult mouse brain from approximately the same locations assayed by MERFISH and Xenium.\n\nThe original notebook used STalign LDDMM. Here we use `squidpy` with `moscot` optimal transport."),
        ("code", "import numpy as np\nimport matplotlib.pyplot as plt\nimport pandas as pd\nimport anndata as ad\nimport squidpy as sq\n\nplt.rcParams[\"figure.figsize\"] = (12, 10)"),
        ("markdown", "## Load data"),
        ("code", "# Source: MERFISH\nfname = '../../merfish_data/datasets_mouse_brain_map_BrainReceptorShowcase_Slice2_Replicate3_cell_metadata_S2R3.csv.gz'\ndf1 = pd.read_csv(fname)\n\ncoords_source = np.column_stack([df1['center_x'].values, df1['center_y'].values])\nadata_source = ad.AnnData(X=np.zeros((len(coords_source), 1)), obs=df1)\nadata_source.obsm['spatial'] = coords_source\n\nfig, ax = plt.subplots()\nax.scatter(coords_source[:, 0], coords_source[:, 1], s=1, alpha=0.2)\nax.set_title('MERFISH (source)')"),
        ("code", "# Target: Xenium\nfname = '../../xenium_data/Xenium_V1_FF_Mouse_Brain_MultiSection_1_cells.csv.gz'\ndf2 = pd.read_csv(fname)\n\ncoords_target = np.column_stack([df2['x_centroid'].values, df2['y_centroid'].values])\nadata_target = ad.AnnData(X=np.zeros((len(coords_target), 1)), obs=df2)\nadata_target.obsm['spatial'] = coords_target\n\nfig, ax = plt.subplots()\nax.scatter(coords_target[:, 0], coords_target[:, 1], s=1, alpha=0.2, c='#ff7f0e')\nax.set_title('Xenium (target)')"),
        ("code", "# Overlay\nfig, ax = plt.subplots()\nax.scatter(coords_source[:, 0], coords_source[:, 1], s=1, alpha=0.2, label='MERFISH')\nax.scatter(coords_target[:, 0], coords_target[:, 1], s=1, alpha=0.1, label='Xenium')\nax.legend(markerscale=10)"),
        ("markdown", "## Align using moscot"),
        ("code", "sq.experimental.tl.align(\n    adata_source,\n    adata_target,\n    method='optimal_transport',\n    verbose=True,\n)"),
        ("markdown", "## Visualize"),
        ("code", "aligned = adata_source.obsm['spatial_aligned']\n\nfig, ax = plt.subplots()\nax.scatter(coords_source[:, 0], coords_source[:, 1], s=1, alpha=0.2, label='MERFISH')\nax.scatter(aligned[:, 0], aligned[:, 1], s=1, alpha=0.1, label='MERFISH aligned')\nax.scatter(coords_target[:, 0], coords_target[:, 1], s=1, alpha=0.1, label='Xenium')\nax.legend(markerscale=10)"),
    ]


# ============================================================================
# 9. visium-visium-alignment-affine-only (point-to-point → moscot)
# ============================================================================
def visium_visium_alignment_affine_only():
    return [
        ("markdown", "# Aligning two Visium datasets (squidpy + moscot)\n\nIn this notebook, we align two spot resolution spatial transcriptomics datasets of serial sections of breast cancer.\n\nThe original notebook used STalign affine-only. Here we use `squidpy` with `moscot` optimal transport."),
        ("code", "import numpy as np\nimport matplotlib.pyplot as plt\nimport pandas as pd\nimport anndata as ad\nimport squidpy as sq\n\nplt.rcParams[\"figure.figsize\"] = (12, 10)"),
        ("markdown", "## Load data"),
        ("code", "# Source\nfname = '../visium_data/slice1_coor.csv'\ndf1 = pd.read_csv(fname)\n\nxI = np.array(df1[df1.columns[0]])\nyI = np.array(df1[df1.columns[1]])\n\ncoords_source = np.column_stack([xI, yI])\nadata_source = ad.AnnData(X=np.zeros((len(coords_source), 1)))\nadata_source.obsm['spatial'] = coords_source\n\nfig, ax = plt.subplots()\nax.scatter(xI, yI, s=20, alpha=0.2, label='source')\nax.legend(markerscale=1)\nax.set_aspect('equal')"),
        ("code", "# Target\nfname = '../visium_data/slice2_coor.csv'\ndf2 = pd.read_csv(fname)\n\nxJ = np.array(df2[df2.columns[0]])\nyJ = np.array(df2[df2.columns[1]])\n\ncoords_target = np.column_stack([xJ, yJ])\nadata_target = ad.AnnData(X=np.zeros((len(coords_target), 1)))\nadata_target.obsm['spatial'] = coords_target\n\nfig, ax = plt.subplots()\nax.scatter(xJ, yJ, s=20, alpha=0.2, c='#ff7f0e', label='target')\nax.legend(markerscale=1)\nax.set_aspect('equal')"),
        ("code", "# Overlay\nfig, ax = plt.subplots()\nax.scatter(xI, yI, s=20, alpha=0.2, label='source')\nax.scatter(xJ, yJ, s=20, alpha=0.1, label='target')\nax.legend(markerscale=1)\nax.set_aspect('equal')"),
        ("markdown", "## Align using moscot"),
        ("code", "sq.experimental.tl.align(\n    adata_source,\n    adata_target,\n    method='optimal_transport',\n    verbose=True,\n)"),
        ("markdown", "## Visualize"),
        ("code", "aligned = adata_source.obsm['spatial_aligned']\n\nfig, ax = plt.subplots()\nax.scatter(xI, yI, s=20, alpha=0.1, label='source')\nax.scatter(aligned[:, 0], aligned[:, 1], s=20, alpha=0.1, label='source aligned')\nax.scatter(xJ, yJ, s=20, alpha=0.1, label='target')\nlgnd = plt.legend(scatterpoints=1, fontsize=10)\nfor handle in lgnd.legend_handles:\n    handle.set_sizes([20.0])\nax.set_aspect('equal')"),
        ("code", "fig, ax = plt.subplots(1, 2, figsize=(20, 8))\nax[0].scatter(xI, yI, s=20, alpha=0.1, label='source')\nax[0].scatter(xJ, yJ, s=20, alpha=0.1, label='target')\nax[0].set_title('Before alignment')\nax[0].set_aspect('equal')\nax[0].legend(markerscale=1)\n\nax[1].scatter(aligned[:, 0], aligned[:, 1], s=20, alpha=0.1, label='source aligned')\nax[1].scatter(xJ, yJ, s=20, alpha=0.1, label='target')\nax[1].set_title('After alignment')\nax[1].set_aspect('equal')\nax[1].legend(markerscale=1)"),
        ("markdown", "## Save results"),
        ("code", "df_aligned = pd.DataFrame({'aligned_x': aligned[:, 0], 'aligned_y': aligned[:, 1]})\nresults = pd.concat([df1, df_aligned], axis=1)\nresults.head()"),
    ]


# ============================================================================
# 10. xenium-starmap-alignment (point-to-point → moscot)
# ============================================================================
def xenium_starmap_alignment():
    return [
        ("markdown", "# Aligning partially matched coronal sections from Xenium and STARmap PLUS (squidpy + moscot)\n\nIn this notebook, we align two single cell resolution spatial transcriptomics datasets of full and hemi coronal sections of the adult mouse brain assayed by Xenium and STARmap PLUS.\n\nThe original notebook used STalign LDDMM. Here we use `squidpy` with `moscot` optimal transport."),
        ("code", "import numpy as np\nimport matplotlib.pyplot as plt\nimport pandas as pd\nimport anndata as ad\nimport squidpy as sq\n\nplt.rcParams[\"figure.figsize\"] = (12, 10)"),
        ("markdown", "## Load data"),
        ("code", "# Source: Xenium\nfname = '../xenium_data/Xenium_V1_FF_Mouse_Brain_MultiSection_1_cells.csv.gz'\ndf1 = pd.read_csv(fname)\n\ncoords_source = np.column_stack([df1['x_centroid'].values, df1['y_centroid'].values])\nadata_source = ad.AnnData(X=np.zeros((len(coords_source), 1)), obs=df1)\nadata_source.obsm['spatial'] = coords_source\n\nfig, ax = plt.subplots()\nax.scatter(coords_source[:, 0], coords_source[:, 1], s=1, alpha=0.2)\nax.set_title('Xenium (source)')"),
        ("code", "# Target: STARmap PLUS\nfname = '../starmap_data/well11_spatial.csv.gz'\ndf2 = pd.read_csv(fname, skiprows=[1])\n\n# Convert to similar scale and flip\nxJ = np.array(df2['Y']) / 5\nyJ = np.array(df2['X']) / 5\nyJ = yJ.max() - yJ\n\ncoords_target = np.column_stack([xJ, yJ])\nadata_target = ad.AnnData(X=np.zeros((len(coords_target), 1)))\nadata_target.obsm['spatial'] = coords_target\n\nfig, ax = plt.subplots()\nax.scatter(coords_target[:, 0], coords_target[:, 1], s=1, alpha=0.2, c='#ff7f0e')\nax.set_title('STARmap PLUS (target)')"),
        ("code", "# Overlay\nfig, ax = plt.subplots()\nax.scatter(coords_source[:, 0], coords_source[:, 1], s=1, alpha=0.1, label='Xenium')\nax.scatter(coords_target[:, 0], coords_target[:, 1], s=1, alpha=0.2, label='STARmap')\nax.legend(markerscale=10)"),
        ("markdown", "## Align using moscot"),
        ("code", "sq.experimental.tl.align(\n    adata_source,\n    adata_target,\n    method='optimal_transport',\n    verbose=True,\n)"),
        ("markdown", "## Visualize"),
        ("code", "aligned = adata_source.obsm['spatial_aligned']\n\nfig, ax = plt.subplots()\nax.scatter(coords_source[:, 0], coords_source[:, 1], s=1, alpha=0.2, label='Xenium')\nax.scatter(aligned[:, 0], aligned[:, 1], s=1, alpha=0.1, label='Xenium aligned')\nax.scatter(coords_target[:, 0], coords_target[:, 1], s=1, alpha=0.2, label='STARmap')\nax.legend(markerscale=10)"),
    ]


# ============================================================================
# 11. xenium-xenium-alignment (point-to-point → moscot)
# ============================================================================
def xenium_xenium_alignment():
    return [
        ("markdown", "# Aligning serial breast cancer Xenium sections (squidpy + moscot)\n\nIn this notebook, we align two single cell resolution spatial transcriptomics datasets of serial breast cancer sections profiled by the Xenium technology.\n\nThe original notebook used STalign LDDMM with landmark points. Here we use `squidpy` with `moscot` optimal transport."),
        ("code", "import numpy as np\nimport matplotlib.pyplot as plt\nimport pandas as pd\nimport anndata as ad\nimport squidpy as sq\n\nplt.rcParams[\"figure.figsize\"] = (10, 8)"),
        ("markdown", "## Load data"),
        ("code", "# Source: Rep1\nfname = '../xenium_data/Xenium_FFPE_Human_Breast_Cancer_Rep1_cells.csv.gz'\ndf1 = pd.read_csv(fname)\n\ncoords_source = np.column_stack([df1['x_centroid'].values, df1['y_centroid'].values])\nadata_source = ad.AnnData(X=np.zeros((len(coords_source), 1)), obs=df1)\nadata_source.obsm['spatial'] = coords_source\n\nfig, ax = plt.subplots()\nax.scatter(coords_source[:, 0], coords_source[:, 1], s=1, alpha=0.2)\nax.set_title('Xenium Rep1 (source)')"),
        ("code", "# Target: Rep2\nfname = '../xenium_data/Xenium_FFPE_Human_Breast_Cancer_Rep2_cells.csv.gz'\ndf2 = pd.read_csv(fname)\n\ncoords_target = np.column_stack([df2['x_centroid'].values, df2['y_centroid'].values])\nadata_target = ad.AnnData(X=np.zeros((len(coords_target), 1)), obs=df2)\nadata_target.obsm['spatial'] = coords_target\n\nfig, ax = plt.subplots()\nax.scatter(coords_target[:, 0], coords_target[:, 1], s=1, alpha=0.2, c='#ff7f0e')\nax.set_title('Xenium Rep2 (target)')"),
        ("code", "# Overlay\nfig, ax = plt.subplots()\nax.scatter(coords_source[:, 0], coords_source[:, 1], s=1, alpha=0.2, label='Rep1')\nax.scatter(coords_target[:, 0], coords_target[:, 1], s=1, alpha=0.2, label='Rep2')\nax.legend(markerscale=10)"),
        ("markdown", "## Align using moscot"),
        ("code", "sq.experimental.tl.align(\n    adata_source,\n    adata_target,\n    method='optimal_transport',\n    verbose=True,\n)"),
        ("markdown", "## Visualize"),
        ("code", "aligned = adata_source.obsm['spatial_aligned']\n\nfig, ax = plt.subplots()\nax.scatter(coords_source[:, 0], coords_source[:, 1], s=1, alpha=0.2, label='Rep1')\nax.scatter(aligned[:, 0], aligned[:, 1], s=1, alpha=0.1, label='Rep1 aligned')\nax.scatter(coords_target[:, 0], coords_target[:, 1], s=1, alpha=0.2, label='Rep2')\nax.legend(markerscale=10)"),
        ("code", "# Also align target to source (reverse direction)\n# With moscot, you can also run align on the target to get it into source space\nadata_target_copy = adata_target.copy()\nsq.experimental.tl.align(\n    adata_target_copy,\n    adata_source,\n    method='optimal_transport',\n    verbose=True,\n)\n\naligned_tgt = adata_target_copy.obsm['spatial_aligned']\n\nfig, ax = plt.subplots()\nax.scatter(coords_source[:, 0], coords_source[:, 1], s=1, alpha=0.2, label='Rep1')\nax.scatter(aligned_tgt[:, 0], aligned_tgt[:, 1], s=1, alpha=0.1, label='Rep2 aligned to Rep1')\nax.legend(markerscale=10)"),
    ]


# ============================================================================
# 12. merfish-visium-alignment (point-to-image → stalign)
# ============================================================================
def merfish_visium_alignment():
    return [
        ("markdown", "# Aligning MERFISH to H&E staining image from Visium (squidpy + STalign)\n\nIn this notebook, we take a single cell resolution spatial transcriptomics dataset of a coronal section of the adult mouse brain profiled by MERFISH and align it to a H&E staining image.\n\nSince this is a point-to-image alignment, we use the STalign/LDDMM backend via `squidpy`."),
        ("code", "import numpy as np\nimport matplotlib.pyplot as plt\nimport pandas as pd\nimport anndata as ad\nimport squidpy as sq\n\nplt.rcParams[\"figure.figsize\"] = (12, 10)"),
        ("markdown", "## Load MERFISH cell data (source)"),
        ("code", "fname = '../merfish_data/datasets_mouse_brain_map_BrainReceptorShowcase_Slice2_Replicate3_cell_metadata_S2R3.csv.gz'\ndf1 = pd.read_csv(fname)\n\ncoords_source = np.column_stack([df1['center_x'].values, df1['center_y'].values])\nadata_source = ad.AnnData(X=np.zeros((len(coords_source), 1)), obs=df1)\nadata_source.obsm['spatial'] = coords_source\n\nfig, ax = plt.subplots()\nax.scatter(coords_source[:, 0], coords_source[:, 1], s=1, alpha=0.2)\nax.set_title('MERFISH cell positions (source)')"),
        ("markdown", "## Load H&E image (target)"),
        ("code", "image_file = '../visium_data/tissue_hires_image.png'\nV = plt.imread(image_file)\n\nfig, ax = plt.subplots()\nax.imshow(V)\nax.set_title('H&E staining image (target)')\nprint(f\"Image shape: {V.shape}\")"),
        ("markdown", "## Load landmark points\n\nFor point-to-image alignment, landmark points help initialize the alignment."),
        ("code", "# Load pre-computed landmark points\ndata = np.load('../visium_data/visium2_points.npz')\npointsI = np.array(data['pointsI'][..., ::-1])  # to x,y\npointsJ = np.array(data['pointsJ'][..., ::-1])  # to x,y\nprint(f\"Source landmarks: {pointsI}\")\nprint(f\"Target landmarks: {pointsJ}\")"),
        ("markdown", "## Align using STalign (point-to-image)\n\n`sq.experimental.tl.align` automatically selects the STalign backend when aligning points to an image."),
        ("code", "# Align MERFISH cell positions to H&E image\n# landmark_source/landmark_target are in (row, col) i.e. (y, x) order in the original STalign convention\n# but sq.experimental.tl.align expects (x, y) order\nsq.experimental.tl.align(\n    adata_source,\n    V,  # target image\n    method='lddmm',\n    resolution=30.0,\n    niter=200,\n    diffeo_start=100,\n    landmark_source=pointsI,  # (y, x) landmarks from original format\n    landmark_target=pointsJ,\n    verbose=True,\n    sigmaM=0.2,\n    sigmaB=0.19,\n    sigmaA=0.3,\n    sigmaP=2e-1,\n    epL=5e-11,\n    epT=5e-4,\n    epV=5e1,\n)"),
        ("markdown", "## Visualize results"),
        ("code", "aligned = adata_source.obsm['spatial_aligned']\nextentJ = [0, V.shape[1], 0, V.shape[0]]\n\nfig, ax = plt.subplots()\nax.imshow(V, extent=extentJ)\nax.scatter(aligned[:, 0], aligned[:, 1], s=1, alpha=0.1, label='MERFISH aligned')\nax.set_title('Aligned MERFISH on H&E')\nax.legend(markerscale=10)"),
        ("markdown", "## Save results"),
        ("code", "df_aligned = pd.DataFrame({'aligned_x': aligned[:, 0], 'aligned_y': aligned[:, 1]})\nresults = pd.concat([df1, df_aligned], axis=1)\nresults.head()"),
    ]


# ============================================================================
# 13. merfish-visium-alignment-with-curve-annotator (point-to-image → stalign)
# ============================================================================
def merfish_visium_alignment_with_curve_annotator():
    return [
        ("markdown", "# Aligning MERFISH to Visium H&E with curve annotations (squidpy + STalign)\n\nIn this notebook, we align MERFISH cell positions to a Visium H&E staining image using curve-based landmark annotations.\n\nSince this is a point-to-image alignment, we use the STalign/LDDMM backend via `squidpy`."),
        ("code", "import numpy as np\nimport matplotlib.pyplot as plt\nimport pandas as pd\nimport anndata as ad\nimport squidpy as sq\n\nplt.rcParams[\"figure.figsize\"] = (12, 10)"),
        ("markdown", "## Load MERFISH data (source)"),
        ("code", "fname = '../merfish_data/datasets_mouse_brain_map_BrainReceptorShowcase_Slice2_Replicate3_cell_metadata_S2R3.csv.gz'\ndf1 = pd.read_csv(fname)\n\ncoords_source = np.column_stack([df1['center_x'].values, df1['center_y'].values])\nadata_source = ad.AnnData(X=np.zeros((len(coords_source), 1)), obs=df1)\nadata_source.obsm['spatial'] = coords_source\n\nfig, ax = plt.subplots()\nax.scatter(coords_source[:, 0], coords_source[:, 1], s=1, alpha=0.2)\nax.set_title('MERFISH cell positions (source)')"),
        ("markdown", "## Load H&E image (target)"),
        ("code", "image_file = '../visium_data/tissue_hires_image.png'\nV = plt.imread(image_file)\n\nfig, ax = plt.subplots()\nax.imshow(V)\nax.set_title('H&E staining image (target)')"),
        ("markdown", "## Load curve landmark points\n\nCurve landmarks were previously annotated using `curve_annotator.py`."),
        ("code", "# Load curve-based landmarks\npointsIlist = np.load('../visium_data/Merfish_S2_R3_curves.npy', allow_pickle=True).tolist()\npointsJlist = np.load('../visium_data/tissue_hires_image_curves.npy', allow_pickle=True).tolist()\n\n# Convert to arrays (y,x order as in original)\npointsI = []\npointsJ = []\nfor i in pointsIlist.keys():\n    for j in range(len(pointsIlist[i])):\n        pointsI.append([pointsIlist[i][j][1], pointsIlist[i][j][0]])\nfor i in pointsJlist.keys():\n    for j in range(len(pointsJlist[i])):\n        pointsJ.append([pointsJlist[i][j][1], pointsJlist[i][j][0]])\n\npointsI = np.array(pointsI)\npointsJ = np.array(pointsJ)\nprint(f\"{len(pointsI)} source landmarks, {len(pointsJ)} target landmarks\")"),
        ("markdown", "## Align using STalign (point-to-image)"),
        ("code", "sq.experimental.tl.align(\n    adata_source,\n    V,\n    method='lddmm',\n    resolution=30.0,\n    niter=10000,\n    landmark_source=pointsI,\n    landmark_target=pointsJ,\n    verbose=True,\n)"),
        ("markdown", "## Visualize results"),
        ("code", "aligned = adata_source.obsm['spatial_aligned']\n\nfig, ax = plt.subplots()\nax.imshow(V)\nax.scatter(aligned[:, 0], aligned[:, 1], s=1, alpha=0.1, label='MERFISH aligned')\nax.set_title('Aligned MERFISH on H&E')\nax.legend(markerscale=10)"),
        ("code", "df_aligned = pd.DataFrame({'aligned_x': aligned[:, 0], 'aligned_y': aligned[:, 1]})\nresults = pd.concat([df1, df_aligned], axis=1)\nresults.head()"),
    ]


# ============================================================================
# 14. merfish-visium-alignment-with-point-annotator (point-to-image → stalign)
# ============================================================================
def merfish_visium_alignment_with_point_annotator():
    return [
        ("markdown", "# Aligning MERFISH to Visium H&E with point annotations (squidpy + STalign)\n\nIn this notebook, we align MERFISH cell positions to a Visium H&E staining image using point-based landmark annotations.\n\nSince this is a point-to-image alignment, we use the STalign/LDDMM backend via `squidpy`."),
        ("code", "import numpy as np\nimport matplotlib.pyplot as plt\nimport pandas as pd\nimport anndata as ad\nimport squidpy as sq\n\nplt.rcParams[\"figure.figsize\"] = (12, 10)"),
        ("markdown", "## Load MERFISH data (source)"),
        ("code", "fname = '../merfish_data/datasets_mouse_brain_map_BrainReceptorShowcase_Slice2_Replicate3_cell_metadata_S2R3.csv.gz'\ndf1 = pd.read_csv(fname)\n\ncoords_source = np.column_stack([df1['center_x'].values, df1['center_y'].values])\nadata_source = ad.AnnData(X=np.zeros((len(coords_source), 1)), obs=df1)\nadata_source.obsm['spatial'] = coords_source\n\nfig, ax = plt.subplots()\nax.scatter(coords_source[:, 0], coords_source[:, 1], s=1, alpha=0.2)\nax.set_title('MERFISH cell positions (source)')"),
        ("markdown", "## Load H&E image (target)"),
        ("code", "image_file = '../visium_data/tissue_hires_image.png'\nV = plt.imread(image_file)\n\nfig, ax = plt.subplots()\nax.imshow(V)\nax.set_title('H&E staining image (target)')"),
        ("markdown", "## Load landmark points"),
        ("code", "# Load point-based landmarks\npointsIlist = np.load('../visium_data/Merfish_S2_R3_points.npy', allow_pickle=True).tolist()\npointsJlist = np.load('../visium_data/tissue_hires_image_points.npy', allow_pickle=True).tolist()\n\npointsI = []\npointsJ = []\nfor i in pointsIlist.keys():\n    for j in range(len(pointsIlist[i])):\n        pointsI.append([pointsIlist[i][j][1], pointsIlist[i][j][0]])\nfor i in pointsJlist.keys():\n    for j in range(len(pointsJlist[i])):\n        pointsJ.append([pointsJlist[i][j][1], pointsJlist[i][j][0]])\n\npointsI = np.array(pointsI)\npointsJ = np.array(pointsJ)\nprint(f\"{len(pointsI)} source landmarks, {len(pointsJ)} target landmarks\")"),
        ("markdown", "## Align using STalign (point-to-image)"),
        ("code", "sq.experimental.tl.align(\n    adata_source,\n    V,\n    method='lddmm',\n    resolution=30.0,\n    niter=200,\n    diffeo_start=100,\n    landmark_source=pointsI,\n    landmark_target=pointsJ,\n    verbose=True,\n    sigmaP=2e-1,\n    sigmaM=0.18,\n    sigmaB=0.18,\n    sigmaA=0.18,\n    epL=5e-11,\n    epT=5e-4,\n    epV=5e1,\n)"),
        ("markdown", "## Visualize results"),
        ("code", "aligned = adata_source.obsm['spatial_aligned']\n\nfig, ax = plt.subplots()\nax.imshow(V)\nax.scatter(aligned[:, 0], aligned[:, 1], s=1, alpha=0.1, label='MERFISH aligned')\nax.scatter(pointsJ[:, 1], pointsJ[:, 0], c='red', label='target landmarks', s=100)\nax.set_title('Aligned MERFISH on H&E')\nax.legend(markerscale=5)"),
        ("code", "df_aligned = pd.DataFrame({'aligned_x': aligned[:, 0], 'aligned_y': aligned[:, 1]})\nresults = pd.concat([df1, df_aligned], axis=1)\nresults.head()"),
    ]


# ============================================================================
# 15. xenium-heimage-alignment (point-to-image → stalign)
# ============================================================================
def xenium_heimage_alignment():
    return [
        ("markdown", "# Aligning Xenium breast cancer data to H&E staining image (squidpy + STalign)\n\nIn this notebook, we align single cell resolution Xenium spatial transcriptomics data to a corresponding H&E staining image of the same tissue section.\n\nSince this is a point-to-image alignment, we use the STalign/LDDMM backend via `squidpy`."),
        ("code", "import numpy as np\nimport matplotlib.pyplot as plt\nimport pandas as pd\nimport anndata as ad\nimport squidpy as sq\n\nplt.rcParams[\"figure.figsize\"] = (12, 10)"),
        ("markdown", "## Load H&E image (target)"),
        ("code", "image_file = '../xenium_data/Xenium_FFPE_Human_Breast_Cancer_Rep1_he_image.png'\nV = plt.imread(image_file)\n\nfig, ax = plt.subplots()\nax.imshow(V)\nax.set_title('H&E image (target)')\nprint(f\"Image shape: {V.shape}\")"),
        ("markdown", "## Load Xenium cell data (source)"),
        ("code", "fname = '../xenium_data/Xenium_FFPE_Human_Breast_Cancer_Rep1_cells.csv.gz'\ndf = pd.read_csv(fname)\n\ncoords_source = np.column_stack([df['x_centroid'].values, df['y_centroid'].values])\nadata_source = ad.AnnData(X=np.zeros((len(coords_source), 1)), obs=df)\nadata_source.obsm['spatial'] = coords_source\n\nfig, ax = plt.subplots()\nax.scatter(coords_source[:, 0], coords_source[:, 1], s=1, alpha=0.2)\nax.set_title('Xenium cell positions (source)')"),
        ("markdown", "## Define landmark points"),
        ("code", "# Manually defined landmarks (in y,x / row,col order)\npointsI = np.array([[1050., 950.], [700., 2200.], [500., 1550.], [1550., 1840.]])\npointsJ = np.array([[3108., 2100.], [4480., 6440.], [5040., 4200.], [1260., 5320.]])"),
        ("markdown", "## Align using STalign (point-to-image)\n\nNote: For this alignment, the H&E image is the source and Xenium positions are being aligned. The squidpy API handles this by aligning the point cloud to the image space."),
        ("code", "# Here we align the H&E (source image) to Xenium cell density (target)\n# Then transform the Xenium points using the inverse\nsq.experimental.tl.align(\n    adata_source,\n    V,\n    method='lddmm',\n    resolution=30.0,\n    niter=2000,\n    landmark_source=pointsI,\n    landmark_target=pointsJ,\n    verbose=True,\n    sigmaM=0.15,\n    sigmaB=0.10,\n    sigmaA=0.11,\n    epV=10,\n)"),
        ("markdown", "## Visualize results"),
        ("code", "aligned = adata_source.obsm['spatial_aligned']\n\nfig, ax = plt.subplots()\nax.imshow(V)\nax.scatter(aligned[:, 0], aligned[:, 1], s=1, alpha=0.1, label='Xenium aligned')\nax.set_title('Aligned Xenium on H&E')\nax.legend(markerscale=10)"),
        ("code", "df_aligned = pd.DataFrame({'aligned_x': aligned[:, 0], 'aligned_y': aligned[:, 1]})\nresults = pd.concat([df, df_aligned], axis=1)\nresults.head()"),
    ]


# ============================================================================
# 16. merfish-allen3Datlas-alignment (3D atlas → not supported in squidpy API)
# ============================================================================
def merfish_allen3Datlas_alignment():
    return [
        ("markdown", "# Aligning MERFISH to the Allen Brain Atlas (squidpy + STalign)\n\nIn this notebook, we align a MERFISH dataset of an adult mouse coronal brain section to the Allen Brain Atlas.\n\n**Note:** 3D atlas-to-2D-slice alignment (`LDDMM_3D_to_slice`) is not yet available in the squidpy high-level API. This notebook uses the STalign functions directly from squidpy's internal LDDMM module. For standard 2D point-to-point alignments, see the moscot-based notebooks."),
        ("code", "import numpy as np\nimport matplotlib.pyplot as plt\nimport pandas as pd\nimport nrrd\nimport torch\nimport squidpy as sq\n\n# Use STalign internals for 3D alignment (not yet in sq.experimental.tl.align)\nfrom squidpy.experimental._lddmm import rasterize\nfrom squidpy.experimental._lddmm._transforms import L_T_from_points\n\n# Also need the original STalign for LDDMM_3D_to_slice\n# which is not yet ported to squidpy\nfrom STalign import STalign"),
        ("markdown", "## Load MERFISH data"),
        ("code", "df = pd.read_csv('../merfish_data/s1r1_metadata.csv.gz')\nx = np.array(df['center_x'])\ny = np.array(df['center_y'])\n\ndx = 10\nblur = 1"),
        ("markdown", "## Load Allen Brain Atlas\n\nThis section remains identical to the STalign version as the 3D atlas alignment is not yet in the squidpy API."),
        ("code", "url = 'http://api.brain-map.org/api/v2/data/query.csv?criteria=model::Structure,rma::criteria,[ontology_id$eq1],rma::options[order$eq%27structures.graph_order%27][num_rows$eqall]'\nontology_name, namesdict = STalign.download_aba_ontology(url, 'allen_ontology.csv')"),
        ("code", "imageurl = 'http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/ara_nissl/ara_nissl_50.nrrd'\nlabelurl = 'http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/ccf_2017/annotation_50.nrrd'\nimagefile, labelfile = STalign.download_aba_image_labels(imageurl, labelurl, 'aba_nissl.nrrd', 'aba_annotation.nrrd')"),
        ("code", "# Rasterize using squidpy's rasterize function\nX_, Y_, W = rasterize(x, y, dx=dx, blur=blur)"),
        ("markdown", "## Initialize alignment parameters\n\nThe 3D atlas alignment procedure uses `LDDMM_3D_to_slice` which is specific to STalign."),
        ("code", "slice_idx = 177\n\nvol, hdr = nrrd.read(imagefile)\nA = vol\nvol, hdr = nrrd.read(labelfile)\nL = vol\n\ndxA = np.diag(hdr['space directions'])\nnxA = A.shape\nxA = [np.arange(n)*d - (n-1)*d/2.0 for n, d in zip(nxA, dxA)]"),
        ("code", "# Alignment parameters\npoints_atlas = np.array([[0, 2580]])\npoints_target = np.array([[8, 2533]])\nLi, Ti = STalign.L_T_from_points(points_atlas, points_target)\n\nxJ = [Y_, X_]\nJ = W[None] / np.mean(np.abs(W))\nxI = xA\nI = A[None] / np.mean(np.abs(A), keepdims=True)\nI = np.concatenate((I, (I - np.mean(I))**2))"),
        ("code", "sigmaA = 2\nsigmaB = 2\nsigmaM = 2\nmuA = torch.tensor([3, 3, 3], device='cpu')\nmuB = torch.tensor([0, 0, 0], device='cpu')\n\nscale_x = 0.9\nscale_y = 0.9\nscale_z = 0.9\ntheta0 = 0\n\nT = np.array([-xI[0][slice_idx], np.mean(xJ[0])-(Ti[0]*scale_y), np.mean(xJ[1])-(Ti[1]*scale_x)])\nscale_atlas = np.array([[scale_z, 0, 0], [0, scale_x, 0], [0, 0, scale_y]])\nL_mat = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(theta0), -np.sin(theta0)], [0.0, np.sin(theta0), np.cos(theta0)]])\nL_mat = np.matmul(L_mat, scale_atlas)"),
        ("markdown", "## Run 3D-to-slice alignment\n\nThis uses the STalign `LDDMM_3D_to_slice` function directly."),
        ("code", "%%time\ntransform = STalign.LDDMM_3D_to_slice(\n    xI, I, xJ, J,\n    T=T, L=L_mat,\n    nt=4, niter=2000,\n    device='cpu',\n    sigmaA=sigmaA, sigmaB=sigmaB, sigmaM=sigmaM,\n    muA=muA, muB=muB,\n)"),
        ("code", "A_out = transform['A']\nv = transform['v']\nxv = transform['xv']\nXs = transform['Xs']"),
        ("markdown", "## Analyze results"),
        ("code", "df_result = STalign.analyze3Dalign(\n    labelfile, xv, v, A_out, xJ, dx,\n    scale_x=scale_x, scale_y=scale_y,\n    x=x, y=y, X_=X_, Y_=Y_,\n    namesdict=namesdict, device='cpu',\n)\ndf_result"),
        ("code", "STalign.plot_brain_regions(df_result)"),
    ]


# ============================================================================
# 17. starmap-allen3Datlas-alignment (3D atlas → not supported in squidpy API)
# ============================================================================
def starmap_allen3Datlas_alignment():
    return [
        ("markdown", "# Aligning STARmap to the Allen Brain Atlas (squidpy + STalign)\n\nIn this notebook, we align a STARmap dataset of the right hemisphere of a mouse brain to the Allen Brain Atlas.\n\n**Note:** 3D atlas-to-2D-slice alignment (`LDDMM_3D_to_slice`) is not yet available in the squidpy high-level API. This notebook uses STalign functions directly."),
        ("code", "import numpy as np\nimport matplotlib.pyplot as plt\nimport pandas as pd\nimport nrrd\nimport torch\nimport squidpy as sq\n\nfrom squidpy.experimental._lddmm import rasterize\nfrom STalign import STalign"),
        ("markdown", "## Load STARmap data"),
        ("code", "df = pd.read_csv('../starmap_data/well11_spatial.csv.gz')\nx = np.array(df['X'])[1:].astype(float)\ny = np.array(df['Y'])[1:].astype(float)\n\ndx = 50\nblur = 1"),
        ("markdown", "## Load Allen Brain Atlas"),
        ("code", "url = 'http://api.brain-map.org/api/v2/data/query.csv?criteria=model::Structure,rma::criteria,[ontology_id$eq1],rma::options[order$eq%27structures.graph_order%27][num_rows$eqall]'\nontology_name, namesdict = STalign.download_aba_ontology(url, 'allen_ontology.csv')"),
        ("code", "imageurl = 'http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/ara_nissl/ara_nissl_50.nrrd'\nlabelurl = 'http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/ccf_2017/annotation_50.nrrd'\nimagefile, labelfile = STalign.download_aba_image_labels(imageurl, labelurl, 'aba_nissl.nrrd', 'aba_annotation.nrrd')"),
        ("code", "# Rasterize\nX_, Y_, W = rasterize(x, y, dx=dx, blur=blur)"),
        ("markdown", "## Initialize alignment"),
        ("code", "slice_idx = 140\n\nvol, hdr = nrrd.read(imagefile)\nA = vol\nvol, hdr = nrrd.read(labelfile)\nL = vol\n\ndxA = np.diag(hdr['space directions'])\nnxA = A.shape\nxA = [np.arange(n)*d - (n-1)*d/2.0 for n, d in zip(nxA, dxA)]"),
        ("code", "theta_deg = 90\n\npoints_atlas = np.array([[0, 0]])\npoints_target = np.array([[-3700, 0]])\nLi, Ti = STalign.L_T_from_points(points_atlas, points_target)\n\nxJ = [Y_, X_]\nJ = W[None] / np.mean(np.abs(W))\nxI = xA\nI = A[None] / np.mean(np.abs(A), keepdims=True)\nI = np.concatenate((I, (I - np.mean(I))**2))\nInorm = STalign.normalize(I, t_min=0, t_max=1)\nJnorm = STalign.normalize(J, t_min=0, t_max=1)"),
        ("code", "sigmaA = 0.1\nsigmaB = 0.1\nsigmaM = 0.1\nmuA = torch.tensor([0.7, 0.7, 0.7], device='cpu')\nmuB = torch.tensor([0, 0, 0], device='cpu')\n\nscale_x = 4\nscale_y = 4\nscale_z = 0.9\ntheta0 = (np.pi/180) * theta_deg\n\nT = np.array([-xI[0][slice_idx], np.mean(xJ[0])-(Ti[0]*scale_y), np.mean(xJ[1])-(Ti[1]*scale_x)])\nscale_atlas = np.array([[scale_z, 0, 0], [0, scale_x, 0], [0, 0, scale_y]])\nL_mat = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(theta0), -np.sin(theta0)], [0.0, np.sin(theta0), np.cos(theta0)]])\nL_mat = np.matmul(L_mat, scale_atlas)"),
        ("markdown", "## Run 3D-to-slice alignment"),
        ("code", "%%time\ntransform = STalign.LDDMM_3D_to_slice(\n    xI, Inorm, xJ, Jnorm,\n    T=T, L=L_mat,\n    nt=4, niter=800,\n    a=250,\n    device='cpu',\n    sigmaA=sigmaA, sigmaB=sigmaB, sigmaM=sigmaM,\n    muA=muA, muB=muB,\n)"),
        ("code", "A_out = transform['A']\nv = transform['v']\nxv = transform['xv']\nXs = transform['Xs']"),
        ("markdown", "## Analyze results"),
        ("code", "df_result = STalign.analyze3Dalign(\n    labelfile, xv, v, A_out, xJ, dx,\n    scale_x=scale_x, scale_y=scale_y,\n    x=x, y=y, X_=X_, Y_=Y_,\n    namesdict=namesdict, device='cpu',\n)\ndf_result"),
        ("code", "STalign.plot_brain_regions(df_result)"),
        ("code", "brain_regions = ['CA1']\nSTalign.plot_subset_brain_regions(df_result, brain_regions)"),
    ]


# ============================================================================
# Main: generate all notebooks
# ============================================================================
NOTEBOOKS = {
    # Point-to-point (moscot)
    "heart-alignment-varying-thickness-squidpy.ipynb": heart_alignment_varying_thickness,
    "heart-alignment-squidpy.ipynb": heart_alignment,
    "merfish-merfish-alignment-affine-only-with-points-squidpy.ipynb": merfish_merfish_alignment_affine_only_with_points,
    "merfish-merfish-alignment-affine-only-squidpy.ipynb": merfish_merfish_alignment_affine_only,
    "merfish-merfish-alignment-simulation-squidpy.ipynb": merfish_merfish_alignment_simulation,
    "merfish-merfish-alignment-using-L-T-squidpy.ipynb": merfish_merfish_alignment_using_LT,
    "merfish-merfish-alignment-squidpy.ipynb": merfish_merfish_alignment,
    "merfish-xenium-alignment-squidpy.ipynb": merfish_xenium_alignment,
    "visium-visium-alignment-affine-only-squidpy.ipynb": visium_visium_alignment_affine_only,
    "xenium-starmap-alignment-squidpy.ipynb": xenium_starmap_alignment,
    "xenium-xenium-alignment-squidpy.ipynb": xenium_xenium_alignment,
    # Point-to-image (stalign via squidpy)
    "merfish-visium-alignment-squidpy.ipynb": merfish_visium_alignment,
    "merfish-visium-alignment-with-curve-annotator-squidpy.ipynb": merfish_visium_alignment_with_curve_annotator,
    "merfish-visium-alignment-with-point-annotator-squidpy.ipynb": merfish_visium_alignment_with_point_annotator,
    "xenium-heimage-alignment-squidpy.ipynb": xenium_heimage_alignment,
    # 3D atlas (STalign-native, no squidpy high-level API yet)
    "merfish-allen3Datlas-alignment-squidpy.ipynb": merfish_allen3Datlas_alignment,
    "starmap-allen3Datlas-alignment-squidpy.ipynb": starmap_allen3Datlas_alignment,
}

if __name__ == "__main__":
    print(f"Generating {len(NOTEBOOKS)} squidpy-compatible notebooks...")
    for name, fn in NOTEBOOKS.items():
        cells = fn()
        write_nb(name, cells)
    print("Done!")
