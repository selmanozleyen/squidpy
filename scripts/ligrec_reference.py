"""Generate reference ligrec results with seed=42, n_jobs=1 (single rng)."""

from itertools import product

import numpy as np
import scanpy as sc

import squidpy as sq

adata = sc.read("tests/_data/test_data.h5ad")
adata.raw = adata.copy()

interactions = tuple(product(adata.raw.var_names[:5], adata.raw.var_names[:5]))

result = sq.gr.ligrec(
    adata,
    "leiden",
    interactions=interactions,
    n_perms=100,
    seed=42,
    n_jobs=1,
    copy=True,
    show_progress_bar=False,
)

means = result["means"].sparse.to_dense()
pvalues = result["pvalues"].sparse.to_dense()

np.savez(
    "scripts/ligrec_reference_seed42.npz",
    means=means.values,
    pvalues=pvalues.values,
    means_index=means.index.to_flat_index().values,
    means_columns=means.columns.to_flat_index().values,
    pvalues_index=pvalues.index.to_flat_index().values,
    pvalues_columns=pvalues.columns.to_flat_index().values,
)

print(f"means shape: {means.shape}")
print(f"pvalues shape: {pvalues.shape}")
print(f"means non-zero: {(means != 0).sum().sum()}")
print(f"pvalues non-NaN: {pvalues.notna().sum().sum()}")
print(f"pvalues sample:\n{pvalues.iloc[:3, :3]}")
print(f"\nSaved to scripts/ligrec_reference_seed42.npz")
