"""Benchmark squidpy.gr.centrality_scores across branches.

Run on each branch:
    source .venv/bin/activate && python benchmarks/bench_closeness.py
"""
from __future__ import annotations

import os
import subprocess
import time

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

import squidpy as sq

SCORES = ["closeness_centrality", "degree_centrality", "average_clustering"]


def make_adata(n_cells: int, n_clusters: int, n_neighs: int = 6) -> AnnData:
    rng = np.random.default_rng(42)
    coords = rng.uniform(0, 1000, size=(n_cells, 2))
    X = rng.standard_normal((n_cells, 10)).astype(np.float32)

    tree = NearestNeighbors(n_neighbors=n_neighs, metric="euclidean")
    tree.fit(coords)
    _, col = tree.kneighbors()
    col = col.reshape(-1)
    row = np.repeat(np.arange(n_cells), n_neighs)
    adj = csr_matrix(
        (np.ones(len(row), dtype=np.float32), (row, col)),
        shape=(n_cells, n_cells),
    )
    adj = adj + adj.T
    adj.data[:] = 1.0
    adj.setdiag(0)
    adj.eliminate_zeros()

    labels = pd.Categorical(rng.integers(0, n_clusters, size=n_cells).astype(str))
    adata = AnnData(X)
    adata.obsm["spatial"] = coords
    adata.obsp["spatial_connectivities"] = adj
    adata.obs["cluster"] = labels

    return adata


def _call(adata: AnnData, score: str | None, n_jobs: int | None) -> None:
    kwargs: dict = dict(
        cluster_key="cluster",
        connectivity_key="spatial",
    )
    if score is not None:
        kwargs["score"] = score
    try:
        sq.gr.centrality_scores(adata, n_jobs=n_jobs, **kwargs)
    except TypeError:
        sq.gr.centrality_scores(adata, **kwargs)


def bench(n_cells: int, n_clusters: int, score: str | None,
          n_jobs: int | None, n_repeats: int = 3) -> float:
    adata = make_adata(n_cells, n_clusters)
    _call(adata.copy(), score, n_jobs)

    times = []
    for _ in range(n_repeats):
        a = adata.copy()
        t0 = time.perf_counter()
        _call(a, score, n_jobs)
        times.append(time.perf_counter() - t0)

    return round(float(np.median(times)), 4)


CONFIGS = [
    (1_000, 10),
    (5_000, 15),
    (10_000, 20),
    (20_000, 20),
    (50_000, 25),
]


if __name__ == "__main__":


    n_threads = min(os.cpu_count() or 4, 8)

    print(f"Threads: {n_threads}")

    score_cases = SCORES + [None]
    score_labels = SCORES + ["all"]

    for label, score_arg in zip(score_labels, score_cases):
        print(f"\n{'=' * 60}")
        print(f"  {label}")
        print(f"{'=' * 60}")

        rows = []
        for n_cells, n_clusters in CONFIGS:
            print(f"  n={n_cells:>6} cls={n_clusters:>3} ...", end="", flush=True)
            t_seq = bench(n_cells, n_clusters, score_arg, n_jobs=None)
            t_thr = bench(n_cells, n_clusters, score_arg, n_jobs=n_threads)
            rows.append({
                "n_cells": n_cells,
                "n_clusters": n_clusters,
                "1 thread (s)": t_seq,
                f"{n_threads} threads (s)": t_thr,
            })
            print(f"  1thr={t_seq:.4f}s  {n_threads}thr={t_thr:.4f}s")

        print()
        print(pd.DataFrame(rows).to_string(index=False))
