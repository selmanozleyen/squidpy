"""
Benchmark script for ligrec() -- compare main vs refactored branch.

Usage:
    python scripts/bench_ligrec.py                          # default config
    python scripts/bench_ligrec.py --n-perms 500            # fewer perms (faster)
    python scripts/bench_ligrec.py --n-cells 50000          # more cells (slower)
    python scripts/bench_ligrec.py --n-runs 5               # average over 5 runs
    python scripts/bench_ligrec.py --n-jobs 4               # 4 workers (main only)
    python scripts/bench_ligrec.py --no-cache               # rebuild data from scratch

Defaults are calibrated to ~30s per run on Apple M-series (1 core):
    30 000 cells, 2 000 genes, 25 clusters, 6 400 interactions, 1 000 perms.

The prepared AnnData + interactions are cached under .pytest_cache/
so repeated runs skip the (slow) data-generation step.
"""

from __future__ import annotations

import argparse
import hashlib
import pickle
import time
import warnings
from itertools import product
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*ImplicitModificationWarning.*")
warnings.filterwarnings("ignore", message=".*Transforming to str index.*")

import numpy as np
import pandas as pd
from anndata import AnnData

from squidpy.gr import ligrec

CACHE_DIR = Path(".pytest_cache") / "bench_ligrec"


def _cache_key(
    n_cells: int,
    n_genes: int,
    n_clusters: int,
    n_interaction_genes: int,
) -> str:
    tag = f"{n_cells}_{n_genes}_{n_clusters}_{n_interaction_genes}"
    return hashlib.sha256(tag.encode()).hexdigest()[:16]


def _build_adata(
    n_cells: int,
    n_genes: int,
    n_clusters: int,
    n_interaction_genes: int,
    use_cache: bool,
) -> tuple[AnnData, list[tuple[str, str]]]:
    key = _cache_key(n_cells, n_genes, n_clusters, n_interaction_genes)
    cache_path = CACHE_DIR / f"{key}.pkl"

    if use_cache and cache_path.exists():
        print(f"Loading cached data from {cache_path}", flush=True)
        with open(cache_path, "rb") as f:
            adata, interactions = pickle.load(f)
        print(
            f"  cells={adata.n_obs}, genes={adata.n_vars}, "
            f"clusters={len(adata.obs['cluster'].cat.categories)}, "
            f"interactions={len(interactions)}",
            flush=True,
        )
        return adata, interactions

    print("Building synthetic AnnData...", flush=True)
    rng = np.random.default_rng(42)
    X = rng.random((n_cells, n_genes))
    cluster_labels = rng.choice([f"c{i}" for i in range(n_clusters)], size=n_cells)
    obs = pd.DataFrame({"cluster": pd.Categorical(cluster_labels)})
    var = pd.DataFrame(index=[f"G{i}" for i in range(n_genes)])
    adata = AnnData(X, obs=obs, var=var)
    adata.raw = adata.copy()

    igenes = list(adata.var_names[:n_interaction_genes])
    interactions = list(product(igenes, igenes))

    print(
        f"  cells={n_cells}, genes={n_genes}, clusters={n_clusters}, "
        f"interaction_genes={n_interaction_genes}, "
        f"interactions={len(interactions)}",
        flush=True,
    )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump((adata, interactions), f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  cached to {cache_path}", flush=True)

    return adata, interactions


def _run_once(
    adata: AnnData,
    interactions: list[tuple[str, str]],
    n_perms: int,
    n_jobs: int | None = None,
) -> float:
    kwargs: dict[str, Any] = {}
    if n_jobs is not None:
        kwargs["n_jobs"] = n_jobs
    t0 = time.perf_counter()
    ligrec(
        adata,
        cluster_key="cluster",
        interactions=interactions,
        n_perms=n_perms,
        copy=True,
        seed=0,
        use_raw=True,
        **kwargs,
    )
    return time.perf_counter() - t0


def main():
    parser = argparse.ArgumentParser(description="Benchmark ligrec()")
    parser.add_argument(
        "--n-cells",
        type=int,
        default=30000,
        help="Number of cells (default 30000)",
    )
    parser.add_argument(
        "--n-genes",
        type=int,
        default=2000,
        help="Total genes in AnnData (default 2000)",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=25,
        help="Number of clusters (default 25)",
    )
    parser.add_argument(
        "--n-interaction-genes",
        type=int,
        default=80,
        help="Genes used in interactions; n^2 pairs (default 80 -> 6400)",
    )
    parser.add_argument(
        "--n-perms",
        type=int,
        default=1000,
        help="Number of permutations (default 1000)",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=3,
        help="Number of timed runs (default 3)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Number of parallel workers (default: all cores)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Rebuild data even if cache exists",
    )
    args = parser.parse_args()

    adata, interactions = _build_adata(
        n_cells=args.n_cells,
        n_genes=args.n_genes,
        n_clusters=args.n_clusters,
        n_interaction_genes=args.n_interaction_genes,
        use_cache=not args.no_cache,
    )

    n_jobs = args.n_jobs

    print("\nWarmup (JIT compile)...", flush=True)
    small = adata[:50, :].copy()
    small.raw = small.copy()
    ligrec(small, cluster_key="cluster", interactions=interactions[:4], n_perms=5, copy=True, seed=0, use_raw=True)
    print("  done.\n", flush=True)

    n_inter = len(interactions)
    n_cls_pairs = len(adata.obs["cluster"].cat.categories) ** 2
    jobs_label = n_jobs if n_jobs is not None else "all"
    print(
        f"Config: {args.n_cells} cells, {args.n_genes} genes, "
        f"{args.n_clusters} clusters, {n_inter} interactions, "
        f"{n_cls_pairs} cluster pairs, {args.n_perms} perms, "
        f"n_jobs={jobs_label}",
        flush=True,
    )
    print(f"Running ligrec() {args.n_runs} time(s)...\n", flush=True)

    times = []
    for i in range(args.n_runs):
        t = _run_once(adata, interactions, args.n_perms, n_jobs=n_jobs)
        times.append(t)
        print(f"  run {i + 1}: {t:.3f}s", flush=True)

    times_arr = np.array(times)
    print(f"\nResults ({args.n_runs} runs):")
    print(f"  mean:   {times_arr.mean():.3f}s")
    print(f"  median: {np.median(times_arr):.3f}s")
    print(f"  min:    {times_arr.min():.3f}s")
    print(f"  max:    {times_arr.max():.3f}s")


if __name__ == "__main__":
    main()
