"""Standalone benchmark runner for comparing main vs current branch.

Usage:
    cd /path/to/squidpy
    python benchmarks/run_benchmarks.py

This runs each benchmark function, timing it on the *current* installed
version of squidpy. To compare branches:

    1. git checkout main && pip install -e .
    2. python benchmarks/run_benchmarks.py 2>&1 | tee results_main.txt
    3. git checkout <refactored> && pip install -e .
    4. python benchmarks/run_benchmarks.py 2>&1 | tee results_new.txt
    5. diff results_main.txt results_new.txt
"""

from __future__ import annotations

import gc
import inspect
import sys
import time
import traceback
from typing import Any

import numpy as np

import squidpy as sq


def _has_param(fn: Any, name: str) -> bool:
    return name in inspect.signature(fn).parameters


def timed(
    fn: Any, *args: Any, warmup: int = 0, repeats: int = 3, **kwargs: Any
) -> dict[str, float]:
    """Time a function call, return min/mean/std over repeats."""
    for _ in range(warmup):
        fn(*args, **kwargs)
    times = []
    for _ in range(repeats):
        gc.collect()
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    arr = np.array(times)
    return {"min": float(arr.min()), "mean": float(arr.mean()), "std": float(arr.std())}


def _njobs_kw(fn: Any, n_jobs: int) -> dict[str, Any]:
    """Build keyword args that are accepted by the function on this branch."""
    kw: dict[str, Any] = {}
    if _has_param(fn, "n_jobs"):
        kw["n_jobs"] = n_jobs
    if _has_param(fn, "backend") and n_jobs > 1:
        kw["backend"] = "loky"
    if _has_param(fn, "show_progress_bar"):
        kw["show_progress_bar"] = False
    return kw


def report(name: str, result: dict[str, float]) -> None:
    print(
        f"  {name:.<55s} "
        f"{result['mean']:.4f}s +/- {result['std']:.4f}s "
        f"(min={result['min']:.4f}s)"
    )


def run_benchmark(title: str, fn: Any) -> None:
    """Run a single benchmark block, catching exceptions."""
    print(f"--- {title} ---")
    try:
        fn()
    except Exception:
        traceback.print_exc()
        print(f"  FAILED: {title}")
    print()


def run_all() -> None:
    print("=" * 70)
    print("Squidpy Benchmark Suite")
    print("=" * 70)

    has_njobs = _has_param(sq.gr.nhood_enrichment, "n_jobs")
    branch_label = (
        "main (with parallelize)" if has_njobs else "refactored (no parallelize)"
    )
    print(f"Branch type: {branch_label}")
    print()

    # ---- Load dataset ----
    print("Loading imc dataset...", flush=True)
    adata_imc = sq.datasets.imc()
    sq.gr.spatial_neighbors(adata_imc)
    print(f"  {adata_imc.n_obs} cells, {adata_imc.n_vars} genes")
    print()

    CK = "cell type"
    n_jobs_list = [1, 2, 4] if has_njobs else [1]

    # ---- nhood_enrichment ----
    def bench_nhood() -> None:
        for nj in n_jobs_list:
            ad = adata_imc.copy()
            kw = _njobs_kw(sq.gr.nhood_enrichment, nj)
            label = f"n_jobs={nj}" if has_njobs else "sequential"
            r = timed(
                sq.gr.nhood_enrichment,
                ad, cluster_key=CK, n_perms=100, seed=42, copy=True,
                **kw, repeats=3,
            )
            report(label, r)

    run_benchmark("nhood_enrichment (100 perms)", bench_nhood)

    # ---- centrality_scores ----
    def bench_centrality() -> None:
        for nj in n_jobs_list:
            ad = adata_imc.copy()
            kw = _njobs_kw(sq.gr.centrality_scores, nj)
            label = f"n_jobs={nj}" if has_njobs else "sequential+tqdm"
            r = timed(
                sq.gr.centrality_scores,
                ad, cluster_key=CK, copy=True,
                **kw, repeats=3,
            )
            report(label, r)

    run_benchmark("centrality_scores", bench_centrality)

    # ---- interaction_matrix ----
    def bench_interaction() -> None:
        ad = adata_imc.copy()
        for label, extra in [
            ("default", {}),
            ("weighted", {"weights": True}),
            ("normalized", {"normalized": True}),
        ]:
            r = timed(
                sq.gr.interaction_matrix,
                ad, cluster_key=CK, copy=True, **extra, repeats=5,
            )
            report(label, r)

    run_benchmark("interaction_matrix", bench_interaction)

    # ---- spatial_autocorr ----
    def bench_autocorr() -> None:
        ad = adata_imc.copy()
        ad.var["highly_variable"] = False
        ad.var.iloc[:20, ad.var.columns.get_loc("highly_variable")] = True
        for mode in ["moran", "geary"]:
            for nj in n_jobs_list:
                kw = _njobs_kw(sq.gr.spatial_autocorr, nj)
                label = (
                    f"{mode} n_jobs={nj}" if has_njobs else f"{mode} sequential"
                )
                r = timed(
                    sq.gr.spatial_autocorr,
                    ad.copy(), mode=mode, n_perms=50, seed=42, copy=True,
                    **kw, repeats=3,
                )
                report(label, r)

    run_benchmark("spatial_autocorr (20 genes, 50 perms)", bench_autocorr)

    # ---- co_occurrence ----
    def bench_cooccurrence() -> None:
        for nj in ([1, 2] if has_njobs else [1]):
            ad = adata_imc.copy()
            kw = _njobs_kw(sq.gr.co_occurrence, nj)
            label = f"n_jobs={nj}" if has_njobs else "numba"
            r = timed(
                sq.gr.co_occurrence,
                ad, cluster_key=CK, copy=True,
                **kw, repeats=2,
            )
            report(label, r)

    run_benchmark("co_occurrence", bench_cooccurrence)

    # ---- ligrec ----
    def bench_ligrec() -> None:
        from squidpy.gr._ligrec import PermutationTest

        ad = adata_imc.copy()
        pt = PermutationTest(ad, use_raw=False)
        try:
            pt.prepare(None)
        except ValueError:
            print("  SKIPPED: not enough genes for default interactions in this dataset")
            return
        n_interactions = len(pt.interactions)
        print(f"  ({n_interactions} interactions found)")

        for nj in ([1, 2] if has_njobs else [1]):
            ad = adata_imc.copy()
            kw: dict[str, Any] = {"n_jobs": nj} if has_njobs else {}
            label = f"n_jobs={nj}" if has_njobs else "sequential+numba"
            r = timed(
                sq.gr.ligrec,
                ad, CK, n_perms=25, seed=42, copy=True, use_raw=False,
                **kw, warmup=1, repeats=2,
            )
            report(label, r)

    run_benchmark("ligrec (25 perms)", bench_ligrec)

    print("=" * 70)
    print("Done.")


if __name__ == "__main__":
    run_all()
