"""Time `sq.experimental.tl.impute_spage` end to end; run once per arm (PYTHONPATH picks the arm)."""

import json
import sys
import time

import numpy as np
import pandas as pd
from anndata import AnnData

import squidpy as sq

n_obs, arm, out = int(sys.argv[1]), sys.argv[2], sys.argv[3]
n_genes, n_measured, rank = 300, 200, 20
rng = np.random.default_rng(0)
loadings = rng.normal(size=(rank, n_genes))


def sample(n):
    return np.maximum(rng.normal(size=(n, rank)) @ loadings + rng.normal(size=(n, n_genes)), 0).astype(np.float32)


genes = [f"g{i}" for i in range(n_genes)]
ref = AnnData(sample(n_obs), var=pd.DataFrame(index=genes))
query = AnnData(sample(n_obs)[:, :n_measured], var=pd.DataFrame(index=genes[:n_measured]))


def run():
    return sq.experimental.tl.impute_spage(ref, query, genes=genes[n_measured:], n_jobs=-1, inplace=False)


run()  # warmup
times = []
for _ in range(3):
    t = time.perf_counter()
    res = run()
    times.append(time.perf_counter() - t)
np.save(f"{out}/{arm}_{n_obs}.npy", res.to_numpy())
print(json.dumps({"arm": arm, "n_obs": n_obs, "median_s": float(np.median(times)), "times": times,
                  "checksum": float(res.to_numpy().sum())}), flush=True)
