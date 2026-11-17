#!/bin/bash
#SBATCH -p cpu_p -q cpu_normal --exclusive --nodes=1 --mem=0 -t 3:00:00 -J spage-knn
#SBATCH -o /home/icb/selman.ozleyen/spage-knn/%j.log
set -uo pipefail
OUT=$HOME/spage-knn/$SLURM_JOB_ID; mkdir -p "$OUT"
S=/localscratch/$USER/$SLURM_JOB_ID; mkdir -p "$S"; cd "$S"
export UV_CACHE_DIR=$S/uv-cache TMPDIR=$S/tmp; mkdir -p "$TMPDIR"
hostname; nproc; df -BG /localscratch | tail -1; free -g | head -2
git clone -q https://github.com/selmanozleyen/squidpy.git repo
git -C repo worktree add -q ../sqeuclidean origin/feat/impute-spage
git -C repo worktree add -q ../cosine origin/bench/spage-cosine
echo "sqeuclidean $(git -C sqeuclidean rev-parse --short HEAD) cosine $(git -C cosine rev-parse --short HEAD)"
~/.local/bin/uv venv -q -p 3.13 venv
VIRTUAL_ENV=$S/venv ~/.local/bin/uv pip install -q -e ./sqeuclidean
venv/bin/python -c "import sklearn, numpy; print('sklearn', sklearn.__version__, 'numpy', numpy.__version__)"
for n in 10000 50000 200000; do
  for jobs in -1 64; do
    for arm in sqeuclidean cosine; do
      PYTHONPATH=$S/$arm/src venv/bin/python -X faulthandler cosine/benchmarks/spage_knn.py $n $arm "$OUT" $jobs 2>&1 \
        | grep -v "FutureWarning\|read_10x" | tee -a "$OUT/results.jsonl"
      echo "exit $arm n=$n n_jobs=$jobs: ${PIPESTATUS[0]}" | tee -a "$OUT/results.jsonl"
    done
    venv/bin/python -c "import numpy as np; a, b = (np.load('$OUT/'+x+'_${n}_$jobs.npy') for x in ('sqeuclidean', 'cosine')); print('n=$n n_jobs=$jobs max abs diff', np.abs(a-b).max(), 'max rel', (np.abs(a-b)/np.maximum(np.abs(a),1e-12)).max())" 2>&1 | tail -1 | tee -a "$OUT/results.jsonl"
  done
done
