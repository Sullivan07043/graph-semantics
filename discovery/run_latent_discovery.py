"""Experiment layer for GPU-RLCD: load a project dataset, hand the matrix to the method.

The method package (causal-learn/gpu_rlcd) is dataset-agnostic. This script owns the dataset
side: it pulls X and variable names from the v6 loaders, writes a matrix npz, and calls the
package runner.

Env: DATASET=dass|wvs|... (a v6 pool_ext loader)  plus any run.py env (STAGE1, MAXK, OUT).
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "v6"))

import numpy as np

import pool_ext  # noqa: E402

DATASET = os.environ.get("DATASET", "dass")


def main():
    ds = pool_ext.LOADERS[DATASET]()
    X = np.asarray(ds["X"], float)
    names = list(ds["graph"].observed)
    npz = os.path.join(HERE, "outputs", f"{DATASET}_matrix.npz")
    os.makedirs(os.path.dirname(npz), exist_ok=True)
    np.savez(npz, X=X, names=np.array(names))
    env = dict(os.environ)
    env["NPZ"] = npz
    env.setdefault("OUT", os.path.join(HERE, "outputs", f"{DATASET}_rlcd_gpu.json"))
    runner = os.path.join(ROOT, "causal-learn", "gpu_rlcd", "run.py")
    sys.exit(subprocess.call([sys.executable, runner], env=env))


if __name__ == "__main__":
    main()
