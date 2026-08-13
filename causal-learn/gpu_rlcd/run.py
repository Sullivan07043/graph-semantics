"""Run GPU-RLCD on one data matrix and save the discovered structure.

The package is dataset-agnostic: input is a matrix, output is a graph json. Dataset loading
belongs to the experiment layer (see discovery/run_latent_discovery.py).

Input, one of:
  NPZ=<file.npz>    arrays: X (n x p); names (p strings, optional)
  CSV=<file.csv>    header row = variable names
  SYNTH=<p>         built-in generator (p variables, p//6 latents), for scale tests

Other env: STAGE1=recboss|ges|all  MAXK=3  STAGE1_DISCOUNT=2  OUT=<json>
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np

from gpu_ranktest import GpuRankTest
from rlcd_gpu import RLCD_gpu

MAXK = int(os.environ.get("MAXK", 3))
STAGE1 = os.environ.get("STAGE1", "recboss")
DISCOUNT = float(os.environ.get("STAGE1_DISCOUNT", 2))


def synth(p, n=8000, seed=0):
    rng = np.random.default_rng(seed)
    n_lat = max(p // 6, 1)
    L = rng.normal(size=(n, n_lat))
    W = np.zeros((n_lat, p))
    for j in range(p):
        W[j % n_lat, j] = rng.uniform(0.6, 1.0)
        if rng.random() < 0.3:
            W[rng.integers(n_lat), j] += rng.uniform(0.1, 0.3)
    return L @ W + 0.6 * rng.normal(size=(n, p))


def load():
    if os.environ.get("NPZ"):
        d = np.load(os.environ["NPZ"], allow_pickle=True)
        X = np.asarray(d["X"], float)
        names = [str(c) for c in d["names"]] if "names" in d else None
        return X, names, os.path.basename(os.environ["NPZ"]).rsplit(".", 1)[0]
    if os.environ.get("CSV"):
        import pandas as pd
        df = pd.read_csv(os.environ["CSV"])
        return df.to_numpy(float), list(df.columns), \
            os.path.basename(os.environ["CSV"]).rsplit(".", 1)[0]
    if os.environ.get("SYNTH"):
        p = int(os.environ["SYNTH"])
        return synth(p), None, f"synth{p}"
    raise SystemExit("set one of NPZ= CSV= SYNTH=")


def main():
    X, names, tag = load()
    if names is None:
        names = [f"X{i + 1}" for i in range(X.shape[1])]
    out = os.environ.get("OUT", os.path.join(HERE, "outputs", f"{tag}_rlcd_gpu.json"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    print(f"[{tag}] n={X.shape[0]} p={X.shape[1]} maxk={MAXK} stage1={STAGE1}", flush=True)
    t0 = time.time()
    cg = RLCD_gpu(X, ranktest_method=GpuRankTest(X), node_names=names,
                  maxk=MAXK, stage1_method=STAGE1, stage1_discount=DISCOUNT)
    dt = time.time() - t0
    A = cg.G.graph
    all_names = [nd.get_name() for nd in cg.G.get_nodes()]
    latents = [nd for nd in all_names if nd.startswith("L")]
    edges = [[all_names[i], all_names[j]]
             for i in range(len(all_names)) for j in range(len(all_names))
             if A[j, i] == 1 and A[i, j] == -1]           # i --> j
    json.dump({"input": tag, "seconds": round(dt, 1), "maxk": MAXK, "stage1": STAGE1,
               "nodes": all_names, "latents": latents, "edges": edges},
              open(out, "w"), indent=1)
    print(f"[{tag}] {dt:.0f}s | {len(latents)} latents, {len(edges)} edges -> {out}", flush=True)
    for L in latents:
        ch = sorted(b for a, b in edges if a == L and not b.startswith("L"))
        print(f"   {L}: {ch}", flush=True)


if __name__ == "__main__":
    main()
