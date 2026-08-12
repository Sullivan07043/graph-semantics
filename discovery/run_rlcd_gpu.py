"""Run certified GPU-RLCD on one dataset and save the discovered structure.

Env: DATASET=dass|synth150 OUT=<json> MAXK=3 STAGE1=ges|all
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "task3_robotics", "task3_pipeline_v1"))

import numpy as np

from gpu_ranktest import GpuRankTest
from rlcd_gpu import RLCD_gpu

DATASET = os.environ.get("DATASET", "dass")
MAXK = int(os.environ.get("MAXK", 3))
STAGE1 = os.environ.get("STAGE1", "ges")
OUT = os.environ.get("OUT", os.path.join(HERE, "outputs", f"{DATASET}_rlcd_gpu.json"))


def load():
    if DATASET == "dass":
        import pool_ext
        ds = pool_ext.dass()
        return np.asarray(ds["X"], float), list(ds["graph"].observed)
    if DATASET.startswith("synth"):
        p = int(DATASET[5:])
        rng = np.random.default_rng(0)
        n, n_lat = 8000, p // 6
        L = rng.normal(size=(n, n_lat))
        W = np.zeros((n_lat, p))
        for j in range(p):
            W[j % n_lat, j] = rng.uniform(0.6, 1.0)
            if rng.random() < 0.3:
                W[rng.integers(n_lat), j] += rng.uniform(0.1, 0.3)
        X = L @ W + 0.6 * rng.normal(size=(n, p))
        return X, [f"X{i + 1}" for i in range(p)]
    raise SystemExit(f"unknown DATASET {DATASET}")


def main():
    X, names = load()
    print(f"[{DATASET}] n={X.shape[0]} p={X.shape[1]} maxk={MAXK} stage1={STAGE1}", flush=True)
    t0 = time.time()
    cg = RLCD_gpu(X, ranktest_method=GpuRankTest(X), node_names=names,
                  maxk=MAXK, stage1_method=STAGE1)
    dt = time.time() - t0
    A = cg.G.graph
    all_names = [n.get_name() for n in cg.G.get_nodes()]
    latents = [n for n in all_names if n.startswith("L")]
    edges = []
    for i in range(len(all_names)):
        for j in range(len(all_names)):
            if A[j, i] == 1 and A[i, j] == -1:      # i --> j
                edges.append([all_names[i], all_names[j]])
    json.dump({"dataset": DATASET, "seconds": round(dt, 1), "maxk": MAXK, "stage1": STAGE1,
               "nodes": all_names, "latents": latents, "edges": edges},
              open(OUT, "w"), indent=1)
    kids = {L: sorted(b for a, b in edges if a == L and not b.startswith("L")) for L in latents}
    print(f"[{DATASET}] {dt:.0f}s | {len(latents)} latents, {len(edges)} edges -> {OUT}", flush=True)
    for L, ch in kids.items():
        print(f"   {L}: {ch}", flush=True)


if __name__ == "__main__":
    main()
