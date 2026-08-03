"""GPU-GIN pilot: full-dataset clustering with the certified stack (certify.py ALL PASS
required first). Prints clusters against published factors, saves JSON next to the other
discovery artifacts.

Env: DATASET=bigfive ROWS=800 ALPHA=0.05 DEVICE=cuda:0 (launch with CUDA_VISIBLE_DEVICES=1).
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V6 = "/data2/shuhao/semantic_interpretation/graph_semantics/v6"
OUT = "/data2/shuhao/semantic_interpretation/graph_semantics/discovery/outputs"
sys.path.insert(0, HERE)
sys.path.insert(0, V6)

import pool                                                            # noqa: E402
import testbeds                                                        # noqa: E402
from gin_gpu import gin_clusters                                       # noqa: E402

NAME = os.environ.get("DATASET", "bigfive")
ROWS = int(os.environ.get("ROWS", 800))
ALPHA = float(os.environ.get("ALPHA", 0.05))
DEVICE = os.environ.get("DEVICE", "cuda:0")

if __name__ == "__main__":
    ds = {**testbeds.LOADERS, **pool.LOADERS}[NAME]()
    g_pub, X = ds["graph"], np.asarray(ds["X"], np.float64)
    obs = list(g_pub.observed)
    rng = np.random.default_rng(0)
    Xs = X[rng.choice(X.shape[0], min(ROWS, X.shape[0]), replace=False)]
    fac_of = {o: F for F in g_pub.latents for o in g_pub.children(F)
              if not g_pub.is_latent(o)}
    print(f"[{NAME}] gpu-gin pilot: {len(obs)} vars, rows={ROWS}, alpha={ALPHA}", flush=True)
    t0 = time.time()
    order, rest = gin_clusters(Xs, alpha=ALPHA, backend='gpu', device=DEVICE)
    dt = time.time() - t0
    print(f"[{NAME}] gpu-gin done in {dt:.0f}s", flush=True)
    all_clusters = [("ordered", c) for c in order] + [("unordered", c) for c in rest]
    covered = 0
    for tag, c in all_clusters:
        names = [obs[i] for i in c]
        covered += len(c)
        facs = sorted({fac_of.get(nm, "?") for nm in names})
        print(f"[{NAME}]   {tag} cluster: {len(c)} items, factors {facs}: {names}",
              flush=True)
    print(f"[{NAME}] {len(all_clusters)} clusters, {covered}/{len(obs)} items covered",
          flush=True)
    json.dump({"dataset": NAME, "rows": ROWS, "alpha": ALPHA, "seconds": dt,
               "causal_order": [[obs[i] for i in c] for c in order],
               "unordered": [[obs[i] for i in c] for c in rest]},
              open(os.path.join(OUT, f"{NAME}_gingpu.json"), "w"), indent=1)
    print(f"[{NAME}] saved {NAME}_gingpu.json", flush=True)
