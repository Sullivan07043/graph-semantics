"""Certification for the GPU GIN stack, on real data only.

Stage 1: batched HSIC vs the official hsic_test_gamma on real column pairs and GIN-style
         e-vectors (float64; require max |p diff| < 1e-8 and stat agreement).
Stage 2: driver equivalence: gin_clusters(backend='gpu') vs gin_clusters(backend='reference')
         on a 15-variable bigfive subset (3 published factors x 5 items). The driver code is
         shared, so this isolates the HSIC backend swap. Requires identical clusters and
         identical causal order.

Env: ROWS=800, DEVICE=cuda:0 (launch with CUDA_VISIBLE_DEVICES=1).
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V6 = "/data2/shuhao/semantic_interpretation/graph_semantics/v6"
sys.path.insert(0, HERE)
sys.path.insert(0, V6)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "upstream", "causal-learn"))

import pool                                                            # noqa: E402
import testbeds                                                        # noqa: E402
from batched_hsic import ColumnBank                                    # noqa: E402
from gin_gpu import gin_clusters, _e_vectors                           # noqa: E402
from causallearn.search.FCMBased.lingam.hsic import hsic_test_gamma    # noqa: E402

ROWS = int(os.environ.get("ROWS", 800))
DEVICE = os.environ.get("DEVICE", "cuda:0")

if __name__ == "__main__":
    ds = {**testbeds.LOADERS, **pool.LOADERS}["bigfive"]()
    g_pub, X = ds["graph"], np.asarray(ds["X"], np.float64)
    obs = list(g_pub.observed)
    rng = np.random.default_rng(0)
    Xs = X[rng.choice(X.shape[0], ROWS, replace=False)]

    # ---- Stage 1: HSIC backend vs official, column pairs + e-vectors ----
    t0 = time.time()
    bank = ColumnBank(Xs, DEVICE)
    cov = np.cov(Xs.T)
    cands = [tuple(rng.choice(Xs.shape[1], 2, replace=False)) for _ in range(20)]
    cands = [tuple(int(v) for v in c) for c in cands]
    E, _ = _e_vectors(Xs, cov, cands, set(range(Xs.shape[1])))
    probes = np.concatenate([Xs.T[:20], E])                            # 40 probe rows
    stat_g, p_g = bank.score(probes)
    worst_p, worst_s = 0.0, 0.0
    checks = [(b, j) for b in range(probes.shape[0]) for j in
              rng.choice(Xs.shape[1], 6, replace=False)]
    for b, j in checks:
        s_ref, p_ref = hsic_test_gamma(probes[b][:, None], Xs[:, [int(j)]])
        worst_p = max(worst_p, abs(p_ref - p_g[b, int(j)]))
        worst_s = max(worst_s, abs(s_ref - stat_g[b, int(j)]) / max(abs(s_ref), 1e-12))
    ok1 = worst_p < 1e-8 and worst_s < 1e-8
    print(f"[stage1] {len(checks)} spot checks: max |p diff| = {worst_p:.2e}, "
          f"max rel stat diff = {worst_s:.2e} -> {'PASS' if ok1 else 'FAIL'} "
          f"({time.time() - t0:.0f}s)", flush=True)

    # ---- Stage 2: driver equivalence on a 15-var subset ----
    fac_items = {}
    for F in g_pub.latents:
        fac_items[F] = [o for o in g_pub.children(F) if not g_pub.is_latent(o)][:5]
    keep_names = [o for F in list(fac_items)[:3] for o in fac_items[F]]
    keep = [obs.index(nm) for nm in keep_names]
    Xsub = Xs[:, keep]
    t0 = time.time()
    ord_ref, rest_ref = gin_clusters(Xsub, backend='reference', verbose=False)
    t_ref = time.time() - t0
    t0 = time.time()
    ord_gpu, rest_gpu = gin_clusters(Xsub, backend='gpu', device=DEVICE, verbose=False)
    t_gpu = time.time() - t0
    ok2 = ord_ref == ord_gpu and rest_ref == rest_gpu
    print(f"[stage2] 15-var driver equivalence: {'PASS' if ok2 else 'FAIL'} "
          f"(reference {t_ref:.0f}s vs gpu {t_gpu:.0f}s)", flush=True)
    print(f"[stage2] clusters (ordered): {ord_gpu} + unordered {rest_gpu}", flush=True)
    fac_of = {o: F for F in g_pub.latents for o in g_pub.children(F)
              if not g_pub.is_latent(o)}
    for c in ord_gpu + rest_gpu:
        print(f"[stage2]   {sorted({fac_of[keep_names[i]] for i in c})} "
              f"<- {[keep_names[i] for i in c]}", flush=True)
    print(f"[certify] {'ALL PASS' if ok1 and ok2 else 'FAILED'}", flush=True)
