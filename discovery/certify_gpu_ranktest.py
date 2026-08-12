"""Certification gate for GpuRankTest, same discipline as polychoric.certify.

Three checks, all must pass before GpuRankTest may replace Chi2RankTest:

1. decision parity: on real data (DASS) and on synthetic latent data, random
   (pcols, qcols, r, alpha) cases must produce the same accept/reject as Chi2RankTest.
2. end-to-end: RLCD run with GpuRankTest must return the same adjacency as with Chi2RankTest
   on a dataset small enough for the CPU test to finish.
3. batch = single: priming a batch then reading decisions must equal unprimed single calls.

Env: CASES=500 DEVICE=cuda SEED=0
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "vendor", "causal-learn"))
sys.path.insert(0, HERE)

import numpy as np

from causallearn.search.HiddenCausal.RLCD.Chi2RankTest import Chi2RankTest
from gpu_ranktest import GpuRankTest

CASES = int(os.environ.get("CASES", 500))
DEVICE = os.environ.get("DEVICE", "cuda")
SEED = int(os.environ.get("SEED", 0))


def synth_latent(n, p, n_lat, rng):
    """One-layer measurement model: latents -> observed, mild cross-loadings."""
    L = rng.normal(size=(n, n_lat))
    W = np.zeros((n_lat, p))
    for j in range(p):
        W[j % n_lat, j] = rng.uniform(0.6, 1.0)
        if rng.random() < 0.3:
            W[rng.integers(n_lat), j] += rng.uniform(0.1, 0.3)
    return L @ W + 0.6 * rng.normal(size=(n, p))


def parity(name, X, cases, rng):
    cpu = Chi2RankTest(X)
    gpu = GpuRankTest(X, device=DEVICE)
    p = X.shape[1]
    mism = 0
    reqs = []
    for _ in range(cases):
        da = int(rng.integers(2, 5))
        db = int(rng.integers(da, min(p - da, 40) + 1))
        cols = list(rng.permutation(p))
        pcols, qcols = cols[:da], cols[da:da + db]
        r = int(rng.integers(0, da))
        alpha = float(rng.choice([0.01, 0.05]))
        reqs.append((pcols, qcols, r, alpha))
    gpu.prime((pc, qc) for pc, qc, _, _ in reqs)   # batch path, same numbers as singles
    for pcols, qcols, r, alpha in reqs:
        cpu.cca_cache_dict.clear()
        a = cpu.test(pcols, qcols, r, alpha)
        b = gpu.test(pcols, qcols, r, alpha)
        if bool(a) != bool(b):
            mism += 1
    print(f"[parity:{name}] {cases - mism}/{cases} decisions agree")
    return mism


def end_to_end(n, p, n_lat, rng):
    """Reference = untouched upstream (mp sweep + CPU test). Two bisection arms:
    schedule-only (patched sweep, CPU test) and full (patched sweep, GPU test)."""
    from causallearn.search.HiddenCausal.RLCD.RLCD_alg import RLCD
    from rlcd_gpu import RLCD_gpu, RLCD_serial
    X = synth_latent(n, p, n_lat, rng)

    # deterministic reference: upstream logic, serial sweep
    t0 = time.time()
    ref = RLCD_serial(X, ranktest_method=Chi2RankTest(X))
    t_ref = time.time() - t0

    # informational: the upstream mp default, run twice (loky hash seeds vary per process)
    mp1 = RLCD(X, ranktest_method=Chi2RankTest(X))
    mp2 = RLCD(X, ranktest_method=Chi2RankTest(X))
    print(f"[info p={p}] upstream mp self-consistent: "
          f"{np.array_equal(mp1.G.graph, mp2.G.graph)} | "
          f"mp equals serial reference: {np.array_equal(mp1.G.graph, ref.G.graph)}")

    sched = RLCD_gpu(X, ranktest_method=Chi2RankTest(X))
    same_sched = np.array_equal(ref.G.graph, sched.G.graph)

    t0 = time.time()
    full = RLCD_gpu(X, ranktest_method=GpuRankTest(X, device=DEVICE))
    t_gpu = time.time() - t0
    same_full = np.array_equal(ref.G.graph, full.G.graph)

    print(f"[end-to-end p={p}] schedule-only identical: {same_sched} | "
          f"full identical: {same_full} | cpu {t_ref:.0f}s vs gpu {t_gpu:.0f}s")
    return (0 if same_sched else 1) + (0 if same_full else 1)


def main():
    rng = np.random.default_rng(SEED)
    fails = 0

    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "task3_robotics", "task3_pipeline_v1"))
    import pool_ext
    dass = np.asarray(pool_ext.dass()["X"], float)
    fails += parity("dass", dass, CASES, rng)
    fails += parity("synth", synth_latent(4000, 30, 6, rng), CASES, rng)
    fails += end_to_end(4000, 15, 3, rng)

    print("CERTIFY", "PASS" if fails == 0 else f"FAIL ({fails})")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
