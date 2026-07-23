"""Phase-1 infrastructure: dependence matrices for the bridge constraint.

For each dataset we compute THREE dependence measures, each at TWO levels:
  - marginal: dep(X_i, X_j)
  - conditional-on-parents: dep(res_i, res_j) where res = X residualized on the latent parent scores
    (the level the graph assigns to sibling pairs)
Measures:
  - pearson: |corr| (the old default; second-order only)
  - dcor:    distance correlation (nonlinear, hyperparameter-free)
  - mi:      kNN (KSG-style via sklearn mutual_info_regression), symmetrized, rescaled to [0,1)
             via dep = sqrt(1 - exp(-2*MI)) (the Gaussian-equivalent correlation transform)
Cached per dataset+measure at outputs/dependence/<ds>_<level>_<measure>.npy (order = graph.observed).

Usage: python pipeline_v3/dependence.py [datasets csv]   (default: DEV+HELDOUT all)
"""
import os, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import pool, optimize
from run_task1 import ALL_LOADERS

OUT = os.path.join(ROOT, "outputs", "dependence")
SUB = int(os.environ.get("DEP_SUBSAMPLE", 5000))         # samples used for dcor/MI (n^2 memory)


def ts():
    return time.strftime("%H:%M:%S")


def _sub(X, rng):
    if X.shape[0] > SUB:
        X = X[rng.choice(X.shape[0], SUB, replace=False)]
    return X


def pearson_mat(X):
    C = np.abs(np.corrcoef(X.T))
    np.fill_diagonal(C, 0.0)
    return C


def dcor_mat(X):
    """Pairwise distance correlation via precomputed double-centered distance matrices.
    Memory = m*n^2*4 bytes; n is adaptively reduced for wide datasets to stay ~<=1GB."""
    n, m = X.shape
    cap = max(500, int(np.sqrt(1e9 / (4 * m))))
    if n > cap:
        X = X[np.random.default_rng(1).choice(n, cap, replace=False)]
        n = cap
    A = np.empty((m, n, n), np.float32)
    for j in range(m):
        d = np.abs(X[:, j][:, None] - X[:, j][None, :]).astype(np.float32)
        A[j] = d - d.mean(0, keepdims=True) - d.mean(1, keepdims=True) + d.mean()
    V = np.array([(A[j] * A[j]).mean() for j in range(m)])
    C = np.zeros((m, m))
    for i in range(m):
        Ai = A[i]
        for j in range(i + 1, m):
            cov = float((Ai * A[j]).mean())
            den = float(np.sqrt(V[i] * V[j])) + 1e-12
            C[i, j] = C[j, i] = np.sqrt(max(cov, 0.0) / den)
    return C


def mi_mat(X):
    from sklearn.feature_selection import mutual_info_regression
    n, m = X.shape
    C = np.zeros((m, m))
    for i in range(m):
        mi = mutual_info_regression(X[:, [j for j in range(m) if j != i]], X[:, i],
                                    n_neighbors=5, random_state=0)
        k = 0
        for j in range(m):
            if j == i:
                continue
            C[i, j] = max(C[i, j], mi[k]); C[j, i] = C[i, j]
            k += 1
    # Gaussian-equivalent correlation: dep = sqrt(1 - exp(-2 MI)) in [0,1)
    return np.sqrt(1.0 - np.exp(-2.0 * np.clip(C, 0, None)))


def residualize(g, X, oi, score):
    R = np.zeros_like(X)
    for k, o in enumerate(g.observed):
        y = X[:, oi[o]]
        regs = [score[p] for p in g.parents(o) if g.is_latent(p) and p in score]
        if regs:
            A = np.stack(regs, 1)
            beta, *_ = np.linalg.lstsq(A, y, rcond=None)
            y = y - A @ beta
        R[:, k] = y
    return (R - R.mean(0)) / (R.std(0) + 1e-9)


def run(name):
    os.makedirs(OUT, exist_ok=True)
    ds = ALL_LOADERS[name]()
    g, X = ds["graph"], ds["X"]
    oi = {o: k for k, o in enumerate(g.observed)}
    _, score = g.estimate_weights(X, oi)
    rng = np.random.default_rng(0)
    levels = {"marginal": _sub(X, rng), "conditional": _sub(residualize(g, X, oi, score), rng)}
    for lv, M in levels.items():
        for meas, fn in [("pearson", pearson_mat), ("dcor", dcor_mat), ("mi", mi_mat)]:
            p = os.path.join(OUT, f"{name}_{lv}_{meas}.npy")
            if os.path.exists(p):
                continue
            t0 = time.time()
            np.save(p, fn(M).astype(np.float32))
            print(f"[{ts()}] {name:10s} {lv:11s} {meas:7s} done in {time.time()-t0:.1f}s", flush=True)


def load(name, level, measure):
    return np.load(os.path.join(OUT, f"{name}_{level}_{measure}.npy"))


if __name__ == "__main__":
    names = (sys.argv[1].split(",") if len(sys.argv) > 1 else pool.DEV + pool.HELDOUT)
    for n in names:
        run(n.strip())
