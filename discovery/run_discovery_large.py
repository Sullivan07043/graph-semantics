"""Discovery for datasets BEYOND RLCD reach (>25 items): the layered composition.

Layer 1 (global, cheap): the marginal-independence skeleton. This is exactly the level-0
skeleton of FCI (Fisher-z at ALPHA on a ROWS subsample), computed directly. causal-learn's
fci() is not usable here at any depth: its possible-d-sep orientation stage does subset
searches that the depth cap does not bound, and on measurement-model data (within-factor
pairs have no observed separating set) it does not terminate. Clusters do not need
orientation, only adjacency.

Layer 2 (local, exact): RLCD (official) inside each cluster of size >= MIN_CLUSTER,
building that cluster's latent layer. Clusters above RLCD reach are declared and get a
single latent over their items (fallback, none expected on 16PF).

Layer 3 (top, small): RLCD over the cluster scores (PC1 per cluster) for the
latent-latent structure.

Output: discovery/outputs/<name>.json in the same schema as run_discovery.py, so
run_downstream.py consumes it unchanged. All parameters recorded in the JSON.

Env: DATASET (default sixteenpf), ROWS=2000, ALPHA=0.001, MIN_CLUSTER=3, RLCD_MAX=25.
"""
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
V6 = os.path.join(os.path.dirname(HERE), "v6")
sys.path.insert(0, V6)

import numpy as np                                                    # noqa: E402
from scipy.stats import norm                                          # noqa: E402
import pool                                                           # noqa: E402
import testbeds                                                       # noqa: E402
from causallearn.search.HiddenCausal.RLCD.RLCD_alg import RLCD        # noqa: E402
from run_discovery import edges_from_cg                               # noqa: E402

NAME = os.environ.get("DATASET", "sixteenpf")
ROWS = int(os.environ.get("ROWS", 2000))
ALPHA = float(os.environ.get("ALPHA", 0.001))
MIN_CLUSTER = int(os.environ.get("MIN_CLUSTER", 3))
RLCD_MAX = int(os.environ.get("RLCD_MAX", 25))
LAT_RE = re.compile(r"^L\d+$")


def marginal_skeleton(X, alpha):
    """Fisher-z marginal independence graph: adjacency iff |z(r)| clears alpha."""
    n, p = X.shape
    R = np.corrcoef(X, rowvar=False)
    np.fill_diagonal(R, 0.0)
    z = 0.5 * np.log((1 + R) / (1 - R + 1e-12))
    crit = norm.ppf(1 - alpha / 2) / np.sqrt(n - 3)
    return np.abs(z) > crit


def components(adj):
    p = adj.shape[0]
    seen, comps = set(), []
    for s in range(p):
        if s in seen:
            continue
        stack, comp = [s], []
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            comp.append(u)
            stack.extend(int(v) for v in np.nonzero(adj[u])[0] if v not in seen)
        comps.append(sorted(comp))
    return comps


if __name__ == "__main__":
    ds = {**testbeds.LOADERS, **pool.LOADERS}[NAME]()
    g_pub, X = ds["graph"], np.asarray(ds["X"], float)
    obs = list(g_pub.observed)
    rng = np.random.default_rng(0)
    Xs = X[rng.choice(X.shape[0], min(ROWS, X.shape[0]), replace=False)]
    t0 = time.time()

    adj = marginal_skeleton(Xs, ALPHA)
    print(f"[{NAME}] alpha skeleton: {int(adj.sum() // 2)} edges (floor only; the cluster "
          f"threshold is chosen structurally below)", flush=True)

    # Psychometric data is globally dependent, so the significance floor alone leaves one
    # giant component. The cluster threshold is chosen by the STRUCTURAL requirement
    # instead: the smallest |r| cut at which the skeleton resolves into RLCD-reach
    # clusters. Selection = maximize items covered by clusters of size
    # [MIN_CLUSTER, RLCD_MAX]; tie-break toward the lower threshold. Sweep printed in full.
    R = np.abs(np.corrcoef(Xs, rowvar=False))
    np.fill_diagonal(R, 0.0)
    best, sweep = None, []
    for thr in np.arange(0.05, 0.61, 0.01):
        comps_t = components((R > thr) & adj)
        good = [c for c in comps_t if MIN_CLUSTER <= len(c) <= RLCD_MAX]
        covered = sum(len(c) for c in good)
        sweep.append((round(float(thr), 2), len(good), covered))
        if best is None or covered > best[2]:
            best = (round(float(thr), 2), len(good), covered, comps_t)
    print(f"[{NAME}] threshold sweep (thr, clusters, items covered): {sweep}", flush=True)
    THR = best[0]
    comps = best[3]
    clusters = [c for c in comps if len(c) >= MIN_CLUSTER]
    singles = [c for c in comps if len(c) < MIN_CLUSTER]
    print(f"[{NAME}] chosen thr={THR}: {len(clusters)} clusters, "
          f"{sum(len(s) for s in singles)} items unclustered (declared, no latent parent)",
          flush=True)
    fac_of = {o: F for F in g_pub.latents for o in g_pub.children(F)
              if not g_pub.is_latent(o)}
    for ci, c in enumerate(clusters):
        names = [obs[i] for i in c]
        facs = sorted({fac_of.get(nm, "?") for nm in names})
        print(f"[{NAME}]   cluster {ci}: {len(c)} items, published factors: {facs}",
              flush=True)

    edges, lat_n = [], 0
    for ci, c in enumerate(clusters):
        names = [obs[i] for i in c]
        if len(c) > RLCD_MAX:
            lat_n += 1
            L = f"L{lat_n}"
            edges += [(L, nm) for nm in names]
            print(f"[{NAME}]   cluster {ci}: {len(c)} > {RLCD_MAX}, single-latent fallback "
                  f"(declared)", flush=True)
            continue
        # RLCD names its internal latents L1, L2, ... and 16PF item codes include L1..L10
        # (scale L), so observed names are prefixed for the call and stripped after.
        cg = RLCD(Xs[:, c], node_names=[f"o::{nm}" for nm in names])
        d, u = edges_from_cg(cg)
        ren = {}
        for e in d + u:
            for x in e:
                if LAT_RE.match(x) and x not in ren:
                    lat_n += 1
                    ren[x] = f"L{lat_n}"

        def deref(x, ren=ren):
            return x[3:] if x.startswith("o::") else ren.get(x, x)

        edges += [(deref(a), deref(b)) for a, b in d + u]
        print(f"[{NAME}]   cluster {ci}: RLCD -> {len(ren)} latents, {len(d) + len(u)} edges",
              flush=True)

    # Directed + undirected RLCD output can compose reciprocal pairs; the downstream graph
    # is a DAG consumer (sign_fix recursion), so dedupe by unordered pair (first orientation
    # wins) and drop self-loops.
    seen_pairs, deduped = set(), []
    for a, b in edges:
        if a == b:
            continue
        k = frozenset((a, b))
        if k in seen_pairs:
            continue
        seen_pairs.add(k)
        deduped.append((a, b))
    if len(deduped) != len(edges):
        print(f"[{NAME}] deduped {len(edges) - len(deduped)} reciprocal/self edges",
              flush=True)
    edges = deduped

    # Layer 3: latent-latent over cluster scores (PC1 per cluster), only if >1 cluster.
    top_edges = []
    if len(clusters) > 1:
        S = []
        for c in clusters:
            Z = Xs[:, c] - Xs[:, c].mean(0)
            _, _, vt = np.linalg.svd(Z, full_matrices=False)
            S.append(Z @ vt[0])
        S = np.asarray(S).T
        snames = [f"C{ci}" for ci in range(len(clusters))]
        try:
            cg_top = RLCD(S, node_names=snames)
            dt_, ut_ = edges_from_cg(cg_top)
            top_edges = dt_ + ut_
            print(f"[{NAME}] top layer: RLCD over {len(clusters)} cluster scores -> "
                  f"{len(top_edges)} edges", flush=True)
        except Exception as e:
            print(f"[{NAME}] top layer failed ({type(e).__name__}: {e}); latents left "
                  f"unconnected (declared)", flush=True)

    out = {"dataset": NAME, "rlcd_directed": [list(e) for e in edges],
           "rlcd_undirected": [], "rlcd_seconds": time.time() - t0,
           "params": {"rows": ROWS, "alpha": ALPHA, "min_cluster": MIN_CLUSTER,
                      "rlcd_max": RLCD_MAX, "cluster_threshold": THR,
                      "skeleton": "marginal fisher-z floor + structural |r| threshold",
                      "top_layer_raw": [list(e) for e in top_edges]}}
    json.dump(out, open(os.path.join(HERE, "outputs", f"{NAME}.json"), "w"), indent=1)
    print(f"[{NAME}] saved outputs/{NAME}.json ({time.time() - t0:.0f}s total)", flush=True)
