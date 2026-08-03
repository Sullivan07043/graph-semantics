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
import pool_ext                                                       # noqa: E402
import testbeds                                                       # noqa: E402
from causallearn.search.HiddenCausal.RLCD.RLCD_alg import RLCD        # noqa: E402
from run_discovery import edges_from_cg                               # noqa: E402

sys.path.insert(0, HERE)
from polychoric import PolychoricRankTest, polychoric_matrix          # noqa: E402

NAME = os.environ.get("DATASET", "sixteenpf")
ROWS = int(os.environ.get("ROWS", 2000))
ALPHA = float(os.environ.get("ALPHA", 0.001))
MIN_CLUSTER = int(os.environ.get("MIN_CLUSTER", 3))
RLCD_MAX = int(os.environ.get("RLCD_MAX", 25))
WORKERS = int(os.environ.get("WORKERS", 1))
# The measure used for the two stages. The main pipeline was made nonlinear in July (dcor targets,
# GBR residualization) but this discovery path was left on Pearson, which is the inconsistency
# these switches address.
#   POLY=1  polychoric correlation in BOTH stages. Pearson attenuates on ordinal data, which
#           biases the skeleton low and inflates the rank test's type I error on skewed items
#           (rank_test_calibration.py). Ordinal data only.
#   SKEL=dcor  distance correlation for the SKELETON only. The skeleton asks a marginal
#           independence question, where dcor is strictly stronger than Pearson because it sees
#           nonlinear dependence. It cannot be used for the rank test: RLCD's identifiability is
#           a rank condition on a cross-covariance matrix, and dcor produces no matrix whose rank
#           carries that meaning. That is also why polychoric, not dcor, is the rank-test fix:
#           the rank theory is stated over the underlying continuous responses, which is exactly
#           what polychoric estimates and what Pearson-of-categories does not.
POLY = os.environ.get("POLY", "0") == "1"
SKEL = os.environ.get("SKEL", "poly" if POLY else "pearson")   # pearson | poly | dcor
# Default SEQUENTIAL, by measurement (16PF, 96 cores): cluster sizes are skewed, the
# critical path is the largest cluster, and sequential RLCD with the whole machine's BLAS
# beats process-parallel workers (235s vs 534s at 6 threads/worker vs 805s at 1 thread).
# Set WORKERS>1 for batches of many similar-size RLCD runs (ratification), where the
# critical path divides by the worker count; results are signature-identical either way
# (certified 2026-08-02).
LAT_RE = re.compile(r"^L\d+$")


def _rlcd_cluster(args):
    """Per-cluster RLCD worker (process pool). Returns raw prefixed edge lists; latent
    renumbering happens in the parent, in cluster order, so parallel output is identical
    to the sequential version. Launch the script with OMP_NUM_THREADS=1 so workers do not
    oversubscribe BLAS threads."""
    ci, Xc, names = args
    rt = PolychoricRankTest(Xc) if POLY else None
    cg = RLCD(Xc, node_names=[f"o::{nm}" for nm in names], ranktest_method=rt)
    d, u = edges_from_cg(cg)
    return ci, d, u


def _skeleton_measure_name():
    return {"pearson": "pearson", "poly": "polychoric", "dcor": "distance correlation"}[SKEL]


def _ratify_job(args):
    """Coverage-ratification worker: RLCD on (cluster core + candidate chunk). A candidate
    is ACCEPTED iff the re-run attaches it to a latent that also carries at least one core
    member of the cluster (i.e. it joins the cluster's measurement structure). Returns the
    accepted candidate names with the core members sharing that latent."""
    ci, Xc, core_names, cand_names = args
    names = core_names + cand_names
    cg = RLCD(Xc, node_names=[f"o::{nm}" for nm in names])
    d, u = edges_from_cg(cg)
    child = {}
    for a, b in d + u:
        if LAT_RE.match(a) and b.startswith("o::"):
            child.setdefault(a, set()).add(b[3:])
    core = set(core_names)
    out = []
    for L, ch in child.items():
        core_ch = ch & core
        if core_ch:
            for cand in ch & set(cand_names):
                out.append((cand, sorted(core_ch)))
    return ci, out


def marginal_skeleton(X, alpha, R=None, measure="pearson"):
    """Marginal dependence graph, used only as a permissive floor: the cluster threshold is chosen
    structurally afterwards.

    For correlation-type measures the edge test is Fisher-z, which is calibrated. Distance
    correlation has a different null distribution, so Fisher-z would be meaningless there; it gets
    the same 2/sqrt(n) magnitude floor the pipeline already uses to zero noise-level dependence.
    That floor is DECLARED as a heuristic, not a calibrated test."""
    n, p = X.shape
    R = np.corrcoef(X, rowvar=False) if R is None else R.copy()
    np.fill_diagonal(R, 0.0)
    if measure == "dcor":
        return R > 2.0 / np.sqrt(n)
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
    ds = {**testbeds.LOADERS, **pool.LOADERS, **pool_ext.LOADERS}[NAME]()
    g_pub, X = ds["graph"], np.asarray(ds["X"], float)
    obs = list(g_pub.observed)
    rng = np.random.default_rng(0)
    Xs = X[rng.choice(X.shape[0], min(ROWS, X.shape[0]), replace=False)]
    t0 = time.time()

    # CORR feeds the skeleton and the threshold sweep. RANK_R feeds RLCD's rank test and stays
    # polychoric-or-Pearson, never dcor (see the SKEL comment at the top).
    CORR, RANK_R = None, None
    off = ~np.eye(Xs.shape[1], dtype=bool)
    if POLY or SKEL == "poly":
        print(f"[{NAME}] polychoric correlation over {Xs.shape[1]} items", flush=True)
        RANK_R = polychoric_matrix(Xs, verbose=True)
        pear = np.corrcoef(Xs, rowvar=False)
        print(f"[{NAME}] mean |r|: pearson {np.abs(pear[off]).mean():.3f} -> "
              f"polychoric {np.abs(RANK_R[off]).mean():.3f} ({time.time() - t0:.0f}s)", flush=True)
    if SKEL == "poly":
        CORR = RANK_R
    elif SKEL == "dcor":
        import dependence as _dep
        print(f"[{NAME}] distance correlation over {Xs.shape[1]} items", flush=True)
        CORR = _dep.dcor_mat(Xs)
        pear = np.corrcoef(Xs, rowvar=False)
        print(f"[{NAME}] mean dependence: |pearson| {np.abs(pear[off]).mean():.3f} -> "
              f"dcor {CORR[off].mean():.3f} ({time.time() - t0:.0f}s)", flush=True)

    adj = marginal_skeleton(Xs, ALPHA, R=CORR, measure=SKEL)
    print(f"[{NAME}] alpha skeleton: {int(adj.sum() // 2)} edges (floor only; the cluster "
          f"threshold is chosen structurally below)", flush=True)

    # Psychometric data is globally dependent, so the significance floor alone leaves one
    # giant component. The cluster threshold is chosen by the STRUCTURAL requirement
    # instead: the smallest |r| cut at which the skeleton resolves into RLCD-reach
    # clusters. Selection = maximize items covered by clusters of size
    # [MIN_CLUSTER, RLCD_MAX]; tie-break toward the lower threshold. Sweep printed in full.
    R = np.abs(np.corrcoef(Xs, rowvar=False) if CORR is None else CORR)
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

    # Per-cluster RLCD runs in parallel processes (clusters are independent subproblems).
    # Renumbering stays in the parent, in cluster order: output is byte-identical to the
    # sequential version. (Observed names are prefixed for the calls because RLCD names its
    # internal latents L1, L2, ... and 16PF item codes include L1..L10.)
    from concurrent.futures import ProcessPoolExecutor

    small = [(ci, c) for ci, c in enumerate(clusters) if len(c) <= RLCD_MAX]
    n_workers = WORKERS or max(1, min(len(small), (os.cpu_count() or 4) - 2))
    print(f"[{NAME}] per-cluster RLCD: {len(small)} clusters on {n_workers} workers",
          flush=True)
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        results = dict()
        for ci, d, u in ex.map(_rlcd_cluster,
                               [(ci, Xs[:, c], [obs[i] for i in c]) for ci, c in small]):
            results[ci] = (d, u)

    edges, lat_n, cluster_lats = [], 0, {}
    for ci, c in enumerate(clusters):
        names = [obs[i] for i in c]
        if len(c) > RLCD_MAX:
            lat_n += 1
            L = f"L{lat_n}"
            edges += [(L, nm) for nm in names]
            print(f"[{NAME}]   cluster {ci}: {len(c)} > {RLCD_MAX}, single-latent fallback "
                  f"(declared)", flush=True)
            continue
        d, u = results[ci]
        ren = {}
        for e in d + u:
            for x in e:
                if LAT_RE.match(x) and x not in ren:
                    lat_n += 1
                    ren[x] = f"L{lat_n}"

        def deref(x, ren=ren):
            return x[3:] if x.startswith("o::") else ren.get(x, x)

        edges += [(deref(a), deref(b)) for a, b in d + u]
        cluster_lats[ci] = sorted(ren.values(), key=lambda s: int(s[1:]))
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

    # ---- Coverage ratification (2026-08-02): clustering proposes, RLCD disposes ----
    # Every item without a latent parent is nominated to its max-mean-|r| cluster, and is
    # accepted only if RLCD, re-run on (cluster core + candidate chunk), attaches it to a
    # latent that carries core members. Accepted items attach to the ORIGINAL cluster
    # latent with maximal child overlap against the ratifying latent's core members.
    # Refused items stay orphaned and are declared. Chunks are many similar small RLCD
    # runs: the process-parallel regime.
    # DEFAULT OFF: ratified attachment was measured harmful (accepted items completed at .205,
    # worse than .255 as orphans, and dragged natively clustered items from .656 to .531). The
    # code is kept so the negative result stays reproducible, not because it is a repair.
    RATIFY = os.environ.get("RATIFY", "0") == "1"
    ratified = []
    if RATIFY:
        parented = {b for a, b in edges if LAT_RE.match(a) and not LAT_RE.match(b)}
        orphans = [i for i, nm in enumerate(obs) if nm not in parented]
        noms = {}
        for i in orphans:
            ci = int(np.argmax([R[i, c].mean() for c in clusters]))
            noms.setdefault(ci, []).append(i)
        # Ratification runs against a REDUCED core: each cluster latent contributes its
        # RATIFY_TOPK highest-|r|-to-candidates indicators (rank identification needs a few
        # pure indicators per latent, not the full cluster), so jobs stay small and fast.
        # Collection is as_completed with a per-job timeout: a straggler forfeits only its
        # own chunk. Declared in the artifact.
        from concurrent.futures import as_completed
        TOPK = int(os.environ.get("RATIFY_TOPK", 4))
        JOB_TIMEOUT = int(os.environ.get("RATIFY_TIMEOUT", 600))
        own = {}
        for ci in cluster_lats:
            own[ci] = {L: sorted(b for x, b in edges
                                 if x == L and not LAT_RE.match(b))
                       for L in cluster_lats[ci]}
        jobs = []
        for ci, cand in sorted(noms.items()):
            if not cluster_lats.get(ci):
                print(f"[{NAME}] ratify: cluster {ci} has no latents, "
                      f"{len(cand)} nominees stay orphaned", flush=True)
                continue
            # total core budget: many-latent clusters must not blow the job size
            # (7 latents x topk 4 = 28 core > RLCD_MAX was measured to invert the saving)
            budget = int(os.environ.get("RATIFY_CORE_BUDGET", 10))
            k_eff = max(2, min(TOPK, budget // max(len(own[ci]), 1)))
            core_idx = []
            for L, ch in own[ci].items():
                ranked = sorted(ch, key=lambda nm: -R[np.ix_(cand, [obs.index(nm)])].mean())
                core_idx += [obs.index(nm) for nm in ranked[:k_eff]]
            core_idx = sorted(set(core_idx))[:budget]
            head = min(max(RLCD_MAX - len(core_idx), 4),
                       int(os.environ.get("RATIFY_CHUNK", 6)))
            for s in range(0, len(cand), head):
                chunk = cand[s:s + head]
                jobs.append((ci, Xs[:, core_idx + chunk],
                             [obs[i] for i in core_idx], [obs[i] for i in chunk]))
        rat_workers = max(1, min(int(os.environ.get("RATIFY_WORKERS", 8)), len(jobs) or 1))
        print(f"[{NAME}] ratify: {len(orphans)} orphans -> {len(jobs)} reduced-core RLCD "
              f"jobs (topk={TOPK}) on {rat_workers} workers", flush=True)
        if jobs:
            accepted_names = set()
            with ProcessPoolExecutor(max_workers=rat_workers) as ex:
                futs = {ex.submit(_ratify_job, j): j[0] for j in jobs}
                for fut in as_completed(futs, timeout=JOB_TIMEOUT * max(1, len(jobs))):
                    try:
                        ci, out = fut.result(timeout=JOB_TIMEOUT)
                    except Exception as e:
                        print(f"[{NAME}] ratify: job on cluster {futs[fut]} dropped "
                              f"({type(e).__name__})", flush=True)
                        continue
                    for cand, core_ch in out:
                        if cand in accepted_names:
                            continue
                        accepted_names.add(cand)
                        L_best = max(cluster_lats[ci],
                                     key=lambda L: len(set(own[ci][L]) & set(core_ch)))
                        edges.append((L_best, cand))
                        ratified.append([L_best, cand])
        print(f"[{NAME}] ratify: {len(ratified)}/{len(orphans)} orphans accepted by RLCD, "
              f"{len(orphans) - len(ratified)} remain orphaned (declared)", flush=True)

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
                      "rank_test": "polychoric" if POLY else "pearson (causal-learn default)",
                      "skeleton": ("marginal fisher-z floor + structural threshold on "
                                   + _skeleton_measure_name()),
                      "ratified_edges": ratified,
                      "top_layer_raw": [list(e) for e in top_edges]}}
    tag = os.environ.get("OUT_TAG", "")
    json.dump(out, open(os.path.join(HERE, "outputs", f"{NAME}{tag}.json"), "w"), indent=1)
    print(f"[{NAME}] saved outputs/{NAME}{tag}.json ({time.time() - t0:.0f}s total)", flush=True)
