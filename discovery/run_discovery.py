"""Discovery phase — the SINGLE entry.

RULING (user 2026-07-28): the RLCD-discovered structure passes DOWNSTREAM UNCHANGED. No
refinement of discovered graphs: the earlier V-driven edit loop was retired after it collapsed
every structure toward a single factor — V is a one-sided statistic (it penalizes violated
independence claims but does not reward held ones), so claim-poor graphs trivially minimize
it; using it as a search objective is invalid. V and cert remain READ-ONLY reports.
himi (n=202) is dropped from the phase.

Per dataset: RLCD (official causal-learn implementation, default parameters) -> adapter ->
outputs/<ds>.json {rlcd_directed, rlcd_undirected, rlcd_seconds} + an informational V report
(discovered vs published, both computed identically).

Env: DATASET (csv, default rse,cfcs), FRESH=1 (ignore cached structure).
Downstream translation/completion consumes these graphs via loader injection into the
official v6 runners (run_bigfive_hier pattern) — no parallel pipeline scripts.
"""
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
V6 = os.path.join(os.path.dirname(HERE), "v6")
sys.path.insert(0, V6)
# pin the repo's causallearn (upstream/ copy); the venv also has a pip causal-learn, never rely on it
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "causal-learn", "upstream", "causal-learn"))

import numpy as np                                                    # noqa: E402
import graph as G                                                     # noqa: E402
import testbeds                                                       # noqa: E402
import pool                                                           # noqa: E402
import terms as TF                                                    # noqa: E402
import nldep as NL                                                    # noqa: E402
import latent_constraints as LC                                       # noqa: E402
import adequacy                                                       # noqa: E402
from causallearn.search.HiddenCausal.RLCD.RLCD_alg import RLCD        # noqa: E402

LAT_RE = re.compile(r"^L\d+$")
OUT = os.path.join(HERE, "outputs")


def edges_from_cg(cg):
    Gg = cg.G
    nodes = Gg.get_nodes()
    names = [n.get_name() for n in nodes]
    A = Gg.graph
    directed, undirected = [], []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if A[i, j] == 0 and A[j, i] == 0:
                continue
            if A[j, i] == 1 and A[i, j] == -1:
                directed.append((names[i], names[j]))
            elif A[i, j] == 1 and A[j, i] == -1:
                directed.append((names[j], names[i]))
            else:
                undirected.append(tuple(sorted((names[i], names[j]))))
    return directed, undirected


def build(edges, obs_all):
    nodes = {x for e in edges for x in e}
    lats = sorted((n for n in nodes if LAT_RE.match(n)), key=lambda s: int(s[1:]))
    return G.Graph(lats, obs_all, edges)


def v_report(g, X, oi):
    """READ-ONLY adequacy numbers for the report (never a search objective)."""
    W, score = g.estimate_weights(X, oi)
    W, score = LC.sign_fix(g, W, score)
    mats = NL.matrices(g, X, oi, score, None)
    ci_full = TF.ci_table(g, X, oi, score, nl=mats, mode="full")
    return adequacy.compute(g, X, oi, score, ci=ci_full)


def run_dataset(name, fresh=False):
    ds = {**testbeds.LOADERS, **pool.LOADERS}[name]()
    g_pub, X = ds["graph"], ds["X"]
    obs = list(g_pub.observed)
    oi = {o: k for k, o in enumerate(obs)}
    path = os.path.join(OUT, f"{name}.json")
    if not fresh and os.path.exists(path):
        d = json.load(open(path))
        directed = [tuple(e) for e in d["rlcd_directed"]]
        undirected = [tuple(e) for e in d["rlcd_undirected"]]
        dt = d.get("rlcd_seconds")
    else:
        t0 = time.time()
        cg = RLCD(np.asarray(X, float), node_names=obs)
        dt = time.time() - t0
        directed, undirected = edges_from_cg(cg)
    g = build(directed + undirected, obs)
    print(f"[{name}] RLCD structure (passed downstream UNCHANGED): "
          f"{len(g.latents)} latents, {len(g.edges)} edges "
          f"({dt if dt else 'cached'}s)", flush=True)
    for L in g.latents:
        print(f"[{name}]   {L}: children = {sorted(g.children(L))}", flush=True)
    adq = v_report(g, X, oi)
    pub_path = os.path.join(V6, "outputs", "diagnostics", f"{name}.json")
    if os.path.exists(pub_path):
        p = json.load(open(pub_path))["adequacy"]
        print(f"[{name}] V (info only): discovered marg {adq['V_marginal']:.2f} / "
              f"cond {adq['V_conditional']:.2f}; published marg {p['V_marginal']:.2f} / "
              f"cond {p['V_conditional']:.2f}", flush=True)
    json.dump({"dataset": name, "rlcd_directed": [list(e) for e in directed],
               "rlcd_undirected": [list(e) for e in undirected], "rlcd_seconds": dt,
               "V_info": {k: adq[k] for k in adq if k != "top_pairs"}},
              open(path, "w"), indent=1)
    print(f"[{name}] saved outputs/{name}.json", flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for nm in [w.strip() for w in os.environ.get("DATASET", "rse,cfcs").split(",")]:
        try:
            run_dataset(nm, fresh=os.environ.get("FRESH", "0") == "1")
        except Exception as e:
            print(f"[{nm}] FAILED: {type(e).__name__}: {e}", flush=True)
    print("[discovery done]", flush=True)
