"""Discovery phase — the SINGLE entry (consolidated 2026-07-28; the earlier check_discovered/
refine/translate_discovered scripts are folded in here or dropped).

Default mode, per dataset: RLCD (raw structure cached in outputs/<ds>.json, reused unless
FRESH=1) -> two-move V-driven refinement loop:
    candidates each iteration = ADD (one latent over the top violating component's roots)
    and MERGE (collapse the most score-dependent latent pair); evaluate V(G,X) for each,
    take the best strict reducer, stop when none reduces V. 10-iteration runaway guard.
-> report V (marginal/conditional) of raw/refined vs the published graph (read from
v6/outputs/diagnostics/<ds>.json) and save everything into ONE json:
outputs/<ds>.json = {rlcd_directed, rlcd_undirected, refined_directed, edits, V_trajectory,
V_published}.

SWEEP=1: RLCD parameter sweep table (alpha x stage1_method x partition_thres) instead of
refinement — prints #latents/#edges/V per config, saves outputs/<ds>_sweep.json.

Env: DATASET (csv, default rse,cfcs,himi), FRESH=1 (rerun RLCD), SWEEP=1.
Translation of discovered graphs is NOT here — when scheduled, it goes through the official
v6 runners via loader injection (run_bigfive_hier pattern), not a parallel script.
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


# ------------------------------------------------------------------ RLCD + adapter
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


def rlcd_edges(name, X, obs, fresh=False, **params):
    """Raw RLCD structure, cached in outputs/<ds>.json unless fresh/params given."""
    path = os.path.join(OUT, f"{name}.json")
    if not fresh and not params and os.path.exists(path):
        d = json.load(open(path))
        if "rlcd_directed" in d:
            return [tuple(e) for e in d["rlcd_directed"]], \
                   [tuple(e) for e in d["rlcd_undirected"]], d.get("rlcd_seconds")
    t0 = time.time()
    cg = RLCD(np.asarray(X, float), node_names=obs, **params)
    dt = time.time() - t0
    directed, undirected = edges_from_cg(cg)
    return directed, undirected, dt


def build(edges, obs_all):
    nodes = {x for e in edges for x in e}
    lats = sorted((n for n in nodes if LAT_RE.match(n)), key=lambda s: int(s[1:]))
    return G.Graph(lats, obs_all, edges)


# ------------------------------------------------------------------ V + edit moves
def v_state(g, X, oi):
    W, score = g.estimate_weights(X, oi)
    W, score = LC.sign_fix(g, W, score)
    mats = NL.matrices(g, X, oi, score, None)
    ci_full = TF.ci_table(g, X, oi, score, nl=mats, mode="full")
    adq = adequacy.compute(g, X, oi, score, ci=ci_full)
    props = adequacy.propose_repairs(g, ci_full)
    return adq, props, mats


def vtotal(adq):
    return adq["V_marginal"] + adq["V_conditional"]


def next_lat(g):
    return f"L{1 + max([int(L[1:]) for L in g.latents], default=0)}"


def cand_add(edges, g, props):
    if not props:
        return None
    comp = props[0]["nodes"]
    comp_set = set(comp)
    roots = [n for n in comp if not any(p in comp_set for p in g.parents(n))]
    L = next_lat(g)
    return edges + [(L, r) for r in roots], {"op": "add", "latent": L, "over": roots}


def cand_merge(edges, g, mats):
    lats = [L for L in g.latents if L in mats["idx"]]
    if len(lats) < 2:
        return None
    best, bv = None, -1.0
    for i, a in enumerate(lats):
        for b in lats[i + 1:]:
            v = mats["marg_dcor"][mats["idx"][a], mats["idx"][b]]
            if v > bv:
                best, bv = (a, b), v
    a, b = best
    M = next_lat(g)
    new_edges, seen = [], set()
    for (u, v) in edges:
        u2 = M if u in (a, b) else u
        v2 = M if v in (a, b) else v
        if u2 == v2 or (u2, v2) in seen:
            continue
        seen.add((u2, v2))
        new_edges.append((u2, v2))
    return new_edges, {"op": "merge", "merged": [a, b], "as": M, "dcor": round(float(bv), 3)}


def refine(name, edges, obs, X, oi):
    g = build(edges, obs)
    edits, traj = [], []
    for it in range(10):
        adq, props, mats = v_state(g, X, oi)
        vt = vtotal(adq)
        traj.append(round(vt, 3))
        print(f"[{name}] iter {it}: V={vt:.2f} (marg {adq['V_marginal']:.2f} + "
              f"cond {adq['V_conditional']:.2f}), latents={g.latents}", flush=True)
        best_edges, best_edit, best_v = None, None, vt
        for cand in (cand_add(edges, g, props), cand_merge(edges, g, mats)):
            if cand is None:
                continue
            cand_edges, edit = cand
            adq_c, _, _ = v_state(build(cand_edges, obs), X, oi)
            v_c = vtotal(adq_c)
            print(f"[{name}]   candidate {edit['op']} "
                  f"({edit.get('latent') or edit.get('as')}): V={v_c:.2f}", flush=True)
            if v_c < best_v:
                best_edges, best_edit, best_v = cand_edges, edit, v_c
        if best_edit is None:
            print(f"[{name}] stop: no candidate reduces V", flush=True)
            break
        edges = best_edges
        edits.append(best_edit)
        print(f"[{name}]   apply: {best_edit}", flush=True)
        g = build(edges, obs)
    return edges, edits, traj


# ------------------------------------------------------------------ modes
def run_dataset(name, fresh=False):
    ds = {**testbeds.LOADERS, **pool.LOADERS}[name]()
    g_pub, X = ds["graph"], ds["X"]
    obs = list(g_pub.observed)
    oi = {o: k for k, o in enumerate(obs)}
    directed, undirected, dt = rlcd_edges(name, X, obs, fresh=fresh)
    print(f"[{name}] RLCD raw: {len(directed)} directed + {len(undirected)} undirected edges "
          f"({dt if dt else 'cached'}s)", flush=True)
    edges0 = directed + undirected
    edges, edits, traj = refine(name, edges0, obs, X, oi)
    pub_path = os.path.join(V6, "outputs", "diagnostics", f"{name}.json")
    v_pub = None
    if os.path.exists(pub_path):
        p = json.load(open(pub_path))["adequacy"]
        v_pub = {k: p[k] for k in p if k != "top_pairs"}
        print(f"[{name}] published graph V: marg {p['V_marginal']:.2f} + "
              f"cond {p['V_conditional']:.2f}", flush=True)
    g_fin = build(edges, obs)
    print(f"[{name}] FINAL: latents={g_fin.latents}, V_traj={traj}", flush=True)
    json.dump({"dataset": name, "rlcd_directed": directed, "rlcd_undirected": undirected,
               "rlcd_seconds": dt, "refined_directed": [list(e) for e in edges],
               "edits": edits, "V_trajectory": traj, "V_published": v_pub},
              open(os.path.join(OUT, f"{name}.json"), "w"), indent=1)
    print(f"[{name}] saved outputs/{name}.json", flush=True)


def run_sweep(name):
    ds = {**testbeds.LOADERS, **pool.LOADERS}[name]()
    g_pub, X = ds["graph"], ds["X"]
    obs = list(g_pub.observed)
    oi = {o: k for k, o in enumerate(obs)}
    grid = []
    for alpha in (0.005, 0.01, 0.05):
        for s1 in ("ges", "fges"):
            for pthres in (2, 3):
                grid.append(dict(alpha_dict={0: alpha, 1: alpha, 2: alpha, 3: alpha},
                                 stage1_method=s1, stage1_partition_thres=pthres))
    rows = []
    for cfg in grid:
        tag = (f"a={list(cfg['alpha_dict'].values())[0]} s1={cfg['stage1_method']} "
               f"pt={cfg['stage1_partition_thres']}")
        try:
            directed, undirected, dt = rlcd_edges(name, X, obs, fresh=True, **cfg)
            g = build(directed + undirected, obs)
            adq, _, _ = v_state(g, X, oi)
            row = dict(cfg=tag, n_latents=len(g.latents), n_edges=len(g.edges),
                       V=round(vtotal(adq), 2), seconds=round(dt, 1),
                       latents={L: sorted(c for c in g.children(L)) for L in g.latents})
            print(f"[{name}] {tag}: latents={row['n_latents']} edges={row['n_edges']} "
                  f"V={row['V']} ({row['seconds']}s)", flush=True)
        except Exception as e:
            row = dict(cfg=tag, error=f"{type(e).__name__}: {e}")
            print(f"[{name}] {tag}: FAILED {row['error']}", flush=True)
        rows.append(row)
    json.dump(rows, open(os.path.join(OUT, f"{name}_sweep.json"), "w"), indent=1)
    print(f"[{name}] saved outputs/{name}_sweep.json", flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    names = [w.strip() for w in os.environ.get("DATASET", "rse,cfcs,himi").split(",")]
    for nm in names:
        try:
            if os.environ.get("SWEEP", "0") == "1":
                run_sweep(nm)
            else:
                run_dataset(nm, fresh=os.environ.get("FRESH", "0") == "1")
        except Exception as e:
            print(f"[{nm}] FAILED: {type(e).__name__}: {e}", flush=True)
    print("[discovery done]", flush=True)
