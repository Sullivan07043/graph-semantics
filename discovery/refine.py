"""Refine a discovered structure: V-driven expansion (the user's check-and-expand loop).

Loop: compute V(G_t, X) on the full CI table -> take the TOP repair proposal -> add ONE new
latent as parent of the proposal component's ROOTS (nodes with no parent inside the
component; latent members enter as children = hierarchy, observed members only if nothing
above them is in the component) -> recompute. Stop when total V stops decreasing, no
proposals remain, or 10 iterations (runaway guard, far above observed need).

New latents continue RLCD's L-numbering (L4, L5, ...). Output: outputs/<ds>_refined.json
(same schema as <ds>_rlcd.json + per-iteration V trajectory + added-latent provenance).
Usage: DATASET=rse python refine.py
"""
import json
import os
import re
import sys

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

LAT_RE = re.compile(r"^L\d+$")


def build(edges, obs_all):
    nodes = {x for e in edges for x in e}
    lats = sorted((n for n in nodes if LAT_RE.match(n)), key=lambda s: int(s[1:]))
    return G.Graph(lats, obs_all, edges)


def v_and_proposals(g, X, oi):
    W, score = g.estimate_weights(X, oi)
    W, score = LC.sign_fix(g, W, score)
    mats = NL.matrices(g, X, oi, score, None)
    ci_full = TF.ci_table(g, X, oi, score, nl=mats, mode="full")
    adq = adequacy.compute(g, X, oi, score, ci=ci_full)
    props = adequacy.propose_repairs(g, ci_full)
    return adq, props, mats


def next_lat(g):
    return f"L{1 + max([int(L[1:]) for L in g.latents], default=0)}"


def cand_add(edges, g, props):
    """ADD: one new latent over the top violating component's roots."""
    if not props:
        return None
    comp = props[0]["nodes"]
    comp_set = set(comp)
    roots = [n for n in comp if not any(p in comp_set for p in g.parents(n))]
    L = next_lat(g)
    return edges + [(L, r) for r in roots], {"op": "add", "latent": L, "over": roots}


def cand_merge(edges, g, mats):
    """MERGE: collapse the most score-dependent latent pair into one latent."""
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


def main():
    name = os.environ.get("DATASET", "rse")
    ds = {**testbeds.LOADERS, **pool.LOADERS}[name]()
    g_pub, X = ds["graph"], ds["X"]
    obs = list(g_pub.observed)
    oi = {o: k for k, o in enumerate(obs)}
    d = json.load(open(os.path.join(HERE, "outputs", f"{name}_rlcd.json")))
    edges = [tuple(e) for e in d["directed"]] + [tuple(sorted(e)) for e in d["undirected"]]

    g = build(edges, obs)
    added, traj = [], []
    for it in range(10):
        adq, props, mats = v_and_proposals(g, X, oi)
        vt = adq["V_marginal"] + adq["V_conditional"]
        traj.append(round(vt, 3))
        print(f"[{name}] iter {it}: V_total={vt:.2f} (marg {adq['V_marginal']:.2f} + "
              f"cond {adq['V_conditional']:.2f}), latents={g.latents}", flush=True)
        cands = [c for c in (cand_add(edges, g, props), cand_merge(edges, g, mats)) if c]
        best_edges, best_edit, best_v = None, None, vt
        for cand_edges, edit in cands:
            g_c = build(cand_edges, obs)
            adq_c, _, _ = v_and_proposals(g_c, X, oi)
            v_c = adq_c["V_marginal"] + adq_c["V_conditional"]
            print(f"[{name}]   candidate {edit['op']}: "
                  f"{edit.get('latent', edit.get('as'))} V={v_c:.2f}", flush=True)
            if v_c < best_v:
                best_edges, best_edit, best_v = cand_edges, edit, v_c
        if best_edit is None:
            print(f"[{name}] stop: no candidate reduces V", flush=True)
            break
        edges = best_edges
        added.append(best_edit)
        print(f"[{name}]   apply: {best_edit}", flush=True)
        g = build(edges, obs)

    out = dict(dataset=name, directed=[list(e) for e in edges], undirected=[],
               added_latents=added, V_trajectory=traj)
    json.dump(out, open(os.path.join(HERE, "outputs", f"{name}_refined.json"), "w"), indent=1)
    print(f"[{name}] final: latents={build(edges, obs).latents}, V_traj={traj}", flush=True)
    print(f"[saved outputs/{name}_refined.json]", flush=True)


if __name__ == "__main__":
    main()
