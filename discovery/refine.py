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
    return adq, props


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
    prev_edges = None
    for it in range(10):
        adq, props = v_and_proposals(g, X, oi)
        vt = adq["V_marginal"] + adq["V_conditional"]
        print(f"[{name}] iter {it}: V_total={vt:.2f} (marg {adq['V_marginal']:.2f} + "
              f"cond {adq['V_conditional']:.2f}), latents={g.latents}", flush=True)
        if traj and vt >= traj[-1]:
            edges = prev_edges                        # last edit did not reduce V: revert
            added.pop()
            print(f"[{name}] stop: V not decreasing — last edit reverted", flush=True)
            break
        traj.append(round(vt, 3))
        if not props:
            print(f"[{name}] stop: no proposals", flush=True)
            break
        comp = props[0]["nodes"]
        comp_set = set(comp)
        roots = [n for n in comp if not any(p in comp_set for p in g.parents(n))]
        new_lat = f"L{1 + max([int(L[1:]) for L in g.latents], default=0)}"
        prev_edges = list(edges)
        edges = edges + [(new_lat, r) for r in roots]
        added.append({"latent": new_lat, "over_roots": roots, "mass": props[0]["mass"]})
        print(f"[{name}]   apply: {new_lat} -> {roots}", flush=True)
        g = build(edges, obs)

    out = dict(dataset=name, directed=[list(e) for e in edges], undirected=[],
               added_latents=added, V_trajectory=traj)
    json.dump(out, open(os.path.join(HERE, "outputs", f"{name}_refined.json"), "w"), indent=1)
    print(f"[{name}] final: latents={build(edges, obs).latents}, V_traj={traj}", flush=True)
    print(f"[saved outputs/{name}_refined.json]", flush=True)


if __name__ == "__main__":
    main()
