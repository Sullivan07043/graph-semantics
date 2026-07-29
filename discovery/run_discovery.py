"""Discovery driver, minimal first version: RLCD on one dataset, print discovered structure
next to the published key. Usage: DATASET=himi python run_discovery.py"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
V6 = os.path.join(os.path.dirname(HERE), "v6")
sys.path.insert(0, V6)

import numpy as np                                                    # noqa: E402
import testbeds                                                       # noqa: E402
import pool                                                           # noqa: E402
from causallearn.search.HiddenCausal.RLCD.RLCD_alg import RLCD        # noqa: E402


def edges_from_cg(cg):
    """-> (directed [(a,b)], undirected [(a,b)]) from causal-learn GeneralGraph."""
    G = cg.G
    nodes = G.get_nodes()
    names = [n.get_name() for n in nodes]
    A = G.graph
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
                undirected.append((names[i], names[j]))
    return directed, undirected


def main():
    name = os.environ.get("DATASET", "himi")
    ds = {**testbeds.LOADERS, **pool.LOADERS}[name]()
    g_pub, X = ds["graph"], ds["X"]
    obs = list(g_pub.observed)
    print(f"[{name}] n={X.shape[0]}, m={len(obs)} observed; published: "
          f"{len(g_pub.latents)} latents, {len(g_pub.edges)} edges", flush=True)
    pub_assign = {L: sorted(c for c in g_pub.children(L) if not g_pub.is_latent(c))
                  for L in g_pub.latents}
    for L, ch in pub_assign.items():
        print(f"  published {L}: {ch}", flush=True)

    t0 = time.time()
    cg = RLCD(np.asarray(X, float), node_names=obs)
    dt = time.time() - t0
    directed, undirected = edges_from_cg(cg)
    lat = sorted({a for a, b in directed + undirected if a.startswith("L")}
                 | {b for a, b in directed + undirected if b.startswith("L")})
    print(f"\n[{name}] RLCD done in {dt:.1f}s; discovered latents: {lat}", flush=True)
    print(f"  directed edges ({len(directed)}):", flush=True)
    for a, b in directed:
        print(f"    {a} -> {b}", flush=True)
    if undirected:
        print(f"  undirected edges ({len(undirected)}):", flush=True)
        for a, b in undirected:
            print(f"    {a} -- {b}", flush=True)

    out = os.path.join(HERE, "outputs")
    os.makedirs(out, exist_ok=True)
    json.dump({"dataset": name, "seconds": dt, "directed": directed,
               "undirected": undirected, "published": pub_assign},
              open(os.path.join(out, f"{name}_rlcd.json"), "w"), indent=1)
    print(f"[saved outputs/{name}_rlcd.json]", flush=True)


if __name__ == "__main__":
    main()
