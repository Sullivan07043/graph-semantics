"""Evidence extraction for the LLM-steering plugin. Deterministic, no API.

For each masked variable (or latent) the evidence package is:
  phrases      the decoded words already saved in the Stage A records (core arm)
  graph_lines  rendered neighbor lines from the graph plus VISIBLE labels only

Questionnaire graphs come from the loaders (given) or the gpurlcd jsons (discovered);
robot graphs come from the summary jsons. Robot edge lists truncate to the TOP_K
strongest edges by |weight| with the omission declared. Nothing here reads the true
label of the masked variable itself.
"""
import json
import os
import re
import sys
from collections import defaultdict

import os
import sys
_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PKG)
sys.path.insert(0, os.path.join(_PKG, "plugin"))
import config
sys.path.insert(0, config.V6)
sys.path.insert(0, config.DISC)
DISC = config.DISC
ROOT = config.GS

TOP_K = 20
LAT_RE = re.compile(r"^L\d+$")


def _loaders():
    import pool
    import pool_ext
    import testbeds
    return {**testbeds.LOADERS, **pool.LOADERS, **pool_ext.LOADERS}


def load_questionnaire(base, source):
    """-> (graph, labels, lat_names). source: given | disc."""
    import graph as G
    ds = _loaders()[base]()
    labels, g_pub = ds["labels"], ds["graph"]
    if source == "given":
        return g_pub, labels, ds.get("latent_gt", {})
    d = json.load(open(os.path.join(DISC, "outputs", f"{base}_gpurlcd.json")))
    edges = [tuple(e) for e in d["rlcd_directed"]] + \
            [tuple(sorted(e)) for e in d["rlcd_undirected"]]
    nodes = {x for e in edges for x in e}
    lats = sorted((n for n in nodes if LAT_RE.match(n)), key=lambda s: int(s[1:]))
    return G.Graph(lats, list(g_pub.observed), edges), labels, {}


def q_item_lines(g, var, labels, hidden, lat_names):
    """T1: visible neighbourhood of one masked item (same rules as the naming
    baseline: sibling items through shared latent parents, direct edges)."""
    lines, seen = [], set()
    for p in g.parents(var):
        if g.is_latent(p):
            sibs = [labels[c] for c in g.children(p)
                    if not g.is_latent(c) and c != var and c not in hidden]
            sibs = [s for s in sibs if s not in seen][:6]
            seen.update(sibs)
            head = (f'a latent factor named "{lat_names[p]}"' if p in lat_names
                    else "an unnamed latent factor")
            if sibs:
                joined = "; ".join(f'"{s}"' for s in sibs)
                lines.append(f"- CAUSE: {head} also causes: {joined}")
            elif p in lat_names:
                lines.append(f"- CAUSE: {head}")
        elif p not in hidden:
            lines.append(f'- CAUSE: "{labels[p]}" influences it')
    for c in g.children(var):
        if not g.is_latent(c) and c not in hidden:
            lines.append(f'- EFFECT: it influences "{labels[c]}"')
    return lines


def q_latent_lines(g, lat, labels):
    """T2: the latent's observed children (all item labels are visible in T2)."""
    kids = [c for c in g.children(lat) if not g.is_latent(c)]
    return [f'- EFFECT: it causes the item "{labels[c]}"' for c in kids[:TOP_K]] + \
        ([f"  (and {len(kids) - TOP_K} more items omitted)"] if len(kids) > TOP_K else [])


def load_robot(base, graph_file):
    import importlib.util
    p3 = config.T3P
    spec = importlib.util.spec_from_file_location("pe3", os.path.join(p3, "pool_ext.py"))
    pe3 = importlib.util.module_from_spec(spec)
    sys.path.insert(0, p3)
    spec.loader.exec_module(pe3)
    ds = pe3.LOADERS[base]()
    g = json.load(open(os.path.join(config.T3, "outputs", graph_file)))
    W, T = {}, {}
    for a, b in g["rlcd_directed"]:
        k = f"{a}->{b}"
        W[(a, b)] = float(g.get("signs", {}).get(k) or 0.0)
        T[(a, b)] = g.get("edge_types", {}).get(k, "lag")
    return ds["labels"], W, T


def r_channel_lines(var, labels, W, T, hidden):
    """T3: typed weighted neighbor lines, top TOP_K by |weight|, truncation declared."""
    rows = []
    for (a, b), w in W.items():
        if a == var and b not in hidden:
            rows.append(("EFFECT", b, w, T[(a, b)]))
        elif b == var and a not in hidden:
            rows.append(("CAUSE", a, w, T[(a, b)]))
    rows.sort(key=lambda r: -abs(r[2]))
    out = []
    for d, n, w, t in rows[:TOP_K]:
        s = "positive" if w >= 0 else "negative"
        out.append(f'- {d} ({t}, weight {abs(w):.3f}, {s}): "{labels[n]}"')
    if len(rows) > TOP_K:
        out.append(f"  (and {len(rows) - TOP_K} weaker edges omitted)")
    return out
