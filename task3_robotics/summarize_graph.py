"""Collapse the BOSS lag graph into a summary graph over the 32 channels.

Translation targets are CHANNELS, not their time copies: not knowing what channel 3 measures means
not knowing it at any time step. So the lag-unrolled graph folds: an edge a@t-1 -> b@t or
a@t -> b@t becomes a -> b, self-edges (a onto its own past) drop out, and duplicates keep the
weight of largest magnitude, sign included.

Env: DISC=<discovered json> ROUTE=boss OUT=<summary json>
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DISC = os.environ.get("DISC", os.path.join(HERE, "outputs", "lift_body_discovered.json"))
ROUTE = os.environ.get("ROUTE", "boss")
OUT = os.environ.get("OUT", os.path.join(HERE, "outputs", "lift_body_summary.json"))


def main():
    edges = json.load(open(DISC))["routes"][ROUTE]
    best, btype = {}, {}
    for e in edges:
        a, b = e["from"].split("@")[0], e["to"].split("@")[0]
        if a == b:
            continue
        k = (a, b)
        if k not in best or abs(e["weight"]) > abs(best[k]):
            best[k] = e["weight"]
            btype[k] = "contemp" if e["from"].endswith("@t") else "lag"
    nodes = sorted({x for k in best for x in k})
    out = {
        "dataset": "liftbody",
        "rlcd_directed": [[a, b] for a, b in sorted(best)],
        "rlcd_undirected": [],
        "signs": {f"{a}->{b}": w for (a, b), w in best.items()},
        "edge_types": {f"{a}->{b}": btype[(a, b)] for (a, b) in best},
        "params": {"route": ROUTE, "source": os.path.basename(DISC),
                   "note": "lag graph folded to channel level; self-edges dropped; "
                           "duplicate edges keep the largest-magnitude weight"},
    }
    json.dump(out, open(OUT, "w"), indent=1)
    acts = [n for n in nodes if n.startswith("action")]
    print(f"[summary] {len(best)} edges over {len(nodes)} channels "
          f"({len(acts)} action channels appear)")
    print(f"[summary] saved {OUT}")


if __name__ == "__main__":
    main()
