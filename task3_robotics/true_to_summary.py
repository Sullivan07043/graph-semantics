"""Fold a per-robot physics truth graph (body_<r>_true.json) into the channel-level summary
format the pipeline loader consumes (same shape as *_boss_summary.json, edge_types included:
kinematics kinds are contemporaneous, everything else is lag).

Env: TRUE=<body_<r>_true.json> OUT=<summary json> DATASET=<loader name>
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
TRUE = os.environ["TRUE"]
OUT = os.environ["OUT"]
DATASET = os.environ["DATASET"]

CONTEMP_KINDS = {"forward_kinematics", "forward_kinematics_rot"}


def main():
    T = json.load(open(TRUE))
    best, btype = {}, {}
    for e in T["edges"]:
        a, b = e["from"].split("@")[0], e["to"].split("@")[0]
        if a == b:
            continue
        w = e.get("weight")
        w = 1.0 if w is None else float(w)
        k = (a, b)
        if k not in best or abs(w) > abs(best[k]):
            best[k] = w
            btype[k] = "contemp" if e["kind"] in CONTEMP_KINDS else "lag"
    out = {
        "dataset": DATASET,
        "rlcd_directed": [[a, b] for a, b in sorted(best)],
        "rlcd_undirected": [],
        "signs": {f"{a}->{b}": w for (a, b), w in best.items()},
        "edge_types": {f"{a}->{b}": btype[(a, b)] for (a, b) in best},
        "params": {"source": os.path.basename(TRUE),
                   "note": "physics truth folded to channel level; self-edges dropped; "
                           "kinematics kinds marked contemporaneous"},
    }
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"[{DATASET}] {len(best)} channel edges -> {OUT}")


if __name__ == "__main__":
    main()
