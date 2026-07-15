"""Assemble the final day tables from the saved records/logs into outputs/final_summary.json + a
printed report. Reads: final_dev_task1/2(+2b), final_heldout_task1/2, gnn_c_*_dev, gnn_v5_*."""
import json, os, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
HELDOUT = ["hexaco", "riasec", "kims"]                   # final split (user 2026-07-14 evening)


def task1_table(path):
    d = json.load(open(os.path.join(OUT, path)))
    return d["summary"]


def task2_summary(paths):
    s = {}
    for p in paths:
        f = os.path.join(OUT, p)
        if os.path.exists(f):
            s.update(json.load(open(f))["summary"])
    return s


def main():
    t1_dev = task1_table("final_dev_task1.json")
    t1_ho = task1_table("final_heldout_task1.json")
    t2 = task2_summary(["final_dev_task2.json", "final_dev_task2b.json", "final_heldout_task2.json"])
    gnn_ab = {}
    for a in ["none", "indep", "res", "coll", "dep", "mb"]:
        f = os.path.join(OUT, f"gnn_c_{a}_dev.json")
        if os.path.exists(f):
            r = json.load(open(f))
            gnn_ab[a] = {n: v["match"] for n, v in r.items()}
    v5 = {}
    for grp in ["dev", "heldout"]:
        f = os.path.join(OUT, f"gnn_v5_{grp}.json")
        if os.path.exists(f):
            v5[grp] = json.load(open(f))

    print("== Task 1 (judge/match), frozen v2 method ==")
    for grp, tab in [("DEV", t1_dev), ("HELDOUT-run", t1_ho)]:
        for n, arms in tab.items():
            row = "  ".join(f"{a}:{v['judge'] if v['judge'] is not None else -1:.3f}/{v['match']:.3f}"
                            for a, v in arms.items())
            tag = "HO " if n in HELDOUT else ("dev" if grp == "DEV" else "dev*")
            print(f"  [{tag}] {n:10s} {row}")
    print("== Task 2 latent judge-ACC (core vs llm_name) ==")
    for n, v in t2.items():
        tag = "HO " if n in HELDOUT else "dev"
        c = v.get("core"); l = v.get("llm_name")
        print(f"  [{tag}] {n:10s} core={c if c is not None else float('nan'):.3f} "
              f"llm={l if l is not None else float('nan'):.3f}")
    print("== GNN constraint ablation (dev match means) ==")
    for a, r in gnn_ab.items():
        print(f"  {a:6s} {np.mean(list(r.values())):.3f}")
    if v5:
        print("== GNN v5 broad-pool ==")
        for grp, r in v5.items():
            print(f"  {grp}: " + "  ".join(f"{n}:{v['match']:.2f}" for n, v in r.items()))
    json.dump({"task1_dev": t1_dev, "task1_heldout_run": t1_ho, "task2": t2,
               "gnn_constraint_ablation_match": gnn_ab, "gnn_v5": v5, "heldout_split": HELDOUT},
              open(os.path.join(OUT, "final_summary.json"), "w"), indent=1)
    print("[saved outputs/final_summary.json]")


if __name__ == "__main__":
    main()
