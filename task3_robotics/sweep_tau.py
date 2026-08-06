"""Sweep the Jacobian threshold against the known truth.

The Jacobian route returns a continuous magnitude per candidate edge, so the edge set is whatever
a cut of that magnitude says it is. The first full run at tau .05 recalled 82 percent of the true
edges but proposed roughly four times too many, so the cut, not the model, is what needs setting.

The threshold is swept on the SAVED Jacobian, so the model is trained once. Scoring is against the
simulator's physics, which is the whole reason for working in this domain: the choice can be made
on evidence rather than on how the graph looks.

Reported per cut: precision, recall, F1, the count of edges the physics forbids, and recall of the
forward-kinematics layer, which is the layer with an analytic ground truth to compare weights to.

Env: JNPZ=<jacobian npz> TRUE=<true graph json> TAUS=0.02,...
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
JNPZ = os.environ.get("JNPZ", os.path.join(HERE, "outputs", "lift_body_jacobian.npz"))
TRUE = os.environ.get("TRUE", os.path.join(HERE, "outputs", "lift_body_true_graph.json"))
TAUS = [float(x) for x in os.environ.get(
    "TAUS", "0.02,0.05,0.10,0.15,0.20,0.30,0.40,0.50,0.60,0.70").split(",")]

TIER = {"robot0_joint_pos": 0, "robot0_joint_vel": 0, "robot0_eef_pos": 1,
        "robot0_eef_quat": 1, "robot0_gripper_qpos": 1, "robot0_gripper_qvel": 1}
FORBIDDEN = [("robot0_joint_pos", "robot0_joint_pos"), ("robot0_eef", "robot0_joint"),
             ("robot0_gripper", "robot0_joint"), ("robot0_joint", "robot0_gripper")]


def tier(col):
    return TIER.get(col.split("@")[0].rsplit(".", 1)[0], 1)


def fam(col):
    b = col.split("@")[0]
    for f in ("robot0_joint_pos", "robot0_joint_vel", "robot0_eef_pos", "robot0_eef_quat",
              "robot0_gripper_qpos", "robot0_gripper_qvel", "action"):
        if b.startswith(f):
            return f
    return "other"


def is_forbidden(a, b):
    fa, fb = fam(a), fam(b)
    for pa, pb in FORBIDDEN:
        if fa.startswith(pa) and fb.startswith(pb) and fa != fb:
            return True
        if pa == pb and fa == fb == pa:
            return True
    return False


def main():
    d = np.load(JNPZ, allow_pickle=True)
    J, src, targets = d["J"], [str(x) for x in d["src"]], [str(x) for x in d["targets"]]
    T = json.load(open(TRUE))
    true_edges = {(e["from"], e["to"]) for e in T["edges"]}
    fk = {(e["from"], e["to"]) for e in T["edges"] if e["kind"].startswith("forward_kinematics")}
    tier0_src = {s for s in src if s.endswith("@t") and tier(s) == 0}

    print(f"truth {len(true_edges)} edges; Jacobian {J.shape[0]} targets x {J.shape[1]} sources\n")
    print(f"{'tau':>6} {'edges':>7} {'prec':>7} {'recall':>7} {'F1':>7} "
          f"{'forbidden':>10} {'fwd-kin':>9}")
    rows = []
    for tau in TAUS:
        got = set()
        for ci, cj in enumerate(targets):
            m = J[ci].max()
            if m <= 0:
                continue
            for pi, sj in enumerate(src):
                if sj == cj:
                    continue
                if tier(cj) == 0 and sj in tier0_src:
                    continue
                if J[ci, pi] >= tau * m:
                    got.add((sj, cj))
        tp = got & true_edges
        prec = len(tp) / max(len(got), 1)
        rec = len(tp) / max(len(true_edges), 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        bad = sum(1 for a, b in got - true_edges if is_forbidden(a, b))
        print(f"{tau:6.2f} {len(got):7d} {prec:7.3f} {rec:7.3f} {f1:7.3f} "
              f"{bad:10d} {len(got & fk):4d}/{len(fk)}")
        rows.append({"tau": tau, "edges": len(got), "precision": prec, "recall": rec,
                     "f1": f1, "forbidden": bad, "fk_recall": len(got & fk)})
    best = max(rows, key=lambda r: r["f1"])
    print(f"\nbest F1 at tau {best['tau']}: {best['edges']} edges, precision "
          f"{best['precision']:.3f}, recall {best['recall']:.3f}, {best['forbidden']} forbidden")
    json.dump(rows, open(os.path.join(HERE, "outputs", "tau_sweep.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
