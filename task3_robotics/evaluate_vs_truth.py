"""Score each discovery route against the known physics of the robot body.

This is the measurement the questionnaire domain cannot supply. There the reference is a published
scoring key, which we have measured to be imperfect, so a poor downstream number cannot be
attributed to discovery or to translation. Here the reference is the simulator's own physics.

Reported per route:
  precision, recall, F1 over the edge set
  a breakdown by edge kind, since the kinds differ in difficulty: integration edges are diagonal
  and easy, the operational-space control edges are dense and configuration dependent, the
  inertial coupling is sparse and is the interesting case
  the false positives that matter, meaning edges the physics says cannot exist
  weight agreement on the edges where the truth is a number: the integration coefficient is the
  control timestep, and the forward-kinematics coefficients are the analytic Jacobian

Env: TRUE=<true graph json> DISC=<discovered json>
"""
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TRUE = os.environ.get("TRUE", os.path.join(HERE, "outputs", "lift_body_true_graph.json"))
DISC = os.environ.get("DISC", os.path.join(HERE, "outputs", "lift_body_discovered.json"))

# edges the physics forbids; a route that proposes these is making a nameable mistake
FORBIDDEN = [
    ("joint_pos", "joint_pos", "positions do not act on each other directly"),
    ("eef", "joint", "the end-effector is a function of the joints, not a cause"),
    ("gripper", "joint", "the finger chain is actuated separately"),
    ("joint", "gripper", "the finger chain is actuated separately"),
]


def base(node):
    return node.split("@")[0]


def family(node):
    b = base(node)
    for f in ("robot0_joint_pos", "robot0_joint_vel", "robot0_eef_pos", "robot0_eef_quat",
              "robot0_gripper_qpos", "robot0_gripper_qvel", "action"):
        if b.startswith(f):
            return f
    return "other"


def forbidden_hit(a, b):
    fa, fb = family(a), family(b)
    for pa, pb, why in FORBIDDEN:
        if pa in fa and pb in fb and not (pa == "joint" and fa == fb):
            return why
    return None


def main():
    T = json.load(open(TRUE))
    D = json.load(open(DISC))
    true_edges = {(e["from"], e["to"]) for e in T["edges"]}
    kind_of = {(e["from"], e["to"]): e["kind"] for e in T["edges"]}
    weight_of = {(e["from"], e["to"]): e["weight"] for e in T["edges"]}
    by_kind = defaultdict(set)
    for e in T["edges"]:
        by_kind[e["kind"]].add((e["from"], e["to"]))

    print(f"truth: {len(true_edges)} edges over {T['n_nodes']} nodes, "
          f"control timestep {T['control_timestep']}\n")

    for route, edges in D["routes"].items():
        got = {(e["from"], e["to"]) for e in edges}
        w_got = {(e["from"], e["to"]): e["weight"] for e in edges}
        tp = got & true_edges
        prec = len(tp) / max(len(got), 1)
        rec = len(tp) / max(len(true_edges), 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        print(f"=== {route}: {len(got)} edges | precision {prec:.3f} | recall {rec:.3f} "
              f"| F1 {f1:.3f}")
        for k in sorted(by_kind):
            s = by_kind[k]
            print(f"      {k:24s} recall {len(got & s)}/{len(s)}")

        bad = defaultdict(int)
        for a, b in got - true_edges:
            why = forbidden_hit(a, b)
            if why:
                bad[why] += 1
        if bad:
            print("      false positives the physics forbids:")
            for why, c in sorted(bad.items(), key=lambda x: -x[1]):
                print(f"        {c:4d}  {why}")
        else:
            print("      no forbidden edges proposed")

        # weight agreement, split by kind. Correlation is only meaningful where the truth
        # actually varies; the integration coefficient is a single constant, so it gets an
        # absolute error instead.
        num = [(e, weight_of[e], w_got[e]) for e in tp if weight_of.get(e) is not None]
        for kind in sorted({kind_of[e] for e, _, _ in num}):
            sub = [x for x in num if kind_of[x[0]] == kind]
            t = np.array([abs(x[1]) for x in sub])
            g = np.array([abs(x[2]) for x in sub])
            if t.std() < 1e-12:
                print(f"      weights, {kind} ({len(sub)} edges): truth {t[0]:.4f}, "
                      f"recovered {g.mean():.4f}, mean abs error {np.abs(g - t).mean():.4f}")
            elif len(sub) > 2:
                r = np.corrcoef(t, g)[0, 1]
                print(f"      weights, {kind} ({len(sub)} edges): |corr| with the analytic "
                      f"value {abs(r):.3f}")
        print()


if __name__ == "__main__":
    main()
