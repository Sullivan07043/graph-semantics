"""Ground-truth body graph for ANY robosuite robot, verified per robot.

Generalizes true_graph.py (the Panda original, kept as the committed reference). Everything that
was Panda-specific is now derived, not assumed:

  joint count, control timestep      read from the live simulator
  positional Jacobian                mujoco mj_jacBody, positional part; zero entries stay absent
  rotational Jacobian                mj_jacBody rotational part, so quaternion edges use the true
                                     sparsity instead of "all joints" as the Panda script did
  coaxial velocity coupling          MEASURED on that robot's own collected data with the same
                                     procedure as on Panda: partial correlation of vel_i(t) with
                                     vel_j(t-1), controlling for vel_i(t-1), pos_i(t-1) and the
                                     full action; pairs above PC_THR enter as inertial_coupling
  action and gripper widths          read from the collected npz

Env: ROBOT=Sawyer NPZ=<steps npz> PC_THR=0.15 OUT=<json>
Runs in the robotics venv with MUJOCO_GL=egl.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROBOT = os.environ.get("ROBOT", "Sawyer")
NPZ = os.environ.get("NPZ", os.path.join(HERE, "outputs", f"body_{ROBOT.lower()}_steps.npz"))
PC_THR = float(os.environ.get("PC_THR", 0.15))
OUT = os.environ.get("OUT", os.path.join(HERE, "outputs", f"body_{ROBOT.lower()}_true.json"))
AXIS = ["x", "y", "z", "w"]


def resid(y, X):
    X = np.c_[X, np.ones(len(X))]
    return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]


def measure_coupling(npz, nj):
    """Cross-joint lagged partial dependence at the velocity level, on this robot's data."""
    d = np.load(npz, allow_pickle=True)
    X, cols = np.asarray(d["X"], float), [str(c) for c in d["names"]]

    def idx(name):
        return cols.index(name)

    pv = [idx(f"robot0_joint_vel.{i}@t-1") for i in range(1, nj + 1)]
    pp = [idx(f"robot0_joint_pos.{i}@t-1") for i in range(1, nj + 1)]
    cv = [idx(f"robot0_joint_vel.{i}@t") for i in range(1, nj + 1)]
    act = [i for i, c in enumerate(cols) if c.startswith("action.")]
    S = X[np.random.default_rng(0).choice(len(X), min(60000, len(X)), replace=False)]
    pairs = []
    for i in range(nj):
        C = np.c_[S[:, pv[i]], S[:, pp[i]], S[:, act]]
        r = resid(S[:, cv[i]], C)
        for j in range(nj):
            if i == j:
                continue
            rj = resid(S[:, pv[j]], C)
            pc = abs(np.corrcoef(r, rj)[0, 1])
            if pc > PC_THR:
                pairs.append((j + 1, i + 1, round(float(pc), 3)))   # vel_j(t-1) -> vel_i(t)
    return pairs


def main():
    import mujoco
    import robosuite as suite
    env = suite.make("Lift", robots=ROBOT, has_renderer=False, has_offscreen_renderer=False,
                     use_camera_obs=False, control_freq=20)
    env.reset()
    dt = float(env.control_timestep)
    r = env.robots[0]
    nj = len(r._ref_joint_vel_indexes)
    m, dd = env.sim.model._model, env.sim.data._data
    eef = next((i for i in range(m.nbody)
                if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or "").endswith("_eef")), None)
    assert eef is not None, "no *_eef body found"
    jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    mujoco.mj_jacBody(m, dd, jacp, jacr, eef)
    Jp = jacp[:, r._ref_joint_vel_indexes]
    Jr = jacr[:, r._ref_joint_vel_indexes]

    d = np.load(NPZ, allow_pickle=True)
    cols = [str(c) for c in d["names"]]
    gq = sorted({c.split("@")[0] for c in cols if c.startswith("robot0_gripper_qpos.")})
    gv = sorted({c.split("@")[0] for c in cols if c.startswith("robot0_gripper_qvel.")})
    act_w = len({c.split("@")[0] for c in cols if c.startswith("action.")})
    quat = sorted({c.split("@")[0] for c in cols if c.startswith("robot0_eef_quat.")
                   and not c.startswith("robot0_eef_quat_site")})

    coupling = measure_coupling(NPZ, nj)

    jp = [f"robot0_joint_pos.{i}" for i in range(1, nj + 1)]
    jv = [f"robot0_joint_vel.{i}" for i in range(1, nj + 1)]
    ep = [f"robot0_eef_pos.{a}" for a in AXIS[:3]]
    acts = [f"action.{i}" for i in range(1, act_w + 1)]

    E = []
    for a in acts[:min(6, act_w - 1)]:
        for v in jv:
            E.append((f"{a}@t-1", f"{v}@t", "control", None))
    for i, v in enumerate(jv):
        E.append((f"{v}@t-1", f"{jp[i]}@t", "integration", dt))
        E.append((f"{v}@t-1", f"{v}@t", "dynamics_self", None))
    for j_from, i_to, pc in coupling:
        E.append((f"robot0_joint_vel.{j_from}@t-1", f"robot0_joint_vel.{i_to}@t",
                  "inertial_coupling", pc))
    for row, coord in enumerate(ep):
        for c, j in enumerate(jp):
            if abs(Jp[row, c]) > 1e-9:
                E.append((f"{j}@t", f"{coord}@t", "forward_kinematics", float(Jp[row, c])))
    for q in quat:
        for c, j in enumerate(jp):
            if np.abs(Jr[:, c]).max() > 1e-9:
                E.append((f"{j}@t", f"{q}@t", "forward_kinematics_rot", None))
    for g in gv:
        E.append((f"{acts[-1]}@t-1", f"{g}@t", "control", None))
    for i, g in enumerate(gv):
        if i < len(gq):
            E.append((f"{g}@t-1", f"{gq[i]}@t", "integration", dt))
        E.append((f"{g}@t-1", f"{g}@t", "dynamics_self", None))

    kinds = {}
    for _, _, k, _ in E:
        kinds[k] = kinds.get(k, 0) + 1
    nodes = sorted({x for e in E for x in e[:2]})
    json.dump({
        "robot": ROBOT, "scope": "robot body only", "control_timestep": dt, "n_joints": nj,
        "n_nodes": len(nodes), "n_edges": len(E), "edges_by_kind": kinds,
        "edges": [{"from": a, "to": b, "kind": k, "weight": w} for a, b, k, w in E],
        "positional_jacobian": Jp.tolist(),
        "measured_coupling": coupling, "pc_threshold": PC_THR,
    }, open(OUT, "w"), indent=1)
    print(f"[{ROBOT}] {nj} joints, {len(nodes)} nodes, {len(E)} edges | "
          f"coupling pairs measured: {len(coupling)} | "
          f"zero columns in positional Jacobian: "
          f"{int((np.abs(Jp).max(0) < 1e-9).sum())}")
    for k, v in sorted(kinds.items()):
        print(f"    {k:22s} {v}")
    print(f"[{ROBOT}] saved {OUT}")


if __name__ == "__main__":
    main()
