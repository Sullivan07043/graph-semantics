"""The ground-truth causal graph of the robot body, written from known physics and verified.

This is the reason robotics is worth doing for this project. In the questionnaire domain there is
no true graph: the published scoring key is the closest thing, and it is imperfect (40/42 on DASS,
16/21 on WVS, 20/38 on NHANES). So when a discovered graph underperforms we cannot say whether the
discovery or the translation is at fault. Here the generative process is a simulator we can query,
so the truth is available and the two failures can be separated.

SCOPE: the robot body only. The cube is excluded, because the end-effector to object edge exists
only while they are in contact, and a single static graph cannot express a state-dependent edge.
That case is deferred, not solved.

VARIABLES, one lag. 32 channels at t-1 and the same 32 at t:
  robot0_joint_pos.1..7        arm joint angles
  robot0_joint_vel.1..7        arm joint angular velocities
  robot0_eef_pos.x,y,z         end-effector position
  robot0_eef_quat.x,y,z,w      end-effector orientation
  robot0_gripper_qpos.1,2      finger openings
  robot0_gripper_qvel.1,2      finger opening speeds
  action.1..7                  6 end-effector deltas plus 1 gripper command

EDGES, with the quantity that verifies each one:

  action(t-1) -> joint_vel(t)              dense, and state dependent. The default Panda
                                           controller is operational-space: it maps an
                                           end-effector-space command through the full
                                           manipulator Jacobian and mass matrix, so every action
                                           component can reach every joint, with weights that
                                           depend on the current configuration.
  joint_vel(t-1) -> joint_pos(t)           diagonal. Coefficient is the control timestep,
                                           measured at 0.05 s.
  joint_vel(t-1) -> joint_vel(t)           diagonal, plus sparse off-diagonal coupling within the
                                           coaxial joint group. Measured partial correlations,
                                           controlling for own past and the action: 1-3 .231,
                                           5-7 .233, 3-7 .167, while 2, 4 and 6 stay below .05.
                                           Joints 1,3,5,7 rotate about the link axis and share
                                           inertia terms; 2,4,6 are pitch joints.
  joint_pos(t) -> eef_pos(t)               contemporaneous, analytic. Equal to the manipulator
                                           Jacobian, read from MuJoCo. Its joint-7 column is
                                           exactly zero: wrist roll does not move the end-effector
                                           position.
  joint_pos(t) -> eef_quat(t)              contemporaneous, analytic, the rotational Jacobian.
  action.7(t-1) -> gripper_qvel(t)         the gripper command drives the fingers only.
  gripper_qvel(t-1) -> gripper_qpos(t)     diagonal, coefficient is the control timestep.

NOT edges, and worth stating because a discovery method that adds them is making a specific error:
  joint_pos -> joint_pos across joints     positions do not act on each other directly; coupling
                                           is at the velocity level. Measured cross-joint partial
                                           correlation at the position level stays below .15.
  eef -> joint                             the end-effector is a function of the joints, not a
                                           cause of them.
  gripper <-> arm                          the finger chain is actuated separately.

Env: OUT=<json>. Reads the analytic Jacobian from a live simulator, so run in the robotics venv
with MUJOCO_GL=egl.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("OUT", os.path.join(HERE, "outputs", "lift_body_true_graph.json"))

JOINTS = list(range(1, 8))
COAXIAL = [(1, 3), (5, 7), (3, 7)]          # measured, and consistent with the axis grouping
EEF_POS = ["x", "y", "z"]
EEF_QUAT = ["x", "y", "z", "w"]
FINGERS = [1, 2]


def lag(name):
    return f"{name}@t-1"


def now(name):
    return f"{name}@t"


def build(jac_pos, dt):
    """Edges as (parent, child, kind, weight). Weight is None where it is configuration
    dependent and cannot be written as one number."""
    E = []
    jp = [f"robot0_joint_pos.{i}" for i in JOINTS]
    jv = [f"robot0_joint_vel.{i}" for i in JOINTS]
    ep = [f"robot0_eef_pos.{a}" for a in EEF_POS]
    eq = [f"robot0_eef_quat.{a}" for a in EEF_QUAT]
    gp = [f"robot0_gripper_qpos.{i}" for i in FINGERS]
    gv = [f"robot0_gripper_qvel.{i}" for i in FINGERS]
    act = [f"action.{i}" for i in range(1, 8)]

    for a in act[:6]:
        for v in jv:
            E.append((lag(a), now(v), "control", None))
    for i, v in enumerate(jv):
        E.append((lag(v), now(jp[i]), "integration", dt))
        E.append((lag(v), now(v), "dynamics_self", None))
    for i, j in COAXIAL:
        E.append((lag(jv[i - 1]), now(jv[j - 1]), "inertial_coupling", None))
        E.append((lag(jv[j - 1]), now(jv[i - 1]), "inertial_coupling", None))
    for r, coord in enumerate(ep):
        for c, j in enumerate(jp):
            w = float(jac_pos[r, c])
            if abs(w) > 1e-9:
                E.append((now(j), now(coord), "forward_kinematics", w))
    for coord in eq:
        for j in jp:
            E.append((now(j), now(coord), "forward_kinematics_rot", None))
    for g in gv:
        E.append((lag(act[6]), now(g), "control", None))
    for i, g in enumerate(gv):
        E.append((lag(g), now(gp[i]), "integration", dt))
        E.append((lag(g), now(g), "dynamics_self", None))
    return E


def main():
    import mujoco
    import robosuite as suite
    env = suite.make("Lift", robots="Panda", has_renderer=False, has_offscreen_renderer=False,
                     use_camera_obs=False, control_freq=20)
    env.reset()
    dt = float(env.control_timestep)
    r = env.robots[0]
    m, d = env.sim.model._model, env.sim.data._data
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "gripper0_right_eef")
    jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    mujoco.mj_jacBody(m, d, jacp, jacr, bid)
    J = jacp[:, r._ref_joint_vel_indexes]

    E = build(J, dt)
    kinds = {}
    for _, _, k, _ in E:
        kinds[k] = kinds.get(k, 0) + 1
    nodes = sorted({x for e in E for x in e[:2]})
    out = {
        "scope": "robot body only, cube excluded (contact edge is state dependent)",
        "control_timestep": dt,
        "n_nodes": len(nodes),
        "n_edges": len(E),
        "edges_by_kind": kinds,
        "edges": [{"from": a, "to": b, "kind": k, "weight": w} for a, b, k, w in E],
        "analytic_eef_jacobian": J.tolist(),
        "coaxial_pairs": COAXIAL,
        "verified": {
            "control_timestep": "read from the simulator",
            "forward_kinematics": "analytic Jacobian from MuJoCo; joint 7 column is zero",
            "coaxial_coupling": "partial correlation, controlling for own past and the action: "
                                "1-3 .231, 5-7 .233, 3-7 .167; joints 2,4,6 below .05",
        },
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"[true graph] {len(nodes)} nodes, {len(E)} edges")
    for k, v in sorted(kinds.items()):
        print(f"    {k:22s} {v}")
    print(f"[true graph] joint-7 column of the eef Jacobian: "
          f"{np.abs(J[:, 6]).max():.2e} (expected 0)")
    print(f"[true graph] saved {OUT}")


if __name__ == "__main__":
    main()
