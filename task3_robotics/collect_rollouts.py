"""Collect robosuite rollouts as a named-variable data matrix.

Runs in the ISOLATED robotics venv (/data2/shuhao/venv_robo), because robosuite pins numpy 1.26
while the certified pipeline runs numpy 2.2 under torch 2.9. The two never share a process. The
interface between them is this script's output file, exactly as discovery/ already hands graphs to
the pipeline.

What it produces, per task:
  X          [n, p] float, one row per simulator step, one column per SCALAR state variable
  names      [p] short variable ids, e.g. robot0_eef_pos.z
  labels     [p] natural-language text for each variable, the Task 1 target
  episode    [n] episode index of each row, needed because rows within an episode are NOT iid
  reward     [n] shaped reward at each step
  success    [n] task success flag at each step

Step rows are NOT the analysis unit. precheck.py measured lag-50 autocorrelation still at .135, so
40000 steps carry about 200 rows of information, one per episode. trajectorize.py turns this file
into one row per episode, which is independent by construction and is also the level at which a
latent's meaning is worth stating.

Degenerate columns are dropped by construction, the same rule used for the NHANES analytes:
  - concatenated mirrors (robot0_proprio-state, object-state) repeat other columns verbatim
  - joint_pos_cos and joint_pos_sin are deterministic functions of joint_pos
  - gripper_to_cube_pos is cube_pos minus eef_pos
Anything still correlating above DEDUP_R after that is reported, not silently removed.

Env: TASK=Lift ROBOT=Panda EPISODES=200 STEPS=200 SEED=0 OUT=<npz path>
Launch with MUJOCO_GL=egl on a headless machine.
"""
import os
import sys

import numpy as np

TASK = os.environ.get("TASK", "Lift")
ROBOT = os.environ.get("ROBOT", "Panda")
EPISODES = int(os.environ.get("EPISODES", 200))
STEPS = int(os.environ.get("STEPS", 200))
SEED = int(os.environ.get("SEED", 0))
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("OUT", os.path.join(HERE, "outputs", f"{TASK.lower()}_rollouts.npz"))
DEDUP_R = 0.999

# concatenations of other observation entries, not variables in their own right
SKIP_KEYS = {"robot0_proprio-state", "object-state"}
# deterministic functions of other kept columns
SKIP_DERIVED = {"robot0_joint_pos_cos", "robot0_joint_pos_sin", "gripper_to_cube_pos"}

AXIS = ["x", "y", "z", "w"]
TEXT = {
    "robot0_joint_pos": "angle of robot arm joint {i}",
    "robot0_joint_vel": "angular velocity of robot arm joint {i}",
    "robot0_joint_acc": "angular acceleration of robot arm joint {i}",
    "robot0_eef_pos": "{ax} coordinate of the robot gripper position",
    "robot0_eef_quat": "{ax} component of the robot gripper orientation quaternion",
    "robot0_eef_quat_site": "{ax} component of the gripper site orientation quaternion",
    "robot0_gripper_qpos": "opening of gripper finger {i}",
    "robot0_gripper_qvel": "closing speed of gripper finger {i}",
    "cube_pos": "{ax} coordinate of the cube position",
    "cube_quat": "{ax} component of the cube orientation quaternion",
}


def scalarize(key, dim):
    """One entry per scalar: (id, text). Vectors of 3 or 4 read as axes, others as indices."""
    out = []
    for j in range(dim):
        if dim in (3, 4) and "quat" not in key or (dim == 4 and "quat" in key):
            tag, sub = AXIS[j], {"ax": AXIS[j]}
        else:
            tag, sub = str(j + 1), {"i": j + 1}
        template = TEXT.get(key)
        text = template.format(**sub) if template else f"{key} component {tag}"
        out.append((f"{key}.{tag}", text))
    return out


def main():
    import robosuite as suite
    env = suite.make(TASK, robots=ROBOT, has_renderer=False, has_offscreen_renderer=False,
                     use_camera_obs=False, control_freq=20, reward_shaping=True)
    rng = np.random.default_rng(SEED)
    lo, hi = env.action_spec
    rows, epis, rews, succ, acts, schema = [], [], [], [], [], None
    for ep in range(EPISODES):
        obs = env.reset()
        if schema is None:
            schema = []
            for k in sorted(obs):
                if k in SKIP_KEYS or k in SKIP_DERIVED:
                    continue
                schema += [(k, *t) for t in scalarize(k, np.atleast_1d(obs[k]).size)]
        for _ in range(STEPS):
            # random actions explore the state space; a policy would concentrate it, which is a
            # different experiment (declared, not an oversight)
            a = rng.uniform(lo, hi)
            obs, r, done, _ = env.step(a)
            rows.append(np.concatenate([np.atleast_1d(obs[k]).ravel()
                                        for k in sorted(obs)
                                        if k not in SKIP_KEYS and k not in SKIP_DERIVED]))
            epis.append(ep)
            acts.append(a.copy())
            rews.append(float(r))
            succ.append(float(env._check_success()))
            if done:
                break
        if (ep + 1) % 25 == 0:
            print(f"[{TASK}] {ep + 1}/{EPISODES} episodes, {len(rows)} steps", flush=True)

    X = np.asarray(rows, float)
    names = [s[1] for s in schema]
    labels = [s[2] for s in schema]
    epis = np.asarray(epis, int)
    assert X.shape[1] == len(names), (X.shape, len(names))

    keep = X.std(0) > 1e-9                       # constant columns carry no dependence
    C = np.corrcoef(X[:, keep], rowvar=False)
    dup = [(np.array(names)[keep][i], np.array(names)[keep][j], float(C[i, j]))
           for i in range(C.shape[0]) for j in range(i + 1, C.shape[0])
           if abs(C[i, j]) > DEDUP_R]
    print(f"[{TASK}] X {X.shape} | constant columns dropped: {int((~keep).sum())}")
    if dup:
        print(f"[{TASK}] pairs still above |r|>{DEDUP_R} (reported, not dropped):")
        for a, b, r in dup[:10]:
            print(f"    {a} ~ {b}  r={r:+.4f}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT, X=X[:, keep], names=np.array(names)[keep], labels=np.array(labels)[keep],
             episode=epis, reward=np.asarray(rews, float), success=np.asarray(succ, float),
             actions=np.asarray(acts, float), task=TASK, robot=ROBOT)
    print(f"[{TASK}] saved {OUT}: {int(keep.sum())} variables, {len(X)} steps, "
          f"{EPISODES} episodes", flush=True)


if __name__ == "__main__":
    sys.exit(main())
