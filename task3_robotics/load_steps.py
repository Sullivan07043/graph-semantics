"""Step-level, lag-1 design matrix for the robot body.

Time is the identification resource here, so it is kept rather than aggregated away. An earlier
trajectory-level version summarized each episode into five statistics per channel; that was wrong,
because the relations among those statistics are created by the aggregation, not by the robot. 54%
of those variables had another statistic of the same base channel as their nearest correlate.

Rows are consecutive step pairs within an episode, so a row is one transition. Columns are the 32
body channels at t-1, the 7 actions at t-1, and the same 32 channels at t. The cube is excluded:
its coupling to the end-effector exists only during contact, and a static graph cannot express a
state-dependent edge.

Env: HDF5=<robomimic file> DEMOS=0 (all) OUT=<npz>
"""
import os

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = "/data2/shuhao/semantic_interpretation/data/pool/robomimic"
HDF5 = os.environ.get("HDF5", os.path.join(DATA, "lift_mg_low_dim_dense_v15.hdf5"))
OUT = os.environ.get("OUT", os.path.join(HERE, "outputs", "lift_body_steps.npz"))
DEMOS = int(os.environ.get("DEMOS", 0))

BODY = [
    ("robot0_joint_pos", 7, "angle of robot arm joint {i}"),
    ("robot0_joint_vel", 7, "angular velocity of robot arm joint {i}"),
    ("robot0_eef_pos", 3, "{a} coordinate of the robot end-effector position"),
    ("robot0_eef_quat", 4, "{a} component of the end-effector orientation quaternion"),
    ("robot0_gripper_qpos", 2, "opening of gripper finger {i}"),
    ("robot0_gripper_qvel", 2, "opening speed of gripper finger {i}"),
]
ACTION_TEXT = ["commanded end-effector motion along {a}",
               "commanded end-effector rotation about {a}"]
AXIS = ["x", "y", "z", "w"]


def schema():
    names, labels = [], []
    for key, dim, text in BODY:
        for j in range(dim):
            if dim in (3, 4):
                names.append(f"{key}.{AXIS[j]}")
                labels.append(text.format(a=AXIS[j]))
            else:
                names.append(f"{key}.{j + 1}")
                labels.append(text.format(i=j + 1))
    act_n, act_l = [], []
    for j in range(7):
        act_n.append(f"action.{j + 1}")
        if j < 3:
            act_l.append(ACTION_TEXT[0].format(a=AXIS[j]))
        elif j < 6:
            act_l.append(ACTION_TEXT[1].format(a=AXIS[j - 3]))
        else:
            act_l.append("commanded gripper opening or closing")
    return names, labels, act_n, act_l


def main():
    names, labels, act_n, act_l = schema()
    f = h5py.File(HDF5, "r")["data"]
    keys = sorted(f.keys(), key=lambda s: int(s.split("_")[-1]))
    if DEMOS:
        keys = keys[:DEMOS]
    prev, act, cur = [], [], []
    for k in keys:
        obs = f[k]["obs"]
        S = np.concatenate([np.atleast_2d(obs[key][:]) for key, _, _ in BODY], axis=1).astype(float)
        A = f[k]["actions"][:].astype(float)
        prev.append(S[:-1])
        act.append(A[:-1])
        cur.append(S[1:])
    P, A, C = np.concatenate(prev), np.concatenate(act), np.concatenate(cur)

    X = np.column_stack([P, A, C])
    cols = ([f"{n}@t-1" for n in names] + [f"{n}@t-1" for n in act_n]
            + [f"{n}@t" for n in names])
    texts = labels + act_l + labels
    assert X.shape[1] == len(cols) == len(texts)

    keep = X.std(0) > 1e-9
    dropped = [c for c, k in zip(cols, keep) if not k]
    X = X[:, keep]
    cols = [c for c, k in zip(cols, keep) if k]
    texts = [t for t, k in zip(texts, keep) if k]

    print(f"[steps] {len(keys)} demos -> {X.shape[0]} transitions x {X.shape[1]} columns")
    print(f"[steps] {len([c for c in cols if c.endswith('@t-1')])} at t-1, "
          f"{len([c for c in cols if c.endswith('@t')])} at t")
    if dropped:
        print(f"[steps] constant columns dropped: {dropped}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT, X=X, names=np.array(cols), labels=np.array(texts),
             n_past=int(sum(1 for c in cols if c.endswith("@t-1"))), source=os.path.basename(HDF5))
    print(f"[steps] saved {OUT}")


if __name__ == "__main__":
    main()
