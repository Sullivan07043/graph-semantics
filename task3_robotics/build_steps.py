"""Collector npz -> lag-1 design matrix, dimension-adaptive.

Same output format as load_steps.py produced from the robomimic file (X with @t-1 and @t columns,
names, labels, n_past), so discover.py and the pipeline loader work unchanged on any robot.

Channel policy:
  - body channels only (robot0_*); cube channels are excluded, as everywhere in the body task
  - robot0_joint_acc is excluded, so every robot carries the same channel FAMILIES as the Panda
    robomimic set the dev pool starts from (that file has no acceleration)
  - action columns come from the recorded actions array, named action.1..N from its width, since
    the arm dimension varies by robot

Env: NPZ=<collector npz> OUT=<design npz>
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NPZ = os.environ["NPZ"]
OUT = os.environ.get("OUT", NPZ.replace("_rollouts.npz", "_steps.npz"))

EXCLUDE_PREFIX = ("cube_", "gripper_to_cube")
EXCLUDE_FAMILY = ("robot0_joint_acc",)
ACTION_TEXT_ARM = ["commanded end-effector motion along {a}",
                   "commanded end-effector rotation about {a}"]
AXIS = ["x", "y", "z"]


def action_schema(width):
    names, labels = [], []
    for j in range(width):
        names.append(f"action.{j + 1}")
        if j < 3:
            labels.append(ACTION_TEXT_ARM[0].format(a=AXIS[j]))
        elif j < 6:
            labels.append(ACTION_TEXT_ARM[1].format(a=AXIS[j - 3]))
        else:
            labels.append("commanded gripper opening or closing")
    return names, labels


def main():
    d = np.load(NPZ, allow_pickle=True)
    X, names = np.asarray(d["X"], float), [str(x) for x in d["names"]]
    labels = [str(x) for x in d["labels"]]
    episode = np.asarray(d["episode"], int)
    A = np.asarray(d["actions"], float)
    assert len(A) == len(X), (A.shape, X.shape)

    keep = [i for i, n in enumerate(names)
            if not n.startswith(EXCLUDE_PREFIX)
            and not any(n.startswith(f) for f in EXCLUDE_FAMILY)]
    names = [names[i] for i in keep]
    labels = [labels[i] for i in keep]
    X = X[:, keep]

    # consecutive step pairs within an episode
    same = episode[1:] == episode[:-1]
    prev, cur, act = X[:-1][same], X[1:][same], A[:-1][same]

    an, al = action_schema(A.shape[1])
    M = np.column_stack([prev, act, cur])
    cols = ([f"{n}@t-1" for n in names] + [f"{n}@t-1" for n in an]
            + [f"{n}@t" for n in names])
    texts = labels + al + labels
    assert M.shape[1] == len(cols) == len(texts)

    nonconst = M.std(0) > 1e-9
    dropped = [c for c, k in zip(cols, nonconst) if not k]
    M = M[:, nonconst]
    cols = [c for c, k in zip(cols, nonconst) if k]
    texts = [t for t, k in zip(texts, nonconst) if k]

    robot = str(d["robot"]) if "robot" in d else "?"
    print(f"[{robot}] {M.shape[0]} transitions x {M.shape[1]} columns "
          f"({len([c for c in cols if c.endswith('@t')])} at t); dropped constant: {len(dropped)}")
    np.savez(OUT, X=M, names=np.array(cols), labels=np.array(texts),
             n_past=int(sum(1 for c in cols if c.endswith("@t-1"))),
             robot=robot, source=os.path.basename(NPZ))
    print(f"[{robot}] saved {OUT}")


if __name__ == "__main__":
    main()
