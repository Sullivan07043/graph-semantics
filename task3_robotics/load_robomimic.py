"""Turn a robomimic HDF5 into the same episode-level matrix trajectorize.py produces.

Why this dataset. Random actions never lift the cube, so with self-collected rollouts the most
meaningful variables degenerate: success is constant false, and the highest cube height equals its
starting height. The robomimic machine-generated set comes from five SAC checkpoints of differing
ability, so 316 of its 1500 trajectories succeed. Success varies, and so does everything downstream
of it. The self-collected random-action set is kept as a sanity-check control: the same pipeline on
purposeless behaviour should not recover task structure.

Output matches trajectorize.py exactly (X, names, labels, task, robot, n_episodes), so structure
discovery, Task 1, Task 2 and the intervention check treat both sources identically.

Two dataset-specific fixes, both declared:
  - `object` is packed. In robosuite Lift it is cube position (3), cube quaternion (4), then the
    gripper-to-cube vector (3). The last is cube_pos minus eef_pos, so it is dropped as derived.
  - `robot0_joint_acc` is present in a live environment but not in this file. Its five statistics
    simply do not exist here.

Env: HDF5=<path> OUT=<npz> DEDUP_R=0.999
"""
import os

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = "/data2/shuhao/semantic_interpretation/data/pool/robomimic"
HDF5 = os.environ.get("HDF5", os.path.join(DATA, "lift_mg_low_dim_dense_v15.hdf5"))
OUT = os.environ.get("OUT", os.path.join(HERE, "outputs", "lift_mg_episode.npz"))
DEDUP_R = float(os.environ.get("DEDUP_R", 0.999))

SKIP = {"robot0_joint_pos_cos", "robot0_joint_pos_sin"}
# unpack robosuite Lift's object-state; the trailing gripper-to-cube vector is derived
OBJECT_PARTS = [("cube_pos", 3), ("cube_quat", 4), ("gripper_to_cube_pos", 3)]
OBJECT_DROP = {"gripper_to_cube_pos"}

AXIS = ["x", "y", "z", "w"]
TEXT = {
    "robot0_joint_pos": "angle of robot arm joint {i}",
    "robot0_joint_vel": "angular velocity of robot arm joint {i}",
    "robot0_eef_pos": "{ax} coordinate of the robot gripper position",
    "robot0_eef_quat": "{ax} component of the robot gripper orientation quaternion",
    "robot0_eef_quat_site": "{ax} component of the gripper site orientation quaternion",
    "robot0_gripper_qpos": "opening of gripper finger {i}",
    "robot0_gripper_qvel": "closing speed of gripper finger {i}",
    "cube_pos": "{ax} coordinate of the cube position",
    "cube_quat": "{ax} component of the cube orientation quaternion",
}
STATS = [
    ("start", lambda Z: Z[0], "the {L} at the start of the episode"),
    ("end", lambda Z: Z[-1], "the {L} at the end of the episode"),
    ("mean", lambda Z: Z.mean(0), "the average {L} over the episode"),
    ("min", lambda Z: Z.min(0), "the lowest {L} during the episode"),
    ("max", lambda Z: Z.max(0), "the highest {L} during the episode"),
]
OUTCOME = [
    ("episode.success", "whether the robot successfully lifted the cube in this episode"),
    ("episode.total_reward", "the total shaped reward earned during the episode"),
    ("episode.max_reward", "the highest shaped reward reached during the episode"),
    ("episode.time_to_first_reward",
     "how long the robot took before earning any reward in this episode"),
]


def scalarize(key, dim):
    out = []
    for j in range(dim):
        if dim == 4 and "quat" in key or (dim == 3 and "quat" not in key):
            tag, sub = AXIS[j], {"ax": AXIS[j]}
        else:
            tag, sub = str(j + 1), {"i": j + 1}
        t = TEXT.get(key)
        out.append((f"{key}.{tag}", t.format(**sub) if t else f"{key} component {tag}"))
    return out


def episode_blocks(demo):
    """[T, k] per base variable, with `object` unpacked and derived parts removed."""
    obs = demo["obs"]
    blocks = []
    for k in sorted(obs):
        if k in SKIP:
            continue
        if k == "object":
            A = obs[k][:]
            off = 0
            for part, dim in OBJECT_PARTS:
                if part not in OBJECT_DROP:
                    blocks.append((part, A[:, off:off + dim]))
                off += dim
        else:
            blocks.append((k, np.atleast_2d(obs[k][:])))
    return sorted(blocks, key=lambda b: b[0])


def main():
    f = h5py.File(HDF5, "r")
    data = f["data"]
    keys = sorted(data.keys(), key=lambda s: int(s.split("_")[-1]))
    schema, rows, outs = None, [], []
    for i, k in enumerate(keys):
        demo = data[k]
        blocks = episode_blocks(demo)
        if schema is None:
            schema = [t for name, Z in blocks for t in scalarize(name, Z.shape[1])]
        Z = np.concatenate([Z for _, Z in blocks], axis=1).astype(float)
        rows.append(np.concatenate([fn(Z) for _, fn, _ in STATS]))
        r = demo["rewards"][:].astype(float)
        first = np.argmax(r > 0) if (r > 0).any() else len(r)
        outs.append([float((r > 0).any()), float(r.sum()), float(r.max()), float(first)])
        if (i + 1) % 300 == 0:
            print(f"[mg] {i + 1}/{len(keys)} demos", flush=True)

    names, labels = [], []
    for tag, _, template in STATS:
        for nm, lb in schema:
            names.append(f"{nm}.{tag}")
            labels.append(template.format(L=lb))
    X = np.column_stack([np.asarray(rows, float), np.asarray(outs, float)])
    names += [n for n, _ in OUTCOME]
    labels += [t for _, t in OUTCOME]
    assert X.shape[1] == len(names), (X.shape, len(names))

    keep = X.std(0) > 1e-9
    X, names, labels = (X[:, keep], [n for n, k in zip(names, keep) if k],
                        [l for l, k in zip(labels, keep) if k])
    C = np.corrcoef(X, rowvar=False)
    dup = [(names[i], names[j], float(C[i, j]))
           for i in range(len(names)) for j in range(i + 1, len(names))
           if abs(C[i, j]) > DEDUP_R]
    succ = X[:, names.index("episode.success")] if "episode.success" in names else None
    print(f"[mg] {X.shape[0]} episodes x {X.shape[1]} variables "
          f"(dropped {int((~keep).sum())} constant)")
    if succ is not None:
        print(f"[mg] success rate {succ.mean():.3f}")
    print(f"[mg] near-duplicate pairs above |r|>{DEDUP_R}: {len(dup)}")
    for a, b, r in dup[:8]:
        print(f"    {a} ~ {b}  r={r:+.4f}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT, X=X, names=np.array(names), labels=np.array(labels),
             task="Lift", robot="Panda", n_episodes=X.shape[0], source=os.path.basename(HDF5))
    print(f"[mg] saved {OUT}")


if __name__ == "__main__":
    main()
