"""Turn step-level rollouts into one row per episode.

Why the episode is the unit. precheck.py measured within-episode autocorrelation still at .135 at
lag 50, so 40000 steps carry about the information of 200 rows. Thinning cannot repair that; the
episode can. Episodes are independent by construction, so a matrix of one row per episode satisfies
the assumption every dependence estimate in the pipeline makes.

It is also the level at which a translated latent says something worth saying. A step-level latent
would mean "this dimension moves joint 3 right now". An episode-level latent means "this dimension
decides how high the cube ends up", which is what interpreting a policy is for.

Variables. Each scalar state variable becomes five episode-level variables (start, end, average,
lowest, highest), each with its own natural-language label built from the base label. Task outcome
variables are added separately, since they are the semantically richest targets. Columns that do
not vary across episodes are dropped: a joint angle that is identical at every reset carries no
dependence, while a randomized initial cube position survives, which is the interesting case.

Env: NPZ=<step-level npz> OUT=<episode-level npz> DEDUP_R=0.999
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NPZ = os.environ.get("NPZ", os.path.join(HERE, "outputs", "lift_traj.npz"))
OUT = os.environ.get("OUT", os.path.join(HERE, "outputs", "lift_episode.npz"))
DEDUP_R = float(os.environ.get("DEDUP_R", 0.999))

STATS = [
    ("start", lambda Z: Z[0], "the {L} at the start of the episode"),
    ("end", lambda Z: Z[-1], "the {L} at the end of the episode"),
    ("mean", lambda Z: Z.mean(0), "the average {L} over the episode"),
    ("min", lambda Z: Z.min(0), "the lowest {L} during the episode"),
    ("max", lambda Z: Z.max(0), "the highest {L} during the episode"),
]


def episode_rows(X, episode, reward, success):
    eps = np.unique(episode)
    per_stat = {tag: [] for tag, _, _ in STATS}
    out = []
    for e in eps:
        m = episode == e
        Z = X[m]
        for tag, fn, _ in STATS:
            per_stat[tag].append(fn(Z))
        r, s = reward[m], success[m]
        out.append([float(r.sum()), float(r.max()), float(r[-1]), float(s.max())])
    stacked = np.column_stack([np.asarray(per_stat[tag]) for tag, _, _ in STATS])
    return stacked, np.asarray(out, float), len(eps)


OUTCOME = [
    ("episode.total_reward", "the total shaped reward earned during the episode"),
    ("episode.max_reward", "the highest shaped reward reached during the episode"),
    ("episode.final_reward", "the shaped reward at the final step of the episode"),
    ("episode.success", "whether the robot successfully lifted the cube in this episode"),
]


def main():
    d = np.load(NPZ, allow_pickle=True)
    X, names, labels = d["X"], list(d["names"]), list(d["labels"])
    episode, reward, success = d["episode"], d["reward"], d["success"]

    S, O, n_eps = episode_rows(X, episode, reward, success)
    stat_names, stat_labels = [], []
    for tag, _, template in STATS:
        for nm, lb in zip(names, labels):
            stat_names.append(f"{nm}.{tag}")
            stat_labels.append(template.format(L=str(lb)))
    E = np.column_stack([S, O])
    enames = stat_names + [n for n, _ in OUTCOME]
    elabels = stat_labels + [t for _, t in OUTCOME]
    assert E.shape[1] == len(enames), (E.shape, len(enames))

    keep = E.std(0) > 1e-9
    E, enames, elabels = (E[:, keep], [n for n, k in zip(enames, keep) if k],
                          [l for l, k in zip(elabels, keep) if k])
    C = np.corrcoef(E, rowvar=False)
    dup = [(enames[i], enames[j], float(C[i, j]))
           for i in range(len(enames)) for j in range(i + 1, len(enames))
           if abs(C[i, j]) > DEDUP_R]

    print(f"[episode] {n_eps} episodes | {E.shape[1]} variables "
          f"(dropped {int((~keep).sum())} that do not vary across episodes)")
    print(f"[episode] task outcomes kept: "
          f"{[n for n in enames if n.startswith('episode.')]}")
    if dup:
        print(f"[episode] pairs above |r|>{DEDUP_R} (reported, not dropped): {len(dup)}")
        for a, b, r in dup[:8]:
            print(f"    {a} ~ {b}  r={r:+.4f}")
    print("[episode] sample labels:")
    for i in list(range(3)) + list(range(len(enames) - 4, len(enames))):
        print(f"    {enames[i]:34s} {elabels[i]}")

    np.savez(OUT, X=E, names=np.array(enames), labels=np.array(elabels),
             task=d["task"], robot=d["robot"], n_episodes=n_eps)
    print(f"[episode] saved {OUT}")


if __name__ == "__main__":
    main()
