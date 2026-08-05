"""P0 pre-checks on the collected rollouts, before any pipeline run.

Two gates, the robotics counterparts of the checks the questionnaire testbeds must pass.

1. DICTIONARY COVERAGE. The decode space is a fixed 522k concept bank of general English. Robot
   state vocabulary was never checked against it. If the words that would have to appear in a
   correct answer are missing from the bank, a low score would say nothing about the method. This
   is the same check run before DASS and NHANES.

2. THE IID GATE. Every dependence estimate in the pipeline (Pearson, distance correlation, the
   rank tests) assumes rows are independent draws. Rollout steps are not: consecutive steps of one
   episode are strongly autocorrelated, so n rows carry far less than n rows of information. This
   measures the autocorrelation and reports the thinning stride at which it decays, plus the
   effective sample size that stride implies. The output is a decision, not a diagnosis: it says
   how to subsample before anything downstream is run.

Runs in the certified pipeline venv, because it needs encode/splice. Reads the npz written by
collect_rollouts.py, which ran in the robotics venv.

Env: NPZ=<path> MAXLAG=50 TARGET_RHO=0.1
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V6 = os.path.join(os.path.dirname(HERE), "v6")
sys.path.insert(0, V6)

NPZ = os.environ.get("NPZ", os.path.join(HERE, "outputs", "lift_rollouts.npz"))
MAXLAG = int(os.environ.get("MAXLAG", 50))
TARGET_RHO = float(os.environ.get("TARGET_RHO", 0.1))


def coverage(labels):
    """Fraction of content words in the label texts that exist as dictionary atoms."""
    import re
    dict_path = os.environ.get("GRAPHSEM_DICT",
                               os.path.join(V6, "outputs", "concept_bank_l3_cog.npz"))
    names = {str(x).lower() for x in np.load(dict_path, allow_pickle=True)["names"]}
    stop = {"of", "the", "a", "an", "and", "to", "in", "for", "at", "on", "component"}
    words = sorted({w for t in labels for w in re.findall(r"[a-z]+", str(t).lower())} - stop)
    hit = [w for w in words if w in names]
    miss = [w for w in words if w not in names]
    print(f"[coverage] label vocabulary {len(words)} distinct content words | "
          f"in dictionary {len(hit)} | missing {len(miss)}")
    if miss:
        print(f"[coverage] MISSING: {', '.join(miss)}")
    # whole phrases matter more than single words for the decode
    phrases = sorted({str(t).lower() for t in labels})
    ph_hit = sum(p in names for p in phrases)
    print(f"[coverage] whole label phrases present as atoms: {ph_hit}/{len(phrases)} "
          f"(low is expected and fine; decode composes atoms)")
    return len(miss)


def autocorr_gate(X, episode, maxlag, target):
    """Within-episode lag-k autocorrelation, averaged over variables and episodes."""
    eps = np.unique(episode)
    p = X.shape[1]
    ac = np.zeros(maxlag + 1)
    cnt = np.zeros(maxlag + 1)
    for e in eps:
        Z = X[episode == e]
        if len(Z) < maxlag + 2:
            continue
        Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
        for k in range(maxlag + 1):
            if len(Z) - k < 5:
                continue
            r = (Z[:len(Z) - k] * Z[k:]).mean(0)          # per-variable lag-k correlation
            ac[k] += np.abs(r).mean()
            cnt[k] += 1
    ac = ac / np.maximum(cnt, 1)
    stride = next((k for k in range(1, maxlag + 1) if ac[k] < target), None)
    n_rows, n_eps = len(X), len(eps)
    print(f"[iid] rows {n_rows} over {n_eps} episodes, {p} variables")
    print("[iid] mean |lag-k autocorrelation| within an episode:")
    for k in [1, 2, 5, 10, 20, 30, 50]:
        if k <= maxlag:
            print(f"        lag {k:3d}: {ac[k]:.3f}")
    if stride is None:
        print(f"[iid] autocorrelation never falls below {target} within lag {maxlag}. "
              f"Thinning cannot fix this: sample ONE row per episode.")
        eff = n_eps
    else:
        eff = int(n_rows / stride)
        print(f"[iid] first lag below {target}: {stride}. Thinning by {stride} gives an effective "
              f"sample of about {eff} rows.")
    print(f"[iid] DECISION: usable independent sample is about {eff} rows "
          f"(raw row count {n_rows} overstates it by {n_rows / max(eff, 1):.0f}x).")
    return eff


if __name__ == "__main__":
    d = np.load(NPZ, allow_pickle=True)
    X, labels, episode = d["X"], d["labels"], d["episode"]
    print(f"=== {str(d['task'])} | X {X.shape} ===")
    miss = coverage(labels)
    print()
    eff = autocorr_gate(X, episode, MAXLAG, TARGET_RHO)
    print()
    print(f"=== P0 verdict: dictionary gaps {miss}, effective sample {eff} ===")
