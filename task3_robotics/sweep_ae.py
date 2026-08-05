"""Choose the autoencoder sparsity so the latents are distinguishable.

The first run at the archive defaults produced 16 latents with 58 children each, and several pairs
whose top loadings were the same variables. Latents that cannot be told apart cannot be translated
apart, which is the failure the questionnaire pool already shows as a large judge-minus-match gap.
So sparsity is not a knob to tune for a better score; it decides whether Task 2 is well posed here.

Selection is by structure, not by any downstream score. Reported per setting:
  median children per latent, wanted in the 5 to 15 band the questionnaire testbeds sit in
  duplicate pairs, latents whose top-5 loaded variables overlap in 4 or more
  outcome coverage, whether any latent loads on reward or success at all
  reconstruction, so a setting is not chosen that has stopped explaining the data

Env: NPZ=<episode npz> K_LIST=8,12,16 LAM_LIST=0.05,0.2,0.5 TAU_LIST=0.2,0.4 AEEPOCHS=600
"""
import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/data2/shuhao/semantic_interpretation/archive_phases/latent_concept/scripts")

NPZ = os.environ.get("NPZ", os.path.join(HERE, "outputs", "lift_mg_episode.npz"))
OUT = os.environ.get("OUT", os.path.join(HERE, "outputs", "ae_sweep.json"))
K_LIST = [int(x) for x in os.environ.get("K_LIST", "8,12,16").split(",")]
LAM_LIST = [float(x) for x in os.environ.get("LAM_LIST", "0.05,0.2,0.5").split(",")]
TAU_LIST = [float(x) for x in os.environ.get("TAU_LIST", "0.2,0.4").split(",")]


def summarize(B, names):
    k = B.shape[1]
    counts = [int((B[:, j] != 0).sum()) for j in range(k)]
    tops = []
    for j in range(k):
        idx = np.argsort(-np.abs(B[:, j]))[:5]
        tops.append({names[i] for i in idx if B[i, j] != 0})
    dup = sum(1 for a in range(k) for b in range(a + 1, k) if len(tops[a] & tops[b]) >= 4)
    outcome = [names[i] for i in range(len(names)) if names[i].startswith("episode.")
               and np.any(B[i] != 0)]
    orphans = int(sum(1 for i in range(len(names)) if not np.any(B[i] != 0)))
    return counts, dup, outcome, orphans


def main():
    import causal_ae
    d = np.load(NPZ, allow_pickle=True)
    X = np.asarray(d["X"], float)
    names = [str(x) for x in d["names"]]
    rows = []
    for K, lam, tau in itertools.product(K_LIST, LAM_LIST, TAU_LIST):
        Z, B, _ = causal_ae.estimate(X, K, lam=lam, tau=tau, verbose=False,
                                     dev=os.environ.get("DEV", "cpu"))
        counts, dup, outcome, orphans = summarize(B, names)
        # reconstruction from the surviving sparse loadings, a floor check that the setting still
        # explains the data rather than having zeroed everything
        rec = float(((np.asarray(Z) @ B.T - (X - X.mean(0)) / (X.std(0) + 1e-9)) ** 2).mean())
        med = int(np.median(counts)) if counts else 0
        band = 5 <= med <= 15
        rows.append({"K": K, "lam": lam, "tau": tau, "k_surviving": int(B.shape[1]),
                     "median_children": med, "duplicate_pairs": dup,
                     "outcome_vars_loaded": outcome, "orphan_observed": orphans,
                     "linear_rec_mse": round(rec, 3), "in_band": band})
        print(f"K={K:3d} lam={lam:<5} tau={tau:<4} -> {B.shape[1]:2d} latents, "
              f"median children {med:3d}{' *' if band else '  '}, dup pairs {dup:2d}, "
              f"orphans {orphans:3d}, outcomes {len(outcome)}, rec {rec:.2f}", flush=True)

    json.dump(rows, open(OUT, "w"), indent=1)
    ok = [r for r in rows if r["in_band"] and r["duplicate_pairs"] == 0]
    print()
    if ok:
        best = min(ok, key=lambda r: (-len(r["outcome_vars_loaded"]), r["linear_rec_mse"]))
        print(f"[sweep] in band and no duplicate latents: {len(ok)} settings")
        print(f"[sweep] pick K={best['K']} lam={best['lam']} tau={best['tau']} "
              f"({best['k_surviving']} latents, median {best['median_children']} children, "
              f"outcome variables loaded: {best['outcome_vars_loaded']})")
    else:
        print("[sweep] NO setting reached the band with distinguishable latents. "
              "The sparsity knobs alone do not fix this.")
    print(f"[sweep] saved {OUT}")


if __name__ == "__main__":
    main()
