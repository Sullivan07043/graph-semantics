"""Type I error of the RLCD rank test on Likert-discretized data.

Motivation: causal-learn's Chi2RankTest standardizes the RAW columns and runs a CCA Wilks
chi-square, i.e. it uses Pearson correlations of the discretized data. arXiv:2501.18990 reports
that this inflates rejection under discretization. If it inflates on OUR data profile, then part
of the ">25 variables does not work" finding is test miscalibration, not search-space size, and
that has to be known before choosing between FOFC and a cluster-then-refine scheme.

Design: a pure one-factor measurement model, so any split of its items into two disjoint sets has
TRUE cross-covariance rank 1. We test the null "rank <= 1", which is true by construction, and
count rejections. A calibrated test rejects at the nominal alpha.

Arms per (n, items-per-side): continuous, then 5-point and 4-point discretizations with both
symmetric and skewed thresholds (clinical items are skewed: most respondents answer at the floor).
Loadings are drawn per replication so the result is not one lucky factor structure.

Env: REPS=200 SEED=0 OUT=<json path>
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CLHOME = os.path.join(os.path.dirname(HERE), "causal-learn")
CL = os.path.join(CLHOME, "upstream", "causal-learn")
sys.path.insert(0, CL)
sys.path.insert(0, CLHOME)          # polychoric.py lives at the causal-learn home
sys.path.insert(0, HERE)
from causallearn.search.HiddenCausal.RLCD.Chi2RankTest import Chi2RankTest  # noqa: E402
from polychoric import PolychoricRankTest                                  # noqa: E402

REPS = int(os.environ.get("REPS", 200))
SEED = int(os.environ.get("SEED", 0))
OUT = os.environ.get("OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "outputs", "rank_calibration.json"))
ALPHAS = [0.01, 0.05, 0.10]
# thresholds are on the standardized latent+noise scale; skewed ones put most mass in category 1
CUTS = {
    "likert5_sym": [-1.5, -0.5, 0.5, 1.5],
    "likert5_skew": [0.25, 0.85, 1.4, 2.0],
    "likert4_sym": [-1.0, 0.0, 1.0],
}


def one_factor(rng, n, k, loading_lo=0.5, loading_hi=0.8):
    """n x (2k) items from a single latent factor. Any k-vs-k split has cross-cov rank 1."""
    f = rng.standard_normal(n)
    lam = rng.uniform(loading_lo, loading_hi, 2 * k)
    e = rng.standard_normal((n, 2 * k)) * np.sqrt(1 - lam ** 2)
    return f[:, None] * lam[None, :] + e


def discretize(X, cuts):
    return np.digitize(X, cuts).astype(float) + 1.0


def run(n, k, reps, seed):
    """Arms are (data profile, rank test). Polychoric is only defined for the discretized profiles,
    so the continuous row exists for the default test only."""
    rng = np.random.default_rng(seed)
    arms = ["continuous"] + [f"{c}{suffix}" for c in CUTS for suffix in ("", "+poly")]
    rej = {a: {al: 0 for al in ALPHAS} for a in arms}
    for _ in range(reps):
        X = one_factor(rng, n, k)
        pcols, qcols = list(range(k)), list(range(k, 2 * k))
        for arm in arms:
            if arm == "continuous":
                t = Chi2RankTest(X)
            else:
                D = discretize(X, CUTS[arm.replace("+poly", "")])
                t = PolychoricRankTest(D) if arm.endswith("+poly") else Chi2RankTest(D)
            for al in ALPHAS:
                # test() returns if_fail_to_reject; a rejection of a TRUE null is a type I error
                if not t.test(pcols, qcols, 1, al):
                    rej[arm][al] += 1
    return {a: {str(al): rej[a][al] / reps for al in ALPHAS} for a in arms}


if __name__ == "__main__":
    out = {"reps": REPS, "seed": SEED, "cuts": CUTS, "cells": {}}
    for n in [2000, 5000]:
        for k in [3, 5]:
            key = f"n{n}_k{k}"
            print(f"[start {key}] {REPS} reps", flush=True)
            out["cells"][key] = run(n, k, REPS, SEED)
            print(f"[{key}] true rank 1, null 'rank<=1' is TRUE; type I error rate:", flush=True)
            for arm, d in out["cells"][key].items():
                print(f"   {arm:14s} " + "  ".join(f"a={a}: {r:.3f}" for a, r in d.items()),
                      flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"[saved {OUT}]", flush=True)
