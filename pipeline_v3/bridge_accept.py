"""Phase-2 acceptance: unified bridge constraint, four arms, same folds, same judge batch.

Arms (all include f_neg + residual channel; only the bridge differs):
  base           frozen config (lower tail lam_zero + Pearson conditional anchor; no upper tail)
  bridge_pearson + upper tail from |Pearson| marginal; conditional anchor unchanged (= exact frozen
                 anchor) -> isolates the UPPER-TAIL addition
  bridge_dcor    upper tail from dcor marginal; conditional anchor magnitudes = dcor conditional,
                 SIGN from Pearson partial corr (dcor/MI are unsigned; hybrid declared)
  bridge_mi      same with kNN-MI (Gaussian-equivalent rescale)
Fixed pre-declared knobs (NOT tuned): lam_upper=0.3, kappa=0.5, q=0.7.
Pre-registered lines (README): dev judge or match mean +0.03 vs base; himi/gcbs guard <=0.05 drop.
Usage: DATASETS=... JUDGE_MODEL=gpt-5.5 python pipeline_v3/bridge_accept.py
"""
import os, sys, json, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import torch
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "6")))   # 6 parallel procs on one box:
                                                                   # unbounded OpenMP threads spin-starve
import encode, metrics, optimize, negop
import judge as judge_mod
from run_task1 import ALL_LOADERS
sys.path.insert(0, os.path.join(ROOT, "pipeline_v3"))
import dependence

FOLDS = 5
LAM_UPPER, KAPPA, Q = 0.3, 0.5, 0.7
ROUND = os.environ.get("BR_ROUND", "v1")
if ROUND == "factorial":
    # corrected design (2026-07-15 night): isolate tail vs anchor; tail arms keep the frozen
    # signed-Pearson anchor; the anchor arm keeps the tail OFF. MI recomputed full-sample upstream.
    ARMS = ["base", "tail_pearson", "tail_mi", "anchor_mi"]
else:
    ARMS = ["base", "bridge_pearson", "bridge_dcor", "bridge_mi"]


def ts():
    return time.strftime("%H:%M:%S")


def run(name, C, cwords, neg):
    ds = ALL_LOADERS[name]()
    g, X, labels = ds["graph"], ds["X"], ds["labels"]
    obs = g.observed
    oi = {o: k for k, o in enumerate(obs)}
    T = encode.embed([labels[o] for o in obs])
    Tn = metrics.norm_rows(T)
    alpha = metrics.pick_alpha(T, C)
    W, score = g.estimate_weights(X, oi)
    pc_names, pc_signed = optimize.partial_residual_corr(g, X, oi, score)
    dep = {m: {lv: dependence.load(name, lv, m) for lv in ["marginal", "conditional"]}
           for m in ["pearson", "dcor", "mi"]}
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(obs))
    folds = [perm[i::FOLDS] for i in range(FOLDS)]
    res = {a: {"judge": [], "match": []} for a in ARMS}
    records = []
    for fno, fold in enumerate(folds):
        masked = sorted(int(i) for i in fold)
        vis = {obs[i]: T[i] for i in range(len(obs)) if i not in set(masked)}
        for arm in ARMS:
            if arm == "base":
                pc, bridge = (pc_names, pc_signed), None
            elif arm.startswith("tail_"):
                meas = arm.split("_")[1]
                pc = (pc_names, pc_signed)                            # anchor FIXED for tail arms
                bridge = dict(obs=obs, dep_marg=dep[meas]["marginal"],
                              lam_upper=LAM_UPPER, kappa=KAPPA, q=Q)
            elif arm == "anchor_mi":
                hyb = np.sign(pc_signed) * dep["mi"]["conditional"]
                np.fill_diagonal(hyb, 0.0)
                pc, bridge = (pc_names, hyb), None                    # tail OFF for anchor arm
            else:
                meas = arm.split("_")[1]
                if meas == "pearson":
                    pc = (pc_names, pc_signed)
                else:
                    hyb = np.sign(pc_signed) * dep[meas]["conditional"]
                    np.fill_diagonal(hyb, 0.0)
                    pc = (pc_names, hyb)
                bridge = dict(obs=obs, dep_marg=dep[meas]["marginal"],
                              lam_upper=LAM_UPPER, kappa=KAPPA, q=Q)
            emb = optimize.optimize_embeddings(g, W, vis, d=T.shape[1], seed=fno,
                                               residual=1.0, lam_res=1.0, partial_corr=pc,
                                               neg_op=neg, bridge=bridge)
            P = np.stack([emb[obs[i]] for i in masked])
            res[arm]["match"].append(metrics.match_acc(P, masked, T))
            words = metrics.decode_words(P, C, cwords, alpha) if judge_mod.available() else None
            if words:
                jacc, verd = metrics.judge_completion(words, [labels[obs[i]] for i in masked])
                if jacc is not None:
                    res[arm]["judge"].append(jacc)
                for r, i in enumerate(masked):
                    records.append({"dataset": name, "fold": fno, "arm": arm, "var": obs[i],
                                    "true_label": labels[obs[i]], "decoded_words": words[r],
                                    "judge": (bool(verd[r]) if verd else None)})
        print(f"[{ts()}]   fold {fno + 1}/{FOLDS} done ({name})", flush=True)
    line = " | ".join(f"{a}: j={np.mean(v['judge']):.3f} m={np.mean(v['match']):.3f}"
                      if v["judge"] else f"{a}: m={np.mean(v['match']):.3f}"
                      for a, v in res.items())
    print(f"[{ts()}] {name:10s} {line}", flush=True)
    out = os.path.join(ROOT, "outputs", f"bridge_{name}.json")
    json.dump({"summary": {a: {k: (float(np.mean(v)) if v else None) for k, v in d_.items()}
                           for a, d_ in res.items()}, "records": records}, open(out, "w"), indent=1)


def main():
    neg = negop.load()
    C, cwords = encode.load_dictionary()
    for n in os.environ.get("DATASETS", "bigfive,rse,mach,sixteenpf,himi,gcbs").split(","):
        run(n.strip(), C, cwords, neg)


if __name__ == "__main__":
    main()
