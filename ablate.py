"""LINE A ablations on the DEV pool (geometric metrics only; judge saved for the final held-out run).
Single-variable discipline: base = ALS+Adam defaults; then free_w; then residual channel over a small
(mu, lam_res) grid; then the combination. All arms share folds, seeds, encoder (current frozen MiniLM).
Output: outputs/ablate.json + printed table."""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import testbeds, pool, encode, metrics, optimize
from run_task1 import ALL_LOADERS

FOLDS = 5
V2 = dict(residual=1.0, lam_res=1.0)                     # frozen v2 config (reference)
ROUND = os.environ.get("ABLATE_ROUND", "v2")
if ROUND == "v3":
    # all-constraints round (user 2026-07-14): shrink = graph-zero blend for the CI anchors;
    # dep = faithfulness dependence floor on trek pairs; coll = explaining-away at v-structures
    ARMS = [
        ("v2", dict(V2)),
        ("v2+shrink", dict(V2, shrink=True)),
        ("v2+dep", dict(V2, lam_dep=0.3)),
        ("v2+coll", dict(V2, lam_coll=0.3)),
        ("v2+dep+shr", dict(V2, shrink=True, lam_dep=0.3)),
        ("v3all", dict(V2, shrink=True, lam_dep=0.3, lam_coll=0.3)),
    ]
else:
    ARMS = [
        ("base", {}),
        ("freew", dict(free_w=True)),
        ("res_.3_1", dict(residual=0.3, lam_res=1.0)),
        ("res_1_1", dict(residual=1.0, lam_res=1.0)),
        ("res_1_.3", dict(residual=1.0, lam_res=0.3)),
        ("fw+res_.3_1", dict(free_w=True, residual=0.3, lam_res=1.0)),
        ("fw+res_1_1", dict(free_w=True, residual=1.0, lam_res=1.0)),
        ("fw+res_1_.3", dict(free_w=True, residual=1.0, lam_res=0.3)),
    ]


def ts():
    return time.strftime("%H:%M:%S")


def main():
    results = {}
    for name in pool.DEV:
        ds = ALL_LOADERS[name]()
        g, X = ds["graph"], ds["X"]
        obs = g.observed
        oi = {o: k for k, o in enumerate(obs)}
        T = encode.embed([ds["labels"][o] for o in obs])
        Tn = metrics.norm_rows(T)
        W, score = g.estimate_weights(X, oi)
        pc = optimize.partial_residual_corr(g, X, oi, score)
        pc_shrunk = (pc[0], optimize.shrink_corr(pc[1], X.shape[0]))
        Craw = np.corrcoef(X.T); np.fill_diagonal(Craw, 0.0)
        dep = (obs, Craw)
        rng = np.random.default_rng(0)
        perm = rng.permutation(len(obs))
        folds = [perm[i::FOLDS] for i in range(FOLDS)]
        results[name] = {}
        for arm, kw in ARMS:
            cs, ms, rr_ = [], [], []
            for fno, fold in enumerate(folds):
                masked = sorted(int(i) for i in fold)
                vis = {obs[i]: T[i] for i in range(len(obs)) if i not in set(masked)}
                kw2 = dict(kw)
                shrink = kw2.pop("shrink", False)
                if kw2.get("residual"):
                    kw2["partial_corr"] = pc_shrunk if shrink else pc
                if kw2.get("lam_dep"):
                    kw2["dep_corr"] = dep
                emb = optimize.optimize_embeddings(g, W, vis, d=T.shape[1], seed=fno, **kw2)
                P = np.stack([emb[obs[i]] for i in masked])
                Pn = metrics.norm_rows(P)
                cs.append(float(np.mean((Pn * Tn[masked]).sum(1))))
                ms.append(metrics.match_acc(P, masked, T))
                S = Pn @ Tn.T
                rr_.append(float(np.mean([1.0 / (1 + int((S[r] > S[r, i]).sum()))
                                          for r, i in enumerate(masked)])))
            results[name][arm] = {"cos": float(np.mean(cs)), "match": float(np.mean(ms)),
                                  "mrr": float(np.mean(rr_))}
            print(f"[{ts()}] {name:9s} {arm:12s} cos={np.mean(cs):.3f} match={np.mean(ms):.3f} "
                  f"mrr={np.mean(rr_):.3f}", flush=True)
    print(f"\n[{ts()}] === dev-pool means ===", flush=True)
    for arm, _ in ARMS:
        m = {k: np.mean([results[n][arm][k] for n in pool.DEV]) for k in ["cos", "match", "mrr"]}
        print(f"  {arm:12s} cos={m['cos']:.3f} match={m['match']:.3f} mrr={m['mrr']:.3f}", flush=True)
    os.makedirs(os.path.join(HERE, "outputs"), exist_ok=True)
    json.dump(results, open(os.path.join(HERE, "outputs", "ablate.json"), "w"), indent=1)
    print("[saved outputs/ablate.json]", flush=True)


if __name__ == "__main__":
    main()
