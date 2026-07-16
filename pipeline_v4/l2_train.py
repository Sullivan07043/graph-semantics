"""L2 outer training: learn constraint-weight multipliers by unrolling the solver on dev graphs.

Data: the 16 dev datasets (pool.DEV). Held-out (hexaco/riasec/kims) NEVER touched.
Split: folds 0-3 of each dataset train, fold 4 validates (checkpoint selection).
Outer loss per (dataset, fold): masked observed nodes' 1-cos to their true label embeddings
+ 0.5 * latents' 1-cos to their GT-name embeddings (latent GT supervision on dev is allowed,
meeting 2026-07-02 note 3).

Arms: ARM=mlp (WeightNet, main) | static (StaticWeights, attribution control).
Env: ARM, K (default 60), INNER_LR (2e-2), OUTER_LR (1e-3), EPOCHS (4), DEVICE (cuda if avail).
Output: outputs/l2_<arm>.pt (best-val checkpoint) + outputs/l2_<arm>_trainlog.json
"""
import os
import sys
import json
import time
import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "pipeline_v3"))
import pool, encode, optimize, negop                                  # noqa: E402
import dependence as depmod                                           # noqa: E402
from run_task1 import ALL_LOADERS                                     # noqa: E402
from pipeline_v4 import core                                          # noqa: E402
from pipeline_v4 import l2_modules as LM                              # noqa: E402

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", 4)))

ARM = os.environ.get("ARM", "mlp")
K = int(os.environ.get("K", 60))
INNER_LR = float(os.environ.get("INNER_LR", 2e-2))
OUTER_LR = float(os.environ.get("OUTER_LR", 1e-3))
EPOCHS = int(os.environ.get("EPOCHS", 4))
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
FOLDS = 5


def ts():
    return time.strftime("%H:%M:%S")


def prep(name):
    ds = ALL_LOADERS[name]()
    g, X, labels, gt = ds["graph"], ds["X"], ds["labels"], ds["latent_gt"]
    obs = g.observed
    oi = {o: k for k, o in enumerate(obs)}
    T = encode.embed([labels[o] for o in obs])
    W, score = g.estimate_weights(X, oi)
    pc = optimize.partial_residual_corr(g, X, oi, score)
    br = dict(obs=list(obs), dep_marg=depmod.load(name, "marginal", "pearson"),
              lam_upper=0.3, kappa=0.5, q=0.7)
    lat_names = [L for L in g.latents if L in gt]
    G = encode.embed([gt[L] for L in lat_names]) if lat_names else None
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(obs))
    folds = [perm[i::FOLDS] for i in range(FOLDS)]
    return dict(name=name, g=g, obs=obs, T=T, W=W, pc=pc, br=br,
                lat_names=lat_names, G=G, folds=folds)


def outer_loss(P, tensors, d_, fold, device):
    """1 - cos of masked observed to truth (+ 0.5 * latent GT term)."""
    obs, T = d_["obs"], d_["T"]
    masked = sorted(int(i) for i in d_["folds"][fold])
    terms = []
    for i in masked:
        n = obs[i]
        if n in tensors:
            t = torch.tensor(T[i], dtype=torch.float32, device=device)
            terms.append(1 - torch.nn.functional.cosine_similarity(tensors[n], t, dim=0))
    lt = []
    if d_["G"] is not None:
        for k, L in enumerate(d_["lat_names"]):
            if L in tensors:
                t = torch.tensor(d_["G"][k], dtype=torch.float32, device=device)
                lt.append(1 - torch.nn.functional.cosine_similarity(tensors[L], t, dim=0))
    lo = torch.stack(terms).mean()
    if lt:
        lo = lo + 0.5 * torch.stack(lt).mean()
    return lo


def solve_pair(d_, fold, module, NEG, train, device):
    obs, T, g, W = d_["obs"], d_["T"], d_["g"], d_["W"]
    masked = set(int(i) for i in d_["folds"][fold])
    vis = {obs[i]: T[i] for i in range(len(obs)) if i not in masked}
    feats = torch.tensor(LM.node_features(g, W, set(vis)), device=device)
    _, tensors = core.solve_unrolled(
        g, W, vis, d=T.shape[1], weight_module=module, K=K, inner_lr=INNER_LR,
        seed=fold, device=device, residual=1.0, lam_res=1.0, partial_corr=d_["pc"],
        neg_op=NEG, bridge=d_["br"], train=train, feats=feats)
    return outer_loss(None, tensors, d_, fold, device)


def main():
    NEG = negop.load().to(DEVICE)
    for p in NEG.parameters():
        p.requires_grad_(False)
    names = list(pool.DEV)
    print(f"[{ts()}] prep {len(names)} dev datasets ...", flush=True)
    data = {}
    for n in names:
        data[n] = prep(n)
        print(f"[{ts()}]   {n}: {len(data[n]['obs'])} obs, {len(data[n]['g'].latents)} latents",
              flush=True)
    module = LM.WeightNet() if ARM == "mlp" else LM.StaticWeights()
    module.to(DEVICE).train()
    opt = torch.optim.Adam(module.parameters(), lr=OUTER_LR)
    pairs = [(n, f) for n in names for f in range(4)]
    log = {"arm": ARM, "K": K, "epochs": []}
    best = float("inf")
    ckpt = os.path.join(HERE, "outputs", f"l2_{ARM}.pt")
    for ep in range(EPOCHS):
        rng = np.random.default_rng(100 + ep)
        order = rng.permutation(len(pairs))
        tl = []
        for j, pi in enumerate(order):
            n, f = pairs[int(pi)]
            opt.zero_grad()
            lo = solve_pair(data[n], f, module, NEG, True, DEVICE)
            lo.backward()
            torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
            opt.step()
            tl.append(float(lo.detach()))
            if j % 16 == 0:
                print(f"[{ts()}] ep{ep} {j}/{len(pairs)} outer_loss={np.mean(tl[-16:]):.4f}",
                      flush=True)
        module.eval()
        # no torch.no_grad(): the inner solve needs grad mode; train=False already detaches per step
        vl = [float(solve_pair(data[n], 4, module, NEG, False, DEVICE).detach()) for n in names]
        module.train()
        v = float(np.mean(vl))
        log["epochs"].append({"train": float(np.mean(tl)), "val": v,
                              "val_per_ds": dict(zip(names, [round(x, 4) for x in vl]))})
        print(f"[{ts()}] EPOCH {ep}: train={np.mean(tl):.4f} val={v:.4f}"
              f" {'(best, saved)' if v < best else ''}", flush=True)
        if v < best:
            best = v
            LM.save(module, ckpt, "static" if ARM == "static" else "mlp")
        json.dump(log, open(os.path.join(HERE, "outputs", f"l2_{ARM}_trainlog.json"), "w"), indent=1)
    # baseline val (multipliers=1, same solver dynamics) for reference
    bl = [float(solve_pair(data[n], 4, None, NEG, False, DEVICE).detach()) for n in names]
    log["baseline_val_mult1"] = float(np.mean(bl))
    json.dump(log, open(os.path.join(HERE, "outputs", f"l2_{ARM}_trainlog.json"), "w"), indent=1)
    print(f"[{ts()}] done. best val={best:.4f} vs mult=1 val={np.mean(bl):.4f}", flush=True)


if __name__ == "__main__":
    main()
