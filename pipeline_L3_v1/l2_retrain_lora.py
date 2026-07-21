"""Retrain WeightNet IN THE L3 LoRA SPACE with a larger unroll budget (fixes two known debts):
  1. the adopted main line reused l2_mlp.pt trained on FROZEN-space embeddings — space mismatch;
  2. K=60 binds on the deepest graphs (mult=1 control lost .05 vs 400 steps).

Solver here: ALS init + K=200 functional-Adam steps, gradients truncated to the last TRUNC=60
steps (first K-TRUNC run detached — constant memory, full-length dynamics).
Labels and latent GT texts are encoded ONCE through the LoRA encoder (frozen during this
training); everything else (folds, losses, controls, held-out purity) matches pipeline_v4/l2_train.
Output: outputs/l2_mlp_lora.pt + outputs/l2_mlp_lora_trainlog.json
Env: K(200) TRUNC(60) EPOCHS(4) OUTER_LR(1e-3) DEVICE
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
import pool, optimize, negop                                          # noqa: E402
import dependence as depmod                                           # noqa: E402
from run_task1 import ALL_LOADERS                                     # noqa: E402
from pipeline_v4 import core, l2_modules as LM                        # noqa: E402
from pipeline_L3_v1 import lora                                       # noqa: E402

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", 8)))
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
K = int(os.environ.get("K", 200))
TRUNC = int(os.environ.get("TRUNC", 60))
EPOCHS = int(os.environ.get("EPOCHS", 4))
OUTER_LR = float(os.environ.get("OUTER_LR", 1e-3))
INNER_LR = 2e-2
FOLDS = 5


def ts():
    return time.strftime("%H:%M:%S")


_ST = None


def lora_embed(texts):
    global _ST
    if _ST is None:
        _ST = lora.load_st(DEVICE)
        lora.inject(_ST)
        lora.load_lora(_ST, os.path.join(HERE, "outputs", "l3_lora.pt"))
        _ST.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 256):
            out.append(lora.encode_grad(_ST, texts[i:i + 256], DEVICE, max_len=128).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def prep(name):
    ds = ALL_LOADERS[name]()
    g, X, labels, gt = ds["graph"], ds["X"], ds["labels"], ds["latent_gt"]
    obs = g.observed
    oi = {o: k for k, o in enumerate(obs)}
    T = lora_embed([labels[o] for o in obs])
    W, score = g.estimate_weights(X, oi)
    pc = optimize.partial_residual_corr(g, X, oi, score)
    br = dict(obs=list(obs), dep_marg=depmod.load(name, "marginal", "pearson"),
              lam_upper=0.3, kappa=0.5, q=0.7)
    lat_names = [L for L in g.latents if L in gt]
    G = lora_embed([gt[L] for L in lat_names]) if lat_names else None
    rng = np.random.default_rng(0)
    folds = [rng.permutation(len(obs))[i::FOLDS] for i in range(FOLDS)]
    return dict(name=name, g=g, obs=obs, T=T, W=W, pc=pc, br=br,
                lat_names=lat_names, G=G, folds=folds)


def solve_trunc(g, W, labeled_emb, d, module, feats, seed, pc, br, NEG, train):
    """ALS init + K steps; grad only through the last TRUNC steps (train=True)."""
    torch.manual_seed(seed)
    free, A, E0 = core._stage1(g, W, labeled_emb, d)
    ctx = core.build_ctx(g, W, dict(W), A, free, d, seed, DEVICE, 1.0, 1.0, pc, NEG, br)
    nw = module(feats, ctx) if module is not None else None
    P = {n: torch.tensor(E0[n], dtype=torch.float32, device=DEVICE) for n in free}
    Rv = {n: torch.tensor(v, dtype=torch.float32, device=DEVICE)
          for n, v in ctx.get("Rv0", {}).items()} if ctx["use_res"] else None
    wt_const = ctx["wt_const"]

    def wt(e):
        return wt_const[e]

    At = ctx["At"]
    b1, b2, eps = 0.9, 0.999, 1e-8
    ps = [P[n] for n in P] + ([Rv[n] for n in Rv] if Rv else [])
    m = [torch.zeros_like(p) for p in ps]
    v = [torch.zeros_like(p) for p in ps]
    for p in ps:
        p.requires_grad_(True)
    for step in range(1, K + 1):
        live = train and step > K - TRUNC

        def emb(n):
            return At[n] if n in At else P[n]
        loss = core.step_loss(ctx, emb, wt, P, Rv, 0.3, 0.1, nw=nw)
        grads = torch.autograd.grad(loss, ps, create_graph=live)
        new_ps = []
        for i, (p, gr) in enumerate(zip(ps, grads)):
            m[i] = b1 * m[i] + (1 - b1) * gr
            v[i] = b2 * v[i] + (1 - b2) * gr * gr
            mh = m[i] / (1 - b1 ** step)
            vh = v[i] / (1 - b2 ** step)
            new_ps.append(p - INNER_LR * mh / (vh.sqrt() + eps))
        ps = new_ps
        if not live:
            ps = [p.detach().requires_grad_(True) for p in ps]
            m = [t.detach() for t in m]
            v = [t.detach() for t in v]
        k = 0
        for n in list(P.keys()):
            P[n] = ps[k]; k += 1
        if Rv is not None:
            for n in list(Rv.keys()):
                Rv[n] = ps[k]; k += 1
    return P


def pair_loss(d_, fold, module, NEG, train):
    obs, T, g, W = d_["obs"], d_["T"], d_["g"], d_["W"]
    masked = sorted(int(i) for i in d_["folds"][fold])
    mset = set(masked)
    vis = {obs[i]: T[i] for i in range(len(obs)) if i not in mset}
    feats = torch.tensor(LM.node_features(g, W, set(vis)), device=DEVICE)
    P = solve_trunc(g, W, vis, T.shape[1], module, feats, fold, d_["pc"], d_["br"], NEG, train)
    terms = []
    for i in masked:
        n = obs[i]
        if n in P:
            t = torch.tensor(T[i], dtype=torch.float32, device=DEVICE)
            terms.append(1 - torch.nn.functional.cosine_similarity(P[n], t, dim=0))
    lt = []
    if d_["G"] is not None:
        for k, L in enumerate(d_["lat_names"]):
            if L in P:
                t = torch.tensor(d_["G"][k], dtype=torch.float32, device=DEVICE)
                lt.append(1 - torch.nn.functional.cosine_similarity(P[L], t, dim=0))
    lo = torch.stack(terms).mean()
    if lt:
        lo = lo + 0.5 * torch.stack(lt).mean()
    return lo


def main():
    NEG = negop.load().to(DEVICE)
    for p in NEG.parameters():
        p.requires_grad_(False)
    names = list(pool.DEV)
    data = {}
    for n in names:
        data[n] = prep(n)
        print(f"[{ts()}]   {n} prepped", flush=True)
    module = LM.WeightNet().to(DEVICE).train()
    opt = torch.optim.Adam(module.parameters(), lr=OUTER_LR)
    pairs = [(n, f) for n in names for f in range(4)]
    log = {"K": K, "TRUNC": TRUNC, "space": "l3_lora", "epochs": []}
    best = float("inf")
    ckpt = os.path.join(HERE, "outputs", "l2_mlp_lora.pt")
    for ep in range(EPOCHS):
        order = np.random.default_rng(100 + ep).permutation(len(pairs))
        tl = []
        for j, pi in enumerate(order):
            n, f = pairs[int(pi)]
            opt.zero_grad()
            lo = pair_loss(data[n], f, module, NEG, True)
            lo.backward()
            torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
            opt.step()
            tl.append(float(lo.detach()))
            if j % 16 == 0:
                print(f"[{ts()}] ep{ep} {j}/{len(pairs)} outer_loss={np.mean(tl[-16:]):.4f}",
                      flush=True)
        vl = [float(pair_loss(data[n], 4, module, NEG, False).detach()) for n in names]
        v = float(np.mean(vl))
        log["epochs"].append({"train": float(np.mean(tl)), "val": v})
        print(f"[{ts()}] EPOCH {ep}: train={np.mean(tl):.4f} val={v:.4f}"
              f" {'(best, saved)' if v < best else ''}", flush=True)
        if v < best:
            best = v
            LM.save(module, ckpt, "mlp")
        json.dump(log, open(os.path.join(HERE, "outputs", "l2_mlp_lora_trainlog.json"), "w"),
                  indent=1)
    # references at the same val protocol: K200 no-learning, and the frozen-space WeightNet
    bl = [float(pair_loss(data[n], 4, None, NEG, False).detach()) for n in names]
    old = LM.load(os.path.join(HERE, "outputs", "l2_mlp.pt"), DEVICE)
    ol = [float(pair_loss(data[n], 4, old, NEG, False).detach()) for n in names]
    log["baseline_val_mult1_K200"] = float(np.mean(bl))
    log["baseline_val_frozen_space_weightnet"] = float(np.mean(ol))
    json.dump(log, open(os.path.join(HERE, "outputs", "l2_mlp_lora_trainlog.json"), "w"), indent=1)
    print(f"[{ts()}] done. best={best:.4f} | mult1@K200={np.mean(bl):.4f} | "
          f"old-frozen-space-WeightNet={np.mean(ol):.4f}", flush=True)


if __name__ == "__main__":
    main()
