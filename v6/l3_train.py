"""L3 training: shape the encoder's space (via LoRA, lora.py) so cosine matches the bridge axiom.

Losses per (dataset, fold) bundle, all on LoRA-encoded label texts:
  bridge   — strongly dependent trek-connected visible pairs keep |cos| >= kappa*dep (hinge^2)
  indep    — pairs d-separated by the empty set get cos^2 -> 0
  neg      — reverse-keyed item ~ f_neg(latent GT text embedding) (f_neg frozen)
  anchor   — 20k dictionary words must NOT move: hinge(0.99 - cos(h', ref))^2, ref = the frozen
             dictionary embeddings (heaviest weight; the v3 drift lesson)
Fold discipline: folds 0-3 of each of the 16 dev sets train, fold 4 validates. Held-out dataset
texts NEVER enter training. A held-back 2k anchor words (never trained on) monitor drift.
No pass/fail thresholds (user ruling 2026-07-16): this script reports facts; adoption is the
user's call on the full comparison table.
Env: EPOCHS(3) LR(1e-4) W_ANCHOR(10) W_BRIDGE(1) W_INDEP(0.3) W_NEG(1) DEVICE
Output: outputs/l3_lora.pt (best fold-4 val constraint loss) + outputs/l3_trainlog.json
"""
import os
import sys
import json
import time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pool, encode, negop                                            # noqa: E402
import dependence as depmod                                           # noqa: E402
from run_task1 import ALL_LOADERS                                     # noqa: E402
import lora                                          # noqa: E402

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", 8)))
DEVICE = os.environ.get("DEVICE", "cuda")
EPOCHS = int(os.environ.get("EPOCHS", 3))
LR = float(os.environ.get("LR", 1e-4))
W = dict(anchor=float(os.environ.get("W_ANCHOR", 10.0)), bridge=float(os.environ.get("W_BRIDGE", 1.0)),
         indep=float(os.environ.get("W_INDEP", 0.3)), neg=float(os.environ.get("W_NEG", 1.0)))
FOLDS, KAPPA, Q = 5, 0.5, 0.7


def ts():
    return time.strftime("%H:%M:%S")


def prep(name):
    ds = ALL_LOADERS[name]()
    g, X, labels, gt = ds["graph"], ds["X"], ds["labels"], ds["latent_gt"]
    obs = list(g.observed)
    oi = {o: k for k, o in enumerate(obs)}
    W_, _ = g.estimate_weights(X, oi)
    Dm = np.asarray(depmod.load(name, "marginal", "pearson"))
    tp = [(a, b) for a, b in g.trek_pairs() if a in oi and b in oi]
    vals = np.array([Dm[oi[a], oi[b]] for a, b in tp]) if tp else np.array([])
    thr = np.quantile(vals, Q) if len(vals) else 1e9
    bridge_pairs = [(a, b, float(v)) for (a, b), v in zip(tp, vals) if v >= thr]
    indep_pairs = [(a, b) for a, b in g.independent_pairs() if a in oi and b in oi]
    neg_edges = [(p, c) for (p, c), w in W_.items() if w < 0 and g.is_latent(p) and p in gt and c in oi]
    rng = np.random.default_rng(0)
    folds = [rng.permutation(len(obs))[i::FOLDS] for i in range(FOLDS)]
    return dict(name=name, obs=obs, oi=oi, labels=labels, gt=gt, folds=folds,
                bridge=bridge_pairs, indep=indep_pairs, neg=neg_edges)


def bundle_loss(st, d_, fold, NEG, anchors, aref, device):
    masked = set(int(i) for i in d_["folds"][fold])
    vis = [o for i, o in enumerate(d_["obs"]) if i not in masked]
    vset = set(vis)
    lat = sorted({p for p, c in d_["neg"]})
    texts = [d_["labels"][o] for o in vis] + [d_["gt"][L] for L in lat]
    idx = {o: k for k, o in enumerate(vis)}
    lidx = {L: len(vis) + k for k, L in enumerate(lat)}
    H = lora.encode_grad(st, texts, device)
    losses = {}
    bp = [(idx[a], idx[b], v) for a, b, v in d_["bridge"] if a in vset and b in vset]
    if bp:
        ia = torch.tensor([x[0] for x in bp], device=device)
        ib = torch.tensor([x[1] for x in bp], device=device)
        fl = torch.tensor([KAPPA * x[2] for x in bp], dtype=torch.float32, device=device)
        cs = (H[ia] * H[ib]).sum(1).abs()
        losses["bridge"] = (torch.relu(fl - cs) ** 2).mean()
    ip = [(idx[a], idx[b]) for a, b in d_["indep"] if a in vset and b in vset]
    if ip:
        ia = torch.tensor([x[0] for x in ip], device=device)
        ib = torch.tensor([x[1] for x in ip], device=device)
        losses["indep"] = (((H[ia] * H[ib]).sum(1)) ** 2).mean()
    np_ = [(lidx[p], idx[c]) for p, c in d_["neg"] if c in vset]
    if np_:
        il = torch.tensor([x[0] for x in np_], device=device)
        ic = torch.tensor([x[1] for x in np_], device=device)
        tgt = torch.nn.functional.normalize(NEG(H[il]), dim=1)
        losses["neg"] = (1 - (H[ic] * tgt).sum(1)).mean()
    # anchors: random 256 of the training anchor pool
    sel = np.random.default_rng(abs(hash((d_["name"], fold))) % 2**31).choice(len(anchors), 256, replace=False)
    Ha = lora.encode_grad(st, [anchors[i] for i in sel], device)
    ra = torch.tensor(aref[sel], dtype=torch.float32, device=device)
    losses["anchor"] = (torch.relu(0.99 - (Ha * ra).sum(1)) ** 2).mean() * 100.0
    total = sum(W[k.replace("anchor", "anchor")] * v if k != "anchor" else W["anchor"] * v
                for k, v in losses.items())
    return total, {k: float(v.detach()) for k, v in losses.items()}


def main():
    st = lora.load_st(DEVICE)
    params = lora.inject(st)
    NEG = negop.load().to(DEVICE)
    for p in NEG.parameters():
        p.requires_grad_(False)
    C, cwords = encode.load_dictionary()
    rng = np.random.default_rng(7)
    pick = rng.choice(len(cwords), 22000, replace=False)
    anchors = [cwords[i] for i in pick[:20000]]
    aref = C[pick[:20000]].astype(np.float32)                        # frozen reference = dict rows
    drift_words = [cwords[i] for i in pick[20000:]]                  # never trained on
    dref = C[pick[20000:]].astype(np.float32)
    names = list(pool.DEV)
    data = {n: prep(n) for n in names}
    print(f"[{ts()}] prepped {len(names)} dev sets; trainable params "
          f"{sum(p.numel() for p in params)}", flush=True)
    opt = torch.optim.Adam(params, lr=LR)
    pairs = [(n, f) for n in names for f in range(4)]
    log = {"epochs": []}
    best = float("inf")

    def drift():
        with torch.no_grad():
            out = []
            for i in range(0, len(drift_words), 512):
                out.append(lora.encode_grad(st, drift_words[i:i + 512], DEVICE).cpu().numpy())
            Hd = np.concatenate(out)
        cs = (Hd * dref).sum(1)
        return float(cs.mean()), float(cs.min())

    for ep in range(EPOCHS):
        order = np.random.default_rng(100 + ep).permutation(len(pairs))
        tl = []
        for j, pi in enumerate(order):
            n, f = pairs[int(pi)]
            opt.zero_grad()
            lo, parts = bundle_loss(st, data[n], f, NEG, anchors, aref, DEVICE)
            lo.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            tl.append(float(lo.detach()))
            if j % 16 == 0:
                print(f"[{ts()}] ep{ep} {j}/{len(pairs)} loss={np.mean(tl[-16:]):.4f} {parts}",
                      flush=True)
        vl = []
        for n in names:
            with torch.no_grad():
                lo, _ = bundle_loss(st, data[n], 4, NEG, anchors, aref, DEVICE)
            vl.append(float(lo))
        v = float(np.mean(vl))
        dm, dmin = drift()
        log["epochs"].append({"train": float(np.mean(tl)), "val": v,
                              "drift_mean_cos": dm, "drift_min_cos": dmin})
        print(f"[{ts()}] EPOCH {ep}: train={np.mean(tl):.4f} val={v:.4f} "
              f"drift(mean/min cos vs frozen)={dm:.4f}/{dmin:.4f}"
              f" {'(best, saved)' if v < best else ''}", flush=True)
        if v < best:
            best = v
            torch.save({"state": lora.lora_state(st), "r": lora.R, "alpha": lora.ALPHA,
                        "layers": lora.N_LAYERS},
                       os.path.join(HERE, "outputs", "l3_lora.pt"))
        json.dump(log, open(os.path.join(HERE, "outputs", "l3_trainlog.json"), "w"), indent=1)
    print(f"[{ts()}] done. best val={best:.4f}", flush=True)


if __name__ == "__main__":
    main()
