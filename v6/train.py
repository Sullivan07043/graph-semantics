"""TRUNK-4 trainer — JOINT training of the generation operator and WeightNet on the v6 core.

Replaces the v5 l2_train.py (kept in v5/ as the l2_mlp.pt lineage). Decisions (declared):
- SPACE: LoRA-adapted encoder (l3_lora.pt) for all label/GT embeddings — training happens in
  the same space inference runs in (the v5 WeightNet was trained in base space; that mismatch
  ends here).
- INIT (strict refinement): WeightNet warm-starts from the frozen v5 l2_mlp.pt; the operator's
  Delta head is zero-initialized. Epoch 0 therefore reproduces the current v6 inference
  behavior exactly; training is a continuation, not a restart.
- JOINT optimization, two param groups: operator Delta at OP_LR (default 1e-3, fresh params),
  WeightNet at WN_LR (default 3e-4, continuation). Grad-clip 1.0 on both.
- OBJECTIVE: v5 outer loss (masked observed 1-cos to true labels + 0.5 * latent 1-cos to GT
  names; dev-only supervision) + LAM_AUDIT (default 0.1) * sign_audit on the dataset's edges
  at the solution (THEORY §4.2: the audit is part of the training objective and reported).
- CONFIG matches inference: latcon (sign_fix + augmented anchors/bridge), ci table, K=60,
  residual=1.0, lam_res=1.0, lam_zero=0.3, lam_norm=0.1, bridge(0.3/.5/.7).
- DATA: 16 dev datasets, folds 0-3 train / fold 4 validation (checkpoint selection by val
  OUTER loss; audit logged separately). Held-out never touched.
- CHECKPOINTS (new filenames — outputs/l2_mlp.pt is a SYMLINK to v5 and must never be
  written): outputs/l2_mlp_v6.pt + outputs/gen_operator.pt, saved as a PAIR at the best-val
  epoch. Log: outputs/train_v6_log.json.
- Lesson carried (2026-07 WeightNet retrain failure): validation loss in embedding space is
  NOT a proxy for decode metrics — adoption is decided by the official free MATCH screens and
  the user, never by this val number.

Env: K, INNER_LR, OP_LR, WN_LR, LAM_AUDIT, EPOCHS (4), DEVICE, TORCH_THREADS.
"""
import os
import sys
import json
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pool                                                           # noqa: E402
import encode                                                         # noqa: E402
import optimize                                                       # noqa: E402
import lora                                                           # noqa: E402
import dependence as depmod                                           # noqa: E402
import latent_constraints as LC                                       # noqa: E402
import terms as TF                                                    # noqa: E402
import gen_operator as GO                                             # noqa: E402
import core                                                           # noqa: E402
import l2_modules as LM                                               # noqa: E402
from run_task1 import ALL_LOADERS                                     # noqa: E402

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", 8)))

K = int(os.environ.get("K", 60))
INNER_LR = float(os.environ.get("INNER_LR", 2e-2))
OP_LR = float(os.environ.get("OP_LR", 1e-3))
WN_LR = float(os.environ.get("WN_LR", 3e-4))
LAM_AUDIT = float(os.environ.get("LAM_AUDIT", 0.1))
EPOCHS = int(os.environ.get("EPOCHS", 4))
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
FOLDS = 5
WN_CKPT = os.path.join(HERE, "outputs", "l2_mlp_v6.pt")
OP_CKPT = os.path.join(HERE, "outputs", "gen_operator.pt")
WN_INIT = os.path.join(HERE, "outputs", "l2_mlp.pt")                  # frozen v5 (read-only)
LORA_CKPT = os.path.join(HERE, "outputs", "l3_lora.pt")


def ts():
    return time.strftime("%H:%M:%S")


# ---- LoRA encoder for all embeddings (same space as inference; main.py wiring) ----
_st = lora.load_st(DEVICE)
lora.inject(_st)
lora.load_lora(_st, LORA_CKPT)
_st.eval()


class _LoraST:
    def encode(self, texts, batch_size=1024, normalize_embeddings=True):
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), 256):
                stripped = [t[len("query: "):] if t.startswith("query: ") else t
                            for t in texts[i:i + 256]]
                out.append(lora.encode_grad(_st, stripped, DEVICE, max_len=128).cpu().numpy())
        return np.concatenate(out)


encode._MODEL = _LoraST()


def prep(name):
    ds = ALL_LOADERS[name]()
    g, X, labels, gt = ds["graph"], ds["X"], ds["labels"], ds["latent_gt"]
    obs = g.observed
    oi = {o: k for k, o in enumerate(obs)}
    T = encode.embed([labels[o] for o in obs])
    W, score = g.estimate_weights(X, oi)
    W, score = LC.sign_fix(g, W, score)
    pc = optimize.partial_residual_corr(g, X, oi, score)
    pc = LC.augmented_partial_corr(g, X, oi, score, pc)
    bn, bD = LC.augmented_bridge(g, list(obs), oi, X, score,
                                 depmod.load(name, "marginal", "pearson"))
    br = dict(obs=bn, dep_marg=bD, lam_upper=0.3, kappa=0.5, q=0.7)
    ci = TF.ci_table(g, X, oi, score)
    edge_par, edge_cond, _ = GO.edge_table(g, W, device=DEVICE)
    lat_names = [L for L in g.latents if L in gt]
    G = encode.embed([gt[L] for L in lat_names]) if lat_names else None
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(obs))
    folds = [perm[i::FOLDS] for i in range(FOLDS)]
    return dict(name=name, g=g, obs=obs, T=T, W=W, pc=pc, br=br, ci=ci,
                edge_par=edge_par, edge_cond=edge_cond,
                lat_names=lat_names, G=G, folds=folds)


def outer_loss(tensors, d_, fold, device):
    obs, T = d_["obs"], d_["T"]
    masked = sorted(int(i) for i in d_["folds"][fold])
    terms_ = []
    for i in masked:
        n = obs[i]
        if n in tensors:
            t = torch.tensor(T[i], dtype=torch.float32, device=device)
            terms_.append(1 - torch.nn.functional.cosine_similarity(tensors[n], t, dim=0))
    lt = []
    if d_["G"] is not None:
        for k, L in enumerate(d_["lat_names"]):
            if L in tensors:
                t = torch.tensor(d_["G"][k], dtype=torch.float32, device=device)
                lt.append(1 - torch.nn.functional.cosine_similarity(tensors[L], t, dim=0))
    lo = torch.stack(terms_).mean()
    if lt:
        lo = lo + 0.5 * torch.stack(lt).mean()
    return lo


def solve_pair(d_, fold, module, gen_op, train, device):
    """-> (outer, audit) losses for one (dataset, fold)."""
    obs, T, g, W = d_["obs"], d_["T"], d_["g"], d_["W"]
    masked = set(int(i) for i in d_["folds"][fold])
    vis = {obs[i]: T[i] for i in range(len(obs)) if i not in masked}
    feats = torch.tensor(LM.node_features(g, W, set(vis)), device=device)
    _, tensors = core.solve_unrolled(
        g, W, vis, d=T.shape[1], gen_op=gen_op, ci=d_["ci"], weight_module=module, K=K,
        inner_lr=INNER_LR, lam_zero=0.3, lam_norm=0.1, seed=fold, device=device,
        residual=1.0, lam_res=1.0, partial_corr=d_["pc"], bridge=d_["br"],
        train=train, feats=feats)
    outer = outer_loss(tensors, d_, fold, device)
    vis_t = {n: torch.tensor(v, dtype=torch.float32, device=device) for n, v in vis.items()}
    Xp = torch.stack([tensors[p] if p in tensors else vis_t[p] for p in d_["edge_par"]])
    audit = gen_op.sign_audit(Xp, d_["edge_cond"])
    return outer, audit


def main():
    names = list(pool.DEV)
    print(f"[{ts()}] prep {len(names)} dev datasets (LoRA space) ...", flush=True)
    data = {}
    for n in names:
        data[n] = prep(n)
        print(f"[{ts()}]   {n}: {len(data[n]['obs'])} obs, {len(data[n]['g'].latents)} latents, "
              f"{sum(len(t) for _, _, t in data[n]['ci'])} ci pairs", flush=True)

    # paired resume: if a previous run saved checkpoints, BOTH modules continue from them;
    # never resume one and reset the other (that trains an inconsistent pair)
    resume = os.path.exists(OP_CKPT) and os.path.exists(WN_CKPT)
    module = LM.load(WN_CKPT if resume else WN_INIT, device=DEVICE)
    module.train()
    for p in module.parameters():
        p.requires_grad_(True)
    gen_op = GO.load_or_init(d=data[names[0]]["T"].shape[1], device=DEVICE,
                             path=OP_CKPT if resume else "/nonexistent")
    gen_op.train()
    print(f"[{ts()}] init: " + ("RESUMED pair (l2_mlp_v6.pt + gen_operator.pt)" if resume else
          "WeightNet <- l2_mlp.pt (v5 frozen), operator ZERO-INIT (== v5 linear+f_neg)"),
          flush=True)
    opt = torch.optim.Adam([
        {"params": gen_op.delta.parameters(), "lr": OP_LR},
        {"params": module.parameters(), "lr": WN_LR},
    ])
    trainable = list(gen_op.delta.parameters()) + list(module.parameters())

    log = {"K": K, "op_lr": OP_LR, "wn_lr": WN_LR, "lam_audit": LAM_AUDIT, "epochs": []}

    def val_pass():
        module.eval(); gen_op.eval()
        vo, va = [], []
        for n in names:
            o, a = solve_pair(data[n], 4, module, gen_op, False, DEVICE)
            vo.append(float(o.detach())); va.append(float(a.detach()))
        module.train(); gen_op.train()
        return vo, va

    vo, va = val_pass()
    log["start_val"] = {"outer": float(np.mean(vo)), "audit": float(np.mean(va)),
                        "per_ds": dict(zip(names, [round(x, 4) for x in vo]))}
    print(f"[{ts()}] START (epoch-0 state == current v6 inference): "
          f"val_outer={np.mean(vo):.4f} val_audit={np.mean(va):.4f}", flush=True)

    pairs = [(n, f) for n in names for f in range(4)]
    best = float("inf")
    for ep in range(EPOCHS):
        rng = np.random.default_rng(100 + ep)
        order = rng.permutation(len(pairs))
        tl, ta = [], []
        for j, pi in enumerate(order):
            n, f = pairs[int(pi)]
            opt.zero_grad()
            outer, audit = solve_pair(data[n], f, module, gen_op, True, DEVICE)
            (outer + LAM_AUDIT * audit).backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            tl.append(float(outer.detach())); ta.append(float(audit.detach()))
            if j % 8 == 0:
                print(f"[{ts()}] ep{ep} {j}/{len(pairs)} outer={np.mean(tl[-8:]):.4f} "
                      f"audit={np.mean(ta[-8:]):.4f}", flush=True)
        vo, va = val_pass()
        v = float(np.mean(vo))
        log["epochs"].append({"train_outer": float(np.mean(tl)), "train_audit": float(np.mean(ta)),
                              "val_outer": v, "val_audit": float(np.mean(va)),
                              "val_per_ds": dict(zip(names, [round(x, 4) for x in vo]))})
        print(f"[{ts()}] EPOCH {ep}: train_outer={np.mean(tl):.4f} val_outer={v:.4f} "
              f"val_audit={np.mean(va):.4f} {'(best, saved pair)' if v < best else ''}",
              flush=True)
        if v < best:
            best = v
            LM.save(module, WN_CKPT, "mlp")
            GO.save(gen_op, OP_CKPT)
        json.dump(log, open(os.path.join(HERE, "outputs", "train_v6_log.json"), "w"), indent=1)
    print(f"[{ts()}] done. best val_outer={best:.4f} (start {log['start_val']['outer']:.4f}). "
          f"Adoption is decided by official free MATCH screens + user, not by this number.",
          flush=True)


if __name__ == "__main__":
    main()
