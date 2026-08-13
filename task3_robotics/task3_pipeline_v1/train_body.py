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
- DATA: 16 dev datasets, folds 0-3 train / fold 4 validation. Held-out never touched.
- SELECTION (revised 2026-07-28 after the first attempt overfit): the canonical pair is the
  epoch with the best mean dev fold-4 MATCH (free Hungarian, the decision metric); the
  embedding-space outer loss is logged but never selects (measured to decouple: val
  .80 -> .22 while held-out match collapsed). EVERY epoch's pair is also saved
  (l2_mlp_v6_ep<N>.pt + gen_operator_ep<N>.pt) so post-hoc screening loses nothing.
- TARGETS: the TRUNK-4a nonlinear stack (nldep: dcor magnitudes, GBR residualization) —
  user order: upgraded BEFORE this retrain.
- CHECKPOINTS (new filenames — outputs/l2_mlp.pt is a SYMLINK to v5 and must never be
  written): canonical pair outputs/l2_mlp_v6.pt + outputs/gen_operator.pt.
  Log: outputs/train_v6_log.json.

ROBOT-BODY VARIANT (train_body.py). Differences from the questionnaire trainer, all deliberate:
  - dev sets are robot body datasets (env DEV_SETS, default liftbody,bodysawyer,bodyiiwa);
    UR5e exists but never enters here, it is the held-out body
  - BOTH modules start FRESH: the operator delta is zero-init (== linear generation) and the
    WeightNet head is zero-init (multipliers exactly 1.0), so the epoch-0 candidate is the
    uniform-weight linear solver, measured and logged before any step
  - no questionnaire checkpoint is loaded anywhere; those were measured to be net-negative on
    robot obs graphs (the monotone strip experiment, 2026-08-06)
  - no f_neg: NEGOP_CKPT is not read; negative edges act by scalar sign

Env: K, INNER_LR, OP_LR, WN_LR, LAM_AUDIT, EPOCHS (4), DEVICE, TORCH_THREADS, DEV_SETS.
"""
import os
import sys
import json
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.abspath(
    os.environ.get("CORE_OUTPUT_DIR", os.path.join(HERE, "outputs"))
)
# The robot-body protocol uses signed physical coefficients, not the questionnaire-trained
# semantic-negation network. Keep the shared module's questionnaire default unchanged elsewhere.
os.environ.setdefault("GENOP_NEGATIVE_MODE", "scalar")
sys.path.insert(0, HERE)
import pool                                                           # noqa: E402
import encode                                                         # noqa: E402
import lora                                                           # noqa: E402
import latent_constraints as LC                                       # noqa: E402
import nldep as NL                                                    # noqa: E402
import terms as TF                                                    # noqa: E402
import gen_operator as GO                                             # noqa: E402
import core                                                           # noqa: E402
import l2_modules as LM                                               # noqa: E402
from run_task1 import ALL_LOADERS                                     # noqa: E402

if os.environ.get("DEPENDENCE_OUT"):
    NL.dep.OUT = os.path.abspath(os.environ["DEPENDENCE_OUT"])

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", 8)))

K = int(os.environ.get("K", 60))
INNER_LR = float(os.environ.get("INNER_LR", 2e-2))
OP_LR = float(os.environ.get("OP_LR", 1e-3))
WN_LR = float(os.environ.get("WN_LR", 3e-4))
LAM_AUDIT = float(os.environ.get("LAM_AUDIT", 0.1))
EPOCHS = int(os.environ.get("EPOCHS", 4))
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
FOLDS = 5
WN_CKPT = os.path.join(ARTIFACT_DIR, "wn_body.pt")
OP_CKPT = os.path.join(ARTIFACT_DIR, "gen_operator_body.pt")
WN_INIT = os.path.join(ARTIFACT_DIR, "l2_mlp.pt")                    # frozen v5 (read-only)
LORA_CKPT = os.environ.get("LORA_CKPT", os.path.join(ARTIFACT_DIR, "l3_lora.pt"))
ENCODER_MODE = os.environ.get("CORE_ENCODER_MODE", "lora").strip().lower()


def ts():
    return time.strftime("%H:%M:%S")


# ---- Encoder for all embeddings (must match inference). ----
if ENCODER_MODE == "lora":
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
elif ENCODER_MODE == "base":
    from sentence_transformers import SentenceTransformer

    encode._MODEL = SentenceTransformer(
        "intfloat/e5-large-v2",
        revision="f169b11e22de13617baa190a028a32f3493550b6",
        device=DEVICE,
        cache_folder=os.environ.get("HF_CACHE"),
        local_files_only=os.environ.get("HF_HUB_OFFLINE", "0") == "1",
    )
else:
    raise ValueError(f"unsupported CORE_ENCODER_MODE={ENCODER_MODE!r}; use 'lora' or 'base'")


def prep(name):
    ds = ALL_LOADERS[name]()
    g, X, labels, gt = ds["graph"], ds["X"], ds["labels"], ds["latent_gt"]
    obs = g.observed
    oi = {o: k for k, o in enumerate(obs)}
    T = encode.embed([labels[o] for o in obs])
    W, score = g.estimate_weights(X, oi)
    W, score = LC.sign_fix(g, W, score)
    # TRUNK-4a nonlinear target stack (user order: BEFORE the retrain)
    mats = NL.matrices(g, X, oi, score, name)
    W = NL.nl_weights(W, mats)
    pc = NL.pc_matrix(mats)
    br = NL.bridge_dict(mats)
    ci = TF.ci_table(g, X, oi, score, nl=mats)   # CI_MODE default marginal_shrink inside
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


def solve_pair(d_, fold, module, gen_op, train, device, want_match=False):
    """-> (outer, audit, match_or_None) for one (dataset, fold). match = the free Hungarian
    match-ACC on the fold's masked observed nodes — the DECISION metric for epoch selection
    (embedding-space outer loss was measured to decouple from decode quality)."""
    obs, T, g, W = d_["obs"], d_["T"], d_["g"], d_["W"]
    masked = sorted(int(i) for i in d_["folds"][fold])
    vis = {obs[i]: T[i] for i in range(len(obs)) if i not in set(masked)}
    feats = torch.tensor(LM.node_features(g, W, set(vis)), device=device)
    emb, tensors = core.solve_unrolled(
        g, W, vis, d=T.shape[1], gen_op=gen_op, ci=d_["ci"], weight_module=module, K=K,
        inner_lr=INNER_LR, lam_zero=0.3, lam_norm=0.1, seed=fold, device=device,
        residual=1.0, lam_res=1.0, partial_corr=d_["pc"], bridge=d_["br"],
        train=train, feats=feats)
    outer = outer_loss(tensors, d_, fold, device)
    vis_t = {n: torch.tensor(v, dtype=torch.float32, device=device) for n, v in vis.items()}
    Xp = torch.stack([tensors[p] if p in tensors else vis_t[p] for p in d_["edge_par"]])
    audit = gen_op.sign_audit(Xp, d_["edge_cond"])
    match = None
    if want_match:
        import metrics
        P = np.stack([emb[obs[i]] for i in masked])
        match = metrics.match_acc(P, masked, T)
    return outer, audit, match


def main():
    names = os.environ.get("DEV_SETS", "liftbody,bodysawyer,bodyiiwa").split(",")
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    print(f"[{ts()}] prep {len(names)} dev datasets ({ENCODER_MODE} E5 space) ...", flush=True)
    data = {}
    for n in names:
        data[n] = prep(n)
        print(f"[{ts()}]   {n}: {len(data[n]['obs'])} obs, {len(data[n]['g'].latents)} latents, "
              f"{sum(len(t) for _, _, t in data[n]['ci'])} ci pairs", flush=True)

    # paired resume: if a previous run saved checkpoints, BOTH modules continue from them;
    # never resume one and reset the other (that trains an inconsistent pair)
    resume = os.path.exists(OP_CKPT) and os.path.exists(WN_CKPT)
    module = LM.load(WN_CKPT, device=DEVICE) if resume else LM.WeightNet().to(DEVICE)
    module.train()
    for p in module.parameters():
        p.requires_grad_(True)
    gen_op = GO.load_or_init(d=data[names[0]]["T"].shape[1], device=DEVICE,
                             path=OP_CKPT if resume else "/nonexistent")
    gen_op.train()
    print(f"[{ts()}] init: " + ("RESUMED body pair" if resume else
          "FRESH pair: WeightNet zero-init head (multipliers==1), operator ZERO-INIT (==linear)"),
          flush=True)
    opt = torch.optim.Adam([
        {"params": gen_op.delta.parameters(), "lr": OP_LR},
        {"params": module.parameters(), "lr": WN_LR},
    ])
    trainable = list(gen_op.delta.parameters()) + list(module.parameters())

    log = {"K": K, "op_lr": OP_LR, "wn_lr": WN_LR, "lam_audit": LAM_AUDIT,
           "encoder_mode": ENCODER_MODE, "dev_sets": names, "epochs": []}

    def val_pass():
        module.eval(); gen_op.eval()
        vo, va, vm = [], [], []
        for n in names:
            o, a, m = solve_pair(data[n], 4, module, gen_op, False, DEVICE, want_match=True)
            vo.append(float(o.detach())); va.append(float(a.detach())); vm.append(float(m))
        module.train(); gen_op.train()
        return vo, va, vm

    vo, va, vm = val_pass()
    log["start_val"] = {"outer": float(np.mean(vo)), "audit": float(np.mean(va)),
                        "match": float(np.mean(vm)),
                        "match_per_ds": dict(zip(names, [round(x, 4) for x in vm]))}
    print(f"[{ts()}] START (epoch-0 state == current v6 inference): "
          f"val_outer={np.mean(vo):.4f} val_audit={np.mean(va):.4f} "
          f"val_MATCH={np.mean(vm):.4f}", flush=True)

    pairs = [(n, f) for n in names for f in range(4)]
    best_match = float(np.mean(vm))            # epoch-0 state is a valid candidate
    LM.save(module, WN_CKPT, "mlp")
    GO.save(gen_op, OP_CKPT)
    for ep in range(EPOCHS):
        rng = np.random.default_rng(100 + ep)
        order = rng.permutation(len(pairs))
        tl, ta = [], []
        for j, pi in enumerate(order):
            n, f = pairs[int(pi)]
            opt.zero_grad()
            outer, audit, _ = solve_pair(data[n], f, module, gen_op, True, DEVICE)
            if not torch.isfinite(outer.detach()):
                print(f"[{ts()}] ep{ep} SKIP {n}/f{f}: non-finite outer", flush=True)
                opt.zero_grad(); continue
            (outer + LAM_AUDIT * audit).backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in trainable):
                print(f"[{ts()}] ep{ep} SKIP {n}/f{f}: non-finite grad", flush=True)
                opt.zero_grad(); continue
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            tl.append(float(outer.detach())); ta.append(float(audit.detach()))
            if j % 8 == 0:
                print(f"[{ts()}] ep{ep} {j}/{len(pairs)} outer={np.mean(tl[-8:]):.4f} "
                      f"audit={np.mean(ta[-8:]):.4f}", flush=True)
        vo, va, vm = val_pass()
        v, m = float(np.mean(vo)), float(np.mean(vm))
        # per-epoch pair always saved (post-hoc screening never loses an epoch)
        LM.save(module, os.path.join(ARTIFACT_DIR, f"wn_body_ep{ep}.pt"), "mlp")
        GO.save(gen_op, os.path.join(ARTIFACT_DIR, f"gen_operator_body_ep{ep}.pt"))
        log["epochs"].append({"train_outer": float(np.mean(tl)), "train_audit": float(np.mean(ta)),
                              "val_outer": v, "val_audit": float(np.mean(va)),
                              "val_match": m,
                              "match_per_ds": dict(zip(names, [round(x, 4) for x in vm]))})
        is_best = m > best_match
        print(f"[{ts()}] EPOCH {ep}: train_outer={np.mean(tl):.4f} val_outer={v:.4f} "
              f"val_MATCH={m:.4f} audit={np.mean(va):.4f} "
              f"{'(best MATCH, canonical pair updated)' if is_best else ''}", flush=True)
        if is_best:                            # canonical pair = best dev fold-4 MATCH
            best_match = m
            LM.save(module, WN_CKPT, "mlp")
            GO.save(gen_op, OP_CKPT)
        json.dump(log, open(os.path.join(ARTIFACT_DIR, "train_body_log.json"), "w"), indent=1)
    print(f"[{ts()}] done. best dev fold-4 MATCH={best_match:.4f} "
          f"(start {log['start_val']['match']:.4f}). Held-out runs ONCE on the canonical pair; "
          f"adoption decided by the user.", flush=True)


if __name__ == "__main__":
    main()
