"""Train L2 constraint multipliers in the final L3 embedding space.

The training order is enforced: this script requires a versioned final L3 checkpoint, installs
that adapted encoder behind ``encode.embed``, and records its SHA-256 identity in the L2
checkpoint.  The default solve is K=120 with 2x60 truncated backpropagation.
"""
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "pipeline_v3"))
import encode
import negop
import optimize
import pool
import dependence as depmod
from pipeline_L3_v1 import lora
from pipeline_v4 import core
from pipeline_v4 import l2_modules as LM
from pipeline_v4 import release
from run_task1 import ALL_LOADERS

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", 4)))

ARM = os.environ.get("ARM", "mlp")
K = int(os.environ.get("K", release.SOLVER_STEPS))
TRUNCATION_STEPS = release.TRUNCATION_STEPS
INNER_LR = float(os.environ.get("INNER_LR", 2e-2))
OUTER_LR = float(os.environ.get("OUTER_LR", 1e-3))
EPOCHS = int(os.environ.get("EPOCHS", 4))
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
FOLDS = 5
L3_CKPT = os.environ.get(
    "L3_CKPT", os.path.join(HERE, "outputs", release.L3_CHECKPOINT_NAME))
L2_OUT = os.environ.get(
    "L2_OUT", os.path.join(HERE, "outputs", release.l2_checkpoint_name(ARM)))
L2_LOG = os.environ.get(
    "L2_LOG", os.path.join(HERE, "outputs", release.l2_trainlog_name(ARM)))


def ts():
    return time.strftime("%H:%M:%S")


def prep(name):
    ds = ALL_LOADERS[name]()
    g, X, labels, gt = ds["graph"], ds["X"], ds["labels"], ds["latent_gt"]
    obs = list(g.observed)
    oi = {o: k for k, o in enumerate(obs)}
    # This call is routed through the installed final L3 encoder.
    T = encode.embed([labels[o] for o in obs])
    W, score = g.estimate_weights(X, oi)
    item_corr = optimize.marginal_corr(g, X, oi)
    pc = optimize.partial_residual_corr(g, X, oi, score)
    residual_pair_info = optimize.leave_pair_out_residual_pairs(g, X, oi)
    independent_info = g.reconcile_independent_pairs(X, oi, score)
    bridge = dict(obs=obs, dep_marg=depmod.load(name, "marginal", "pearson"),
                  lam_upper=0.3, kappa=0.5, q=0.7)
    lat_names = [latent for latent in g.latents if latent in gt]
    G = encode.embed([gt[latent] for latent in lat_names]) if lat_names else None
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(obs))
    folds = [perm[i::FOLDS] for i in range(FOLDS)]
    return dict(name=name, g=g, obs=obs, T=T, W=W, pc=pc, item_corr=item_corr,
                residual_pair_info=residual_pair_info, bridge=bridge,
                independent_info=independent_info, n_samples=int(X.shape[0]),
                lat_names=lat_names, G=G, folds=folds)


def outer_loss(tensors, data, fold, device):
    """Dev-only cosine, relative identity, and latent supervision."""
    obs, T = data["obs"], data["T"]
    masked = sorted(int(i) for i in data["folds"][fold])
    terms, predicted, targets = [], [], []
    for i in masked:
        node = obs[i]
        if node in tensors:
            target = torch.tensor(T[i], dtype=torch.float32, device=device)
            predicted.append(tensors[node])
            targets.append(target)
            terms.append(1.0 - torch.nn.functional.cosine_similarity(
                tensors[node], target, dim=0))
    if not terms:
        raise RuntimeError(f"No masked observed tensors for {data['name']} fold {fold}")
    loss = torch.stack(terms).mean()
    if len(predicted) > 1:
        pred = torch.nn.functional.normalize(torch.stack(predicted), dim=1)
        target = torch.nn.functional.normalize(torch.stack(targets), dim=1)
        similarity = pred @ target.T
        target_similarity = target @ target.T
        base_margin = 1.0 - target_similarity
        adapted_margin = similarity.diag()[:, None] - similarity
        scale = torch.clamp(base_margin, min=0.05)
        offdiag = ~torch.eye(len(predicted), dtype=torch.bool, device=device)
        relative = torch.relu((base_margin - adapted_margin) / scale) ** 2
        loss = loss + relative[offdiag].mean()
    latent_terms = []
    if data["G"] is not None:
        for k, latent in enumerate(data["lat_names"]):
            if latent in tensors:
                target = torch.tensor(data["G"][k], dtype=torch.float32, device=device)
                latent_terms.append(1.0 - torch.nn.functional.cosine_similarity(
                    tensors[latent], target, dim=0))
    if latent_terms:
        loss = loss + 0.5 * torch.stack(latent_terms).mean()
    return loss


def fold_inputs(data, fold, NEG, device):
    obs, T, g, W = data["obs"], data["T"], data["g"], data["W"]
    masked = set(int(i) for i in data["folds"][fold])
    visible = {obs[i]: T[i] for i in range(len(obs)) if i not in masked}
    item_info = core.prepare_item_identity(
        g, visible, data["item_corr"], data["n_samples"], neg_op=NEG, device=device)
    feats = torch.tensor(
        LM.node_features(g, W, set(visible), item_info=item_info,
                         independent_info=data["independent_info"]),
        dtype=torch.float32, device=device)
    return visible, item_info, feats


def solve_pair(data, fold, module, NEG, train, device):
    visible, item_info, feats = fold_inputs(data, fold, NEG, device)
    _, tensors = core.solve_unrolled(
        data["g"], data["W"], visible, d=data["T"].shape[1],
        weight_module=module, K=K, inner_lr=INNER_LR, seed=fold, device=device,
        residual=1.0, lam_res=1.0, partial_corr=data["pc"],
        neg_op=NEG, bridge=data["bridge"], n_samples=data["n_samples"],
        independent_info=data["independent_info"], item_info=item_info,
        item_corr=data["item_corr"], residual_pair_info=data["residual_pair_info"],
        train=train, feats=feats,
        truncation_steps=TRUNCATION_STEPS if train else None)
    return outer_loss(tensors, data, fold, device)


def multiplier_distribution(module, data, NEG, device, fold=4):
    """Summarize the actually applicable node multipliers for all six constraint types."""
    values = {term: [] for term in LM.TERMS}
    with torch.no_grad():
        for d_ in data.values():
            visible, item_info, feats = fold_inputs(d_, fold, NEG, device)
            mult = module.multipliers(feats).detach().cpu().numpy()
            ni = {n: i for i, n in enumerate(d_["g"].nodes)}
            gen = [ni[n] for n in d_["g"].nodes if d_["g"].parents(n)]
            free = [ni[n] for n in d_["g"].nodes if n not in visible]
            # ``step_loss`` consumes the sparse leave-pair-out residual relations when present,
            # so report only endpoints that can actually receive an anchor multiplier.
            anchor_nodes = {
                node
                for a, b, _ in d_["residual_pair_info"].get("pairs", [])
                for node in (a, b)
            }
            anchor = [ni[n] for n in d_["g"].nodes if n in anchor_nodes]
            item = [ni[n] for n in item_info["nodes"]]
            index = {"gen": gen, "resnorm": gen, "anchor": anchor,
                     "node": list(range(len(d_["g"].nodes))), "norm": free, "item": item}
            for column, term in enumerate(LM.TERMS):
                if index[term]:
                    values[term].extend(mult[index[term], column].tolist())
    lower, upper = np.exp(-LM.BOUND), np.exp(LM.BOUND)
    out = {}
    for term, vals in values.items():
        arr = np.asarray(vals, dtype=float)
        out[term] = {
            "count": int(arr.size),
            "mean": float(arr.mean()) if arr.size else None,
            "std": float(arr.std()) if arr.size else None,
            "min": float(arr.min()) if arr.size else None,
            "max": float(arr.max()) if arr.size else None,
            "near_lower_fraction": float(np.mean(arr <= lower * 1.01)) if arr.size else None,
            "near_upper_fraction": float(np.mean(arr >= upper / 1.01)) if arr.size else None,
        }
    return out


def main():
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    if K != release.SOLVER_STEPS:
        raise ValueError("The main-line L2 training budget is fixed at K=120 (2x60); "
                         "K=240 is not supported by this training script.")
    # Enforce L3 -> L2 ordering and retain the model object for the duration of training.
    l3_st, l3_sha256 = lora.install_as_encode_model(encode, L3_CKPT, DEVICE)
    print(f"[{ts()}] installed final L3 encoder {l3_sha256[:12]} on {DEVICE}", flush=True)

    NEG = negop.load().to(DEVICE)
    for p in NEG.parameters():
        p.requires_grad_(False)
    names = list(pool.DEV)
    print(f"[{ts()}] prep {len(names)} dev datasets in final L3 space ...", flush=True)
    data = {}
    for name in names:
        data[name] = prep(name)
        info = data[name]["independent_info"]
        print(f"[{ts()}]   {name}: {len(data[name]['obs'])} obs, "
              f"{len(data[name]['g'].latents)} latents; independent "
              f"{info['raw_count']}->{info['retained_count']} "
              f"(conflicts={info['conflict_count']})", flush=True)

    if ARM not in ("mlp", "static"):
        raise ValueError("ARM must be 'mlp' or 'static'")
    module = LM.WeightNet() if ARM == "mlp" else LM.StaticWeights()
    module.to(DEVICE).train()
    optimizer = torch.optim.Adam(module.parameters(), lr=OUTER_LR)
    pairs = [(name, fold) for name in names for fold in range(4)]
    log = {"release_version": release.RELEASE_VERSION,
           "arm": ARM, "K": K, "truncation_steps": TRUNCATION_STEPS,
           "device": DEVICE, "l3_checkpoint": os.path.abspath(L3_CKPT),
           "l3_checkpoint_sha256": l3_sha256, "epochs": []}
    best = float("inf")
    output_dir = os.path.join(HERE, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(L2_OUT)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(L2_LOG)), exist_ok=True)
    ckpt = L2_OUT

    for epoch in range(EPOCHS):
        order = np.random.default_rng(100 + epoch).permutation(len(pairs))
        train_losses = []
        for j, pair_index in enumerate(order):
            name, fold = pairs[int(pair_index)]
            optimizer.zero_grad()
            loss = solve_pair(data[name], fold, module, NEG, True, DEVICE)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.detach()))
            if j % 16 == 0:
                print(f"[{ts()}] ep{epoch} {j}/{len(pairs)} "
                      f"outer_loss={np.mean(train_losses[-16:]):.4f}", flush=True)

        module.eval()
        # Inner inference still needs autograd to calculate each optimization step.
        val_losses = [float(solve_pair(data[name], 4, module, NEG, False, DEVICE).detach())
                      for name in names]
        module.train()
        val = float(np.mean(val_losses))
        is_best = val < best
        log["epochs"].append({
            "train": float(np.mean(train_losses)),
            "val": val,
            "val_per_ds": dict(zip(names, [round(value, 4) for value in val_losses])),
        })
        print(f"[{ts()}] EPOCH {epoch}: train={np.mean(train_losses):.4f} val={val:.4f}"
              f"{' (best, saved)' if is_best else ''}", flush=True)
        if is_best:
            best = val
            LM.save(module, ckpt, "static" if ARM == "static" else "mlp",
                    metadata={"l3_checkpoint_sha256": l3_sha256, "K": K,
                              "truncation_steps": TRUNCATION_STEPS,
                              "release_version": release.RELEASE_VERSION,
                              "trained_in_embedding_space": "E5-large-v2 + final L3 LoRA"})
        with open(L2_LOG, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=1)

    best_module = LM.load(ckpt, DEVICE, expected_l3_sha256=l3_sha256)
    baseline = [float(solve_pair(data[name], 4, None, NEG, False, DEVICE).detach())
                for name in names]
    log["baseline_val_mult1"] = float(np.mean(baseline))
    log["multiplier_distribution_fold4"] = multiplier_distribution(
        best_module, data, NEG, DEVICE)
    with open(L2_LOG, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=1)
    print(f"[{ts()}] done. best val={best:.4f} vs mult=1 val={np.mean(baseline):.4f}",
          flush=True)
    print(json.dumps(log["multiplier_distribution_fold4"], indent=1), flush=True)
    # Keep the adapted encoder strongly referenced until all embedding work is complete.
    _ = l3_st


if __name__ == "__main__":
    main()
