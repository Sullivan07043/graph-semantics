"""Train identity-preserving L3 LoRA on DEV labels only.

The original identity hinge is retained.  A graph-balanced relational loss and deterministic
cross-dataset replay preserve the frozen same-parent geometry and prevent cross-parent collapse.
Data-gated L3 independence is frozen-relative (pairs may separate but may not move closer).
Checkpoint selection is constrained by every multi-factor DEV dataset's frozen cosine gap; when
no adapted epoch is feasible, the valid zero-LoRA checkpoint is retained.
"""
import json
import os
import sys
import time
import zlib

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "pipeline_v3"))
import encode
import negop
import pool
import dependence as depmod
from pipeline_L3_v1 import lora
from run_task1 import ALL_LOADERS

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", 8)))
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = int(os.environ.get("EPOCHS", 3))
LR = float(os.environ.get("LR", 1e-4))
IDENTITY_WEIGHT = 1.0
RELATIONAL_WEIGHT = 1.0
REPLAY_WEIGHT = 1.0
REPLAY_GROUPS_PER_TYPE = 4
GUARD_TOLERANCE = 1e-4
W = {
    "anchor": float(os.environ.get("W_ANCHOR", 10.0)),
    "bridge": float(os.environ.get("W_BRIDGE", 1.0)),
    "indep": float(os.environ.get("W_INDEP", 0.3)),
    "neg": float(os.environ.get("W_NEG", 1.0)),
    "identity": IDENTITY_WEIGHT,
    "relational": RELATIONAL_WEIGHT,
}
FOLDS, KAPPA, Q = 5, 0.5, 0.7
OUTPUT_DIR = os.path.join(HERE, "outputs")
CKPT_PATH = os.environ.get("L3_OUT", os.path.join(OUTPUT_DIR, "l3_lora_rel.pt"))
LOG_PATH = os.environ.get("L3_LOG", os.path.join(OUTPUT_DIR, "l3_rel_trainlog.json"))


def ts():
    return time.strftime("%H:%M:%S")


def prep(name):
    ds = ALL_LOADERS[name]()
    g, X, labels, gt = ds["graph"], ds["X"], ds["labels"], ds["latent_gt"]
    obs = list(g.observed)
    oi = {o: k for k, o in enumerate(obs)}
    W_, score = g.estimate_weights(X, oi)
    indep_info = g.reconcile_independent_pairs(X, oi, score)
    Dm = np.asarray(depmod.load(name, "marginal", "pearson"))
    tp = [(a, b) for a, b in g.trek_pairs() if a in oi and b in oi]
    vals = np.array([Dm[oi[a], oi[b]] for a, b in tp]) if tp else np.array([])
    threshold = np.quantile(vals, Q) if len(vals) else 1e9
    bridge_pairs = [(a, b, float(v)) for (a, b), v in zip(tp, vals) if v >= threshold]
    indep_pairs = [(a, b) for a, b in indep_info["pairs"] if a in oi and b in oi]
    neg_edges = [(p, c) for (p, c), w in W_.items()
                 if w < 0 and g.is_latent(p) and p in gt and c in oi]
    # Directed pairs are intentional: adapted_margin(i,j) is anchored at item i.
    identity_pairs = []
    parent_sets = {o: set(g.parents(o)) for o in obs}
    for a in obs:
        for b in obs:
            if a != b and parent_sets[a] & parent_sets[b]:
                identity_pairs.append((a, b))
    same_groups = []
    for parent in g.nodes:
        members = [o for o in obs if parent in parent_sets[o]]
        pairs = [(a, b) for k, a in enumerate(members) for b in members[k + 1:]]
        if pairs:
            same_groups.append(pairs)
    cross_group_map = {}
    for k, a in enumerate(obs):
        for b in obs[k + 1:]:
            if parent_sets[a] & parent_sets[b]:
                continue
            left, right = tuple(sorted(parent_sets[a])), tuple(sorted(parent_sets[b]))
            key = tuple(sorted((left, right)))
            cross_group_map.setdefault(key, []).append((a, b))
    cross_groups = list(cross_group_map.values())
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(obs))
    folds = [perm[i::FOLDS] for i in range(FOLDS)]
    return dict(name=name, g=g, obs=obs, oi=oi, labels=labels, gt=gt, folds=folds,
                bridge=bridge_pairs, indep=indep_pairs, neg=neg_edges,
                identity=identity_pairs, same_groups=same_groups,
                cross_groups=cross_groups, indep_info=indep_info)


def encode_no_grad(st, texts, device, batch_size=256):
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            out.append(lora.encode_grad(st, texts[i:i + batch_size], device,
                                        max_len=128).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def identity_loss(H, H0, pairs):
    """The fixed identity-preserving hinge, factored out for mechanism testing."""
    if not pairs:
        return H.new_zeros(())
    ia = torch.tensor([a for a, _ in pairs], dtype=torch.long, device=H.device)
    ib = torch.tensor([b for _, b in pairs], dtype=torch.long, device=H.device)
    base_margin = 1.0 - (H0[ia] * H0[ib]).sum(1)
    adapted_margin = (H[ia] * H0[ia]).sum(1) - (H[ia] * H0[ib]).sum(1)
    return (torch.relu(base_margin - adapted_margin) ** 2).mean()


def relational_loss(H, H0, same_groups, cross_groups):
    """Graph-balanced frozen-geometry safeguard.

    Same-parent pairs preserve their frozen cosine in both directions.  Cross-parent pairs incur
    loss only when they become more similar, so structured independence remains free to separate
    them.  Averaging first within graph groups prevents large factors from receiving quadratic
    weight merely because they contain more item pairs.
    """
    def group_losses(groups, one_sided):
        values = []
        for pairs in groups:
            if not pairs:
                continue
            ia = torch.tensor([a for a, _ in pairs], dtype=torch.long, device=H.device)
            ib = torch.tensor([b for _, b in pairs], dtype=torch.long, device=H.device)
            current = (H[ia] * H[ib]).sum(1)
            frozen = (H0[ia] * H0[ib]).sum(1)
            scale = torch.clamp(1.0 - frozen, min=0.05)
            delta = (current - frozen) / scale
            term = torch.relu(delta) ** 2 if one_sided else delta ** 2
            values.append(term.mean())
        return torch.stack(values).mean() if values else H.new_zeros(())

    same = group_losses(same_groups, one_sided=False)
    cross = group_losses(cross_groups, one_sided=True)
    if same_groups and cross_groups:
        return 0.5 * (same + cross)
    return same + cross


def _visible_relation_groups(groups, visible_index):
    out = []
    for group in groups:
        pairs = [(visible_index[a], visible_index[b]) for a, b in group
                 if a in visible_index and b in visible_index]
        if pairs:
            out.append(pairs)
    return out


def relational_replay_loss(st, data, step, device):
    """One deterministic, graph-balanced relation pair per type and DEV dataset."""
    texts, frozen, same_pairs, cross_pairs = [], [], [], []
    text_index = {}

    def add(d_, node):
        key = (d_["name"], node)
        if key not in text_index:
            text_index[key] = len(texts)
            texts.append(d_["labels"][node])
            frozen.append(d_["h0"][d_["oi"][node]])
        return text_index[key]

    for offset, d_ in enumerate(data.values()):
        masked = set(int(i) for i in d_["folds"][(step + offset) % 4])
        visible = {o for i, o in enumerate(d_["obs"]) if i not in masked}
        same_available = [[pair for pair in group if pair[0] in visible and pair[1] in visible]
                          for group in d_["same_groups"]]
        same_available = [group for group in same_available if group]
        cross_available = [[pair for pair in group if pair[0] in visible and pair[1] in visible]
                           for group in d_["cross_groups"]]
        cross_available = [group for group in cross_available if group]
        for groups, destination in ((same_available, same_pairs),
                                    (cross_available, cross_pairs)):
            if not groups:
                continue
            count = min(REPLAY_GROUPS_PER_TYPE, len(groups))
            for pick in range(count):
                group = groups[(step + offset + pick) % len(groups)]
                a, b = group[(step // max(len(data), 1) + pick) % len(group)]
                destination.append([(add(d_, a), add(d_, b))])
    if not texts:
        return next(st.parameters()).new_zeros(())
    H = lora.encode_grad(st, texts, device)
    H0 = torch.tensor(np.stack(frozen), dtype=torch.float32, device=device)
    return relational_loss(H, H0, same_pairs, cross_pairs)


def identity_safe_gradients(task_loss, safeguard_loss, params):
    """Gradient-balanced PCGrad for the task/safeguard pair.

    The safeguard is zero at the frozen encoder, so a fixed coefficient badly understates it once
    other objectives begin to move the geometry.  Its gradient is normalized to the task-gradient
    norm without a searched scalar.  A conflicting task component is projected off the safeguard
    direction, guaranteeing a nonnegative first-order inner product.
    """
    task_grad = torch.autograd.grad(
        task_loss, params, retain_graph=True, allow_unused=True)
    safe_grad = torch.autograd.grad(
        safeguard_loss, params, allow_unused=True)
    task_grad = [torch.zeros_like(p) if g is None else g for p, g in zip(params, task_grad)]
    safe_grad = [torch.zeros_like(p) if g is None else g for p, g in zip(params, safe_grad)]
    task_norm = torch.sqrt(sum((g * g).sum() for g in task_grad))
    safe_norm = torch.sqrt(sum((g * g).sum() for g in safe_grad))
    if float(safeguard_loss.detach()) < 1e-10 or float(safe_norm.detach()) < 1e-8:
        return task_grad, {
            "task_grad_norm": float(task_norm.detach()),
            "safe_grad_norm": float(safe_norm.detach()),
            "grad_cosine": None,
            "safe_balance": 0.0,
        }
    balance = (task_norm / safe_norm.clamp(min=1e-12)).detach()
    balanced_safe = [balance * g for g in safe_grad]
    dot = sum((a * b).sum() for a, b in zip(task_grad, balanced_safe))
    balanced_norm2 = sum((g * g).sum() for g in balanced_safe).clamp(min=1e-12)
    if float(dot.detach()) < 0:
        task_grad = [a - dot / balanced_norm2 * b
                     for a, b in zip(task_grad, balanced_safe)]
    final = [a + b for a, b in zip(task_grad, balanced_safe)]
    cosine = dot / (task_norm * torch.sqrt(balanced_norm2)).clamp(min=1e-12)
    return final, {
        "task_grad_norm": float(task_norm.detach()),
        "safe_grad_norm": float(safe_norm.detach()),
        "grad_cosine": float(cosine.detach()),
        "safe_balance": float(balance.detach()),
    }


def bundle_loss(st, d_, fold, NEG, anchors, aref, device, return_tensors=False):
    masked = set(int(i) for i in d_["folds"][fold])
    vis = [o for i, o in enumerate(d_["obs"]) if i not in masked]
    vset = set(vis)
    lat = sorted({p for p, _ in d_["neg"]})
    texts = [d_["labels"][o] for o in vis] + [d_["gt"][L] for L in lat]
    idx = {o: k for k, o in enumerate(vis)}
    lidx = {L: len(vis) + k for k, L in enumerate(lat)}
    H = lora.encode_grad(st, texts, device)
    H0 = torch.tensor(d_["h0"][[d_["oi"][o] for o in vis]], dtype=torch.float32, device=device)
    losses = {}

    bp = [(idx[a], idx[b], v) for a, b, v in d_["bridge"] if a in vset and b in vset]
    if bp:
        ia = torch.tensor([x[0] for x in bp], device=device)
        ib = torch.tensor([x[1] for x in bp], device=device)
        floor = torch.tensor([KAPPA * x[2] for x in bp], dtype=torch.float32, device=device)
        cosine = (H[ia] * H[ib]).sum(1).abs()
        losses["bridge"] = (torch.relu(floor - cosine) ** 2).mean()

    ip = [(idx[a], idx[b]) for a, b in d_["indep"] if a in vset and b in vset]
    if ip:
        ia = torch.tensor([x[0] for x in ip], device=device)
        ib = torch.tensor([x[1] for x in ip], device=device)
        current = (H[ia] * H[ib]).sum(1)
        frozen = (H0[ia] * H0[ib]).sum(1)
        losses["indep"] = (torch.relu(current - frozen) ** 2).mean()

    neg_pairs = [(lidx[p], idx[c]) for p, c in d_["neg"] if c in vset]
    if neg_pairs:
        il = torch.tensor([x[0] for x in neg_pairs], device=device)
        ic = torch.tensor([x[1] for x in neg_pairs], device=device)
        target = torch.nn.functional.normalize(NEG(H[il]), dim=1)
        losses["neg"] = (1.0 - (H[ic] * target).sum(1)).mean()

    id_pairs = [(idx[a], idx[b]) for a, b in d_["identity"] if a in vset and b in vset]
    if id_pairs:
        losses["identity"] = identity_loss(H[:len(vis)], H0, id_pairs)

    same_groups = _visible_relation_groups(d_["same_groups"], idx)
    cross_groups = _visible_relation_groups(d_["cross_groups"], idx)
    if same_groups or cross_groups:
        losses["relational"] = relational_loss(
            H[:len(vis)], H0, same_groups, cross_groups)

    # Preserve the existing dictionary drift anchor: 256 sampled training anchors per bundle,
    # a 0.99 cosine hinge, and the historical internal x100 scale.
    stable_seed = zlib.crc32(f"{d_['name']}:{fold}".encode("utf-8"))
    sel = np.random.default_rng(stable_seed).choice(len(anchors), 256, replace=False)
    Ha = lora.encode_grad(st, [anchors[i] for i in sel], device)
    reference = torch.tensor(aref[sel], dtype=torch.float32, device=device)
    losses["anchor"] = (torch.relu(0.99 - (Ha * reference).sum(1)) ** 2).mean() * 100.0

    total = sum(W[k] * value for k, value in losses.items())
    parts = {k: float(value.detach()) for k, value in losses.items()}
    return (total, parts, losses) if return_tensors else (total, parts)


def _mean_parts(rows):
    keys = sorted({k for row in rows for k in row})
    return {k: float(np.mean([row[k] for row in rows if k in row and row[k] is not None]))
            for k in keys if any(k in row and row[k] is not None for row in rows)}


def cosine_gap(H, d_):
    same, cross = [], []
    parent_sets = {o: set(d_["g"].parents(o)) for o in d_["obs"]}
    for i, a in enumerate(d_["obs"]):
        for j in range(i + 1, len(d_["obs"])):
            b = d_["obs"][j]
            value = float(np.dot(H[i], H[j]))
            (same if parent_sets[a] & parent_sets[b] else cross).append(value)
    sm = float(np.mean(same)) if same else None
    cm = float(np.mean(cross)) if cross else None
    return {"same_parent_mean": sm, "cross_parent_mean": cm,
            "gap": (sm - cm if sm is not None and cm is not None else None)}


def all_geometry(st, data, device):
    out = {}
    for name, d_ in data.items():
        embeddings = encode_no_grad(
            st, [d_["labels"][o] for o in d_["obs"]], device)
        out[name] = cosine_gap(embeddings, d_)
    return out


def main():
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    names = list(pool.DEV)
    data = {name: prep(name) for name in names}
    st = lora.load_st(DEVICE)
    # Frozen E5 identities are computed for every dev label before any LoRA module is injected.
    for name in names:
        d_ = data[name]
        d_["h0"] = encode_no_grad(st, [d_["labels"][o] for o in d_["obs"]], DEVICE)
        info = d_["indep_info"]
        print(f"[{ts()}] {name}: independent raw={info['raw_count']} "
              f"retained={info['retained_count']} conflicts={info['conflict_count']} "
              f"tau={info['tau']:.5f}", flush=True)
    frozen_geometry = {name: cosine_gap(d_["h0"], d_) for name, d_ in data.items()}
    himi_frozen = frozen_geometry["himi"]

    params = lora.inject(st)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(CKPT_PATH)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(LOG_PATH)), exist_ok=True)
    # A zero-LoRA checkpoint is always feasible and is retained unless an adapted epoch preserves
    # every multi-factor DEV gap while improving the original validation objective.
    torch.save(lora.checkpoint_payload(
        st, metadata={"selection": "zero-lora relational fallback",
                      "guard_tolerance": GUARD_TOLERANCE,
                      "dev_datasets": names}), CKPT_PATH)
    NEG = negop.load().to(DEVICE)
    for p in NEG.parameters():
        p.requires_grad_(False)
    C, cwords = encode.load_dictionary()
    rng = np.random.default_rng(7)
    pick = rng.choice(len(cwords), 22000, replace=False)
    anchors = [cwords[i] for i in pick[:20000]]
    aref = C[pick[:20000]].astype(np.float32)
    drift_words = [cwords[i] for i in pick[20000:]]
    dref = C[pick[20000:]].astype(np.float32)
    del C

    print(f"[{ts()}] prepped {len(names)} dev sets; trainable params "
          f"{sum(p.numel() for p in params)}; identity_weight={IDENTITY_WEIGHT}", flush=True)
    opt = torch.optim.Adam(params, lr=LR)
    pairs = [(name, fold) for name in names for fold in range(4)]
    log = {
        "checkpoint_version": lora.CHECKPOINT_VERSION,
        "device": DEVICE,
        "weights": W,
        "replay_weight": REPLAY_WEIGHT,
        "replay_groups_per_type": REPLAY_GROUPS_PER_TYPE,
        "guard_tolerance": GUARD_TOLERANCE,
        "identity_weight_note": "fixed at 1.0; no hyperparameter grid search",
        "independence": {name: {k: data[name]["indep_info"][k]
                                for k in ("raw_count", "retained_count", "conflict_count",
                                          "tau", "n_samples")}
                         for name in names},
        "himi_frozen_gap": himi_frozen,
        "frozen_geometry": frozen_geometry,
        "epochs": [],
    }
    best = float("inf")
    best_source = "zero-lora relational fallback"

    def drift():
        Hd = encode_no_grad(st, drift_words, DEVICE, batch_size=512)
        cosine = (Hd * dref).sum(1)
        return float(cosine.mean()), float(cosine.min())

    for epoch in range(EPOCHS):
        order = np.random.default_rng(100 + epoch).permutation(len(pairs))
        train_losses, train_parts, gradient_parts = [], [], []
        for j, pair_index in enumerate(order):
            name, fold = pairs[int(pair_index)]
            opt.zero_grad()
            loss, parts, tensor_parts = bundle_loss(
                st, data[name], fold, NEG, anchors, aref, DEVICE, return_tensors=True)
            replay = relational_replay_loss(
                st, data, epoch * len(pairs) + j, DEVICE)
            loss = loss + REPLAY_WEIGHT * replay
            parts["replay"] = float(replay.detach())
            safe = W["identity"] * tensor_parts.get("identity", loss.new_zeros(())) + \
                W["relational"] * tensor_parts.get("relational", loss.new_zeros(())) + \
                REPLAY_WEIGHT * replay
            task = loss - safe
            gradients, grad_info = identity_safe_gradients(task, safe, params)
            for parameter, gradient in zip(params, gradients):
                parameter.grad = gradient
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            train_losses.append(float(loss.detach()))
            train_parts.append(parts)
            gradient_parts.append(grad_info)
            if j % 16 == 0:
                print(f"[{ts()}] ep{epoch} {j}/{len(pairs)} "
                      f"loss={np.mean(train_losses[-16:]):.4f} {parts}", flush=True)

        val_losses, val_parts = [], []
        for name in names:
            with torch.no_grad():
                loss, parts = bundle_loss(st, data[name], 4, NEG, anchors, aref, DEVICE)
            val_losses.append(float(loss))
            val_parts.append(parts)
        val = float(np.mean(val_losses))
        drift_mean, drift_min = drift()
        adapted_geometry = all_geometry(st, data, DEVICE)
        himi_l3 = adapted_geometry["himi"]
        guard_failures = {
            name: {
                "frozen_gap": frozen_geometry[name]["gap"],
                "adapted_gap": geometry["gap"],
                "change": geometry["gap"] - frozen_geometry[name]["gap"],
            }
            for name, geometry in adapted_geometry.items()
            if frozen_geometry[name]["gap"] is not None
            and geometry["gap"] < frozen_geometry[name]["gap"] - GUARD_TOLERANCE
        }
        guard_ok = not guard_failures
        entry = {
            "train": float(np.mean(train_losses)),
            "val": val,
            "train_parts": _mean_parts(train_parts),
            "gradient_parts": _mean_parts(gradient_parts),
            "val_parts": _mean_parts(val_parts),
            "identity_train": _mean_parts(train_parts).get("identity"),
            "identity_val": _mean_parts(val_parts).get("identity"),
            "drift_mean_cos": drift_mean,
            "drift_min_cos": drift_min,
            "himi_frozen_gap": himi_frozen,
            "himi_l3_gap": himi_l3,
            "himi_gap_change": (himi_l3["gap"] - himi_frozen["gap"]
                                if himi_l3["gap"] is not None else None),
            "adapted_geometry": adapted_geometry,
            "relational_guard_ok": guard_ok,
            "relational_guard_failures": guard_failures,
        }
        log["epochs"].append(entry)
        is_best = guard_ok and val < best
        print(f"[{ts()}] EPOCH {epoch}: train={entry['train']:.4f} val={val:.4f} "
              f"identity(train/val)={entry['identity_train']:.6f}/"
              f"{entry['identity_val']:.6f} drift(mean/min)={drift_mean:.4f}/{drift_min:.4f} "
              f"Himi gap frozen/L3={himi_frozen['gap']:.4f}/{himi_l3['gap']:.4f}"
              f" guard={'ok' if guard_ok else 'fail:' + str(len(guard_failures))}"
              f"{' (best, saved)' if is_best else ''}", flush=True)
        if is_best:
            best = val
            best_source = f"epoch {epoch}"
            torch.save(lora.checkpoint_payload(
                st, metadata={"identity_weight": IDENTITY_WEIGHT,
                              "relational_weight": RELATIONAL_WEIGHT,
                              "replay_weight": REPLAY_WEIGHT,
                              "guard_tolerance": GUARD_TOLERANCE,
                              "dev_datasets": names, "best_epoch": epoch,
                              "best_val": best}), CKPT_PATH)
        log["selected_checkpoint"] = best_source
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=1)
    print(f"[{ts()}] done. selected={best_source}; "
          f"best feasible val={best if np.isfinite(best) else None}", flush=True)


if __name__ == "__main__":
    main()
