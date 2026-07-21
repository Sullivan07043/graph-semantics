"""PIPELINE V4 — L2 learned (unrolled) optimizer.

Two solvers over the SAME normalized objective:
  solve_frozen   — scalar-weight Adam reference using the shared normalized objective.
  solve_unrolled — the L2 solver: ALS init + K functional-Adam steps, fully differentiable, with
                   per-node constraint-weight MULTIPLIERS coming from a weight module. With the
                   identity module (all multipliers 1) it is the frozen objective under different
                   solver dynamics (the solver-dynamics control arm).

Weight modules (l2_modules.py): StaticWeights (6 learned scalars, attribution control) and
WeightNet (node-context MLP, the main nonlinear arm; zero-init head => multipliers start at 1).
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root
sys.path.insert(0, HERE)
import optimize as O                                                  # noqa: E402

TERMS = ["gen", "resnorm", "anchor", "node", "norm", "item"]


def prepare_item_identity(g, labeled_emb, marginal_corr, n_samples, neg_op=None, device="cpu"):
    """Build leak-free whole-item targets from original observed-column correlations.

    The local semantic target remains restricted to visible ``g.mb_observed(i)`` nodes.  Marginal
    Pearson correlation is used here because the target is the complete item embedding; parent-
    residual correlations belong to the residual-vector channel.  Positive and negative
    (NegOp-transformed) candidates are normalized as separate polarity views before being combined,
    so the larger pole cannot drown out the other merely through candidate count or total weight.
    A second, non-candidate profile contrast is enabled only when a reliable visible association
    exists outside the graph-local neighborhood; a one-parent star therefore does not duplicate
    its local item constraint under a different name.

    Reliable weights are noise-floor excesses ``(|rho|-2/sqrt(n))_+``.  Negative associations are
    usable only through ``neg_op`` and are never represented by vector negation.  ``confidence``
    remains the auditable max-|rho| feature.  Once a node passes the reliable-candidate gate,
    ``L_item`` itself uses unit strength because reliability was already applied both as a hard
    threshold and inside the target weights.
    """
    import torch
    visible = set(labeled_emb)
    node_features = np.zeros((len(g.nodes), 7), dtype=np.float32)
    empty = {"nodes": [], "targets": np.zeros((0, 0), dtype=np.float32),
             "confidence": np.zeros(0, dtype=np.float32), "features": node_features,
             "profile_nodes": [], "profile_hi": np.zeros((0, 0), dtype=np.float32),
             "profile_lo": np.zeros((0, 0), dtype=np.float32),
             "profile_confidence": np.zeros(0, dtype=np.float32), "tau": None}
    if marginal_corr is None or not n_samples:
        return empty
    corr_names, Cmat = marginal_corr
    corr_names = list(corr_names)
    cidx = {n: i for i, n in enumerate(corr_names)}
    Cmat = np.asarray(Cmat, dtype=float)
    tau = 2.0 / max(np.sqrt(int(n_samples)), 1.0)

    specs = []
    negative_visible = set()
    for i in g.observed:
        if i in visible or i not in cidx:
            continue
        local = [j for j in g.mb_observed(i) if j in visible and j in cidx]
        accepted = []
        for j in local:
            rho = float(Cmat[cidx[i], cidx[j]])
            if not np.isfinite(rho) or abs(rho) <= tau:
                continue
            if rho < 0 and neg_op is None:
                continue
            accepted.append((j, rho))
            if rho < 0:
                negative_visible.add(j)
        global_assoc, global_weak = [], []
        for j in sorted(visible):
            if j not in cidx or j == i:
                continue
            rho = float(Cmat[cidx[i], cidx[j]])
            if not np.isfinite(rho):
                continue
            if abs(rho) <= tau:
                global_weak.append((j, rho))
            elif rho >= 0 or neg_op is not None:
                global_assoc.append((j, rho))
                if rho < 0:
                    negative_visible.add(j)
        local_set = set(local)
        has_nonlocal_assoc = any(j not in local_set for j, _ in global_assoc)
        specs.append((i, local, accepted, global_assoc, global_weak,
                      has_nonlocal_assoc))

    neg_embeddings = {}
    if negative_visible:
        names = sorted(negative_visible)
        X = torch.tensor(np.stack([labeled_emb[n] for n in names]), dtype=torch.float32,
                         device=device)
        neg_op = neg_op.to(device)
        with torch.no_grad():
            Y = torch.nn.functional.normalize(neg_op(X), dim=1).detach().cpu().numpy()
        neg_embeddings = {n: Y[k] for k, n in enumerate(names)}

    nodes, targets, confidence = [], [], []
    profile_nodes, profile_hi, profile_lo, profile_confidence = [], [], [], []
    ni = {n: i for i, n in enumerate(g.nodes)}
    for i, local, accepted, global_assoc, global_weak, has_nonlocal_assoc in specs:
        if not accepted:
            node_features[ni[i], 3] = float(len(local))
        else:
            abs_corr = np.array([abs(rho) for _, rho in accepted], dtype=np.float32)
            coeff = np.maximum(abs_corr - tau, 0.0)
            polarity_views = []
            for negative in (False, True):
                selected = [k for k, (_, rho) in enumerate(accepted)
                            if (rho < 0) == negative]
                if not selected:
                    continue
                semantic = [
                    neg_embeddings[accepted[k][0]] if negative
                    else np.asarray(labeled_emb[accepted[k][0]], dtype=np.float32)
                    for k in selected
                ]
                view = (coeff[selected, None] * np.stack(semantic)).sum(0)
                view_norm = float(np.linalg.norm(view))
                if np.isfinite(view_norm) and view_norm >= 1e-9:
                    polarity_views.append(view / view_norm)
            z = np.stack(polarity_views).sum(0) if polarity_views else \
                np.zeros_like(np.asarray(labeled_emb[accepted[0][0]], dtype=np.float32))
            zn = float(np.linalg.norm(z))
            if np.isfinite(zn) and zn >= 1e-9:
                z = z / zn
                conf = float(abs_corr.max())
                prob = coeff / max(float(coeff.sum()), 1e-12)
                entropy = (-float(np.sum(prob * np.log(prob + 1e-12))) /
                           max(float(np.log(len(prob))), 1.0)) if len(prob) > 1 else 0.0
                effective = float(1.0 / max(float(np.sum(prob ** 2)), 1e-12))
                nodes.append(i)
                targets.append(z.astype(np.float32))
                confidence.append(conf)
                node_features[ni[i]] = [
                    float(len(accepted)), conf, float(abs_corr.mean()), float(len(local)),
                    conf, entropy, effective,
                ]

        if has_nonlocal_assoc and len(global_assoc) >= 2:
            ordered = sorted(global_assoc, key=lambda x: abs(x[1]), reverse=True)
            split = max(1, len(ordered) // 2)
            high = ordered[:split]
            low = global_weak if global_weak else ordered[split:]
            if low:
                hi_abs = np.array([abs(rho) for _, rho in high], dtype=np.float32)
                hi_w = np.maximum(hi_abs - tau, 0.0)
                hi_sem = [neg_embeddings[j] if rho < 0
                          else np.asarray(labeled_emb[j], dtype=np.float32)
                          for j, rho in high]
                lo_sem = [np.asarray(labeled_emb[j], dtype=np.float32)
                          if abs(rho) <= tau or rho >= 0 else neg_embeddings[j]
                          for j, rho in low]
                lo_abs = np.array([abs(rho) for _, rho in low], dtype=np.float32)
                lo_w = np.ones(len(low), dtype=np.float32) if global_weak \
                    else np.maximum(lo_abs - tau, 1e-6)
                zh = (hi_w[:, None] * np.stack(hi_sem)).sum(0)
                zl = (lo_w[:, None] * np.stack(lo_sem)).sum(0)
                nh, nl = float(np.linalg.norm(zh)), float(np.linalg.norm(zl))
                gap = float(hi_abs.mean() - lo_abs.mean())
                if nh >= 1e-9 and nl >= 1e-9 and gap > 0:
                    profile_nodes.append(i)
                    profile_hi.append((zh / nh).astype(np.float32))
                    profile_lo.append((zl / nl).astype(np.float32))
                    profile_confidence.append(gap)
    d = len(next(iter(labeled_emb.values()))) if labeled_emb else 0
    return {"nodes": nodes,
            "targets": np.stack(targets).astype(np.float32)
            if targets else np.zeros((0, d), dtype=np.float32),
            "confidence": np.asarray(confidence, dtype=np.float32),
            "features": node_features,
            "profile_nodes": profile_nodes,
            "profile_hi": np.stack(profile_hi).astype(np.float32)
            if profile_hi else np.zeros((0, d), dtype=np.float32),
            "profile_lo": np.stack(profile_lo).astype(np.float32)
            if profile_lo else np.zeros((0, d), dtype=np.float32),
            "profile_confidence": np.asarray(profile_confidence, dtype=np.float32),
            "tau": tau}


# --------------------------------------------------------------------------- shared precomputation
def build_ctx(g, W, Wcur, A, free, d, seed, device,
              residual, lam_res, partial_corr, lam_dep, dep_corr, dep_kappa, lam_coll,
              neg_op, bridge, gen_op, independent_info=None, item_info=None,
              residual_pair_info=None):
    """All static precomputation for the stage-2 objective (no Parameters created here).
    Static context shared by the scalar and unrolled normalized objectives."""
    import torch
    ctx = {"g": g, "W": W, "free": free, "d": d, "device": device}
    ctx["all_nodes"] = list(g.nodes)
    ctx["At"] = {n: torch.tensor(v, dtype=torch.float32, device=device) for n, v in A.items()}
    visible_observed = [n for n in g.observed if n in A]
    if visible_observed:
        center_np = np.mean(np.stack([np.asarray(A[n], dtype=np.float32)
                                     for n in visible_observed]), axis=0)
    else:
        center_np = np.zeros(d, dtype=np.float32)
    ctx["semantic_center"] = torch.tensor(center_np, dtype=torch.float32, device=device)
    gen_nodes = [n for n in g.nodes if g.parents(n)]
    ctx["gen_nodes"] = gen_nodes
    ctx["parents"] = {n: list(g.parents(n)) for n in gen_nodes}
    ctx["wt_const"] = {e: torch.tensor(float(v), dtype=torch.float32, device=device)
                       for e, v in Wcur.items()}

    ctx["use_res"] = residual > 0
    ctx["residual"] = residual
    ctx["lam_res"] = lam_res
    ctx["pc_nodes"] = []
    ctx["pc_pair_terms"] = None
    if ctx["use_res"]:
        gnr = np.random.default_rng(seed)
        ctx["Rv0"] = {n: gnr.normal(0, 1e-3, d) for n in gen_nodes}
        rv_set = set(gen_nodes)
        if lam_res > 0 and residual_pair_info is not None:
            rp = [(a, b, float(rho)) for a, b, rho in residual_pair_info.get("pairs", [])
                  if a in rv_set and b in rv_set and np.isfinite(rho)]
            if rp:
                pc_nodes = sorted({n for a, b, _ in rp for n in (a, b)})
                pci = {n: i for i, n in enumerate(pc_nodes)}
                ctx["pc_nodes"] = pc_nodes
                ctx["pc_pair_terms"] = (
                    torch.tensor([pci[a] for a, _, _ in rp], dtype=torch.long, device=device),
                    torch.tensor([pci[b] for _, b, _ in rp], dtype=torch.long, device=device),
                    torch.tensor([rho for _, _, rho in rp], dtype=torch.float32, device=device),
                    torch.tensor([
                        max(abs(rho) - float(residual_pair_info.get("tau", 0.0)), 0.0)
                        for _, _, rho in rp
                    ], dtype=torch.float32, device=device),
                )
        elif lam_res > 0 and partial_corr is not None:
            pc_names, Pmat = partial_corr
            pc_nodes = [n for n in pc_names if n in rv_set]
            pidx = [list(pc_names).index(n) for n in pc_nodes]
            ctx["pc_nodes"] = pc_nodes
            ctx["Pt"] = torch.tensor(Pmat[np.ix_(pidx, pidx)], dtype=torch.float32, device=device)
            ctx["offdiag"] = ~torch.eye(len(pc_nodes), dtype=torch.bool, device=device)

    node_idx = {n: i for i, n in enumerate(g.nodes)}
    ctx["node_idx"] = node_idx
    raw_zp_pairs = (list(independent_info["pairs"]) if independent_info is not None
                    else g.independent_pairs())
    free_set = set(free)
    zp_pairs = [(a, b) for a, b in raw_zp_pairs if a in free_set or b in free_set]
    fixed_zp_pairs = [(a, b) for a, b in raw_zp_pairs if a not in free_set and b not in free_set]
    ctx["zp_pairs"] = zp_pairs
    ctx["zero_pair_counts"] = {
        "raw": len(raw_zp_pairs), "active": len(zp_pairs), "fixed_fixed": len(fixed_zp_pairs)
    }
    ctx["ia"] = torch.tensor([node_idx[a] for a, b in zp_pairs], dtype=torch.long, device=device)
    ctx["ib"] = torch.tensor([node_idx[b] for a, b in zp_pairs], dtype=torch.long, device=device)
    fixed_zero_cos = []
    for a, b in fixed_zp_pairs:
        va = np.asarray(A[a], dtype=float) - center_np if a in A else None
        vb = np.asarray(A[b], dtype=float) - center_np if b in A else None
        if va is None or vb is None:
            continue
        denom = np.linalg.norm(va) * np.linalg.norm(vb)
        if denom >= 1e-12:
            fixed_zero_cos.append(float(np.dot(va, vb) / denom))
    ctx["zero_target"] = torch.tensor(
        float(np.median(fixed_zero_cos)) if fixed_zero_cos else 0.0,
        dtype=torch.float32, device=device)

    ctx["dep_terms"] = None
    if lam_dep > 0 and dep_corr is not None:
        dn, Rm_ = dep_corr
        di = {n: k for k, n in enumerate(dn)}
        obs_set = set(dn)
        pairs = [(a, b) for a, b in g.trek_pairs() if a in obs_set and b in obs_set]
        if pairs:
            da = torch.tensor([node_idx[a] for a, b in pairs], dtype=torch.long, device=device)
            db = torch.tensor([node_idx[b] for a, b in pairs], dtype=torch.long, device=device)
            floor = torch.tensor([dep_kappa * abs(float(Rm_[di[a], di[b]])) for a, b in pairs],
                                 dtype=torch.float32, device=device)
            ctx["dep_terms"] = (da, db, floor)
    ctx["colls"] = [(node_idx[p1], node_idx[p2], node_idx[c]) for p1, p2, c in g.v_structures()] \
        if lam_coll > 0 else []

    ctx["gen_op"] = None
    ctx["neg_op"] = None
    ctx["neg_parents"] = []
    if gen_op is not None:
        gen_op = gen_op.to(device)
        ctx["gen_op"] = gen_op
        neg_op = None
        ge = [(p, n, float(W.get((p, n), 0.0))) for n in gen_nodes for p in g.parents(n)]
        ctx["ge_par"] = [p for p, n, w in ge]
        lat = set(g.latents)
        ctx["ge_cond"] = torch.tensor([[1.0 if w >= 0 else -1.0, abs(w),
                                        1.0 if (p in lat and n in lat) else 0.0] for p, n, w in ge],
                                      dtype=torch.float32, device=device)
        ctx["ge_absw"] = torch.tensor([abs(w) for p, n, w in ge], dtype=torch.float32, device=device)
        ge_child = {}
        for r, (p, n, w) in enumerate(ge):
            ge_child.setdefault(n, []).append(r)
        ctx["ge_child"] = ge_child
    if neg_op is not None:
        ctx["neg_op"] = neg_op.to(device)
        ctx["neg_parents"] = sorted({p for (p, n), w in W.items() if w < 0})

    ctx["br_terms"] = None
    if bridge is not None and bridge.get("lam_upper", 0) > 0:
        bi = {n: k for k, n in enumerate(bridge["obs"])}
        Dm = np.asarray(bridge["dep_marg"])
        tp = [(a, b) for a, b in g.trek_pairs() if a in bi and b in bi]
        if tp:
            vals = np.array([Dm[bi[a], bi[b]] for a, b in tp])
            thr = np.quantile(vals, bridge.get("q", 0.7))
            keep = [(a, b, v) for (a, b), v in zip(tp, vals) if v >= thr]
            if keep:
                fixed_bridge_cos = []
                for a, b, _ in keep:
                    if a not in A or b not in A:
                        continue
                    va = np.asarray(A[a], dtype=float) - center_np
                    vb = np.asarray(A[b], dtype=float) - center_np
                    denom = np.linalg.norm(va) * np.linalg.norm(vb)
                    if denom >= 1e-12:
                        fixed_bridge_cos.append(abs(float(np.dot(va, vb) / denom)))
                active = [(a, b, v) for a, b, v in keep if a in free_set or b in free_set]
                calibrated = float(np.median(fixed_bridge_cos)) if fixed_bridge_cos else None
                ctx["bridge_pair_counts"] = {
                    "selected": len(keep), "active": len(active),
                    "fixed_fixed": len(keep) - len(active), "calibrated_floor": calibrated,
                }
                if active:
                    ba = torch.tensor([node_idx[a] for a, b, v in active],
                                      dtype=torch.long, device=device)
                    bb = torch.tensor([node_idx[b] for a, b, v in active],
                                      dtype=torch.long, device=device)
                    bfloor = torch.tensor([
                        calibrated if calibrated is not None
                        else bridge.get("kappa", 0.5) * float(v)
                        for _, _, v in active
                    ], dtype=torch.float32, device=device)
                    ctx["br_terms"] = (ba, bb, bfloor, float(bridge["lam_upper"]))

    ctx["item_nodes"] = []
    ctx["item_idx"] = torch.zeros(0, dtype=torch.long, device=device)
    ctx["item_targets"] = torch.zeros((0, d), dtype=torch.float32, device=device)
    ctx["item_confidence"] = torch.zeros(0, dtype=torch.float32, device=device)
    ctx["profile_nodes"] = []
    ctx["profile_hi"] = torch.zeros((0, d), dtype=torch.float32, device=device)
    ctx["profile_lo"] = torch.zeros((0, d), dtype=torch.float32, device=device)
    ctx["profile_confidence"] = torch.zeros(0, dtype=torch.float32, device=device)
    if item_info is not None and item_info.get("nodes"):
        ctx["item_nodes"] = list(item_info["nodes"])
        ctx["item_idx"] = torch.tensor([node_idx[n] for n in ctx["item_nodes"]],
                                        dtype=torch.long, device=device)
        ctx["item_targets"] = torch.tensor(item_info["targets"], dtype=torch.float32,
                                            device=device)
        ctx["item_confidence"] = torch.tensor(item_info["confidence"], dtype=torch.float32,
                                               device=device)
    if item_info is not None and item_info.get("profile_nodes"):
        ctx["profile_nodes"] = list(item_info["profile_nodes"])
        ctx["profile_hi"] = torch.tensor(item_info["profile_hi"], dtype=torch.float32,
                                          device=device)
        ctx["profile_lo"] = torch.tensor(item_info["profile_lo"], dtype=torch.float32,
                                          device=device)
        ctx["profile_confidence"] = torch.tensor(
            item_info["profile_confidence"], dtype=torch.float32, device=device)
    return ctx


# --------------------------------------------------------------------------- shared objective
def step_loss(ctx, emb, wt, E, Rv, lam_zero, lam_norm, lam_dep, lam_coll,
              lam_item=1.0, nw=None):
    """One evaluation of the full stage-2 objective. emb(n) -> tensor; wt(edge) -> weight tensor;
    E: dict free-node -> tensor (norm term); Rv: dict gen-node -> residual tensor (or None).
    nw=None: the normalized scalar objective with all multipliers equal to one.
    nw=dict of per-node MULTIPLIER tensors (1.0 = frozen): {"gen": [len(gen_nodes)],
    "resnorm": [len(gen_nodes)], "anchor": [len(pc_nodes)], "node": [len(all_nodes)]
    (pair terms use the two endpoints' mean), "norm": [len(free)],
    "item": [len(reliable masked observed nodes)]}."""
    import torch
    W = ctx["W"]
    gen_nodes = ctx["gen_nodes"]
    At = ctx["At"]
    use_res = ctx["use_res"] and Rv is not None
    loss = 0.0
    neg_cache = {}
    if ctx["neg_parents"]:
        NP = ctx["neg_op"](torch.stack([emb(p) for p in ctx["neg_parents"]]))
        neg_cache = {p: NP[i] for i, p in enumerate(ctx["neg_parents"])}
    gout = None
    if ctx["gen_op"] is not None:
        Xp = torch.stack([emb(p) for p in ctx["ge_par"]])
        gout = ctx["ge_absw"][:, None] * ctx["gen_op"](Xp, ctx["ge_cond"])
    gen_terms = []
    for k, n in enumerate(gen_nodes):
        if gout is not None:
            tot = gout[ctx["ge_child"][n]].sum(0)
        else:
            tot = None
            for p in ctx["parents"][n]:
                w_e = wt((p, n))
                if p in neg_cache and float(W.get((p, n), 0.0)) < 0:
                    t = torch.abs(w_e) * neg_cache[p]
                else:
                    t = w_e * emb(p)
                tot = t if tot is None else tot + t
        if use_res:
            tot = tot + Rv[n]
        tgt = At[n] if n in At else E[n]
        term = ((tgt - tot) ** 2).sum()
        gen_terms.append(term if nw is None else nw["gen"][k] * term)
    if gen_terms:
        # Every structured term is an average over its applicable nodes/pairs.  In particular,
        # generation no longer grows linearly with graph width.
        loss = loss + torch.stack(gen_terms).mean()
    if use_res:
        Rn = torch.stack([Rv[n] for n in gen_nodes])
        rn2 = (Rn ** 2).sum(1)
        loss = loss + ctx["residual"] * (rn2.mean() if nw is None else (nw["resnorm"] * rn2).mean())
        if len(ctx["pc_nodes"]) > 1:
            Rm = torch.stack([Rv[n] for n in ctx["pc_nodes"]])
            Rm = torch.nn.functional.normalize(Rm, dim=1)
            if ctx["pc_pair_terms"] is not None:
                pa, pb, target, reliability = ctx["pc_pair_terms"]
                aerr = ((Rm[pa] * Rm[pb]).sum(1) - target) ** 2
                if nw is None:
                    weighted = (reliability * aerr).sum() / reliability.sum().clamp(min=1e-9)
                else:
                    wa = nw["anchor"]
                    pair_weight = 0.5 * (wa[pa] + wa[pb])
                    weighted = (reliability * pair_weight * aerr).sum() / \
                        reliability.sum().clamp(min=1e-9)
                loss = loss + ctx["lam_res"] * weighted
            else:
                aerr = ((Rm @ Rm.T) - ctx["Pt"]) ** 2
                if nw is None:
                    loss = loss + ctx["lam_res"] * aerr[ctx["offdiag"]].mean()
                else:
                    wa = nw["anchor"]
                    pw = 0.5 * (wa[:, None] + wa[None, :])
                    loss = loss + ctx["lam_res"] * (pw * aerr)[ctx["offdiag"]].mean()
    need_M = (len(ctx["zp_pairs"]) and lam_zero > 0) or ctx["dep_terms"] is not None \
        or ctx["colls"] or ctx["br_terms"] is not None
    pairw = None
    if need_M:
        M = torch.stack([emb(n) for n in ctx["all_nodes"]])
        Mn = torch.nn.functional.normalize(M - ctx["semantic_center"], dim=1)
        pairw = None if nw is None else nw["node"]
    if len(ctx["zp_pairs"]) and lam_zero > 0:
        t = ((Mn[ctx["ia"]] * Mn[ctx["ib"]]).sum(1) - ctx["zero_target"]) ** 2
        if pairw is not None:
            t = 0.5 * (pairw[ctx["ia"]] + pairw[ctx["ib"]]) * t
        loss = loss + lam_zero * t.mean()
    if ctx["dep_terms"] is not None:
        da, db, floor = ctx["dep_terms"]
        cs = (Mn[da] * Mn[db]).sum(1).abs()
        t = torch.relu(floor - cs) ** 2
        if pairw is not None:
            t = 0.5 * (pairw[da] + pairw[db]) * t
        loss = loss + lam_dep * t.mean()
    if ctx["br_terms"] is not None:
        ba, bb, bfloor, lam_up = ctx["br_terms"]
        cs = (Mn[ba] * Mn[bb]).sum(1).abs()
        t = torch.relu(bfloor - cs) ** 2
        if pairw is not None:
            t = 0.5 * (pairw[ba] + pairw[bb]) * t
        loss = loss + lam_up * t.mean()
    if ctx["colls"]:
        cl = 0.0
        for i1, i2, ic in ctx["colls"]:
            u1 = Mn[i1] - (Mn[i1] @ Mn[ic]) * Mn[ic]
            u2 = Mn[i2] - (Mn[i2] @ Mn[ic]) * Mn[ic]
            cnum = (u1 @ u2) / (u1.norm() * u2.norm() + 1e-9)
            cl = cl + torch.relu(cnum) ** 2
        loss = loss + lam_coll * cl / len(ctx["colls"])
    if ctx["item_nodes"] and lam_item > 0:
        item_emb = torch.stack([emb(n) for n in ctx["item_nodes"]])
        item_emb = torch.nn.functional.normalize(item_emb, dim=1)
        target = torch.nn.functional.normalize(ctx["item_targets"], dim=1)
        # Candidate reliability has already gated the node and weighted the two-pole target.
        # Applying max-|rho| again here systematically weakens wide single-factor scales.
        t = 1.0 - (item_emb * target).sum(1)
        if nw is not None:
            t = nw["item"] * t
        loss = loss + lam_item * t.mean()
    if ctx["profile_nodes"] and lam_item > 0:
        profile_emb = torch.stack([emb(n) for n in ctx["profile_nodes"]])
        profile_emb = torch.nn.functional.normalize(
            profile_emb - ctx["semantic_center"], dim=1)
        hi = torch.nn.functional.normalize(
            ctx["profile_hi"] - ctx["semantic_center"], dim=1)
        lo = torch.nn.functional.normalize(
            ctx["profile_lo"] - ctx["semantic_center"], dim=1)
        similarity_gap = (profile_emb * hi).sum(1) - (profile_emb * lo).sum(1)
        t = ctx["profile_confidence"] * torch.relu(-similarity_gap) ** 2
        if nw is not None:
            ni = torch.tensor([ctx["node_idx"][n] for n in ctx["profile_nodes"]],
                              dtype=torch.long, device=t.device)
            t = nw["node"][ni] * t
        loss = loss + lam_item * t.mean()
    if lam_norm > 0:
        nr = torch.stack([E[n].norm() for n in ctx["free"]])
        t = (nr - 1.0) ** 2
        if nw is not None:
            t = nw["norm"] * t
        loss = loss + lam_norm * t.mean()
    return loss


def _stage1(g, W, labeled_emb, d):
    labeled = set(labeled_emb)
    free = [n for n in g.nodes if n not in labeled]
    A = {n: np.asarray(v, np.float64) for n, v in labeled_emb.items()}
    E0 = O._solve_embeddings(g, dict(W), A, free, d) if free else {}
    return free, A, E0


# --------------------------------------------------------------------------- frozen-path replica
def solve_frozen(g, W, labeled_emb, d, steps=400, lr=2e-2, lam_zero=0.3, lam_norm=0.1, seed=0,
                 device="cpu", residual=0.0, lam_res=0.0, partial_corr=None,
                 lam_dep=0.0, dep_corr=None, dep_kappa=0.5, lam_coll=0.0,
                 neg_op=None, bridge=None, lam_item=1.0, n_samples=None,
                 independent_info=None, item_info=None, item_corr=None,
                 residual_pair_info=None):
    """Scalar-weight reference solve built on the shared normalized context/objective."""
    import torch
    torch.manual_seed(seed)
    free, A, E0 = _stage1(g, W, labeled_emb, d)
    if not free:
        return dict(labeled_emb)
    if item_info is None:
        item_info = prepare_item_identity(g, labeled_emb, item_corr, n_samples,
                                          neg_op=neg_op, device=device)
    ctx = build_ctx(g, W, dict(W), A, free, d, seed, device,
                    residual, lam_res, partial_corr, lam_dep, dep_corr, dep_kappa, lam_coll,
                    neg_op, bridge, None, independent_info=independent_info,
                    item_info=item_info, residual_pair_info=residual_pair_info)
    E = {n: torch.nn.Parameter(torch.tensor(E0[n], dtype=torch.float32, device=device)) for n in free}
    params = list(E.values())
    wt_const = ctx["wt_const"]

    def wt(e):
        return wt_const[e]

    Rv = None
    if ctx["use_res"]:
        Rv = {n: torch.nn.Parameter(torch.tensor(v, dtype=torch.float32, device=device))
              for n, v in ctx["Rv0"].items()}
        params += list(Rv.values())
    At = ctx["At"]

    def emb(n):
        return At[n] if n in At else E[n]

    opt = torch.optim.Adam(params, lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = step_loss(ctx, emb, wt, E, Rv, lam_zero, lam_norm, lam_dep, lam_coll,
                         lam_item=lam_item)
        loss.backward()
        opt.step()
    out = {n: v.detach().cpu().numpy().astype(np.float64) for n, v in E.items()}
    out.update({n: np.asarray(v, np.float64) for n, v in labeled_emb.items()})
    return out


# --------------------------------------------------------------------------- L2 unrolled solver
def solve_unrolled(g, W, labeled_emb, d, weight_module=None, K=120, inner_lr=2e-2,
                   lam_zero=0.3, lam_norm=0.1, seed=0, device="cpu",
                   residual=0.0, lam_res=0.0, partial_corr=None,
                   lam_dep=0.0, dep_corr=None, dep_kappa=0.5, lam_coll=0.0,
                   neg_op=None, bridge=None, lam_item=1.0, n_samples=None,
                   independent_info=None, item_info=None, train=False, feats=None,
                   truncation_steps=60, item_corr=None, residual_pair_info=None):
    """ALS init + K functional-Adam steps. Differentiable end-to-end when train=True (gradients
    reach weight_module through every unrolled step). weight_module(feats, ctx) -> nw dict of
    per-node multipliers (see step_loss); None -> multipliers 1 (solver-dynamics control).
    Returns (emb_dict, tensors) where tensors maps free node -> final torch tensor (kept on graph
    when train=True so the caller can build the outer loss without a numpy round-trip).

    Training defaults to 2x60-step truncated backpropagation for K=120: after step 60 the
    embedding, residual, and functional-Adam histories are detached without changing their
    numerical values.  The second segment still depends on the same weight-module output.  Set
    ``truncation_steps=None`` for an ordinary full backward graph; inference always executes all K
    numerical steps.
    """
    import torch
    torch.manual_seed(seed)
    free, A, E0 = _stage1(g, W, labeled_emb, d)
    if not free:
        return dict(labeled_emb), {}
    if item_info is None:
        item_info = prepare_item_identity(g, labeled_emb, item_corr, n_samples,
                                          neg_op=neg_op, device=device)
    ctx = build_ctx(g, W, dict(W), A, free, d, seed, device,
                    residual, lam_res, partial_corr, lam_dep, dep_corr, dep_kappa, lam_coll,
                    neg_op, bridge, None, independent_info=independent_info,
                    item_info=item_info, residual_pair_info=residual_pair_info)
    nw = None
    if weight_module is not None:
        nw = weight_module(feats, ctx)

    P = {n: torch.tensor(E0[n], dtype=torch.float32, device=device) for n in free}
    Rv = {n: torch.tensor(v, dtype=torch.float32, device=device)
          for n, v in ctx.get("Rv0", {}).items()} if ctx["use_res"] else None
    def flat():
        ps = [P[n] for n in P] + ([Rv[n] for n in Rv] if Rv else [])
        return ps

    wt_const = ctx["wt_const"]

    def wt(e):
        return wt_const[e]

    At = ctx["At"]
    # functional Adam state (out-of-place; differentiable w.r.t. the weight module when train=True)
    b1, b2, eps = 0.9, 0.999, 1e-8
    ps = flat()
    m = [torch.zeros_like(p) for p in ps]
    v = [torch.zeros_like(p) for p in ps]
    for p in ps:
        p.requires_grad_(True)
    for step in range(1, K + 1):
        def emb(n):
            return At[n] if n in At else P[n]
        loss = step_loss(ctx, emb, wt, P, Rv, lam_zero, lam_norm, lam_dep, lam_coll,
                         lam_item=lam_item, nw=nw)
        grads = torch.autograd.grad(loss, ps, create_graph=train)
        new_ps = []
        for i, (p, gr) in enumerate(zip(ps, grads)):
            m[i] = b1 * m[i] + (1 - b1) * gr
            v[i] = b2 * v[i] + (1 - b2) * gr * gr
            mh = m[i] / (1 - b1 ** step)
            vh = v[i] / (1 - b2 ** step)
            # sqrt(v + eps^2) is numerically equivalent to Adam's sqrt(v)+eps floor at v=0,
            # while avoiding an infinite second derivative at exactly-zero gradient during the
            # differentiable outer solve.
            new_ps.append(p - inner_lr * mh / torch.sqrt(vh + eps * eps))
        ps = new_ps
        if not train:
            ps = [p.detach().requires_grad_(True) for p in ps]
            m = [t.detach() for t in m]
            v = [t.detach() for t in v]
        elif truncation_steps and step % truncation_steps == 0 and step < K:
            # Preserve forward values exactly while cutting only the history before this boundary.
            ps = [p.detach().requires_grad_(True) for p in ps]
            m = [t.detach() for t in m]
            v = [t.detach() for t in v]
        k = 0
        for n in list(P.keys()):
            P[n] = ps[k]; k += 1
        if Rv is not None:
            for n in list(Rv.keys()):
                Rv[n] = ps[k]; k += 1
    tensors = dict(P)
    out = {n: t.detach().cpu().numpy().astype(np.float64) for n, t in P.items()}
    out.update({n: np.asarray(vv, np.float64) for n, vv in labeled_emb.items()})
    return out, tensors
