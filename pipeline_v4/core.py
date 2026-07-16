"""PIPELINE V4 — L2 learned (unrolled) optimizer. Lives entirely in pipeline_v4/; the frozen
pipeline (optimize.py and the runners) is NOT touched.

Two solvers over the SAME objective:
  solve_frozen   — reproduces optimize.optimize_embeddings stage 2 (ALS init + Adam, scalar lambdas)
                   operation-for-operation. Exists only to prove the objective port is exact
                   (identity check against outputs/refcheck_prerefactor.npz).
  solve_unrolled — the L2 solver: ALS init + K functional-Adam steps, fully differentiable, with
                   per-node constraint-weight MULTIPLIERS coming from a weight module. With the
                   identity module (all multipliers 1) it is the frozen objective under different
                   solver dynamics (the solver-dynamics control arm).

Weight modules (l2_modules.py): StaticWeights (5 learned scalars, attribution control) and
WeightNet (node-context MLP, the main nonlinear arm; zero-init head => multipliers start at 1).
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root
sys.path.insert(0, HERE)
import optimize as O                                                  # noqa: E402

TERMS = ["gen", "resnorm", "anchor", "node", "norm"]


# --------------------------------------------------------------------------- shared precomputation
def build_ctx(g, W, Wcur, A, free, d, seed, device,
              residual, lam_res, partial_corr, lam_dep, dep_corr, dep_kappa, lam_coll,
              neg_op, bridge, gen_op):
    """All static precomputation for the stage-2 objective (no Parameters created here).
    Port of the corresponding block of optimize.optimize_embeddings; the objective terms consumed
    by step_loss below are the same operations in the same order."""
    import torch
    ctx = {"g": g, "W": W, "free": free, "d": d, "device": device}
    ctx["all_nodes"] = list(g.nodes)
    ctx["At"] = {n: torch.tensor(v, dtype=torch.float32, device=device) for n, v in A.items()}
    gen_nodes = [n for n in g.nodes if g.parents(n)]
    ctx["gen_nodes"] = gen_nodes
    ctx["parents"] = {n: list(g.parents(n)) for n in gen_nodes}
    ctx["wt_const"] = {e: torch.tensor(float(v), dtype=torch.float32, device=device)
                       for e, v in Wcur.items()}

    ctx["use_res"] = residual > 0
    ctx["residual"] = residual
    ctx["lam_res"] = lam_res
    ctx["pc_nodes"] = []
    if ctx["use_res"]:
        gnr = np.random.default_rng(seed)
        ctx["Rv0"] = {n: gnr.normal(0, 1e-3, d) for n in gen_nodes}
        if lam_res > 0 and partial_corr is not None:
            pc_names, Pmat = partial_corr
            rv_set = set(gen_nodes)
            pc_nodes = [n for n in pc_names if n in rv_set]
            pidx = [list(pc_names).index(n) for n in pc_nodes]
            ctx["pc_nodes"] = pc_nodes
            ctx["Pt"] = torch.tensor(Pmat[np.ix_(pidx, pidx)], dtype=torch.float32, device=device)
            ctx["offdiag"] = ~torch.eye(len(pc_nodes), dtype=torch.bool, device=device)

    node_idx = {n: i for i, n in enumerate(g.nodes)}
    ctx["node_idx"] = node_idx
    zp_pairs = g.independent_pairs()
    ctx["zp_pairs"] = zp_pairs
    ctx["ia"] = torch.tensor([node_idx[a] for a, b in zp_pairs], dtype=torch.long, device=device)
    ctx["ib"] = torch.tensor([node_idx[b] for a, b in zp_pairs], dtype=torch.long, device=device)

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
                ba = torch.tensor([node_idx[a] for a, b, v in keep], dtype=torch.long, device=device)
                bb = torch.tensor([node_idx[b] for a, b, v in keep], dtype=torch.long, device=device)
                bfloor = torch.tensor([bridge.get("kappa", 0.5) * float(v) for a, b, v in keep],
                                      dtype=torch.float32, device=device)
                ctx["br_terms"] = (ba, bb, bfloor, float(bridge["lam_upper"]))
    return ctx


# --------------------------------------------------------------------------- shared objective
def step_loss(ctx, emb, wt, E, Rv, lam_zero, lam_norm, lam_dep, lam_coll, nw=None):
    """One evaluation of the full stage-2 objective. emb(n) -> tensor; wt(edge) -> weight tensor;
    E: dict free-node -> tensor (norm term); Rv: dict gen-node -> residual tensor (or None).
    nw=None: the original scalar objective, operation-for-operation.
    nw=dict of per-node MULTIPLIER tensors (1.0 = frozen): {"gen": [len(gen_nodes)],
    "resnorm": [len(gen_nodes)], "anchor": [len(pc_nodes)], "node": [len(all_nodes)]
    (pair terms use the two endpoints' mean), "norm": [len(free)]}."""
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
        loss = loss + (term if nw is None else nw["gen"][k] * term)
    if use_res:
        Rn = torch.stack([Rv[n] for n in gen_nodes])
        rn2 = (Rn ** 2).sum(1)
        loss = loss + ctx["residual"] * (rn2.mean() if nw is None else (nw["resnorm"] * rn2).mean())
        if len(ctx["pc_nodes"]) > 1:
            Rm = torch.stack([Rv[n] for n in ctx["pc_nodes"]])
            Rm = torch.nn.functional.normalize(Rm, dim=1)
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
        Mn = torch.nn.functional.normalize(M, dim=1)
        pairw = None if nw is None else nw["node"]
    if len(ctx["zp_pairs"]) and lam_zero > 0:
        t = ((Mn[ctx["ia"]] * Mn[ctx["ib"]]).sum(1)) ** 2
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
                 neg_op=None, bridge=None):
    """Exact replica of optimize.optimize_embeddings (free_w=False, gen_op=None) built on the
    shared ctx/objective. Used ONLY for the identity check."""
    import torch
    torch.manual_seed(seed)
    free, A, E0 = _stage1(g, W, labeled_emb, d)
    if not free:
        return dict(labeled_emb)
    ctx = build_ctx(g, W, dict(W), A, free, d, seed, device,
                    residual, lam_res, partial_corr, lam_dep, dep_corr, dep_kappa, lam_coll,
                    neg_op, bridge, None)
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
        loss = step_loss(ctx, emb, wt, E, Rv, lam_zero, lam_norm, lam_dep, lam_coll)
        loss.backward()
        opt.step()
    out = {n: v.detach().cpu().numpy().astype(np.float64) for n, v in E.items()}
    out.update({n: np.asarray(v, np.float64) for n, v in labeled_emb.items()})
    return out


# --------------------------------------------------------------------------- L2 unrolled solver
def solve_unrolled(g, W, labeled_emb, d, weight_module=None, K=60, inner_lr=2e-2,
                   lam_zero=0.3, lam_norm=0.1, seed=0, device="cpu",
                   residual=0.0, lam_res=0.0, partial_corr=None,
                   lam_dep=0.0, dep_corr=None, dep_kappa=0.5, lam_coll=0.0,
                   neg_op=None, bridge=None, train=False, feats=None):
    """ALS init + K functional-Adam steps. Differentiable end-to-end when train=True (gradients
    reach weight_module through every unrolled step). weight_module(feats, ctx) -> nw dict of
    per-node multipliers (see step_loss); None -> multipliers 1 (solver-dynamics control).
    Returns (emb_dict, tensors) where tensors maps free node -> final torch tensor (kept on graph
    when train=True so the caller can build the outer loss without a numpy round-trip)."""
    import torch
    torch.manual_seed(seed)
    free, A, E0 = _stage1(g, W, labeled_emb, d)
    if not free:
        return dict(labeled_emb), {}
    ctx = build_ctx(g, W, dict(W), A, free, d, seed, device,
                    residual, lam_res, partial_corr, lam_dep, dep_corr, dep_kappa, lam_coll,
                    neg_op, bridge, None)
    nw = None
    if weight_module is not None:
        nw = weight_module(feats, ctx)

    P = {n: torch.tensor(E0[n], dtype=torch.float32, device=device) for n in free}
    Rv = {n: torch.tensor(v, dtype=torch.float32, device=device)
          for n, v in ctx.get("Rv0", {}).items()} if ctx["use_res"] else None
    names = list(P.keys()) + ([f"r::{n}" for n in Rv] if Rv else [])

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
        loss = step_loss(ctx, emb, wt, P, Rv, lam_zero, lam_norm, lam_dep, lam_coll, nw=nw)
        grads = torch.autograd.grad(loss, ps, create_graph=train)
        new_ps = []
        for i, (p, gr) in enumerate(zip(ps, grads)):
            m[i] = b1 * m[i] + (1 - b1) * gr
            v[i] = b2 * v[i] + (1 - b2) * gr * gr
            mh = m[i] / (1 - b1 ** step)
            vh = v[i] / (1 - b2 ** step)
            new_ps.append(p - inner_lr * mh / (vh.sqrt() + eps))
        ps = new_ps
        if not train:
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
