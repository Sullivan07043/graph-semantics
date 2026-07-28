"""v6 CORE solver — the term-factory objective around the Jacobian-locked nonlinear operator.

Supersedes v5's l2_solver.py (removed from v6; the v5 tree keeps the lineage). One objective,
assembled from terms.py tables (THEORY Definition 2):

  R1  generation through gen_operator.GenOperator — THE generation path; there is no separate
      linear branch. The untrained (zero-init) operator reproduces v5's linear+f_neg exactly.
  R2/R3  unified conditional-independence rule (terms.ci_table): residualized-cosine matching
      on every non-ancestral pair given its common-ancestor set; S = empty set is the old
      marginal decorrelation. Weighted by lam_zero (name kept from v5 for runner compat).
  R2 upper tail  dependence floor on trek pairs (terms.dep_floor_table).
  R3 at S = pa  residual channel + partial-correlation Gram anchor (unchanged from v5).
  R4  labeled anchors fixed by parameterization; unit-norm term on free nodes.

Solver: ALS linear init (valid initializer per THEORY §4.3) + K functional-Adam steps,
differentiable end-to-end with train=True (gradients reach BOTH the weight module and the
generation operator through every unrolled step — the TRUNK-4 training path).

WeightNet interface is preserved: weight_module(feats, ctx) -> nw dict with channels
gen/resnorm/anchor/node/norm (l2_modules._slice_nw ctx keys unchanged); CI and floor pair
terms ride the "node" channel via endpoint means, exactly as v5's pair terms did.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import optimize as O                                                  # noqa: E402
import terms as TF                                                    # noqa: E402
import gen_operator as GO                                             # noqa: E402


# --------------------------------------------------------------------------- static context
def build_ctx(g, W, A, free, d, seed, device,
              residual, lam_res, partial_corr, bridge, ci, gen_op):
    """All static precomputation (no Parameters created here)."""
    import torch
    ctx = {"g": g, "W": W, "free": free, "d": d, "device": device}
    ctx["all_nodes"] = list(g.nodes)
    ctx["At"] = {n: torch.tensor(v, dtype=torch.float32, device=device) for n, v in A.items()}
    node_idx = {n: i for i, n in enumerate(g.nodes)}
    ctx["node_idx"] = node_idx
    gen_nodes = [n for n in g.nodes if g.parents(n)]
    ctx["gen_nodes"] = gen_nodes

    ctx["gen_op"] = gen_op.to(device)
    ctx["edge_par"], ctx["edge_cond"], ctx["child_rows"] = GO.edge_table(g, W, device=device)

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

    ctx["ci_groups"] = []
    ctx["ci_count"] = 0
    if ci:
        for S, ia, ib, tg in TF.ci_tensors(ci, node_idx, device=device):
            S_idx = torch.tensor([node_idx[s] for s in S], dtype=torch.long, device=device) \
                if S else None
            ctx["ci_groups"].append((S_idx, ia, ib, tg))
            ctx["ci_count"] += len(tg)

    ctx["br_terms"] = TF.dep_floor_table(g, node_idx, bridge, device=device)
    return ctx


# --------------------------------------------------------------------------- objective
def step_loss(ctx, emb, E, Rv, lam_zero, lam_norm, nw=None):
    """One evaluation of the full objective. emb(n) -> tensor; E: dict free node -> tensor;
    Rv: dict gen node -> residual tensor (or None). nw=None: multipliers 1; else the
    l2_modules nw dict (channels gen/resnorm/anchor/node/norm)."""
    import torch
    gen_nodes = ctx["gen_nodes"]
    At = ctx["At"]
    use_res = ctx["use_res"] and Rv is not None
    loss = 0.0

    # R1 generation: one batched operator forward per step over all edges
    Xp = torch.stack([emb(p) for p in ctx["edge_par"]])
    contrib = ctx["gen_op"](Xp, ctx["edge_cond"])
    for k, n in enumerate(gen_nodes):
        tot = contrib[ctx["child_rows"][n]].sum(0)
        if use_res:
            tot = tot + Rv[n]
        tgt = At[n] if n in At else E[n]
        term = ((tgt - tot) ** 2).sum()
        loss = loss + (term if nw is None else nw["gen"][k] * term)

    # R3 at S = pa: residual norms + partial-correlation Gram anchor
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

    need_M = (ctx["ci_count"] and lam_zero > 0) or ctx["br_terms"] is not None
    pairw = None if nw is None else nw["node"]
    if need_M:
        M = torch.stack([emb(n) for n in ctx["all_nodes"]])

    # R2/R3 unified CI rule: residualized cosine -> shrunk partial correlation
    if ctx["ci_count"] and lam_zero > 0:
        tot_ci = 0.0
        for S_idx, ia, ib, tg in ctx["ci_groups"]:
            cs = TF.ci_cos(M, (S_idx, ia, ib))
            t = (cs - tg) ** 2
            if pairw is not None:
                t = 0.5 * (pairw[ia] + pairw[ib]) * t
            tot_ci = tot_ci + t.sum()
        loss = loss + lam_zero * tot_ci / ctx["ci_count"]

    # R2 upper tail: dependence floor on strongly-dependent trek pairs
    if ctx["br_terms"] is not None:
        Mn = torch.nn.functional.normalize(M, dim=1)
        ba, bb, bfloor, lam_up = ctx["br_terms"]
        cs = (Mn[ba] * Mn[bb]).sum(1).abs()
        t = torch.relu(bfloor - cs) ** 2
        if pairw is not None:
            t = 0.5 * (pairw[ba] + pairw[bb]) * t
        loss = loss + lam_up * t.mean()

    # R4 unit norm on free nodes
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


# --------------------------------------------------------------------------- unrolled solver
def solve_unrolled(g, W, labeled_emb, d, gen_op, ci, weight_module=None, K=60, inner_lr=2e-2,
                   lam_zero=0.3, lam_norm=0.1, seed=0, device="cpu",
                   residual=0.0, lam_res=0.0, partial_corr=None,
                   bridge=None, train=False, feats=None):
    """ALS linear init + K functional-Adam steps on the term-factory objective.
    gen_op: gen_operator.GenOperator (REQUIRED — the generation path).
    ci: terms.ci_table output (REQUIRED — the unified CI support).
    Differentiable end-to-end when train=True (gradients reach weight_module and gen_op).
    Returns (emb_dict, tensors)."""
    import torch
    assert gen_op is not None and ci is not None, "v6 core requires gen_op and ci tables"
    torch.manual_seed(seed)
    free, A, E0 = _stage1(g, W, labeled_emb, d)
    if not free:
        return dict(labeled_emb), {}
    ctx = build_ctx(g, W, A, free, d, seed, device,
                    residual, lam_res, partial_corr, bridge, ci, gen_op)
    nw = weight_module(feats, ctx) if weight_module is not None else None

    P = {n: torch.tensor(E0[n], dtype=torch.float32, device=device) for n in free}
    Rv = {n: torch.tensor(v, dtype=torch.float32, device=device)
          for n, v in ctx.get("Rv0", {}).items()} if ctx["use_res"] else None
    At = ctx["At"]

    b1, b2, eps = 0.9, 0.999, 1e-8
    ps = [P[n] for n in P] + ([Rv[n] for n in Rv] if Rv else [])
    m = [torch.zeros_like(p) for p in ps]
    v = [torch.zeros_like(p) for p in ps]
    for p in ps:
        p.requires_grad_(True)
    for step in range(1, K + 1):
        def emb(n):
            return At[n] if n in At else P[n]
        loss = step_loss(ctx, emb, P, Rv, lam_zero, lam_norm, nw=nw)
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
