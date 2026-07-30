"""P2 — per-node certainty score (THEORY Definition 4). Read-only post-solve diagnostic.

cert(i) = sigma_min(J P_i): the smallest singular value of the Jacobian of the ACTIVE
constraint set at the solution, restricted to variations of e_i. Low cert(i) = the evidence
leaves directions of node i unfixed (null-space freedom; THEORY §2.4 pathologies).

Rows of J P_i (rows not involving e_i are zero in the i-block and drop out of sigma_min):
  R1  node i's own generation equation (identity block, if i has parents) and, for every child
      c of i, the negated operator Jacobian -dT(e_i; cond_ic)/de_i (exact autograd Jacobian
      through Delta AND f_neg). The residual channel is the moment-condition slack and is not a
      constraint on e_i (declared).
  R2/R3  every CI pair whose endpoint OR conditioning set contains i: gradient row of
      cos(res_S e_a, res_S e_b) w.r.t. e_i.
  R2 floor  ACTIVE hinge pairs only (|cos| < floor) touching i.
  R4  the unit-norm row e_i / ||e_i||.

Falsification tests this score must reproduce (THEORY §2.3): (a) riasec cross-run variance on
type nodes, (b) metatraits lowest among latents, (c) correlation with judge correctness on dev.
Running those tests is a user-approved step (TRUNK-4 / P2 validation), not done here.
"""
import numpy as np
import torch

import terms as TF
import gen_operator as GO


def _edge_jacobian(gen_op, e_p, cond_row):
    """Exact d x d Jacobian of T(e; cond) at e_p (both branches of the sign action are smooth)."""
    def f(e):
        c = cond_row.unsqueeze(0)
        base = e.unsqueeze(0) if float(cond_row[0]) >= 0 else gen_op.neg(e.unsqueeze(0))
        return (c[:, 1:2] * (base + gen_op.delta(torch.cat([e.unsqueeze(0), c], 1))))[0]
    return torch.autograd.functional.jacobian(f, e_p, vectorize=True)


def compute(g, W, emb, gen_op, ci, labeled, bridge=None):
    """emb: dict node -> np.array (FULL solution). labeled: labeled node names (not free).
    -> dict free node -> cert (float). Pure diagnostic; nothing is modified."""
    free = [n for n in g.nodes if n not in set(labeled)]
    d = len(next(iter(emb.values())))
    E = {n: torch.tensor(np.asarray(emb[n], np.float64), dtype=torch.float32,
                         requires_grad=(n in set(free))) for n in g.nodes}
    node_idx = {n: i for i, n in enumerate(g.nodes)}
    rows = {n: [] for n in free}

    # R1: generation rows
    par, cond, child_rows = GO.edge_table(g, W)
    for n in free:
        if g.parents(n):
            rows[n].append(torch.eye(d))
    r = 0
    for c in [x for x in g.nodes if g.parents(x)]:
        for p in g.parents(c):
            if p in rows:
                rows[p].append(-_edge_jacobian(gen_op, E[p].detach(), cond[r]))
            r += 1

    # R2/R3: CI pair rows (one backward per pair, distributed to every free node it touches)
    M = torch.stack([E[n] for n in g.nodes])
    for S, pairs, tg in ci:
        S_idx = torch.tensor([node_idx[s] for s in S], dtype=torch.long) if S else None
        touched_S = [s for s in S if s in rows]
        for (a, b), _t in zip(pairs, tg):
            leaves = [x for x in {a, b, *touched_S} if x in rows]
            if not leaves:
                continue
            ia = torch.tensor([node_idx[a]], dtype=torch.long)
            ib = torch.tensor([node_idx[b]], dtype=torch.long)
            f = TF.ci_cos(M, (S_idx, ia, ib))[0]
            gs = torch.autograd.grad(f, [E[x] for x in leaves], allow_unused=True,
                                     retain_graph=True)
            for x, gx in zip(leaves, gs):
                if gx is not None:
                    rows[x].append(gx.detach().unsqueeze(0))

    # R2 upper tail: ACTIVE floor hinges
    br = TF.dep_floor_table(g, node_idx, bridge) if bridge is not None else None
    if br is not None:
        ba, bb, bfloor, _lam = br
        Mn = torch.nn.functional.normalize(M, dim=1)
        cs = (Mn[ba] * Mn[bb]).sum(1).abs()
        for k in (cs < bfloor).nonzero(as_tuple=True)[0].tolist():
            a, b = g.nodes[int(ba[k])], g.nodes[int(bb[k])]
            leaves = [x for x in (a, b) if x in rows]
            if not leaves:
                continue
            f = bfloor[k] - (Mn[ba[k]] * Mn[bb[k]]).sum().abs()
            gs = torch.autograd.grad(f, [E[x] for x in leaves], allow_unused=True,
                                     retain_graph=True)
            for x, gx in zip(leaves, gs):
                if gx is not None:
                    rows[x].append(gx.detach().unsqueeze(0))

    # R4: unit-norm row
    out = {}
    for n in free:
        e = E[n].detach()
        rows[n].append((e / (e.norm() + 1e-9)).unsqueeze(0))
        J = torch.cat(rows[n], 0)
        out[n] = float(torch.linalg.svdvals(J.double())[-1])
    return out
