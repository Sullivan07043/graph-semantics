"""P3 — influence-weighted decoding for deep latents (THEORY §1.1 / §2.4). Read-only helper.

Object computed: the GENERATIVE influence of a latent u on every observed node — the Jacobian
of the generation composition along directed paths u -> ... -> o, evaluated at the solution
(Definition 1's B = E[dX/dZ] on the embedding side). This is the "direct influence readout
across two hops" of THEORY §2.4: first-order evidence for metatraits that bypasses the flat
psi(psi(.)) curvature.

DECLARED DEVIATION from PLAN P3's wording ("d(observed solutions)/d(latent u) through the
differentiable solve"): through the solve, text-bearing (labeled) observed nodes are PINNED,
so that derivative is identically zero exactly where the footprint needs weight. The
generative-path Jacobian is the object T1 actually names, is nonzero on all descendants, and
is what §2.4 prescribes. Flagged to the user 2026-07-28.

Estimator: forward JVP of the operator composition. Probe tangents v at u propagate in
topological order: delta_c = sum_{p in pa(c)} JVP of T(. ; cond_pc) at e_p applied to delta_p.
weight(o) = mean over probes of ||delta_o||  (magnitude of influence);
sign(o)   = sign of corr(score_u, X_o)      (data-grounded polarity, as in W estimation).

Footprint and blend (decode inputs; decoding itself stays metrics.decode_words):
  footprint(u) = normalize( sum_o weight(o) * [e_o if sign(o) >= 0 else f_neg(e_o)] )
  blended(u)   = normalize( (1 - beta) * normalize(e_u) + beta * footprint(u) ), beta=0.5 default.
Intended use: latents >= 2 hops from any text-bearing node (metatraits); caller filters by
hop_depth(). No thresholds; both raw and blended embeddings are returned for evidence tables.
"""
import numpy as np
import torch


def hop_depth(g, u):
    """Min directed-path length from u to any observed node (np.inf if none reachable)."""
    obs = set(g.observed)
    depth, frontier, seen = 0, {u}, {u}
    while frontier:
        depth += 1
        nxt = {c for x in frontier for c in g.children(x) if c not in seen}
        if nxt & obs:
            return depth
        seen |= nxt
        frontier = nxt
    return float("inf")


def _jvp_edge(gen_op, e_p, cond_row, tang):
    def f(e):
        c = cond_row.unsqueeze(0)
        base = e.unsqueeze(0) if float(cond_row[0]) >= 0 else gen_op.neg(e.unsqueeze(0))
        return (c[:, 1:2] * (base + gen_op.delta(torch.cat([e.unsqueeze(0), c], 1))))[0]
    return torch.func.jvp(f, (e_p,), (tang,))[1]


def gen_influence(g, W, emb, gen_op, u, probes=8, seed=0):
    """-> dict observed node -> mean ||delta_o|| over unit probe tangents at u."""
    lat = set(g.latents)
    cond_of = {}
    for c in g.nodes:
        for p in g.parents(c):
            w = float(W.get((p, c), 0.0))
            cond_of[(p, c)] = torch.tensor(
                [1.0 if w >= 0 else -1.0, abs(w), 1.0 if (p in lat and c in lat) else 0.0])
    order = []
    seen = {u}
    frontier = [u]
    while frontier:                              # topological by BFS layers (DAG, small)
        nxt = []
        for x in frontier:
            for c in g.children(x):
                if c not in seen:
                    seen.add(c)
                    nxt.append(c)
        order += nxt
        frontier = nxt
    E = {n: torch.tensor(np.asarray(emb[n], np.float64), dtype=torch.float32) for n in g.nodes}
    d = E[u].shape[0]
    rng = np.random.default_rng(seed)
    acc = {o: 0.0 for o in g.observed}
    for _ in range(probes):
        v = rng.normal(size=d)
        v = torch.tensor(v / np.linalg.norm(v), dtype=torch.float32)
        delta = {u: v}
        for c in order:
            t = None
            for p in g.parents(c):
                if p in delta:
                    contrib = _jvp_edge(gen_op, E[p], cond_of[(p, c)], delta[p])
                    t = contrib if t is None else t + contrib
            if t is not None:
                delta[c] = t
        for o in g.observed:
            if o in delta:
                acc[o] += float(delta[o].norm())
    return {o: acc[o] / probes for o in g.observed if acc[o] > 0}


def data_signs(g, X, obs_index, score, u):
    """sign(corr(score_u, X_o)) per observed o (polarity source, same as W estimation)."""
    if u not in score:
        return {}
    s = np.asarray(score[u], float)
    out = {}
    for o in g.observed:
        c = np.corrcoef(s, X[:, obs_index[o]])[0, 1]
        out[o] = 1.0 if (np.isnan(c) or c >= 0) else -1.0
    return out


def footprint(emb, weights, signs, fneg=None):
    """normalize(sum_o w_o * [e_o | f_neg(e_o)]). fneg: frozen negop module (None => -e_o
    NEVER substituted; negative-sign nodes are simply skipped, declared)."""
    acc = None
    for o, w in weights.items():
        e = np.asarray(emb[o], np.float64)
        if signs.get(o, 1.0) < 0:
            if fneg is None:
                continue
            with torch.no_grad():
                e = fneg(torch.tensor(e, dtype=torch.float32).unsqueeze(0))[0].numpy()
        acc = w * e if acc is None else acc + w * e
    if acc is None:
        return None
    return acc / (np.linalg.norm(acc) + 1e-9)


def blended(e_u, f_u, beta=0.5):
    eu = np.asarray(e_u, np.float64)
    eu = eu / (np.linalg.norm(eu) + 1e-9)
    if f_u is None:
        return eu
    b = (1.0 - beta) * eu + beta * np.asarray(f_u, np.float64)
    return b / (np.linalg.norm(b) + 1e-9)
