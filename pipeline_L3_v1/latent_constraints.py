"""Latent-level constraints, landed (user order 2026-07-22). Three pieces:

1. sign_fix(g, W, score)      — upper-latent sign convention: walk latents bottom-up; if the
                                summed weight to LATENT children is negative, flip that latent's
                                score (the PC1 polarity followed the dominant child, which can be
                                the construct's NEGATIVE pole — the bigfive stability bug).
2. augmented_partial_corr(..) — observed residual correlations (existing) EXTENDED with latent
                                residuals (each latent score regressed on its latent parents'
                                scores). Feeding the combined (names, P) into the solver anchors
                                LATENT residual directions too (pc filter is name-membership).
3. augmented_bridge(..)       — dependence matrix over observed+latents (|corr| of scores for the
                                latent blocks). Trek-connected latent pairs then receive the
                                similarity lower bound exactly like observed pairs.

No solver changes: build_ctx filters constraint pairs by matrix-name membership, so augmented
inputs activate the latent terms automatically.
"""
import numpy as np


def _topo_latents_bottom_up(g):
    lats = list(g.latents)
    depth = {}

    def d(L):
        if L in depth:
            return depth[L]
        ps = [p for p in g.parents(L) if g.is_latent(p)]
        depth[L] = 0 if not ps else 1 + max(d(p) for p in ps)
        return depth[L]
    for L in lats:
        d(L)
    # bottom-up = leaves (children) first = larger depth first? depth counts distance from roots;
    # we flip parents AFTER their children are settled, so process by DESCENDING depth = children
    # of deep chains first, roots last.
    return sorted(lats, key=lambda L: -depth[L])


def sign_fix(g, W, score):
    """Flip upper-latent scores whose summed latent-child edge weight is negative; recompute the
    incident W entries from the flipped scores. Returns (W, score) updated copies."""
    W = dict(W)
    score = {k: v.copy() for k, v in score.items()}
    obs_cols = {}

    def corr(a, b):
        return float(np.corrcoef(a, b)[0, 1])

    for L in _topo_latents_bottom_up(g):
        lat_children = [c for c in g.nodes if g.is_latent(c) and L in g.parents(c)]
        if not lat_children:
            continue
        s = sum(W.get((L, c), 0.0) for c in lat_children)
        if s < 0:
            score[L] = -score[L]
            for c in lat_children:
                if (L, c) in W:
                    W[(L, c)] = corr(score[L], score[c])
            for p in g.parents(L):
                if (p, L) in W and p in score:
                    W[(p, L)] = corr(score[p], score[L])
            for c in g.nodes:
                if not g.is_latent(c) and L in g.parents(c):
                    pass  # bipartite latents have no observed children here; guard for generality
    return W, score


def augmented_partial_corr(g, X, obs_index, score, base_partial_corr):
    """Extend the observed residual-correlation anchor with latent rows.
    base_partial_corr: (obs_names, P) from optimize.partial_residual_corr.
    Latent residual = latent score regressed on its LATENT parents' scores (roots get z-scored
    score itself, which then carries no anchor weight against observeds beyond raw correlation)."""
    obs_names, P = base_partial_corr
    lat_gen = [L for L in g.latents
               if any(g.is_latent(p) for p in g.parents(L)) and L in score]
    if not lat_gen:
        return base_partial_corr
    n = X.shape[0]
    R_obs = np.zeros((n, len(obs_names)))
    for k, o in enumerate(obs_names):
        y = X[:, obs_index[o]]
        regs = [score[p] for p in g.parents(o) if g.is_latent(p) and p in score]
        if regs:
            A = np.stack(regs, 1)
            beta, *_ = np.linalg.lstsq(A, y, rcond=None)
            y = y - A @ beta
        R_obs[:, k] = y
    R_lat = np.zeros((n, len(lat_gen)))
    for k, L in enumerate(lat_gen):
        y = score[L].astype(float)
        regs = [score[p] for p in g.parents(L) if g.is_latent(p) and p in score]
        if regs:
            A = np.stack(regs, 1)
            beta, *_ = np.linalg.lstsq(A, y, rcond=None)
            y = y - A @ beta
        R_lat[:, k] = y
    R = np.concatenate([R_obs, R_lat], 1)
    R = R - R.mean(0)
    R = R / (R.std(0) + 1e-9)
    P_aug = np.corrcoef(R.T)
    np.fill_diagonal(P_aug, 0.0)
    return list(obs_names) + lat_gen, P_aug


def augmented_bridge(g, obs_names, obs_index, X, score, base_dep):
    """Dependence-magnitude matrix over observed+latents for the similarity lower bound.
    Observed block = base_dep (cached Pearson magnitudes); latent blocks = |corr| of latent
    scores with each other and with observed columns."""
    lats = [L for L in g.latents if L in score]
    m, k = len(obs_names), len(lats)
    D = np.zeros((m + k, m + k))
    D[:m, :m] = np.asarray(base_dep)
    S = np.stack([score[L] for L in lats], 1)
    Sn = (S - S.mean(0)) / (S.std(0) + 1e-9)
    Xo = np.stack([X[:, obs_index[o]] for o in obs_names], 1)
    Xn = (Xo - Xo.mean(0)) / (Xo.std(0) + 1e-9)
    LO = np.abs(Sn.T @ Xn / len(Sn))                     # [k, m]
    LL = np.abs(np.corrcoef(Sn.T)) if k > 1 else np.ones((k, k))
    np.fill_diagonal(LL, 0.0)
    D[m:, :m] = LO
    D[:m, m:] = LO.T
    D[m:, m:] = LL
    return list(obs_names) + lats, D
