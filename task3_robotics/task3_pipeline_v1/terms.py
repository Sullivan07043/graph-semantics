"""Term-factory (THEORY Definition 2): structure pattern -> moment condition tables.

Each function reads a pattern off the given causal graph (+ data) and emits a static table
the solver turns into one objective term. The five hand-written v5 terms become instances:

  R1 generation      -> gen_operator.edge_table (Jacobian-locked operator routing)
  R2 zero end + R3   -> ci_table (THE unified conditional-independence rule, P4): every pair
                        d-separated GIVEN its common-ancestor set S gets
                        cos(res_S e_i, res_S e_j) -> rho_hat(i,j|S); S = empty set IS the v5
                        marginal decorrelation special case. One code path, flat and
                        hierarchical graphs alike.
  R2 upper tail      -> dep_floor_table (trek-connected pairs keep |cos| >= kappa * dep)
  R3 at S = pa       -> residual Gram anchor (unchanged, assembled in core.build_ctx)
  R4 anchors/norm    -> labeled embeddings fixed by parameterization + unit-norm term

ci_table notes:
- S(i,j) = ALL common ancestors anc(i) & anc(j). Every trek between a non-ancestral pair tops
  at a common ancestor, so conditioning on the full set blocks every trek; conditioning on
  ancestors can never open a collider path between non-ancestrally-related nodes (a common
  ancestor that descends from such a collider would create a cycle).
- Targets rho_hat = data partial correlation of the pair's signals given the S signals
  (latent signal = its estimated score, observed signal = its data column), with the
  shrink rule: values below the ~2-sigma noise floor 2/sqrt(n) become the graph's claimed 0.
  This is psi(D_{ij|S}) with psi = identity on the shrunk value (psi calibration is an open
  THEORY item; P0's model-implied correlations are the planned calibrator).
- Latents without an estimated score (no observed descendants) are skipped as endpoints and
  as conditioning signals; the graph-side pattern is unchanged.
"""
import os

import numpy as np
import torch


# ------------------------------------------------------------------ unified CI rule (P4)
def ci_table(g, X, obs_index, score, nl=None, mode=None):
    """-> list of groups (S_names tuple, pairs [(a, b)], targets float32 [n_pairs]).
    Support: every unordered non-ancestral pair; group key = its common-ancestor set S.
    nl: nldep.matrices dict — marginal targets become sign(Pearson) * dcor with the Pearson
    noise-floor keep/zero decision (TRUNK-4a); conditional-group targets stay Pearson-based
    (diagnostics only). mode (default env CI_MODE=marginal_shrink, ruling 2026-07-28):
    'marginal_shrink' = marginal groups with shrink targets (THE objective default);
    'marginal' = marginal groups, hard-zero targets (v5 semantics); 'full' = everything
    (diagnostics/attribution only — measured harmful in the objective)."""
    anc = {n: g.ancestors(n) for n in g.nodes}
    sig = {}
    for n in g.nodes:
        if g.is_latent(n):
            if n in score:
                sig[n] = np.asarray(score[n], float)
        else:
            sig[n] = X[:, obs_index[n]].astype(float)
    tau = 2.0 / max(np.sqrt(X.shape[0]), 1.0)
    groups = {}
    for i, a in enumerate(g.nodes):
        for b in g.nodes[i + 1:]:
            if a in anc[b] or b in anc[a]:
                continue
            groups.setdefault(tuple(sorted(anc[a] & anc[b])), []).append((a, b))
    out = []
    for S in sorted(groups):
        Ssig = [sig[s] for s in S if s in sig]
        A = None
        if Ssig:
            A = np.stack(Ssig, 1)
            A = A - A.mean(0)
        cache = {}

        def resid(n):
            if n not in cache:
                y = sig[n] - sig[n].mean()
                if A is not None:
                    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
                    y = y - A @ beta
                cache[n] = y
            return cache[n]

        keep, tg = [], []
        for a, b in groups[S]:
            if a not in sig or b not in sig:
                continue
            if not S and nl is not None and a in nl["idx"] and b in nl["idx"]:
                ia_, ib_ = nl["idx"][a], nl["idx"][b]
                p_ = nl["marg_pear"][ia_, ib_]
                rho = float(np.sign(p_) * nl["marg_dcor"][ia_, ib_]) if abs(p_) >= tau else 0.0
                keep.append((a, b))
                tg.append(rho)
                continue
            ra, rb = resid(a), resid(b)
            den = np.linalg.norm(ra) * np.linalg.norm(rb)
            rho = float(ra @ rb / den) if den > 1e-9 else 0.0
            keep.append((a, b))
            tg.append(rho if abs(rho) >= tau else 0.0)
        if keep:
            out.append((S, keep, np.array(tg, np.float32)))
    mode = mode or os.environ.get("CI_MODE", "marginal_shrink")
    if mode == "marginal":
        out = [(S, p, t * 0) for S, p, t in out if not S]
    elif mode == "marginal_shrink":
        out = [(S, p, t) for S, p, t in out if not S]
    return out


def ci_tensors(ci, node_idx, device="cpu"):
    """Static tensors per CI group for the solver.
    -> list of (S_names, ia LongTensor, ib LongTensor, targets FloatTensor)."""
    out = []
    for S, pairs, tg in ci:
        ia = torch.tensor([node_idx[a] for a, b in pairs], dtype=torch.long, device=device)
        ib = torch.tensor([node_idx[b] for a, b in pairs], dtype=torch.long, device=device)
        out.append((S, ia, ib, torch.tensor(tg, dtype=torch.float32, device=device)))
    return out


# ------------------------------------------------------------------ R2 upper tail
def dep_floor_table(g, node_idx, bridge, device="cpu"):
    """Trek-connected pairs in the top (1-q) dependence quantile keep |cos| >= kappa * dep.
    bridge: dict(obs=names, dep_marg=[m, m], lam_upper, kappa, q) — observed-only or the
    latcon-augmented observed+latent version. -> (ia, ib, floor, lam_upper) or None."""
    if bridge is None or bridge.get("lam_upper", 0) <= 0:
        return None
    bi = {n: k for k, n in enumerate(bridge["obs"])}
    Dm = np.asarray(bridge["dep_marg"])
    tp = [(a, b) for a, b in g.trek_pairs() if a in bi and b in bi]
    if not tp:
        return None
    vals = np.array([Dm[bi[a], bi[b]] for a, b in tp])
    thr = np.quantile(vals, bridge.get("q", 0.7))
    keep = [(a, b, v) for (a, b), v in zip(tp, vals) if v >= thr]
    if not keep:
        return None
    ia = torch.tensor([node_idx[a] for a, b, v in keep], dtype=torch.long, device=device)
    ib = torch.tensor([node_idx[b] for a, b, v in keep], dtype=torch.long, device=device)
    floor = torch.tensor([bridge.get("kappa", 0.5) * float(v) for a, b, v in keep],
                         dtype=torch.float32, device=device)
    return ia, ib, floor, float(bridge["lam_upper"])


# ------------------------------------------------------------------ shared eval helper
def ci_cos(M, group, Gram=None, eps=1e-9):
    """Residualized cosine per pair, from NODE-level quantities only (algebraic identity:
    <res_S a, res_S b> = <a,b> - <a,Q><Q,b> for orthonormal Q spanning the S embeddings).
    Per-pair tensors are scalars — memory O(N^2 + n_pairs), never O(n_pairs x d). This is
    load-bearing for the training path (6.7k-pair graphs x 60 unrolled steps with retained
    graphs OOMed the 40GB GPU under the direct per-pair gather; same objective value).
    Gram: optional precomputed M @ M.T (share it across groups within one solver step)."""
    S_idx, ia, ib = group
    if Gram is None:
        Gram = M @ M.T
    if S_idx is not None and len(S_idx):
        Q, _ = torch.linalg.qr(M[S_idx].T)                     # [d, |S|] orthonormal span
        MQ = M @ Q                                             # [N, |S|]
        num = Gram[ia, ib] - (MQ[ia] * MQ[ib]).sum(1)
        sq = (MQ * MQ).sum(1)
        da = Gram[ia, ia] - sq[ia]
        db = Gram[ib, ib] - sq[ib]
    else:
        num = Gram[ia, ib]
        da, db = Gram[ia, ia], Gram[ib, ib]
    # clamp BEFORE sqrt: sqrt'(0) is inf, and near-zero norms are REAL here (ALS gives
    # childless roots ~0 vectors) — clamp-after-sqrt still produces 0 * inf = nan in backward
    da = torch.clamp(da, min=1e-12)
    db = torch.clamp(db, min=1e-12)
    return num / (torch.sqrt(da) * torch.sqrt(db) + eps)
