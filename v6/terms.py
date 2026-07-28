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
import numpy as np
import torch


# ------------------------------------------------------------------ unified CI rule (P4)
def ci_table(g, X, obs_index, score):
    """-> list of groups (S_names tuple, pairs [(a, b)], targets float32 [n_pairs]).
    Support: every unordered non-ancestral pair; group key = its common-ancestor set S."""
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
            ra, rb = resid(a), resid(b)
            den = np.linalg.norm(ra) * np.linalg.norm(rb)
            rho = float(ra @ rb / den) if den > 1e-9 else 0.0
            keep.append((a, b))
            tg.append(rho if abs(rho) >= tau else 0.0)
        if keep:
            out.append((S, keep, np.array(tg, np.float32)))
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
def ci_cos(M, group, eps=1e-9):
    """Residualized cosine for one CI group. M: [N, d] raw node embeddings (solver order);
    group: (S_idx LongTensor or None, ia, ib). Projects the endpoint embeddings off the span
    of the S embeddings (differentiable, current values), then row-normalizes."""
    S_idx, ia, ib = group
    Ea, Eb = M[ia], M[ib]
    if S_idx is not None and len(S_idx):
        Q, _ = torch.linalg.qr(M[S_idx].T)                     # [d, |S|] orthonormal span
        Ea = Ea - (Ea @ Q) @ Q.T
        Eb = Eb - (Eb @ Q) @ Q.T
    Ea = Ea / (Ea.norm(dim=1, keepdim=True) + eps)
    Eb = Eb / (Eb.norm(dim=1, keepdim=True) + eps)
    return (Ea * Eb).sum(1)
