"""THE CORE (meeting 2026-07-02): optimize semantic embeddings under the causal constraints derived from a
GIVEN graph. The optimization variables are the EMBEDDINGS themselves — latent u_j and unlabeled-observed
s_i — plus, optionally, embedding-space edge magnitudes and per-node residual vectors. No encoder is
trained (labeled embeddings a_i come from a frozen encoder and are fixed), so there is no semantic-drift
failure mode.

v2 (2026-07-14):
  - Deterministic ALS core: the generation terms are QUADRATIC in the embeddings, so with fixed edge
    weights the free embeddings solve a small sparse least-squares system in closed form (per dimension).
    With free_w=True we alternate: closed-form embedding solve <-> per-node sign-constrained least squares
    for the edge magnitudes (support and sign stay locked to the given graph + data; only magnitudes move,
    because data-space correlations need not equal embedding-space combination coefficients).
  - Adam refinement adds the non-quadratic terms (independence decorrelation, unit-norm, residual anchor)
    starting from the ALS solution; fixed seed keeps it reproducible.
  - Residual channel (residual mu > 0): e_n = gen(n) + r_n with small ||r_n||. Residual DIRECTIONS are
    anchored to the data via partial correlations: cos(r_i, r_k) is pulled toward the correlation of the
    observed residuals after regressing out the parents (identity signal beyond the factor).
"""
import numpy as np


# --------------------------------------------------------------------------- data-side helper
def partial_residual_corr(g, X, obs_index, score):
    """Correlation of data residuals after regressing each observed column on its parents' latent scores.
    -> (obs_names, P [n_obs, n_obs]). `score` is the latent-score dict from graph.estimate_weights."""
    obs = g.observed
    R = np.zeros((X.shape[0], len(obs)))
    for k, o in enumerate(obs):
        y = X[:, obs_index[o]]
        regs = [score[p] for p in g.parents(o) if g.is_latent(p) and p in score]
        regs += [X[:, obs_index[p]] for p in g.parents(o) if not g.is_latent(p)]
        if regs:
            A = np.stack(regs, 1)
            beta, *_ = np.linalg.lstsq(A, y, rcond=None)
            y = y - A @ beta
        R[:, k] = y
    R = R - R.mean(0)
    R = R / (R.std(0) + 1e-9)
    P = np.corrcoef(R.T)
    np.fill_diagonal(P, 0.0)
    return obs, P


def shrink_corr(P, n_samples):
    """Graph-vs-data blend for conditional-independence anchors: the graph's local Markov claim is
    rho = 0; keep a data partial correlation only when it clears the ~2-sigma noise floor 2/sqrt(n),
    else trust the graph's zero."""
    tau = 2.0 / max(np.sqrt(n_samples), 1.0)
    Q = P.copy()
    Q[np.abs(Q) < tau] = 0.0
    return Q


# --------------------------------------------------------------------------- ALS pieces
def _solve_embeddings(g, W, A, free, d):
    """Closed-form least squares of the generation objective over the free embeddings.
    Rows: one equation per generated node n:  t_n - sum_p W[p,n] t_p = 0 (labeled t are constants)."""
    fidx = {n: i for i, n in enumerate(free)}
    rows, rhs = [], []
    for n in g.nodes:
        ps = g.parents(n)
        if not ps:
            continue
        row = np.zeros(len(free))
        b = np.zeros(d)
        if n in fidx:
            row[fidx[n]] += 1.0
        else:
            b -= A[n]
        for p in ps:
            w = float(W.get((p, n), 0.0))
            if p in fidx:
                row[fidx[p]] -= w
            else:
                b += w * A[p]
        rows.append(row)
        rhs.append(b)
    M = np.stack(rows)                                   # [n_eq, F]
    B = np.stack(rhs)                                    # [n_eq, d]
    # tiny ridge keeps under-determined nodes (e.g. childless roots) finite
    U = np.linalg.solve(M.T @ M + 1e-6 * np.eye(len(free)), M.T @ B)
    return {n: U[fidx[n]] for n in free}


def _solve_weights(g, W_sign, emb):
    """Per-node sign-constrained least squares for embedding-space edge magnitudes (support+sign locked).
    Column p flipped by its sign -> nonnegativity -> scipy nnls; returns a new weight dict."""
    from scipy.optimize import nnls
    W_new = {}
    for n in g.nodes:
        ps = g.parents(n)
        if not ps:
            continue
        signs = [1.0 if float(W_sign.get((p, n), 0.0)) >= 0 else -1.0 for p in ps]
        A = np.stack([s * emb[p] for s, p in zip(signs, ps)], 1)       # [d, P]
        coef, _ = nnls(A, emb[n])
        for p, s, c in zip(ps, signs, coef):
            W_new[(p, n)] = float(s * c)
    return W_new


def optimize_embeddings(g, W, labeled_emb, d, steps=400, lr=2e-2, lam_zero=0.3, lam_norm=0.1,
                        seed=0, device="cpu", free_w=False, als_rounds=5,
                        residual=0.0, lam_res=0.0, partial_corr=None,
                        lam_dep=0.0, dep_corr=None, dep_kappa=0.5, lam_coll=0.0,
                        neg_op=None, bridge=None, gen_op=None, verbose=False):
    """g: graph.Graph; W: dict edge->signed weight (given support, data-estimated);
    labeled_emb: dict observed_name -> np.array[d] (frozen, VISIBLE labels only).
    free_w: also optimize embedding-space edge magnitudes (sign/support locked to the given graph).
    residual (mu>0): per-node residual channel; lam_res + partial_corr=(obs_names, P) anchor residual
    directions to the data's partial correlations (pass P through shrink_corr for the graph-zero blend).
    lam_dep + dep_corr=(obs_names, R): faithfulness floor — trek-connected observed pairs keep
    |cos(e_i,e_k)| >= dep_kappa*|rho_ik| (hinge). lam_coll: explaining-away at v-structures — after
    projecting the two parents orthogonal to their common child, penalize any remaining POSITIVE cos.
    neg_op: frozen semantic-negation module (negop.NegOp); when set, a NEGATIVE edge's generation
    contribution becomes |w| * neg_op(e_p) instead of w * e_p in the Adam stage (a reverse item is
    the semantic negation of its factor, not the vector negation; the linear ALS init is a declared
    approximation corrected here). Gradients flow through the frozen operator to the parent.
    bridge: UNIFIED BRIDGE CONSTRAINT (pipeline_v3; the explicit semantic-dependence monotonicity
    axiom). dict(obs=list, dep_marg=[m,m] magnitudes, lam_upper=float, kappa=float, q=float):
    trek-connected OBSERVED pairs whose marginal dependence is in the top (1-q) quantile get a
    lower bound hinge(kappa*dep - |cos|)^2 — the UPPER tail; the lower tail stays lam_zero on
    independent pairs; the conditional tail rides through partial_corr (whose magnitudes the caller
    may replace with dcor/MI, sign from Pearson).
    Returns dict node_name -> np.array[d] for ALL nodes (labeled ones pass through unchanged)."""
    import torch
    torch.manual_seed(seed)
    labeled = set(labeled_emb)
    free = [n for n in g.nodes if n not in labeled]                  # latents + unlabeled observed
    if not free:
        return dict(labeled_emb)
    A = {n: np.asarray(v, np.float64) for n, v in labeled_emb.items()}

    # ---- stage 1: deterministic ALS (closed-form embeddings <-> signed-LS magnitudes)
    Wcur = dict(W)
    E0 = _solve_embeddings(g, Wcur, A, free, d)
    if free_w:
        emb_all = {**A, **E0}
        for _ in range(als_rounds):
            Wcur = _solve_weights(g, W, emb_all)
            E0 = _solve_embeddings(g, Wcur, A, free, d)
            emb_all = {**A, **E0}

    # ---- stage 2: Adam refinement with the full (non-quadratic) objective
    E = {n: torch.nn.Parameter(torch.tensor(E0[n], dtype=torch.float32, device=device)) for n in free}
    At = {n: torch.tensor(v, dtype=torch.float32, device=device) for n, v in A.items()}
    params = list(E.values())

    gen_nodes = [n for n in g.nodes if g.parents(n)]
    if free_w:
        sgn = {e: (1.0 if float(W.get(e, 0.0)) >= 0 else -1.0) for e in Wcur}
        theta = {e: torch.nn.Parameter(torch.tensor(
            float(np.log(np.expm1(max(abs(Wcur[e]), 1e-4)))), dtype=torch.float32)) for e in Wcur}
        params += list(theta.values())

        def wt(e):
            return sgn[e] * torch.nn.functional.softplus(theta[e])
    else:
        wt_const = {e: torch.tensor(float(v), dtype=torch.float32) for e, v in Wcur.items()}

        def wt(e):
            return wt_const[e]

    use_res = residual > 0
    pc_nodes = []
    if use_res:
        gnr = np.random.default_rng(seed)
        Rv = {n: torch.nn.Parameter(torch.tensor(gnr.normal(0, 1e-3, d), dtype=torch.float32,
                                                 device=device)) for n in gen_nodes}
        params += list(Rv.values())
        if lam_res > 0 and partial_corr is not None:
            pc_names, Pmat = partial_corr
            pc_nodes = [n for n in pc_names if n in Rv]
            pidx = [list(pc_names).index(n) for n in pc_nodes]
            Pt = torch.tensor(Pmat[np.ix_(pidx, pidx)], dtype=torch.float32, device=device)
            offdiag = ~torch.eye(len(pc_nodes), dtype=torch.bool, device=device)

    def emb(n):
        return At[n] if n in At else E[n]

    node_idx = {n: i for i, n in enumerate(g.nodes)}
    zp_pairs = g.independent_pairs()
    ia = torch.tensor([node_idx[a] for a, b in zp_pairs], dtype=torch.long, device=device)
    ib = torch.tensor([node_idx[b] for a, b in zp_pairs], dtype=torch.long, device=device)
    # faithfulness floor: trek-connected OBSERVED pairs with a data-corr-scaled minimum |cos|
    dep_terms = None
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
            dep_terms = (da, db, floor)
    colls = [(node_idx[p1], node_idx[p2], node_idx[c]) for p1, p2, c in g.v_structures()] \
        if lam_coll > 0 else []
    neg_parents = []
    if gen_op is not None:                                # g_phi supersedes the neg_op special case
        gen_op = gen_op.to(device)
        neg_op = None
        ge = [(p, n, float(W.get((p, n), 0.0))) for n in gen_nodes for p in g.parents(n)]
        ge_par = [p for p, n, w in ge]
        lat = set(g.latents)
        ge_cond = torch.tensor([[1.0 if w >= 0 else -1.0, abs(w),
                                 1.0 if (p in lat and n in lat) else 0.0] for p, n, w in ge],
                               dtype=torch.float32, device=device)
        ge_absw = torch.tensor([abs(w) for p, n, w in ge], dtype=torch.float32, device=device)
        ge_child = {}
        for r, (p, n, w) in enumerate(ge):
            ge_child.setdefault(n, []).append(r)
    if neg_op is not None:
        neg_op = neg_op.to(device)
        neg_parents = sorted({p for (p, n), w in W.items() if w < 0})
    # unified bridge: UPPER tail on strongly-dependent trek-connected observed pairs
    br_terms = None
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
                br_terms = (ba, bb, bfloor, float(bridge["lam_upper"]))
    opt = torch.optim.Adam(params, lr=lr)
    for step in range(steps):
        opt.zero_grad()
        loss = 0.0
        # one batched f_neg forward per step for every parent with a negative out-edge (the naive
        # per-edge call was 100x slower: n_edges x steps MLP calls, recomputing shared parents)
        neg_cache = {}
        if neg_parents:
            NP = neg_op(torch.stack([emb(p) for p in neg_parents]))
            neg_cache = {p: NP[i] for i, p in enumerate(neg_parents)}
        gout = None
        if gen_op is not None:                            # one batched g_phi forward per step
            Xp = torch.stack([emb(p) for p in ge_par])
            gout = ge_absw[:, None] * gen_op(Xp, ge_cond)
        for n in gen_nodes:
            if gout is not None:
                tot = gout[ge_child[n]].sum(0)
            else:
                tot = None
                for p in g.parents(n):
                    w_e = wt((p, n))
                    if p in neg_cache and float(W.get((p, n), 0.0)) < 0:
                        t = torch.abs(w_e) * neg_cache[p]
                    else:
                        t = w_e * emb(p)
                    tot = t if tot is None else tot + t
            if use_res:
                tot = tot + Rv[n]
            tgt = At[n] if n in At else E[n]
            loss = loss + ((tgt - tot) ** 2).sum()
        if use_res:
            Rn = torch.stack([Rv[n] for n in gen_nodes])
            loss = loss + residual * (Rn ** 2).sum(1).mean()
            if len(pc_nodes) > 1:
                Rm = torch.stack([Rv[n] for n in pc_nodes])
                Rm = torch.nn.functional.normalize(Rm, dim=1)
                loss = loss + lam_res * (((Rm @ Rm.T) - Pt)[offdiag] ** 2).mean()
        need_M = (len(zp_pairs) and lam_zero > 0) or dep_terms is not None or colls \
            or br_terms is not None
        if need_M:
            M = torch.stack([emb(n) for n in g.nodes])
            Mn = torch.nn.functional.normalize(M, dim=1)
        if len(zp_pairs) and lam_zero > 0:                           # vectorized independence decorrelation
            loss = loss + lam_zero * (((Mn[ia] * Mn[ib]).sum(1)) ** 2).mean()
        if dep_terms is not None:                                    # faithfulness dependence floor
            da, db, floor = dep_terms
            cs = (Mn[da] * Mn[db]).sum(1).abs()
            loss = loss + lam_dep * (torch.relu(floor - cs) ** 2).mean()
        if br_terms is not None:                                     # bridge UPPER tail
            ba, bb, bfloor, lam_up = br_terms
            cs = (Mn[ba] * Mn[bb]).sum(1).abs()
            loss = loss + lam_up * (torch.relu(bfloor - cs) ** 2).mean()
        if colls:                                                    # explaining away at v-structures
            cl = 0.0
            for i1, i2, ic in colls:
                u1 = Mn[i1] - (Mn[i1] @ Mn[ic]) * Mn[ic]
                u2 = Mn[i2] - (Mn[i2] @ Mn[ic]) * Mn[ic]
                cnum = (u1 @ u2) / (u1.norm() * u2.norm() + 1e-9)
                cl = cl + torch.relu(cnum) ** 2
            loss = loss + lam_coll * cl / len(colls)
        if lam_norm > 0:
            nr = torch.stack([E[n].norm() for n in free])
            loss = loss + lam_norm * ((nr - 1.0) ** 2).mean()
        loss.backward()
        opt.step()
        if verbose and (step % 100 == 0 or step == steps - 1):
            print(f"  [optimize step {step}/{steps} loss={float(loss):.4f}]", flush=True)
    out = {n: v.detach().cpu().numpy().astype(np.float64) for n, v in E.items()}
    out.update({n: np.asarray(v, np.float64) for n, v in labeled_emb.items()})
    return out
