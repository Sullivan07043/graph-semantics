"""SWAP INTERVENTION VALIDATION (J-space style): does a latent's representation causally govern its
children's completions? Swap two latents' representations; a masked child should now be predicted as
the OTHER family's meaning.

Success for masked child c of latent p, swapped with q:
    cos(pred_swapped(c), ref(q)) > cos(pred_swapped(c), ref(p))
where ref(L) = normalized mean of L's VISIBLE observed descendants' label embeddings (fold-local).
Baseline sanity: pre-swap prediction should sit closer to its own family (reported as base_ok).

Line A (graph-opt): solve embeddings under the frozen config, then replace the parent contribution
in the generation equation: pred_swap(c) = s_c - W[p,c]*u_p + W[p,c]*u_q (residual kept).
Line B (GNN): swap the two latents' hidden states after layer SWAP_LAYER and continue the forward.

Geometric only (no judge). Output: outputs/intervene.json + printed per-dataset rates.
"""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
sys.path.insert(0, HERE)
import pool, encode, metrics, optimize
from run_task1 import ALL_LOADERS

FOLDS = 5
MAX_PAIRS = 15
SWAP_LAYER = int(os.environ.get("SWAP_LAYER", 2))
GNN_CKPT = os.environ.get("GNN_CKPT", os.path.join(HERE, "outputs", "gnn_latsup.pt"))
_NEG = None
DATASETS = os.environ.get("DATASETS", "himi,bigfive,gcbs,sd3,hexaco,riasec,kims").split(",")


def ts():
    return time.strftime("%H:%M:%S")


def family_refs(g, T, obs_i, visible_set, W=None):
    """Family reference = the latent's POSITIVE pole: mean of positively-loaded visible descendants.
    (Reverse-keyed items point away from the pole; including them made the reference near-random on
    bigfive/hexaco — the sign fix registered 2026-07-15.) Falls back to all visible if no positives."""
    def load_sign(o):
        if W is None:
            return 1.0
        ps = [p for p in g.parents(o) if g.is_latent(p)]
        return W.get((ps[0], o), 0.0) if ps else 1.0

    refs = {}
    for L in g.latents:
        vis = [o for o in g.observed_descendants(L) if obs_i[o] in visible_set]
        pos = [o for o in vis if load_sign(o) >= 0]
        rows = [T[obs_i[o]] for o in (pos if pos else vis)]
        if rows:
            v = np.mean(rows, 0)
            refs[L] = v / (np.linalg.norm(v) + 1e-9)
    return refs


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def latent_pairs(g, rng):
    lats = [L for L in g.latents]
    pairs = [(a, b) for i, a in enumerate(lats) for b in lats[i + 1:]]
    if len(pairs) > MAX_PAIRS:
        pairs = [pairs[i] for i in rng.choice(len(pairs), MAX_PAIRS, replace=False)]
    return pairs


def run_dataset(name):
    ds = ALL_LOADERS[name]()
    g, X, labels = ds["graph"], ds["X"], ds["labels"]
    obs = g.observed
    obs_i = {o: k for k, o in enumerate(obs)}
    T = encode.embed([labels[o] for o in obs])
    W, score = g.estimate_weights(X, obs_i)
    pc = optimize.partial_residual_corr(g, X, obs_i, score)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(obs))
    folds = [perm[i::FOLDS] for i in range(FOLDS)]
    prng = np.random.default_rng(1)
    pairs = latent_pairs(g, prng)
    parent_of = {o: next((p for p in g.parents(o) if g.is_latent(p)), None) for o in obs}

    # ---- line B setup
    import torch, gnn as gnn_mod
    ck = torch.load(GNN_CKPT, map_location=gnn_mod.DEVICE)
    m = gnn_mod.CompletionGNN(ck["d"], ck["hid"], ck["layers"]).to(gnn_mod.DEVICE)
    m.load_state_dict(ck["state"], strict=False); m.eval()
    gt_ = gnn_mod.graph_tensors(ds)
    nidx = {n_: i for i, n_ in enumerate(g.nodes)}

    A_succ, A_base, B_succ, B_base = [], [], [], []
    for fno, fold in enumerate(folds):
        masked = sorted(int(i) for i in fold)
        mset = set(masked)
        visible_set = set(range(len(obs))) - mset
        vis_emb = {obs[i]: T[i] for i in visible_set}
        refs = family_refs(g, T, obs_i, visible_set, W)
        import negop as _negop
        emb = optimize.optimize_embeddings(g, W, vis_emb, d=T.shape[1], seed=fno,
                                           residual=1.0, lam_res=1.0, partial_corr=pc,
                                           neg_op=_NEG)
        ne, lb, ce = gnn_mod.masked_inputs(gt_, masked)
        with torch.no_grad():
            out0 = m(gt_, ne, lb, ce).cpu().numpy().astype(np.float64)
        for (p, q) in pairs:
            if p not in refs or q not in refs:
                continue
            with torch.no_grad():
                outS = m(gt_, ne, lb, ce, swap=(nidx[p], nidx[q], SWAP_LAYER)
                         ).cpu().numpy().astype(np.float64)
            for c_i in masked:
                c = obs[c_i]
                pa = parent_of[c]
                if pa not in (p, q):
                    continue
                other = q if pa == p else p
                # line A: replace the parent contribution in the generation equation
                w = float(W.get((pa, c), 0.0))
                s_c = emb[c]
                predA = s_c - w * emb[pa] + w * emb[other]
                A_base.append(cos(s_c, refs[pa]) > cos(s_c, refs[other]))
                A_succ.append(cos(predA, refs[other]) > cos(predA, refs[pa]))
                # line B: mid-network hidden swap
                b0 = out0[nidx[c]]
                bS = outS[nidx[c]]
                B_base.append(cos(b0, refs[pa]) > cos(b0, refs[other]))
                B_succ.append(cos(bS, refs[other]) > cos(bS, refs[pa]))
    res = {"graph_opt": {"swap_success": float(np.mean(A_succ)) if A_succ else None,
                         "base_ok": float(np.mean(A_base)) if A_base else None,
                         "n": len(A_succ)},
           "gnn": {"swap_success": float(np.mean(B_succ)) if B_succ else None,
                   "base_ok": float(np.mean(B_base)) if B_base else None, "n": len(B_succ)}}
    def _f(v):
        return f"{v:.3f}" if v is not None else "n/a"
    print(f"[{ts()}] {name:10s} graph-opt swap={_f(res['graph_opt']['swap_success'])} "
          f"(base {_f(res['graph_opt']['base_ok'])}, n={res['graph_opt']['n']}) | "
          f"gnn swap={_f(res['gnn']['swap_success'])} (base {_f(res['gnn']['base_ok'])})", flush=True)
    return res


def _init_neg():
    global _NEG
    import negop
    _NEG = negop.load()


def main():
    _init_neg()
    path = os.path.join(HERE, "outputs", "intervene.json")
    out = {}
    if os.path.exists(path) and os.environ.get("RESUME"):
        out = json.load(open(path))
    for name in DATASETS:
        name = name.strip()
        if name in out:
            continue
        out[name] = run_dataset(name)
        json.dump(out, open(path, "w"), indent=1)
    print("[saved outputs/intervene.json]", flush=True)


if __name__ == "__main__":
    main()
