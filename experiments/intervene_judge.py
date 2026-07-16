"""SWAP INTERVENTION - JUDGE VERSION. Same swaps as intervene.py, but success is judged by language,
not geometry: decode the swapped prediction into dictionary words, then ask the judge whether the
words refer to the OTHER latent's construct (latent_gt text). Baseline sanity: pre-swap decode should
be judged as referring to the original parent's construct.

Lines: A = graph-opt (frozen config incl f_neg), B = GNN hidden-swap.
Cases capped at MAX_CASES per dataset (deterministic subsample) to bound judge API cost.
Output: outputs/intervene_judge.json
"""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import pool, encode, metrics, optimize
import judge as judge_mod
from run_task1 import ALL_LOADERS
from experiments.intervene import family_refs, latent_pairs, cos, FOLDS, SWAP_LAYER, GNN_CKPT

MAX_CASES = int(os.environ.get("MAX_CASES", 40))
DATASETS = os.environ.get("DATASETS", "tlvd,himi,bigfive,hs,gcbs,sixteenpf,hsq,sd3,hexaco,riasec,kims").split(",")
_NEG = None


def ts():
    return time.strftime("%H:%M:%S")


def run_dataset(name, C, cwords):
    ds = ALL_LOADERS[name]()
    g, X, labels, lgt = ds["graph"], ds["X"], ds["labels"], ds["latent_gt"]
    obs = g.observed
    obs_i = {o: k for k, o in enumerate(obs)}
    T = encode.embed([labels[o] for o in obs])
    alpha = metrics.pick_alpha(T, C)
    W, score = g.estimate_weights(X, obs_i)
    pc = optimize.partial_residual_corr(g, X, obs_i, score)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(obs))
    folds = [perm[i::FOLDS] for i in range(FOLDS)]
    prng = np.random.default_rng(1)
    pairs = latent_pairs(g, prng)
    parent_of = {o: next((p for p in g.parents(o) if g.is_latent(p)), None) for o in obs}

    import torch, gnn as gnn_mod
    ck = torch.load(GNN_CKPT, map_location=gnn_mod.DEVICE)
    m = gnn_mod.CompletionGNN(ck["d"], ck["hid"], ck["layers"]).to(gnn_mod.DEVICE)
    m.load_state_dict(ck["state"], strict=False); m.eval()
    gt_ = gnn_mod.graph_tensors(ds)
    nidx = {n_: i for i, n_ in enumerate(g.nodes)}

    # collect cases: (vecA_swap, vecA_base, vecB_swap, vecB_base, gt_other, gt_parent)
    cases = []
    for fno, fold in enumerate(folds):
        masked = sorted(int(i) for i in fold)
        mset = set(masked)
        visible_set = set(range(len(obs))) - mset
        vis_emb = {obs[i]: T[i] for i in visible_set}
        emb = optimize.optimize_embeddings(g, W, vis_emb, d=T.shape[1], seed=fno,
                                           residual=1.0, lam_res=1.0, partial_corr=pc,
                                           neg_op=_NEG)
        ne, lb, ce = gnn_mod.masked_inputs(gt_, masked)
        with torch.no_grad():
            out0 = m(gt_, ne, lb, ce).cpu().numpy().astype(np.float64)
        for (p, q) in pairs:
            if p not in lgt or q not in lgt:
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
                w = float(W.get((pa, c), 0.0))
                predA = emb[c] - w * emb[pa] + w * emb[other]
                cases.append((predA, emb[c], outS[nidx[c]], out0[nidx[c]], lgt[other], lgt[pa]))
    if not cases:
        print(f"[{ts()}] {name:10s} n/a (no cases)", flush=True)
        return {"n": 0}
    if len(cases) > MAX_CASES:
        sub = np.random.default_rng(2).choice(len(cases), MAX_CASES, replace=False)
        cases = [cases[i] for i in sub]

    res = {"n": len(cases)}
    for line, iv, ib in [("graph_opt", 0, 1), ("gnn", 2, 3)]:
        Vsw = np.stack([c[iv] for c in cases])
        Vb = np.stack([c[ib] for c in cases])
        wsw = metrics.decode_words(Vsw, C, cwords, alpha)
        wb = metrics.decode_words(Vb, C, cwords, alpha)
        succ = judge_mod.judge_batch([(wsw[i], cases[i][4]) for i in range(len(cases))], "latent")
        base = judge_mod.judge_batch([(wb[i], cases[i][5]) for i in range(len(cases))], "latent")
        for key, v in [("swap_judge", succ), ("base_judge", base)]:
            ok = [x for x in (v or []) if x is not None]
            res.setdefault(line, {})[key] = float(np.mean(ok)) if ok else None
    def _f(v):
        return f"{v:.3f}" if v is not None else "n/a"
    print(f"[{ts()}] {name:10s} graph-opt swap={_f(res['graph_opt']['swap_judge'])} "
          f"(base {_f(res['graph_opt']['base_judge'])}, n={res['n']}) | "
          f"gnn swap={_f(res['gnn']['swap_judge'])} (base {_f(res['gnn']['base_judge'])})", flush=True)
    return res


def main():
    global _NEG
    import negop
    _NEG = negop.load()
    import experiments.intervene as iv
    iv._NEG = _NEG
    C, cwords = encode.load_dictionary()
    path = os.path.join(HERE, "outputs", "intervene_judge.json")
    out = {}
    if os.path.exists(path) and os.environ.get("RESUME"):
        out = json.load(open(path))
    for name in DATASETS:
        name = name.strip()
        if name in out:
            continue
        out[name] = run_dataset(name, C, cwords)
        json.dump(out, open(path, "w"), indent=1)
    print("[saved outputs/intervene_judge.json]", flush=True)


if __name__ == "__main__":
    main()
