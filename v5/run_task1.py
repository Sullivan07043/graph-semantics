"""Task 1 — complete the semantics of unlabeled OBSERVED variables, given the causal graph + a labeled subset.
Protocol: FOLDS-fold masking over observed labels (labels hidden; data kept; the graph is GIVEN and fixed).
Arms (same three metrics each): uniform / raw-corr baselines (no graph), and CORE = graph-constrained
embedding optimization (optimize.py). Full per-item records -> RECORDS_OUT."""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import testbeds, pool, encode, metrics, optimize
import judge as judge_mod

ALL_LOADERS = {**testbeds.LOADERS, **pool.LOADERS}


def select_datasets(which):
    if which == "all":
        return list(ALL_LOADERS)
    if which == "dev":
        return list(pool.DEV)
    if which == "heldout":
        return list(pool.HELDOUT)
    return [w.strip() for w in which.split(",")]

FOLDS = int(os.environ.get("FOLDS", 5))
STEPS = int(os.environ.get("STEPS", 400))
LAM_ZERO = float(os.environ.get("LAM_ZERO", 0.3))
LAM_NORM = float(os.environ.get("LAM_NORM", 0.1))
FREE_W = os.environ.get("FREE_W", "0") == "1"
RESIDUAL = float(os.environ.get("RESIDUAL", 0.0))
LAM_RES = float(os.environ.get("LAM_RES", 0.0))
SHRINK = os.environ.get("SHRINK", "0") == "1"            # graph-zero blend for CI anchors
LAM_DEP = float(os.environ.get("LAM_DEP", 0.0))          # faithfulness dependence floor
LAM_COLL = float(os.environ.get("LAM_COLL", 0.0))        # explaining-away at v-structures
NEGOP = os.environ.get("NEGOP", "0") == "1"
BRIDGE = os.environ.get("BRIDGE", "")             # "pearson" = frozen upper-tail bridge (2026-07-15)          # semantic negation operator on negative edges
GNN_ARM = os.environ.get("GNN_ARM", "0") == "1"          # line-B zero-shot arm (needs outputs/gnn.pt)


NEG_OP = None
if NEGOP:
    import negop
    NEG_OP = negop.load()
GENPHI = os.environ.get("GENPHI", "0") == "1"
GEN_OP = None
if GENPHI:
    import genphi as _genphi
    GEN_OP = _genphi.load()


def ts():
    return time.strftime("%H:%M:%S")


def run_dataset(ds, C, cwords, records):
    g, X, labels = ds["graph"], ds["X"], ds["labels"]
    obs = g.observed
    oi = {o: k for k, o in enumerate(obs)}
    T = encode.embed([labels[o] for o in obs])
    Tn = metrics.norm_rows(T)
    alpha = metrics.pick_alpha(T, C)
    W, score = g.estimate_weights(X, oi)
    pc = optimize.partial_residual_corr(g, X, oi, score) if RESIDUAL > 0 else None
    if pc is not None and SHRINK:
        pc = (pc[0], optimize.shrink_corr(pc[1], X.shape[0]))
    Craw = np.corrcoef(X.T); np.fill_diagonal(Craw, 0.0)
    br = None
    if BRIDGE:
        import dependence as _dep
        br = dict(obs=list(obs), dep_marg=_dep.load(ds["name"], "marginal", BRIDGE),
                  lam_upper=0.3, kappa=0.5, q=0.7)
    dep = (obs, Craw) if LAM_DEP > 0 else None
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(obs))
    folds = [perm[i::FOLDS] for i in range(FOLDS)]
    arm_names = ["uniform", "rawcorr", "core"] + (["gnn"] if GNN_ARM else [])
    gnn_ctx = None
    if GNN_ARM:
        import torch, gnn as gnn_mod
        ck = torch.load(gnn_mod.CKPT, map_location=gnn_mod.DEVICE)
        gmodel = gnn_mod.CompletionGNN(ck["d"], ck["hid"], ck["layers"]).to(gnn_mod.DEVICE)
        gmodel.load_state_dict(ck["state"], strict=False); gmodel.eval()
        gnn_ctx = (gnn_mod, gmodel, gnn_mod.graph_tensors(ds))
    arms = {a: {"judge": [], "match": [], "exact": []} for a in arm_names}
    print(f"[{ts()}] {ds['name']}: {X.shape[0]}x{len(obs)} | graph: {len(g.latents)} latents, "
          f"{len(g.edges)} edges, {len(g.independent_pairs())} independent pairs | alpha={alpha:.2e}", flush=True)

    for fno, fold in enumerate(folds):
        masked = sorted(int(i) for i in fold)
        visible = [i for i in range(len(obs)) if i not in set(masked)]
        vis_emb = {obs[i]: T[i] for i in visible}
        # baselines: affinity-weighted mean of visible labels
        preds = {}
        for name, A in (("uniform", np.ones_like(Craw)), ("rawcorr", np.clip(Craw, 0, None))):
            P = np.zeros((len(masked), T.shape[1]))
            for r, i in enumerate(masked):
                w = np.zeros(len(obs)); w[visible] = A[i, visible]
                if w.sum() < 1e-9:
                    w[visible] = 1.0
                P[r] = (w / w.sum()) @ T
            preds[name] = P
        # CORE: graph-constrained embedding optimization
        emb = optimize.optimize_embeddings(g, W, vis_emb, d=T.shape[1], steps=STEPS,
                                           lam_zero=LAM_ZERO, lam_norm=LAM_NORM, seed=fno,
                                           free_w=FREE_W, residual=RESIDUAL, lam_res=LAM_RES,
                                           partial_corr=pc, lam_dep=LAM_DEP, dep_corr=dep,
                                           lam_coll=LAM_COLL, neg_op=NEG_OP, bridge=br, gen_op=GEN_OP)
        preds["core"] = np.stack([emb[obs[i]] for i in masked])
        if gnn_ctx is not None:
            import torch
            gnn_mod, gmodel, gt_ = gnn_ctx
            with torch.no_grad():
                o = gnn_mod.masked_forward(gmodel, gt_, masked)
            preds["gnn"] = o[gt_["obs_pos"][masked].to(gnn_mod.DEVICE)].cpu().numpy().astype(np.float64)
        for a, P in preds.items():
            arms[a]["exact"].append(metrics.exact_acc(P, masked, Tn))
            arms[a]["match"].append(metrics.match_acc(P, masked, T))
            words = metrics.decode_words(P, C, cwords, alpha) if judge_mod.available() else None
            jacc, verd = (metrics.judge_completion(words, [labels[obs[i]] for i in masked])
                          if words else (None, None))
            if jacc is not None:
                arms[a]["judge"].append(jacc)
            for r, i in enumerate(masked):
                records.append({"task": 1, "dataset": ds["name"], "fold": fno, "arm": a,
                                "var": obs[i], "true_label": labels[obs[i]],
                                "decoded_words": (words[r] if words else None),
                                "judge": (bool(verd[r]) if verd and verd[r] is not None else None)})
        print(f"[{ts()}]   fold {fno + 1}/{FOLDS} done", flush=True)

    print(f"\n[{ts()}] === Task 1 results: {ds['name']} ===   judge   match(chance~{1/len(folds[0]):.2f})   exact",
          flush=True)
    for a in arm_names:
        j = f"{np.mean(arms[a]['judge']):.3f}" if arms[a]["judge"] else "  -  "
        print(f"  {a:10s}: {j}   {np.mean(arms[a]['match']):.3f}            {np.mean(arms[a]['exact']):.3f}",
              flush=True)
    return {a: {k: (float(np.mean(v)) if v else None) for k, v in arms[a].items()} for a in arms}


def main():
    which = os.environ.get("DATASET", "all")
    names = select_datasets(which)
    C, cwords = encode.load_dictionary()
    records, summary = [], {}
    for n in names:
        summary[n] = run_dataset(ALL_LOADERS[n](), C, cwords, records)
    out = os.environ.get("RECORDS_OUT", os.path.join(HERE, "outputs", "task1_records.json"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"summary": summary, "records": records}, open(out, "w"), ensure_ascii=False, indent=1)
    print(f"[saved {out} ({len(records)} items)]", flush=True)


if __name__ == "__main__":
    main()
