"""Task 2 — Task 1 + translate the LATENT variables (same given-graph setting).
The same graph-constrained optimization already produces latent embeddings u_j; this runner decodes them and
judges against the dataset's latent ground-truth descriptions (see testbeds.py; on TLVD the GT texts are the
four construct descriptions shipped in TLVD's own description file). Latent baseline: LLM-naming (TLVD-style
single call over the latent's children labels), judged by the same judge. Records -> RECORDS_OUT."""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import testbeds, pool, encode, metrics, optimize
import judge as judge_mod
from run_task1 import ALL_LOADERS, select_datasets

FOLDS = int(os.environ.get("FOLDS", 5))
STEPS = int(os.environ.get("STEPS", 400))
LAM_ZERO = float(os.environ.get("LAM_ZERO", 0.3))
LAM_NORM = float(os.environ.get("LAM_NORM", 0.1))
FREE_W = os.environ.get("FREE_W", "0") == "1"
RESIDUAL = float(os.environ.get("RESIDUAL", 0.0))
LAM_RES = float(os.environ.get("LAM_RES", 0.0))
SHRINK = os.environ.get("SHRINK", "0") == "1"
LAM_DEP = float(os.environ.get("LAM_DEP", 0.0))
LAM_COLL = float(os.environ.get("LAM_COLL", 0.0))
NEGOP = os.environ.get("NEGOP", "0") == "1"          # semantic negation operator on negative edges
GNN_ARM = os.environ.get("GNN_ARM", "0") == "1"          # decode the GNN's latent-node outputs too
GNN_JAC = os.environ.get("GNN_JAC", "0") == "1"          # Jacobian read-off latent translation
GNN_GEN = os.environ.get("GNN_GEN", "0") == "1"          # generation-head read-off (needs gen-trained ckpt)


NEG_OP = None
if NEGOP:
    import negop
    NEG_OP = negop.load()


def ts():
    return time.strftime("%H:%M:%S")


def llm_name(child_labels, model="gpt-4o-mini"):
    import urllib.request
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return None
    prompt = ("The following observed measures all load on one hidden latent factor:\n- "
              + "\n- ".join(child_labels) +
              "\n\nName the single construct this latent factor represents, in 1-4 words. "
              "Answer with only the name.")
    body = json.dumps({"model": model, "temperature": 0,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request("https://api.openai.com/v1/chat/completions", data=body,
                                 headers={"Authorization": f"Bearer {key}",
                                          "Content-Type": "application/json"})
    try:
        r = json.loads(urllib.request.urlopen(req, timeout=60).read())
        return r["choices"][0]["message"]["content"].strip()
    except Exception:
        return None


def run_dataset(ds, C, cwords, records):
    g, X, labels, gt = ds["graph"], ds["X"], ds["labels"], ds["latent_gt"]
    obs = g.observed
    oi = {o: k for k, o in enumerate(obs)}
    T = encode.embed([labels[o] for o in obs])
    alpha = metrics.pick_alpha(T, C)
    W, score = g.estimate_weights(X, oi)
    pc = optimize.partial_residual_corr(g, X, oi, score) if RESIDUAL > 0 else None
    if pc is not None and SHRINK:
        pc = (pc[0], optimize.shrink_corr(pc[1], X.shape[0]))
    Craw = np.corrcoef(X.T); np.fill_diagonal(Craw, 0.0)
    dep = ([o for o in obs], Craw) if LAM_DEP > 0 else None
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(obs))
    folds = [perm[i::FOLDS] for i in range(FOLDS)]
    lat_names = [L for L in g.latents if L in gt]
    core_accs, qual, llm_accs = [], [], []
    gnn_ctx, gnn_accs, jac_accs, gen_accs = None, [], [], []
    if GNN_ARM or GNN_JAC or GNN_GEN:
        import torch, gnn as gnn_mod
        ck = torch.load(gnn_mod.CKPT, map_location=gnn_mod.DEVICE)
        gmodel = gnn_mod.CompletionGNN(ck["d"], ck["hid"], ck["layers"]).to(gnn_mod.DEVICE)
        gmodel.load_state_dict(ck["state"], strict=False); gmodel.eval()
        gt_ = gnn_mod.graph_tensors(ds)
        nidx = {n: i for i, n in enumerate(g.nodes)}
        gnn_ctx = (gnn_mod, gmodel, gt_, [nidx[L] for L in lat_names])
    print(f"[{ts()}] {ds['name']}: Task 2 over {len(lat_names)} latents x {FOLDS} folds", flush=True)

    for fno, fold in enumerate(folds):
        masked = set(int(i) for i in fold)
        vis_emb = {obs[i]: T[i] for i in range(len(obs)) if i not in masked}
        emb = optimize.optimize_embeddings(g, W, vis_emb, d=T.shape[1], steps=STEPS,
                                           lam_zero=LAM_ZERO, lam_norm=LAM_NORM, seed=fno,
                                           free_w=FREE_W, residual=RESIDUAL, lam_res=LAM_RES,
                                           partial_corr=pc, lam_dep=LAM_DEP, dep_corr=dep,
                                           lam_coll=LAM_COLL, neg_op=NEG_OP)
        U = np.stack([emb[L] for L in lat_names])
        words = metrics.decode_words(U, C, cwords, alpha)
        jacc, verd = metrics.judge_latents(words, [gt[L] for L in lat_names])
        if jacc is not None:
            core_accs.append(jacc)
        for L, w_, ok in zip(lat_names, words, verd or [None] * len(lat_names)):
            records.append({"task": 2, "dataset": ds["name"], "fold": fno, "arm": "core", "latent": L,
                            "gt": gt[L], "decoded_words": w_,
                            "judge": (bool(ok) if ok is not None else None)})
        if fno == 0:
            qual = list(zip(lat_names, words, verd or []))
        if gnn_ctx is not None:
            import torch
            gnn_mod, gmodel, gt_, lat_idx = gnn_ctx
            if GNN_ARM:
                with torch.no_grad():
                    o = gnn_mod.masked_forward(gmodel, gt_, sorted(masked))
                Ug = o[torch.tensor(lat_idx, device=gnn_mod.DEVICE)].cpu().numpy().astype(np.float64)
                gwords = metrics.decode_words(Ug, C, cwords, alpha)
                gacc, gverd = metrics.judge_latents(gwords, [gt[L] for L in lat_names])
                if gacc is not None:
                    gnn_accs.append(gacc)
                for L, w_, ok in zip(lat_names, gwords, gverd or [None] * len(lat_names)):
                    records.append({"task": 2, "dataset": ds["name"], "fold": fno, "arm": "gnn",
                                    "latent": L, "gt": gt[L], "decoded_words": w_,
                                    "judge": (bool(ok) if ok is not None else None)})
            if GNN_JAC:
                Uj = gnn_mod.jacobian_readoff(gmodel, gt_, sorted(masked), lat_names)
                jwords = metrics.decode_words(Uj, C, cwords, alpha)
                jacc_, jverd = metrics.judge_latents(jwords, [gt[L] for L in lat_names])
                if jacc_ is not None:
                    jac_accs.append(jacc_)
                for L, w_, ok in zip(lat_names, jwords, jverd or [None] * len(lat_names)):
                    records.append({"task": 2, "dataset": ds["name"], "fold": fno,
                                    "arm": "gnn_jacread", "latent": L, "gt": gt[L],
                                    "decoded_words": w_,
                                    "judge": (bool(ok) if ok is not None else None)})
            if GNN_GEN:
                Ug2 = gnn_mod.genhead_readoff(gmodel, gt_, sorted(masked), lat_names)
                w2 = metrics.decode_words(Ug2, C, cwords, alpha)
                a2, v2 = metrics.judge_latents(w2, [gt[L] for L in lat_names])
                if a2 is not None:
                    gen_accs.append(a2)
                for L, w_, ok in zip(lat_names, w2, v2 or [None] * len(lat_names)):
                    records.append({"task": 2, "dataset": ds["name"], "fold": fno,
                                    "arm": "gnn_genhead", "latent": L, "gt": gt[L],
                                    "decoded_words": w_,
                                    "judge": (bool(ok) if ok is not None else None)})
        # LLM-naming baseline, ALIGNED TO THE TASK (user correction 2026-07-15): the task gives a
        # SUBSET of labels, so the baseline names from the fold's VISIBLE children only — never from
        # labels our method cannot see. Evaluated per fold, averaged like core.
        if judge_mod.available():
            items, meta = [], []
            for L in lat_names:
                ch = [labels[c] for c in g.observed_descendants(L) if oi[c] not in masked][:6]
                nm = llm_name(ch) if ch else None
                if nm is None:
                    items = []
                    break
                items.append(([nm], gt[L])); meta.append((L, nm))
            if len(items) == len(lat_names):
                import judge
                v = judge.judge_batch(items, "latent")
                if v:
                    llm_accs.append(float(np.mean(v)))
                for (L, nm), ok in zip(meta, v or []):
                    records.append({"task": 2, "dataset": ds["name"], "fold": fno, "arm": "llm_name",
                                    "latent": L, "gt": gt[L], "decoded_words": [nm],
                                    "judge": bool(ok)})
        print(f"[{ts()}]   fold {fno + 1}/{FOLDS} done", flush=True)

    base_acc = float(np.mean(llm_accs)) if llm_accs else None

    print(f"\n[{ts()}] === Task 2 results: {ds['name']} (latent judge-ACC) ===", flush=True)
    print(f"  core (graph-optimized embeddings): "
          f"{np.mean(core_accs):.3f}" if core_accs else "  core: (judge off)", flush=True)
    if gnn_accs:
        print(f"  gnn (trained completion operator): {np.mean(gnn_accs):.3f}", flush=True)
    if jac_accs:
        print(f"  gnn jacobian read-off            : {np.mean(jac_accs):.3f}", flush=True)
    if gen_accs:
        print(f"  gnn generation-head read-off     : {np.mean(gen_accs):.3f}", flush=True)
    print(f"  LLM-naming baseline              : "
          f"{base_acc:.3f}" if base_acc is not None else "  LLM-naming baseline: (skipped)", flush=True)
    for L, w_, ok in qual:
        print(f"    {L} (gt: {ds['latent_gt'][L][:50]}...) <- {', '.join(w_)}"
              f"  [{'OK' if ok else 'X'}]" if ok is not None else "", flush=True)
    return {"core": (float(np.mean(core_accs)) if core_accs else None), "llm_name": base_acc,
            "gnn": (float(np.mean(gnn_accs)) if gnn_accs else None),
            "gnn_jacread": (float(np.mean(jac_accs)) if jac_accs else None),
            "gnn_genhead": (float(np.mean(gen_accs)) if gen_accs else None)}


def main():
    which = os.environ.get("DATASET", "all")
    names = select_datasets(which)
    C, cwords = encode.load_dictionary()
    records, summary = [], {}
    for n in names:
        summary[n] = run_dataset(ALL_LOADERS[n](), C, cwords, records)
    out = os.environ.get("RECORDS_OUT", os.path.join(HERE, "outputs", "task2_records.json"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"summary": summary, "records": records}, open(out, "w"), ensure_ascii=False, indent=1)
    print(f"[saved {out} ({len(records)} items)]", flush=True)


if __name__ == "__main__":
    main()
