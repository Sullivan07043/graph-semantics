"""Task 2 — Task 1 + translate the LATENT variables (same given-graph setting).
The same graph-constrained optimization already produces latent embeddings u_j; this runner decodes them and
judges against the dataset's latent ground-truth descriptions (see testbeds.py; on TLVD the GT texts are the
four construct descriptions shipped in TLVD's own description file). Latent baselines: loading-centroid and
LLM-naming (single call over a fold-visible typed Markov blanket), judged by the same judge. Records -> RECORDS_OUT."""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import testbeds, pool, encode, metrics, optimize, external_baselines
import judge as judge_mod
from run_task1 import ALL_LOADERS, select_datasets

FOLDS = int(os.environ.get("FOLDS", 5))
LATCON = os.environ.get("LATCON", "1") == "1"   # latent-level constraints (main-line default 2026-07-22)
STEPS = int(os.environ.get("STEPS", 400))
LAM_ZERO = float(os.environ.get("LAM_ZERO", 0.3))
LAM_NORM = float(os.environ.get("LAM_NORM", 0.1))
FREE_W = os.environ.get("FREE_W", "0") == "1"
RESIDUAL = float(os.environ.get("RESIDUAL", 0.0))
LAM_RES = float(os.environ.get("LAM_RES", 0.0))
SHRINK = os.environ.get("SHRINK", "0") == "1"
LAM_DEP = float(os.environ.get("LAM_DEP", 0.0))
LAM_COLL = float(os.environ.get("LAM_COLL", 0.0))
NLDEP = os.environ.get("NLDEP", "1") == "1"      # TRUNK-4a: dcor targets + GBR residualization
NEGOP = os.environ.get("NEGOP", "0") == "1"
BRIDGE = os.environ.get("BRIDGE", "")             # "pearson" = frozen upper-tail bridge (2026-07-15)          # semantic negation operator on negative edges
GNN_ARM = os.environ.get("GNN_ARM", "0") == "1"          # decode the GNN's latent-node outputs too
GNN_JAC = os.environ.get("GNN_JAC", "0") == "1"          # Jacobian read-off latent translation
GNN_GEN = os.environ.get("GNN_GEN", "0") == "1"          # generation-head read-off (needs gen-trained ckpt)
LOADING_ARM = os.environ.get("LOADING_ARM", "0") == "1"  # PC1-loading centroid of visible indicators


LLM_BASELINE_MODEL = os.environ.get("LLM_BASELINE_MODEL", "gpt-4o-mini")
MB_LLM_PROMPT_VERSION = "single-llm-visible-typed-mb-v1"
RUN_METADATA = {
    "protocol_version": "visible-label-folds-v2",
    "folds": FOLDS,
    "decode_alpha": "selected separately per fold from visible labels only",
    "loading_centroid": {},
    "mb_llm_name": {
        "model": LLM_BASELINE_MODEL,
        "judge_model": judge_mod.MODEL,
        "prompt_version": MB_LLM_PROMPT_VERSION,
        "input": "typed Markov blanket with visible observed labels only",
        "official_tlvd_reproduction": False,
    },
}


NEG_OP = None
if NEGOP:
    import negop
    NEG_OP = negop.load()
GENOP = os.environ.get("GENOP", "1") == "1"      # v6: Jacobian-locked operator IS the gen path
GEN_OP = None
if GENOP:
    import gen_operator as _go
    GEN_OP = _go.load_or_init()


def ts():
    return time.strftime("%H:%M:%S")


def llm_name(context, model=None):
    """Name a latent from a leakage-safe typed Markov blanket."""
    rendered = external_baselines.format_latent_markov_context(context)
    prompt = (
        "Infer the single most plausible semantic construct represented by an anonymous "
        "latent variable from its typed Markov blanket below. Use only the visible "
        "measure descriptions and structural roles shown; labels hidden by the fold are "
        "unavailable.\n\n"
        + rendered
        + "\n\nReturn only a concise 1-4 word construct name."
    )
    name = judge_mod.chat(prompt, model=model or LLM_BASELINE_MODEL)
    name = name.strip().strip("\"'")
    if not name or len(name) > 120:
        raise RuntimeError("LLM naming returned an empty or overlong response")
    return name


def run_dataset(ds, C, cwords, records):
    g, X, labels, gt = ds["graph"], ds["X"], ds["labels"], ds["latent_gt"]
    obs = g.observed
    oi = {o: k for k, o in enumerate(obs)}
    T = encode.embed([labels[o] for o in obs])
    W, score = g.estimate_weights(X, oi)
    # Freeze a conventional PC1-loading baseline before CORE-only sign/nonlinear
    # processing. It is only meaningful for measurement DAGs (latent-only sources).
    loading_applicable, loading_reason = external_baselines.loading_centroid_applicability(g)
    loading_weights = {}
    if LOADING_ARM:
        RUN_METADATA["loading_centroid"][ds["name"]] = {
            "applicable": bool(loading_applicable),
            "reason": loading_reason,
            "method_version": external_baselines.LOADING_CENTROID_VERSION,
        }
    if LOADING_ARM and loading_applicable:
        for L, s in score.items():
            for o in g.observed_descendants(L):
                value = float(np.corrcoef(s, X[:, oi[o]])[0, 1])
                if np.isfinite(value):
                    loading_weights[(L, o)] = value
    if LATCON:
        import latent_constraints as _LC
        W, score = _LC.sign_fix(g, W, score)
    _mats = None
    if NLDEP:                                    # TRUNK-4a nonlinear target stack (default)
        import nldep as _nl
        _mats = _nl.matrices(g, X, oi, score, ds["name"])
        W = _nl.nl_weights(W, _mats)
        pc = _nl.pc_matrix(_mats) if RESIDUAL > 0 else None
        br = _nl.bridge_dict(_mats) if BRIDGE != "off" else None
    else:                                        # legacy Pearson path (attribution only)
        pc = optimize.partial_residual_corr(g, X, oi, score) if RESIDUAL > 0 else None
        if LATCON and pc is not None:
            import latent_constraints as _LC
            pc = _LC.augmented_partial_corr(g, X, oi, score, pc)
        br = None
        if BRIDGE and BRIDGE != "off":
            import dependence as _dep
            br = dict(obs=list(obs), dep_marg=_dep.load(ds["name"], "marginal", BRIDGE),
                      lam_upper=0.3, kappa=0.5, q=0.7)
            if LATCON:
                import latent_constraints as _LC
                _bn, _bD = _LC.augmented_bridge(g, list(obs), oi, X, score, br["dep_marg"])
                br = dict(obs=_bn, dep_marg=_bD, lam_upper=0.3, kappa=0.5, q=0.7)
    if pc is not None and SHRINK:
        pc = (pc[0], optimize.shrink_corr(pc[1], X.shape[0]))
    Craw = np.corrcoef(X.T); np.fill_diagonal(Craw, 0.0)
    dep = ([o for o in obs], Craw) if LAM_DEP > 0 else None
    import terms as _terms
    # built AFTER sign_fix; CI_MODE default marginal_shrink handled inside ci_table
    ci = _terms.ci_table(g, X, oi, score, nl=_mats)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(obs))
    folds = [perm[i::FOLDS] for i in range(FOLDS)]
    lat_names = [L for L in g.latents if L in gt]
    core_accs, loading_accs, qual, llm_verdicts = [], [], [], []
    llm_attempted = 0
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
    if LOADING_ARM and not loading_applicable:
        print(f"[{ts()}]   loading-centroid skipped: {loading_reason}", flush=True)

    for fno, fold in enumerate(folds):
        masked = set(int(i) for i in fold)
        visible = [i for i in range(len(obs)) if i not in masked]
        visible_nodes = {obs[i] for i in visible}
        vis_emb = {obs[i]: T[i] for i in visible}
        alpha = metrics.fold_alpha(T, C, visible)
        emb = optimize.optimize_embeddings(g, W, vis_emb, d=T.shape[1], steps=STEPS,
                                           lam_zero=LAM_ZERO, lam_norm=LAM_NORM, seed=fno,
                                           free_w=FREE_W, residual=RESIDUAL, lam_res=LAM_RES,
                                           partial_corr=pc, lam_dep=LAM_DEP, dep_corr=dep,
                                           lam_coll=LAM_COLL, neg_op=NEG_OP, bridge=br,
                                           gen_op=GEN_OP, ci=ci)
        U = np.stack([emb[L] for L in lat_names])
        words = metrics.decode_words(U, C, cwords, alpha)
        jacc, verd = metrics.judge_latents(words, [gt[L] for L in lat_names])
        if jacc is not None:
            core_accs.append(jacc)
        for L, w_, ok in zip(lat_names, words, verd or [None] * len(lat_names)):
            records.append({"task": 2, "dataset": ds["name"], "fold": fno, "arm": "core", "latent": L,
                            "gt": gt[L], "decode_alpha": float(alpha), "decoded_words": w_,
                            "judge": (bool(ok) if ok is not None else None)})
        if fno == 0:
            qual = list(zip(lat_names, words, verd or []))
        if LOADING_ARM and loading_applicable:
            loading = external_baselines.loading_centroid(g, vis_emb, W=loading_weights)
            Ul = np.stack([loading[L] for L in lat_names])
            lwords = metrics.decode_words(Ul, C, cwords, alpha)
            lacc, lverd = metrics.judge_latents(lwords, [gt[L] for L in lat_names])
            if lacc is not None:
                loading_accs.append(lacc)
            for L, w_, ok in zip(lat_names, lwords, lverd or [None] * len(lat_names)):
                records.append({"task": 2, "dataset": ds["name"], "fold": fno,
                                "arm": "loading_centroid", "latent": L, "gt": gt[L],
                                "method_version": external_baselines.LOADING_CENTROID_VERSION,
                                "decode_alpha": float(alpha), "decoded_words": w_,
                                "judge": (bool(ok) if ok is not None else None)})
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
                                    "latent": L, "gt": gt[L], "decode_alpha": float(alpha),
                                    "decoded_words": w_,
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
                                    "decode_alpha": float(alpha), "decoded_words": w_,
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
                                    "decode_alpha": float(alpha), "decoded_words": w_,
                                    "judge": (bool(ok) if ok is not None else None)})
        # Simplified TLVD-style comparator: one LLM sees only the fold-visible,
        # typed Markov blanket. This is not the official multi-agent BNE/evidence system.
        if judge_mod.available():
            llm_attempted += len(lat_names)
            fold_rows, generated = [], []
            for L in lat_names:
                context = external_baselines.latent_markov_context(
                    g, L, labels, visible_nodes
                )
                try:
                    nm = llm_name(context)
                    generation_error = None
                except Exception as exc:
                    nm = None
                    generation_error = f"{type(exc).__name__}: {exc}"[:300]
                row = {
                    "task": 2, "dataset": ds["name"], "fold": fno,
                    "arm": "mb_llm_name", "latent": L, "gt": gt[L],
                    "decoded_words": ([nm] if nm else None), "judge": None,
                    "llm_model": LLM_BASELINE_MODEL,
                    "judge_model": judge_mod.MODEL,
                    "prompt_version": MB_LLM_PROMPT_VERSION,
                    "context": context,
                    "generation_error": generation_error,
                }
                fold_rows.append(row)
                if nm is not None:
                    generated.append((len(fold_rows) - 1, L, nm))
            if generated:
                verdicts = judge_mod.judge_batch(
                    [([nm], gt[L]) for _, L, nm in generated], "latent"
                )
                if verdicts is None or len(verdicts) != len(generated):
                    verdicts = [None] * len(generated)
                for (row_index, _, _), ok in zip(generated, verdicts):
                    fold_rows[row_index]["judge"] = (
                        bool(ok) if ok is not None else None
                    )
                    if ok is not None:
                        llm_verdicts.append(bool(ok))
            records.extend(fold_rows)
        print(f"[{ts()}]   fold {fno + 1}/{FOLDS} done", flush=True)

    base_acc = float(np.mean(llm_verdicts)) if llm_verdicts else None
    llm_coverage = (len(llm_verdicts) / llm_attempted) if llm_attempted else None

    print(f"\n[{ts()}] === Task 2 results: {ds['name']} (latent judge-ACC) ===", flush=True)
    print(f"  core (graph-optimized embeddings): "
          f"{np.mean(core_accs):.3f}" if core_accs else "  core: (judge off)", flush=True)
    if LOADING_ARM and loading_applicable:
        print(f"  loading-centroid baseline         : "
              f"{np.mean(loading_accs):.3f}" if loading_accs
              else "  loading-centroid baseline: (judge off)", flush=True)
    elif LOADING_ARM:
        print("  loading-centroid baseline         : N/A for this general DAG", flush=True)
    if gnn_accs:
        print(f"  gnn (trained completion operator): {np.mean(gnn_accs):.3f}", flush=True)
    if jac_accs:
        print(f"  gnn jacobian read-off            : {np.mean(jac_accs):.3f}", flush=True)
    if gen_accs:
        print(f"  gnn generation-head read-off     : {np.mean(gen_accs):.3f}", flush=True)
    print(f"  MB-LLM naming adaptation         : "
          f"{base_acc:.3f} (coverage={llm_coverage:.3f})"
          if base_acc is not None else "  MB-LLM naming adaptation: (skipped)", flush=True)
    for L, w_, ok in qual:
        print(f"    {L} (gt: {ds['latent_gt'][L][:50]}...) <- {', '.join(w_)}"
              f"  [{'OK' if ok else 'X'}]" if ok is not None else "", flush=True)
    return {"core": (float(np.mean(core_accs)) if core_accs else None),
            "loading_centroid": (float(np.mean(loading_accs)) if loading_accs else None),
            "mb_llm_name": base_acc,
            "mb_llm_name_coverage": llm_coverage,
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
    json.dump({"summary": summary, "records": records, "run_metadata": RUN_METADATA},
              open(out, "w"), ensure_ascii=False, indent=1)
    print(f"[saved {out} ({len(records)} items)]", flush=True)


if __name__ == "__main__":
    main()
