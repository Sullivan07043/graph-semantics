"""Task-2 DEPLOYMENT-SETTING translation: all observed labels VISIBLE (the actual use case — masking
is a Task-1 evaluation device; LLM-naming always saw all labels, so per-fold-masked translation
under-served our side), plus J-space-style EXPECTATION AVERAGING: solve the graph optimization under
several seeds and average each latent's embedding before decoding (their J is stable because it is
averaged over ~1000 prompts; our single-solve u_L was not).

Arms: full (all labels, seed-averaged u) vs the old masked-fold protocol numbers (from records).
Judge gpt-5.5. Output: RECORDS_OUT (default outputs/t2full_<ds>.json).
"""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
sys.path.insert(0, HERE)
import pool, encode, metrics, optimize
from run_task1 import ALL_LOADERS

SEEDS = int(os.environ.get("SEEDS", 5))


def ts():
    return time.strftime("%H:%M:%S")


def run(name, C, cwords):
    ds = ALL_LOADERS[name]()
    g, X, labels, gt = ds["graph"], ds["X"], ds["labels"], ds["latent_gt"]
    obs = g.observed
    oi = {o: k for k, o in enumerate(obs)}
    T = encode.embed([labels[o] for o in obs])
    alpha = metrics.pick_alpha(T, C)
    W, score = g.estimate_weights(X, oi)
    pc = optimize.partial_residual_corr(g, X, oi, score)
    lat_names = [L for L in g.latents if L in gt]
    vis = {obs[i]: T[i] for i in range(len(obs))}                    # ALL labels visible
    U = None
    for s in range(SEEDS):
        emb = optimize.optimize_embeddings(g, W, vis, d=T.shape[1], seed=s,
                                           residual=1.0, lam_res=1.0, partial_corr=pc)
        Us = np.stack([emb[L] for L in lat_names])
        Us = Us / (np.linalg.norm(Us, axis=1, keepdims=True) + 1e-9)
        U = Us if U is None else U + Us
    U = U / SEEDS
    words = metrics.decode_words(U, C, cwords, alpha)
    acc, verd = metrics.judge_latents(words, [gt[L] for L in lat_names])
    recs = [{"dataset": name, "arm": "full_avg", "latent": L, "gt": gt[L], "decoded_words": w_,
             "judge": (bool(v) if v is not None else None)}
            for L, w_, v in zip(lat_names, words, verd or [None] * len(lat_names))]
    print(f"[{ts()}] {name:10s} full-label seed-averaged translation judge-ACC = "
          f"{acc if acc is not None else float('nan'):.3f}  ({len(lat_names)} latents)", flush=True)
    return acc, recs


def main():
    C, cwords = encode.load_dictionary()
    names = os.environ.get("DATASETS", "himi,bigfive,gcbs,sd3,hexaco,riasec,kims").split(",")
    summary, records = {}, []
    for n in names:
        summary[n], r = run(n.strip(), C, cwords)
        records += r
    out = os.environ.get("RECORDS_OUT", os.path.join(HERE, "outputs", "t2full.json"))
    json.dump({"summary": summary, "records": records}, open(out, "w"), indent=1)
    print(f"[saved {out}]", flush=True)


if __name__ == "__main__":
    main()
