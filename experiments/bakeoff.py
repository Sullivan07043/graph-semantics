"""Encoder bake-off (GLOBAL choice, dev pool only, geometric metrics only — no judge, no dictionary).
Candidates: all-MiniLM-L6-v2 (current), thenlper/gte-large, intfloat/e5-large-v2 (symmetric 'query: '
prefix per its model card). Protocol: identical 5-fold masking; arms rawcorr and core (ALS+Adam defaults);
metrics: mean cosine of the predicted embedding to the true held-out label embedding, and matching-ACC.
The winner is picked on the dev mean of the CORE arm and then frozen for everything downstream.
Output: outputs/bakeoff.json + printed table."""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
sys.path.insert(0, HERE)
import testbeds, pool, metrics, optimize
from run_task1 import ALL_LOADERS

ENCODERS = [
    ("minilm", "sentence-transformers/all-MiniLM-L6-v2", ""),
    ("gte-large", "thenlper/gte-large", ""),
    ("e5-large", "intfloat/e5-large-v2", "query: "),
]
FOLDS = 5


def ts():
    return time.strftime("%H:%M:%S")


def embed_with(model_name, prefix, texts, dev):
    from sentence_transformers import SentenceTransformer
    key = f"_m_{model_name}"
    m = globals().get(key)
    if m is None:
        m = SentenceTransformer(model_name, cache_folder=os.environ.get("HF_CACHE",
                                "/data2/shuhao/hf_cache"), device=dev)
        globals()[key] = m
    return np.asarray(m.encode([prefix + t for t in texts], normalize_embeddings=True), np.float64)


def eval_dataset(ds, T):
    g, X = ds["graph"], ds["X"]
    obs = g.observed
    oi = {o: k for k, o in enumerate(obs)}
    Tn = metrics.norm_rows(T)
    W, _ = g.estimate_weights(X, oi)
    Craw = np.corrcoef(X.T); np.fill_diagonal(Craw, 0.0)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(obs))
    folds = [perm[i::FOLDS] for i in range(FOLDS)]
    out = {a: {"cos": [], "match": [], "mrr": []} for a in ["rawcorr", "core"]}
    for fno, fold in enumerate(folds):
        masked = sorted(int(i) for i in fold)
        visible = [i for i in range(len(obs)) if i not in set(masked)]
        P = np.zeros((len(masked), T.shape[1]))
        for r, i in enumerate(masked):
            w = np.zeros(len(obs)); w[visible] = np.clip(Craw, 0, None)[i, visible]
            if w.sum() < 1e-9:
                w[visible] = 1.0
            P[r] = (w / w.sum()) @ T
        preds = {"rawcorr": P}
        emb = optimize.optimize_embeddings(g, W, {obs[i]: T[i] for i in visible}, d=T.shape[1], seed=fno)
        preds["core"] = np.stack([emb[obs[i]] for i in masked])
        for a, P_ in preds.items():
            Pn = metrics.norm_rows(P_)
            out[a]["cos"].append(float(np.mean((Pn * Tn[masked]).sum(1))))
            out[a]["match"].append(metrics.match_acc(P_, masked, T))
            # encoder-invariant retrieval: reciprocal rank of the true label among ALL dataset labels
            S = Pn @ Tn.T                                            # [n_masked, n_obs]
            rr = [1.0 / (1 + int((S[r] > S[r, i]).sum())) for r, i in enumerate(masked)]
            out[a]["mrr"].append(float(np.mean(rr)))
    return {a: {k: float(np.mean(v)) for k, v in d_.items()} for a, d_ in out.items()}


def main():
    import torch
    dev = "cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else "cpu"
    names = pool.DEV
    data = {n: ALL_LOADERS[n]() for n in names}
    results = {}
    for enc, model, prefix in ENCODERS:
        results[enc] = {}
        for n in names:
            ds = data[n]
            T = embed_with(model, prefix, [ds["labels"][o] for o in ds["graph"].observed], dev)
            results[enc][n] = eval_dataset(ds, T)
            print(f"[{ts()}] {enc:9s} {n:9s} core cos={results[enc][n]['core']['cos']:.3f} "
                  f"match={results[enc][n]['core']['match']:.3f} | rawcorr "
                  f"cos={results[enc][n]['rawcorr']['cos']:.3f}", flush=True)
        core = np.mean([results[enc][n]["core"]["cos"] for n in names])
        cm = np.mean([results[enc][n]["core"]["match"] for n in names])
        print(f"[{ts()}] == {enc}: dev-mean core cos={core:.3f} match={cm:.3f}", flush=True)
        results[enc]["_dev_mean"] = {"core_cos": float(core), "core_match": float(cm)}
    os.makedirs(os.path.join(HERE, "outputs"), exist_ok=True)
    json.dump(results, open(os.path.join(HERE, "outputs", "bakeoff.json"), "w"), indent=1)
    print("[saved outputs/bakeoff.json]", flush=True)


if __name__ == "__main__":
    main()
