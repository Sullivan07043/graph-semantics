"""Step C: forward-beta GENERATION-FOOTPRINT latent translation + ensemble (J-space lesson: read
meaning at the OUTPUT side, not only the internal state; lineage: the forward-beta read-off).

Footprint of latent L (fold-local, visible labels only):
    u_fwd(L) = normalize( sum_{c in obsdesc(L), c visible} |w_path(L,c)| * phi(a_c) )
    phi(a_c) = a_c if the path sign is positive else f_neg(a_c)   (semantic negation operator)
w_path = product of edge weights along the (tree) chain from L down to c.

Arms judged on the SAME folds in one batch: core (frozen v2 optimizer u_L — deliberately WITHOUT the
new negop integration so step C is not conflated with step A), fwdbeta (u_fwd alone), ens (decode of
the normalized mean of unit u_core and unit u_fwd). Records -> outputs/fwdbeta_<ds>.json."""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
sys.path.insert(0, HERE)
import pool, encode, metrics, optimize
from run_task1 import ALL_LOADERS

FOLDS = 5


def ts():
    return time.strftime("%H:%M:%S")


def path_weight(g, W, L, c):
    """Signed product of edge weights climbing the (tree) parent chain from c up to L."""
    w, node = 1.0, c
    while node != L:
        ps = g.parents(node)
        if not ps:
            return 0.0
        p = ps[0]
        w *= float(W.get((p, node), 0.0))
        node = p
    return w


def run(name, C, cwords, neg):
    import torch
    ds = ALL_LOADERS[name]()
    g, X, labels, gt = ds["graph"], ds["X"], ds["labels"], ds["latent_gt"]
    obs = g.observed
    oi = {o: k for k, o in enumerate(obs)}
    T = encode.embed([labels[o] for o in obs])
    alpha = metrics.pick_alpha(T, C)
    W, score = g.estimate_weights(X, oi)
    pc = optimize.partial_residual_corr(g, X, oi, score)
    lat_names = [L for L in g.latents if L in gt]
    with torch.no_grad():
        Tneg = neg(torch.tensor(T, dtype=torch.float32)).numpy().astype(np.float64)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(obs))
    folds = [perm[i::FOLDS] for i in range(FOLDS)]
    accs = {a: [] for a in ["core", "fwdbeta", "ens"]}
    records = []
    for fno, fold in enumerate(folds):
        masked = set(int(i) for i in fold)
        vis = {obs[i]: T[i] for i in range(len(obs)) if i not in masked}
        emb = optimize.optimize_embeddings(g, W, vis, d=T.shape[1], seed=fno,
                                           residual=1.0, lam_res=1.0, partial_corr=pc)
        Uc = np.stack([emb[L] for L in lat_names])
        Uf = np.zeros_like(Uc)
        for li, L in enumerate(lat_names):
            v = np.zeros(T.shape[1])
            for c in g.observed_descendants(L):
                k = oi[c]
                if k in masked:
                    continue
                w = path_weight(g, W, L, c)
                v += abs(w) * (T[k] if w >= 0 else Tneg[k])
            Uf[li] = v
        nrm = lambda M: M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
        Ue = nrm(nrm(Uc) + nrm(Uf))
        for arm, U in [("core", Uc), ("fwdbeta", Uf), ("ens", Ue)]:
            words = metrics.decode_words(U, C, cwords, alpha)
            acc, verd = metrics.judge_latents(words, [gt[L] for L in lat_names])
            if acc is not None:
                accs[arm].append(acc)
            for L, w_, ok in zip(lat_names, words, verd or [None] * len(lat_names)):
                records.append({"dataset": name, "fold": fno, "arm": arm, "latent": L, "gt": gt[L],
                                "decoded_words": w_, "judge": (bool(ok) if ok is not None else None)})
    line = " ".join(f"{a}={np.mean(v):.3f}" if v else f"{a}=-" for a, v in accs.items())
    print(f"[{ts()}] {name:10s} {line}", flush=True)
    out = os.path.join(HERE, "outputs", f"fwdbeta_{name}.json")
    json.dump({"summary": {a: (float(np.mean(v)) if v else None) for a, v in accs.items()},
               "records": records}, open(out, "w"), indent=1)


def main():
    import negop
    neg = negop.load()
    C, cwords = encode.load_dictionary()
    for n in os.environ.get("DATASETS", "himi,bigfive,gcbs,sd3,hexaco,riasec,kims").split(","):
        run(n.strip(), C, cwords, neg)


if __name__ == "__main__":
    main()
