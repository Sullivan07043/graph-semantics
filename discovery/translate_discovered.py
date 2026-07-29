"""Translate the latents of a discovered/refined structure — the auto-naming step.

Protocol: the standard 5-fold T2 solve (mask 20% observed labels per fold, solve with the v6
canonical pair on the discovered graph), decode every latent per fold with the SpLiCE
dictionary. Keyless = free (no judge; discovered latents have no GT names — output is the
translation itself plus each latent's children for human reading).
Usage: DATASET=rse SUFFIX=refined python translate_discovered.py
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
V6 = os.path.join(os.path.dirname(HERE), "v6")
sys.path.insert(0, V6)

os.environ.setdefault("GRAPHSEM_DICT", os.path.join(V6, "outputs", "concept_bank_l3_cog.npz"))

import numpy as np                                                    # noqa: E402
import torch                                                          # noqa: E402
import graph as G                                                     # noqa: E402
import testbeds                                                       # noqa: E402
import pool                                                           # noqa: E402
import encode                                                         # noqa: E402
import lora                                                           # noqa: E402
import core                                                           # noqa: E402
import terms as TF                                                    # noqa: E402
import nldep as NL                                                    # noqa: E402
import gen_operator as GO                                             # noqa: E402
import l2_modules as LM                                               # noqa: E402
import latent_constraints as LC                                       # noqa: E402
import metrics                                                        # noqa: E402

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", 8)))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAT_RE = re.compile(r"^L\d+$")


def main():
    name = os.environ.get("DATASET", "rse")
    suffix = os.environ.get("SUFFIX", "refined")
    ds = {**testbeds.LOADERS, **pool.LOADERS}[name]()
    g_pub, X, labels = ds["graph"], ds["X"], ds["labels"]
    obs = list(g_pub.observed)
    oi = {o: k for k, o in enumerate(obs)}
    d = json.load(open(os.path.join(HERE, "outputs", f"{name}_{suffix}.json")))
    edges = [tuple(e) for e in d["directed"]] + [tuple(sorted(e)) for e in d.get("undirected", [])]
    nodes = {x for e in edges for x in e}
    lats = sorted((n for n in nodes if LAT_RE.match(n)), key=lambda s: int(s[1:]))
    g = G.Graph(lats, obs, edges)
    for L in lats:
        ch = [c for c in g.children(L)]
        print(f"[{name}] {L}: children = {ch}", flush=True)

    W, score = g.estimate_weights(X, oi)
    W, score = LC.sign_fix(g, W, score)
    mats = NL.matrices(g, X, oi, score, None)
    W = NL.nl_weights(W, mats)
    pc = NL.pc_matrix(mats)
    br = NL.bridge_dict(mats)
    ci = TF.ci_table(g, X, oi, score, nl=mats)

    st = lora.load_st(DEVICE)
    lora.inject(st)
    lora.load_lora(st, os.path.join(V6, "outputs", "l3_lora.pt"))
    st.eval()

    class _LoraST:
        def encode(self, texts, batch_size=1024, normalize_embeddings=True):
            out = []
            with torch.no_grad():
                for i in range(0, len(texts), 256):
                    stripped = [t[len("query: "):] if t.startswith("query: ") else t
                                for t in texts[i:i + 256]]
                    out.append(lora.encode_grad(st, stripped, DEVICE, max_len=128).cpu().numpy())
            return np.concatenate(out)

    encode._MODEL = _LoraST()
    T = encode.embed([labels[o] for o in obs])
    C, cwords = encode.load_dictionary()
    alpha = metrics.pick_alpha(T, C)
    gen_op = GO.load_or_init(path=os.path.join(V6, "outputs", "gen_operator.pt"))
    module = LM.load(os.path.join(V6, "outputs", "l2_mlp_v6.pt"))

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(obs))
    words_per_latent = {L: [] for L in lats}
    for fold in range(5):
        masked = set(int(i) for i in perm[fold::5])
        vis = {obs[i]: T[i] for i in range(len(obs)) if i not in masked}
        feats = torch.tensor(LM.node_features(g, W, set(vis)))
        emb, _ = core.solve_unrolled(
            g, W, vis, d=T.shape[1], gen_op=gen_op, ci=ci, weight_module=module, K=60,
            lam_zero=0.3, lam_norm=0.1, seed=fold, residual=1.0, lam_res=1.0,
            partial_corr=pc, bridge=br, train=False, feats=feats)
        U = np.stack([emb[L] for L in lats])
        wl = metrics.decode_words(U, C, cwords, alpha)
        for L, w in zip(lats, wl):
            words_per_latent[L].append(w[:6])
    print(flush=True)
    for L in lats:
        print(f"[{name}] {L} translations across folds:", flush=True)
        for f, w in enumerate(words_per_latent[L]):
            print(f"    fold {f}: {w}", flush=True)
    json.dump({"dataset": name, "suffix": suffix,
               "children": {L: g.children(L) for L in lats},
               "translations": words_per_latent},
              open(os.path.join(HERE, "outputs", f"{name}_{suffix}_translations.json"), "w"),
              indent=1)
    print(f"[saved outputs/{name}_{suffix}_translations.json]", flush=True)


if __name__ == "__main__":
    main()
