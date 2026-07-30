"""Check a discovered structure with the v6 diagnostics.

Adapter: outputs/<ds>_rlcd.json -> graph.Graph. Latents = nodes matching ^L\\d+$ (RLCD
naming). Undirected leftover edges are oriented lexicographically — a DECLARED arbitrary
convention, reported, never hidden. All dataset observed columns are kept (nodes RLCD left
isolated become roots).

Checks (same code paths as the published-graph diagnostics):
  P5  V(G_hat, X) + repair proposals, printed next to V(G_pub, X) from
      v6/outputs/diagnostics/<ds>.json (computed last night, FOLD=0 protocol).
  P2  cert on the discovered latents after a fold-0 solve in LoRA space with the v6
      canonical trained pair (same protocol as the published-graph cert numbers).

Usage: DATASET=rse python check_discovered.py
"""
import json
import os
import re
import sys
import time

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
import adequacy                                                       # noqa: E402
import certainty                                                      # noqa: E402

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", 8)))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAT_RE = re.compile(r"^L\d+$")


def load_discovered(name, obs_all):
    d = json.load(open(os.path.join(HERE, "outputs", f"{name}_rlcd.json")))
    edges = [tuple(e) for e in d["directed"]]
    oriented = [tuple(sorted(e)) for e in d["undirected"]]
    edges += oriented
    nodes = {x for e in edges for x in e}
    lats = sorted(n for n in nodes if LAT_RE.match(n))
    g = G.Graph(lats, obs_all, edges)
    return g, oriented


def main():
    name = os.environ.get("DATASET", "rse")
    ds = {**testbeds.LOADERS, **pool.LOADERS}[name]()
    g_pub, X, labels = ds["graph"], ds["X"], ds["labels"]
    obs = list(g_pub.observed)
    oi = {o: k for k, o in enumerate(obs)}
    g_hat, oriented = load_discovered(name, obs)
    print(f"[{name}] discovered: {len(g_hat.latents)} latents, {len(g_hat.edges)} edges "
          f"({len(oriented)} undirected edges oriented lexicographically: {oriented})",
          flush=True)

    # data-side prep on the DISCOVERED graph (nl cache under its own name)
    W, score = g_hat.estimate_weights(X, oi)
    W, score = LC.sign_fix(g_hat, W, score)
    mats = NL.matrices(g_hat, X, oi, score, f"{name}-rlcd")
    W = NL.nl_weights(W, mats)
    pc = NL.pc_matrix(mats)
    br = NL.bridge_dict(mats)
    ci_obj = TF.ci_table(g_hat, X, oi, score, nl=mats)
    ci_full = TF.ci_table(g_hat, X, oi, score, nl=mats, mode="full")

    # P5: V on discovered vs published (published read from last night's archive)
    adq = adequacy.compute(g_hat, X, oi, score, ci=ci_full)
    props = adequacy.propose_repairs(g_hat, ci_full)
    pub = json.load(open(os.path.join(V6, "outputs", "diagnostics", f"{name}.json")))["adequacy"]
    print(f"[{name}] V marginal:    discovered {adq['V_marginal']:8.2f} "
          f"({adq['n_marginal_violations']}/{adq['n_marginal_claims']})   "
          f"published {pub['V_marginal']:8.2f} "
          f"({pub['n_marginal_violations']}/{pub['n_marginal_claims']})", flush=True)
    print(f"[{name}] V conditional: discovered {adq['V_conditional']:8.2f} "
          f"({adq['n_conditional_violations']}/{adq['n_conditional_claims']})   "
          f"published {pub['V_conditional']:8.2f} "
          f"({pub['n_conditional_violations']}/{pub['n_conditional_claims']})", flush=True)
    if props:
        print(f"[{name}] top repair proposal on discovered: {props[0]['proposal']} "
              f"{props[0]['nodes'][:6]} (mass {props[0]['mass']:.2f})", flush=True)

    # P2: cert after a fold-0 solve (v6 canonical pair, LoRA space)
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
    gen_op = GO.load_or_init(path=os.path.join(V6, "outputs", "gen_operator.pt"))
    module = LM.load(os.path.join(V6, "outputs", "l2_mlp_v6.pt"))
    rng = np.random.default_rng(0)
    fold = sorted(int(i) for i in rng.permutation(len(obs))[0::5])
    vis = {obs[i]: T[i] for i in range(len(obs)) if i not in set(fold)}
    feats = torch.tensor(LM.node_features(g_hat, W, set(vis)))
    emb, _ = core.solve_unrolled(
        g_hat, W, vis, d=T.shape[1], gen_op=gen_op, ci=ci_obj, weight_module=module, K=60,
        lam_zero=0.3, lam_norm=0.1, seed=0, residual=1.0, lam_res=1.0,
        partial_corr=pc, bridge=br, train=False, feats=feats)
    cert = certainty.compute(g_hat, W, emb, gen_op, ci_obj, labeled=set(vis), bridge=br)
    lat_cert = {L: round(cert[L], 4) for L in g_hat.latents if L in cert}
    print(f"[{name}] discovered-latent cert: {lat_cert}", flush=True)

    out = dict(dataset=name, latents=g_hat.latents, oriented=oriented,
               V_discovered={k: adq[k] for k in adq if k != 'top_pairs'},
               V_published={k: pub[k] for k in pub if k != 'top_pairs'},
               top_proposal=props[0] if props else None, latent_cert=lat_cert)
    json.dump(out, open(os.path.join(HERE, "outputs", f"{name}_check.json"), "w"),
              indent=1, default=str)
    print(f"[saved outputs/{name}_check.json]", flush=True)


if __name__ == "__main__":
    main()
