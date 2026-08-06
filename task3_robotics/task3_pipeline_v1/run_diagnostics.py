"""P6 single entry: one pass per dataset = official-config solve + all three diagnostics.

Per dataset (13 reporting + bigfive2):
  solve   fold-FOLD (default 0) under the official v6 config: LoRA space, canonical trained
          pair (l2_mlp_v6.pt + gen_operator.pt), nldep targets, CI marginal_shrink, RCHAN=hard.
  P2      certainty.compute: cert(i) for every free node (masked observed + latents).
  P5      adequacy.compute (full CI table) + propose_repairs (proposals only).
  P3      for latents >= 2 hops from text: generative-path influence weights, footprint,
          blended decode words next to the direct decode (free inspection; judge runs
          separately when the user approves spend).
Output: outputs/diagnostics/<ds>.json + printed summary lines. Read-only: no artifact of the
main line is modified. Env: DATASET (csv | all13 | +bigfive2 default), FOLD, TASKDIAG_DECODE=1.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

DICT = os.environ.get("GRAPHSEM_DICT") or os.path.join(HERE, "outputs", "concept_bank_l3_cog.npz")
os.environ["GRAPHSEM_DICT"] = DICT

import numpy as np                                                    # noqa: E402
import torch                                                          # noqa: E402
import encode                                                         # noqa: E402
import lora                                                           # noqa: E402
import graph as G                                                     # noqa: E402
import core                                                           # noqa: E402
import terms as TF                                                    # noqa: E402
import nldep as NL                                                    # noqa: E402
import gen_operator as GO                                             # noqa: E402
import l2_modules as LM                                               # noqa: E402
import latent_constraints as LC                                       # noqa: E402
import certainty                                                      # noqa: E402
import adequacy                                                       # noqa: E402
import influence_decode as INF                                        # noqa: E402
import metrics                                                        # noqa: E402
import negop                                                          # noqa: E402

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", 8)))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FOLD = int(os.environ.get("FOLD", 0))
OUT = os.path.join(HERE, "outputs", "diagnostics")
K = int(os.environ.get("K", 60))

# ---- LoRA encoder (main.py wiring) ----
CKPT = os.path.join(HERE, "outputs", "l3_lora.pt")
_st = lora.load_st(DEVICE)
lora.inject(_st)
lora.load_lora(_st, CKPT)
_st.eval()


class _LoraST:
    def encode(self, texts, batch_size=1024, normalize_embeddings=True):
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), 256):
                stripped = [t[len("query: "):] if t.startswith("query: ") else t
                            for t in texts[i:i + 256]]
                out.append(lora.encode_grad(_st, stripped, DEVICE, max_len=128).cpu().numpy())
        return np.concatenate(out)


encode._MODEL = _LoraST()

import run_task1 as RT1                                               # noqa: E402

GEN_OP = GO.load_or_init(path=os.path.join(HERE, "outputs", "gen_operator.pt"))
MODULE = LM.load(os.path.join(HERE, "outputs", "l2_mlp_v6.pt"))
FNEG = negop.load()

# kept in sync with run_bigfive_hier (its module level loads the encoder, so no import)
BIG_TWO = {"stability": ["agreeableness", "conscientiousness", "neuroticism"],
           "plasticity": ["extraversion", "openness"]}
GT_NEW = {
    "stability": "the higher-order personality metatrait: stability (alpha) --- shared variance "
                 "of agreeableness, conscientiousness, and low neuroticism",
    "plasticity": "the higher-order personality metatrait: plasticity (beta) --- shared variance "
                  "of extraversion and openness",
    "GFP": "the general factor of personality (shared variance of all five factors)",
}


def load_bigfive2():
    ds = RT1.ALL_LOADERS["bigfive"]()
    g0 = ds["graph"]
    edges = list(g0.edges)
    for up, downs in BIG_TWO.items():
        for d_ in downs:
            edges.append((up, d_))
    edges += [("GFP", "stability"), ("GFP", "plasticity")]
    g = G.Graph(list(g0.latents) + ["stability", "plasticity", "GFP"], list(g0.observed), edges)
    gt = dict(ds["latent_gt"])
    gt.update(GT_NEW)
    return dict(name="bigfive2", graph=g, X=ds["X"], labels=ds["labels"], latent_gt=gt)


def run_dataset(name):
    ds = load_bigfive2() if name == "bigfive2" else RT1.ALL_LOADERS[name]()
    g, X, labels = ds["graph"], ds["X"], ds["labels"]
    obs = g.observed
    oi = {o: k for k, o in enumerate(obs)}
    T = encode.embed([labels[o] for o in obs])
    W, score = g.estimate_weights(X, oi)
    W, score = LC.sign_fix(g, W, score)
    mats = NL.matrices(g, X, oi, score, name)
    W = NL.nl_weights(W, mats)
    pc = NL.pc_matrix(mats)
    br = NL.bridge_dict(mats)
    ci_obj = TF.ci_table(g, X, oi, score, nl=mats)                    # objective (marginal_shrink)
    ci_full = TF.ci_table(g, X, oi, score, nl=mats, mode="full")      # diagnostics

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(obs))
    fold = sorted(int(i) for i in perm[FOLD::5])
    vis = {obs[i]: T[i] for i in range(len(obs)) if i not in set(fold)}
    feats = torch.tensor(LM.node_features(g, W, set(vis)))
    emb, _ = core.solve_unrolled(
        g, W, vis, d=T.shape[1], gen_op=GEN_OP, ci=ci_obj, weight_module=MODULE, K=K,
        lam_zero=0.3, lam_norm=0.1, seed=FOLD, residual=1.0, lam_res=1.0,
        partial_corr=pc, bridge=br, train=False, feats=feats)

    rec = {"dataset": name, "fold": FOLD}

    # P2 certainty
    cert = certainty.compute(g, W, emb, GEN_OP, ci_obj, labeled=set(vis), bridge=br)
    lat_cert = {n: v for n, v in cert.items() if g.is_latent(n)}
    rec["cert"] = {n: round(v, 6) for n, v in sorted(cert.items(), key=lambda kv: kv[1])}
    print(f"[{name}] P2 cert: lowest 5 = "
          f"{[(n, round(v, 4)) for n, v in sorted(cert.items(), key=lambda kv: kv[1])[:5]]}",
          flush=True)
    if lat_cert:
        print(f"[{name}]    latent cert range: min={min(lat_cert.values()):.4f} "
              f"max={max(lat_cert.values()):.4f}", flush=True)

    # P5 adequacy + repair proposals
    adq = adequacy.compute(g, X, oi, score, ci=ci_full)
    props = adequacy.propose_repairs(g, ci_full)
    rec["adequacy"] = adq
    rec["repair_proposals"] = props[:10]
    print(f"[{name}] P5 V(G,X): marginal {adq['V_marginal']:.2f} "
          f"({adq['n_marginal_violations']}/{adq['n_marginal_claims']}), conditional "
          f"{adq['V_conditional']:.2f} ({adq['n_conditional_violations']}/"
          f"{adq['n_conditional_claims']}); top proposal: "
          f"{props[0]['proposal'] if props else 'none'} "
          f"({props[0]['nodes'][:6] if props else []})", flush=True)

    # P3 influence decode for deep latents
    deep = [L for L in g.latents if INF.hop_depth(g, L) >= 2]
    if deep and os.environ.get("DIAG_DECODE", "1") == "1":
        C, cwords = encode.load_dictionary()
        alpha = metrics.pick_alpha(T, C)
        p3 = {}
        for L in deep:
            wts = INF.gen_influence(g, W, emb, GEN_OP, L, probes=8)
            signs = INF.data_signs(g, X, oi, score, L)
            fp = INF.footprint(emb, wts, signs, fneg=FNEG)
            bl = INF.blended(emb[L], fp)
            direct_w = metrics.decode_words(np.asarray(emb[L])[None, :], C, cwords, alpha)[0][:6]
            blend_w = metrics.decode_words(np.asarray(bl)[None, :], C, cwords, alpha)[0][:6]
            p3[L] = {"direct": direct_w, "blended": blend_w,
                     "top_influence": sorted(wts, key=wts.get, reverse=True)[:5]}
            print(f"[{name}] P3 {L}: direct={direct_w} | blended={blend_w}", flush=True)
        rec["p3"] = p3

    os.makedirs(OUT, exist_ok=True)
    json.dump(rec, open(os.path.join(OUT, f"{name}.json"), "w"), indent=1, default=str)
    return rec


if __name__ == "__main__":
    which = os.environ.get("DATASET", "")
    if which:
        names = [w.strip() for w in which.split(",")]
    else:
        names = list(RT1.select_datasets("all")) + ["bigfive2"]
    for nm in names:
        try:
            run_dataset(nm)
        except Exception as e:                                        # keep the sweep alive
            print(f"[{nm}] FAILED: {type(e).__name__}: {e}", flush=True)
    print("[diagnostics done]", flush=True)
