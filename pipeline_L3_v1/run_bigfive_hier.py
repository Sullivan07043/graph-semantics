"""Big Five with a published 2-level latent SUPERSTRUCTURE, evaluated by the current main line.

Added latents (literature-given, not fitted):
  GFP (general factor of personality)
    -> stability (alpha):   agreeableness, conscientiousness, neuroticism (sign from data)
    -> plasticity (beta):   extraversion, openness
Everything else identical to the official protocol: same folds, same solver (LoRA space +
WeightNet), same decode. Registered as dataset name "bigfive2" (original loaders untouched).
Env: TASK=1|2, RECORDS_OUT; keyless => free metrics only.
"""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "pipeline_v3"))

DICT = os.path.join(HERE, "outputs", "concept_bank_l3.npz")
os.environ["GRAPHSEM_DICT"] = DICT
os.environ.setdefault("DATASET", "bigfive2")

import numpy as np                                                    # noqa: E402
import torch                                                          # noqa: E402
import encode                                                         # noqa: E402
import graph as G                                                     # noqa: E402
from pipeline_L3_v1 import lora                                       # noqa: E402
from pipeline_v4 import core, l2_modules as LM                        # noqa: E402

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", 4)))

BIG_TWO = {
    "stability": ["agreeableness", "conscientiousness", "neuroticism"],
    "plasticity": ["extraversion", "openness"],
}
GT_NEW = {
    "stability": "the higher-order personality metatrait: stability (alpha) --- shared variance of "
                 "agreeableness, conscientiousness, and low neuroticism",
    "plasticity": "the higher-order personality metatrait: plasticity (beta) --- shared variance of "
                  "extraversion and openness",
    "GFP": "the general factor of personality (shared variance of all five factors)",
}


def load_bigfive2():
    import run_task1
    ds = run_task1.ALL_LOADERS["bigfive"]()
    g0 = ds["graph"]
    edges = list(g0.edges)
    for up, downs in BIG_TWO.items():
        for d in downs:
            edges.append((up, d))
    edges += [("GFP", "stability"), ("GFP", "plasticity")]
    lats = list(g0.latents) + ["stability", "plasticity", "GFP"]
    g = G.Graph(lats, list(g0.observed), edges)
    gt = dict(ds["latent_gt"])
    gt.update(GT_NEW)
    return dict(name="bigfive2", graph=g, X=ds["X"], labels=ds["labels"], latent_gt=gt)


# ---- LoRA encoder + l3 dictionary (identical to run_eval_l3) ----
CKPT = os.path.join(HERE, "outputs", "l3_lora.pt")
_v = np.load(DICT, allow_pickle=True)
assert abs(float(_v["lora_version"]) - os.path.getmtime(CKPT)) < 1.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
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

# ---- WeightNet solver (identical to run_eval_l3, L2_ARM=mlp) ----
import optimize                                                       # noqa: E402
from pipeline_L3_v1 import latent_constraints as LC                   # noqa: E402
LATCON = os.environ.get("LATCON", "0") == "1"
_DS_CACHE = {}


def _latcon_inputs(g, W_in):
    """Recompute score, apply sign convention, build augmented anchors/dependence."""
    key = id(g)
    if key not in _DS_CACHE:
        ds = load_bigfive2()
        X = ds["X"]
        oi = {o: k for k, o in enumerate(g.observed)}
        W0, score0 = g.estimate_weights(X, oi)
        Wf, scoref = LC.sign_fix(g, W0, score0)
        base_pc = optimize.partial_residual_corr(g, X, oi, scoref)
        pc_aug = LC.augmented_partial_corr(g, X, oi, scoref, base_pc)
        import dependence as depmod
        base_dep = depmod.load("bigfive2", "marginal", "pearson")
        br_names, br_D = LC.augmented_bridge(g, list(g.observed), oi, X, scoref, base_dep)
        _DS_CACHE[key] = (Wf, pc_aug, dict(obs=br_names, dep_marg=br_D,
                                           lam_upper=0.3, kappa=0.5, q=0.7))
    return _DS_CACHE[key]
_MODULE = LM.load(os.environ.get("L2_CKPT", os.path.join(HERE, "outputs", "l2_mlp.pt")))
K = int(os.environ.get("K", 60))


def _l2_solve(g, W, labeled_emb, d, steps=400, lr=2e-2, lam_zero=0.3, lam_norm=0.1,
              seed=0, device="cpu", free_w=False, als_rounds=5,
              residual=0.0, lam_res=0.0, partial_corr=None,
              lam_dep=0.0, dep_corr=None, dep_kappa=0.5, lam_coll=0.0,
              neg_op=None, bridge=None, gen_op=None, verbose=False):
    if LATCON:
        W, partial_corr, bridge = _latcon_inputs(g, W)
    feats = torch.tensor(LM.node_features(g, W, set(labeled_emb)), device=device)
    emb, _ = core.solve_unrolled(
        g, W, labeled_emb, d, weight_module=_MODULE, K=K, inner_lr=2e-2,
        lam_zero=lam_zero, lam_norm=lam_norm, seed=seed, device=device,
        residual=residual, lam_res=lam_res, partial_corr=partial_corr,
        neg_op=neg_op, bridge=bridge, train=False, feats=feats)
    return emb


optimize.optimize_embeddings = _l2_solve

if __name__ == "__main__":
    task = os.environ.get("TASK", "1")
    if task == "1":
        import run_task1 as runner                                    # noqa: E402
    else:
        import run_task2 as runner                                    # noqa: E402
    runner.ALL_LOADERS["bigfive2"] = load_bigfive2
    runner.main()
