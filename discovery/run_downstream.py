"""Downstream evaluation of DISCOVERED structures through the OFFICIAL v6 runners.

Loader injection only (run_bigfive_hier pattern): the RLCD structure from
discovery/outputs/<BASE>.json is registered as dataset "<BASE>rlcd" and run_task1/run_task2
run their standard protocol on it — same solver, checkpoints, targets, metrics, judge.

Judge targets for discovered latents: each discovered latent's reference is the published
construct whose item set overlaps its observed children the most (majority overlap; ties by
first). On single-construct scales (rse, cfcs) this reduces to the published construct
description by construction. The published key is a CLUSTER-LEVEL REFERENCE for judging,
never an input to discovery (declared).

Env: BASE (any dataset with a discovery/outputs/<BASE>.json), TASK=1|2, RECORDS_OUT, and the
usual v6 knobs. JUDGE_MODEL must be set explicitly for judged runs (official: gpt-5.5).
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
V6 = os.path.join(os.path.dirname(HERE), "v6")
sys.path.insert(0, V6)

BASE = os.environ["BASE"]
os.environ["DATASET"] = f"{BASE}rlcd"
DICT = os.environ.get("GRAPHSEM_DICT") or os.path.join(V6, "outputs", "concept_bank_l3_cog.npz")
os.environ["GRAPHSEM_DICT"] = DICT

import numpy as np                                                    # noqa: E402
import torch                                                          # noqa: E402
import encode                                                         # noqa: E402
import lora                                                           # noqa: E402
import graph as G                                                     # noqa: E402

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", 8)))
CKPT = os.path.join(V6, "outputs", "l3_lora.pt")
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

import optimize                                                       # noqa: E402
import core                                                           # noqa: E402
import l2_modules as LM                                               # noqa: E402

_MODULE = LM.load(os.environ.get("L2_CKPT", os.path.join(V6, "outputs", "l2_mlp_v6.pt")))
K = int(os.environ.get("K", 60))


def _l2_solve(g, W, labeled_emb, d, steps=400, lr=2e-2, lam_zero=0.3, lam_norm=0.1,
              seed=0, device="cpu", free_w=False, als_rounds=5,
              residual=0.0, lam_res=0.0, partial_corr=None,
              lam_dep=0.0, dep_corr=None, dep_kappa=0.5, lam_coll=0.0,
              neg_op=None, bridge=None, gen_op=None, ci=None, verbose=False):
    feats = torch.tensor(LM.node_features(g, W, set(labeled_emb)), device=device)
    emb, _ = core.solve_unrolled(
        g, W, labeled_emb, d, gen_op=gen_op, ci=ci, weight_module=_MODULE, K=K,
        inner_lr=2e-2, lam_zero=lam_zero, lam_norm=lam_norm, seed=seed, device=device,
        residual=residual, lam_res=lam_res, partial_corr=partial_corr,
        bridge=bridge, train=False, feats=feats)
    return emb


optimize.optimize_embeddings = _l2_solve

LAT_RE = re.compile(r"^L\d+$")

if __name__ == "__main__":
    task = os.environ.get("TASK", "1")
    if task == "1":
        import run_task1 as runner                                    # noqa: E402
    else:
        import run_task2 as runner                                    # noqa: E402

    def load_discovered():
        base = runner.ALL_LOADERS[BASE]()
        d = json.load(open(os.path.join(HERE, "outputs", f"{BASE}.json")))
        edges = [tuple(e) for e in d["rlcd_directed"]] + \
                [tuple(sorted(e)) for e in d["rlcd_undirected"]]
        nodes = {x for e in edges for x in e}
        lats = sorted((n for n in nodes if LAT_RE.match(n)), key=lambda s: int(s[1:]))
        g = G.Graph(lats, list(base["graph"].observed), edges)
        g_pub = base["graph"]
        fac_of = {o: F for F in g_pub.latents for o in g_pub.children(F)
                  if not g_pub.is_latent(o)}

        def obs_leaves(gr, node, seen=None):
            seen = seen if seen is not None else set()
            out = []
            for c in gr.children(node):
                if c in seen:
                    continue
                seen.add(c)
                out += obs_leaves(gr, c, seen) if gr.is_latent(c) else [c]
            return out

        gt = {}
        for L in lats:
            votes = [fac_of[o] for o in obs_leaves(g, L) if o in fac_of]
            if votes:
                top = max(sorted(set(votes)), key=votes.count)
                gt[L] = base["latent_gt"].get(top, next(iter(base["latent_gt"].values()), ""))
            elif base["latent_gt"]:
                gt[L] = next(iter(base["latent_gt"].values()))
        return dict(name=f"{BASE}rlcd", graph=g, X=base["X"], labels=base["labels"],
                    latent_gt=gt)

    runner.ALL_LOADERS[f"{BASE}rlcd"] = load_discovered
    runner.main()
