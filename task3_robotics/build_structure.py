"""Assemble the three-layer structure for an episode-level robot dataset.

The point of Task 3 is that this structure is largely COMPUTED, not discovered. In the questionnaire
setting the latent-to-observed structure has to be inferred from rank constraints, and that is the
step that stops working above 25 variables. Here a differentiable model exists, so its Jacobian
gives the same structure directly.

  latent -> observed   decoder Jacobian of the sparse autoencoder (archive causal_ae, the
                       Thought-Communication estimator: reconstruction plus L1 on the decoder
                       Jacobian, whose zero pattern IS the d-separation pattern). COMPUTED.
  observed -> observed CauScale, run separately by run_causcale.py. DISCOVERED, and the only layer
                       where a discovery method is needed at all.
  latent -> latent     graphical lasso on the recovered latent scores, the post-hoc fix recorded in
                       semantic/02_method: causal_ae's model class is single-layer, so hierarchy
                       leaks into correlated scores instead of appearing as edges.

Output is a graph JSON in the same shape discovery/ already writes, so run_downstream.py can take
it without changes.

Env: NPZ=<episode npz> K=16 AELAM=0.05 AEEPOCHS=800 CAUSCALE_NPZ=<obs-obs npz> EDGE_T=0.9
     GLASSO_A=0.1 OUT=<json>
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ARCHIVE = "/data2/shuhao/semantic_interpretation/archive_phases/latent_concept/scripts"
sys.path.insert(0, ARCHIVE)

NPZ = os.environ.get("NPZ", os.path.join(HERE, "outputs", "lift_mg_episode.npz"))
CAUSCALE_NPZ = os.environ.get("CAUSCALE_NPZ",
                              os.path.join(HERE, "outputs", "causcale", "obs_obs_adjacency.npz"))
OUT = os.environ.get("OUT", os.path.join(HERE, "outputs", "lift_mg_structure.json"))
K = int(os.environ.get("K", 16))
EDGE_T = float(os.environ.get("EDGE_T", 0.9))
GLASSO_A = float(os.environ.get("GLASSO_A", 0.1))


def latent_latent(Z, alpha):
    """Zeros in the precision matrix are latent-latent conditional independences."""
    from sklearn.covariance import GraphicalLasso
    try:
        g = GraphicalLasso(alpha=alpha, max_iter=200).fit(Z)
        P = g.precision_
    except Exception as e:
        print(f"[latent-latent] glasso failed ({type(e).__name__}), falling back to "
              f"thresholded partial correlation", flush=True)
        C = np.corrcoef(Z, rowvar=False)
        P = np.linalg.pinv(C)
    d = np.sqrt(np.abs(np.diag(P)))
    R = -P / np.outer(d, d)                 # partial correlation
    np.fill_diagonal(R, 0.0)
    return R


def main():
    d = np.load(NPZ, allow_pickle=True)
    X = np.asarray(d["X"], float)
    names = [str(x) for x in d["names"]]
    labels = [str(x) for x in d["labels"]]

    import causal_ae
    print(f"[ae] {X.shape[0]} episodes x {X.shape[1]} variables, K={K}", flush=True)
    Z, B, keep = causal_ae.estimate(X, K)
    k = Z.shape[1]
    lat = [f"L{j}" for j in range(k)]
    print(f"[ae] {k} surviving latents of {K}; children per latent: "
          f"{[int((B[:, j] != 0).sum()) for j in range(k)]}", flush=True)

    edges, signs = [], {}
    for j in range(k):
        for i in range(len(names)):
            if B[i, j] != 0:
                edges.append([lat[j], names[i]])
                signs[f"{lat[j]}->{names[i]}"] = float(B[i, j])
    orphans = [names[i] for i in range(len(names)) if not np.any(B[i] != 0)]
    print(f"[ae] latent->observed edges: {len(edges)} | observed with no latent parent: "
          f"{len(orphans)}", flush=True)

    R = latent_latent(Z, GLASSO_A)
    ll_edges = []
    thr = 0.1
    for a in range(k):
        for b in range(a + 1, k):
            if abs(R[a, b]) > thr:
                ll_edges.append([lat[a], lat[b]])
                signs[f"{lat[a]}--{lat[b]}"] = float(R[a, b])
    print(f"[latent-latent] {len(ll_edges)} partial correlations above {thr}", flush=True)

    oo_edges = []
    if os.path.exists(CAUSCALE_NPZ):
        c = np.load(CAUSCALE_NPZ, allow_pickle=True)
        cn = [str(x) for x in c["names"]]
        assert cn == names, "CauScale ran on a different variable set"
        E = c["edge"]
        iu = np.triu_indices(len(names), 1)
        for i, j in zip(*iu):
            if E[i, j] > EDGE_T:
                oo_edges.append([names[i], names[j]])
        print(f"[obs-obs] {len(oo_edges)} pairs above edge probability {EDGE_T}", flush=True)
    else:
        print(f"[obs-obs] {CAUSCALE_NPZ} missing; layer omitted", flush=True)

    out = {
        "dataset": f"robot_{str(d['task'])}",
        "rlcd_directed": edges,
        "rlcd_undirected": ll_edges + oo_edges,
        "params": {"source": os.path.basename(NPZ), "n_episodes": int(d["n_episodes"]),
                   "K_requested": K, "K_surviving": k, "edge_threshold": EDGE_T,
                   "glasso_alpha": GLASSO_A,
                   "latent_observed": "sparse-Jacobian autoencoder (computed, not discovered)",
                   "observed_observed": "CauScale amortized inference",
                   "latent_latent": "graphical lasso on recovered latent scores",
                   "orphan_observed": orphans},
        "signs": signs,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    np.savez(os.path.join(os.path.dirname(OUT), "lift_mg_latents.npz"),
             Z=Z, B=B, latents=np.array(lat), names=np.array(names), labels=np.array(labels))
    print(f"[structure] saved {OUT} and lift_mg_latents.npz", flush=True)

    print("\n[structure] strongest children of each latent (sign, variable):")
    for j in range(k):
        idx = np.argsort(-np.abs(B[:, j]))[:4]
        idx = [i for i in idx if B[i, j] != 0]
        desc = ", ".join(f"{'+' if B[i, j] > 0 else '-'}{names[i]}" for i in idx)
        print(f"   {lat[j]:4s} {desc}")


if __name__ == "__main__":
    main()
