"""Run CauScale on an episode-level matrix and return the observed-to-observed adjacency.

CauScale is amortized: one forward pass of a pretrained network gives a probabilistic adjacency,
which is why it reaches hundreds of nodes where search-based methods stop. It assumes causal
sufficiency and models no latents, so it is used ONLY for the observed-to-observed layer here. The
latent layers come from the model Jacobian, which is computed rather than discovered.

Declared limitation: the released checkpoint is trained on synthetic data. Robot episode statistics
are out of that distribution, and the paper's own out-of-distribution number is 84.4% mAP against
99.6% in-distribution. Treat the output as a proposal to be checked, not as ground truth.

Its inference path expects a manifest CSV whose rows point at three .npy files: the observations,
the true adjacency, and the per-sample intervention indicator. We have no true adjacency, so a zero
matrix is passed. Our data is purely observational, so the intervention indicator is zero as well.

We do NOT call their src/inference.py. That script scores the prediction against the true graph,
and `causcale.py:155` calls exit() the moment a metric is NaN. With an all-zero placeholder graph
AUROC is NaN by construction, so the process ends with status 0 and writes nothing, after the
prediction has already been computed. Calling the encoder directly skips scoring entirely, which
is what we want: we have no ground truth to score against.

Runs in the causcale conda env (python 3.10, torch 2.1), never in the pipeline venv.

Env: NPZ=<episode npz> OUTDIR=<workdir> GPU=1 SAMPLE_SIZE=1000
"""
import csv
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAUSCALE = "/data2/shuhao/semantic_interpretation/CauScale"
PY = "/data2/shuhao/miniforge3/envs/causcale/bin/python"
CKPT = os.path.join(CAUSCALE, "checkpoints", "synthetic", "auprc=0.905_migrated.ckpt")

NPZ = os.environ.get("NPZ", os.path.join(HERE, "outputs", "lift_mg_episode.npz"))
OUTDIR = os.environ.get("OUTDIR", os.path.join(HERE, "outputs", "causcale"))
GPU = os.environ.get("GPU", "1")
SAMPLE_SIZE = os.environ.get("SAMPLE_SIZE", "1000")


def prepare(npz, outdir):
    d = np.load(npz, allow_pickle=True)
    X = np.asarray(d["X"], np.float32)
    names = [str(x) for x in d["names"]]
    p = X.shape[1]
    os.makedirs(outdir, exist_ok=True)
    fp_data = os.path.join(outdir, "data.npy")
    fp_graph = os.path.join(outdir, "graph_placeholder.npy")
    fp_regime = os.path.join(outdir, "regime.npy")
    np.save(fp_data, X)
    np.save(fp_graph, np.zeros((p, p), np.int64))          # no ground truth; metrics are ignored
    np.save(fp_regime, np.zeros((X.shape[0], p), np.int64))  # purely observational
    manifest = os.path.join(outdir, "manifest.csv")
    with open(manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "fp_data", "fp_graph", "fp_regime"])
        w.writeheader()
        w.writerow({"split": "test", "fp_data": fp_data, "fp_graph": fp_graph,
                    "fp_regime": fp_regime})
    print(f"[causcale] {X.shape[0]} episodes x {p} variables -> {manifest}", flush=True)
    return manifest, names, p


def main():
    if not os.path.exists(CKPT):
        sys.exit(f"missing checkpoint {CKPT}")
    manifest, names, p = prepare(NPZ, OUTDIR)

    os.chdir(CAUSCALE)
    sys.path.insert(0, os.path.join(CAUSCALE, "src"))
    sys.argv = ["inference.py", "--config_file", "config/inference.yaml", "--gpu", GPU,
                "--data_file", manifest, "--sample_size", SAMPLE_SIZE, "--batch_size", "1",
                "--results_prefix", os.path.join(OUTDIR, "results"),
                "--checkpoint_path", CKPT]
    import torch
    from args import parse_args
    from data import InferenceDataModule
    from model import CauScale

    args = parse_args()
    dm = InferenceDataModule(args)
    model = CauScale(args)
    state = torch.load(CKPT, map_location="cpu")
    model.load_state_dict(state["state_dict"] if "state_dict" in state else state)
    dev = f"cuda:{GPU}" if int(GPU) >= 0 and torch.cuda.is_available() else "cpu"
    model.to(dev).eval()

    batch = next(iter(dm.predict_dataloader()))
    for k, v in batch.items():
        if torch.is_tensor(v):
            batch[k] = v.to(dev)
    # the encoder returns pair embeddings; symmetrize() runs the top layer that turns each
    # upper-triangular pair into three logits: no edge, i->j, j->i. Its second return value is
    # the label, which is meaningless here and is discarded.
    with torch.no_grad():
        out = model.encoder(batch)
        pair_logits, _ = model.symmetrize(out, batch, reduce=False)
        P = torch.softmax(pair_logits[0], dim=-1).cpu().numpy()      # (n_pairs*2, 3)

    # one row per upper-triangular pair; the three columns are no edge, i->j, j->i
    n = len(names)
    iu = np.triu_indices(n, 1)
    assert len(P) == len(iu[0]), (len(P), len(iu[0]))
    prob = np.zeros((n, n, 3))
    prob[iu[0], iu[1]] = P
    edge = np.zeros((n, n))
    edge[iu[0], iu[1]] = 1.0 - P[:, 0]
    edge = edge + edge.T
    np.savez(os.path.join(OUTDIR, "obs_obs_adjacency.npz"),
             prob=prob, edge=edge, names=np.array(names), source=os.path.basename(NPZ))
    print(f"[causcale] adjacency {prob.shape} saved to {OUTDIR}/obs_obs_adjacency.npz")
    e = edge[iu]
    print(f"[causcale] pair edge probability: mean {e.mean():.3f}, max {e.max():.3f}, "
          f"min {e.min():.3f}; above 0.5: {int((e > 0.5).sum())} of {len(e)}; "
          f"above 0.9: {int((e > 0.9).sum())}")


if __name__ == "__main__":
    main()
