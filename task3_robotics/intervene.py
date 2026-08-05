"""Define each latent's meaning by intervening on it, and write the Task 2 reference.

Robot latents have no published names. The questionnaire testbeds get their Task 2 reference from
an instrument's own scoring key; there is no such thing here. What robotics has instead, and what
questionnaires never have, is the ability to actually intervene: set one latent, decode, and read
which named variables move and in which direction. That signed response over named observed
variables is this project's own definition of meaning, so it is the reference used here.

DECLARED RELAXATION, and it is a real one. The reference is machine-constructed from the same model
whose Jacobian also produced the graph, so Task 2 here asks whether the pipeline can put the
structure's implication into language. The questionnaire Task 2 asks something stronger: whether it
recovers a name that a human field settled on independently. These two numbers are not the same
kind of evidence and must not be averaged or compared directly.

Two mitigations against the reference being trivially echoed:
  - the response is measured by finite difference through the NONLINEAR decoder, not read off the
    mean Jacobian the graph was thresholded from, so magnitudes and ordering can differ;
  - the reference sentence names the top responders by their natural-language labels, which the
    pipeline must reach through the graph rather than by copying a name it was handed.

Also reports, per latent, the response norm. A latent whose intervention moves nothing is
meaningless regardless of what any metric says, and that check needs no judge.

Env: NPZ=<episode npz> K=16 AELAM=0.05 AEEPOCHS=800 DELTA=2.0 TOPK=4 OUT=<json>
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/data2/shuhao/semantic_interpretation/archive_phases/latent_concept/scripts")

NPZ = os.environ.get("NPZ", os.path.join(HERE, "outputs", "lift_mg_episode.npz"))
OUT = os.environ.get("OUT", os.path.join(HERE, "outputs", "lift_mg_latent_gt.json"))
K = int(os.environ.get("K", 16))
DELTA = float(os.environ.get("DELTA", 2.0))
TOPK = int(os.environ.get("TOPK", 4))
DEV = os.environ.get("DEV", "cpu")


def train_ae(X, k, seed=0):
    """Same recipe and seed as build_structure.py, so the latents are the same ones."""
    import causal_ae
    from torch.func import jacrev, vmap
    lam = float(os.environ.get("AELAM", 0.05))
    epochs = int(os.environ.get("AEEPOCHS", 800))
    torch.manual_seed(seed)
    Xn = causal_ae.z(X)
    Xt = torch.tensor(Xn, dtype=torch.float32, device=DEV)
    n, m = Xt.shape
    net = causal_ae._AE(m, k, 64).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    for ep in range(epochs):
        opt.zero_grad()
        xhat, zc = net(Xt)
        rec = ((xhat - Xt) ** 2).mean()
        idx = torch.randint(0, n, (min(256, n),), device=DEV)
        jl1 = vmap(jacrev(net.dec))(zc[idx].detach()).abs().mean()
        (rec + lam * jl1).backward()
        opt.step()
    with torch.no_grad():
        Z = net.enc(Xt)
    return net, Z


def intervene(net, Z, delta):
    """Finite-difference response: hold every episode's latents, push one latent by +/- delta of
    its own spread, decode both, and average the difference. Shape (k, m)."""
    k = Z.shape[1]
    sd = Z.std(0, keepdim=True)
    resp = []
    with torch.no_grad():
        for j in range(k):
            Zp, Zm = Z.clone(), Z.clone()
            Zp[:, j] += delta * sd[0, j]
            Zm[:, j] -= delta * sd[0, j]
            resp.append((net.dec(Zp) - net.dec(Zm)).mean(0) / (2 * delta * sd[0, j]))
    return torch.stack(resp).cpu().numpy()


def sentence(row, labels, topk):
    order = np.argsort(-np.abs(row))[:topk]
    up = [labels[i] for i in order if row[i] > 0]
    dn = [labels[i] for i in order if row[i] < 0]
    parts = []
    if up:
        parts.append("raises " + "; ".join(up))
    if dn:
        parts.append("lowers " + "; ".join(dn))
    return "an episode-level factor that " + ", and ".join(parts)


def main():
    d = np.load(NPZ, allow_pickle=True)
    X = np.asarray(d["X"], float)
    names = [str(x) for x in d["names"]]
    labels = [str(x) for x in d["labels"]]
    print(f"[intervene] {X.shape[0]} episodes x {X.shape[1]} variables, K={K}", flush=True)

    net, Z = train_ae(X, K)
    R = intervene(net, Z, DELTA)
    norms = np.linalg.norm(R, axis=1)
    live = [j for j in range(len(R)) if norms[j] > 1e-6 * norms.max()]
    print(f"[intervene] response norms: max {norms.max():.3f}, min {norms.min():.3f}; "
          f"latents that move something: {len(live)}/{len(R)}", flush=True)

    gt, sig = {}, {}
    for j in live:
        L = f"L{j}"
        gt[L] = sentence(R[j], labels, TOPK)
        sig[L] = {names[i]: float(R[j, i]) for i in np.argsort(-np.abs(R[j]))[:10]}
    json.dump({"latent_gt": gt, "signature": sig,
               "params": {"K": K, "delta": DELTA, "topk": TOPK,
                          "source": os.path.basename(NPZ),
                          "reference": "intervention response through the decoder; machine "
                                       "constructed, not a published construct name"}},
              open(OUT, "w"), indent=1)
    print(f"[intervene] wrote {len(gt)} latent references to {OUT}\n")
    for L in list(gt)[:6]:
        print(f"  {L}: {gt[L][:150]}")


if __name__ == "__main__":
    main()
