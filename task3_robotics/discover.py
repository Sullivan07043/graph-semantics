"""Two discovery routes on the same step-level data, scored against the same known truth.

Both were suggested in the 2026-08-05 meeting. They differ in what they produce and in what they
can express, and the whole point of running them on a system whose graph we know is that the
difference becomes measurable rather than arguable.

Both routes must be able to express CONTEMPORANEOUS edges. Forward kinematics is instantaneous:
the end-effector pose at t is a function of the joint angles at t, not of anything at t-1. That is
45 of the 113 true edges, and a first version of this script could not reach any of them, because
it only modelled past to current. The fix is to let a current channel have current parents, with
self-edges forbidden and an acyclic order imposed from the physics: joints come before the
end-effector and before the gripper pose.

ROUTE jacobian
    Learn the transition map f(x_{t-1}, a_{t-1}, x_t^{allowed}) -> x_t with an L1 penalty on its
    input Jacobian,
    then read the structure off that Jacobian. Structure and weights come from one object, so no
    separate coefficient fit is needed. The Jacobian is local, so a configuration-dependent edge
    such as the operational-space controller's action-to-joint map is representable.
    The latent-scale problem that defeated the L1 penalty in the autoencoder version does not
    arise here: inputs and outputs are both observed channels with fixed scales, so the network
    cannot shrink the Jacobian by inflating something in between.

ROUTE boss
    BOSS with the BIC-from-covariance score, as instructed. Returns a CPDAG over all columns; only
    edges running from t-1 to t are kept, since a backward edge in time is not a hypothesis worth
    entertaining. Weights are then the least-squares coefficients of each node on its parents,
    which is the maximum-likelihood fit under the same linear-Gaussian model BIC scores. Its output
    is one static graph, so a configuration-dependent edge can only be represented as present
    or absent.

ROUTE recboss
    Same algorithm, the C implementation (causal-learn/upstream/causal-get, J. Ramsey /
    B. Andrews). Runs from the correlation matrix in seconds instead of tens of minutes, so the
    BIC penalty DISCOUNT becomes sweepable. discount=1 equals the causal-learn default; the
    post-filter and the least-squares weights are shared with ROUTE boss. Build the site/ dir
    first (bash causal-learn/upstream/causal-get/build.sh).

Env: NPZ=<step npz> ROUTE=jacobian|boss|recboss|both EPOCHS=400 LAM=0.02 TAU=0.05 DEV=cpu ROWS=0
     DISCOUNT=2 (recboss only)
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "causal-learn", "upstream", "causal-learn"))

NPZ = os.environ.get("NPZ", os.path.join(HERE, "outputs", "lift_body_steps.npz"))
ROUTE = os.environ.get("ROUTE", "both")
EPOCHS = int(os.environ.get("EPOCHS", 400))
LAM = float(os.environ.get("LAM", 0.02))
TAU = float(os.environ.get("TAU", 0.05))
DEV = os.environ.get("DEV", "cpu")
ROWS = int(os.environ.get("ROWS", 0))


# contemporaneous order, from the physics: joint angles and velocities are upstream of the
# end-effector pose and of the gripper pose at the same instant
TIER = {"robot0_joint_pos": 0, "robot0_joint_vel": 0, "robot0_eef_pos": 1,
        "robot0_eef_quat": 1, "robot0_gripper_qpos": 1, "robot0_gripper_qvel": 1}


def tier(col):
    b = col.split("@")[0].rsplit(".", 1)[0]
    return TIER.get(b, 1)


def load():
    d = np.load(NPZ, allow_pickle=True)
    X = np.asarray(d["X"], float)
    cols = [str(c) for c in d["names"]]
    if ROWS and len(X) > ROWS:
        X = X[np.random.default_rng(0).choice(len(X), ROWS, replace=False)]
    past = [i for i, c in enumerate(cols) if c.endswith("@t-1")]
    cur = [i for i, c in enumerate(cols) if c.endswith("@t")]
    return X, cols, past, cur


def allowed_contemporaneous(cols, cur):
    """For each current column, which other current columns may be its parents."""
    return {j: [i for i in cur if i != j and tier(cols[i]) < tier(cols[j])] for j in cur}


def route_jacobian(X, cols, past, cur):
    import torch
    import torch.nn as nn
    from torch.func import jacrev, vmap

    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = torch.tensor((X - mu) / sd, dtype=torch.float32, device=DEV)
    # inputs are the past columns AND the tier-0 current columns, so an instantaneous edge from a
    # joint angle to the end-effector pose is representable. Tier-0 targets keep only past inputs,
    # which forbids self-edges and keeps the contemporaneous part acyclic.
    tier0 = [i for i in cur if tier(cols[i]) == 0]
    src = past + tier0
    P, C = Z[:, src], Z[:, cur]
    net = nn.Sequential(nn.Linear(len(src), 128), nn.GELU(),
                        nn.Linear(128, 128), nn.GELU(),
                        nn.Linear(128, len(cur))).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    n = len(P)
    t0 = time.time()
    for ep in range(EPOCHS):
        opt.zero_grad()
        idx = torch.randint(0, n, (min(4096, n),), device=DEV)
        rec = ((net(P[idx]) - C[idx]) ** 2).mean()
        sub = P[idx[:128]].detach()
        jl1 = vmap(jacrev(net))(sub).abs().mean()
        (rec + LAM * jl1).backward()
        opt.step()
        if ep % 100 == 0 or ep == EPOCHS - 1:
            print(f"  [jacobian] ep{ep} rec={rec.item():.4f} jac_l1={jl1.item():.4f}", flush=True)
    with torch.no_grad():
        pass
    idx = torch.randperm(n, device=DEV)[:512]
    J = vmap(jacrev(net))(P[idx].detach()).abs().mean(0).detach().cpu().numpy()   # [cur, src]
    print(f"  [jacobian] done in {time.time() - t0:.0f}s", flush=True)
    # a column-relative cut, so a strong parent set does not hide a weak one elsewhere
    edges = []
    for ci, cj in enumerate(cur):
        m = J[ci].max()
        for pi, sj in enumerate(src):
            if sj == cj:
                continue                                   # no self-edge
            if tier(cols[cj]) == 0 and sj in tier0:
                continue                                   # tier-0 targets take past inputs only
            if m > 0 and J[ci, pi] >= TAU * m:
                edges.append((cols[sj], cols[cj], float(J[ci, pi])))
    return edges, J, [cols[i] for i in src], [cols[i] for i in cur]


def filter_and_weight(directed, X, cols, past, cur):
    """Shared tail of the BOSS routes: time/tier filter, then least-squares weights.

    directed = iterable of (i, j) index pairs meaning i -> j. Keeps i -> j when it runs across
    time (past to current) or contemporaneously along the physics tier order. Weights are the
    least squares of each child on its retained parents, the ML fit under the same
    linear-Gaussian model BIC scores.
    """
    pastset, curset = set(past), set(cur)
    keep = set()
    for i, j in directed:
        if i == j:
            continue
        if i in pastset and j in curset:
            keep.add((i, j))                                # across time, always allowed
        elif i in curset and j in curset and tier(cols[i]) < tier(cols[j]):
            keep.add((i, j))                                # contemporaneous, physics order
    edges = []
    for j in curset:
        ps = sorted(i for (i, jj) in keep if jj == j)
        if not ps:
            continue
        A_ = np.c_[X[:, ps], np.ones(len(X))]
        beta = np.linalg.lstsq(A_, X[:, j], rcond=None)[0][:-1]
        for k, i in enumerate(ps):
            edges.append((cols[i], cols[j], float(beta[k])))
    return edges


def route_boss(X, cols, past, cur):
    from causallearn.search.PermutationBased.BOSS import boss
    t0 = time.time()
    g = boss(X, score_func="local_score_BIC_from_cov", node_names=cols, verbose=False)
    print(f"  [boss] search done in {time.time() - t0:.0f}s", flush=True)
    A = g.graph
    # causal-learn encodes i -> j as graph[j, i] = 1 and graph[i, j] = -1
    directed = [(i, j) for i in range(len(cols)) for j in range(len(cols))
                if A[j, i] == 1 and A[i, j] == -1]
    return filter_and_weight(directed, X, cols, past, cur), None


def route_recboss(X, cols, past, cur):
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "causal-learn", "upstream", "causal-get", "site"))
    import causalget as cg
    discount = float(os.environ.get("DISCOUNT", 2))
    restarts = int(os.environ.get("RESTARTS", 1))
    t0 = time.time()
    R = np.corrcoef(X, rowvar=False)
    dag = cg.boss(R, n=len(X), discount=discount, restarts=restarts, seed=1)
    print(f"  [recboss] search done in {time.time() - t0:.1f}s "
          f"(discount={discount}, restarts={restarts}, {int(dag.sum())} raw edges)", flush=True)
    directed = [(int(j), int(i)) for i, j in zip(*np.nonzero(dag))]   # dag[i, j] = 1 means j -> i
    return filter_and_weight(directed, X, cols, past, cur), None


def main():
    X, cols, past, cur = load()
    print(f"[discover] {X.shape[0]} transitions, {len(past)} past columns, "
          f"{len(cur)} current columns", flush=True)
    out = {}
    if ROUTE in ("jacobian", "both"):
        print("[discover] route: jacobian", flush=True)
        e, J, src, cur_cols = route_jacobian(X, cols, past, cur)
        out["jacobian"] = [{"from": a, "to": b, "weight": w} for a, b, w in e]
        # save the raw Jacobian so the threshold can be swept without retraining
        np.savez(os.path.join(HERE, "outputs", "lift_body_jacobian.npz"),
                 J=J, src=np.array(src), targets=np.array(cur_cols))
        print(f"[discover] jacobian: {len(e)} edges", flush=True)
    if ROUTE in ("boss", "both"):
        print("[discover] route: boss", flush=True)
        e, _ = route_boss(X, cols, past, cur)
        out["boss"] = [{"from": a, "to": b, "weight": w} for a, b, w in e]
        print(f"[discover] boss: {len(e)} forward-in-time edges", flush=True)
    if ROUTE == "recboss":
        print("[discover] route: recboss", flush=True)
        e, _ = route_recboss(X, cols, past, cur)
        # stored under the "boss" key: same algorithm, and every downstream reader
        # (summarize_graph.py, evaluate_vs_truth.py) selects routes["boss"]
        out["boss"] = [{"from": a, "to": b, "weight": w} for a, b, w in e]
        print(f"[discover] recboss: {len(e)} kept edges", flush=True)
    path = os.environ.get("OUT", os.path.join(HERE, "outputs", "lift_body_discovered.json"))
    json.dump({"source": os.path.basename(NPZ), "tau": TAU, "lam": LAM, "routes": out},
              open(path, "w"), indent=1)
    print(f"[discover] saved {path}", flush=True)


if __name__ == "__main__":
    main()
