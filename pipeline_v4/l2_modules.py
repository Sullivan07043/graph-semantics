"""L2 modules mapping node context to per-node constraint multipliers.

Both arms start exactly at the structured objective (all multipliers are 1.0):
``StaticWeights`` has one learned scalar for each of six terms, while ``WeightNet`` uses
22 graph/data features.  Multipliers remain bounded to ``[exp(-1.5), exp(1.5)]``.
"""
import numpy as np
import torch

TERMS = ["gen", "resnorm", "anchor", "node", "norm", "item"]
BOUND = 1.5
FEATURE_DIM = 22
CHECKPOINT_VERSION = 4
CHECKPOINT_FORMAT = "graph-semantics-weightnet"


def node_features(g, W, labeled_set, item_info=None, independent_info=None):
    """Return ``[N, 22]`` graph/data features without masked-label semantics.

    Item diagnostics include candidate count, max/mean marginal correlation, local visibility,
    confidence, target entropy, and effective candidate count.  The final three columns describe
    each node's retained-zero/conflict degree and mean conflict strength.
    """
    children = {}
    for (p, n) in W:
        children.setdefault(p, []).append(n)
    lat = set(g.latents)
    N = len(g.nodes)
    F = np.zeros((N, FEATURE_DIM), np.float32)
    for i, n in enumerate(g.nodes):
        ps = list(g.parents(n))
        ch = children.get(n, [])
        pw = [float(W.get((p, n), 0.0)) for p in ps]
        cw = [float(W.get((n, c), 0.0)) for c in ch]
        F[i, 0] = 1.0 if n in lat else 0.0
        F[i, 1] = 0.0 if n in labeled_set else 1.0
        F[i, 2] = np.log1p(len(ps))
        F[i, 3] = np.log1p(len(ch))
        F[i, 4] = np.mean([w < 0 for w in pw]) if pw else 0.0
        F[i, 5] = np.mean([w < 0 for w in cw]) if cw else 0.0
        F[i, 6] = np.mean([abs(w) for w in pw]) if pw else 0.0
        F[i, 7] = np.mean([abs(w) for w in cw]) if cw else 0.0
        F[i, 8] = 1.0 if any(p in lat for p in ps) else 0.0
        F[i, 9] = np.log1p(len(g.observed_descendants(n))) if n in lat else 0.0
        F[i, 10] = np.log1p(N) / 6.0
        F[i, 11] = np.mean([p in labeled_set for p in ps]) if ps else 0.0
    if item_info is not None:
        item_features = np.asarray(item_info.get("features"), dtype=np.float32)
        if item_features.shape != (N, 7):
            raise ValueError(f"item_info features must have shape {(N, 7)}, got "
                             f"{item_features.shape}")
        F[:, 12] = np.log1p(item_features[:, 0])
        F[:, 13] = item_features[:, 1]
        F[:, 14] = item_features[:, 2]
        F[:, 15] = np.log1p(item_features[:, 3])
        F[:, 16] = item_features[:, 4]
        F[:, 17] = item_features[:, 5]
        F[:, 18] = np.log1p(item_features[:, 6])
    if independent_info is not None:
        ni = {n: i for i, n in enumerate(g.nodes)}
        retained_degree = np.zeros(N, dtype=np.float32)
        conflict_degree = np.zeros(N, dtype=np.float32)
        conflict_strength = [[] for _ in range(N)]
        for a, b in independent_info.get("pairs", []):
            if a in ni and b in ni:
                retained_degree[ni[a]] += 1
                retained_degree[ni[b]] += 1
        for a, b, rho in independent_info.get("conflicts", []):
            if a in ni and b in ni:
                conflict_degree[ni[a]] += 1
                conflict_degree[ni[b]] += 1
                conflict_strength[ni[a]].append(abs(float(rho)))
                conflict_strength[ni[b]].append(abs(float(rho)))
        F[:, 19] = np.log1p(retained_degree)
        F[:, 20] = np.log1p(conflict_degree)
        F[:, 21] = np.array([np.mean(v) if v else 0.0 for v in conflict_strength],
                            dtype=np.float32)
    return F


def _slice_nw(mult, ctx):
    """Slice ``[N, 6]`` node multipliers into the arrays consumed by ``step_loss``."""
    ni = ctx["node_idx"]
    gi = torch.tensor([ni[n] for n in ctx["gen_nodes"]], dtype=torch.long, device=mult.device)
    fi = torch.tensor([ni[n] for n in ctx["free"]], dtype=torch.long, device=mult.device)
    nw = {"gen": mult[gi, 0], "resnorm": mult[gi, 1],
          "node": mult[:, 3], "norm": mult[fi, 4],
          "item": mult[ctx["item_idx"], 5]}
    if ctx["pc_nodes"]:
        ai = torch.tensor([ni[n] for n in ctx["pc_nodes"]], dtype=torch.long, device=mult.device)
        nw["anchor"] = mult[ai, 2]
    else:
        nw["anchor"] = None
    return nw


class StaticWeights(torch.nn.Module):
    """One learned log-multiplier per term, identical for every node."""

    def __init__(self):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.zeros(len(TERMS)))

    def multipliers(self, feats):
        m = torch.exp(BOUND * torch.tanh(self.theta))
        return m[None, :].expand(feats.shape[0], len(TERMS))

    def forward(self, feats, ctx):
        mult = self.multipliers(feats)
        if mult.shape[0] != len(ctx["all_nodes"]):
            raise ValueError(f"feature/node mismatch: {mult.shape[0]} != "
                             f"{len(ctx['all_nodes'])}")
        return _slice_nw(mult, ctx)


class WeightNet(torch.nn.Module):
    """Node-context MLP with a zero head, hence exactly unit multipliers at initialization."""

    def __init__(self, fdim=FEATURE_DIM, hid=64):
        super().__init__()
        self.fdim = int(fdim)
        self.hid = int(hid)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(fdim, hid), torch.nn.Tanh(),
            torch.nn.Linear(hid, hid), torch.nn.Tanh(),
            torch.nn.Linear(hid, len(TERMS)))
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

    def multipliers(self, feats):
        raw = self.net(feats)
        return torch.exp(BOUND * torch.tanh(raw))

    def forward(self, feats, ctx):
        return _slice_nw(self.multipliers(feats), ctx)


def save(module, path, kind, metadata=None):
    torch.save({"format": CHECKPOINT_FORMAT, "version": CHECKPOINT_VERSION, "kind": kind,
                "feature_dim": FEATURE_DIM, "terms": list(TERMS),
                "state": module.state_dict(), "metadata": dict(metadata or {})}, path)


def load(path, device="cpu", expected_l3_sha256=None):
    ck = torch.load(path, map_location=device)
    if not isinstance(ck, dict) or ck.get("format") != CHECKPOINT_FORMAT \
            or ck.get("version") != CHECKPOINT_VERSION:
        got = ck.get("version", "legacy/unversioned") if isinstance(ck, dict) else type(ck).__name__
        raise RuntimeError(
            f"Incompatible L2 checkpoint {path!r}: got {got}, expected format "
            f"{CHECKPOINT_FORMAT!r} version {CHECKPOINT_VERSION}. The two-pole/unit-item-anchor "
            "objective is not compatible with the old checkpoint even though the "
            "22-feature/6-term tensor schema is unchanged; retrain pipeline_v4/l2_train.py.")
    if ck.get("feature_dim") != FEATURE_DIM or ck.get("terms") != TERMS:
        raise RuntimeError(
            f"Incompatible L2 checkpoint schema in {path!r}: expected feature_dim={FEATURE_DIM} "
            f"and terms={TERMS}, got feature_dim={ck.get('feature_dim')} terms={ck.get('terms')}.")
    kind = ck.get("kind")
    if kind not in ("static", "mlp"):
        raise RuntimeError(f"Unknown L2 checkpoint kind {kind!r} in {path!r}")
    if expected_l3_sha256 is not None:
        actual = ck.get("metadata", {}).get("l3_checkpoint_sha256")
        if actual != expected_l3_sha256:
            raise RuntimeError(
                "WeightNet was not trained with the requested final L3 checkpoint: "
                f"checkpoint metadata has {actual!r}, expected {expected_l3_sha256!r}. "
                "Retrain pipeline_v4/l2_train.py after L3 training.")
    mod = StaticWeights() if kind == "static" else WeightNet()
    try:
        mod.load_state_dict(ck["state"])
    except (KeyError, RuntimeError) as exc:
        raise RuntimeError(f"L2 checkpoint state is incompatible with version "
                           f"{CHECKPOINT_VERSION}: {exc}") from exc
    mod.to(device).eval()
    return mod
