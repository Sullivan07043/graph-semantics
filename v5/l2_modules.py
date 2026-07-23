"""L2 weight modules: map node context -> per-node constraint-weight multipliers.

Both start EXACTLY at the frozen objective (all multipliers 1.0):
  StaticWeights — 5 learned scalars (one per term), the linear/static attribution control.
  WeightNet     — node-context MLP (nonlinear, the main arm); zero-init head.
Multipliers are bounded to [e^-1.5, e^1.5] ~ [0.22, 4.5] so the learned objective cannot silently
delete or explode a constraint.
"""
import numpy as np
import torch

TERMS = ["gen", "resnorm", "anchor", "node", "norm"]
BOUND = 1.5


def node_features(g, W, labeled_set):
    """[N, 12] float32 features for every node in g.nodes order. Graph + data only (no labels)."""
    children = {}
    for (p, n) in W:
        children.setdefault(p, []).append(n)
    lat = set(g.latents)
    N = len(g.nodes)
    F = np.zeros((N, 12), np.float32)
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
    return F


def _slice_nw(mult, ctx):
    """mult: [N, 5] multipliers aligned with ctx['all_nodes'] -> the nw dict step_loss expects."""
    ni = ctx["node_idx"]
    gi = torch.tensor([ni[n] for n in ctx["gen_nodes"]], dtype=torch.long, device=mult.device)
    fi = torch.tensor([ni[n] for n in ctx["free"]], dtype=torch.long, device=mult.device)
    nw = {"gen": mult[gi, 0], "resnorm": mult[gi, 1], "node": mult[:, 3], "norm": mult[fi, 4]}
    if ctx["pc_nodes"]:
        ai = torch.tensor([ni[n] for n in ctx["pc_nodes"]], dtype=torch.long, device=mult.device)
        nw["anchor"] = mult[ai, 2]
    else:
        nw["anchor"] = None
    return nw


class StaticWeights(torch.nn.Module):
    """One learned log-multiplier per term, same for every node. exp(0)=1 at init."""

    def __init__(self):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.zeros(5))

    def forward(self, feats, ctx):
        m = torch.exp(BOUND * torch.tanh(self.theta))          # [5]
        N = len(ctx["all_nodes"])
        mult = m[None, :].expand(N, 5)
        return _slice_nw(mult, ctx)


class WeightNet(torch.nn.Module):
    """Node-context MLP -> per-node, per-term multipliers. Zero-init head => exactly 1.0 at init."""

    def __init__(self, fdim=12, hid=64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(fdim, hid), torch.nn.Tanh(),
            torch.nn.Linear(hid, hid), torch.nn.Tanh(),
            torch.nn.Linear(hid, 5))
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

    def forward(self, feats, ctx):
        raw = self.net(feats)                                   # [N, 5]
        mult = torch.exp(BOUND * torch.tanh(raw))
        return _slice_nw(mult, ctx)


def save(module, path, kind):
    torch.save({"kind": kind, "state": module.state_dict()}, path)


def load(path, device="cpu"):
    ck = torch.load(path, map_location=device)
    mod = StaticWeights() if ck["kind"] == "static" else WeightNet()
    mod.load_state_dict(ck["state"])
    mod.to(device).eval()
    return mod
