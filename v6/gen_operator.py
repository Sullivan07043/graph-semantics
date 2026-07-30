"""Jacobian-locked nonlinear generation operator T_theta (THEORY §4.2) — THE v6 generation path.

g_c(e_pa(c)) = sum_{p in pa(c)} T_theta(e_p ; s_pc, |w_pc|, tau_pc)

- Additive per-parent routing => Jacobian SPARSITY lock by construction (a child's contribution
  from non-parents is identically zero; no penalty needed).
- One shared transform T_theta conditioned on edge sign s (+-1), magnitude |w|, and edge type
  tau (1 iff latent->latent).
- T(e_p; cond) = |w| * (base(e_p, s) + Delta_theta([e_p, cond])), where base = e_p on positive
  edges and f_neg(e_p) on negative edges (f_neg frozen INSIDE the operator: the sign action is
  intrinsic, THEORY §4.1 arg 2).
- Zero-init identity discipline: Delta head zero-initialized, so the untrained operator is
  EXACTLY the v5 generation term (w * e_p positive / |w| * f_neg(e_p) negative). v5 = epoch 0.
- Sign lock is NOT by construction (MLP): sign_audit() is the audit penalty, part of the
  TRUNK-4 training objective and reported, never silently assumed (THEORY §4.2).

The operator's forward returns the FULL per-edge contribution (|w| scaling inside) — callers
sum contributions per child, nothing else.
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.environ.get("GENOP_CKPT", os.path.join(HERE, "outputs", "gen_operator.pt"))
COND = 3   # [sign(+-1), |w|, 1{latent->latent}]


class GenOperator(nn.Module):
    def __init__(self, d, neg_op, hid=256):
        super().__init__()
        self.d = d
        self.neg = neg_op
        for p in self.neg.parameters():
            p.requires_grad_(False)
        self.delta = nn.Sequential(nn.Linear(d + COND, hid), nn.GELU(), nn.Linear(hid, d))
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)

    def _base(self, E_par, cond):
        base = E_par
        nidx = (cond[:, 0] < 0).nonzero(as_tuple=True)[0]
        if len(nidx):
            base = E_par.clone()
            base[nidx] = self.neg(E_par[nidx])
        return base

    def forward(self, E_par, cond):
        """E_par [m, d] parent embedding per edge; cond [m, COND]. -> [m, d] contribution."""
        absw = cond[:, 1:2]
        return absw * (self._base(E_par, cond) + self.delta(torch.cat([E_par, cond], 1)))

    def sign_audit(self, E_par, cond):
        """Audit penalty (THEORY §4.2): <T(e;+,.), e_p> >= 0 on positive edges and
        <T(e;-,.), f_neg(e_p)> >= 0 on negative edges; hinge on the cosine."""
        out = self.forward(E_par, cond)
        ref = self._base(E_par, cond)
        cs = F.cosine_similarity(out, ref, dim=1)
        return torch.relu(-cs).pow(2).mean()


def edge_table(g, W, device="cpu"):
    """Static per-edge tables for the batched operator forward.
    -> (edge_parents [list of parent names, one per edge], cond [m, COND] tensor,
        child_rows {child -> LongTensor of edge-row indices})."""
    gen_nodes = [n for n in g.nodes if g.parents(n)]
    lat = set(g.latents)
    par, cond, child_rows = [], [], {}
    for n in gen_nodes:
        for p in g.parents(n):
            w = float(W.get((p, n), 0.0))
            child_rows.setdefault(n, []).append(len(par))
            par.append(p)
            cond.append([1.0 if w >= 0 else -1.0, abs(w), 1.0 if (p in lat and n in lat) else 0.0])
    cond_t = torch.tensor(cond, dtype=torch.float32, device=device)
    child_rows = {n: torch.tensor(r, dtype=torch.long, device=device)
                  for n, r in child_rows.items()}
    return par, cond_t, child_rows


def save(op, path=CKPT):
    torch.save({"d": op.d, "hid": op.delta[0].out_features, "state": op.delta.state_dict()}, path)


def load_or_init(d=1024, device="cpu", path=CKPT):
    """Trained checkpoint if present, else zero-init (== exact v5 linear+f_neg behavior).
    f_neg always comes frozen from negop.CKPT."""
    import negop
    fneg = negop.load()
    if os.path.exists(path):
        ck = torch.load(path, map_location="cpu")
        op = GenOperator(ck["d"], fneg, hid=ck["hid"])
        op.delta.load_state_dict(ck["state"])
    else:
        op = GenOperator(d, fneg)
    op.to(device).eval()
    return op
