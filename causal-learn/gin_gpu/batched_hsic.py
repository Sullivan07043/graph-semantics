"""GPU-batched HSIC gamma test, a faithful port of causal-learn's
`causallearn.search.FCMBased.lingam.hsic.hsic_test_gamma` (mdbs bandwidth path).

Faithfulness contract (certified in certify.py against the official function, float64):
  - bandwidth: median heuristic over the FIRST 100 samples, strict-upper-triangle positive
    distances, numpy-median semantics (average of the two middle values on even counts);
  - RBF gram K = exp(-D / (2 w^2)); double centering Kc = K - (col+row)/n + all/n^2;
  - stat = (1/n) sum(Kc.T * Lc);
  - var  = 72 (n-4)(n-5) / (n(n-1)(n-2)(n-3)) * (1/(n(n-1))) * (sum - trace) of (Kc*Lc/6)^2;
  - mean = (1 + mu_X mu_Y - mu_X - mu_Y)/n with mu from zero-diagonal K, L;
  - p = 1 - gamma.cdf(stat, mean^2/var, scale = var n / mean).

Batched structure: the Y side (data columns) is precomputed ONCE into a ColumnBank
(centered grams flattened), then a batch of X vectors is scored against every column with
two GEMMs: sum(Kc*Lc) = <Kc, Lc>_F and sum((Kc*Lc)^2) = <Kc^2, Lc^2>_F (both symmetric).
All heavy math float64 on the GPU; the gamma tail runs in scipy on CPU.
"""
import numpy as np
import torch
from scipy.stats import gamma


def _widths(Xb: torch.Tensor) -> torch.Tensor:
    """Median-heuristic bandwidth per row of Xb (B, n), numpy-median compatible."""
    m = Xb[:, :100]                                       # first 100 samples, as reference
    d = (m * m).sum(-1, keepdim=False)
    D = m.unsqueeze(2).sub(m.unsqueeze(1)).pow_(2)        # (B, k, k) squared distances
    k = m.shape[1]
    iu = torch.triu_indices(k, k, offset=1)
    vals = D[:, iu[0], iu[1]]                             # (B, k(k-1)/2) strict upper
    pos = vals > 0
    cnt = pos.sum(1)
    if (cnt == 0).any():
        raise ValueError("constant column: median-heuristic bandwidth undefined")
    vals = torch.where(pos, vals, torch.full_like(vals, torch.inf))
    vals, _ = vals.sort(dim=1)
    lo = vals.gather(1, ((cnt - 1) // 2).unsqueeze(1)).squeeze(1)
    hi = vals.gather(1, (cnt // 2).unsqueeze(1)).squeeze(1)
    return torch.sqrt(0.5 * (lo + hi) / 2)


def _grams(Xb: torch.Tensor, w: torch.Tensor):
    """K, Kc, mu, diag(Kc) for each row of Xb (B, n). K has TRUE diagonal (=1); mu uses
    the zero-diagonal sum exactly as the reference."""
    B, n = Xb.shape
    D = Xb.unsqueeze(2).sub(Xb.unsqueeze(1)).pow_(2)      # (B, n, n)
    K = torch.exp(-D / (2.0 * (w ** 2)).view(B, 1, 1))
    rs = K.sum(2)                                         # row sums (B, n)
    cs = K.sum(1)                                         # col sums (== rs, symmetric)
    al = rs.sum(1)                                        # all sum (B,)
    Kc = K - (cs.unsqueeze(1) + rs.unsqueeze(2)) / n + (al / n ** 2).view(B, 1, 1)
    mu = (al - torch.diagonal(K, dim1=1, dim2=2).sum(1)) / (n * (n - 1))
    return K, Kc, mu


class ColumnBank:
    """Precomputed Y-side for a fixed data matrix: one entry per column."""

    def __init__(self, data: np.ndarray, device: str):
        self.device = device
        Y = torch.tensor(np.asarray(data, np.float64).T, device=device)   # (p, n)
        self.n = Y.shape[1]
        w = _widths(Y)
        _, Lc, mu = _grams(Y, w)
        self.Lc_flat = Lc.reshape(Y.shape[0], -1)                          # (p, n^2)
        self.Lc2_flat = (Lc ** 2).reshape(Y.shape[0], -1)
        self.Lc_diag2 = torch.diagonal(Lc, dim1=1, dim2=2) ** 2            # (p, n)
        self.muY = mu

    def score(self, E: np.ndarray, chunk: int = 96):
        """HSIC(stat, p) of every E row against every bank column.
        E: (B, n) float64. Returns stats, pvals as numpy (B, p)."""
        n = self.n
        Et = torch.tensor(np.asarray(E, np.float64), device=self.device)
        stats, means, vars = [], [], []
        for s in range(0, Et.shape[0], chunk):
            Xb = Et[s:s + chunk]
            w = _widths(Xb)
            _, Kc, muX = _grams(Xb, w)
            Kc_flat = Kc.reshape(Xb.shape[0], -1)
            st = (Kc_flat @ self.Lc_flat.T) / n                            # (b, p)
            s2 = (Kc_flat ** 2) @ self.Lc2_flat.T
            tr = (torch.diagonal(Kc, dim1=1, dim2=2) ** 2) @ self.Lc_diag2.T
            v = (s2 - tr) / 36.0
            v = v / n / (n - 1)
            v = 72.0 * (n - 4) * (n - 5) / n / (n - 1) / (n - 2) / (n - 3) * v
            mn = (1.0 + muX.unsqueeze(1) * self.muY.unsqueeze(0)
                  - muX.unsqueeze(1) - self.muY.unsqueeze(0)) / n
            stats.append(st.cpu().numpy())
            means.append(mn.cpu().numpy())
            vars.append(v.cpu().numpy())
        stat = np.concatenate(stats)
        mean = np.concatenate(means)
        var = np.concatenate(vars)
        al = mean ** 2 / var
        beta = var * n / mean
        p = 1.0 - gamma.cdf(stat, al, scale=beta)
        return stat, p
