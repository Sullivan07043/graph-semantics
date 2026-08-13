"""GPU-batched canonical-correlation rank test, drop-in for RLCD's Chi2RankTest.

Same test, two speeds. The statistic and the decision rule are copied verbatim from
Chi2RankTest.test (Wilks chi-square on canonical correlations), so a single .test() call is
exchangeable with the CPU implementation. What changes is where the canonical correlations come
from: they are computed from the correlation matrix alone (no per-sample work), in float64, on
the GPU, and .prime() computes them for many tests in one batched call. RLCD's cluster sweep
primes each chunk of surviving subsets before testing them, which turns the sweep's serial
ms-per-test into a microsecond amortized cost.

float64 is a requirement, not a preference: the recboss episode measured a ~1e-3 decision noise
floor for float32 scoring, and rank decisions at questionnaire sample sizes sit below it.

Certification: certify_gpu_ranktest.py requires decision-level agreement with Chi2RankTest on
real data before this class may be used (same discipline as polychoric.certify).
"""
import numpy as np
from math import log
from scipy.stats import chi2

_EPS_EIG = 1e-12          # relative eigenvalue clip for whitening, mirrors CanCorr tolerance


class GpuRankTest(object):
    """Duck-types Chi2RankTest: __init__(data, N_scaling), .test(pcols, qcols, r, alpha).

    Extra API: .prime(requests) batch-computes canonical correlations for
    requests = iterable of (pcols, qcols) and fills the cache; .clear() empties the cache
    (call between chunks to bound memory).
    """

    def __init__(self, data, N_scaling=1, device=None, max_batch=2048):
        import torch
        self.torch = torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.N = data.shape[0]
        self.N_scaling = N_scaling
        self.max_batch = max_batch

        X = np.asarray(data, np.float64)
        X = X - X.mean(axis=0)
        sd = X.std(axis=0)
        sd[sd == 0] = 1.0
        X = X / sd
        R = (X.T @ X) / self.N
        self.R = torch.tensor(R, dtype=torch.float64, device=device)
        self._cache = {}

    @staticmethod
    def _key(pcols, qcols):
        return (tuple(sorted(pcols)), tuple(sorted(qcols)))

    def clear(self):
        self._cache = {}

    def _whiten_inv_sqrt(self, M):
        """Batched M^{-1/2} via eigh with relative clipping. M: (B, d, d)."""
        torch = self.torch
        w, V = torch.linalg.eigh(M)
        wmax = w[:, -1:].clamp(min=_EPS_EIG)
        good = w > _EPS_EIG * wmax
        inv = torch.where(good, w.clamp(min=1e-300).rsqrt(), torch.zeros_like(w))
        return (V * inv.unsqueeze(1)) @ V.transpose(1, 2)

    def _cancorr_batch(self, pcols_list, qcols_list):
        """Canonical correlations for same-shape requests. Returns (B, min(dp, dq)) ndarray."""
        torch = self.torch
        iP = torch.tensor(pcols_list, dtype=torch.long, device=self.device)
        iQ = torch.tensor(qcols_list, dtype=torch.long, device=self.device)
        Rpp = self.R[iP[:, :, None], iP[:, None, :]]
        Rqq = self.R[iQ[:, :, None], iQ[:, None, :]]
        Rpq = self.R[iP[:, :, None], iQ[:, None, :]]
        Wp = self._whiten_inv_sqrt(Rpp)
        Wq = self._whiten_inv_sqrt(Rqq)
        M = Wp @ Rpq @ Wq
        s = torch.linalg.svdvals(M).clamp(max=1.0)
        return s.cpu().numpy()

    def _compute(self, requests):
        """requests: list of (pcols, qcols). Groups by shape, batches, fills cache."""
        groups = {}
        for pcols, qcols in requests:
            k = self._key(pcols, qcols)
            if k in self._cache:
                continue
            groups.setdefault((len(k[0]), len(k[1])), {})[k] = None
        for (dp, dq), keys in groups.items():
            keys = list(keys)
            for lo in range(0, len(keys), self.max_batch):
                part = keys[lo:lo + self.max_batch]
                cc = self._cancorr_batch([list(a) for a, _ in part], [list(b) for _, b in part])
                for k, row in zip(part, cc):
                    self._cache[k] = row

    def prime(self, requests):
        self._compute(list(requests))

    def test(self, pcols, qcols, r, alpha):
        """Verbatim Chi2RankTest decision on GPU-computed canonical correlations."""
        k = self._key(pcols, qcols)
        if k not in self._cache:
            self._compute([(pcols, qcols)])
        cancorr = self._cache[k]

        testStat = 0.0
        p = len(pcols)
        q = len(qcols)
        for li in cancorr[r:]:
            li = min(li, 1 - 1e-15)
            testStat += log(1) - log(1 - li * li)
        ratio = 0.0
        for i in range(r):
            li = cancorr[i]
            ratio += 1 / (li * li) - 1
        ratio += self.N * self.N_scaling - r - 0.5 * (p + q + 1)
        testStat = testStat * ratio

        dfreedom = (p - r) * (q - r)
        criticalValue = chi2.ppf(1 - alpha, dfreedom)
        return testStat <= criticalValue
