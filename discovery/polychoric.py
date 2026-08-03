"""Polychoric correlation for ordinal items, plus a drop-in rank test for RLCD.

Why this exists. Pearson correlation on discretized ordinal data is attenuated: it estimates the
correlation of the categories, not of the latent responses that generated them. Our whole discovery
stack reads Pearson correlations of Likert columns, in the marginal skeleton and inside RLCD's rank
test, and `rank_test_calibration.py` measured the resulting type I error inflation on skewed items.
Polychoric correlation is the standard psychometric correction: it estimates the correlation of the
underlying bivariate-normal latents from the contingency table.

Estimator: the usual two-step method. Thresholds come from the marginal category frequencies via the
normal quantile function, which is exact given normal latents. The correlation is then the scalar
that maximizes the multinomial log-likelihood of the observed table, solved by bounded scalar
optimization on the bivariate normal rectangle probabilities.

PolychoricRankTest duck-types causal-learn's Chi2RankTest (`test(pcols, qcols, r, alpha)`), so it can
be handed to `RLCD(..., ranktest_method=...)` without touching causal-learn. It runs the same Wilks
statistic; the only change is the correlation matrix the canonical correlations are computed from.
"""
import os

import numpy as np
from scipy.optimize import brentq
from scipy.stats import chi2, norm

# Gauss-Legendre nodes for the bivariate normal CDF. Drezner-Wesolowsky form:
#   Phi2(h,k,rho) = Phi(h)Phi(k) + (1/2pi) * int_0^rho exp(-(h^2 - 2 t h k + k^2)/(2(1-t^2)))
#                                            / sqrt(1-t^2) dt
# The integrand is smooth on [0, rho], so fixed nodes are accurate and fully vectorized, which
# matters here: a 162-variable matrix is 13041 pairs and each pair needs a root search.
_GL_X, _GL_W = np.polynomial.legendre.leggauss(48)


def _bvn_cdf(h, k, rho):
    """P(H <= h, K <= k) for standard bivariate normal with correlation rho. Vectorized over h, k."""
    h = np.asarray(h, float)
    k = np.asarray(k, float)
    base = norm.cdf(h) * norm.cdf(k)
    if abs(rho) < 1e-12:
        return base
    t = 0.5 * rho * (_GL_X + 1.0)                                  # nodes on [0, rho]
    w = 0.5 * rho * _GL_W
    t = t.reshape((-1,) + (1,) * h.ndim)
    w = w.reshape((-1,) + (1,) * h.ndim)
    om = 1.0 - t ** 2
    finite = np.isfinite(h) & np.isfinite(k)
    hs = np.where(finite, h, 0.0)
    ks = np.where(finite, k, 0.0)
    integrand = np.exp(-(hs ** 2 - 2.0 * t * hs * ks + ks ** 2) / (2.0 * om)) / np.sqrt(om)
    # with an infinite argument the bivariate CDF collapses to the univariate one, already in base
    integrand = np.where(finite, integrand, 0.0)
    return base + (w * integrand).sum(0) / (2.0 * np.pi)


def _bvn_pdf(h, k, rho):
    """Standard bivariate normal density. This is exactly d/drho of _bvn_cdf, which supplies the
    analytic score used by the root search below. Density is zero at an infinite argument."""
    om = 1.0 - rho ** 2
    finite = np.isfinite(h) & np.isfinite(k)
    hs = np.where(finite, h, 0.0)
    ks = np.where(finite, k, 0.0)
    out = np.exp(-(hs ** 2 - 2.0 * rho * hs * ks + ks ** 2) / (2.0 * om)) / (2.0 * np.pi * np.sqrt(om))
    return np.where(finite, out, 0.0)


def _thresholds(col):
    """Normal cut points implied by the observed category frequencies."""
    vals, counts = np.unique(col, return_counts=True)
    cum = np.cumsum(counts)[:-1] / len(col)
    return norm.ppf(np.clip(cum, 1e-9, 1 - 1e-9)), vals


def _table(x, y, vx, vy):
    ix = np.searchsorted(vx, x)
    iy = np.searchsorted(vy, y)
    T = np.zeros((len(vx), len(vy)))
    np.add.at(T, (ix, iy), 1.0)
    return T


def _corner_grid(a, b):
    """Threshold grids padded with the infinite outer edges, as a cell corner mesh."""
    A = np.concatenate(([-np.inf], a, [np.inf]))
    B = np.concatenate(([-np.inf], b, [np.inf]))
    return np.meshgrid(A, B, indexing="ij")


def _cells(H, K, rho, fn):
    """Inclusion-exclusion over cell corners: turns a corner-evaluated function into cell values."""
    V = fn(H, K, rho)
    return V[1:, 1:] - V[:-1, 1:] - V[1:, :-1] + V[:-1, :-1]


def polychoric_pair(x, y):
    """Two-step polychoric correlation between two ordinal columns.

    Thresholds are fixed from the margins, then the correlation is the root of the analytic
    log-likelihood score sum_cells T_ij * (dP_ij/drho) / P_ij, where dP/drho is the bivariate
    normal density differenced over the cell corners.
    """
    a, vx = _thresholds(x)
    b, vy = _thresholds(y)
    if len(vx) < 2 or len(vy) < 2:
        return 0.0
    T = _table(x, y, vx, vy)
    H, K = _corner_grid(a, b)

    def score(rho):
        P = np.clip(_cells(H, K, rho, _bvn_cdf), 1e-12, None)
        dP = _cells(H, K, rho, _bvn_pdf)
        return float((T * dP / P).sum())

    try:
        lo, hi = score(-0.995), score(0.995)
        if lo * hi > 0:
            return -0.995 if lo < 0 else 0.995
        return float(brentq(score, -0.995, 0.995, xtol=1e-5))
    except Exception:
        return float(np.corrcoef(x, y)[0, 1])


def polychoric_matrix(X, verbose=False):
    """[n, p] ordinal data -> [p, p] polychoric correlation matrix."""
    X = np.asarray(X)
    p = X.shape[1]
    R = np.eye(p)
    for i in range(p):
        for j in range(i + 1, p):
            R[i, j] = R[j, i] = polychoric_pair(X[:, i], X[:, j])
        if verbose and (i + 1) % 10 == 0:
            print(f"[polychoric] {i + 1}/{p} rows done", flush=True)
    return _nearest_psd(R)


def _nearest_psd(R, eps=1e-6):
    """Two-step polychoric estimates are pairwise, so the matrix can be indefinite."""
    w, V = np.linalg.eigh((R + R.T) / 2)
    if w.min() >= eps:
        return R
    w = np.clip(w, eps, None)
    A = V @ np.diag(w) @ V.T
    d = np.sqrt(np.diag(A))
    return A / np.outer(d, d)


class PolychoricRankTest:
    """Drop-in for causal-learn's Chi2RankTest, reading a polychoric correlation matrix.

    Same Wilks statistic and same decision rule. Canonical correlations come from the correlation
    matrix rather than from the standardized raw columns, which is the only difference.
    """

    def __init__(self, data, N_scaling=1, R=None, verbose=False):
        self.data = np.asarray(data, float)
        self.N = self.data.shape[0]
        self.N_scaling = N_scaling
        self.R = polychoric_matrix(self.data, verbose=verbose) if R is None else np.asarray(R, float)
        self._cache = {}

    def _cancorr(self, pcols, qcols):
        key = (tuple(sorted(pcols)), tuple(sorted(qcols)))
        if key in self._cache:
            return self._cache[key]
        Rpp = self.R[np.ix_(pcols, pcols)]
        Rqq = self.R[np.ix_(qcols, qcols)]
        Rpq = self.R[np.ix_(pcols, qcols)]
        # canonical correlations = singular values of Rpp^{-1/2} Rpq Rqq^{-1/2}
        M = _inv_sqrt(Rpp) @ Rpq @ _inv_sqrt(Rqq)
        s = np.linalg.svd(M, compute_uv=False)
        s = np.clip(s, 0.0, 1 - 1e-15)
        self._cache[key] = s
        return s

    def test(self, pcols, qcols, r, alpha):
        """Null: rank(cross-covariance) <= r. Returns if_fail_to_reject, as Chi2RankTest does."""
        s = self._cancorr(list(pcols), list(qcols))
        p, q = len(pcols), len(qcols)
        if r >= min(p, q):
            return True
        stat = -np.log1p(-s[r:] ** 2).sum()
        ratio = sum(1.0 / (s[i] ** 2) - 1.0 for i in range(r) if s[i] > 0)
        ratio += self.N * self.N_scaling - r - 0.5 * (p + q + 1)
        stat *= ratio
        df = (p - r) * (q - r)
        return bool(stat <= chi2.ppf(1 - alpha, df))


def _inv_sqrt(A):
    """Symmetric inverse square root via eigendecomposition (A is a correlation block)."""
    w, V = np.linalg.eigh((A + A.T) / 2)
    w = np.clip(w, 1e-10, None)
    return V @ np.diag(1.0 / np.sqrt(w)) @ V.T


def certify(seed=0, n=3000, verbose=True):
    """Separate the implementation from the input. Fed the Pearson correlation matrix, this test
    must reproduce causal-learn's Chi2RankTest, whose canonical correlations come from the raw
    standardized columns. Any disagreement is a bug in the rank test here, not an effect of
    polychoric estimation."""
    import sys
    sys.path.insert(0, "/data2/shuhao/semantic_interpretation/causal-learn")
    from causallearn.search.HiddenCausal.RLCD.Chi2RankTest import Chi2RankTest

    rng = np.random.default_rng(seed)
    f = rng.standard_normal((n, 2))
    lam = rng.uniform(0.5, 0.9, (2, 8))
    X = f @ lam + rng.standard_normal((n, 8)) * 0.5
    mine = PolychoricRankTest(X, R=np.corrcoef(X, rowvar=False))
    ref = Chi2RankTest(X)
    worst, bad = 0.0, 0
    for pcols, qcols in [([0, 1], [2, 3]), ([0, 1, 2], [3, 4, 5]), ([0], [1, 2, 3]),
                         ([0, 1, 2, 3], [4, 5, 6, 7])]:
        for r in range(min(len(pcols), len(qcols))):
            for alpha in (0.01, 0.05, 0.1):
                a = mine.test(pcols, qcols, r, alpha)
                b = bool(ref.test(pcols, qcols, r, alpha))
                bad += a != b
        s_mine = mine._cancorr(pcols, qcols)
        s_ref = ref._cancorr(pcols, qcols) if hasattr(ref, "_cancorr") else None
        if s_ref is not None:
            worst = max(worst, float(np.abs(np.sort(s_mine) - np.sort(s_ref)).max()))
    if verbose:
        print(f"[certify] decisions disagreeing with causal-learn: {bad} "
              f"({'PASS' if bad == 0 else 'FAIL'})")
    return bad


if __name__ == "__main__":
    certify()
    # self-check: on continuous-normal data discretized finely, polychoric must recover the
    # generating correlation better than Pearson does
    rng = np.random.default_rng(0)
    n = 4000
    for rho in [0.3, 0.6]:
        Z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], n)
        cuts = [0.25, 0.85, 1.4, 2.0]                       # the skewed 5-point profile
        D = np.digitize(Z, cuts).astype(float)
        pear = np.corrcoef(D[:, 0], D[:, 1])[0, 1]
        poly = polychoric_pair(D[:, 0], D[:, 1])
        print(f"true {rho:.2f} | pearson {pear:.3f} (err {abs(pear - rho):.3f}) | "
              f"polychoric {poly:.3f} (err {abs(poly - rho):.3f})")
