"""
SpLiCE (Bhalla et al., NeurIPS 2024): decompose an embedding as a sparse NONNEGATIVE combination of a fixed
concept dictionary. min_{w>=0} ||C^T w - e||^2 + lambda ||w||_1. Dataset-agnostic and bias-free: the
dictionary is a fixed, auditable concept space; the sparse solve (not an LLM) picks the concepts.

Used for BOTH (a) defining a variable's meaning target = SpLiCE(its context embedding) and (b) decoding any
predicted meaning into a few weighted concepts, and the concept supports give a dataset-agnostic eval metric.
"""
import numpy as np
from sklearn.linear_model import Lasso


def _norm(M):
    return M / (np.linalg.norm(M, axis=-1, keepdims=True) + 1e-9)


def splice_batch(E, C, alpha=0.01, max_iter=3000):
    """E: [m, d] embeddings; C: [V, d] dictionary. Returns W: [m, V] sparse nonneg weights."""
    E = _norm(np.atleast_2d(np.asarray(E, np.float64)))
    Cn = _norm(np.asarray(C, np.float64))
    Xd = Cn.T                                          # [d, V] design (columns = concept embeddings)
    W = np.zeros((E.shape[0], Cn.shape[0]), np.float32)
    for i in range(E.shape[0]):
        m = Lasso(alpha=alpha, positive=True, fit_intercept=False, max_iter=max_iter)
        m.fit(Xd, E[i])
        W[i] = m.coef_
    return W


def auto_alpha(E, C, target_l0=8, lo=1e-5, hi=2e-2, iters=12):
    """Pick alpha so the mean nonzero count (l0) is ~target_l0 (l0 decreases as alpha grows)."""
    for _ in range(iters):
        mid = (lo * hi) ** 0.5
        l0 = (splice_batch(E, C, alpha=mid) > 1e-6).sum(1).mean()
        if l0 > target_l0:
            lo = mid
        else:
            hi = mid
    return (lo * hi) ** 0.5


def recon(W, C):
    """Concept-space projection: normalize(W @ C_normalized)."""
    Cn = _norm(np.asarray(C, np.float64))
    return _norm(np.asarray(W, np.float64) @ Cn).astype(np.float32)


def top_concepts(w, words, k=8, eps=1e-6):
    idx = np.argsort(w)[::-1]
    return [(words[j], float(w[j])) for j in idx[:k] if w[j] > eps]


def support(w, k=10, eps=1e-6):
    idx = [j for j in np.argsort(w)[::-1][:k] if w[j] > eps]
    return set(int(j) for j in idx)
