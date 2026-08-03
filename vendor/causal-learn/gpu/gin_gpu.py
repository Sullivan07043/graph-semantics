"""GIN with a corrected driver and a GPU-batched HSIC backend.

Base: causallearn.search.HiddenCausal.GIN.GIN (cluster phase + causal-order phase).

DECLARED DEVIATION from the official GIN(): the official cluster loop tests
`data[:, [z]] for z in range(len(remain_var_set))`, i.e. it treats POSITIONS as COLUMN
INDICES and tests e against the first |remain| columns of the data (including the
candidate cluster's own columns). The paper's GIN condition, and the same file's GIN_MI
implementation, test e against the actual remaining variables. This port does the latter.
The same correction applies to the causal-order phase.

Two backends share one driver, so certification (certify.py) isolates the backend swap:
  backend='reference': loops the official hsic_test_gamma per test (slow, exact);
  backend='gpu':       ColumnBank batched HSIC (batched_hsic.py).

API: gin_clusters(data, alpha) -> (causal_order, unordered_clusters), with variable
indices into data's columns. No GeneralGraph is built; downstream composition is ours.
"""
import os
import sys
from itertools import combinations

import numpy as np
from scipy.stats import chi2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from batched_hsic import ColumnBank                                    # noqa: E402
from causallearn.search.FCMBased.lingam.hsic import hsic_test_gamma    # noqa: E402
from causallearn.search.HiddenCausal.GIN.GIN import (                  # noqa: E402
    merge_overlaping_cluster, array_split)


def _fisher(pvals: np.ndarray) -> float:
    pvals = np.clip(pvals, 1e-5, None)
    stat = -2.0 * np.log(pvals).sum()
    return 1.0 - chi2.cdf(stat, 2 * len(pvals))


def _e_vectors(data, cov, cands, var_set):
    """Batched GIN surrogates: e = data[:, X] @ (last right-singular vector of cov[Z, X]).
    All candidates at one phase share |Z|, so the SVDs batch."""
    Zs = [sorted(var_set - set(c)) for c in cands]
    Xs = [list(c) for c in cands]
    sub = np.stack([cov[np.ix_(Z, X)] for Z, X in zip(Zs, Xs)])        # (B, m, k)
    _, _, vt = np.linalg.svd(sub)
    omega = vt.transpose(0, 2, 1)[:, :, -1]                            # (B, k) = V[:, -1]
    cols = data[:, np.asarray(Xs)].swapaxes(0, 1)                      # (B, n, k)
    E = np.einsum('bnk,bk->bn', cols, omega)
    return E, Zs


def gin_clusters(data, alpha=0.05, backend='gpu', device='cuda:0', verbose=True):
    data = np.asarray(data, np.float64)
    n, p = data.shape
    cov = np.cov(data.T)
    bank = ColumnBank(data, device) if backend == 'gpu' else None

    def pvals_vs_all(E):
        """(B, p) p-values of every e row against every data column."""
        if backend == 'gpu':
            return bank.score(E)[1]
        out = np.empty((E.shape[0], p))
        for b in range(E.shape[0]):
            for j in range(p):
                out[b, j] = hsic_test_gamma(E[b][:, None], data[:, [j]])[1]
        return out

    var_set = set(range(p))
    clusters_list = []
    cluster_size = 2
    max_cands = int(os.environ.get("GIN_MAX_CANDS", 100_000))
    while cluster_size < len(var_set):
        cands = list(combinations(sorted(var_set), cluster_size))
        if not cands:
            break
        if len(cands) > max_cands:
            # circuit breaker: a phase that accepted nothing does not shrink var_set, and
            # candidate counts then grow combinatorially; stop and declare instead of
            # burning the GPU on a doomed enumeration.
            print(f"[gin] STOP: {len(cands)} candidates at size {cluster_size} exceeds "
                  f"GIN_MAX_CANDS={max_cands}; returning clusters found so far", flush=True)
            break
        E, Zs = _e_vectors(data, cov, cands, var_set)
        P = pvals_vs_all(E)
        accepted = []
        for b, (c, Z) in enumerate(zip(cands, Zs)):
            if _fisher(P[b, Z]) >= alpha:
                accepted.append(c)
        accepted = merge_overlaping_cluster(accepted)
        clusters_list += accepted
        for c in accepted:
            var_set -= set(c)
        if verbose:
            print(f"[gin] size {cluster_size}: {len(cands)} candidates -> "
                  f"{len(accepted)} clusters, {len(var_set)} vars left", flush=True)
        cluster_size += 1

    # causal-order phase (official logic, corrected z-indexing)
    causal_order = []
    updated = True
    while updated:
        updated = False
        X0, Z0 = [], []
        for ck in causal_order:
            c1, c2 = array_split(ck, 2)
            X0 += c1
            Z0 += c2
        for i, ci in enumerate(clusters_list):
            is_root = True
            ci1, ci2 = array_split(list(ci), 2)
            for j, cj in enumerate(clusters_list):
                if i == j:
                    continue
                cj1, _ = array_split(list(cj), 2)
                Z = Z0 + ci2
                Ei, _ = _e_vectors(data, cov, [tuple(X0 + ci1 + cj1)],
                                   set(X0 + ci1 + cj1) | set(Z))
                Pi = pvals_vs_all(Ei)
                if _fisher(Pi[0, Z]) < alpha:
                    is_root = False
                    break
            if is_root:
                causal_order.append(list(ci))
                clusters_list.remove(ci)
                updated = True
                break

    return causal_order, [list(c) for c in clusters_list]
