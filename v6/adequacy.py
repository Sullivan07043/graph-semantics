"""P5 — structure-adequacy statistic V(G, X) (THEORY Definition 5). Read-only diagnostic.

V(G, X) = sum over pairs where G claims independence of w(D) * 1{D > tau_n}, tau_n = 2/sqrt(n),
w increasing (implemented: w = identity, declared). It is the mass of data dependence the graph
FORBIDS — the unmodeled-confounder / missing-structure signature; per-pair contributions
localize the defect. Large V => the CI zero-end constraints are actively harmful (attack
surface, THEORY §6).

Implementation rides on terms.ci_table: its per-pair shrunk partial correlations are exactly
the D_{ij|S} values with the tau_n floor already applied (nonzero target == violation). The
S = empty-set group is Definition 5 verbatim (marginal claims); the S != empty groups are its
conditional extension under the unified rule. Reported separately and combined; no thresholds
beyond tau_n, no pass/fail — evidence tables only.

Graph-repair proposals stay OFF (the extend/correct decision is not made here; PLAN P5).
"""
import numpy as np


def compute(g, X, obs_index, score, ci=None, top=20):
    """-> dict with marginal/conditional/total V, pair counts, and top offending pairs.
    ci: terms.ci_table output (built here if None)."""
    if ci is None:
        import terms
        ci = terms.ci_table(g, X, obs_index, score)
    rows = []                                     # (|rho|, rho, a, b, S)
    marg_mass = cond_mass = 0.0
    marg_viol = cond_viol = marg_n = cond_n = 0
    for S, pairs, tg in ci:
        for (a, b), rho in zip(pairs, tg):
            rho = float(rho)
            if not S:
                marg_n += 1
            else:
                cond_n += 1
            if rho == 0.0:
                continue
            if not S:
                marg_mass += abs(rho); marg_viol += 1
            else:
                cond_mass += abs(rho); cond_viol += 1
            rows.append((abs(rho), rho, a, b, list(S)))
    rows.sort(key=lambda r: -r[0])
    return {
        "n_samples": int(X.shape[0]),
        "tau_n": float(2.0 / max(np.sqrt(X.shape[0]), 1.0)),
        "V_marginal": marg_mass, "n_marginal_claims": marg_n, "n_marginal_violations": marg_viol,
        "V_conditional": cond_mass, "n_conditional_claims": cond_n,
        "n_conditional_violations": cond_viol,
        "V_total": marg_mass + cond_mass,
        "top_pairs": [{"a": a, "b": b, "S": S, "rho": rho} for _, rho, a, b, S in rows[:top]],
    }
