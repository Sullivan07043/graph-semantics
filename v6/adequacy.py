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

Graph-repair PROPOSALS are ON (user ruling 2026-07-28: the next phase discovers structure, so
the violation pattern should already speak in structural terms) — but the solver keeps using
the GIVEN graph; nothing here modifies it. propose_repairs() emits ranked hypotheses only.
"""
import numpy as np


def compute(g, X, obs_index, score, ci=None, top=20):
    """-> dict with marginal/conditional/total V, pair counts, and top offending pairs.
    ci: terms.ci_table output (built here if None)."""
    if ci is None:
        import terms
        ci = terms.ci_table(g, X, obs_index, score, mode="full")   # diagnostics need all groups
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


def propose_repairs(g, ci):
    """Structural hypotheses from the violation pattern. PROPOSALS ONLY — the given graph is
    never modified here. Heuristics (evidence-ranked):
      - a connected component of the violation graph with >= 3 nodes -> one shared latent over
        those nodes (confounder-cluster hypothesis);
      - an isolated violating pair -> pairwise repair (missing edge or pairwise confounder;
        direction is NOT identifiable from V alone, stated as such).
    -> list of dicts sorted by violation mass, each carrying its supporting pairs."""
    viol = [(a, b, float(rho), list(S)) for S, pairs, tg in ci
            for (a, b), rho in zip(pairs, tg) if float(rho) != 0.0]
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, _, _ in viol:
        parent[find(a)] = find(b)
    comps = {}
    for a, b, rho, S in viol:
        comps.setdefault(find(a), []).append((a, b, rho, S))
    out = []
    for _, vpairs in comps.items():
        nodes = sorted({x for a, b, _, _ in vpairs for x in (a, b)})
        mass = sum(abs(r) for _, _, r, _ in vpairs)
        kind = "add-shared-latent" if len(nodes) >= 3 else "pairwise-edge-or-confounder"
        out.append({"proposal": kind, "nodes": nodes, "mass": mass,
                    "n_pairs": len(vpairs),
                    "pairs": [{"a": a, "b": b, "rho": r, "S": S}
                              for a, b, r, S in sorted(vpairs, key=lambda v: -abs(v[2]))]})
    out.sort(key=lambda p: -p["mass"])
    return out
