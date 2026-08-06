"""Given-graph representation and the constraints derived from it.

A Graph holds latent node names, observed node names, and DIRECTED typed edges (any of latent->latent,
latent->observed, observed->observed, observed->latent). Constraints read off the graph:
  - parents(node): the generation channel (who generates whom);
  - independent_pairs(): marginal independence = in the DAG, two nodes are marginally dependent only if one
    is an ancestor of the other or they share a common ancestor (a trek connects them); every other pair is
    d-separated by the empty set -> independence constraint;
  - estimate_weights(X): signed strengths on the GIVEN support, from data: each latent's score = first
    principal component of its observed descendants; weight(edge) = corr of source score/column with target
    column (data-grounded, deterministic; the graph itself is never altered).
"""
import re
import numpy as np


class Graph:
    def __init__(self, latents, observed, edges):
        self.latents = list(latents)
        self.observed = list(observed)
        self.edges = [(a, b) for a, b in edges]
        self.nodes = self.latents + self.observed
        self._pa = {n: [] for n in self.nodes}
        self._ch = {n: [] for n in self.nodes}
        for a, b in self.edges:
            self._pa[b].append(a)
            self._ch[a].append(b)

    # ---------------------------------------------------------------- basics
    def parents(self, n):
        return list(self._pa[n])

    def children(self, n):
        return list(self._ch[n])

    def is_latent(self, n):
        return n in set(self.latents)

    def ancestors(self, n):
        out, stack = set(), [n]
        while stack:
            u = stack.pop()
            for p in self._pa[u]:
                if p not in out:
                    out.add(p)
                    stack.append(p)
        return out

    def observed_descendants(self, n):
        out, seen, stack = [], set(), [n]
        while stack:
            u = stack.pop()
            for c in self._ch[u]:
                if c in seen:
                    continue
                seen.add(c)
                stack.append(c)
                if c in set(self.observed):
                    out.append(c)
        return out

    # ---------------------------------------------------------------- constraints
    def independent_pairs(self):
        """Pairs marginally d-separated (empty conditioning set) in the DAG: neither is an ancestor of the
        other and they share no common ancestor (counting a node as its own ancestor)."""
        anc = {n: self.ancestors(n) | {n} for n in self.nodes}
        pairs = []
        for i, a in enumerate(self.nodes):
            for b in self.nodes[i + 1:]:
                if anc[a] & anc[b]:
                    continue
                pairs.append((a, b))
        return pairs

    def markov_blanket(self, n):
        """Parents + children + spouses (other parents of children)."""
        mb = set(self._pa[n]) | set(self._ch[n])
        for c in self._ch[n]:
            mb |= set(self._pa[c])
        mb.discard(n)
        return mb

    def mb_observed(self, n):
        """Observed projection of the Markov-blanket closure: observed members of MB(n) plus the
        children of n's parents (siblings). In a bipartite design graph MB(n) is just the latent
        parent (no label), so the operational locality set for label completion is the siblings —
        'everything relevant to n flows through its parents'."""
        s = self.markov_blanket(n)
        for p in self._pa[n]:
            s |= set(self._ch[p])
        s.discard(n)
        obs = set(self.observed)
        return sorted(x for x in s if x in obs)

    def trek_pairs(self):
        """Pairs CONNECTED by a trek (one is an ancestor of the other, or they share a common
        ancestor) — the complement of independent_pairs(). Faithfulness direction: these pairs are
        dependent, so their semantics should not collapse to orthogonality."""
        anc = {n: self.ancestors(n) | {n} for n in self.nodes}
        return [(a, b) for i, a in enumerate(self.nodes) for b in self.nodes[i + 1:]
                if anc[a] & anc[b]]

    def v_structures(self):
        """(p1, p2, c) triples where p1 -> c <- p2 and p1, p2 are marginally independent (no trek).
        Conditioning on the collider c induces dependence between the parents (explaining away)."""
        anc = {n: self.ancestors(n) | {n} for n in self.nodes}
        out = []
        for c in self.nodes:
            ps = self._pa[c]
            for i, p1 in enumerate(ps):
                for p2 in ps[i + 1:]:
                    if not (anc[p1] & anc[p2]):
                        out.append((p1, p2, c))
        return out

    def estimate_weights(self, X, obs_index):
        """Signed edge strengths on the given support. X: [n_samples, n_observed]; obs_index: name -> col.
        Latent score = PC1 of its observed descendants (sign-aligned to positive mean loading)."""
        score = {}
        for L in self.latents:
            dobs = self.observed_descendants(L)
            if not dobs:
                continue
            sub = X[:, [obs_index[o] for o in dobs]]
            sub = sub - sub.mean(0)
            _, _, vt = np.linalg.svd(sub, full_matrices=False)
            s = sub @ vt[0]
            if np.mean([np.corrcoef(s, X[:, obs_index[o]])[0, 1] for o in dobs]) < 0:
                s = -s
            score[L] = (s - s.mean()) / (s.std() + 1e-9)
        W = {}
        for a, b in self.edges:
            va = score.get(a) if self.is_latent(a) else X[:, obs_index[a]]
            vb = score.get(b) if self.is_latent(b) else X[:, obs_index[b]]
            if va is None or vb is None:
                W[(a, b)] = 0.0
            else:
                W[(a, b)] = float(np.corrcoef(va, vb)[0, 1])
        return W, score


def from_dot(path):
    """Parse a TLVD-style .dot: nodes colored red (latent) / blue (observed); 'a -> b' edges."""
    txt = open(path).read()
    latents, observed = [], []
    for m in re.finditer(r"(\w+)\s*\[color\s*=\s*(red|blue)\]", txt):
        (latents if m.group(2) == "red" else observed).append(m.group(1))
    edges = re.findall(r"(\w+)\s*->\s*(\w+)", txt)
    return Graph(latents, observed, edges)


def bipartite(construct_of):
    """Design graph: latent(construct) -> observed, from a mapping observed_name -> construct_name."""
    observed = list(construct_of)
    latents = sorted(set(construct_of.values()))
    edges = [(construct_of[o], o) for o in observed]
    return Graph(latents, observed, edges)
