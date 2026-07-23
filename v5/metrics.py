"""Metrics + records. Same three-ruler design as the parent project: judge-ACC (semantic, LLM),
matching-ACC (identity, Hungarian, LLM-free), exact top-1 (strict reference). SpLiCE feeds the judge and
the qualitative output only — never a geometric score."""
import numpy as np
import splice_decode as splice
import judge


def norm_rows(M):
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)


def pick_alpha(E, C, target_l0=8):
    E = np.asarray(E)
    if E.shape[0] > 8:
        E = E[np.linspace(0, E.shape[0] - 1, 8).astype(int)]
    return splice.auto_alpha(E, C, target_l0=target_l0)


def decode_words(V, C, cwords, alpha, topk=6):
    # Negative-loading direction: a node generated with net-negative weight points OPPOSITE its parent in
    # embedding space, and nonnegative SpLiCE selects no concept for such vectors. When the straight decode
    # is empty, decode -v instead and mark the antonym direction with a "low " prefix.
    out = []
    for v in np.atleast_2d(V):
        nv = np.linalg.norm(v)
        if nv < 1e-9:
            out.append([])
            continue
        W = splice.splice_batch(v[None] / nv, C, alpha=alpha)
        ws = [w for w, _ in splice.top_concepts(W[0], cwords, k=topk)]
        if not ws:
            W = splice.splice_batch(-v[None] / nv, C, alpha=alpha)
            ws = ["low " + w for w, _ in splice.top_concepts(W[0], cwords, k=topk)]
        out.append(ws)
    return out


def exact_acc(pred, masked_idx, Tn):
    hits = 0
    for r, i in enumerate(masked_idx):
        p = pred[r] / (np.linalg.norm(pred[r]) + 1e-9)
        hits += int(np.argmax(Tn @ p) == i)
    return hits / max(len(masked_idx), 1)


def match_acc(pred, masked_idx, T):
    from scipy.optimize import linear_sum_assignment
    S = norm_rows(pred) @ norm_rows(T[masked_idx]).T
    r, c = linear_sum_assignment(-S)
    return float(np.mean(c == np.arange(len(masked_idx))))


def judge_completion(words_list, true_texts):
    """-> (acc | None, verdicts | None); judge mode 'completion' (dominant meaning, synonyms count)."""
    if not judge.available():
        return None, None
    v = judge.judge_batch([(w, t) for w, t in zip(words_list, true_texts)], "completion")
    ok = [x for x in (v or []) if x is not None]
    return (float(np.mean(ok)), v) if ok else (None, None)


def judge_latents(words_list, gt_texts):
    if not judge.available():
        return None, None
    v = judge.judge_batch([(w, t) for w, t in zip(words_list, gt_texts)], "latent")
    ok = [x for x in (v or []) if x is not None]
    return (float(np.mean(ok)), v) if ok else (None, None)
