"""Referee-space evaluation metrics (2026-08-14 metric revision).

Motivation, from the group-meeting critique of match: fold-wise Hungarian assignment measures
relative discrimination among 7-9 targets, tolerates sibling substitution (label copying scores
high), and its value depends on each method's own embedding space. These two metrics fix those
defects and are computed from decoded TEXT, so every method is scored in ONE fixed public
referee space (base e5-large-v2, pinned revision), never in its own.

NRR  (neutral-referee retrieval): rank the true label among the dataset's full label set by
     referee cosine to the prediction. Report top-1 accuracy and MRR.
SDA  (sibling-discrimination accuracy): the prediction must be strictly closer to the true
     label than to EVERY sibling label (same published construct, or same channel family on
     robots). Punishes exactly the family-level substitution that inflates match.

Both report a chance-normalized companion: (x - chance) / (1 - chance).
"""
import os

import numpy as np

_REF = None
_REV = "f169b11e22de13617baa190a028a32f3493550b6"


def referee():
    global _REF
    if _REF is None:
        import torch
        from sentence_transformers import SentenceTransformer
        _REF = SentenceTransformer(
            "intfloat/e5-large-v2", revision=_REV,
            device="cuda" if torch.cuda.is_available() else "cpu",
            cache_folder=os.environ.get("HF_CACHE", "/data2/shuhao/hf_cache"))
    return _REF


def embed(texts):
    return np.asarray(referee().encode([f"query: {t}" for t in texts],
                                       batch_size=256, normalize_embeddings=True))


def nrr(pred_texts, true_idx, candidate_labels, pred_emb=None, cand_emb=None):
    """Rank the true label among all candidate labels for each prediction.

    pred_texts: list of decoded predictions (None allowed -> scored as rank last).
    true_idx: index of each prediction's true label inside candidate_labels.
    Returns (top1, mrr, ranks). Ties count against the prediction (worst rank among ties).
    """
    cand_emb = cand_emb if cand_emb is not None else embed(candidate_labels)
    have = [i for i, t in enumerate(pred_texts) if t]
    ranks = np.full(len(pred_texts), len(candidate_labels), float)
    if have:
        pe = pred_emb if pred_emb is not None else embed([pred_texts[i] for i in have])
        sims = pe @ cand_emb.T                                    # [n_have, n_cand]
        for r, i in enumerate(have):
            s = sims[r]
            ranks[i] = 1 + (s > s[true_idx[i]]).sum() + 0.5 * ((s == s[true_idx[i]]).sum() - 1)
    top1 = float((ranks == 1).mean())
    mrr = float((1.0 / ranks).mean())
    return top1, mrr, ranks.tolist()


def sda(pred_texts, true_idx, sibling_idx, candidate_labels, cand_emb=None):
    """Strictly closer to the true label than to every sibling. None predictions fail.

    sibling_idx: per prediction, the indices (into candidate_labels) of its sibling labels.
    Items without siblings are skipped (returned count excludes them).
    """
    cand_emb = cand_emb if cand_emb is not None else embed(candidate_labels)
    have = [i for i, t in enumerate(pred_texts) if t]
    pe = embed([pred_texts[i] for i in have]) if have else np.zeros((0, cand_emb.shape[1]))
    pe_of = dict(zip(have, pe))
    wins, n = 0, 0
    for i, t in enumerate(pred_texts):
        sibs = [s for s in sibling_idx[i] if s != true_idx[i]]
        if not sibs:
            continue
        n += 1
        if t is None or i not in pe_of:
            continue
        s = pe_of[i] @ cand_emb.T
        if all(s[true_idx[i]] > s[j] for j in sibs):
            wins += 1
    return (wins / n if n else None), n


def chance_norm(x, chance):
    if x is None or chance is None or chance >= 1:
        return None
    return (x - chance) / (1 - chance)
