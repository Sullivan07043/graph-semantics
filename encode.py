"""Frozen sentence encoder (all-MiniLM-L6-v2) for label texts, plus the fixed decode dictionary loader.
The encoder is FROZEN by design at this stage: the optimization variables are embeddings, not networks."""
import os
import numpy as np

_MODEL = None
DICT_PATH = os.environ.get("GRAPHSEM_DICT", os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "wikipedia", "outputs", "concept_bank_wn.npz")))


def embed(texts):
    """list[str] -> [n, 384] unit-normalized embeddings (frozen)."""
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2",
                                     cache_folder=os.environ.get("HF_CACHE", "/data2/shuhao/hf_cache"))
    return np.asarray(_MODEL.encode(list(texts), normalize_embeddings=True), np.float64)


def load_dictionary():
    """-> (C [V,384], words list[V]) — the fixed auditable decode space."""
    d = np.load(DICT_PATH, allow_pickle=True)
    return d["emb"].astype(np.float64), list(d["names"])
