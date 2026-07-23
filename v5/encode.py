"""Frozen sentence encoder for label texts, plus the fixed decode dictionary loader.
The encoder is FROZEN by design at this stage: the optimization variables are embeddings, not networks.

Encoder selection is a GLOBAL choice (env GRAPHSEM_ENCODER), fitted once on the dev pool by bakeoff.py
and then frozen — never per dataset. Bake-off 2026-07-14 (dev-pool matching-ACC, encoder-invariant):
e5-large 0.737 > gte-large 0.704 > minilm 0.625 -> default e5-large."""
import os
import numpy as np

SPECS = {
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", "", "concept_bank_big_minilm.npz"),
    "gte-large": ("thenlper/gte-large", "", "concept_bank_big_gte.npz"),
    "e5-large": ("intfloat/e5-large-v2", "query: ", "concept_bank_big_e5.npz"),
}
ENCODER = os.environ.get("GRAPHSEM_ENCODER", "e5-large")
_MODEL_NAME, _PREFIX, _DICT_FILE = SPECS[ENCODER]
_DICT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "..", "..", "wikipedia", "outputs"))
DICT_PATH = os.environ.get("GRAPHSEM_DICT", os.path.join(_DICT_DIR, _DICT_FILE))

_MODEL = None


def embed(texts, batch_size=1024):
    """list[str] -> [n, d] unit-normalized embeddings (frozen encoder, symmetric prefix if required)."""
    global _MODEL
    if _MODEL is None:
        import torch
        from sentence_transformers import SentenceTransformer
        dev = None
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            dev = "cuda:1"
        _MODEL = SentenceTransformer(_MODEL_NAME, device=dev,
                                     cache_folder=os.environ.get("HF_CACHE", "/data2/shuhao/hf_cache"))
    return np.asarray(_MODEL.encode([_PREFIX + t for t in texts], batch_size=batch_size,
                                    normalize_embeddings=True), np.float64)


def load_dictionary():
    """-> (C [V,d], words list[V]) — the fixed auditable decode space (must match ENCODER)."""
    d = np.load(DICT_PATH, allow_pickle=True)
    return d["emb"].astype(np.float32), [str(x) for x in d["names"]]
