"""Build the expanded decode dictionary: WordNet concept bank (existing) UNION ConceptNet Numberbatch
English terms (multi-word phrases included). One dictionary for ALL datasets; alpha stays globally fixed
(pick_alpha); nothing here is ever tuned per dataset.

Filter for Numberbatch terms: lowercase alphabetic words joined by underscores, 1-3 words, each word >= 2
chars, total length >= 3, no digits. Terms are embedded with the CURRENT frozen encoder (encode.embed) on
GPU if available, and saved as npz {names, emb} next to the WordNet bank.

Usage: python build_dictionary.py [out.npz]   (env: GRAPHSEM_NB = numberbatch txt.gz path)
"""
import gzip, os, re, sys
import numpy as np
import encode

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("GRAPHSEM_DATA", os.path.abspath(os.path.join(HERE, "..", "data")))
NB = os.environ.get("GRAPHSEM_NB", os.path.join(DATA, "numberbatch-en-19.08.txt.gz"))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(encode.DICT_PATH), "concept_bank_big_minilm.npz")

TERM_RE = re.compile(r"^[a-z]{2,}(_[a-z]{2,}){0,2}$")


def numberbatch_terms(path):
    out = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        next(f)                                          # header: "<n> <dim>"
        for line in f:
            t = line.split(" ", 1)[0]
            if TERM_RE.match(t):
                out.append(t.replace("_", " "))
    return out


def main():
    wn = np.load(encode.DICT_PATH, allow_pickle=True)
    wn_names = [str(x) for x in wn["names"]]
    nb = numberbatch_terms(NB)
    seen = {n.lower() for n in wn_names}
    extra = [t for t in nb if t.lower() not in seen]
    print(f"wordnet {len(wn_names)} + numberbatch {len(nb)} -> {len(extra)} new terms", flush=True)

    import torch
    from sentence_transformers import SentenceTransformer
    dev = "cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else "cpu"
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2",
                                cache_folder=os.environ.get("HF_CACHE", "/data2/shuhao/hf_cache"),
                                device=dev)
    E = model.encode(extra, batch_size=2048, normalize_embeddings=True, show_progress_bar=True)
    names = np.array(wn_names + extra, object)
    emb = np.concatenate([wn["emb"].astype(np.float32), np.asarray(E, np.float32)])
    # WordNet bank rows are MiniLM embeddings too but may predate normalization; normalize everything
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    np.savez_compressed(OUT, names=names, emb=emb)
    print(f"saved {OUT}: {emb.shape}", flush=True)


if __name__ == "__main__":
    main()
