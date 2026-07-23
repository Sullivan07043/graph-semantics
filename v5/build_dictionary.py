"""Build the expanded decode dictionary: WordNet concept names (existing bank) UNION ConceptNet
Numberbatch English terms (multi-word phrases included), all embedded with the CURRENT frozen encoder
(env GRAPHSEM_ENCODER via encode.py). One dictionary for ALL datasets; alpha stays globally fixed
(pick_alpha); nothing here is ever tuned per dataset.

Filter for Numberbatch terms: lowercase alphabetic words joined by underscores, 1-3 words, each word >= 2
chars, no digits. Saved as npz {names, emb} at encode.DICT_PATH (or argv[1]).

Usage: [GRAPHSEM_ENCODER=e5-large] python build_dictionary.py [out.npz]
       (env: GRAPHSEM_NB = numberbatch txt.gz path)
"""
import gzip, os, re, sys
import numpy as np
import encode

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("GRAPHSEM_DATA", os.path.abspath(os.path.join(HERE, "..", "data")))
NB = os.environ.get("GRAPHSEM_NB", os.path.join(DATA, "numberbatch-en-19.08.txt.gz"))
WN_BANK = os.path.join(os.path.dirname(encode.DICT_PATH), "concept_bank_wn.npz")
OUT = sys.argv[1] if len(sys.argv) > 1 else encode.DICT_PATH

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
    wn_names = [str(x) for x in np.load(WN_BANK, allow_pickle=True)["names"]]
    nb = numberbatch_terms(NB)
    seen = {n.lower() for n in wn_names}
    extra = [t for t in nb if t.lower() not in seen]
    names = wn_names + extra
    print(f"encoder={encode.ENCODER}: wordnet {len(wn_names)} + numberbatch-new {len(extra)} "
          f"= {len(names)} terms", flush=True)
    E = encode.embed(names, batch_size=1024).astype(np.float32)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    np.savez_compressed(OUT, names=np.array(names, object), emb=E)
    print(f"saved {OUT}: {E.shape}", flush=True)


if __name__ == "__main__":
    main()
