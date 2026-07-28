"""Expand the decode dictionary with Cognitive Atlas vocabulary (user order 2026-07-28).

Diagnosed gap: the 521k bank is general English; it lacks cognitive-science terminology, which
caps decode quality on the tlvd family regardless of embedding placement. Source: Cognitive
Atlas ontology (concepts + tasks; the standard vocabulary for cognitive tasks), fetched to
<data>/cogatlas_{concepts,tasks}.json.

Decode-only change: NOTHING upstream retrains (f_neg / WeightNet / operator / LoRA never see
the dictionary). New terms are encoded with the SAME frozen LoRA encoder and appended to the
existing bank; output is a NEW file outputs/concept_bank_l3_cog.npz carrying the same
lora_version stamp (the base concept_bank_l3.npz — a v5 symlink — is never touched).
Runs point GRAPHSEM_DICT at the new file to use it; match metrics are dictionary-free and
unaffected by construction.
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lora                                                           # noqa: E402

DATA = os.environ.get("GRAPHSEM_DATA", os.path.abspath(os.path.join(HERE, "..", "..", "data")))
BASE = os.path.join(HERE, "outputs", "concept_bank_l3.npz")
CKPT = os.path.join(HERE, "outputs", "l3_lora.pt")
OUT = os.path.join(HERE, "outputs", "concept_bank_l3_cog.npz")
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")


def cogatlas_terms():
    terms = []
    for fn in ("cogatlas_concepts.json", "cogatlas_tasks.json"):
        for item in json.load(open(os.path.join(DATA, fn))):
            name = (item.get("name") or "").strip().lower()
            if 2 <= len(name) <= 60 and not any(ch.isdigit() for ch in name):
                terms.append(name)
    return sorted(set(terms))


def main():
    bank = np.load(BASE, allow_pickle=True)
    assert abs(float(bank["lora_version"]) - os.path.getmtime(CKPT)) < 1.0, \
        "base bank was encoded with a DIFFERENT lora checkpoint"
    names = [str(x) for x in bank["names"]]
    seen = {n.lower() for n in names}
    new = [t for t in cogatlas_terms() if t not in seen]
    print(f"base bank {len(names)} entries; cogatlas new terms {len(new)}", flush=True)

    st = lora.load_st(DEVICE)
    lora.inject(st)
    lora.load_lora(st, CKPT)
    st.eval()
    embs = []
    with torch.no_grad():
        for i in range(0, len(new), 512):
            embs.append(lora.encode_grad(st, new[i:i + 512], DEVICE).cpu().numpy())
    E_new = np.concatenate(embs).astype(np.float32)

    emb = np.concatenate([bank["emb"], E_new])
    all_names = np.array(names + new, dtype=object)
    np.savez(OUT, emb=emb, names=all_names, lora_version=float(bank["lora_version"]))
    print(f"saved {OUT}: {len(all_names)} entries (+{len(new)})", flush=True)


if __name__ == "__main__":
    main()
