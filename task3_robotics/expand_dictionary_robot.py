"""Add index-bearing and axis atoms to the decode dictionary (robot domain).

Diagnosed gap: robot channel labels differ only by an index or an axis letter ("angle of robot arm
joint 3" against "... joint 5"), and the 522k concept bank has no atom that can express that
difference, so decodes of correctly-placed embeddings come out identical and the judge has nothing
to discriminate with. The coverage precheck missed this because it tests whether words exist, not
whether the vocabulary is fine-grained enough to separate the labels.

Same recipe as v6/tools/expand_dictionary.py, and the same guarantees: decode-side only, nothing
retrains, new terms are encoded with the SAME frozen LoRA encoder and appended; output is a NEW
bank carrying the same lora_version stamp. The certified pool keeps using the cog bank; robot runs
point GRAPHSEM_DICT at this one. match is dictionary-free and unaffected by construction.
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
V6 = os.path.join(os.path.dirname(HERE), "v6")
sys.path.insert(0, V6)
import lora                                                           # noqa: E402

BASE = os.path.join(V6, "outputs", "concept_bank_l3_cog.npz")
CKPT = os.path.join(V6, "outputs", "l3_lora.pt")
OUT = os.path.join(V6, "outputs", "concept_bank_l3_robot.npz")
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

TERMS = (
    [f"joint {i}" for i in range(1, 8)]
    + [f"robot arm joint {i}" for i in range(1, 8)]
    + [f"gripper finger {i}" for i in (1, 2)]
    + [f"finger {i}" for i in (1, 2)]
    + ["x axis", "y axis", "z axis",
       "x coordinate", "y coordinate", "z coordinate",
       "x component", "y component", "z component", "w component",
       "rotation about the x axis", "rotation about the y axis", "rotation about the z axis",
       "quaternion component", "end-effector position", "commanded motion", "gripper command"]
)


def main():
    d = np.load(BASE, allow_pickle=True)
    names = [str(x) for x in d["names"]]
    have = set(n.lower() for n in names)
    new = [t for t in TERMS if t.lower() not in have]
    print(f"[robot dict] base {len(names)} atoms | adding {len(new)} "
          f"({len(TERMS) - len(new)} already present)")

    st = lora.load_st(DEVICE)
    lora.inject(st)
    lora.load_lora(st, CKPT)
    st.eval()
    with torch.no_grad():
        E = lora.encode_grad(st, new, DEVICE, max_len=128).cpu().numpy()
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)

    emb = np.concatenate([d["emb"], E.astype(d["emb"].dtype)])
    out_names = np.array(names + new)
    np.savez(OUT, emb=emb, names=out_names, lora_version=d["lora_version"])
    print(f"[robot dict] saved {OUT}: {len(out_names)} atoms "
          f"(lora_version preserved: {float(d['lora_version']):.0f})")


if __name__ == "__main__":
    main()
