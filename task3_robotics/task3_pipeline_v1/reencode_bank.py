"""Re-encode the robot concept bank in a trained adapter's space.

The dictionary and the encoder must live in one space: decode searches the bank by cosine, and
main.py refuses a mismatched pair (lora_version). After TRAIN_LORA the adapter has moved, so
each run re-encodes the bank once with its own adapter and stamps lora_version to the adapter's
mtime.

Env: LORA_CKPT=<adapter> SRC=<source bank npz> OUT=<npz>. Runs on the GPU, ~5 min for 522k.
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import lora  # noqa: E402

LORA_CKPT = os.environ["LORA_CKPT"]
SRC = os.environ.get("SRC", os.path.join(HERE, "..", "..", "v6", "outputs",
                                         "concept_bank_l3_robot.npz"))
OUT = os.environ["OUT"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    d = np.load(SRC, allow_pickle=True)
    names = [str(n) for n in d["names"]]
    st = lora.load_st(DEVICE)
    lora.inject(st)
    lora.load_lora(st, LORA_CKPT)
    st.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(names), 512):
            out.append(lora.encode_grad(st, names[i:i + 512], DEVICE, max_len=64).cpu().numpy())
            if i % 51200 == 0:
                print(f"[reencode] {i}/{len(names)}", flush=True)
    emb = np.concatenate(out).astype(d["emb"].dtype)
    np.savez(OUT, emb=emb, names=d["names"], lora_version=os.path.getmtime(LORA_CKPT))
    print(f"[reencode] {emb.shape} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
