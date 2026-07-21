"""Re-encode the full decode dictionary with the final, versioned L3 checkpoint."""
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import encode
from pipeline_L3_v1 import lora
from pipeline_v4 import release

DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
CKPT = os.environ.get(
    "L3_CKPT", os.path.join(HERE, "outputs", release.L3_CHECKPOINT_NAME))
OUT = os.environ.get(
    "L3_DICT", os.path.join(HERE, "outputs", release.L3_DICTIONARY_NAME))


def main():
    frozen, words = encode.load_dictionary()
    st = lora.load_st(DEVICE)
    lora.inject(st)
    lora.load_lora(st, CKPT)
    st.eval()
    out = np.zeros_like(frozen)
    started = time.time()
    with torch.no_grad():
        for i in range(0, len(words), 2048):
            out[i:i + 2048] = lora.encode_grad(
                st, words[i:i + 2048], DEVICE).cpu().numpy()
            if (i // 2048) % 32 == 0:
                print(f"[{time.strftime('%H:%M:%S')}] {i}/{len(words)} "
                      f"({(time.time() - started) / 60:.1f} min)", flush=True)
    shift = 1.0 - (out * frozen).sum(1)
    print(f"dictionary shift vs frozen: mean {shift.mean():.5f}, "
          f"p99 {np.quantile(shift, .99):.5f}, max {shift.max():.5f}", flush=True)
    ckpt_sha256 = lora.checkpoint_sha256(CKPT)
    temp = OUT + ".tmp.npz"
    np.savez(temp, emb=out.astype(np.float32), names=np.array(words, dtype=object),
             format=np.array(lora.DICTIONARY_FORMAT),
             version=np.array(lora.DICTIONARY_VERSION, dtype=np.int64),
             encoder=np.array(lora.ENCODER_NAME),
             lora_checkpoint_sha256=np.array(ckpt_sha256),
             shift_mean=np.array(float(shift.mean())),
             shift_p99=np.array(float(np.quantile(shift, .99))),
             shift_max=np.array(float(shift.max())))
    os.replace(temp, OUT)
    print(f"[saved {OUT}; l3={ckpt_sha256[:12]}]", flush=True)


if __name__ == "__main__":
    main()
