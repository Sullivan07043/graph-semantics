"""Re-encode the 521k decode dictionary through the L3 LoRA encoder.
The optimization space and the decode space must be the SAME space (version consistency);
output: outputs/concept_bank_l3.npz (emb + names + the lora checkpoint's mtime as version tag).
"""
import os
import sys
import time
import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import encode                                                         # noqa: E402
from pipeline_L3_v1 import lora                                       # noqa: E402

DEVICE = os.environ.get("DEVICE", "cuda")
CKPT = os.path.join(HERE, "outputs", "l3_lora.pt")
OUT = os.path.join(HERE, "outputs", "concept_bank_l3.npz")


def main():
    C, words = encode.load_dictionary()
    st = lora.load_st(DEVICE)
    lora.inject(st)
    lora.load_lora(st, CKPT)
    st.eval()
    out = np.zeros_like(C)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(words), 2048):
            out[i:i + 2048] = lora.encode_grad(st, words[i:i + 2048], DEVICE).cpu().numpy()
            if (i // 2048) % 32 == 0:
                print(f"[{time.strftime('%H:%M:%S')}] {i}/{len(words)} "
                      f"({(time.time()-t0)/60:.1f} min)", flush=True)
    shift = 1 - (out * C).sum(1)
    print(f"dictionary shift vs frozen: mean {shift.mean():.5f}, p99 {np.quantile(shift, .99):.5f}, "
          f"max {shift.max():.5f}", flush=True)
    np.savez(OUT, emb=out.astype(np.float32), names=np.array(words, dtype=object),
             lora_version=os.path.getmtime(CKPT))
    print(f"[saved {OUT}]", flush=True)


if __name__ == "__main__":
    main()
