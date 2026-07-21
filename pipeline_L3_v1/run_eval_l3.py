"""L3 evaluation under the official protocol, API-FREE by default (match/exact only; judge runs
only when OPENAI_API_KEY is set — final-candidate stage, user-approved spend).

Reuses run_task1/run_task2 unchanged. Swaps, before the runners import anything:
  encode._MODEL     -> the LoRA-adapted SentenceTransformer (encode.embed then uses it as-is)
  GRAPHSEM_DICT env -> outputs/concept_bank_l3.npz (decode space = optimization space; version
                       asserted against the lora checkpoint)
  optionally the solver -> L2 WeightNet unrolled solver (L2_ARM=mlp), same as pipeline_v4/run_eval.

Env: TASK=1|2, DATASET, RECORDS_OUT, L2_ARM=frozen|mlp (frozen = 400-step official solver).
"""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "pipeline_v3"))

DICT = os.path.join(HERE, "outputs", "concept_bank_l3.npz")
os.environ["GRAPHSEM_DICT"] = DICT

import numpy as np                                                    # noqa: E402
import torch                                                          # noqa: E402
import encode                                                         # noqa: E402
from pipeline_L3_v1 import lora                                       # noqa: E402

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", 4)))
CKPT = os.path.join(HERE, "outputs", "l3_lora.pt")
_v = np.load(DICT, allow_pickle=True)
assert abs(float(_v["lora_version"]) - os.path.getmtime(CKPT)) < 1.0, \
    "dictionary was encoded with a DIFFERENT lora checkpoint — re-run reencode_dict.py"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_st = lora.load_st(DEVICE)
lora.inject(_st)
lora.load_lora(_st, CKPT)
_st.eval()


class _LoraST:
    """Duck-types the bit of SentenceTransformer that encode.embed uses."""

    def encode(self, texts, batch_size=1024, normalize_embeddings=True):
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), 256):
                # encode.embed already prepended the prefix; encode_grad adds it too -> strip here
                stripped = [t[len("query: "):] if t.startswith("query: ") else t
                            for t in texts[i:i + 256]]
                out.append(lora.encode_grad(_st, stripped, DEVICE, max_len=128).cpu().numpy())
        return np.concatenate(out)


encode._MODEL = _LoraST()

ARM = os.environ.get("L2_ARM", "frozen")
if ARM == "mlp":
    import optimize                                                   # noqa: E402
    from pipeline_v4 import core, l2_modules as LM                    # noqa: E402
    _MODULE = LM.load(os.environ.get("L2_CKPT", os.path.join(HERE, "outputs", "l2_mlp.pt")))
    K = int(os.environ.get("K", 60))

    def _l2_solve(g, W, labeled_emb, d, steps=400, lr=2e-2, lam_zero=0.3, lam_norm=0.1,
                  seed=0, device="cpu", free_w=False, als_rounds=5,
                  residual=0.0, lam_res=0.0, partial_corr=None,
                  lam_dep=0.0, dep_corr=None, dep_kappa=0.5, lam_coll=0.0,
                  neg_op=None, bridge=None, gen_op=None, verbose=False):
        feats = torch.tensor(LM.node_features(g, W, set(labeled_emb)), device=device)
        emb, _ = core.solve_unrolled(
            g, W, labeled_emb, d, weight_module=_MODULE, K=K, inner_lr=2e-2,
            lam_zero=lam_zero, lam_norm=lam_norm, seed=seed, device=device,
            residual=residual, lam_res=lam_res, partial_corr=partial_corr,
            neg_op=neg_op, bridge=bridge, train=False, feats=feats)
        return emb

    optimize.optimize_embeddings = _l2_solve

task = os.environ.get("TASK", "1")
if task == "1":
    import run_task1 as runner                                        # noqa: E402
else:
    import run_task2 as runner                                        # noqa: E402
runner.main()
