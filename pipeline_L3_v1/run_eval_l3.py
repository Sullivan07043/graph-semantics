"""Evaluate the final L3 encoder + L2 WeightNet under the official runners.

The L3 dictionary and L2 checkpoint are cryptographically tied to the same versioned LoRA
checkpoint.  API-free match/exact/cosine evaluation is obtained by leaving OPENAI_API_KEY unset.
"""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "pipeline_v3"))

from pipeline_v4 import release

DICT = os.environ.get(
    "L3_DICT", os.path.join(HERE, "outputs", release.L3_DICTIONARY_NAME))
CKPT = os.environ.get(
    "L3_CKPT", os.path.join(HERE, "outputs", release.L3_CHECKPOINT_NAME))
L2_CKPT = os.environ.get(
    "L2_CKPT", os.path.join(HERE, "outputs", release.l2_checkpoint_name("mlp")))
os.environ["GRAPHSEM_DICT"] = DICT

import numpy as np
import torch
import encode
from pipeline_L3_v1 import lora

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", 4)))
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
release_manifest = release.load_manifest(HERE)
release.verify_artifact(release_manifest, HERE, "l3_checkpoint", CKPT)
release.verify_artifact(
    release_manifest, HERE, "l3_dictionary", DICT,
    verify_sha256=os.environ.get("VERIFY_RELEASE_SHA256", "0") == "1")
ckpt_sha256 = lora.checkpoint_sha256(CKPT)
dictionary = np.load(DICT, allow_pickle=True)
try:
    dictionary_format = str(dictionary["format"])
    dictionary_version = int(dictionary["version"])
    dictionary_sha256 = str(dictionary["lora_checkpoint_sha256"])
except KeyError as exc:
    raise RuntimeError(
        "Incompatible legacy L3 dictionary: re-run pipeline_L3_v1/reencode_dict.py with the "
        "final versioned L3 checkpoint.") from exc
if dictionary_format != lora.DICTIONARY_FORMAT \
        or dictionary_version != lora.DICTIONARY_VERSION:
    raise RuntimeError(
        f"Incompatible L3 dictionary format/version: {dictionary_format!r}/"
        f"{dictionary_version}; expected {lora.DICTIONARY_FORMAT}/"
        f"{lora.DICTIONARY_VERSION}. Re-run reencode_dict.py.")
if dictionary_sha256 != ckpt_sha256:
    raise RuntimeError(
        "Dictionary was encoded with a different L3 checkpoint. Re-run "
        "pipeline_L3_v1/reencode_dict.py before evaluation.")
dictionary.close()

_st, _ = lora.install_as_encode_model(encode, CKPT, DEVICE)

# The adopted main line uses the same structured terms as L2 training.  Explicit caller values
# still win, but a plain invocation no longer silently drops residual, bridge, or negation terms.
os.environ.setdefault("NEGOP", "1")
os.environ.setdefault("RESIDUAL", "1")
os.environ.setdefault("LAM_RES", "1")
os.environ.setdefault("BRIDGE", "pearson")

ARM = os.environ.get("L2_ARM", "mlp")
if ARM == "mlp":
    import optimize
    from pipeline_v4 import core
    from pipeline_v4 import l2_modules as LM

    K = int(os.environ.get("K", release.SOLVER_STEPS))
    if K != release.SOLVER_STEPS:
        raise ValueError("Final main-line inference is fixed at K=120.")
    release.verify_artifact(release_manifest, HERE, "l2_checkpoint", L2_CKPT)
    _MODULE = LM.load(L2_CKPT, DEVICE,
                      expected_l3_sha256=ckpt_sha256)

    def _l2_solve(g, W, labeled_emb, d, steps=400, lr=2e-2, lam_zero=0.3,
                  lam_norm=0.1, seed=0, device="cpu", free_w=False, als_rounds=5,
                  residual=0.0, lam_res=0.0, partial_corr=None,
                  lam_dep=0.0, dep_corr=None, dep_kappa=0.5, lam_coll=0.0,
                  neg_op=None, bridge=None, gen_op=None, verbose=False,
                  n_samples=None, independent_info=None, item_info=None,
                  item_corr=None, residual_pair_info=None):
        if item_info is None:
            item_info = core.prepare_item_identity(
                g, labeled_emb, item_corr, n_samples, neg_op=neg_op, device=DEVICE)
        feats = torch.tensor(
            LM.node_features(g, W, set(labeled_emb), item_info=item_info,
                             independent_info=independent_info),
            dtype=torch.float32, device=DEVICE)
        embeddings, _ = core.solve_unrolled(
            g, W, labeled_emb, d, weight_module=_MODULE, K=K, inner_lr=2e-2,
            lam_zero=lam_zero, lam_norm=lam_norm, seed=seed, device=DEVICE,
            residual=residual, lam_res=lam_res, partial_corr=partial_corr,
            lam_dep=lam_dep, dep_corr=dep_corr, dep_kappa=dep_kappa,
            lam_coll=lam_coll, neg_op=neg_op, bridge=bridge,
            n_samples=n_samples, independent_info=independent_info, item_info=item_info,
            item_corr=item_corr, residual_pair_info=residual_pair_info,
            train=False, feats=feats)
        return embeddings

    optimize.optimize_embeddings = _l2_solve
elif ARM != "frozen":
    raise ValueError("L2_ARM must be 'mlp' or 'frozen'")

task = os.environ.get("TASK", "1")
if task == "1":
    import run_task1 as runner
else:
    import run_task2 as runner
runner.main()
