"""Evaluate the L2 solver through the official task runners.

This frozen-encoder reference entry point uses the same K=120 numerical solve as the main line.
The final adopted L3+L2 path is ``pipeline_L3_v1/run_eval_l3.py``.
"""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "pipeline_v3"))

import torch
import optimize
from pipeline_v4 import core
from pipeline_v4 import l2_modules as LM
from pipeline_v4 import release

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", 4)))
ARM = os.environ.get("L2_ARM", "mlp")
K = int(os.environ.get("K", release.SOLVER_STEPS))
INNER_LR = float(os.environ.get("INNER_LR", 2e-2))
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
if K != release.SOLVER_STEPS:
    raise ValueError("The main-line L2 inference budget is fixed at K=120.")

_MODULE = None
if ARM in ("static", "mlp"):
    checkpoint = os.environ.get("L2_CKPT")
    if not checkpoint:
        raise RuntimeError(
            "pipeline_v4/run_eval.py is the frozen-E5 reference path and has no adopted "
            "v4.1 WeightNet checkpoint. Set L2_CKPT to an explicitly compatible frozen-space "
            "checkpoint, use L2_ARM=mult1, or run the v4.1 main-line entry point "
            "pipeline_L3_v1/run_eval_l3.py.")
    _MODULE = LM.load(checkpoint, DEVICE)
elif ARM != "mult1":
    raise ValueError("L2_ARM must be mult1, static, or mlp")


def _l2_solve(g, W, labeled_emb, d, steps=400, lr=2e-2, lam_zero=0.3, lam_norm=0.1,
              seed=0, device="cpu", free_w=False, als_rounds=5,
              residual=0.0, lam_res=0.0, partial_corr=None,
              lam_dep=0.0, dep_corr=None, dep_kappa=0.5, lam_coll=0.0,
              neg_op=None, bridge=None, gen_op=None, verbose=False,
              n_samples=None, independent_info=None, item_info=None,
              item_corr=None, residual_pair_info=None):
    if item_info is None:
        item_info = core.prepare_item_identity(
            g, labeled_emb, item_corr, n_samples, neg_op=neg_op, device=DEVICE)
    feats = None
    if _MODULE is not None:
        feats = torch.tensor(
            LM.node_features(g, W, set(labeled_emb), item_info=item_info,
                             independent_info=independent_info),
            dtype=torch.float32, device=DEVICE)
    embeddings, _ = core.solve_unrolled(
        g, W, labeled_emb, d, weight_module=_MODULE, K=K, inner_lr=INNER_LR,
        lam_zero=lam_zero, lam_norm=lam_norm, seed=seed, device=DEVICE,
        residual=residual, lam_res=lam_res, partial_corr=partial_corr,
        lam_dep=lam_dep, dep_corr=dep_corr, dep_kappa=dep_kappa, lam_coll=lam_coll,
        neg_op=neg_op, bridge=bridge, n_samples=n_samples,
        independent_info=independent_info, item_info=item_info,
        item_corr=item_corr, residual_pair_info=residual_pair_info,
        train=False, feats=feats)
    return embeddings


optimize.optimize_embeddings = _l2_solve

task = os.environ.get("TASK", "1")
if task == "1":
    import run_task1 as runner
else:
    import run_task2 as runner
runner.optimize = optimize
runner.main()
