"""L2 evaluation under the OFFICIAL protocol. Reuses run_task1/run_task2 unchanged (their
run_dataset, folds, judge, records) by swapping optimize.optimize_embeddings for the L2 unrolled
solver. This keeps every number directly comparable with the frozen tables (t1final/t2final).

Env: L2_ARM = mult1 | static | mlp   (mult1 = no module: solver-dynamics control)
     K (60), INNER_LR (2e-2), TASK = 1 | 2, DATASETS (passed through to the runner)
Outputs land in outputs/ as t1l2<arm>_*.json / t2l2<arm>_*.json (runner OUT_PREFIX).
"""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "pipeline_v3"))

import torch                                                          # noqa: E402
import optimize                                                       # noqa: E402
from pipeline_v4 import core, l2_modules as LM                        # noqa: E402

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", 4)))
ARM = os.environ.get("L2_ARM", "mlp")
K = int(os.environ.get("K", 60))
INNER_LR = float(os.environ.get("INNER_LR", 2e-2))

_MODULE = None
if ARM in ("static", "mlp"):
    _MODULE = LM.load(os.path.join(HERE, "outputs", f"l2_{ARM}.pt"))


def _l2_solve(g, W, labeled_emb, d, steps=400, lr=2e-2, lam_zero=0.3, lam_norm=0.1,
              seed=0, device="cpu", free_w=False, als_rounds=5,
              residual=0.0, lam_res=0.0, partial_corr=None,
              lam_dep=0.0, dep_corr=None, dep_kappa=0.5, lam_coll=0.0,
              neg_op=None, bridge=None, gen_op=None, verbose=False):
    """Drop-in replacement for optimize.optimize_embeddings (steps/lr/free_w/gen_op ignored:
    the L2 solver defines its own dynamics; free_w and gen_op are not part of the frozen config)."""
    feats = None
    if _MODULE is not None:
        feats = torch.tensor(LM.node_features(g, W, set(labeled_emb)), device=device)
    emb, _ = core.solve_unrolled(
        g, W, labeled_emb, d, weight_module=_MODULE, K=K, inner_lr=INNER_LR,
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
runner.optimize = optimize    # runner already imported optimize by reference; keep both aligned
runner.main()
