# Archive — superseded code, kept intact

Everything here is preserved exactly as it was when v5 was assembled (2026-07-22).
These files reference the OLD repository layout (root-level modules, outputs/ at repo root),
so they are NOT runnable in place. To run any archived code, check out the complete snapshot:

    git checkout pre-v5        # tag: last commit before the v5 restructure

Contents:
- pipeline_v3/  — bridge acceptance experiments, g_phi (rejected operator); dependence.py moved to v5
- pipeline_v4/  — L2 development (core/l2_modules/l2_train moved to v5; run_eval.py = frozen-space eval)
- pipeline_L3_v1/ — L3 development (active parts moved to v5; l2_retrain_lora.py = rejected retrain)
- experiments/  — swap interventions, ablation-era scripts, bake-off, forward-beta
- gnn.py        — the paused GNN completion line (swap-intervention reference model)
