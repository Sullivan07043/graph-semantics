# External baseline code

Reusable implementations and experiment entry points live in this package.
No external-baseline implementation or runner is placed in the `v6` root.

| Baseline | Canonical implementation | Task 1 runner | Task 2 runner | OpenAI API |
|---|---|---|---|---|
| Feature Propagation | `feature_propagation.py` | `v6.baselines.runners.feature_propagation_task1` | `v6.baselines.runners.feature_propagation_task2` | No |
| GraphMAE-GCN | `graphmae_gcn.py` | `v6.baselines.runners.graphmae_task1` | `v6.baselines.runners.graphmae_task2` | No |
| CLIP-Dissect (E5 text adaptation) | `clip_dissect_e5.py` + `clip_dissect_bank.py` | `v6.baselines.runners.clip_dissect_task1` | `v6.baselines.runners.interpretability_task2 --baselines text-dissect` | No |
| Automated Interpretability | `automated_interpretability.py` | `v6.baselines.runners.llm_interpretability_task1 --baselines autointerp` | `v6.baselines.runners.interpretability_task2 --baselines autointerp` | Yes |
| Delphi | `delphi.py` | `v6.baselines.runners.llm_interpretability_task1 --baselines delphi` | `v6.baselines.runners.interpretability_task2 --baselines delphi` | Yes |

Run these commands with the repository environment; GraphMAE-GCN requires
PyTorch, while the CLIP-Dissect adaptation uses the project's frozen E5
encoder. Only Automated Interpretability and Delphi read `OPENAI_API_KEY`.
Their default model is `gpt-4o-mini`, and `--model` can override it.

Shared infrastructure is isolated in `api.py` and `protocol.py`. Automated
Interpretability and Delphi share private validation and sampling code in
`_llm_interpretability.py`; their public imports are separated so callers do
not need to depend on the combined implementation module.

CLI commands use `python -m` before the runner module shown above. The internal
method key `text-dissect` is retained only in result artifacts and cache keys to
keep previous runs reproducible; the report-facing name is CLIP-Dissect E5.
