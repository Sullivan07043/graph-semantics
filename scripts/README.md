# Experiment Utilities

Run these commands from the repository root so data, cache, and output paths
resolve consistently.

| Script | Purpose |
|---|---|
| `make_oracle_datasets.py` | Generate the controlled oracle datasets under `data/oracle_*`. |
| `run_task1_ablation.py` | Run the real-data Task 1 constraint sweep and per-item diagnostics. |
| `run_oracle_diagnostics.py` | Evaluate selected Task 1 configurations on the oracle datasets. |
| `run_polarity_ablation.py` | Run the focused polarity-aware oracle comparison. |

```text
python scripts/make_oracle_datasets.py
python scripts/run_task1_ablation.py
python scripts/run_oracle_diagnostics.py
python scripts/run_polarity_ablation.py
```

The stable user-facing Task 1 and Task 2 entry points remain `run_task1.py` and
`run_task2.py` in the repository root. Experiment utilities keep their existing
environment variables and write to the repository-level `outputs/` directory by
default.
