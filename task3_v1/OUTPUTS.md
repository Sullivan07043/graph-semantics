# Task 3 artifact layout

## Results

- `outputs/preflight/`: J-lens and intervention preflight checks.
- `outputs/calibration/`: 32-concept write calibration and row-level records.
- `outputs/discovery/`: raw discovery matrices and innovation-residual matrices.
- `outputs/causcale/`: CauScale smoke tests and bootstrap candidate graphs.
- `outputs/validation/`: held-out and targeted intervention results.
- `outputs/semantic/`: direct J-space semantic-completion diagnostics.
- `outputs/paper_aligned/`: separate paper-aligned Qwen read-and-swap substrate check.

The current Stage 1 result chain is:

1. `outputs/discovery/innovation_matrix_1000x128.*`
2. `outputs/causcale/innovation_causcale_bootstrap_20.*`
3. `outputs/validation/innovation_heldout_graph_validation.*`

The current direct semantic diagnostic is:

- `outputs/semantic/jspace_semantic_direct_smoke.json`

Files without the `innovation_` prefix are retained as earlier raw-coordinate
comparisons.

## Logs

- `logs/calibration/`: write-calibration standard output and error logs.
- `logs/semantic/`: direct semantic-test standard output and error logs.
- `logs/paper_aligned/`: prompt-preparation and paper-aligned runner logs.

Logs are operational records. Numerical results and experiment metadata remain
under `outputs/`.
