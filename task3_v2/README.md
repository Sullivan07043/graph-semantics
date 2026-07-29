# Task 3 V2

Status: **E0″ complete — F (`broader_solver_or_metric_failure`); Task 3 is
paused for solver/evaluation audit**. E0′ remains NO-GO, and neither old E1 nor
S0 is allowed. The frozen results were not used to tune the solver, graphs,
labels, seeds, or objective coefficients.

V2 redesigns Task 3 around the same core problem as Tasks 1/2: given a
graph and partial semantic labels, recover the remaining node meanings. The
fixed- or controlled-graph semantic-recovery experiment should be primary.
J-space causal discovery, if retained, should be a separate optional diagnostic
rather than a prerequisite.

## Current experiment: E0′ — Oracle Causal-Graph Bridge Test

E0′ asks whether the frozen Stage-3 Task 1 graph-semantic solver still improves
masked semantic recovery when it is transferred from psychometric,
scoring-key, and hierarchy graphs to three fixed directed causal DAGs with
natural semantic labels. The primary comparison uses the oracle DAG structure
with data-estimated edge weights; shuffled, reversed, raw-correlation, and
uniform/no-structure arms use the same data, folds, encoder, dictionary,
decoder, and seeds.

This experiment does not use an LLM, J-space, CauScale, activation
interventions, latent-node discovery, or Task 2 latent translation. The
experiment definition and frozen inputs live in
[`experiments/e0_oracle_bridge/`](experiments/e0_oracle_bridge/); its primary
report is
[`experiments/e0_oracle_bridge/results/report.md`](experiments/e0_oracle_bridge/results/report.md).

## Historical branch

The original token-anchored J-space Stage 1 is preserved in
[`../task3_v1/`](../task3_v1/) as an **independent, paused historical branch**.
Its NO-GO result blocks causal interpretation of that discovered graph. It did
not determine E0′; E0′ was run independently and reached the separate NO-GO
reported above. V1 code, artifacts, thresholds, reports, and numerical results
remain available for audit.

## Experiment index: E0′ and E0″

The canonical experiment names and current decisions are:

- **E0′ — Oracle Causal-Graph Bridge Test:** frozen oracle-DAG transfer test,
  completed with **NO-GO**. See the
  [experiment definition](experiments/e0_oracle_bridge/) and
  [formal report](experiments/e0_oracle_bridge/results/report.md). This result
  did not authorize the old E1.
- **E0″ — Orientation and Constraint Audit:** frozen explanation audit for the
  E0′ NO-GO, completed with **F — `broader_solver_or_metric_failure`**. The orientation
  interface passed, selected Stage-3 bundle behavior reproduced, and no E0′
  rerun was required. Because the same-module positive control failed and local
  metrics conflict materially, Task 3 is paused for solver/evaluation audit.
  The old E1 and S0 are both disallowed; Judge remains pending. See the
  [complete audit specification](experiments/e0_orientation_constraint_audit/TASK3_E0_AUDIT.md),
  [ten-section report](experiments/e0_orientation_constraint_audit/results/report.md),
  and
  [machine-readable decision](experiments/e0_orientation_constraint_audit/results/decision.json).

E0″ preserves E0′ labels, graphs, SCM data, folds, seeds, Stage-3 artifacts,
objective coefficients, candidate dictionary, and decoder. It is a diagnostic
continuation, not a tuned rerun or a new causal-discovery stage.