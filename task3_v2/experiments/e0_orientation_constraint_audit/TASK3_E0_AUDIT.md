# Task 3 E0″ — Orientation and Constraint Audit

Status: **formal run complete**. The frozen audit classified the E0′ failure as
**F — `broader_solver_or_metric_failure`**. The orientation interface passed, the
selected Stage-3 bundle behavior reproduced, and no E0′ rerun is required.
The old E1 remains blocked; the S0 semantic-support benchmark is also not
authorized by this result.

The concise numerical report is [`results/report.md`](results/report.md), the
machine-readable verdict is [`results/decision.json`](results/decision.json),
and the frozen protocol is [`config.yaml`](config.yaml).

## 1. Purpose and scope

E0″ explains the E0′ NO-GO; it does not replace or tune E0′. It asks the
following diagnostic questions:

1. Did an edge-orientation or source/target interface error invalidate E0′?
2. Did the frozen Stage-3 checkpoint/bundle cease to reproduce its known
   qualitative behavior?
3. Is the failure concentrated at roots or nodes without a visible structural
   anchor?
4. Is the transferred generation constraint the failing objective component?
5. Does a semantic-support graph succeed where the causal DAG fails?
6. If none of those explanations is sufficient, is this a broader
   solver/metric failure?

The audit does **not** run J-space, CauScale, activation interventions,
latent-node discovery, Task-2 translation, or the old E1. It does not train or
modify a semantic encoder, LoRA adapter, WeightNet, negation operator, concept
bank, graph, fold, label, seed, objective coefficient, or decoder.

## 2. Frozen inheritance from E0′

The source experiment is
[`../e0_oracle_bridge/`](../e0_oracle_bridge/), with source configuration
[`../e0_oracle_bridge/config.yaml`](../e0_oracle_bridge/config.yaml) at frozen
SHA-256
`bc8004a8d288d8a3bddd00bef12c44eab26d372e276a106f4c95340a0d0a8a44`.
Its required input decision is NO-GO.

E0″ reuses without modification:

- the three 20-node oracle causal DAGs and their natural-language labels;
- the linear additive Gaussian SCM data, graph seeds, data seeds, and split
  assignments;
- 2,000 samples per graph, split into 1,200 train, 400 development, and 400
  test samples;
- train-fitted standardization and train/development-only edge estimation;
  the test split is never used for model selection or estimation;
- five fixed masking folds per graph, four masked nodes per fold, so each of
  the 60 graph-local nodes is evaluated once;
- the E0′ data-estimated weights, raw-correlation baseline, uniform baseline,
  20 fixed shuffled-graph permutations per graph, and all random seeds;
- oracle-estimated-weight ALS masked-node initialization, identical for every
  audit arm;
- the all-20-gold-label candidate set within each graph and the frozen Stage-3
  splice decoder.

The frozen Stage-3 runtime is:

| component | frozen value |
| --- | --- |
| encoder | `intfloat/e5-large-v2`, snapshot `f169b11e22de13617baa190a028a32f3493550b6` |
| calibrated space | LoRA rank 8, alpha 16, last 2 layers |
| LoRA SHA-256 | `d90b024e7fb030e3ee1545c8d19606cf032cf810fc7f4a758749dfefc95a49d5` |
| WeightNet SHA-256 | `70ffc4fcf668b57d943240fde67a8b339c702fa0093a5594f34e3297a5c50bfd` |
| negation operator SHA-256 | `6f30f0d68ee653d52aef93bdae97feeac8abd17189e688ef49f594e18574690e` |
| concept bank SHA-256 | `6da2de255dcb2fa559fa1c2a8bfba25fd0e4fcfecb5c768ecf229b6ce4e7bb9e` |
| solver | 60-step functional Adam, learning rate 0.02, CPU |
| objective | `lam_zero=0.3`, `lam_norm=0.1`, residual and `lam_res` both 1.0 |
| bridge | absolute train/development Pearson, `lam_upper=0.3`, `kappa=0.5`, `q=0.7` |

At protocol freeze, the worktree commit was
`02f8112f88cc1dddc22d2b445fbe1a14480be542`, the latest-main code authority
was `8d58ee99855dbe7a44c26a2b5c8642d01ba736ac`, and the `v5` tree was verified
byte-equivalent to that authority. Main was used read-only; no checkout or
merge was needed.

## 3. Fail-closed orientation gate

The executable orientation audit is
[`../../scripts/e0_orientation_audit.py`](../../scripts/e0_orientation_audit.py);
its derivation is documented in
[`orientation_unit_test.md`](orientation_unit_test.md).

The gate traces JSON `source → target` through:

- source-row/target-column adjacency;
- the E0′ ordered-edge adapter;
- `Graph.parents` and `Graph.children`;
- SCM generation and edge-weight estimation;
- ALS initialization; and
- the frozen Stage-3 generation loss and gradients.

The fixed probe is `A --0.7→ B --0.4→ C`, with
`z_A=(1,0)`, `z_B=(0,0)`, and `z_C=(0,1)`. It requires:

```text
L_gen = ||z_B - 0.7 z_A||² + ||z_C - 0.4 z_B||² = 1.49
dL/dz_B = (-1.4, -0.8)
```

Thus B receives an incoming-equation pull toward A and a downstream-equation
pull toward C. That behavior is the derivative of the frozen quadratic
compatibility objective, not a transposed edge interface. A deliberately
transposed `A ← B ← C` fixture must be rejected, and unit-weight forward and
fully reversed chains must expose their equal generation energy/gradient.

The formal gate passed. All 15 graph-by-fold canonical full solves matched the
unmodified E0′ solver, and all 60 full-oracle node metrics matched E0′ within
the frozen tolerances. Consequently:

- `orientation_interface_bug=false`;
- no orientation fix was applied;
- an orientation-triggered E0′ rerun is forbidden.

Structured evidence is in
[`results/orientation_audit.json`](results/orientation_audit.json) and
[`results/parity_gate.json`](results/parity_gate.json).

## 4. Structural strata

Every masked node is assigned to the following preregistered groups. Groups
overlap intentionally.

| group | definition | observed nodes |
| --- | --- | ---: |
| `root` | no oracle-DAG parent | 18 |
| `non_root` | at least one oracle-DAG parent | 42 |
| `non_root_visible_parent` | non-root with at least one unmasked/visible parent in its fold | 42 |
| `no_visible_parent_visible_child` | no visible parent but at least one visible child | 17 |
| `no_visible_structural_anchor` | neither a visible parent nor a visible child | 1 |

For the frozen causal objective, a *structural anchor* means a visible parent
or visible child. Visible same-module nodes are recorded separately and are
not retroactively counted as causal anchors.

The full oracle arm is paired within node against `reversed_full`,
`shuffled_full`, `uniform`, and `raw_correlation`. Motif annotations allow
multiple roles (`chain`, `fork`, `collider`, `mediator`, `other`) while the
single primary role uses precedence
`collider > fork > mediator > chain > other`.

## 5. Constraint-decomposition arms

For optimizer arms, only six registered term switches change. Their fixed
order is:

```text
generation, residual_norm, residual_alignment, independence, bridge, norm
```

| arm | term mask | graph |
| --- | --- | --- |
| `full_oracle` | `1 1 1 1 1 1` | directed oracle DAG |
| `generation_only_oracle` | `1 0 0 0 0 0` | directed oracle DAG |
| `oracle_without_generation` | `0 1 1 1 1 1` | directed oracle DAG |
| `residual_only_oracle` | `0 1 1 0 0 0` | directed oracle DAG |
| `independence_only_oracle` | `0 0 0 1 1 0` | directed oracle DAG |
| `symmetrized_oracle` | `1 1 1 1 1 1` | bidirected oracle skeleton |
| `markov_blanket_oracle` | `1 1 1 1 1 1` | bidirected parents/children/co-parents |
| `same_module_graph` | `1 1 1 1 1 1` | bidirected within-module cliques |
| `reversed_full` | `1 1 1 1 1 1` | all oracle edges reversed |
| `shuffled_full` | `1 1 1 1 1 1` | 20 fixed graph-support relabelings per graph; node/data/semantic identities stay fixed |
| `raw_correlation` | baseline | frozen E0′ closed-form baseline |
| `uniform` | baseline | frozen E0′ no-structure baseline |

All optimizer arms begin from the exact same masked-node ALS vectors and use
a common, node-ID-mapped residual initialization. Baselines report the same
common ALS initial vector and their deterministic closed-form final vector.

Two intentional edge cases are diagnostic:

- `generation_only_oracle` retains the residual inside the generation
  equation while disabling its penalties. This degeneracy is measured, not
  silently repaired.
- `residual_only_oracle` disconnects the active residual penalties from the
  semantic embeddings. Zero semantic-embedding gradient and displacement are
  therefore expected.

The three bidirected diagnostic adapters are not causal DAGs. Symmetric weights
are absolute train/development Pearson correlations assigned identically in
both directions. Under these cyclic adapters, the frozen `v5`
ancestor/trek/independence machinery reduces to connected-component
reachability. These arms are structural/semantic diagnostics and must not be
given a causal interpretation.

## 6. Metrics and paired inference

Each node/arm record contains:

- gold cosine;
- graph-centered cosine, using the centroid of all 20 gold-label embeddings;
- prediction margin;
- mean reciprocal rank (MRR);
- recall at 1;
- recall at 5;
- Match accuracy; and
- exact accuracy.

The frozen decoder fits its alpha using visible labels in the current fold
only. Primary classification never relies on aggregate raw cosine alone; it
combines semantic metrics (`gold_cosine`, `centered_cosine`,
`prediction_margin`) with retrieval metrics (`mrr`, `recall_at_5`,
`match_acc`).

Every paired contrast uses a fixed-seed, **10,000-draw hierarchical bootstrap**
in the order:

```text
graph → fold → masked node
```

For a shuffled comparison, each leaf is first averaged over its 20 frozen
permutations and is then paired to the same graph/fold/node. A positive cell is
*supported* only when the row is complete, its paired mean is above zero, and
the lower bound of its 95% interval is also above zero. The bootstrap seed is
`88173`.

Raw paired leaves are in
[`results/paired_deltas.csv`](results/paired_deltas.csv); intervals are in
[`results/bootstrap_summary.csv`](results/bootstrap_summary.csv).

## 7. Frozen A–F decision map

The conditions are evaluated together. A and B are validity confounds. For
C–F, a supported positive cell is not a stable advantage when the same
comparison also has supported adverse evidence; a failed same-module positive
control or material cross-metric conflict blocks C and selects F.

| category | frozen trigger | required action |
| --- | --- | --- |
| **A — orientation interface bug** | the fail-closed orientation audit reports an interface bug | repair only that interface and rerun frozen E0′ once |
| **B — checkpoint/bundle drift** | the selected bundle replication does not preserve its frozen qualitative behavior | restore a behaviorally trusted Stage-3 bundle, then rerun frozen E0′ |
| **C — root/anchor boundary** | failure is concentrated more strongly at roots, while visible-parent non-roots have stable support across semantic and retrieval families on at least two baselines, without supported adverse cells; the same-module positive control must also pass | redefine the task boundary around observable structural anchors; do not enter old E1 |
| **D — generation-constraint mismatch** | `oracle_without_generation − full_oracle` is supported on at least 2 semantic/retrieval metrics and `full_oracle − generation_only` is supported on at least 2 | close unchanged generation transfer and redesign the semantic constraint before another benchmark |
| **E — causal-graph/semantic-constraint mismatch** | orientation and bundle pass; visible-parent support is below 3; full causal semantic support against shuffle/uniform is below 2; and same-module beats both shuffle and uniform on at least 2 semantic/retrieval metrics each | close generic causal-DAG transfer; S0 semantic-support benchmark may proceed |
| **F — broader solver/metric failure** | the same-module positive control is not sufficiently supported, or local metrics contain material supported conflicts | pause Task 3 and audit solver/evaluation |

Only A or B can require an E0′ rerun. The old E1 is disallowed for every
category. S0 is allowed only under E with a passing bundle and same-module
positive control.

## 8. Formal outcome

The executed decision was **F — `broader_solver_or_metric_failure`**:

- orientation interface bug: false;
- selected bundle/checkpoint drift: false;
- root/anchor boundary rule: false;
- generation mismatch rule: false;
- causal-graph/semantic-constraint mismatch rule: false;
- same-module multi-metric positive diagnostic: false;
- material cross-metric conflict: true;
- E0′ rerun required: false;
- old E1 allowed: false;
- S0 allowed: false.

There were 18 roots and 42 non-roots. Roots contributed about 83.7% of the
summed reversed-minus-oracle gold-cosine advantage. The 42 visible-parent
non-roots still showed a metric conflict: full oracle lost gold cosine to
uniform/raw-correlation while improving centered cosine and retrieval measures
such as MRR and recall at 5. Only one visible-parent baseline comparison met the stable cross-family rule,
while two contained both supported positive and supported adverse metrics.
The same-module control had only one supported cell versus shuffle and also
contained adverse gold-cosine/margin evidence versus uniform. Roots remain an
important stratified finding, but they do not sufficiently explain the full
failure; these conflicts produce category F.

The first generated report incorrectly labeled this pattern C by accumulating
positive cells while ignoring supported adverse cells. QA corrected that
report-layer consistency bug and refinalized `decision.json`, `report.md`,
`provenance.json`, and `run_manifest.json` from the completed CSV/NPZ results.
No solver, decoder, bootstrap, graph, or checkpoint was rerun or changed.

The bundle probe reproduced the required qualitative trends without retraining:

- Himi Match: core `0.800` > raw correlation `0.767` > uniform `0.383`;
- Kims Match: raw correlation `0.900` > core `0.693` > uniform `0.154`;
- BigFive2 hierarchy Match: latent constraints on `0.620` versus off `0.540`
  (delta `+0.080`).

See [`bundle_replication.md`](bundle_replication.md) and
[`results/bundle_replication.json`](results/bundle_replication.json).

## 9. API-free Judge status

The formal local run did not call a Judge API and did not fabricate or impute a
Judge score. Missing verdicts are **pending**, never scored as zero:

- [`results/judge_requests.jsonl`](results/judge_requests.jsonl) contains
  1,860 unique E0″ requests;
- [`results/bundle_judge_requests.jsonl`](results/bundle_judge_requests.jsonl)
  contains 268 bundle-replication requests.

Judge accuracy and Judge trend are therefore not evaluable in this API-free
run and do not enter the A–F decision. The local cosine, ranking, decoder, loss,
gradient, parity, and bootstrap outputs are complete.

## 10. Reproduction commands

Run from the repository root in this order:

```powershell
# Orientation audit and targeted tests
.\.venv\Scripts\python.exe task3_v2\scripts\e0_orientation_audit.py
.\.venv\Scripts\python.exe -m unittest discover -s task3_v2\tests -p test_e0_orientation.py -v

# Optional fail-fast validation of frozen files and adapters
.\.venv\Scripts\python.exe task3_v2\scripts\run_e0_audit.py --config task3_v2\experiments\e0_orientation_constraint_audit\config.yaml --validate-only

# Required behavior probe, then the formal audit
.\.venv\Scripts\python.exe task3_v2\scripts\run_e0_bundle_replication.py
.\.venv\Scripts\python.exe task3_v2\scripts\run_e0_audit.py --config task3_v2\experiments\e0_orientation_constraint_audit\config.yaml

# Report-only QA refinalization; reuses completed CSV/NPZ outputs
.\.venv\Scripts\python.exe task3_v2\scripts\run_e0_audit.py --config task3_v2\experiments\e0_orientation_constraint_audit\config.yaml --refinalize-existing

# Complete Task-3-v2 unit suite
.\.venv\Scripts\python.exe -m unittest discover -s task3_v2\tests -v
```

`--skip-decode` is debug-only and does not produce a valid formal E0″ result.
`--refinalize-existing` refuses debug outputs and does not rerun a solver,
decoder, or bootstrap.

## 11. Artifact index

Protocol and implementation:

- [`config.yaml`](config.yaml) — frozen human-readable protocol;
- [`../../scripts/e0_orientation_audit.py`](../../scripts/e0_orientation_audit.py)
  — orientation gate;
- [`../../scripts/e0_audit_solver.py`](../../scripts/e0_audit_solver.py) —
  term-mask audit solver;
- [`../../scripts/run_e0_bundle_replication.py`](../../scripts/run_e0_bundle_replication.py)
  — API-free selected-bundle replication;
- [`../../scripts/run_e0_audit.py`](../../scripts/run_e0_audit.py) — formal
  runner;
- [`../../tests/test_e0_orientation.py`](../../tests/test_e0_orientation.py),
  [`../../tests/test_e0_audit_solver.py`](../../tests/test_e0_audit_solver.py),
  and [`../../tests/test_e0_audit_decision.py`](../../tests/test_e0_audit_decision.py)
  — executable unit coverage.

Primary outputs:

- [`results/report.md`](results/report.md) — ten-section human-readable report;
- [`results/decision.json`](results/decision.json) — A–F verdict;
- [`results/per_node_audit.csv`](results/per_node_audit.csv) — 1,860 node/arm
  rows;
- [`results/per_group.csv`](results/per_group.csv) and
  [`results/per_arm.csv`](results/per_arm.csv) — aggregates;
- [`results/loss_terms.csv`](results/loss_terms.csv) and
  [`results/gradient_norms.csv`](results/gradient_norms.csv) — decomposition
  diagnostics;
- [`results/embeddings.npz`](results/embeddings.npz) — initial/final float32
  embeddings, shape `(1860, 1024)`;
- [`results/graph_arm_metadata.json`](results/graph_arm_metadata.json) —
  graph-adapter and cyclic-graph metadata;
- [`results/provenance.json`](results/provenance.json) — hashes, commands,
  row counts, split-use assertions, and parity summary;
- [`results/run_manifest.json`](results/run_manifest.json) — output hashes.

## 12. Interpretation limits

- The test covers three synthetic linear-Gaussian, 20-node DAGs. It is not a
  claim about all causal graphs, nonlinear SCMs, or unrestricted natural
  language semantics.
- The candidate set contains the 20 gold labels within each graph. Retrieval
  results do not establish open-vocabulary recovery.
- The completely unanchored stratum has only one node; it must not carry an
  independent population-level conclusion.
- Generation energy propagates through both endpoints and has weak orientation
  identifiability at unit weights even though the software interface is
  correctly oriented.
- Semantic, centered, margin, and retrieval metrics disagree in important
  strata. The frozen multi-metric rule preserves that disagreement instead of
  selecting a favorable metric after seeing results.
- The same-module positive control failed its preregistered multi-metric rule,
  and its bidirected graph has no causal interpretation.
- `generation_only` and `residual_only` intentionally expose degeneracy and
  disconnection; they are diagnostic ablations, not proposed solvers.
- The original reported release bundle is absent. The local artifacts had been
  retrained before E0′, so provenance is limited even though the selected
  behavioral orderings reproduced and no artifact was retrained in E0″.
- Judge remains pending in this API-free environment. Any later Judge results
  must be attached to the existing request IDs and reported separately; they
  must not rewrite the frozen local A–F decision.
