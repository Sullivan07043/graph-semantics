# 2026-07-06 Constraint Ablation

## Purpose

Task 1 baseline results showed that the original graph-constrained `core` method was unstable:

- It worked reasonably on Himi judge-ACC.
- It matched raw correlation on TLVD matching-ACC but did not beat it on judge-ACC.
- It failed badly on Big Five, especially compared with `rawcorr`.

This experiment tested whether changing graph-derived constraints improves observed-variable semantic
completion.

## Code Changes

Added constraint controls:

| Env Var | Values | Effect |
|---|---|---|
| `EDGE_WEIGHT_MODE` | `signed`, `abs`, `positive` | Converts data correlations into semantic edge weights. |
| `NORMALIZE_GEN` | `0`, `1` | If enabled, parent generation uses weighted average instead of raw weighted sum. |
| `LAM_OBS_PRIOR` | float | Adds an observed-variable prior from visible labels. |
| `OBS_PRIOR_SCOPE` | `siblings`, `all` | Selects which visible observed variables can form the prior. |

New constraint terms:

```text
e_n =
  fixed label embedding a_n, if n is a visible observed variable
  optimized embedding z_n, otherwise

w'_{p,n} =
  w_{p,n},        if EDGE_WEIGHT_MODE=signed
  |w_{p,n}|,      if EDGE_WEIGHT_MODE=abs
  max(w_{p,n},0), if EDGE_WEIGHT_MODE=positive

gen(n) =
  sum_{p in Pa(n)} w'_{p,n} e_p,                                      if NORMALIZE_GEN=0
  sum_{p in Pa(n)} w'_{p,n} e_p / (sum_{p in Pa(n)} |w'_{p,n}| + eps), if NORMALIZE_GEN=1
```

Original optimization objective:

```text
L_original =
  sum_{n: Pa(n) != empty, n visible} || gen(n) - a_n ||_2^2
  + sum_{n: Pa(n) != empty, n free} || z_n - gen(n) ||_2^2
  + LAM_ZERO / |I| * sum_{(u,v) in I} cos(e_u, e_v)^2
  + LAM_NORM / |F| * sum_{n in F} (||z_n||_2 - 1)^2
```

where:

```text
I = graph-derived marginally independent node pairs
F = free optimized nodes, i.e. latent variables and masked observed variables
```

With `LAM_OBS_PRIOR > 0`, masked observed variables also receive a graph-gated item-level prior:

```text
S_i =
  {j: j is visible and Pa(i) intersects Pa(j)}, if OBS_PRIOR_SCOPE=siblings
  {j: j is visible},                            if OBS_PRIOR_SCOPE=all

r_{i,j} = max(corr(X_i, X_j), 0)

prior_i =
  sum_{j in S_i} r_{i,j} a_j / (sum_{j in S_i} r_{i,j} + eps)

L_obs_prior =
  LAM_OBS_PRIOR / |P| * sum_{i in P} || z_i - normalize(prior_i) ||_2^2

L_new = L_original + L_obs_prior
```

Here `P` is the set of masked observed variables whose candidate set `S_i` is non-empty and whose positive
correlation weights have non-zero sum. If no valid prior can be formed for a masked variable, that variable is
not included in `L_obs_prior`.

For `OBS_PRIOR_SCOPE=siblings`, only visible observed variables sharing a parent latent are used.

Motivation:

- Directly using signed correlations as semantic weights can be harmful because `-embedding` is not a reliable
  semantic antonym.
- Pure parent generation can collapse all observed items under one latent into a shared factor-level meaning.
- A sibling observed prior preserves item-level distinctions while still respecting graph structure.

## Common Settings

These settings apply to both experiments unless explicitly overridden.

Task and data:

| Setting | Value |
|---|---|
| Runner | `run_task1.py` |
| Dataset selection | `DATASET=all` |
| Datasets | `tlvd`, `himi`, `bigfive` |
| Fold count | `FOLDS=5` |
| Fold split seed | `np.random.default_rng(0)` |
| Masking protocol | Observed labels are masked by fold; observed data and graph are kept fixed. |
| Optimization seed | Fold index: `seed=fno` |
| Records format | JSON with `summary` and per-item `records` |

Embedding and decoding:

| Setting | Value |
|---|---|
| Frozen encoder | `sentence-transformers/all-MiniLM-L6-v2` |
| Encoder output | 384-dimensional unit-normalized embeddings |
| Encoder cache env | `HF_CACHE`; default in code is `/data2/shuhao/hf_cache` |
| Decode dictionary env | `GRAPHSEM_DICT` |
| Decode dictionary used by default | `../wikipedia/outputs/concept_bank_wn.npz` |
| Sparse decoder alpha | `metrics.pick_alpha(T, C)`, dataset-specific |

Optimization:

| Setting | Value |
|---|---|
| Optimized variables | Free latent embeddings and masked observed-variable embeddings |
| Fixed variables | Visible observed-label embeddings |
| Optimizer | Adam |
| Learning rate | `lr=5e-2` |
| Steps | `STEPS=1500` |
| Norm regularization | `LAM_NORM=0.1` |
| Independence regularization | `LAM_ZERO=0.3` unless otherwise stated |
| Device | `DEVICE=cpu` for the new full run; original baseline device was whatever the original code default resolved to |

Evaluation:

| Metric | Definition |
|---|---|
| Judge-ACC | LLM yes/no semantic judgment over decoded words |
| Matching-ACC | Hungarian matching accuracy between predicted embeddings and masked true label embeddings |
| Exact | Exact nearest-label top-1 accuracy |

Judge:

| Setting | Value |
|---|---|
| API key source | `OPENAI_API_KEY` environment variable |
| Judge model env | `JUDGE_MODEL` |
| Judge model default | `gpt-4o-mini` |
| Temperature | `0` |
| Timeout | `120s` per request |
| Retry envs | `JUDGE_RETRIES`, `JUDGE_RETRY_BASE` |
| Retry defaults | `JUDGE_RETRIES=5`, `JUDGE_RETRY_BASE=1.0` |
| Failure handling | If judge fails after retries, judge result is recorded as unavailable rather than false. |

## Experiment 0: Original Full Baseline

Configuration:

```text
DATASET=all
FOLDS=5
STEPS=1500
LAM_ZERO=0.3
LAM_NORM=0.1
DEVICE=original code default
EDGE_WEIGHT_MODE=signed
NORMALIZE_GEN=0
LAM_OBS_PRIOR=0
OBS_PRIOR_SCOPE=not used
optimizer=Adam
lr=5e-2
LLM judge enabled
```

Note: this run was produced before the new constraint environment variables were added. The listed constraint
settings are the equivalent interpretation under the current code path.

Output:

```text
outputs/task1_records.json
```

Results:

| Dataset | Arm | Judge-ACC | Matching-ACC | Exact |
|---|---|---:|---:|---:|
| TLVD | uniform | 0.600 | 0.400 | 0.000 |
| TLVD | rawcorr | 0.700 | 1.000 | 0.100 |
| TLVD | core | 0.600 | 1.000 | 0.000 |
| Himi | uniform | 0.317 | 0.400 | 0.000 |
| Himi | rawcorr | 0.300 | 0.867 | 0.000 |
| Himi | core | 0.483 | 0.800 | 0.000 |
| Big Five | uniform | 0.100 | 0.160 | 0.000 |
| Big Five | rawcorr | 0.400 | 0.700 | 0.000 |
| Big Five | core | 0.100 | 0.240 | 0.000 |

## Experiment 1: Full Judge Run With New Constraints

Goal: rerun the new constraint configuration with full settings aligned to the original baseline.

Configuration:

```text
DATASET=all
FOLDS=5
STEPS=1500
LAM_ZERO=0.3
LAM_NORM=0.1
DEVICE=cpu
EDGE_WEIGHT_MODE=abs
NORMALIZE_GEN=1
LAM_OBS_PRIOR=0.5
OBS_PRIOR_SCOPE=siblings
optimizer=Adam
lr=5e-2
fold split seed=0
optimization seeds=fold index
LLM judge enabled
RECORDS_OUT=outputs/exp_task1_all_constraints_full_judge.json
```

Results:

| Dataset | Arm | Judge-ACC | Matching-ACC | Exact |
|---|---|---:|---:|---:|
| TLVD | uniform | 0.600 | 0.400 | 0.000 |
| TLVD | rawcorr | 0.700 | 1.000 | 0.100 |
| TLVD | core | 0.400 | 1.000 | 0.000 |
| Himi | uniform | 0.317 | 0.400 | 0.000 |
| Himi | rawcorr | 0.300 | 0.867 | 0.000 |
| Himi | core | 0.483 | 0.767 | 0.000 |
| Big Five | uniform | 0.100 | 0.160 | 0.000 |
| Big Five | rawcorr | 0.400 | 0.700 | 0.000 |
| Big Five | core | 0.260 | 0.640 | 0.000 |

Full-run core comparison:

| Dataset | Metric | Original Core | New Constraint Core | Change |
|---|---|---:|---:|---:|
| TLVD | Judge-ACC | 0.600 | 0.400 | -0.200 |
| TLVD | Matching-ACC | 1.000 | 1.000 | +0.000 |
| Himi | Judge-ACC | 0.483 | 0.483 | +0.000 |
| Himi | Matching-ACC | 0.800 | 0.767 | -0.033 |
| Big Five | Judge-ACC | 0.100 | 0.260 | +0.160 |
| Big Five | Matching-ACC | 0.240 | 0.640 | +0.400 |

## Interim Summary After Experiment 1

Confirmed improvements:

- Big Five improved substantially:
  - Judge-ACC: `0.100 -> 0.260`
  - Matching-ACC: `0.240 -> 0.640`

Negative or neutral changes:

- TLVD judge-ACC decreased: `0.600 -> 0.400`.
- Himi judge-ACC stayed unchanged: `0.483 -> 0.483`.
- Himi matching-ACC decreased slightly in the full run: `0.800 -> 0.767`.

Conclusion:

- The new constraints are useful for Big Five.
- They are not a universal replacement for the original constraints.
- For now, keep one global constraint configuration during each run, so later ablations are easier to interpret.

Open issue:

- We need a constraint design that preserves TLVD judge-ACC and Himi matching-ACC while improving Big Five.
- The next round should avoid dataset-specific or graph-type-specific switching and instead test globally applied objective changes.

## Experiment 2: Full No-Judge Constraint Sweep

### Goal

Before spending API budget on semantic judging, run a complete global constraint ablation over the currently
implemented controls. This is a screening experiment only: it uses matching-ACC and exact top-1 to identify
promising configurations, while deliberately leaving judge-ACC unavailable. It does not establish semantic
quality by itself.

### Configuration

```text
Runner=run_task1_ablation.py
DATASET=all
FOLDS=5
STEPS=1500
LAM_ZERO=0.3
LAM_NORM=0.1
DEVICE=cpu
RUN_JUDGE=0
OBS_PRIOR_SCOPE=siblings
optimizer=Adam
lr=5e-2
fold split seed=0
optimization seed=fold index
```

The sweep evaluates all 14 combinations implemented by the runner. `Macro matching` is the equal-weight average
of the three dataset-level core matching-ACC values.

| Config | Edge weights | Normalize | `LAM_OBS_PRIOR` | TLVD matching | Himi matching | Big Five matching | Macro matching |
|---|---|---:|---:|---:|---:|---:|---:|
| `A_original` | signed | 0 | 0.0 | 1.000 | 0.767 | 0.280 | 0.682 |
| `B_abs_only` | abs | 0 | 0.0 | 1.000 | 0.767 | 0.180 | 0.649 |
| `C_positive_only` | positive | 0 | 0.0 | 1.000 | 0.767 | 0.320 | 0.696 |
| `D_norm_only` | signed | 1 | 0.0 | 1.000 | 0.867 | 0.240 | 0.702 |
| `E_abs_norm` | abs | 1 | 0.0 | 1.000 | 0.867 | 0.300 | 0.722 |
| `F_prior_only_05` | signed | 0 | 0.5 | 1.000 | 0.867 | 0.360 | 0.742 |
| `G_norm_prior_05` | signed | 1 | 0.5 | 1.000 | 0.867 | 0.360 | 0.742 |
| `H_abs_norm_prior_05` | abs | 1 | 0.5 | 1.000 | 0.867 | 0.600 | 0.822 |
| `I_signed_norm_prior_01` | signed | 1 | 0.1 | 1.000 | 0.767 | 0.300 | 0.689 |
| `J_signed_norm_prior_03` | signed | 1 | 0.3 | 1.000 | 0.867 | 0.300 | 0.722 |
| `K_signed_norm_prior_10` | signed | 1 | 1.0 | 1.000 | 0.900 | 0.360 | 0.753 |
| `L_abs_norm_prior_01` | abs | 1 | 0.1 | 1.000 | 0.767 | 0.620 | 0.796 |
| `M_abs_norm_prior_03` | abs | 1 | 0.3 | 1.000 | 0.867 | 0.640 | 0.836 |
| `N_abs_norm_prior_10` | abs | 1 | 1.0 | 1.000 | 0.900 | 0.580 | 0.827 |

All core exact top-1 values are `0.000`, except `F_prior_only_05` on Big Five (`0.020`). The fixed
`rawcorr` baseline scores `1.000`, `0.867`, and `0.700` matching-ACC on TLVD, Himi, and Big Five respectively.

### Screening Interpretation

- TLVD matching-ACC is saturated at `1.000` for every core configuration, so this metric cannot choose among
  constraints for TLVD.
- Himi is strongest for `K_signed_norm_prior_10` and `N_abs_norm_prior_10` (`0.900`).
- Big Five and macro matching are strongest for `M_abs_norm_prior_03` (`0.640` and `0.836`).
- `H_abs_norm_prior_05` is a close Big Five candidate (`0.600`) with a stronger sibling prior, while
  `D_norm_only` isolates the effect of normalized generation.

The no-judge sweep therefore selected `A_original`, `D_norm_only`, `H_abs_norm_prior_05`,
`K_signed_norm_prior_10`, and `M_abs_norm_prior_03` for the subsequent semantic-quality evaluation.

Outputs:

```text
outputs/ablations/ablation_manifest.json
outputs/ablations/summary.csv
outputs/ablations/summary.md
outputs/diagnostics/per_item_diagnostics.csv
outputs/diagnostics/error_report.md
```

## Experiment 3: Targeted Judged Selection for Task 1 v1.1

### Goal

The first full judged run used one new configuration (`abs` weights, normalized generation, and sibling prior
weight `0.5`) and showed a Big Five improvement, but it did not establish whether the gain reflected semantic
quality or only matching behavior. Experiment 2 then selected five candidates using a full no-judge sweep.
This targeted run compared those candidates under the same five-fold protocol, with the LLM judge enabled for
all three arms.

The decision criterion was semantic quality first: retain a candidate only if it improves or preserves judge-ACC
relative to the original core and does not materially degrade TLVD. Matching-ACC was used as a complementary
identity metric, not as a substitute for semantic correctness.

### Common Run Settings

```text
DATASET=all
FOLDS=5
STEPS=1500
LAM_ZERO=0.3
LAM_NORM=0.1
DEVICE=cpu
RUN_JUDGE=1
JUDGE_MODEL=gpt-4o-mini
JUDGE_RETRIES=5
OBS_PRIOR_SCOPE=siblings
optimizer=Adam
lr=5e-2
fold split seed=0
optimization seed=fold index
```

All 1,140 per-item arm records received a judge verdict. Outputs:

```text
outputs/ablations_judged/ablation_manifest.json
outputs/ablations_judged/summary.csv
outputs/ablations_judged/summary.md
outputs/ablations_judged/judged_error_report.md
outputs/ablations_judged/per_item_diagnostics.csv
```

### Candidate Configurations

| Name | `EDGE_WEIGHT_MODE` | `NORMALIZE_GEN` | `LAM_OBS_PRIOR` | `OBS_PRIOR_SCOPE` |
|---|---|---:|---:|---|
| `A_original` | `signed` | `0` | `0.0` | `siblings` |
| `D_norm_only` | `signed` | `1` | `0.0` | `siblings` |
| `H_abs_norm_prior_05` | `abs` | `1` | `0.5` | `siblings` |
| `K_signed_norm_prior_10` | `signed` | `1` | `1.0` | `siblings` |
| `M_abs_norm_prior_03` | `abs` | `1` | `0.3` | `siblings` |

### Macro Results

Each dataset contributes equally to the macro average.

| Core setting | Judge-ACC | Matching-ACC | Exact top-1 |
|---|---:|---:|---:|
| `A_original` | 0.352 | 0.682 | 0.000 |
| `D_norm_only` | 0.366 | 0.702 | 0.000 |
| `H_abs_norm_prior_05` | 0.361 | 0.822 | 0.000 |
| `K_signed_norm_prior_10` | 0.399 | 0.753 | 0.000 |
| **`M_abs_norm_prior_03`** | **0.501** | **0.836** | **0.000** |
| `rawcorr` baseline | 0.480 | 0.856 | 0.033 |

`M_abs_norm_prior_03` relative to `A_original`:

- Macro judge-ACC: `+0.149`.
- Macro matching-ACC: `+0.153`.
- Macro exact top-1: `+0.000`.

`M_abs_norm_prior_03` relative to `rawcorr`:

- Macro judge-ACC: `+0.021`.
- Macro matching-ACC: `-0.020`.
- Macro exact top-1: `-0.033`.

### Dataset-Level Results for the Selected Configuration

| Dataset | A original core: judge / match | M core: judge / match | Rawcorr: judge / match |
|---|---:|---:|---:|
| TLVD | 0.400 / 1.000 | **0.700 / 1.000** | 0.700 / 1.000 |
| Himi | 0.417 / 0.767 | **0.483 / 0.867** | 0.300 / 0.867 |
| Big Five | 0.240 / 0.280 | **0.320 / 0.640** | 0.440 / 0.700 |

### Per-Item Semantic Diagnosis

- **TLVD:** matching-ACC is saturated at `1.000` for both `A_original` and M, but M improves judge-ACC from
  `0.400` to `0.700`. Relative to A, M gains four judged items and loses two:
  `X_AverageQus_Par2` and `X_AverageQus_Par3`. Thus matching saturation does not hide an overall semantic
  degradation in this setting, although it does hide item-level regressions.
- **Big Five:** M improves both judge-ACC (`0.240 -> 0.320`) and matching-ACC (`0.280 -> 0.640`) over A.
  The per-item judge comparison contains 13 gains and 9 losses. This supports a real semantic improvement over
  A, but not a Big Five win over `rawcorr`, which remains stronger on both judge and matching.
- **Himi:** M preserves the `0.867` matching-ACC of `rawcorr` while increasing judge-ACC to `0.483`.
  Relational integration reaches judge-ACC `0.667` under M. Divided attention remains at judge-ACC `0.000`
  under A, D, and M, so the current graph constraints and decoder do not yet robustly recover that construct's
  specific semantics.

## Final Decision: Task 1 v1.1

**Fix `M_abs_norm_prior_03` as the global Task 1 v1.1 configuration.**

```text
EDGE_WEIGHT_MODE=abs
NORMALIZE_GEN=1
LAM_OBS_PRIOR=0.3
OBS_PRIOR_SCOPE=siblings
```

Rationale:

- It is the strongest tested setting on macro judge-ACC (`0.501`) and nearly matches `rawcorr` on macro
  matching-ACC (`0.836` vs `0.856`).
- It improves judge-ACC over the original core on all three datasets, improves matching-ACC on Himi and
  Big Five, and preserves the saturated TLVD matching-ACC.
- It improves TLVD judge-ACC rather than causing the previously observed concern about a semantic decline.
- It does not use dataset-specific or graph-type-specific switching; the same objective is applied to every
  graph.

Limitations:

- This is a five-fold comparison over three datasets, not a statistical significance claim.
- Strict exact top-1 remains `0.000` for every core configuration; the current decoder has not achieved exact
  label recovery.
- `rawcorr` is still stronger on Big Five, and Himi divided attention remains semantically unresolved.

## Experiment 4: Controlled Synthetic/Oracle Graph Diagnosis

### Goal

Experiment 3 selected `M_abs_norm_prior_03` as the fixed real-data Task 1 v1.1 configuration. Experiment 4
tests whether its apparent semantic benefit survives four controlled graph structures with known loadings,
parents, polarity, and item types. This phase is diagnostic only: it does not add Task 2, GNN, MI/CMI, a new
decoder, or a new optimization algorithm.

The six evaluated arms are `uniform`, `rawcorr`, `A_original`, `D_norm_only`,
`M_abs_norm_prior_03`, and `K_signed_norm_prior_10`. The controlled comparisons isolate whether failures arise
from the graph-constraint representation or from generic embedding optimization.

### Oracle Data Generation

The generator samples independent latent scores and produces observed variables through a known linear model:

```text
L_k ~ Normal(0, 1)

X_i = sum_{k in Pa(i)} loading_{k,i} L_k + epsilon_i
epsilon_i ~ Normal(0, sigma_i^2)
```

The loader z-scores each observed column before estimating graph edge weights. Each dataset has 1,200 samples.
The base generator seed is `20260710`; successive datasets use `20260710 + dataset_offset`.

| Dataset | Latents and observed structure | Loadings / noise | Diagnostic purpose |
|---|---|---|---|
| `oracle_clean` | Working memory, relational integration, divided attention, and task switching; four semantically consistent observed labels per latent | All loadings `+0.90`; noise SD `0.40` | Ideal dense positive-loading sibling graph |
| `oracle_polarity` | Extraversion and neuroticism; three positive items and one reverse-coded item per latent | Positive `+0.85` to `+0.90`; reverse `-0.90`; noise SD `0.38` | Whether absolute weights or the sibling prior erase polarity |
| `oracle_mixed_parent` | Working memory, divided attention, and relational reasoning; nine pure items and three two-parent items | Pure `+0.84` to `+0.90`; mixed `+0.64` per parent; noise SD `0.40` | Whether parent aggregation represents compositional labels |
| `oracle_sparse_sibling` | Same four constructs as `oracle_clean`, but only two observed items per latent | All loadings `+0.90`; noise SD `0.40` | Dependence on sibling density |

Generation script:

```text
scripts/make_oracle_datasets.py
```

Each `data/oracle_*` directory contains `data.csv`, `codebook.txt`, `graph.dot`, `latent_labels.json`, and
`oracle_metadata.json`. The metadata records the exact loading, parent set, polarity, item type, and sibling
count for every observed variable.

### Configuration

```text
Runner=run_oracle_diagnostics.py
ORACLE_DATASET=all
FOLDS=5
STEPS=1500
LAM_ZERO=0.3
LAM_NORM=0.1
DEVICE=cpu
OBS_PRIOR_SCOPE=siblings
optimizer=Adam
lr=5e-2
fold split seed=0
optimization seed=fold index
encoder=sentence-transformers/all-MiniLM-L6-v2
embedding dimension=384
LLM judge=disabled
```

`run_task1.py` also accepts each oracle dataset through `DATASET=oracle_clean`, `oracle_polarity`,
`oracle_mixed_parent`, or `oracle_sparse_sibling`. Its historical `DATASET=all` behavior remains restricted to
TLVD, Himi, and Big Five.

| Arm | Edge weights | Normalized generation | `LAM_OBS_PRIOR` | Purpose |
|---|---|---:|---:|---|
| `uniform` | n/a | n/a | n/a | Structure-free visible-label average |
| `rawcorr` | n/a | n/a | n/a | Positive raw-correlation baseline |
| `A_original` | `signed` | `0` | `0.0` | Original graph objective |
| `D_norm_only` | `signed` | `1` | `0.0` | Isolate normalized generation |
| `M_abs_norm_prior_03` | `abs` | `1` | `0.3` | Fixed Task 1 v1.1 setting |
| `K_signed_norm_prior_10` | `signed` | `1` | `1.0` | Signed-edge, strong-prior control |

### Oracle-Specific Evaluation

The LLM judge is intentionally disabled because the oracle metadata provides direct structural diagnostics.
In addition to matching-ACC and exact top-1, the experiment records:

```text
cosine_i = cos(predicted_embedding_i, true_label_embedding_i)

anchor_l = normalize(mean true-label embedding of pure items under latent l)
parent_score(i,l) = cos(predicted_embedding_i, anchor_l)
parent-set correct_i = 1 if the top-|Pa(i)| anchors equal Pa(i), else 0

polarity_margin_i = cos(predicted_i, true_i)
                    - mean_{j: same parents, opposite polarity} cos(predicted_i, true_j)
polarity correct_i = 1 if polarity_margin_i > 0, else 0
```

`prior coverage` is the fraction of masked items for which the positive-correlation sibling mixture is nonzero.
Matching-ACC remains a fold-local Hungarian assignment metric and can saturate when a sparse fold has few
competitors; cosine and oracle structural metrics are therefore required for interpretation.

### Full Results

| Dataset | Arm | Matching-ACC | Exact | Mean cosine | Parent-set ACC | Polarity ACC | Prior coverage |
|---|---|---:|---:|---:|---:|---:|---:|
| Clean | `uniform` | 0.438 | 0.000 | 0.480 | 0.000 | - | 0.000 |
| Clean | `rawcorr` | 0.625 | 0.000 | 0.562 | 1.000 | - | 0.000 |
| Clean | `A_original` | 0.750 | 0.000 | 0.554 | 1.000 | - | 0.000 |
| Clean | `D_norm_only` | 0.875 | 0.000 | 0.554 | 1.000 | - | 0.000 |
| Clean | `M_abs_norm_prior_03` | 0.625 | 0.000 | 0.555 | 1.000 | - | 1.000 |
| Clean | `K_signed_norm_prior_10` | 0.625 | 0.000 | 0.555 | 1.000 | - | 1.000 |
| Polarity | `uniform` | 0.500 | 0.000 | 0.539 | 0.250 | 0.000 | 0.000 |
| Polarity | `rawcorr` | 0.750 | 0.000 | 0.475 | 0.750 | 0.625 | 0.000 |
| Polarity | `A_original` | 0.500 | 0.000 | -0.004 | 0.500 | 1.000 | 0.000 |
| Polarity | `D_norm_only` | 0.750 | 0.000 | -0.004 | 0.500 | 1.000 | 0.000 |
| Polarity | `M_abs_norm_prior_03` | 1.000 | 0.000 | 0.577 | 1.000 | 0.000 | 0.750 |
| Polarity | `K_signed_norm_prior_10` | 0.750 | 0.000 | 0.158 | 0.750 | 0.875 | 0.750 |
| Mixed parent | `uniform` | 0.250 | 0.000 | 0.477 | 0.083 | - | 0.000 |
| Mixed parent | `rawcorr` | 0.833 | 0.083 | 0.551 | 1.000 | - | 0.000 |
| Mixed parent | `A_original` | 1.000 | 0.167 | 0.517 | 1.000 | - | 0.000 |
| Mixed parent | `D_norm_only` | 0.833 | 0.167 | 0.524 | 1.000 | - | 0.000 |
| Mixed parent | `M_abs_norm_prior_03` | 0.833 | 0.167 | 0.533 | 1.000 | - | 1.000 |
| Mixed parent | `K_signed_norm_prior_10` | 0.833 | 0.167 | 0.542 | 1.000 | - | 1.000 |
| Sparse sibling | `uniform` | 0.750 | 0.000 | 0.355 | 0.000 | - | 0.000 |
| Sparse sibling | `rawcorr` | 1.000 | 0.000 | 0.359 | 1.000 | - | 0.000 |
| Sparse sibling | `A_original` | 1.000 | 0.000 | 0.354 | 1.000 | - | 0.000 |
| Sparse sibling | `D_norm_only` | 1.000 | 0.000 | 0.353 | 1.000 | - | 0.000 |
| Sparse sibling | `M_abs_norm_prior_03` | 1.000 | 0.000 | 0.354 | 1.000 | - | 1.000 |
| Sparse sibling | `K_signed_norm_prior_10` | 1.000 | 0.000 | 0.356 | 1.000 | - | 1.000 |
| Macro average | `uniform` | 0.484 | 0.000 | 0.463 | 0.083 | 0.000 | 0.000 |
| Macro average | `rawcorr` | 0.802 | 0.021 | 0.486 | 0.938 | 0.625 | 0.000 |
| Macro average | `A_original` | 0.812 | 0.042 | 0.355 | 0.875 | 1.000 | 0.000 |
| Macro average | `D_norm_only` | 0.865 | 0.042 | 0.357 | 0.875 | 1.000 | 0.000 |
| Macro average | `M_abs_norm_prior_03` | 0.865 | 0.042 | 0.505 | 1.000 | 0.000 | 0.938 |
| Macro average | `K_signed_norm_prior_10` | 0.802 | 0.042 | 0.403 | 0.938 | 0.875 | 0.938 |

Polarity ACC and polarity margin are defined only for `oracle_polarity`; the macro row reports that dataset's
value rather than treating unavailable values as zero.

### Controlled Diagnosis

- **Clean graph:** M recovers the correct latent parent for every item (`parent-set ACC=1.000`) and obtains mean
  cosine `0.555`, so the optimizer passes the basic structural sanity check. However, M matching-ACC is `0.625`:
  it ties `rawcorr` and is `0.250` below `D_norm_only`. The sibling prior is not a clean-graph improvement.
- **Polarity:** M obtains matching-ACC `1.000` but polarity ACC `0.000` with mean polarity margin `-0.191`.
  `D_norm_only` preserves polarity perfectly (`1.000`) and K reaches `0.875`. The absolute edge transform erases
  negative loading direction, while the positive-correlation prior gives reverse-coded items zero prior coverage.
  This is direct evidence that high matching can hide semantic-direction errors.
- **Mixed parents:** M reaches parent-set ACC `1.000`. Its mixed items have matching-ACC `1.000` and mean cosine
  `0.707`, compared with `0.778` and `0.476` for pure items. Parent aggregation succeeds in this oracle.
- **Sparse siblings:** M matching-ACC remains `1.000`, but mean cosine falls from `0.555` on the clean graph to
  `0.354`. Each sparse item has one visible sibling on average, versus `2.5` for the clean graph. The matching
  metric is saturated by the smaller candidate set and does not reveal this semantic-quality decline.

### Small Polarity-Aware Follow-up

The initial oracle diagnosis could not distinguish whether M's polarity failure came from the absolute edge
transformation or from the sibling prior. A focused follow-up therefore runs only `oracle_polarity` with the same
five folds, 1,500 optimization steps, `LAM_ZERO=0.3`, `LAM_NORM=0.1`, normalized generation, and CPU. The judge
remains disabled.

The core design is a small factorial comparison:

- Edge generation: `signed` versus `abs`.
- Prior: none, the current positive-correlation prior, or an explicit loading-aware prior.
- The explicit prior retains a sibling only when the two edge-loading signs agree under their shared latent parent.
  This relation is invariant to a global sign flip of the estimated latent score.

| Arm | Edge | Prior | Matching | Mean cosine | Polarity ACC | Reverse cosine |
|---|---|---|---:|---:|---:|---:|
| `uniform` | - | - | 0.500 | 0.539 | 0.000 | 0.597 |
| `rawcorr` | - | - | 0.750 | 0.475 | 0.625 | 0.375 |
| D | `signed` | none | 0.750 | -0.004 | 1.000 | -0.605 |
| E | `abs` | none | 0.750 | 0.585 | 0.000 | 0.605 |
| J | `signed` | current correlation prior, 0.3 | 0.750 | 0.086 | 1.000 | -0.605 |
| M | `abs` | current correlation prior, 0.3 | 1.000 | 0.577 | 0.000 | 0.605 |
| P | `signed` | loading-aware prior, 0.3 | 0.750 | 0.086 | 1.000 | -0.605 |
| Q | `abs` | loading-aware prior, 0.3 | 1.000 | 0.577 | 0.000 | 0.605 |

Exact top-1 is `0.000` for every arm. Three direct equivalence checks are also exactly zero:

```text
max L2(current prior, loading-aware prior) = 0.00000000
max L2(J prediction, P prediction)          = 0.00000000
max L2(M prediction, Q prediction)          = 0.00000000
```

This follow-up changes the interpretation in two ways:

1. Explicit loading-sign filtering does not improve or alter the sibling prior on this oracle. The existing
   `max(corr, 0)` weighting already admits zero opposite-loading contributors, so it is implicitly polarity-selective
   under the clean linear generator.
2. Edge generation, rather than prior construction, causes the polarity-ACC collapse. However, `signed` generation
   is not a semantic solution: its reverse-item prediction has cosine `-0.605` to the true reverse label. It obtains
   polarity ACC `1.000` by negating the latent vector, but a negative semantic embedding is not an antonym embedding.

Thus neither `abs` nor direct signed multiplication correctly represents a negative-loading semantic relation.
A successful method must improve both cosine and polarity, rather than trading one for the other.

### Interpretation and Decision

The combined Exp4 evidence points to constraint representation rather than a generic optimizer failure. Clean and
mixed-parent graphs recover their known parent structure, and the focused follow-up shows that explicit
polarity-aware sibling filtering is numerically redundant with the current positive-correlation prior. The unresolved
problem is the semantic relation attached to a negative-loading edge: `abs` discards direction, while signed scalar
multiplication produces an anti-aligned vector rather than the reverse-coded text meaning.

Configuration M remains the fixed real-data Task 1 v1.1 benchmark selected by Experiment 3, but the oracle
results show that it is not yet a universal graph-semantic rule. In particular, `abs` weights cannot be assumed
safe when edge sign has known semantic meaning, and sibling-prior density affects embedding quality even when
matching-ACC is saturated.

The next smallest controlled test should use a separate balanced polarity oracle with at least two reverse items
per latent and estimate positive/reverse sign-group semantic prototypes from visible labels. This non-parametric
relation representation should be tested before learning a parameterized positive/reverse transform or replacing
the general optimizer.

Outputs:

```text
outputs/oracle/oracle_manifest.json
outputs/oracle/summary.csv
outputs/oracle/summary.md
outputs/oracle/oracle_error_report.md
outputs/oracle/per_item_diagnostics.csv

outputs/polarity_ablation/manifest.json
outputs/polarity_ablation/summary.csv
outputs/polarity_ablation/summary.md
outputs/polarity_ablation/report.md
outputs/polarity_ablation/per_item_diagnostics.csv
```

## Next Steps

1. Retain M as the frozen real-data Task 1 v1.1 benchmark, with `A_original`, `D_norm_only`, `K`, and `rawcorr`
   as controlled references rather than treating M as universally valid.
2. Add a separate balanced polarity oracle with at least two positive and two reverse items per latent. Estimate
   sign-group semantic prototypes from visible labels without learning a new model.
3. Test whether the sign-group representation improves both cosine and polarity. Only if it is insufficient should
   a parameterized positive/reverse relation transform and its training algorithm be introduced.
4. Repeat the oracle diagnosis over additional generator seeds and fold splits. Prioritize cosine, parent-set ACC,
   and polarity ACC over matching-ACC on sparse folds.
5. Isolate why M blurs item identity on `oracle_clean` and loses cosine under sparse sibling support. Revisit the
   optimizer only if clean structural recovery becomes unstable across seeds.
6. Re-evaluate the focused change on TLVD, Himi, and Big Five before making a stronger generalization claim.
