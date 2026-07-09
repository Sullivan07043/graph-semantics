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

## Summary

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

## Next Experiments

1. Tune one global `LAM_ZERO`: `0`, `0.01`, `0.05`, `0.1`, `0.3`.
2. Test softer observed priors: smaller `LAM_OBS_PRIOR`, top-k correlated siblings, or confidence-weighted priors.
3. Test edge-weight transforms globally: `signed`, `positive`, `abs`, and clipped/rescaled variants.
4. Add per-fold and per-item diagnostic summaries for `core` vs `rawcorr`.
5. Run Task 2 with both original and best global constraint settings.
6. Vectorize independence decorrelation; the current pairwise loop is slow for dense independent-pair sets.
