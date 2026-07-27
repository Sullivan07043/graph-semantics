# Task 3 J-lens Preflight Report

Date: 2026-07-23

## Executive conclusion

**Stage-0 measurement/write GO; Stage-1 causal-graph NO-GO under the
preregistered comparison.** The official Qwen3.5-4B Jacobian lens runs on the
local RTX 5090, produces non-degenerate coordinates at layers 8, 16, 24, and
30, and supports precise minimum-norm dual writes. The official CauScale
checkpoint also accepts the 1,000 x 128 pilot matrix and assigns scores that
predict held-out intervention effects well. However, it does not outperform
the absolute-correlation baseline on AUPRC, and a same-concept cross-layer
heuristic performs better still. The present result therefore supports a
predictive dependency/carryover interpretation, not a causal-graph claim.


A focused Stage-1 repair was subsequently run: discovery-only grouped ridge
residualization removed predictable same-concept cross-layer carryover,
same-concept edges were excluded from the primary graph score, max-product
path probability replaced direct-edge probability for total-effect prediction,
and all 32 target token IDs were excluded from held-out prompts. The corrected
validation found zero practically positive cross-concept effects among 1,085
eligible pairs. Stage 1 therefore remains NO-GO; Stage 2 has not started.

## Environment

| component | value |
|---|---|
| model | `Qwen/Qwen3.5-4B` |
| lens | `neuronpedia/jacobian-lens`, revision `qwen-n1000` |
| `jlens` source | Anthropic commit `581d398613e5602a5af361e1c34d3a92ea82ba8e` |
| GPU | NVIDIA GeForce RTX 5090 |
| PyTorch | 2.13.0+cu130 |
| Transformers | 5.14.1 |
| Python | 3.13.9 |
| dtype | bf16 |

The optional optimized Qwen attention path is not installed; Transformers
used its PyTorch fallback. This affects throughput, not the interpretation of
the preflight.

## Readout result

The official quartile-layer rule resolves to layers 8, 16, 24, and 30.
Cold download plus first load took 692.1 seconds and used 7.98 GiB peak CUDA
memory. After caching, selected-layer J-lens inference averaged 0.255 seconds
per prompt in the 20-prompt matrix run.

Examples:

- For the phrase ending in "country shaped like a boot", layer-8 J-lens
  surfaced `boots` and `heel`, while the layer-8 logit lens was mostly
  uninterpretable fragments. Layer 24 surfaced `shape`/`shaped`.
- For the programming prompt about an infinite loop, layer-24 J-lens surfaced
  `incorrectly`, `never`, and `wrong`.
- The analogy prompt mostly produced blank/answer-slot tokens. This is a
  negative qualitative example and shows that prompt/read-position selection
  still needs calibration.

## Minimal feature matrix

The preflight selected eight verified single-token concepts:

`water`, `fire`, `music`, `danger`, `Italy`, `code`, `animal`, and `happy`.

Across 20 prompts and four layers, the resulting selected-token J-lens score
matrix has shape `20 × 32`.

| check | result |
|---|---:|
| NaNs | 0 |
| zero-variance columns | 0 |
| minimum column SD | 1.311 |
| median column SD | 1.803 |
| maximum column SD | 3.153 |
| cached total runtime | 10.55 seconds |
| peak CUDA memory | 7.98 GiB |

These are J-lens logits, not yet the final sparse J-space coordinates or
validated causal variables.

## One-node dual-write audit

Configuration:

- source: `water`;
- layer: 24;
- audit prompt: `After hours in the sun, the exhausted hiker opened the container.`;
- coordinate system: eight normalized static J-lens vectors;
- write operator: ridge-regularized minimum-norm dual;
- calibration: 20 prompts;
- Gram condition number: 3.30.

| dose | target error (SD) | mean off-target (SD) | max off-target (SD) | `water` output Δlogit |
|---:|---:|---:|---:|---:|
| -1 SD | 0.0075 | 0.0047 | 0.0081 | -1.0313 |
| +1 SD | 0.0068 | 0.0048 | 0.0087 | +0.8125 |

For the +1 SD intervention, the largest positive output-logit changes were
`water`, `Water`, `WATER`, and other water-token variants. For -1 SD, water
decreased and fire-related tokens were among the largest positive changes.

This is a local write-precision result for one source, prompt, layer, feature
set, and dose pair. It does not establish downstream causal edges, robustness
across prompts, or semantic recovery.

## 32-concept write calibration

The expanded calibration used 32 verified single-token concepts, 100 prompts
that omit the four initial source words, four layers, four source concepts,
10 audit prompts, doses {-1, +1} SD, and four arms. Batched execution produced
1,280 intervention rows.

| arm | median target error (SD) | median mean off-target (SD) | direction accuracy | coordinate pass rate |
|---|---:|---:|---:|---:|
| ridge-dual target | 0.0193 | 0.0175 | 1.000 | 1.000 |
| naive target direction | 0.0194 | 0.3060 | 1.000 | 0.000 |
| wrong dual, norm matched | 1.0005 | 0.0524 | 0.488 | 0.000 |
| random direction, norm matched | 1.0017 | 0.0274 | 0.463 | 0.000 |

The ridge-dual method passed all 320 target-write cases. The naive direction
hit the requested target but caused roughly 17 times the median off-target
movement, empirically justifying the minimum-norm dual operator. Direct
J-lens logits and normalized static coordinates had median Pearson
correlation 0.977 across 128 layer-concept columns (minimum 0.812), so the
coordinate is related to but not identical to the nonlinear final-norm
J-lens readout.

## Pilot discovery matrix

The discovery matrix uses 1,000 WikiText-2 validation paragraphs selected
deterministically round-robin across 56 article groups and truncated to 128
tokens. It contains 32 concepts at each of four layers, for 128 columns.

| check | result |
|---|---:|
| shape | 1,000 x 128 |
| NaNs | 0 |
| zero-variance columns | 0 |
| column SD range | 0.176-1.641 |
| extraction time | 223.6 seconds |
| mean time per prompt | 0.217 seconds |
| peak CUDA memory | 8.02 GiB |

This is an observational feature matrix, not a graph or a validated set of
causal variables.

## CauScale architecture smoke and bootstrap

The released synthetic CauScale checkpoint loaded strictly with all 938 state
keys and no missing or unexpected keys. Its released Python 3.10 and
PyTorch-Lightning 1.9 interface was retained; only PyTorch was updated to a
CUDA 13 build so the RTX 5090 is supported.

The 1,000 x 128 forward pass took 0.301 seconds with the Ledoit-Wolf precision
prior. Of 6,144 layer-allowed directed pairs, 111 received probability at
least 0.5. A zero-prior ablation produced only two such pairs. The two arms
had Top-100 Jaccard 0.176 and allowed-edge Spearman 0.756, showing substantial
prior sensitivity.

Twenty sample bootstraps were combined with concept-group feature
bootstraps: each run sampled 1,000 rows with replacement and retained 26 of
32 concepts at all four layers. With probability at least 0.5 and selection
frequency at least 0.8:

- 33 stable candidates remained;
- 32/33 were the same token-anchored concept across layers;
- the sole stable cross-concept candidate was risk@24 -> danger@30;
- conditional pairwise edge-set Jaccard had median 0.213.

The predominance of same-concept carryover is a warning that the graph mostly
rediscovers how the token-anchored coordinate dictionary propagates through
depth.

## Held-out activation-intervention validation

The primary validation froze the graph and source-selection rule before
measuring effects. It used 16 source nodes, 20 WikiText-2 test prompts from
20 test-only article groups, and doses {-2, -1, +1, +2} SD. Tested source
words were excluded from prompts. There were 1,280 writes and 896 eligible
later-layer source-target pairs.

| write check | result |
|---|---:|
| median target error | 0.0160 SD |
| p95 target error | 0.0593 SD |
| median mean off-target movement | 0.0173 SD |
| p95 mean off-target movement | 0.0498 SD |
| pass rate (error <= 0.1, off-target <= 0.1) | 1.000 |

A downstream pair was counted positive only when a 4,096-draw paired
prompt-level sign-flip test had Benjamini-Hochberg q < 0.05 and RMS
standardized effect was at least 0.1. This yielded 23 positive pairs.

| predictor | AUROC | AUPRC | Precision@10 |
|---|---:|---:|---:|
| CauScale mean probability | 0.976 | 0.714 | 0.900 |
| bootstrap selection frequency | 0.940 | 0.623 | 0.900 |
| absolute correlation | 0.954 | 0.741 | 1.000 |
| same-concept cross-layer heuristic | 0.997 | 0.821 | 1.000 |
| architecture-only layer distance | 0.589 | 0.031 | 0.100 |
| seeded random | 0.471 | 0.029 | 0.000 |

All 23 positive effects were same-concept cross-layer effects. Eight stable
CauScale edges were covered by the primary source set and all eight had a
positive total effect; their median RMS effect was 1.064 SD, versus 0.036 SD
for unrelated predicted non-edges. This is strong predictive evidence, but
CauScale fails the preregistered requirement to beat the correlation baseline
and is also beaten by the same-concept heuristic.

The frozen sole cross-concept candidate was tested separately:

| pair | bootstrap probability | frequency | RMS effect | BH q | practical result |
|---|---:|---:|---:|---:|---|
| risk@24 -> danger@30 | 0.708 | 0.800 | 0.054 SD | 0.0039 | fail: below 0.1 SD |
| risk@24 -> risk@30 positive control | - | - | 1.105 SD | 0.0039 | pass |

The statistical risk-to-danger change is repeatable but too small to meet the
preregistered practical-effect threshold. The same-concept positive control
confirms that this decision is not caused by a failed write.

Single-source interventions validate total downstream effect or
reachability. They do not prove that any surviving candidate is a direct
edge; mediator/path-blocking interventions would still be required.


## Corrected Stage-1 innovation validation

The focused repair used one fixed configuration rather than a threshold or
model ablation sweep. For each concept at layers 16, 24, and 30, its raw
coordinate was regressed on all earlier-layer raw coordinates of the same
concept using ridge alpha 1.0. Fits used discovery data only; five folds grouped
by WikiText article audited out-of-group prediction, and the final
full-discovery coefficients were frozen before held-out evaluation.

| innovation check | result |
|---|---:|
| grouped out-of-fold R2, median | 0.594 |
| same-concept earlier/later absolute correlation, median before | 0.633 |
| same-concept earlier/innovation absolute correlation, median after | 0.00069 |
| NaNs / zero-variance columns | 0 / 0 |

CauScale was then rerun with the same released checkpoint and the same 20
sample/concept-group bootstrap seeds. It produced 24 stable edges: 23 were
same-concept and one was cross-concept, \`animal@8 -> city@24\`. For primary
prediction, same-concept edges were removed and a pair received the maximum
product of mean CauScale probabilities over all layer-ordered cross-concept
paths. This aligns the graph score with the total downstream effect measured
by a single-source intervention.

The final held-out pass excluded every one of the 32 target token IDs at the
tokenizer level. It ran 1,280 writes over 16 sources, four doses, and 20
test-only article groups. One \`Italy@8\`, +1 SD write had target error 0.10065 SD
and was marked invalid; its source-prompt dose group was excluded from the
paired statistics. The remaining audit pass rate was 99.92%, with at least 19
fully valid prompts per evaluated pair.

| corrected held-out result | value |
|---|---:|
| token-ID leakage count | 0 |
| evaluated cross-concept pairs | 1,085 |
| cross-concept pairs with groupwise BH q < 0.05 and RMS >= 0.1 SD | 0 |
| stable cross-concept candidates evaluated | 1 |
| \`animal@8 -> city@24\` RMS innovation effect | 0.0222 SD |
| \`animal@8 -> city@24\` groupwise BH q | 0.838 |
| unrelated cross-concept non-edge median RMS | 0.0283 SD |
| raw same-concept positive controls | 27 |
| raw same-concept positive-control median RMS | 0.276 SD |

Because the corrected primary endpoint has zero cross-concept positives,
AUROC and AUPRC are undefined rather than evidence for or against one
predictor ranking. The sole stable cross-concept candidate is weaker than the
median unrelated pair and fails both statistical and practical-effect gates.
The retained raw same-concept controls show that the null cross-concept result
is not explained by a generally broken write operator.

## Decision and next experiment

Do not advance this configuration to graph-conditioned semantic recovery as
if Stage 1 had passed. The current result is:

1. **GO** for the J-lens measurement interface and ridge-dual intervention
   operator.
2. **GO** for computational feasibility of a 1,000 x 128 matrix and official
   CauScale inference on the RTX 5090.
3. **NO-GO** for a causal-graph claim because CauScale does not beat
   correlation AUPRC, a same-concept heuristic beats both, bootstrap graph
   stability is modest, and the only stable cross-concept edge misses the
   practical intervention threshold.

The residualization repair has now been completed and did not reveal a
practically measurable cross-concept endpoint. Do not continue with a large
ablation grid or enter semantic reconstruction. The single proposed fallback
is a new, separately frozen concept-prototype pilot: define each concept from
3-5 prespecified synonymous token directions, rerun this same innovation/path
pipeline once, and retain the same statistical and RMS >= 0.1 SD gates. If
that pilot also yields too few cross-concept positives for a stable comparison,
terminate Stage 1 for token-anchored J features and redesign the feature track
before any Stage-2 claim.

## Artifacts

- task3_v1/preflight_jlens.py
- task3_v1/preflight_matrix.py
- task3_v1/preflight_intervention.py
- task3_v1/run_write_calibration.py
- task3_v1/build_discovery_matrix.py
- task3_v1/build_innovation_matrix.py
- task3_v1/run_causcale_smoke.py
- task3_v1/run_causcale_bootstrap.py
- task3_v1/run_heldout_graph_validation.py
- task3_v1/run_innovation_graph_validation.py
- task3_v1/run_targeted_cross_edge.py
- ignored runtime artifacts under `task3_v1/outputs/` and `task3_v1/logs/`
