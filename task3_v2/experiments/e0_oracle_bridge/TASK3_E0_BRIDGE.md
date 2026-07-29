# Task 3 E0′ — Oracle Causal-Graph Bridge Test

## Question

Does the frozen Stage-3 Task-1 graph-semantic solver, which was developed on
psychometric/scoring-key/hierarchy graphs, still use correctly aligned graph
structure to improve masked semantic recovery on natural-semantic directed
causal DAGs?

E0′ is a bridge test, not a new training exercise and not a causal-discovery
experiment. The graph structure is oracle-provided in the primary arm.

## Frozen method

- Formal source entrypoint: `v5/main.py` with `TASK=1`, `L2_ARM=mlp`, and
  `K=60`.
- Encoder space: `intfloat/e5-large-v2` plus the existing last-two-layer
  rank-8 LoRA checkpoint.
- Solver: the existing frozen WeightNet-weighted unrolled optimizer.
- Objective: unchanged Stage-3 generation, residual, residual-correlation,
  marginal-independence, Pearson upper-tail bridge, and norm terms with the
  frozen coefficients in `config.yaml`.
- Decoder and baselines: the existing SpLiCE decoder, raw-correlation
  baseline, and uniform baseline.
- No LoRA, WeightNet, encoder, dictionary, or solver retraining is allowed.

The E0′ runner is an interface adapter around the existing modules. It supplies
an observed-only `v5.graph.Graph`, externally selected oracle/reversed/shuffled
support, and either pipeline-estimated or SCM-true edge weights. It does not
change `v5/` or the Stage-3 loss.

## Fixed causal micro-worlds

The three graph specifications under `graphs/` were fixed before evaluation:

1. industrial cooling system;
2. logistics and delivery system;
3. water-treatment system.

Each contains 20 observed anonymous nodes, 24–32 positive directed edges,
maximum indegree three, four local modules, and explicit chain, fork, collider,
and mediator motifs. There are no latent nodes or hidden confounders. Gold
labels and descriptions live only in the fixture/evaluation layer. The solver
receives anonymous IDs, graph edges, numeric weights, data, and the embeddings
of labels that are visible in the current fold.

## Data and masking

Each graph generates 2,000 samples from its fixed linear additive Gaussian SCM.
The fixed split is 1,200/400/400. Standardization statistics are fit on train
only and applied to every split. Pipeline edge-weight estimation uses train and
dev; test is never used for estimation, calibration, or selection.

The five fixed folds mask four of 20 labels each, so every node is masked
exactly once. All arms share data, folds, label candidates, semantic encoder,
decoder, and random seeds. Decoder alpha uses visible labels only in each fold;
this is a necessary anti-leak adapter because masked gold labels may be used
only for final evaluation.

## Arms

- `core_oracle_estimated_weights` — primary: oracle support and the formal
  pipeline's data-estimated edge weights.
- `core_oracle_true_weights` — diagnostic upper bound using fixed SCM
  coefficients.
- `core_shuffled_graph` — 20 pre-seeded node relabelings per graph with data
  columns and semantic identities held fixed.
- `core_reversed_graph` — every oracle edge reversed.
- `raw_correlation` — the existing positive raw-correlation baseline.
- `uniform` — the existing no-structure baseline.

## Metrics and statistics

Reported local metrics are Match-ACC, gold-embedding cosine, MRR, Recall@1,
Recall@5, and the existing exact top-1 decode metric. Judge-ACC is never
fabricated: this run writes complete requests and marks it pending because API
spend is outside the frozen local run.

Paired deltas use the same masked nodes. Hierarchical bootstrap resamples
graph → fold → masked node for 10,000 fixed-seed draws and reports mean delta,
95% percentile CI, paired win rate, graph-specific results, and the aggregate.

## Frozen decision rule

`GO` requires positive oracle-minus-shuffle and oracle-minus-no-graph effects
on a primary semantic metric with aggregate CI excluding zero, consistent
direction on at least two of three graphs, and a stable oracle advantage over
the reversed graph. `NO-GO` applies when correctly aligned structure does not
beat shuffled/no-graph structure and reversal is comparable or better.
Otherwise the result is `INCONCLUSIVE`. Thresholds, labels, graphs, seeds, and
hyperparameters are not changed after seeing results.

## Explicit non-goals

E0′ does not run an LLM, extract J-space, run CauScale, write activations, use
innovation residuals, discover latent nodes, use `concept@layer` nodes, perform
Task-2 latent translation, or sweep graph size.
