# pipeline_v3 — Unified Bridge-Constraint Version

Designed on 2026-07-15 with theory-first discipline.

This directory contains the independent implementation of the newer pipeline. Shared root modules
remain in use: graph, testbeds, pool, encode, judge, metrics, splice_decode, and negop. The
experiments/ directory is a historical experiment archive. The old run_task1 and run_task2
runners are retained to reproduce previously reported numbers.

## Method and differences from earlier versions

**Explicit bridge axiom:** semantic similarity \(\cos(e_i,e_j)\) is a monotone function of the
statistical dependence strength \(\operatorname{dep}(X_i,X_j)\). The conditional form corresponds
to conditional dependence. The old independence decorrelation term for the lower tail and the
residual Pearson anchor for the conditional case are special cases of the same principle. The
upper-tail dependence bound previously failed under simple vector negation and was re-evaluated in
the semantic f_neg basis.

Components:

1. **dependence.py** — Dependence-matrix infrastructure for Pearson correlation, distance
   correlation, and k-nearest-neighbor mutual information, each at two levels: marginal and
   conditional after residualization on parent scores. Per-dataset matrices are cached as NPZ
   files. This also implements the MI/CMI direction described as Yujia's second constraint.
2. **bridge.py** — Unified bridge constraints with graph-based layering: no-trek pairs use the
   marginal layer, while sibling pairs use the conditional layer. The loss combines a lower tail
   for independent-pair cosine squared, an upper-tail hinge
   \(\kappa\operatorname{dep}_q-|\cos|\), and conditional residual alignment. Pearson, distance
   correlation, and mutual information are alternative comparison arms.
3. **solve.py** — Objective combining signed generation through f_neg, residual consistency,
   bridge constraints, and unit norm. Optimization uses an ALS closed-form initialization followed
   by deterministic Adam refinement.
4. **translate.py** — Two parallel Task 2 readouts: decoding the optimized latent embedding and a
   forward-beta readout based on path-weight aggregation, with f_neg on reverse paths. The primary
   readout is chosen from the three-way comparison evidence.
5. **intervene.py** — Complete swap intervention with judged and positive-pole-reference variants,
   used as the third formal metric.
6. **run_pipeline.py** — Unified Task 1, Task 2, and swap entry point. Records are written to disk.
   The LLM-naming baseline is fold-aligned and sees only visible child nodes.

## Experimental discipline

Masking 20 percent is part of the task definition. All fitting and design decisions use the 16 DEV
datasets; the three held-out datasets, HEXACO, RIASEC, and KIMS, are evaluated only once. Every new
constraint arm requires a mechanism hypothesis, a preregistered success threshold, and a
same-batch control. Losing candidates are rejected immediately. Global components such as the
encoder, dictionary, and f_neg use only WordNet and DEV data for training.

## Stage-2 bridge-constraint preregistration

- Success criterion: mean DEV-pool Task 1 judge or match improves by at least 0.03 over the current
  frozen configuration of generation, f_neg, and residual Pearson alignment.
- Guardrail: Himi and GCBS may regress by at most 0.05.
- Comparison arms: Pearson, distance-correlation, and mutual-information bridge variants, plus the
  current frozen baseline, evaluated in the same batch.
- Failure disposition: the complete bridge package is not frozen into the main line; only the
  dependence.py infrastructure and diagnostic conclusions are retained.
