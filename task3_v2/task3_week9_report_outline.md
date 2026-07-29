# Slide 1｜Task 3 Goal

## Extend Tasks 1 and 2 to LLM scale

- **Tasks 1 and 2:** given a graph and partial semantic labels, recover the missing observed and latent semantics.
- **Task 3:** test whether the same graph-semantic translation method can work on LLM internal representations.

### Planned pipeline

\[
\text{J-space features}
\rightarrow
\text{CauScale graph}
\rightarrow
\text{semantic translation}
\]

### Main question

> Can graph structure help translate unlabeled variables inside an LLM?

---

# Slide 2｜Problems with Last Week's Task 3

1. **Stage 1 became a difficult causal-discovery gate.**  
   Before testing semantic translation, we first had to discover and intervention-validate a causal graph from J-space.

2. **The nodes were repeated views of the same concepts.**  
   Nodes were defined as `concept@layer`, so same-concept propagation across layers naturally dominated.

3. **The 32 probe tokens did not form a known causal system.**  
   A sparse and stable cross-concept causal DAG was therefore not guaranteed.

4. **A J-space coordinate is not automatically a causal variable.**  
   It is a projection of a hidden state. Unmeasured dimensions may violate causal sufficiency, and a single-source write measures total downstream effect rather than a direct edge.

### Conclusion

> The previous design mixed two problems: causal discovery and semantic translation.

---

# Slide 3｜New Experimental Plan

Before testing the full LLM-scale pipeline, separate three questions:

1. **Can our current method use a causal DAG?**
2. **Does J-space preserve useful variable information?**
3. **Can CauScale provide a graph that is useful for semantic translation?**

### Diagnostic ladder

\[
E0′:\quad X^{SCM}+G^*
\]

Clean SCM data + oracle causal graph

\[
E1:\quad X^{J}+G^*
\]

J-space measurements + oracle causal graph

\[
E2:\quad X^{J}+\widehat{G}_{\text{CauScale}}
\]

J-space measurements + CauScale graph

### Original interpretation

| Result | Meaning |
|---|---|
| E0′ fails | Current semantic constraints do not transfer to generic causal DAGs |
| E0′ passes, E1 fails | J-space measurement is the bottleneck |
| E0′ and E1 pass, E2 fails | CauScale graph discovery is the bottleneck |
| All pass | The full Task 3 pipeline is supported |

> E0′ returned NO-GO, so E0″ was added before E1/E2 to audit the cause of failure.

---

# Slide 4｜E0′: Oracle Causal-Graph Bridge Test

## Research question

> Can the frozen Stage-3 Task-1 solver use a correct causal DAG to improve masked semantic recovery?

### Three causal micro-worlds

- Industrial cooling system
- Logistics and delivery system
- Water-treatment system

### Experimental setup

| Item | Setting |
|---|---|
| Graphs | 3 |
| Observed nodes | 20 per graph |
| Directed edges | 24–32 per graph |
| Latent nodes | None |
| Hidden confounders | None |
| SCM samples | 2,000 per graph |
| Masking | 5 folds; 4 of 20 labels masked per fold |

### Experimental arms

- Oracle graph + estimated weights
- Oracle graph + true weights
- Shuffled graph
- Reversed graph
- Raw correlation
- Uniform baseline

---

# Slide 5｜E0′ Result: NO-GO

## Primary cosine comparison

| Comparison | Delta | 95% CI |
|---|---:|---:|
| Oracle vs shuffled | +0.0131 | [-0.1106, 0.0868] |
| Oracle vs reversed | -0.1305 | [-0.2747, -0.0484] |
| Oracle vs uniform | -0.1698 | [-0.3295, -0.0826] |
| Oracle vs raw correlation | -0.1704 | [-0.3287, -0.0839] |

### Main observations

- Oracle graph had **no stable advantage** over shuffled graphs.
- Reversed graph performed significantly better than oracle.
- Uniform and raw correlation also performed significantly better.
- The oracle advantage was positive on **0 of 3 graphs**.

\[
\boxed{\text{NO-GO}}
\]

### Decision

> Do not proceed directly to E1.

---

# Slide 6｜Initial E0′ Hypothesis: Graph-Type Mismatch

## Tasks 1 and 2 used a different type of graph

Their datasets mainly use:

- questionnaire scoring keys;
- factor-to-item graphs;
- psychometric hierarchies;
- published measurement structures.

Connected nodes often belong to the same semantic family.

### Existing results

| Task | Ours | Best comparison |
|---|---:|---:|
| Task 1 Judge-ACC, all 13 datasets | 0.717 | Raw correlation: 0.610 |
| Task 2 Judge-ACC, mean | 0.930 | LLM naming: 0.695 |

## E0′ used mechanism-level causal edges

Example:

\[
\text{blockage}
\rightarrow
\text{pressure}
\rightarrow
\text{temperature}
\rightarrow
\text{shutdown}
\]

A causal edge does not imply that two node meanings should be close in embedding space:

\[
\text{causal relation}
\neq
\text{semantic composition}
\]

### Initial interpretation after E0′

- Tasks 1 and 2 are **not disproved**.
- J-space and CauScale are **not disproved**.
- The failed route is:

\[
\boxed{
\text{CauScale causal graph}
\rightarrow
\text{unchanged Task 1/2 solver}
}
\]

### Initial proposed next step

> Redesign how causal structure constrains semantic embeddings, instead of treating every causal edge as a semantic-generation edge.

The graphs are structurally related but semantically different.
Task 1/2 mainly use measurement and hierarchy graphs, where connected nodes usually share a semantic family. E0 uses mechanism-level causal DAGs, where causal influence does not imply semantic similarity. This graph-type shift was the leading explanation after E0′, but E0′ alone could not establish that it was the main cause. E0″ therefore tests whether this explanation is sufficient.
The CauScale graph is closer to E0’s mechanism-level causal DAG than to the measurement and hierarchy graphs used in Tasks 1/2. In the current J-space setting, it is more safely interpreted as a directed dependency graph over internal coordinates, so it should not be passed unchanged into the existing semantic solver.

causcale: causal sufficiency, not guaranteed in jspace

---

# Slide 7｜E0″ Audit: Validity Passes, but Controls Conflict

## Validity checks

| Check | Result |
|---|---|
| Orientation from JSON → adapter → SCM/ALS/generation | Pass |
| Solver and metric parity | 15/15 solves; 60/60 node metrics |
| Selected Stage-3 bundle behavior | Reproduced |

## Key diagnostic evidence

| Test | Main result | Interpretation |
|---|---|---|
| Roots | 18 roots explain **83.67%** of reversal's cosine advantage | Important, but not sufficient |
| Visible-parent nodes | Gold cosine **-0.0656**; centered cosine **+0.5101**; MRR **+0.1027** vs uniform | Material metric conflict |
| Generation ablation | Removing generation changes gold cosine by **-0.4443** | Generation is not the isolated failure |
| Same-module control | Centered/retrieval gains coexist with adverse cosine/margin | Positive control fails |

### Audit scope

- Same E0′ graphs, folds, checkpoints, solver, and decoder.
- No retraining, coefficient tuning, J-space, CauScale, or E1.

> Orientation error, bundle drift, root-only, generation-only, and causal-graph-only explanations are all insufficient.

---

# Slide 8｜E0″ Final Result: Category F

\[
\boxed{
\text{F — broader solver or metric failure}
}
\]

### Updated conclusion

- E0′ remains a **NO-GO for the frozen local solver/evaluation stack** on three synthetic causal DAGs.
- Orientation and bundle checks pass, but local semantic and retrieval metrics disagree.
- The graph-type shift may contribute, but it is not a sufficient explanation.

### What this does not show

> Tasks 1 and 2, J-space, CauScale, and causal graphs in general are not disproved.

### Decision

> **No E0′ rerun, no old E1, and no S0. Pause Task 3 and audit the solver/evaluation stack.**

### Next step

> Reconcile gold cosine with centered/retrieval metrics and establish a passing positive control before redesigning causal constraints or returning to J-space.
