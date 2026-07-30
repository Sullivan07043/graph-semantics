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
| E0′ fails | The unchanged local solver is not validated for this causal-DAG bridge; diagnose before E1/E2 |
| E0′ passes, E1 fails | J-space measurement is the bottleneck |
| E0′ and E1 pass, E2 fails | CauScale graph discovery is the bottleneck |
| All pass | The full Task 3 pipeline is supported |

> E0′ returned NO-GO; E0″ classified the failure as Category F, so E1/E2 remain paused.

---

# Slide 4｜E0′: Oracle Graph Gives No Stable Advantage

## Research question

> Can the frozen local solver use a correct causal DAG to improve masked semantic recovery?

### Test

- Three hand/LLM-designed synthetic linear-Gaussian SCM worlds: industrial cooling, logistics, and water treatment.
- 20 observed nodes per graph; 26–29 edges; no latent nodes or hidden confounders.
- 2,000 samples per graph; 5 folds with 4 of 20 labels masked.
- Oracle estimated/true weights vs shuffled, reversed, raw correlation, and uniform.

> **Bundle:** Frozen local reproduction; original release artifacts unavailable.

### Primary gold-embedding cosine

| Comparison | Delta |
|---|---:|
| Oracle vs shuffled | +0.0131 |
| Oracle vs reversed | -0.1305 |
| Oracle vs uniform | -0.1698 |
| Oracle vs raw correlation | -0.1704 |

### Result

- No stable oracle advantage over shuffled graphs; consistent advantage on **0 of 3 graphs**.
- Reversed and no-graph baselines achieved higher primary cosine.

\[
\boxed{\text{NO-GO — E1 blocked}}
\]

---

# Slide 5｜Initial E0′ Hypothesis: Graph-Type Mismatch

## Tasks 1 and 2 predominantly used measurement/keying graphs

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
| Task 2 Judge-ACC, mean over 7 datasets | 0.930 | LLM naming: 0.695 |

### Scope of existing evidence

> Task 1/2 evidence is mainly measurement/keying. TLVD is a mixed directed exception but stays semantically narrow; gains are not universal. Transfer of the unchanged solver to semantically diverse mechanism DAGs is unvalidated.

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

The unchanged solver imposes an embedding-space generation equation:

\[
e_{\text{child}}
\approx
\sum_p w_{p,c}e_p
\]

For mechanism-level edges, causal influence does not necessarily imply this semantic composition.

### Initial interpretation after E0′

- Tasks 1 and 2 are **not disproved**.
- J-space and CauScale are **not disproved**.
- The unsupported direct-transfer route is:

\[
\boxed{
\text{mechanism-style directed graph}
\rightarrow
\text{unchanged Task 1/2 solver}
}
\]

### Relevance to Task 3

> The planned J-space/CauScale output is closer in role to a mechanism-style directed dependency graph than to a measurement hierarchy. Because causal sufficiency in J-space is not established, it should not yet be treated as a validated causal DAG.

---

# Slide 6｜E0″ Conclusion: Pause Task 3

\[
\boxed{
\text{F — broader solver or metric failure}
}
\]

## Why?

| Audit finding | Result |
|---|---|
| Orientation and E0′ parity | Pass: 15/15 canonical solves; 60/60 oracle node metrics match |
| Bundle check | Selected API-free trend reproduced; Judge pending |
| Root effect | Roots explain **83.67%** of reversal's cosine advantage, but not the full failure |
| Metric/control diagnosis | Visible-parent metrics conflict; generation is not isolated; same-module positive control fails |

### Updated conclusion

- E0′ remains a **NO-GO for the frozen local solver/evaluation stack** on three synthetic causal DAGs.
- E0″ does not support graph-type mismatch as a sufficient explanation.
- Material cross-metric conflict and the failed positive control lead to **Category F: broader solver or metric failure**.

> Tasks 1 and 2, J-space, CauScale, and causal graphs in general are **not** disproved.

### Decision

> **Pause Task 3: no E0′ rerun, no old E1, and no S0.**

### Next step

> **Near term:** return to Tasks 1 and 2. Before reviving Task 3 or reusing the unchanged solver on mechanism-style DAGs, audit the solver/evaluation stack and require a passing positive control.
