# v6 PLAN — theory upgrades and pipeline rebuild (reference document)

Status: PLANNING (2026-07-28). This file is the authoritative checklist for v6. Nothing below is
implemented until its box is checked. v6 starts as a full copy of v5 (frozen artifacts symlinked
from `v5/outputs/`); every change is made faithfully and at full scale — no silent downgrades.

Baseline being improved (v5, all numbers held-out unless noted): Task 1 judge .698 / match .642
vs rawcorr .595/.850; Task 2 judge .882 vs LLM-naming .796; all-13 .717/.751, T2 .930.
Known open problems inherited: metatrait decoding 5/15; riasec instability (judge .23–.43 across
runs); hierarchical graphs lose the independence tail (bigfive+GFP: 1210 → 0 pairs); no recovery
theorem.

---

## PART A — THEORY CHANGES

### T1. Meaning as Jacobian influence (definition change)
Replace the informal "meaning = constrained position" with the formal primitive:
the semantics of a node is its influence structure **B = E[∂X/∂Z]** on nameable anchors
(derivative-as-dependence; same primitive as J-space at LLM scale, TC's B(J_f) at identification).
- [ ] Write the definition: semantic assignment `e: nodes → S^{d-1}` such that pairwise geometry
      of `e` realizes the influence structure of the causal model.
- [ ] Rewrite each v5 constraint as a moment condition on the Jacobian (generation = first-order
      expansion; independence = zero blocks; floor/anchors = magnitude conditions).
- DELIVERABLE: `v6/THEORY.md` §1, referenced by every pipeline component.

### T2. Recovery statement (the missing theorem)
There is currently NO formal guarantee that the constraint system's solution recovers true label
embeddings. Target statement (conjecture to be proven or honestly bounded):
under (i) faithful graph, (ii) bridge assumption (T3), (iii) anchor density condition (every free
node trek-connected to ≥ k labeled anchors through identifiable paths), (iv) sign conventions,
the solution is unique up to an explicit equivalence class, and the masked-node solution converges
to the true label embedding as anchor density grows.
- [ ] Characterize the NULL SPACE of the constraint system at a solution (which directions are
      free). Hypothesis: riasec chaos and metatrait genericity are exactly large null spaces.
- [ ] State and prove the linear-SEM case first (v5 objective is quadratic + hinges; the quadratic
      core has closed-form analysis). Nonlinear case (T4) as extension or assumption.
- [ ] Derive per-node identifiability score usable at runtime (→ P2).
- DELIVERABLE: `v6/THEORY.md` §2 with proofs or explicit counterexamples.

### T3. Bridge axiom as a stated assumption with a violation statistic
- [ ] Formalize: |cos(e_i,e_j)| = g(dep(X_i,X_j)) for monotone g in class G, on trek-connected
      pairs; independence ⇒ orthogonality on d-separated pairs. LoRA calibration = enforcing this
      assumption on the encoder; the anchor loss bounds the enforcement's distortion (drift ≤ ε).
- [ ] Define the violation statistic V(G,X) = mass of pairs where the graph claims ⊥ but data
      dependence is high (the unmodeled-confounder signature). This is the structure-adequacy
      diagnostic; also quantifies the "extend/correct the graph?" question.
- DELIVERABLE: `v6/THEORY.md` §3; V(G,X) implemented in P5.

### T4. Nonlinear generation with Jacobian-locked structure
Replace the linear SEM patchwork (constant w + f_neg on negative edges) with a trainable
nonlinear generation operator g whose **Jacobian sparsity and sign pattern is locked to the
graph**: ∂g_c/∂e_p ≠ 0 iff p ∈ pa(c), sign(∂g_c/∂e_p) = sign(w_pc) (directional derivative along
the parent embedding). Linear+f_neg is the exact zero-init special case (identity discipline).
This is the principled revival of g_φ: the earlier failure mode (himi corruption from
unconstrained transforms) is directly addressed by the Jacobian lock.
- [ ] Formal spec: operator class, the Jacobian penalty/parameterization that enforces the lock,
      why d-separation semantics is preserved under the locked pattern.
- DELIVERABLE: `v6/THEORY.md` §4; implementation in P1.

### T5. Conditional-independence lower tail (hierarchy fix)
Marginal d-separation vanishes under a global root (bigfive+GFP: 0 independent pairs), deleting
the entire lower tail of the bridge axiom. Fix: decorrelation applies to RESIDUALIZED embeddings —
pairs d-separated GIVEN ancestors get cos(residual_i, residual_j) → ρ̂ (partial corr), extending
the residual-anchor logic from observed pairs to all pairs with a blocking ancestor set.
- [ ] Formal: which conditional statements the graph licenses; the embedding-side operator
      (project out ancestor span, then decorrelate).
- DELIVERABLE: `v6/THEORY.md` §5; implementation in P4.

### T6. Position on unmodeled confounders
No new machinery; write down what holds: residual channel absorbs confounder footprints
(partial-corr anchors), V(G,X) detects them, the independence term is the attack surface,
CauScale's causal-sufficiency assumption is a Task-3 boundary handled by the TC layer.
- DELIVERABLE: `v6/THEORY.md` §6 (one page, honest).

---

## PART B — PIPELINE CHANGES (v6 work packages)

### P1 (← T4) Jacobian-locked nonlinear generation  [THE substantive trained component]
- New `v6/gen_operator.py`: operator g (small conditioned MLP per edge-type), zero-init to
  linear+f_neg; Jacobian lock enforced by construction (masked input routing) + penalty audit.
- Trained across the 16 dev graphs (folds 0–3/4 discipline unchanged, held-out untouched);
  the training objective includes the Jacobian sign/sparsity audit terms.
- WeightNet NOTE: solver dynamics change under g ⇒ WeightNet must be retrained jointly or after
  (decide at implementation; both runs are dev-only).
- Screens (free): all-13 match + per-dataset guard vs v5; decode-word inspection on himi/tlvd
  (the historical corruption case); judge only at the end on the final candidate.
- Risk register: himi-style semantic corruption (watch decodes); training instability (clip,
  zero-init); overfit to outer cosine (lesson recorded — screen at MATCH level, not embedding).

### P2 (← T2) Identifiability / certainty diagnostics
- New `v6/certainty.py`: at solution, constraint-system Jacobian spectrum; per-node certainty
  score = alignment of the node's solution with the well-determined subspace.
- Output: per-translation certainty attached to records; flags low-certainty latents instead of
  confidently decoding them.
- Validation (free): certainty must predict (a) riasec cross-run variance, (b) metatrait failures,
  (c) judge correctness correlation on dev. No thresholds — report the correlations.

### P3 (← T1) Influence-weighted decoding for deep latents
- New `v6/influence_decode.py`: ∂(observed solutions)/∂(latent u) through the differentiable
  solve → influence weights over text-bearing nodes → footprint-assisted decode for latents ≥ 2
  hops from text (metatraits; facets optional).
- Validation: bigfive2 metatraits (currently 5/15) and hexaco facet/factor decodes, free first
  (word inspection + match), judge last (~$1).

### P4 (← T5) Conditional-independence decorrelation
- Extend `latent_constraints.py`: residualized decorrelation pairs for graphs whose marginal
  independence set is empty/small; automatic (structure-driven support, like everything else).
- Validation: bigfive2 hierarchy (does T1 recover the flat-graph gap fully?); all-13 regression
  screen.

### P5 (← T3) Structure-adequacy diagnostic V(G,X)
- New `v6/adequacy.py`: compute V(G,X) per dataset; report alongside results. Graph-repair
  proposals stay OFF (the extend/correct decision is not made here).

### P6 Housekeeping (no science)
- `dependence.py`: lazy auto-compute on cache miss (kills the bigfive2-crash class).
- Single entry `v6/main.py` = solve + diagnostics (certainty, adequacy) in one pass.
- Carry latcon default; port LoRA/f_neg unchanged until P1 supersedes f_neg's role.

---

## ORDER OF WORK
1. THEORY.md §1–§3 drafts (T1–T3) — before any code, since P1–P3 implement them.
2. P2 + P3 (free, fast, each closes an open problem; no training).
3. P4 (small, closes the hierarchy tail).
4. P1 (the heavy build: spec → implement → train → screen).  P2's diagnostics gate P1's audit.
5. P5 + P6 alongside.
6. Full evaluation: free metrics first, held-out primary, judge once on the final configuration.

## DISCIPLINE (carried over, binding)
- Held-out (hexaco/riasec/kims) never touches training or design decisions; generalization claims
  attach to held-out numbers only; dev reported as fit with the "seen" annotation.
- No pass/fail thresholds; full-evidence tables; adoption decided by the user.
- Every trained component zero-inits to the previous method (strict refinement).
- Free screens before any judge spend; judge cache on; OPENAI_API_KEY explicitly blanked for
  free runs; CUDA_VISIBLE_DEVICES always explicit; ≤15 concurrent eval processes.
