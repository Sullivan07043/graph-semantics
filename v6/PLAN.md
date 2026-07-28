# v6 PLAN — theory upgrades and pipeline rebuild (reference document)

Status: TRUNK-1 THEORY DRAFTED (2026-07-28); next = term-factory core (TRUNK-2). This file is the authoritative checklist for v6. Nothing below is
implemented until its box is checked. v6 starts as a full copy of v5 (frozen artifacts symlinked
from `v5/outputs/`); every change is made faithfully and at full scale — no silent downgrades.

Baseline being improved (v5, all numbers held-out unless noted): Task 1 judge .698 / match .642
vs rawcorr .595/.850; Task 2 judge .882 vs LLM-naming .796; all-13 .717/.751, T2 .930.
Known open problems inherited: metatrait decoding 5/15; riasec instability (judge .23–.43 across
runs); hierarchical graphs lose the independence tail (bigfive+GFP: 1210 → 0 pairs); no recovery
theorem.

---

## THE MAIN LINE (convergence statement, 2026-07-28)

**One principle**: a semantic embedding is a **geometric realization of the causal dependence
structure** — the embedding geometry (cosines, orthogonality, the generative map) must realize,
item by item, the dependence structure the causal model asserts.

Everything below is a corollary, not a parallel idea:
- T1 defines the object to be realized (influence structure, B = E[∂X/∂Z]);
- T2 is the central mathematical question (given anchors, is the realization unique?);
- T3 is the realization condition + whether the encoder space can support it (LoRA = enforcement)
  + where the model fails to fit (V(G,X));
- T4 is faithfulness of the generative map (use the true map, Jacobian pattern locked to the
  graph; linear+f_neg = its first-order special case);
- T5 is the conditional part of the same condition; T6 the unmodeled part.
- WeightNet / LoRA / f_neg are NOT theory: numerics, satisfiability engineering, and a special
  case of T4 respectively — demoted to implementation notes in THEORY.md.

**Anti-divergence rule (binding)**: any future addition must first answer "is it a corollary of
the realization principle?" If not, it does not enter v6.

**Engineering consequence**: the five hand-written objective terms are five instances of one
abstraction (structure pattern → moment condition). The v6 core is rewritten as a single
term-factory; the f_neg special-case branch is absorbed by T4's operator; flat and hierarchical
graphs share one conditional-independence rule (P4). Net: core shrinks and unifies; P2/P3/P5 are
read-only post-solve consumers (low coupling); **the single complexity hotspot is P1** (training
infrastructure + Jacobian-lock audit + WeightNet co-retraining) — scheduled LAST, after the
unified frame stands on P2-P5.

## PART A — THEORY CHANGES

### T1. Meaning as Jacobian influence (definition change)
Replace the informal "meaning = constrained position" with the formal primitive:
the semantics of a node is its influence structure **B = E[∂X/∂Z]** on nameable anchors
(derivative-as-dependence; same primitive as J-space at LLM scale, TC's B(J_f) at identification).
- [x] Write the definition: semantic assignment `e: nodes → S^{d-1}` such that pairwise geometry
      of `e` realizes the influence structure of the causal model.
- [x] Rewrite each v5 constraint as a moment condition on the Jacobian (generation = first-order
      expansion; independence = zero blocks; floor/anchors = magnitude conditions).
- DELIVERABLE: `v6/THEORY.md` §1, referenced by every pipeline component.

### T2. Recovery statement (the missing theorem)
There is currently NO formal guarantee that the constraint system's solution recovers true label
embeddings. Target statement (conjecture to be proven or honestly bounded):
under (i) faithful graph, (ii) bridge assumption (T3), (iii) anchor density condition (every free
node trek-connected to ≥ k labeled anchors through identifiable paths), (iv) sign conventions,
the solution is unique up to an explicit equivalence class, and the masked-node solution converges
to the true label embedding as anchor density grows.
- [x] (Prop 1/2 + Def 4 drafted; C1 stated as conjecture) Characterize the NULL SPACE of the constraint system at a solution (which directions are
      free). Hypothesis: riasec chaos and metatrait genericity are exactly large null spaces.
- [~] Linear generation core proved (Prop 1); C1 with bridge constraints open (v5 objective is quadratic + hinges; the quadratic
      core has closed-form analysis). Nonlinear case (T4) as extension or assumption.
- [x] Derive per-node identifiability score usable at runtime (→ P2).
- DELIVERABLE: `v6/THEORY.md` §2 with proofs or explicit counterexamples.

### T3. Bridge axiom as a stated assumption with a violation statistic
- [x] Formalize: |cos(e_i,e_j)| = g(dep(X_i,X_j)) for monotone g in class G, on trek-connected
      pairs; independence ⇒ orthogonality on d-separated pairs. LoRA calibration = enforcing this
      assumption on the encoder; the anchor loss bounds the enforcement's distortion (drift ≤ ε).
- [x] Define the violation statistic V(G,X) = mass of pairs where the graph claims ⊥ but data
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
- [x] Formal spec: operator class, the Jacobian penalty/parameterization that enforces the lock,
      why d-separation semantics is preserved under the locked pattern.
- DELIVERABLE: `v6/THEORY.md` §4; implementation in P1.

### T5. Conditional-independence lower tail (hierarchy fix)
Marginal d-separation vanishes under a global root (bigfive+GFP: 0 independent pairs), deleting
the entire lower tail of the bridge axiom. Fix: decorrelation applies to RESIDUALIZED embeddings —
pairs d-separated GIVEN ancestors get cos(residual_i, residual_j) → ρ̂ (partial corr), extending
the residual-anchor logic from observed pairs to all pairs with a blocking ancestor set.
- [x] Formal: which conditional statements the graph licenses; the embedding-side operator
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
- [x] ARCHITECTURE (2026-07-28, part of TRUNK-2): `v6/gen_operator.py` — shared T_theta
  conditioned on (sign, |w|, lat-lat), |w|*(base + Delta), base = e_p / f_neg(e_p), zero-init
  Delta head, sign_audit() hinge. TRAINING = TRUNK-4 (not started).
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
- [x] IMPLEMENTED (TRUNK-3): `v6/influence_decode.py` — GENERATIVE-path influence (ruling
  2026-07-28, THEORY §2.4): forward JVP of the locked operator along directed paths u → o at
  the solution (Definition 1's B), data-grounded polarity, f_neg-aware footprint, blend.
  (Original wording "∂(observed solutions)/∂u through the solve" was superseded: that
  derivative is identically zero on pinned text-bearing nodes.)
- Validation: bigfive2 metatraits (currently 5/15) and hexaco facet/factor decodes, free first
  (word inspection + match), judge last (~$1).

### P4 (← T5) Conditional-independence decorrelation
- [x] IMPLEMENTED (2026-07-28, part of TRUNK-2), then REVISED same day after held-out
  attribution: embedding-level conditional pairs measured harmful (.658 → .422 untrained;
  double enforcement against the residual channel). v6 DEFAULT is now CI_MODE=marginal_shrink
  (marginal support + shrink targets, .672); conditional statements live in the residual Gram
  channel (observed + latcon latent rows). Full conditional ci_table kept for DIAGNOSTICS only
  (V(G,X)). See THEORY §5 (one statement, one channel).
- Extend `latent_constraints.py`: residualized decorrelation pairs for graphs whose marginal
  independence set is empty/small; automatic (structure-driven support, like everything else).
- Validation: bigfive2 hierarchy (does T1 recover the flat-graph gap fully?); all-13 regression
  screen.

### P5 (← T3) Structure-adequacy diagnostic V(G,X)
- [x] IMPLEMENTED (TRUNK-3): `v6/adequacy.py` — V(G,X) from ci_table targets (marginal Def 5
  verbatim + conditional extension), per-pair localization. Graph-repair PROPOSALS ON (ruling
  2026-07-28: next phase discovers structure): `propose_repairs()` emits ranked hypotheses
  (shared-latent clusters / pairwise repairs) — proposals ONLY, the given graph stays in use
  and is never modified.

### P0 Loading-matrix estimation upgrade (graph-side input to the trunk)
The linear block of the influence structure IS the loading matrix; v5 estimates it by PC1-score
correlations (crude). Upgrade: confirmatory factor analysis (ML/WLS) on the GIVEN structure
(second-order CFA for hierarchies) via an existing library — no home-grown estimator.
Buys: (a) proper magnitudes/signs (subsumes the manual sign convention), (b) loading standard
errors → feed P2 certainty and constraint weighting, (c) model-implied correlations (Wright's
rules) → calibrate ψ (THEORY open item 2). Passes the anti-divergence test: better estimator of
Definition 1's object.

### P6 Housekeeping (no science)
- `dependence.py`: lazy auto-compute on cache miss (kills the bigfive2-crash class).
- Single entry `v6/main.py` = solve + diagnostics (certainty, adequacy) in one pass.
- Carry latcon default; port LoRA/f_neg unchanged until P1 supersedes f_neg's role.

---

## THE TRUNK (主干工程 — complete before branches; binding; REVISED 2026-07-28 per user)
  TRUNK-1  `v6/THEORY.md` §1–§4: the realization principle formalized, INCLUDING the nonlinear
           generative map (T4) — nonlinearity is part of the main line, not an appendix.  [DONE]
  TRUNK-2  Term-factory core WITH the nonlinear operator BUILT IN: one objective engine
           (structure pattern → moment condition, THEORY Def 2 R1–R4). The generation path IS
           the Jacobian-locked nonlinear operator (P1's architecture) — there is NO separate
           linear default. "Linear+f_neg" survives ONLY as the operator's zero-initialization
           (strict-refinement discipline: untrained operator ≡ v5 behavior). P4's unified
           conditional-independence support rule also lives here (constraint generation, not
           diagnostics). WeightNet differentiable-solver interface preserved; latcon default on.
           [BUILT 2026-07-28: gen_operator.py (T_theta, zero-init=linear+f_neg, sign_audit) +
           terms.py (ci_table/ci_cos/dep_floor_table) + core.py (solve_unrolled, train=True
           reaches operator AND WeightNet) + wiring (run_task1/2 pass ci + GENOP default on,
           optimize.py same contracts for the frozen arm, main.py default L2_ARM=mlp,
           run_bigfive_hier on core). Syntax-compiled only — NO run of any kind yet.]
  TRUNK-3  Diagnostics: P2 (certainty), P3 (influence-weighted decode), P5 (adequacy V(G,X)) —
           read-only post-solve consumers of the core.
           [BUILT 2026-07-28: certainty.py (Def 4 exact — active-set Jacobian rows incl.
           autograd operator Jacobians, sigma_min per free node), influence_decode.py
           (generative-path JVP influence + data-sign footprint + blend; deviation from the
           original P3 wording RATIFIED by user 2026-07-28 and canonicalized in THEORY §2.4),
           adequacy.py (Def 5 on ci_table targets; marginal verbatim + conditional extension;
           propose_repairs() ON per ruling 2026-07-28 — hypotheses only, given graph in use).
           Syntax-compiled only — NO run of any kind; validation runs (P2 falsification tests,
           P3 metatrait decode, P5 tables) await user approval.]
  TRUNK-4  Full-scale training, REVISED ORDER (user 2026-07-28: "不允许线性近似、线性训练…
           排在重训之前"):
           4a. NONLINEAR TARGET STACK first — all constraint-target MAGNITUDES move from
               Pearson to dcor (marginal CI targets, bridge floor incl. latcon latent rows,
               |w| edge conditioning); conditional residualization moves from linear lstsq to
               gradient-boosted regression, residual dependence = signed dcor; new caches for
               all 19 datasets + bigfive2 (old caches untouched). Signs stay Pearson-sign
               (polarity is binary; dcor is unsigned). DECLARED linear remnants: ALS as
               initializer only (THEORY §4.3); PC1 latent scores as conditioning summaries
               (proper replacement = Task-3 TC layer).
           4a+. RCHAN=hard default (ruling 2026-07-28): residual coordinate by IDENTITY
               (r_c = e_c - sum T(e_p)); no auxiliary r variables; soft form kept for
               attribution only. Training restarted on this form (soft-form run discarded
               per the "don't train on a form about to be replaced" principle).
           4b. Trainer rework: per-epoch checkpoint PAIRS + epoch selection by dev fold-4
               MATCH (embedding val loss measured to decouple: val .80→.22 while held-out
               match collapsed .658→.378).
           4c. Joint retrain (operator + WeightNet) on the corrected objective + nonlinear
               targets; then held-out ONCE as the report number (judge spend user-approved).
           [First attempt 2026-07-28 FAILED and was attributed: duplicated conditional pairs
           (objective, -.236) + overfit-by-val-selection (training, -.044); both fixed above.]
Branches after the trunk: P0 (CFA loading estimation — can slot before TRUNK-4 if the user
orders it), P6 (housekeeping).

**Verification rule (user order 2026-07-28, binding)**: NO smoke tests, identity checks,
verification runs, or validation passes of any kind without asking the user first. Build
faithfully, then ask what to verify.

## DISCIPLINE (carried over, binding)
- Held-out (hexaco/riasec/kims) never touches training or design decisions; generalization claims
  attach to held-out numbers only; dev reported as fit with the "seen" annotation.
- No pass/fail thresholds; full-evidence tables; adoption decided by the user.
- Every trained component zero-inits to the previous method (strict refinement).
- Free screens before any judge spend; judge cache on; OPENAI_API_KEY explicitly blanked for
  free runs; CUDA_VISIBLE_DEVICES always explicit; ≤15 concurrent eval processes.
