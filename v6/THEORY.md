# v6 THEORY — Semantic Embeddings as Geometric Realizations of Causal Dependence

Status: DRAFT 1 (2026-07-28). Sections 1–4 are the trunk spec; every v6 component implements a
definition or condition stated here. Claims are labeled **Definition / Assumption / Proposition
(proved) / Conjecture (not proved)**. Nothing unlabeled is load-bearing.

Notation. Causal DAG `G` over nodes `V = Z ∪ X` (latents `Z`, observed `X`, |X| = m, |Z| = k),
signed edge set `E`. Structural model: each node `V_c = f_c(V_{pa(c)}, ε_c)` with jointly
independent noises. Data: n i.i.d. samples of `X`. Encoder: a fixed map `Φ : texts → S^{d-1}`.
A subset `L ⊆ X` carries labels with embeddings `a_i = Φ(ℓ_i)`. A *semantic assignment* is
`e : V → R^d` with `e_i = a_i` for `i ∈ L`. `D_{ij}` denotes a population dependence magnitude
between `V_i, V_j` (linear-Gaussian default: `|corr|`), and `D_{ij|S}` its conditional version
given `S ⊆ V`.

---

## §1 The realization principle

### 1.1 The object to be realized

**Definition 1 (influence structure).** For the structural model above, the *influence
structure* is the signed pattern `B` with entries `B_{cp} = E[∂ f_c / ∂ V_p]` for `p ∈ pa(c)`
(direct), extended to arbitrary pairs by summing path products along directed paths (total
influence), and to symmetric dependence by trek sums. Under linear-Gaussian SEM, trek-summed
influence coincides with covariance (Wright's path rules); in general it is the first-order
footprint of the mechanism. This is TC's identified object `B(J_f)` and the object J-space reads
against token anchors; we take it, not raw correlation, as the primitive carrier of meaning.

### 1.2 What "realization" means

**Definition 2 (semantic realization).** Fix a monotone *bridge function* `ψ : [0,1] → [0,1]`
with `ψ(0)=0`, and a tolerance `η ≥ 0`. An assignment `e` is an *(η, ψ)-realization* of `(G, P)`
if:

- **(R1) Generation.** For every non-root `c`: `e_c = g_c(e_{pa(c)}) + r_c`, `‖r_c‖ ≤ η`, where
  the map `g_c` is *Jacobian-locked to G*: `∂g_c/∂e_p ≠ 0` iff `p ∈ pa(c)`, and the directional
  derivative along each parent respects the edge sign.
- **(R2) Marginal bridge.** For trek-connected pairs: `| |cos(e_i, e_j)| − ψ(D_{ij}) | ≤ η`.
  In particular d-separated (by ∅) pairs satisfy `|cos| ≤ η` (the zero end).
- **(R3) Conditional bridge.** For any pair `(i,j)` and blocking-relevant ancestor set `S`:
  `| cos(res_S e_i, res_S e_j) − sign·ψ(D_{ij|S}) | ≤ η`, where `res_S` projects onto the
  orthogonal complement of `span{e_s : s ∈ S}`. (R2) is the case `S = ∅`; v5's residual-anchor
  term is the case `S = pa`, applied to the residual vectors `r`.
- **(R4) Boundary.** `e_i = a_i` on `L`.

**Remark 1 (v5 as a relaxation).** The v5 objective is a Lagrangian relaxation of (R1)–(R4)
with three restrictions: `g_c` linear (`Σ w_pc e_p`, with the `f_neg` patch approximating the
sign lock on negative edges), (R3) instantiated only at `S = ∅` (independence decorrelation) and
`S = pa` on residuals (residual alignment), and (R2) enforced only at the top dependence
quantile (the floor) plus the zero end. Every v5 term is one moment condition of Definition 2:

| v5 term | condition | restriction in v5 |
|---|---|---|
| generation loss | (R1) | linear g + f_neg patch |
| residual norm μ | (R1) | `η` as soft penalty |
| residual alignment | (R3), S = pa | residual vectors only |
| independence decorrelation | (R2) zero end | marginal only |
| similarity floor | (R2) upper tail | top-30% pairs, one-sided |
| unit norm | — | numerical, not part of the theory |

### 1.3 Meaning and translation

**Definition 3 (meaning).** The *meaning* of a node `V_i` under a realization `e` is its
influence profile over the anchored subspace — operationally: the position `e_i`, whose
geometry, by (R1)–(R3), encodes exactly `V_i`'s dependence relations to all labeled nodes.
*Translation* is the decoding of `e_i` against a fixed dictionary embedded by the same `Φ`
(sparse nonnegative decomposition). Justification: under a realization, two nodes with
identical dependence profiles receive identical embeddings — meaning cannot contain more than
the structure expresses (this bounds the method: construct surplus beyond indicators is
unrecoverable in principle; cf. §6).

**Consequence for the tasks.** Task 1 and Task 2 are the same operation applied at different
nodes: read out an unanchored node's realized position. Their difficulty differs through the
*evidence geometry*: a masked observed node is one hop from sibling anchors and owns a data
column; a latent is determined as the common source of its children (R1 aggregated over
children); an upper latent is two `ψ`-compositions away from any anchor — the empirically
observed difficulty ordering (observed < latent < metatrait) is the attenuation of constraint
curvature with anchor distance (§2.4).

---

## §2 Recovery: uniqueness, null space, certainty

### 2.1 The linear generation core

Take `g_c` linear with fixed weights `W` (v5's quadratic core), `η = 0` in (R1), boundary (R4).
Stack the generation equations into `M e_free = C(a_L)` where `M` depends on `(G, W)` only.

**Proposition 1 (proved; linear algebra).** The generation+boundary system determines `e_free`
uniquely iff `M` has full column rank. The set of solutions is an affine subspace whose
direction space is `null(M)`; v5's ridge (`M^T M + 1e-6 I`) selects the **minimum-norm** element,
i.e., a canonical but *arbitrary* choice inside the null space.

*Proof sketch.* Standard least-squares theory; the ridge term makes the normal equations
strictly convex, and its minimizer converges to the min-norm solution as the ridge → 0. ∎

This identifies the failure mode precisely: **whenever `null(M)` is nontrivial, part of every
free embedding is chosen by the regularizer, not by evidence.** The (R2)/(R3) terms then act as
the only forces inside that subspace — and they are nonconvex hinges, which is why weakly
determined solutions vary across runs.

### 2.2 What (R2)/(R3) add

**Proposition 2 (proved; immediate).** Any direction `δ ∈ null(M)` that changes some claimed
cosine — i.e., `∃ (i,j)` in the constraint support with `∂ cos(e_i,e_j)/∂δ ≠ 0` at the solution
— is excluded from the residual freedom of an exact realization. Hence the *effective* freedom
of a realization is `null(M) ∩ null(J_bridge)`, the joint null space of the generation system
and the bridge-constraint Jacobian at the solution. ∎

**Conjecture C1 (recovery; NOT proved).** Call node `i` *anchor-separated* if its vector of
trek dependences to labeled nodes `(D_{iℓ})_{ℓ∈L}` differs from that of every other free node by
at least `γ > 0` in sup-norm. If every free node is anchor-separated and `ψ` is strictly
monotone with slope bounded below on the occupied range, then the (0, ψ)-realization is unique,
and under encoder consistency (Assumption A1, §3) it coincides with the true label embeddings
on masked nodes up to error `O(η + drift)`. — This is the theorem to prove or bound honestly;
the anchor-separation condition is checkable from data, which makes C1 falsifiable per dataset.

### 2.3 Certainty score (spec for the runtime diagnostic)

**Definition 4 (per-node certainty).** At a solution `e*`, let `J` be the Jacobian of the full
active constraint set w.r.t. the free coordinates. The *certainty* of node `i` is the smallest
singular value of `J` restricted to variations supported on `e_i`:
`cert(i) = σ_min( J P_i )`, `P_i` the coordinate projector. Low `cert(i)` = node `i` has
directions the evidence does not fix.

Predictions this score must reproduce (these are its falsification tests): (a) riasec's
cross-run judge variance traces to low `cert` on its type nodes; (b) metatraits have the lowest
`cert` among latents; (c) `cert` correlates with judge correctness on dev.

### 2.4 Known pathologies as null-space instances

- **riasec (circumplex).** Adjacent-type dependence is approximately rotation-invariant along
  the hexagon; a rotation of the six type embeddings along the circumplex approximately
  preserves all (R1)–(R3) quantities ⇒ an approximate symmetry direction in the joint null
  space ⇒ solution position along it is chosen by initialization/float noise — the observed
  .23–.43 cross-run judge swing. Fix class: a symmetry-breaking prior (e.g., Prediger axes as
  two anchored pseudo-nodes), i.e., shrink the null space, not reweight existing terms (both
  reweighting experiments already failed, consistently with this analysis).
- **Metatraits.** Their constraints reach anchors only through children: curvature availed to a
  metatrait direction is a composition `ψ∘ψ` of two hops, hence flat ⇒ low `cert`, generic
  decodes. Fix class: add direct influence readout across two hops (P3), which supplies
  first-order evidence bypassing the composition.

---

## §3 The bridge assumption, its enforcement, and its violation statistic

**Assumption A1 (encoder realizability).** There exist `ψ` monotone and `η₀` small such that
the *true* label embeddings `{Φ(ℓ_i)}` form an `(η₀, ψ)`-realization of `(G, P)` restricted to
labeled nodes. In words: language, as embedded by `Φ`, already mirrors the dependence structure
of the measured system, up to tolerance.

This is the load-bearing assumption of the whole method — meaning can be recovered from
structure only if structure is echoed in the semantic space. It is *testable on labeled data*:
define the empirical realizability loss `R(Φ) =` mean violation of (R2)/(R3) over labeled
pairs. LoRA calibration is precisely `min R(Φ_adapted)` subject to `drift(Φ_adapted, Φ) ≤ ε`
(the anchor loss); the drift bound is what keeps the enforcement from destroying `Φ`'s general
semantics (the measured `ε`: dictionary shift ≤ .02).

**Definition 5 (structure-adequacy statistic).** For dataset `(G, X)`:
`V(G, X) = Σ_{(i,j): G ⊨ i ⊥ j} w(D_{ij}) · 1{ D_{ij} > τ_n }`, `τ_n = 2/√n` the noise floor,
`w` increasing. `V` is the mass of data dependence that the graph *forbids* — the unmodeled-
confounder / missing-structure signature. Per-pair contributions localize the defect. `V` is the
quantitative form of the "extend/correct the graph?" question, and the attack-surface meter for
the independence term (large `V` ⇒ (R2) zero-end constraints are actively harmful).

---

## §4 The nonlinear generative map with a Jacobian lock (the trunk's operator)

### 4.1 Why linear is insufficient (two arguments)

1. **Composition.** Even if each single hop admitted a linear bridge, two-hop dependence
   composes as `ψ(ψ(·))` which is not linear; a single linear `g` cannot satisfy (R2) at both
   hops of a hierarchy simultaneously except in degenerate cases. Metatraits sit exactly there.
2. **Sign action.** Negation is a semantic operator, not scalar `−1` (measured: `−u` has
   cos −.6 to true reverse labels). v5 patches this with `f_neg` on negative edges only — a
   first-order, one-direction correction. The general map should carry the sign action
   intrinsically.

### 4.2 Operator class

`g_c(e_{pa(c)}) = Σ_{p ∈ pa(c)} T_θ(e_p ; s_{pc}, |w_{pc}|, τ_{pc})` — additive across parents,
one shared transform `T_θ` conditioned on edge sign `s`, magnitude `|w|`, and edge type `τ`
(latent-latent vs latent-observed).

- **Sparsity lock by construction:** additive per-parent routing ⇒ `∂g_c/∂e_p = ∂T_θ/∂e_p` for
  parents and ≡ 0 for non-parents. No penalty needed for the pattern.
- **Sign lock by audit:** require `⟨ T_θ(e_p;+,…), e_p ⟩ ≥ 0` and
  `⟨ T_θ(e_p;−,…), f_neg-direction ⟩ ≥ 0` on training distribution; enforced as an audit
  penalty (exact by-construction sign for an MLP is not available; the audit is part of the
  training objective and reported, not silently assumed).
- **Zero-init (identity discipline):** `T_θ` initialized so that `T = w·e_p` on positive edges
  and `|w|·f_neg(e_p)` on negative edges exactly; v5 is the epoch-0 special case.
- **Restriction declared:** additivity excludes parent-interaction terms. This is the v6 class;
  interactions would still respect the lock but are deferred until additivity's residual `V`
  and `cert` diagnostics show they are needed (anti-divergence rule).

### 4.3 What is preserved

Under the locked class, the d-separation reading of the graph is preserved: a child's embedding
is a function of its parents' embeddings and its own residual only — conditional-independence
semantics (R3) remain interpretable, ALS on the linearization remains a valid initializer, and
Proposition 1's null-space analysis applies to the linearization at any point.

---

## §5 Conditional-independence tail (one rule for flat and hierarchical graphs)

The marginal zero end of (R2) has empty support under a global root (measured: bigfive+GFP,
1210 → 0 pairs). The general rule is (R3): for every pair `(i,j)` d-separated *given* an
ancestor set `S(i,j)` (minimal blocking set from the DAG), constrain
`cos(res_S e_i, res_S e_j) → ψ(D_{ij|S})`, with `D_{ij|S}` the partial dependence from data.
Marginal independence is the special case `S = ∅`; v5's residual anchors are the projection of
this rule onto residual vectors at `S = pa`. One implementation (P4) replaces both special
cases; support is derived from the DAG automatically (blocking sets), so flat and hierarchical
graphs are treated by the same code path.

---

## §6 Unmodeled confounders (position)

Latents in `G` *are* modeled confounders; translating them is Task 2. For confounders absent
from `G`: (i) their footprint appears in `V(G,X)` (Definition 5) — detection; (ii) the residual
channel absorbs their induced covariance via (R3) at `S = pa` — damage containment; (iii) the
marginal zero end of (R2) is the attack surface — constraints wrongly applied to dependent
pairs; (iv) repair (adding latents) is a *structure* decision, out of scope for the solver by
the given-graph contract; `V`'s per-pair localization is the evidence such a decision would
consume. At Task-3 scale, CauScale's causal-sufficiency assumption makes (iii) live; the TC
autoencoder layer is the designated confounder-carrier there.

---

## Implementation notes (explicitly NOT theory)

- **WeightNet**: numerical method for finding realizations (per-node weighting of the
  Lagrangian); does not alter what a realization is.
- **LoRA**: enforcement of Assumption A1 under a drift budget; a property of `Φ`, not of the
  realization definition.
- **f_neg**: the sign action of `T_θ` at zero-init; absorbed by §4.
- **Judge/match**: estimators of translation quality; external to the theory.

## What remains open (honest list)

1. Conjecture C1 (recovery under anchor-separation) — prove, bound, or produce a
   counterexample; the linear case with (R2) as equality constraints is the tractable target.
2. The bridge function ψ: currently implicit (κ-scaled linear on the floor); a calibrated
   estimate of ψ from labeled pairs would sharpen (R2)/(R3) and C1.
3. Symmetry-breaking priors for circumplex-class graphs (riasec) — theory says shrink the null
   space; the specific prior is a modeling decision.
4. Whether additive `T_θ` suffices (§4.2's declared restriction) — decided by diagnostics, not
   taste.
