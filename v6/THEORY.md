# v6 THEORY — Semantic Embeddings as Geometric Realizations of Causal Dependence

Status: DRAFT 2 (2026-07-28; L2-formalized). Sections 1–4 are the trunk spec; every v6 component implements a
definition or condition stated here. Claims are labeled **Definition / Assumption / Proposition
(proved) / Conjecture (not proved)**. Nothing unlabeled is load-bearing.

Notation. Causal DAG `G` over nodes `V = Z ∪ X` (latents `Z`, observed `X`, |X| = m, |Z| = k),
signed edge set `E`. Structural model: each node `V_c = f_c(V_{pa(c)}, ε_c)` with jointly
independent noises. Data: n i.i.d. samples of `X`. Encoder: a fixed map `Φ : texts → S^{d-1}`.
A subset `L ⊆ X` carries labels with embeddings `a_i = Φ(ℓ_i)`. A *semantic assignment* is
`e : V → R^d` with `e_i = a_i` for `i ∈ L`.

**The two geometries.** Let `H = L²(P)` be the Hilbert space of centered unit-variance random
variables with `⟨U,W⟩_H = E[UW]`, and write `V̄_i` for the standardized version of `V_i` (latents
enter through their structural definition; empirically through their scores). Two standard facts
anchor everything below:
  (F1) `corr(V_i, V_j) = cos_H(V̄_i, V̄_j)` — correlation IS the cosine of the angle in `H`;
  (F2) the partial correlation `ρ_{ij·S}` IS the cosine between the residuals of `V̄_i, V̄_j`
       after orthogonal projection onto `span{V̄_s : s ∈ S}` in `H` — probabilistic conditioning
       IS orthogonal projection in `L²`.
The other geometry is the semantic sphere `S^{d-1} ⊂ R^d` with its cosine. `D_{ij}` denotes the
chosen dependence functional (default `|cos_H| = |corr|`; a declared modeling choice — MI/dcor
variants were tested and added nothing on Likert data), and `D_{ij|S}` its conditional version
(default `|ρ_{ij·S}|`).

---

## §0 Plain statement (read this first)

In three sentences. The causal graph tells us which dependence relations must hold among the
variables — which pairs are related, how strongly, which become unrelated once you account for
their causes, and who generates whom. We ask for the arrangement of points on the semantic
sphere whose similarity structure obeys exactly those relations, with the known labels pinned in
place. The unknown nodes' positions are then forced, and reading a position off against a
dictionary is translation.

### §0.1 Where causality enters (and where it does not)

Observational data alone pins down only DEPENDENCE structure — this is a basic fact of causal
inference, not a choice of ours; (R2)/(R3) are written in correlation language because that is
what the data side can supply. The causal content enters through the GIVEN graph, at three
places:

1. **Constraint support is causal.** Which pairs must be orthogonal, given whom, and which pairs
   are trek-connected are consequences of the causal Markov condition of `G` — d-separation is a
   causal-graph concept, not a data pattern. We enforce the GRAPH's claims, not the data's
   appearance; conflicts between the two are precisely what `V(G,X)` (Definition 5) measures.
2. **Generation is directed.** (R1) runs along the causal arrows: Markov-equivalent DAGs share
   dependence structure but yield DIFFERENT generation systems and different realizations. This
   is exactly where the given-graph task setting injects the information that observational data
   cannot supply (the choice within the Markov equivalence class).
3. **Validation is interventional.** The swap test performs do-operations on latent
   representations and checks downstream semantics — a causal criterion with no correlational
   counterpart.

What we do NOT claim: no structure discovery (the graph is given), no causal-effect estimation
(no do-effects are computed). The correct classification of this method is **dependence geometry
constrained by a causal model** — observational fitting through the causal model's dependence
implications, with direction and intervention semantics supplied by the model.

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

**Definition 2 (semantic realization).** Fix a strictly increasing `ψ : [0,1] → [0,1]` with
`ψ(0) = 0` (the *similarity transfer function*; monotone only — order-preserving, not
isometric, since the encoder's similarity scale need not equal the correlation scale) and a
tolerance `η ≥ 0`. An assignment `e` is an *(η, ψ)-realization* of `(G, P)` if the map
`ι : V̄_i ∈ H ↦ e_i ∈ S^{d-1}` is an **approximate similarity-structure homomorphism that
intertwines the projection operators**, together with a graph-faithful generative
parameterization:

- **(R1) Generation.** For every non-root `c`: `e_c = g_c(e_{pa(c)}) + r_c`, `‖r_c‖ ≤ η`, where
  `g_c` is *Jacobian-locked to G*: `∂g_c/∂e_p ≠ 0` iff `p ∈ pa(c)`, and the directional
  derivative along each parent respects the edge sign.
- **(R2) Gram correspondence (marginal).** For all `(i,j)` in the support set `T` (pairs with
  nonzero model-implied dependence, i.e. trek-connected in `G`):
  `| |cos_{R^d}(e_i, e_j)| − ψ(|cos_H(V̄_i, V̄_j)|) | ≤ η`.
  By (F1) the argument of ψ is `D_{ij}`. Pairs d-separated by ∅ have model-implied dependence 0,
  so `|cos_{R^d}(e_i, e_j)| ≤ η` — the zero end is a special case, not a separate condition.
- **(R3) Projection intertwining (conditional).** For any pair `(i,j)` with blocking set `S`,
  writing `P_S^H` for projection onto `span{V̄_s}^⊥` in `H` and `P_S^e` for projection onto
  `span{e_s : s ∈ S}^⊥` in `R^d`:
  `| cos_{R^d}(P_S^e e_i, P_S^e e_j) − sgn · ψ(|cos_H(P_S^H V̄_i, P_S^H V̄_j)|) | ≤ η`.
  By (F2) the argument of ψ is `|ρ_{ij·S}|`. (R2) is the case `S = ∅`; v5's residual-anchor
  term is the case `S = pa` applied to the residual vectors; d-separation given `S` implies
  `ρ_{ij·S} = 0`, making conditional orthogonality another special case.
- **(R4) Boundary.** `e_i = a_i` on `L`.

**One-sentence characterization.** `e` transports the Gram structure of the random variables in
`L²(P)` to the Gram structure of the embeddings on the sphere, up to the monotone
reparameterization ψ and tolerance η, in a way that commutes with orthogonal projection
(= conditioning). The entire method is then: *find the point configuration on the sphere whose
Gram structure realizes the (conditional) correlation structure of the causal model, pinned at
the anchors.*

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
  The influence readout is defined **generatively** (ruling 2026-07-28): for latent `u` and
  observed `o`, `B_{uo}` is the Jacobian of the composed generative map along all directed
  paths `u → … → o`, evaluated at the solution — Definition 1's `B` on the embedding side,
  estimated by forward JVP through the locked operator `T_θ`. It is deliberately NOT the
  derivative of the solver output: labeled nodes are pinned inputs of the solve, so that
  derivative vanishes identically on exactly the text-bearing nodes, which is degenerate for
  the readout's purpose; the generative Jacobian is defined on all descendants.

---

## §3 The bridge assumption, its enforcement, and its violation statistic

**Assumption A1 (encoder realizability).** There exist a strictly increasing `ψ` and `η₀`
small such that the *true* label embeddings `{Φ(ℓ_i)}` satisfy (R2)–(R3) with tolerance `η₀` on
labeled pairs — i.e., the restriction of `ι` to labeled nodes is already an approximate
similarity-structure homomorphism. In words: language, as embedded by `Φ`, mirrors the
dependence structure of the measured system up to a monotone rescaling. Define the *empirical
realizability loss* `R(Φ) = mean over labeled pairs of the (R2)/(R3) violation`; A1 asserts
`R(Φ) ≤ η₀`, and A1 is thereby testable on labeled data before any solving.

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

## §5 Conditional-independence tail: one statement, one channel (REVISED 2026-07-28)

**Principle.** Every independence statement the graph licenses is enforced exactly once, in the
coordinates the model itself defines. The generative decomposition `e_c = Σ_p T_θ(e_p) + r_c`
supplies a conditional coordinate for every generated node: `r_c` *is* "what remains of `c`
given its parents". Accordingly:

- **Marginal statements** (`S = ∅`): embedding-cosine decorrelation, target
  `shrink(D_ij)` — the graph's zero unless the data dependence clears the `2/√n` noise floor,
  in which case the data value is kept and `V(G,X)` records the conflict.
- **Conditional statements**: residual-coordinate matching, `cos(r_i, r_j) → D_{ij·pa}`
  (partial dependence from data), for observed AND latent generated nodes (the latent rows are
  the latcon augmentation). Completeness: for any non-ancestral pair, conditioning each node on
  its own parents blocks every trek, so this channel covers the entire conditional tail;
  root–root pairs are marginal or trek-connected and are covered by (R2).

**What was rejected, and why (measured).** The earlier draft enforced conditional statements a
SECOND time on span-projected embeddings (`cos(P_S^⊥ e_i, P_S^⊥ e_j) → ψ(D_{ij|S})`).
This double-loads the degrees of freedom the generation term is shaping, dilutes the marginal
term (shared mean), and projects against ancestor spans that are themselves moving during the
solve. Held-out attribution (2026-07-28, Task 1 match, untrained arms): v5 objective .658;
marginal-only (v5 semantics, new implementation) .658 — digit-for-digit, certifying the
implementation; marginal + shrink targets **.672** (adopted); + duplicated conditional pairs
.422; + training on the duplicated objective .378. The duplicated channel is retained in code
for diagnostics only (`V(G,X)` localization), never in the objective.

**Hierarchy status.** Under a global root the marginal support shrinks (bigfive+GFP: 1210 → 0
pairs); the conditional structure is then carried by the residual channel's latent rows —
which is exactly the configuration that produced the hierarchy pilot result (hier + latcon
match .740 > flat .720), with no embedding-level conditional pairs involved.

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
