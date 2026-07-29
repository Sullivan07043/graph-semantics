# Discovery Phase PLAN (drafted 2026-07-28, FOR USER REVIEW — no code until approved)

Goal: structure discovery FROM DATA ALONE. Input = X only (partial labels never touch
discovery; they enter only downstream completion/translation and the cert post-filter).
Output = G_hat: latent count, latent-observed assignment, latent-latent skeleton with
identifiable directions, optional obs-obs edges, each part tagged with its stability.
Non-goals (binding): no arbitrary-DAG search, no LLM scale, no v6 solver changes.

## 1. Method

### D1 — RLCD primary search (on X_train only)
Rank-deficiency framework: rank tests on cross-covariance submatrices; tetrad is the rank-1
special case. Covers the full target class (latent hierarchies, latent-latent edges) and is
the project's own lineage (RLCD -> Markov-blanket-optimize; RLCD+MI variants).
- Implementation: project lineage code from archive/ if usable, else the authors' release;
  rank tests via SVD/canonical correlations with the paper's default test level.
- Declared parameters (fixed across datasets, never tuned per dataset): rank-test alpha
  (paper default), >= 3 pure indicators per latent.
- FOFC/BPC run as sanity BASELINES only.

### D2 — Orientation
Directions only where identifiable (rank orientation rules / collider tests on the latent
skeleton). Everything else stays undirected: the deliverable states the equivalence class,
not a guess.

### D3 — V-loop finite-sample refinement (still X_train only)
while above-noise mass remains in V(G_t, X_train):
  take the single highest-mass repair proposal (add shared latent over a violating component
  / add latent-latent edge / split a cluster) -> G_{t+1};
  ACCEPT only if the change reproduces in >= 70% of B=50 bootstrap re-discoveries
  (declared algorithm parameter; sensitivity to 60/80% reported once on dev).
Terminate when addressable V mass is exhausted or all proposals fail stability.
Labels are never read in D1-D3.

### Post-filter (labels enter here, after structure is frozen)
cert(i) computed under the standard fold protocol on G_hat; latents the evidence cannot pin
down are tagged "not translatable" (reported both included and excluded — nothing silently
dropped).

## 2. Sample and dataset discipline
- Per dataset: rows split 50/50 train/test, seed 0. D1-D3 see X_train only.
- Design iteration on FOUR dev datasets: bigfive (flat 5-factor), hs (g-factor candidate),
  himi (common-EF candidate), rse (single factor, the null case). tlvd kept as a stretch
  probe for obs-obs chains, not a design driver.
- hexaco/riasec/kims: untouched until the design is frozen; ONE final run.

## 3. Evaluation — four legs, in this order
1. OUT-OF-SAMPLE STRUCTURE FIT (pipeline-independent, primary):
   V(G_hat, X_test) vs V(G_published, X_test), identical tau_n and code path.
2. STABILITY: bootstrap (B=50) cluster/edge recovery table; unstable parts reported as
   "not asserted", never silently kept.
3. REFERENCE AGREEMENT: cluster-level ARI vs the published key + a per-deviation
   explanation row (a discovered GFP/g layer is a known key omission, not an error).
4. TASK UTILITY (last): official mask-20% T1/T2 with the UNCHANGED v6 pipeline on G_hat vs
   G_published. Free match first; judge only with user-approved spend.

## 4. Translation on discovered graphs
- Cluster-to-reference matching (Hungarian on indicator overlap): aligned latents are judged
  against the reference construct name; unaligned (newly discovered) latents get qualitative
  decode + swap intervention only — no fabricated ACC.

## 5. Milestones (each ends with a report to the user)
- M1: D1 running; raw structures + stability tables on the 4 dev datasets.
- M2: D3 loop implemented; refined structures + V trajectories on dev.
- M3: full four-leg dev report -> DESIGN FREEZE (user reviews before held-out).
- M4: held-out one-shot; final structure report.
- M5: translation on discovered graphs (judge spend asked separately).

## 6. Code layout (new, v6 untouched)
discovery/: run_discovery.py (single entry), rlcd.py (or vendored lineage code), vloop.py,
stability.py, evaluate.py. Read-only reuse of v6 modules (terms.ci_table mode=full,
adequacy.compute/propose_repairs, nldep caches for targets, the v6 solver for leg 4).

## 7. Declared limits and risks
- Rank tests are covariance (linear-family) statistics: structure decisions are DISCRETE and
  ride the linear identifiability theorems; continuous targets stay on the nonlinear stack.
  If the V-loop persistently finds mass rank tests cannot explain, that is recorded as
  evidence of nonlinear structure — not chased in this phase.
- Small-sample datasets (kims n=521): the stability leg is expected to be the binding
  constraint there; reported, not patched.
- obs-obs chains (tlvd family): out of D1's class; a second-stage PC-on-residuals is added
  ONLY if leg-1 V localizes obs-obs mass after the latent structure is in place.
- Directions inside the equivalence class are not claimed.
