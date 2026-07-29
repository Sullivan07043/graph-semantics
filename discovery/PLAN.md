# Discovery Phase PLAN (v2, simplified 2026-07-28 per user: no invented apparatus)

The task is simple: input data (+labels for downstream only), discover the structure with
RLCD, check it with our own diagnostics, use it.

1. INPUT: X per dataset. Labels never enter discovery; they are for downstream
   completion/translation as always.

2. DISCOVER: RLCD (project lineage code from archive/, else the authors' release) with its
   default parameters, per dataset -> G_hat: latent count, latent-observed assignment,
   latent hierarchy; directions only where the rank rules identify them (a property of the
   math, not a process rule). FOFC/BPC optional sanity baselines.

3. CHECK AND EXPAND (the user's ruling: diagnostics are the check):
   - V(G_hat, X): remaining above-noise dependence mass; repair proposals say where to
     extend (add shared latent / latent-latent edge / split cluster). Apply, re-run, until
     no above-noise mass remains or proposals stop reducing it.
   - cert on G_hat: which discovered latents the evidence pins down; unpinned ones are
     reported as such.

4. USE: unchanged v6 pipeline on G_hat — completion (T1) and translation (T2) — reported
   next to the published-graph numbers (free match first; judge on approval). Discovered
   latents aligned to reference constructs get judged against those names; genuinely new
   layers (GFP/g/common-EF style) get decode + swap, no fabricated ACC.

Report per dataset: discovered vs published structure (matches, additions, differences),
V before/after expansion, cert table, T1/T2 numbers.

Code: discovery/run_discovery.py + RLCD integration only. No sample splits, no design-freeze
staging, no acceptance thresholds, no changes to v6.
