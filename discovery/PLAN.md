# Discovery Phase PLAN (v2, simplified 2026-07-28 per user: no invented apparatus)

The task is simple: input data (+labels for downstream only), discover the structure with
RLCD, check it with our own diagnostics, use it.

1. INPUT: X per dataset. Labels never enter discovery; they are for downstream
   completion/translation as always.

2. DISCOVER: RLCD (project lineage code from archive/, else the authors' release) with its
   default parameters, per dataset -> G_hat: latent count, latent-observed assignment,
   latent hierarchy; directions only where the rank rules identify them (a property of the
   math, not a process rule). FOFC/BPC optional sanity baselines.

3. CHECK = READ-ONLY REPORT (REVISED, user ruling 2026-07-28 evening): the discovered
   structure passes downstream UNCHANGED. The V-driven edit loop was tried and RETIRED:
   V is one-sided (penalizes violated independence claims, never rewards held ones), so
   claim-poor graphs trivially minimize it and the loop collapsed every structure toward a
   single factor. V and cert are reported as descriptions of the graph-data relationship,
   never used to edit the graph. himi (n=202) dropped from the phase.

4. USE: unchanged v6 pipeline on G_hat — completion (T1) and translation (T2) — reported
   next to the published-graph numbers (free match first; judge on approval). Discovered
   latents aligned to reference constructs get judged against those names; genuinely new
   layers (GFP/g/common-EF style) get decode + swap, no fabricated ACC.

Report per dataset: discovered vs published structure (matches, additions, differences),
V before/after expansion, cert table, T1/T2 numbers.

Code: discovery/run_discovery.py + RLCD integration only. No sample splits, no design-freeze
staging, no acceptance thresholds, no changes to v6.
