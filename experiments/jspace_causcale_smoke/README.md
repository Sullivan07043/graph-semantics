# J-space × CauScale × graph-semantics v4.1 smoke experiment

> Status (2026-07-22): the hard-do pilot is complete. See `RESULTS.md` for the
> measured results and current negative conclusion about incremental v4.1 lift.
> The original planning estimates at the end are retained as historical context.

This directory contains a small, gated experiment.  The question is whether the
**frozen v4.1 semantic completion system** benefits from a graph inferred over
measured Jacobian-lens coordinates.

The experiment deliberately separates engineering checks from scientific
evidence:

1. **Fixture gate (not evidence).** Generate a 20-node SCM with known graph and
   interventions. Validate the array contract, graph post-processing, and the
   frozen v4.1 custom-data path.
2. **J-lens readout gate.** Load the official Qwen3.5-4B n=1000 Jacobian lens,
   verify 20 preregistered `(layer, token, position-rule)` coordinates on 50
   prompts, then collect the full 1000-row matrix.
3. **CauScale gate.** Run the official synthetic checkpoint through a
   prediction-only wrapper, retaining all three pair probabilities. Export a
   thresholded acyclic graph and edge stability diagnostics.
4. **Semantic test.** Blind coordinate labels fold-by-fold and compare frozen
   v4.1 on the CauScale graph against correlation, no-graph, shuffled-graph, and
   oracle-graph controls.
5. **Causal test.** On held-out prompts, inject or ablate one early-layer
   coordinate and test whether predicted descendants and their semantic changes
   agree. Include matched-norm random-direction controls.

The primary acceptance contrast is:

```text
v4.1(CauScale graph) > v4.1(shuffled/no graph)
```

and, when an oracle graph is available:

```text
v4.1(CauScale graph) approaches v4.1(oracle graph).
```

No claim is made from a visually plausible graph alone. CauScale sees measured
coordinates, so this is a Task-1-style test of unnamed observed variables, not
discovery of completely unmeasured latent variables.

## Frozen inputs

- graph-semantics release: `v4.1`
- Qwen model: `Qwen/Qwen3.5-4B`
- J-lens source: commit `581d398613e5602a5af361e1c34d3a92ea82ba8e`
- lens: `neuronpedia/jacobian-lens`, commit
  `b62c39069a0740aebcc70462231b68612cae367f`, SHA-256
  `1f9a8f8fd593f0ffec1a9640993257ca4560f8ae3e5602315643d5cc6818534e`
- CauScale source: commit `9d4766bbe5efd118f9ae696545956e89ca8e4e4d`
- CauScale checkpoint: `synthetic/auprc=0.905_migrated.ckpt`

Exact paths, hashes, seeds, thresholds, and row counts are written to each run
manifest.

## Data contract

Every dataset directory contains:

```text
X.npy                  float32 [N,d], raw continuous coordinates
interventions.npy      uint8   [N,d], one-hot or all-zero per row
nodes.json             ordered metadata for the d columns
labels.json            held-out semantic labels (never passed to CauScale)
oracle_graph.npy       uint8   [d,d], optional; only for fixture/calibration
manifest.json          provenance, split and preprocessing information
```

Rows are independent prompts/rollouts. Token positions or layers are never
treated as independent rows. Standardization statistics are fitted on the DEV
rows only and copied into `manifest.json`.

## Initial commands

From the repository root:

```powershell
.venv\Scripts\python.exe experiments\jspace_causcale_smoke\src\preflight.py
.venv\Scripts\python.exe experiments\jspace_causcale_smoke\src\make_fixture.py
.venv\Scripts\python.exe experiments\jspace_causcale_smoke\src\graph_postprocess.py `
  --probabilities experiments\jspace_causcale_smoke\runs\fixture\oracle_pair_probs.npz `
  --output experiments\jspace_causcale_smoke\runs\fixture\discovered_graph.npz
```

The real-model collector and CauScale wrapper have their own isolated
environments. They exchange only `.npy`, `.npz`, and JSON files with v4.1.

## Time budget on this workstation

- fixture + adapters + frozen-v4.1 smoke: about 2–4 hours;
- model/lens download and first J-lens readout: about 1–3 hours if the network is
  healthy (roughly 10 GB must be downloaded);
- first interpretable clean/intervention end-to-end table: 1–2 working days;
- three seeds plus stability and matched controls: 3–5 working days.
