# causal_llm_plugin

A lightweight, causality-based plugin that steers a large language model. The plugin
discovers causal structure from raw data, solves constraint-based embeddings on that
structure, and hands both to the LLM as evidence. The LLM writes the final
translation of each unlabeled variable. LLM + plugin together form the main method.

The LLM can be an API model (current experiments: gpt-5.5 through OpenRouter) or an
open-weight model (planned: continuous prefix conditioning, see `prefix/`).

## Layout

- `config.py` - absolute roots. Edit this first on a new machine.
- `plugin/` - the plugin itself:
  - `prompts.py` - the frozen prompt template (versioned; never tuned per dataset)
  - `evidence.py` - evidence extraction (decoded phrases + graph neighborhoods)
  - `run_plugin.py` - batch driver; one LLM call per variable per arm, cached
- `discovery/` - adapters that call the discovery stack (GPU-RLCD for
  questionnaires, BOSS for robots) in `graph_semantics/causal-learn/`
- `scoring/` - referee-space metrics (NRR, SDA) and report generators;
  `score_records.py` writes per-dataset detail files to `outputs/scores/`
- `experiments/` - lane scripts for every experiment family (pilot arms, the
  label-masking stress test, the merged translation task, funding gates)
- `prefix/` - reserved for the open-weight continuous-prefix variant
- `outputs/` - scores and gate logs (records live with the pipeline, see below)

## Evaluation arms

Every experiment runs block-level ablations of the one frozen template:

- `llmfull` - phrases + causal neighbors (the main method)
- `llmgraph` - causal neighbors only = the LLM-ONLY baseline (nothing from the
  pipeline; on discovered graphs even this consumes our discovery output)
- `llmphrase` - phrases only
- `llmplacebo` - another variable's evidence (prior-leakage control)

## Reproduce

Requirements: the `graph_semantics` repo, python env with torch and
sentence-transformers (`/data2/shuhao/venv` here), an OpenRouter key in
`OPENAI_API_KEY` with `JUDGE_BASE_URL=https://openrouter.ai/api/v1`.

1. Discover graphs (skip if given graphs are used):
   `graph_semantics/discovery/run_latent_discovery.py` per dataset writes
   `discovery/outputs/<ds>_gpurlcd.json`; robots use BOSS summaries.
2. Produce pipeline records (decoded phrases per masked variable) with the frozen
   config: `graph_semantics/discovery/stageA_env.sh` +
   `experiments/run_joint_given.sh` / `run_joint_disc.sh` (merged task) or the
   stage A lanes (single tasks). Records land in
   `graph_semantics/discovery/outputs/rec_v2*/`.
3. Run plugin arms: `experiments/run_joint_plugin.sh` (or `run_pilot.sh` for the
   single-task pilot). One json per dataset x source x arm, plus a call cache.
4. Score: `scoring/score_records.py` per file (per-set details in
   `outputs/scores/`), `scoring/gen_llm_report.py` for the aggregate tables.
5. Judge (optional, costs API): `graph_semantics/discovery/stageB_judge.py` fills
   judge verdicts into every record file; rerun step 4 afterwards.

Determinism: temperature 0, versioned prompts, seeded masking, per-call caches.
Numbers reproduce up to LLM API nondeterminism.
