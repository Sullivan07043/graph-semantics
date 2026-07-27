# controlled-current32 v1

This directory contains a deterministic 1,000-prompt candidate for Stage 1
discovery. It is **not** an intervention set and it is **not behaviorally
frozen**.

Status: `CANDIDATE_STATIC_VALIDATION_PASSED_NOT_BEHAVIORALLY_FROZEN`

The only concept inventory is the ordered `CONCEPTS` literal in
`task3/build_discovery_matrix.py`. Leading spaces, capitalization, order, and
the total of 32 tokens are preserved in `concept_inventory.json` and every
prompt record.

All prompts end exactly with `Answer:`. Answers and concept labels live only in
metadata, except that `explicit_single` deliberately names its primary concept.
The other four conditions do not literally name their target concepts.

Files:

- `prompts.jsonl`: 1,000 Stage-1 discovery candidate records.
- `concept_inventory.json`: exact concept list plus source provenance.
- `fold_assignments.json`: five folds of 200 with fold-exclusive entities and
  template families.
- `validation_report.json`: static schema, balance, duplicate, and leakage
  checks.
- `generation_manifest.json`: generation settings, complete concept list,
  provenance, hashes, and repository snapshot.

Generation and validation logs are stored separately under
`task3/logs/controlled_current32/v1/`.

No artificial concept co-occurrence in this dataset is asserted to be causal
ground truth. Future intervention prompts must be generated separately after
candidate edges have been selected.
