# v6 file map

Entry point: `main.py` (TASK=1|2, DATASET, L2_ARM=mlp default). Canon: `PLAN.md`, `THEORY.md`.

| Group | Files | Role |
|---|---|---|
| Core solve | `core.py` `terms.py` `gen_operator.py` `l2_modules.py` `optimize.py` | unrolled solver / term factory (Def 2) / Jacobian-locked operator / WeightNet / frozen-arm objective + shared helpers (ALS, shrink) |
| Graph and data | `graph.py` `testbeds.py` `pool.py` `dependence.py` `nldep.py` | given-graph + d-separation / 13 reporting datasets / 16-dataset training pool / dcor-MI infrastructure / nonlinear target stack (TRUNK-4a) |
| Encoder and dictionary | `encode.py` `lora.py` `build_dictionary.py` `reencode_dict.py` `expand_dictionary.py` | e5 wrapper / L3 LoRA / 521k bank build / LoRA-space re-encode / +CogAtlas expansion |
| Training | `train.py` `negop.py` `l3_train.py` | joint operator+WeightNet trainer (dev fold-4 MATCH selection) / f_neg / LoRA training lineage |
| Runners | `run_task1.py` `run_task2.py` `run_bigfive_hier.py` | official protocols; hierarchy pilot |
| Metrics and decode | `metrics.py` `judge.py` `splice_decode.py` | match/exact / LLM judge + disk cache / SpLiCE decode |
| Diagnostics (read-only) | `certainty.py` `adequacy.py` `influence_decode.py` | cert(i) Def 4 / V(G,X) Def 5 + repair proposals / generative-path influence (P3) |
| Partially superseded | `latent_constraints.py` | `sign_fix` load-bearing; `augmented_*` superseded by nldep (legacy Pearson path only) |

Frozen artifacts in `outputs/` are symlinks to `v5/outputs/`; v6-trained files use NEW names
(`l2_mlp_v6*.pt`, `gen_operator*.pt`, `concept_bank_l3_cog.npz`) and never overwrite v5.
