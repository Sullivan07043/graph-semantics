# v6 file map

Entry point: `main.py` (TASK=1|2, DATASET, L2_ARM=mlp default). Canon: `PLAN.md`, `THEORY.md`.

| Group | Files | Role |
|---|---|---|
| Core solve | `core.py` `terms.py` `gen_operator.py` `l2_modules.py` `optimize.py` | unrolled solver / term factory (Def 2) / Jacobian-locked operator / WeightNet / frozen-arm objective + shared helpers (ALS, shrink) |
| Graph and data | `graph.py` `testbeds.py` `pool.py` `dependence.py` `nldep.py` | given-graph + d-separation / 13 reporting datasets / 16-dataset training pool / dcor-MI infrastructure / nonlinear target stack (TRUNK-4a) |
| Encoder and dictionary | `encode.py` `lora.py` | e5 wrapper / L3 LoRA (runtime) |
| Training | `train.py` `negop.py` | joint operator+WeightNet trainer (dev fold-4 MATCH selection) / f_neg |
| Offline tools | `tools/` | one-time builders, not runtime imports: dictionary build (`build_dictionary.py`), LoRA-space re-encode (`reencode_dict.py`), CogAtlas expansion (`expand_dictionary.py`), LoRA training (`l3_train.py`) |
| Runners | `run_task1.py` `run_task2.py` `run_bigfive_hier.py` | official protocols; hierarchy pilot |
| External baselines | `baselines/` (`runners/` contains Task 1/2 entry points; `reports/` contains the report) | five canonical implementations plus their shared protocol, API, tests, runners, and report; see [`baselines/README.md`](baselines/README.md) |
| Metrics and decode | `metrics.py` `judge.py` `splice_decode.py` | match/exact / LLM judge + disk cache / SpLiCE decode |
| Diagnostics (read-only) | `certainty.py` `adequacy.py` `influence_decode.py` | cert(i) Def 4 / V(G,X) Def 5 + repair proposals / generative-path influence (P3) |
| Partially superseded | `latent_constraints.py` | `sign_fix` load-bearing; `augmented_*` superseded by nldep (legacy Pearson path only) |

Frozen artifacts in `outputs/` are symlinks to `v5/outputs/`; v6-trained files use NEW names
(`l2_mlp_v6*.pt`, `gen_operator*.pt`, `concept_bank_l3_cog.npz`) and never overwrite v5.
