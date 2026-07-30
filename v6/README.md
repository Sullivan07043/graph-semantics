# v6 file map

Entry point: `main.py` (TASK=1|2, DATASET, L2_ARM=mlp default). Canon: `PLAN.md`, `THEORY.md`.

| Group | Files | Role |
|---|---|---|
| Core solve | `core.py` `terms.py` `gen_operator.py` `l2_modules.py` `optimize.py` | unrolled solver / term factory (Def 2) / Jacobian-locked operator / WeightNet / frozen-arm objective + shared helpers (ALS, shrink) |
| External baselines | `external_baselines.py` `graphmae.py` | Feature Propagation / PC1-loading centroid / pure-Torch GraphMAE-GCN adaptation |
| Graph and data | `graph.py` `testbeds.py` `pool.py` `dependence.py` `nldep.py` | given-graph + d-separation / 13 reporting datasets / 16-dataset training pool / dcor-MI infrastructure / nonlinear target stack (TRUNK-4a) |
| Encoder and dictionary | `encode.py` `lora.py` | e5 wrapper / L3 LoRA (runtime) |
| Training | `train.py` `negop.py` `graphmae.py train` | joint operator+WeightNet trainer (dev fold-4 MATCH selection) / f_neg / dev-only GraphMAE baseline |
| Offline tools | `tools/` | one-time builders, not runtime imports: dictionary build (`build_dictionary.py`), LoRA-space re-encode (`reencode_dict.py`), CogAtlas expansion (`expand_dictionary.py`), LoRA training (`l3_train.py`) |
| Runners | `run_task1.py` `run_task2.py` `run_bigfive_hier.py` | official protocols; hierarchy pilot |
| Metrics and decode | `metrics.py` `judge.py` `splice_decode.py` | match/exact / LLM judge + disk cache / SpLiCE decode |
| Diagnostics (read-only) | `certainty.py` `adequacy.py` `influence_decode.py` | cert(i) Def 4 / V(G,X) Def 5 + repair proposals / generative-path influence (P3) |
| Partially superseded | `latent_constraints.py` | `sign_fix` load-bearing; `augmented_*` superseded by nldep (legacy Pearson path only) |

Frozen artifacts in `outputs/` are symlinks to `v5/outputs/`; v6-trained files use NEW names
(`l2_mlp_v6*.pt`, `gen_operator*.pt`, `concept_bank_l3_cog.npz`) and never overwrite v5.

## External baseline switches

The original runner behavior is unchanged unless an arm is explicitly enabled:

- Task 1: `FP_ARM=1` adds `feature_prop`; `GRAPHMAE_ARM=1` adds
  `graphmae_gcn` for zero-shot evaluation.
- Task 2: `LOADING_ARM=1` adds `loading_centroid` only on measurement DAGs. General
  DAGs with observed-source edges (TLVD) are explicitly marked not applicable.
- With an API key, Task 2 adds `mb_llm_name`, a single-LLM typed-Markov-blanket
  adaptation. Configure its generator independently with `LLM_BASELINE_MODEL`; the
  evaluator remains `JUDGE_MODEL`.
- `GRAPHMAE_CKPT` defaults to `outputs/graphmae.pt`.
- GraphMAE evaluation rejects datasets listed in checkpoint `train_datasets`. Setting
  `GRAPHMAE_ALLOW_TRAIN_EVAL=1` renames the arm to `graphmae_gcn_in_sample` and tags
  every record; those diagnostics are not reportable zero-shot results.

Feature Propagation uses Rossi et al.'s symmetrically normalized adjacency, zero
initialization, repeated diffusion, and clamping update, with the paper's 40-step default.
Projecting a directed causal graph to a binary undirected graph (plus explicit handling of
unanchored components) is the benchmark adaptation. GraphMAE is a PyTorch-only GCN adaptation retaining the GraphMAE
mask token, decoder re-mask, and scaled-cosine reconstruction objective. It is not the
official node-classification pipeline. Train it only from `pool.DEV` with
`python v6/graphmae.py train`. The CLI rejects datasets outside DEV, saves the exact
`train_datasets`, encoder name, and LoRA SHA-256, and supports
`GRAPHMAE_EXCLUDE=tlvd` for a run that intentionally leaves TLVD untouched.

The PC1-loading centroid is a transparent classical measurement-model comparator rather
than a reproduction of a single published package. The MB-LLM arm uses the TLVD paper's
Markov-blanket input principle but omits its multi-agent BNE training and evidence-search
stage, so outputs identify it as an adaptation rather than official TLVD.
