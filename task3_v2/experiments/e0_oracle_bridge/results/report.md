# Task 3 E0-prime -- Oracle Causal-Graph Bridge Test: Results

Run status: **FORMAL LOCAL RUN**  
Decision: **NO-GO**  
Judge-ACC: **pending** (requests written; no API was called)

## Frozen Stage-3 method and audit

Formal source entrypoint: `v5/main.py`. Actual E0 adapter: `task3_v2/scripts/run_e0_bridge.py`. The adapter imports library modules directly and never imports or executes the side-effectful formal entrypoint.

The encoder is `intfloat/e5-large-v2` at local snapshot `f169b11e22de13617baa190a028a32f3493550b6`, with the frozen last-two-layer rank-8, alpha-16 LoRA. The frozen WeightNet runs K=60 functional-Adam inner steps at lr=0.02. WeightNet, negop, and solver run on CPU; only the encoder uses CUDA.

Frozen objective: lam_zero=0.3, lam_norm=0.1, residual=1.0, lam_res=1.0, Pearson bridge lambda=0.3, kappa=0.5, q=0.7. This adapter changes interface shape only, not the loss.

The frozen residual term is the solver's embedding residual plus a train+dev parent-regressed partial-correlation anchor. It is **not** the forbidden J-space innovation-residual preprocessing.

Training audit (recorded, not rerun): the WeightNet procedure ran outer Adam lr=0.001 for 4 epochs / 256 total updates; best-checkpoint selection retained epoch 3 / update 192. It used K=60 inner steps. LoRA Adam ran lr=1e-4 for 3 epochs / 192 updates, with anchor=10, bridge=1, independence=0.3, negation=1 loss weights. No component was retrained in E0-prime.

| Artifact | Path | SHA-256 |
|---|---|---|
| lora | D:\research\translate latent variables\graph-semantics\v5\outputs\l3_lora.pt | d90b024e7fb030e3ee1545c8d19606cf032cf810fc7f4a758749dfefc95a49d5 |
| dictionary | D:\research\translate latent variables\graph-semantics\v5\outputs\concept_bank_l3.npz | 6da2de255dcb2fa559fa1c2a8bfba25fd0e4fcfecb5c768ecf229b6ce4e7bb9e |
| weightnet | D:\research\translate latent variables\graph-semantics\v5\outputs\l2_mlp.pt | 70ffc4fcf668b57d943240fde67a8b339c702fa0093a5594f34e3297a5c50bfd |
| negop | D:\research\translate latent variables\graph-semantics\v5\outputs\negop.pt | 6f30f0d68ee653d52aef93bdae97feeac8abd17189e688ef49f594e18574690e |

Artifact limitation: the original ignored v5-compatible release bundle is absent. These checkpoints were retrained locally at the recorded artifact commit and frozen. The local reproduction did not exactly reproduce README metrics; E0-prime tests this local frozen bundle.

## Graph fixtures

| Graph | World | Nodes | Edges | Roots | Max in | Chain | Fork | Collider | Mediator |
|---|---|---|---|---|---|---|---|---|---|
| graph_00 | industrial_cooling_system | 20 | 26 | 6 | 3 | 38 | 5 | 11 | 13 |
| graph_01 | logistics_and_delivery_system | 20 | 29 | 6 | 3 | 35 | 9 | 14 | 13 |
| graph_02 | water_treatment_system | 20 | 29 | 6 | 3 | 39 | 9 | 14 | 13 |

Solver nodes are anonymous IDs. Each masked solve receives support, numeric X/W, and only the 16 visible-label embeddings; gold metadata remains in the fixture/evaluation layer.

## SCM data QA

| Graph | Data SHA-256 | Train mean err | Train SD err | Edge corr min | Finite |
|---|---|---|---|---|---|
| graph_00 | 264477b50971cbb72c5f379fc698673723e4346173708003db7b1bfea1804b3f | 4.95e-17 | 1.11e-15 | 0.4252 | yes |
| graph_01 | 61d6b6c7cd869222a8f72ac919560c7731e2dc36354f689fa083b77c9af259c6 | 6.45e-17 | 8.88e-16 | 0.4313 | yes |
| graph_02 | 9b8534cccb7ce3004c1db20a0c2eace4ce907af70d878930fb483b3a7ee48bb2 | 8.40e-17 | 1.78e-15 | 0.4218 | yes |

Each graph has 2,000 rows split 1,200/400/400. Z-score statistics use train only. W, bridge, partial residual correlation, and rawcorr use train+dev only; test is excluded from selection.

## All-arm local metrics

| Arm | Judge | Match | Cosine | MRR | R@1 | R@5 | Exact |
|---|---|---|---|---|---|---|---|
| core_oracle_estimated_weights | pending | 0.6833 | 0.7455 | 0.2205 | 0.0000 | 0.5500 | 0.0000 |
| core_oracle_true_weights | pending | 0.6833 | 0.7870 | 0.2109 | 0.0000 | 0.5333 | 0.0000 |
| core_shuffled_graph | pending | 0.3758 | 0.7324 | 0.1506 | 0.0033 | 0.2658 | 0.0033 |
| core_reversed_graph | pending | 0.7167 | 0.8760 | 0.2067 | 0.0000 | 0.4500 | 0.0000 |
| raw_correlation | pending | 0.7167 | 0.9159 | 0.1164 | 0.0000 | 0.1500 | 0.0000 |
| uniform | pending | 0.2667 | 0.9153 | 0.0920 | 0.0000 | 0.0667 | 0.0000 |

The shuffled arm is averaged across 20 permutations per graph; `shuffle_null.csv` contains the full exactly-60-permutation null distribution.

## Per-graph all-arm results

| Graph | Arm | Match | Cosine | MRR | R@1 | R@5 | Exact |
|---|---|---|---|---|---|---|---|
| graph_00 | core_oracle_estimated_weights | 1.0000 | 0.6756 | 0.2846 | 0.0000 | 0.7500 | 0.0000 |
| graph_00 | core_oracle_true_weights | 0.9000 | 0.7345 | 0.2719 | 0.0000 | 0.7000 | 0.0000 |
| graph_00 | core_shuffled_graph | 0.3800 | 0.7104 | 0.1547 | 0.0050 | 0.2825 | 0.0050 |
| graph_00 | core_reversed_graph | 0.9000 | 0.8637 | 0.2732 | 0.0000 | 0.5500 | 0.0000 |
| graph_00 | raw_correlation | 0.8000 | 0.9146 | 0.1145 | 0.0000 | 0.2000 | 0.0000 |
| graph_00 | uniform | 0.4000 | 0.9140 | 0.0796 | 0.0000 | 0.0000 | 0.0000 |
| graph_01 | core_oracle_estimated_weights | 0.5000 | 0.8116 | 0.2045 | 0.0000 | 0.5000 | 0.0000 |
| graph_01 | core_oracle_true_weights | 0.5000 | 0.8272 | 0.1969 | 0.0000 | 0.4500 | 0.0000 |
| graph_01 | core_shuffled_graph | 0.3625 | 0.7425 | 0.1468 | 0.0025 | 0.2500 | 0.0025 |
| graph_01 | core_reversed_graph | 0.6500 | 0.8845 | 0.1686 | 0.0000 | 0.3500 | 0.0000 |
| graph_01 | raw_correlation | 0.5500 | 0.9178 | 0.1200 | 0.0000 | 0.1500 | 0.0000 |
| graph_01 | uniform | 0.2000 | 0.9171 | 0.0951 | 0.0000 | 0.1000 | 0.0000 |
| graph_02 | core_oracle_estimated_weights | 0.5500 | 0.7493 | 0.1723 | 0.0000 | 0.4000 | 0.0000 |
| graph_02 | core_oracle_true_weights | 0.6500 | 0.7993 | 0.1638 | 0.0000 | 0.4500 | 0.0000 |
| graph_02 | core_shuffled_graph | 0.3850 | 0.7444 | 0.1503 | 0.0025 | 0.2650 | 0.0025 |
| graph_02 | core_reversed_graph | 0.6000 | 0.8798 | 0.1782 | 0.0000 | 0.4500 | 0.0000 |
| graph_02 | raw_correlation | 0.8000 | 0.9153 | 0.1146 | 0.0000 | 0.1000 | 0.0000 |
| graph_02 | uniform | 0.2000 | 0.9149 | 0.1014 | 0.0000 | 0.1000 | 0.0000 |

Every graph retains all 20 nodes and all six arms; shuffled entries average 20 permutations.

## Structural diagnostics

| Diagnostic | Stratum | n | Match | Cosine | MRR |
|---|---|---|---|---|---|
| root_status | non_root | 42 | 0.6667 | 0.8500 | 0.1930 |
| root_status | root | 18 | 0.7222 | 0.5017 | 0.2846 |
| visible_parent | no | 18 | 0.7222 | 0.5017 | 0.2846 |
| visible_parent | yes | 42 | 0.6667 | 0.8500 | 0.1930 |
| visible_child | no | 5 | 0.8000 | 0.5657 | 0.1861 |
| visible_child | yes | 55 | 0.6727 | 0.7619 | 0.2236 |
| visible_same_module | yes | 60 | 0.6833 | 0.7455 | 0.2205 |

No difficult node was excluded. Per-graph/all-arm strata are in `structural_diagnostics.csv`.

## Paired hierarchical bootstrap

| Scope | Graph | Comparison | Mean delta | 95% CI | Paired win |
|---|---|---|---|---|---|
| aggregate | aggregate | oracle_vs_shuffle | 0.0131 | [-0.1106, 0.0868] | 0.7667 |
| graph | graph_00 | oracle_vs_shuffle | -0.0348 | [-0.3173, 0.1241] | 0.7500 |
| graph | graph_01 | oracle_vs_shuffle | 0.0691 | [0.0024, 0.1275] | 0.8500 |
| graph | graph_02 | oracle_vs_shuffle | 0.0050 | [-0.0809, 0.0897] | 0.7000 |
| aggregate | aggregate | oracle_vs_reverse | -0.1305 | [-0.2747, -0.0484] | 0.3500 |
| graph | graph_00 | oracle_vs_reverse | -0.1881 | [-0.5132, -0.0087] | 0.2500 |
| graph | graph_01 | oracle_vs_reverse | -0.0729 | [-0.1515, -0.0100] | 0.4000 |
| graph | graph_02 | oracle_vs_reverse | -0.1305 | [-0.2561, -0.0419] | 0.4000 |
| aggregate | aggregate | oracle_vs_no_graph | -0.1698 | [-0.3295, -0.0826] | 0.0833 |
| graph | graph_00 | oracle_vs_no_graph | -0.2384 | [-0.5879, -0.0439] | 0.2500 |
| graph | graph_01 | oracle_vs_no_graph | -0.1056 | [-0.1841, -0.0467] | 0.0000 |
| graph | graph_02 | oracle_vs_no_graph | -0.1655 | [-0.3035, -0.0777] | 0.0000 |
| aggregate | aggregate | oracle_vs_rawcorr | -0.1704 | [-0.3287, -0.0839] | 0.0667 |
| graph | graph_00 | oracle_vs_rawcorr | -0.2390 | [-0.5859, -0.0469] | 0.2000 |
| graph | graph_01 | oracle_vs_rawcorr | -0.1062 | [-0.1835, -0.0479] | 0.0000 |
| graph | graph_02 | oracle_vs_rawcorr | -0.1659 | [-0.3014, -0.0784] | 0.0000 |

For oracle-vs-shuffle, 20 shuffle values are averaged within each masked node first. All comparisons then resample graph -> fold -> masked node for exactly 10,000 fixed-seed draws. `paired_deltas.csv` includes all metrics, aggregate rows, and graph-specific rows.

## Judge requests

`judge_requests.jsonl` contains 1500 cache-compatible requests frozen to model `gpt-5.5`, mode `completion`, with `rec`, `tgt`, arm, and shuffle provenance. Formal-run requests are all pending; no network/API call or fabricated verdict occurs.

## Decision and attribution

**NO-GO** on `gold_embedding_cosine`. Consistent positive graphs: 0/3 (none).

Supported-positive means mean >0 and aggregate 95% CI lower bound >0. GO requires supported-positive oracle-vs-shuffle, oracle-vs-uniform, and oracle-vs-reverse effects, plus joint positive direction in >=2 graphs. NO-GO requires both core comparisons (oracle-vs-shuffle and oracle-vs-uniform) to be unsupported, reverse mean <=0, and the two core means not both positive. Directionally positive core means that miss CI or cross-graph requirements, and all other conflicting patterns, are INCONCLUSIVE.

Failure-source label: **causal_graph_semantic_constraint_mismatch**. This is a diagnostic attribution heuristic from preregistered arm point estimates, not a proven causal attribution.

E1 allowed: **no**. diagnose graph-to-semantic constraint transfer; E1 is not allowed.

## Reproducibility

- Current git commit: `02f8112f88cc1dddc22d2b445fbe1a14480be542`
- Working tree dirty: `True`
- Artifact training commit: `70efde7ed488229667ae7958237116c7bdb40e45`
- Config: `D:\research\translate latent variables\graph-semantics\task3_v2\experiments\e0_oracle_bridge\config.yaml`
- Config SHA-256: `e89b2c74721b8fb9f34728987dfdc1bb9ebb1eb0b1ac25c5714af0605d8605ea`
- Actual command: `"D:\research\translate latent variables\graph-semantics\.venv\Scripts\python.exe" task3_v2\scripts\run_e0_bridge.py --config task3_v2\experiments\e0_oracle_bridge\config.yaml`
- e5 snapshot revision: `f169b11e22de13617baa190a028a32f3493550b6`
- Packages, GPU inventory, git status, checkpoint metadata, and hashes: `run_manifest.json`.
