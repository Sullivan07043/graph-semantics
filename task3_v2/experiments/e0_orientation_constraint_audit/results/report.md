# Task 3 E0-double-prime — Orientation and Constraint Audit

**Final classification: F — `broader_solver_or_metric_failure`.**

This is a frozen diagnostic run. It did not run an LLM, J-space, CauScale, activation writes, latent discovery, or training, and it did not tune any loss coefficient.

## 1. Orientation verdict

Orientation audit passed: **True**. JSON `source -> target`, source-row/target-column adjacency, the adapter, `Graph.parents/children`, SCM generation, ALS, and Stage-3 generation all agree. The negative transposition control was rejected.

For `A -> B -> C`, the implemented generation loss gives `dL/dz_B=(-1.4,-0.8)` in the fixed probe: B's own equation pulls toward parent A, while C's equation also back-propagates through B and pulls toward child C. This is bidirectional quadratic compatibility, not an interface transpose.

All 15 canonical full solves passed parity; embeddings file SHA-256 is `8cfc774f015d267299544b033981378caf677a767096efaa44e8135270d4ee8b`. The 60 full-oracle node metrics also match frozen E0-prime within the preregistered tolerance.

## 2. Bundle replication

Selected-dataset behavioral trend reproduced: **True**; bundle result status: `formal_local_api_free`. See `../bundle_replication.md` and `bundle_replication.json` for the dev, held-out, and BigFive2 hierarchy/latent-constraint reruns. Judge remains pending where no cache/API verdict was available. The absent original release artifacts remain a provenance limitation, not by themselves a fabricated drift verdict.

## 3. Root / non-root strata

| group | arm | nodes | gold cos | centered cos | MRR | R@5 | Match |
| --- | --- | --- | --- | --- | --- | --- | --- |
| root | full_oracle | 18 | 0.5017 | 0.1205 | 0.2846 | 0.6111 | 0.7222 |
| root | reversed_full | 18 | 0.8673 | 0.1023 | 0.2709 | 0.6667 | 0.7222 |
| root | shuffled_full | 18 | 0.6207 | 0.0060 | 0.1697 | 0.3139 | 0.3944 |
| root | uniform | 18 | 0.9147 | -0.4330 | 0.0961 | 0.1111 | 0.2222 |
| root | raw_correlation | 18 | 0.9125 | 0.0010 | 0.1360 | 0.2222 | 0.6667 |
| non_root | full_oracle | 42 | 0.8500 | 0.1019 | 0.1930 | 0.5238 | 0.6667 |
| non_root | reversed_full | 42 | 0.8806 | 0.0763 | 0.1838 | 0.4048 | 0.7143 |
| non_root | shuffled_full | 42 | 0.7980 | -0.0005 | 0.1442 | 0.2500 | 0.3679 |
| non_root | uniform | 42 | 0.9156 | -0.4082 | 0.0903 | 0.0476 | 0.2857 |
| non_root | raw_correlation | 42 | 0.9173 | -0.0996 | 0.1080 | 0.1190 | 0.7381 |
| non_root_visible_parent | full_oracle | 42 | 0.8500 | 0.1019 | 0.1930 | 0.5238 | 0.6667 |
| non_root_visible_parent | reversed_full | 42 | 0.8806 | 0.0763 | 0.1838 | 0.4048 | 0.7143 |
| non_root_visible_parent | shuffled_full | 42 | 0.7980 | -0.0005 | 0.1442 | 0.2500 | 0.3679 |
| non_root_visible_parent | uniform | 42 | 0.9156 | -0.4082 | 0.0903 | 0.0476 | 0.2857 |
| non_root_visible_parent | raw_correlation | 42 | 0.9173 | -0.0996 | 0.1080 | 0.1190 | 0.7381 |
| no_visible_parent_visible_child | full_oracle | 17 | 0.5694 | 0.1249 | 0.2896 | 0.5882 | 0.7059 |
| no_visible_parent_visible_child | reversed_full | 17 | 0.8694 | 0.1090 | 0.2721 | 0.6471 | 0.7647 |
| no_visible_parent_visible_child | shuffled_full | 17 | 0.6339 | 0.0075 | 0.1688 | 0.3118 | 0.4000 |
| no_visible_parent_visible_child | uniform | 17 | 0.9141 | -0.4342 | 0.0933 | 0.1176 | 0.2353 |
| no_visible_parent_visible_child | raw_correlation | 17 | 0.9130 | 0.0035 | 0.1355 | 0.2353 | 0.7059 |
| no_visible_structural_anchor | full_oracle | 1 | -0.6492 | 0.0461 | 0.2000 | 1.0000 | 1.0000 |
| no_visible_structural_anchor | reversed_full | 1 | 0.8318 | -0.0113 | 0.2500 | 1.0000 | 0.0000 |
| no_visible_structural_anchor | shuffled_full | 1 | 0.3962 | -0.0198 | 0.1859 | 0.3500 | 0.3000 |
| no_visible_structural_anchor | uniform | 1 | 0.9247 | -0.4128 | 0.1429 | 0.0000 | 0.0000 |
| no_visible_structural_anchor | raw_correlation | 1 | 0.9048 | -0.0414 | 0.1429 | 0.0000 | 0.0000 |

Reversed-minus-oracle gold-cosine advantage attributable to roots: **83.67%**. Roots therefore explain most of reversed's aggregate cosine advantage, but they are not a sufficient explanation of the complete failure.

## 4. Visible-parent strata

All 42 non-root nodes have at least one visible parent. Their semantic and retrieval metrics must be read together: the frozen E0-prime pattern can lose cosine to no-graph baselines while improving MRR/R@5. This metric conflict is retained rather than resolved by selecting one metric.

| group | comparison | metric | mean Δ | CI low | CI high | n |
| --- | --- | --- | --- | --- | --- | --- |
| root | full_oracle_minus_reversed_full | gold_cosine | -0.3657 | -0.5840 | -0.2292 | 18 |
| root | full_oracle_minus_reversed_full | centered_cosine | 0.0182 | -0.0756 | 0.0994 | 18 |
| root | full_oracle_minus_reversed_full | prediction_margin | -0.1020 | -0.1443 | -0.0702 | 18 |
| root | full_oracle_minus_reversed_full | mrr | 0.0137 | -0.0257 | 0.0579 | 18 |
| root | full_oracle_minus_reversed_full | recall_at_5 | -0.0556 | -0.2941 | 0.1667 | 18 |
| non_root_visible_parent | full_oracle_minus_uniform | gold_cosine | -0.0656 | -0.1835 | -0.0194 | 42 |
| non_root_visible_parent | full_oracle_minus_uniform | centered_cosine | 0.5101 | 0.4405 | 0.5978 | 42 |
| non_root_visible_parent | full_oracle_minus_uniform | prediction_margin | -0.0464 | -0.0608 | -0.0322 | 42 |
| non_root_visible_parent | full_oracle_minus_uniform | mrr | 0.1027 | 0.0459 | 0.1761 | 42 |
| non_root_visible_parent | full_oracle_minus_uniform | recall_at_5 | 0.4762 | 0.2000 | 0.7500 | 42 |
| all_nodes | full_oracle_minus_generation_only | gold_cosine | 0.0226 | 0.0017 | 0.0437 | 60 |
| all_nodes | full_oracle_minus_generation_only | centered_cosine | 0.0184 | -0.0049 | 0.0464 | 60 |
| all_nodes | full_oracle_minus_generation_only | prediction_margin | -0.0018 | -0.0149 | 0.0095 | 60 |
| all_nodes | full_oracle_minus_generation_only | mrr | -0.0067 | -0.0337 | 0.0190 | 60 |
| all_nodes | full_oracle_minus_generation_only | recall_at_5 | 0.0500 | -0.1337 | 0.2167 | 60 |
| all_nodes | oracle_without_generation_minus_full_oracle | gold_cosine | -0.4443 | -0.5348 | -0.3287 | 60 |
| all_nodes | oracle_without_generation_minus_full_oracle | centered_cosine | -0.0443 | -0.0935 | 0.0040 | 60 |
| all_nodes | oracle_without_generation_minus_full_oracle | prediction_margin | -0.0031 | -0.0260 | 0.0195 | 60 |
| all_nodes | oracle_without_generation_minus_full_oracle | mrr | -0.0161 | -0.0590 | 0.0350 | 60 |
| all_nodes | oracle_without_generation_minus_full_oracle | recall_at_5 | -0.1500 | -0.3667 | 0.0667 | 60 |
| all_nodes | same_module_minus_shuffled_full | gold_cosine | -0.0754 | -0.2069 | 0.0318 | 60 |
| all_nodes | same_module_minus_shuffled_full | centered_cosine | 0.0320 | 0.0072 | 0.0607 | 60 |
| all_nodes | same_module_minus_shuffled_full | prediction_margin | -0.0059 | -0.0274 | 0.0131 | 60 |
| all_nodes | same_module_minus_shuffled_full | mrr | 0.0319 | -0.0009 | 0.0684 | 60 |
| all_nodes | same_module_minus_shuffled_full | recall_at_5 | 0.1308 | -0.0100 | 0.2792 | 60 |
| all_nodes | same_module_minus_uniform | gold_cosine | -0.2460 | -0.3837 | -0.1367 | 60 |
| all_nodes | same_module_minus_uniform | centered_cosine | 0.4490 | 0.4115 | 0.4904 | 60 |
| all_nodes | same_module_minus_uniform | prediction_margin | -0.1129 | -0.1364 | -0.0924 | 60 |
| all_nodes | same_module_minus_uniform | mrr | 0.0917 | 0.0501 | 0.1387 | 60 |
| all_nodes | same_module_minus_uniform | recall_at_5 | 0.3333 | 0.1500 | 0.5333 | 60 |

`no_visible_structural_anchor` means no visible parent or child in the current oracle objective. Visible same-module candidates are reported separately because they are not part of that objective.

## 5. Constraint decomposition

| arm | n | gold cos | centered cos | margin | MRR | R@5 | Match |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full_oracle | 60 | 0.7455 | 0.1075 | -0.1114 | 0.2205 | 0.5500 | 0.6833 |
| generation_only_oracle | 60 | 0.7229 | 0.0891 | -0.1096 | 0.2272 | 0.5000 | 0.6833 |
| oracle_without_generation | 60 | 0.3012 | 0.0632 | -0.1145 | 0.2044 | 0.4000 | 0.6500 |
| residual_only_oracle | 60 | 0.7725 | 0.1069 | -0.1009 | 0.2184 | 0.5167 | 0.7667 |
| independence_only_oracle | 60 | 0.3203 | 0.1068 | -0.1266 | 0.2214 | 0.4500 | 0.6833 |
| symmetrized_oracle | 60 | 0.4123 | 0.0803 | -0.1402 | 0.2285 | 0.6000 | 0.7500 |
| markov_blanket_oracle | 60 | 0.1453 | 0.0549 | -0.1416 | 0.2157 | 0.5000 | 0.6000 |
| same_module_graph | 60 | 0.6694 | 0.0334 | -0.1365 | 0.1837 | 0.4000 | 0.5167 |
| reversed_full | 60 | 0.8766 | 0.0841 | -0.0795 | 0.2099 | 0.4833 | 0.7167 |
| shuffled_full | 1200 | 0.7448 | 0.0014 | -0.1306 | 0.1518 | 0.2692 | 0.3758 |
| raw_correlation | 60 | 0.9159 | -0.0694 | -0.0295 | 0.1164 | 0.1500 | 0.7167 |
| uniform | 60 | 0.9153 | -0.4157 | -0.0237 | 0.0920 | 0.0667 | 0.2667 |

Every optimizer arm starts from the exact same oracle ALS embeddings. The decomposition changes only the six registered term switches. `generation_only` intentionally leaves the residual inside the frozen generation equation without its penalties; `residual_only` is mathematically disconnected from semantic embeddings, so zero displacement/embedding gradient is expected.

| arm | term | active | initial | final | final-initial |
| --- | --- | --- | --- | --- | --- |
| full_oracle | generation | yes | 1.6137 | 0.2992 | -1.3145 |
| full_oracle | residual_norm | yes | 0.0041 | 0.3890 | 0.3849 |
| full_oracle | residual_alignment | yes | 0.0017 | 0.0406 | 0.0389 |
| full_oracle | independence | yes | 0.0563 | 0.0533 | -0.0030 |
| full_oracle | bridge | yes | 0.0000 | 0.0000 | 0.0000 |
| full_oracle | norm | yes | 0.0039 | 0.0026 | -0.0014 |
| generation_only_oracle | generation | yes | 1.6137 | 0.0026 | -1.6111 |
| generation_only_oracle | residual_norm | no | 0.0041 | 1.2461 | 1.2419 |
| generation_only_oracle | residual_alignment | no | 0.0017 | 0.0700 | 0.0682 |
| generation_only_oracle | independence | no | 0.0563 | 0.0549 | -0.0015 |
| generation_only_oracle | bridge | no | 0.0000 | 0.0000 | 0.0000 |
| generation_only_oracle | norm | no | 0.0039 | 0.0036 | -0.0004 |
| oracle_without_generation | generation | no | 1.6137 | 3.7385 | 2.1247 |
| oracle_without_generation | residual_norm | yes | 0.0041 | 0.0662 | 0.0621 |
| oracle_without_generation | residual_alignment | yes | 0.0017 | 0.0369 | 0.0352 |
| oracle_without_generation | independence | yes | 0.0563 | 0.0410 | -0.0153 |
| oracle_without_generation | bridge | yes | 0.0000 | 0.0000 | 0.0000 |
| oracle_without_generation | norm | yes | 0.0039 | 0.0000 | -0.0039 |
| residual_only_oracle | generation | no | 1.6137 | 1.7024 | 0.0886 |
| residual_only_oracle | residual_norm | yes | 0.0041 | 0.0662 | 0.0621 |
| residual_only_oracle | residual_alignment | yes | 0.0017 | 0.0369 | 0.0352 |
| residual_only_oracle | independence | no | 0.0563 | 0.0563 | 0.0000 |
| residual_only_oracle | bridge | no | 0.0000 | 0.0000 | 0.0000 |
| residual_only_oracle | norm | no | 0.0039 | 0.0039 | 0.0000 |
| independence_only_oracle | generation | no | 1.6137 | 37.3780 | 35.7643 |
| independence_only_oracle | residual_norm | no | 0.0041 | 0.0041 | 0.0000 |
| independence_only_oracle | residual_alignment | no | 0.0017 | 0.0017 | 0.0000 |
| independence_only_oracle | independence | yes | 0.0563 | 0.0410 | -0.0153 |
| independence_only_oracle | bridge | yes | 0.0000 | 0.0000 | 0.0000 |
| independence_only_oracle | norm | no | 0.0039 | 0.4329 | 0.4289 |

Gradient diagnostics: 10440 applicable node-term rows; 5175 active near-zero, 0 exploding, 0 non-finite. Thresholds were frozen at 1e-10 and 1.0e3.

## 6. Same-module positive diagnostic

Same-module passed the preregistered multi-metric positive diagnostic: **False**. This graph is a semantic-support control, not a causal method. Its bidirected adapter also makes v5 trek/independence operations reduce to connected-component reachability; the graph-arm metadata records this limitation explicitly.

## 7. Where reversed's advantage comes from

Roots contribute 6.5823 summed reverse-minus-oracle cosine versus 1.2847 from non-roots. The effect is distributed across graphs and persists among roots with a visible child; it is not caused only by one completely unanchored node. Visible-parent non-roots show supported centered/retrieval gains but also supported adverse gold-cosine and margin cells. That is material metric conflict, not a stable category-C advantage.

## 8. Final failure classification

Primary category: **F — `broader_solver_or_metric_failure`**. Evidence counts: `{"causal_full_supported_semantic_cells": 2, "full_over_generation_only_supported_cells": 1, "root_semantic_failure_cells": 7, "same_module_adverse_cells": 2, "same_module_conflicting_comparisons": 1, "same_module_vs_shuffle_supported_cells": 1, "same_module_vs_uniform_supported_cells": 3, "visible_parent_adverse_semantic_cells": 4, "visible_parent_conflicting_comparisons": 2, "visible_parent_stable_baseline_comparisons": 1, "visible_parent_supported_semantic_cells": 4, "without_generation_supported_cells": 0}`.

Material cross-metric conflict: **True**; same-module positive control: **False**. A supported positive cell does not count as a stable advantage when the same comparison also contains supported adverse evidence.

The classification combines root/visible-parent strata, centered cosine, margin, retrieval metrics, the loss decomposition, bundle replication, and same-module positive control. It is not an aggregate-cosine-only verdict.

## 9. Rerun E0-prime?

Required: **False**. pause Task 3 and audit solver/evaluation because the semantic positive control failed or local metrics conflict materially. There was no orientation repair, so an orientation-triggered E0-prime rerun is explicitly forbidden.

## 10. S0 / old E1 decision

Old E1 allowed: **False**. S0 latent-to-observed semantic-support graph benchmark allowed: **False**.

## Statistics, decoder, and provenance

Paired inference uses graph -> fold -> masked node hierarchical bootstrap with exactly 10,000 fixed-seed draws. Shuffled leaves are means over all 20 fixed permutations before pairing. `judge_requests.jsonl` contains 1,860 unique pending requests; no Judge-ACC was fabricated.

Worktree commit at freeze: `02f8112f88cc1dddc22d2b445fbe1a14480be542`; frozen latest-main authority: `8d58ee99855dbe7a44c26a2b5c8642d01ba736ac`; current main at report time: `9ec3e264fb43f824a276f965061f9c9bf464ef59`. The working `v5` is byte-for-byte tree-equivalent to both main snapshots, so no dirty-branch checkout or merge was needed.

Formal commands are recorded in `provenance.json`. Full machine-readable values are in `per_node_audit.csv`, `per_group.csv`, `per_arm.csv`, `loss_terms.csv`, `gradient_norms.csv`, `paired_deltas.csv`, and `bootstrap_summary.csv`.
