# Robot Task 1 Baselines: Local Regeneration and Historical Reference

## Scope

This report evaluates channel-name recovery on four robot datasets. Panda, Sawyer, and IIWA are
development datasets; UR5e is strictly held out. The controlled comparison uses five-fold mean
Match-ACC, locally generated BOSS graphs, and frozen base E5. A separate README-style Core rerun
uses the historical joint-dev, LoRA-E5, and semantic-negation code path. Only Match-ACC is reported;
LLM Judge is not run.

## All baselines and Core

All available results are shown in one table. Bold marks the best result among the current local,
controlled base-E5 runs in each column. The README-style rerun and README historical rows are
included for context but are not eligible for the local winner because they use different training
and evaluator conditions.

| Method | Run / source | Panda (dev) | Sawyer (dev) | IIWA (dev) | Dev macro | UR5e (held-out) |
|---|---|---:|---:|---:|---:|---:|
| Uniform | Local base-E5 | .248 | .111 | .044 | .134 | .100 |
| RawCorr | Local base-E5 | .152 | .196 | **.547** | .299 | **.406** |
| Feature Propagation | Local base-E5 | .129 | .193 | .300 | .207 | .217 |
| GraphMAE-GCN | Local base-E5 | **.310** | **.436** | .364 | **.370** | .350 |
| CLIP-Dissect (E5 robot adaptation) | Local base-E5 | .095 | .229 | .272 | .199 | .267 |
| Automated Interpretability | Local base-E5 | .152 | .193 | .111 | .152 | .239 |
| Delphi | Local base-E5 | .033 | .143 | .136 | .104 | .100 |
| Core (ours) | Local base-E5 Robot-LODO | .248 | .168 | .231 | .215 | .217 |
| Core (ours) | Local README-style: joint-dev, LoRA-E5, semantic negation | .586 | .400 | .589 | .525 | .339 |
| Uniform | README historical | .257 | - | - | - | .217 |
| RawCorr | README historical | .152 | - | - | - | .328 |
| Linear | README historical | .248 | - | - | - | .242 |
| Naming | README historical | .248 | - | - | - | .361 |
| Core (ours) | README historical | .281 | - | - | - | .358 |

The eight local rows use the same regenerated step files, BOSS graphs, five folds, and frozen
base-E5 Match-ACC, so they form the controlled comparison. GraphMAE-GCN is the strongest of the
five new baselines on every robot. When Uniform and RawCorr are also included, GraphMAE-GCN is
best on Panda and Sawyer, while RawCorr is best on IIWA and held-out UR5e. Local Core does not win
any dataset: it scores `.248`, `.168`, `.231`, and `.217` on Panda, Sawyer, IIWA, and UR5e. On
Panda it ties the local Uniform control.

The local README-style Core rerun trains one model jointly on Panda, Sawyer, and IIWA folds 0--3,
uses the three fold-4 scores for checkpoint selection, and evaluates the selected model on all five
folds. It uses LoRA-E5 and the questionnaire-trained semantic-negation operator loaded by the
historical code. Its Match-ACC is `.586`, `.400`, `.589`, and `.339` on Panda, Sawyer, IIWA, and
UR5e, with a dev macro of `.525`. The selected checkpoint is epoch 18: mean dev fold-4 Match rises
from `.2222` to `.2778`. UR5e remains genuinely held out and is only `.019` below the README value
(`.339` versus `.358`). The three dev scores are in-domain results: folds 0--3 were directly used
for supervised training, so they must not be ranked against Robot-LODO results. In particular,
Panda `.586` is not a held-out generalization score.

GraphMAE-GCN uses Robot-LODO: each development target is trained on the other two development
robots, and the UR5e checkpoint is trained only on Panda, Sawyer, and IIWA. Questionnaire
checkpoints are never loaded.

Local Core was retrained separately for every target using the same Robot-LODO split as GraphMAE:
Panda uses Sawyer and IIWA; Sawyer uses Panda and IIWA; IIWA uses Panda and Sawyer; UR5e uses all
three development robots and never enters training or checkpoint selection. Every model uses 20
epochs, `K=400`, `K_GRAD=60`, source-robot fold 4 for checkpoint selection, and the pinned base-E5
evaluation space. Negative physical edges use scalar sign, and no questionnaire-trained semantic
negation checkpoint is loaded. The selected epochs are 19, 7, 12, and 4 respectively.

An earlier target-specific diagnostic run accidentally loaded the questionnaire semantic-negation
checkpoint despite the robot trainer's documented scalar-sign setting. Because 160--250 edges per
robot are negative, that run was materially affected and remains excluded. The new README-style
row intentionally enables semantic negation to reproduce the historical code path. The shared
operator now records its negative-edge mode in each checkpoint and rejects mode-mismatched loads.

The README Core values remain separate historical rows. No matching local checkpoint or fold
records exist for that earlier run, and the README reports dataset-level Core only for Panda and
UR5e. Its dev fold-4 value `.3175` is a checkpoint-selection statistic, not a Sawyer, IIWA, or
five-fold Dev-macro result. Relative to README historical Core, local Robot-LODO Core is `.033`
lower on Panda (`.248` versus `.281`) and `.141` lower on UR5e (`.217` versus `.358`). These are not
controlled regressions because the data artifacts, training split, and evaluator conditions differ.
The README-style rerun narrows the held-out UR5e difference to `.019`, supporting protocol and
implementation differences as the main explanation for the earlier gap.

Automated Interpretability produced only 10 unique names for 154 channels; `Gripper Control` and
`Joint Activation` account for 90.3% of its predictions. Delphi generated 46 unique names for 154
channels, and its held-out detection AUROC is near chance on every robot (`.499` to `.522`). These
native results are consistent with their low channel-name Match-ACC.

## Configuration

- Task: recover the names of 20% masked robot channels.
- Splits: five interleaved folds, seed 0.
- Metric: fold-local Hungarian Match-ACC, averaged across five folds.
- Encoder: `intfloat/e5-large-v2`, revision
  `f169b11e22de13617baa190a028a32f3493550b6`, with `query:` prefix and no project LoRA.
- LLM model: `gpt-4o-mini`; API-based methods use only fold-visible channel text.
- CLIP-Dissect bank: fixed 4,096-concept robot/physics WordNet bank with static joint/axis atoms.
- Core: target-specific Robot-LODO training, 20 epochs, `K=400`, `K_GRAD=60`, source fold-4 Match
  checkpoint selection, scalar-sign negative physical edges, no questionnaire negation checkpoint,
  and frozen base-E5 evaluation.
- README-style Core rerun: one joint model trained on Panda, Sawyer, and IIWA folds 0--3, fold 4
  selection, 20 epochs, `K=400`, `K_GRAD=60`, project LoRA-E5, and semantic negation. Epoch 18 is
  selected at `.2778` mean dev fold-4 Match, from a `.2222` starting value.
- Exact match: not reported (the legacy Core runner computes it internally, but it is not used for
  selection or comparison). LLM Judge: off.

## API execution

- Requested model: `gpt-4o-mini`; returned model: `gpt-4o-mini-2024-07-18`.
- Successful cached responses: 770 (308 Automated Interpretability, 462 Delphi).
- Token usage: 5,889,990 input and 584,386 output tokens.
- Client-estimated cost at the configured rates: `$1.2341`.
- Semantic correction requests: 0; every first response passed the local format checks.
- Per-case scans found the target gold label and raw node ID in 0 of 770 cached prompts. Case
  artifacts add the raw node ID after the API response, and metric artifacts add gold labels only
  during evaluation.
- The first four-worker attempt reached the account's 200k-token/minute limit. The resumable run
  completed with one case worker; already cached responses were not purchased again.
- API wall time was approximately 66 minutes including the rate-limit pause and resume.

## Comparison caveats

1. Sawyer, IIWA, and UR5e rollouts were regenerated locally. Panda uses the official robomimic
   data, but its BOSS graph was also regenerated locally.
2. The rollout script seeds random actions but not every simulator reset, and the BOSS script does
   not fix Python's internal shuffle seed. The local artifacts are therefore a same-configuration
   rerun, not an exact reconstruction of the README artifacts.
3. The controlled rows use one pinned frozen base-E5 evaluator. The explicitly marked README-style
   row uses project LoRA-E5, while the historical README table mixes that earlier LoRA-backed
   pipeline with a base-E5 naming baseline.
4. The original per-fold records, rollout arrays, BOSS summaries, and trained robot checkpoints
   underlying the README table are not committed as one frozen bundle, so an exact paired rerun is
   unavailable locally.
5. The README text says robot negative edges use scalar sign, but its historical implementation
   actually loads the questionnaire semantic-negation checkpoint. The README-style rerun follows
   the executed historical code rather than that contradictory prose description.

For these reasons, the local tables are the controlled comparison. README values are historical
context only.

## Artifacts

- External summary:
  `task3_robotics/outputs/baselines/task1_external_robot4_boss_seed0_v1/summary.json`
- Local Uniform/RawCorr controls:
  `task3_robotics/outputs/baselines/task1_uniform_rawcorr_base_e5_boss_seed0_v1.json`
- Local Core summary:
  `task3_robotics/outputs/core_local_base_e5_lodo_scalar_v2/summary.json`
- Local README-style Core summary:
  `task3_robotics/outputs/core_local_readme_legacy_lora_joint_v1/summary.json`
- Local README-style Core records and training log:
  `task3_robotics/outputs/core_local_readme_legacy_lora_joint_v1/task1_all4_records.json` and
  `task3_robotics/outputs/core_local_readme_legacy_lora_joint_v1/train_body_log.json`
- Historical reference: `task3_robotics/README.md`
