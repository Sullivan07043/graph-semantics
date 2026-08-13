# Task 3: robot channel translation

Task 1 on robot data: mask 20% of the channel names of a robot log, recover them from causal
constraints. Observed variables only, no latents (meeting decision 2026-08-05). A channel is one
column of the step-level log, e.g. `robot0_joint_pos.3` = "angle of robot arm joint 3".

Robots: Panda (robomimic policy data, dev), Sawyer + IIWA (self-collected, dev),
UR5e (self-collected, held-out; 6 joints, never enters training or selection).

## Environments

Three, isolated; they exchange files, never a process.

| env | path | used for |
|---|---|---|
| certified pipeline | `/data2/shuhao/venv` | discovery, training, evaluation (numpy 2.2, torch 2.9) |
| robotics | `/data2/shuhao/venv_robo` | robosuite simulation (numpy 1.26, mujoco 3.3.7) |
| CauScale | `/data2/shuhao/miniforge3/envs/causcale` | not used on this line anymore |

Simulation is headless: always `MUJOCO_GL=egl`. robosuite 1.5.2 needs mujoco==3.3.7 (3.11 renamed
`qM`).

## Data flow

```
collect_rollouts.py  (venv_robo)   robot -> body_<r>_rollouts.npz   [steps, actions, episode ids]
build_steps.py                     rollouts -> body_<r>_steps.npz   [lag-1 design matrix]
discover.py          ROUTE=boss    steps -> body_<r>_discovered.json
summarize_graph.py                 lag graph -> body_<r>_boss_summary.json  [channel level,
                                   self-edges dropped, edge_types lag/contemp kept]
true_graph_any.py    (venv_robo)   simulator physics -> body_<r>_true.json  [verified per robot]
evaluate_vs_truth.py               discovered vs true: precision/recall by edge kind
task3_pipeline_v1/main.py          Task 1 protocol (mask 20%, 5 folds)
task3_pipeline_v1/train_body.py    operator + WeightNet training on the dev robots
t1_naming_baseline.py              LLM naming arm (gpt-4o-mini over graph neighbours)
```

Panda uses robomimic instead of collection: `lift_mg_low_dim_dense_v15.hdf5` under
`data/pool/robomimic/`, converted by `load_steps.py` to `lift_body_steps.npz`.

## Reproduce

```bash
T=/data2/shuhao/semantic_interpretation/graph_semantics/task3_robotics
RV=/data2/shuhao/venv_robo/bin/python
PV=/data2/shuhao/venv/bin/python

# 1. collect (per robot; ~40 min each; Panda comes from robomimic instead)
MUJOCO_GL=egl TASK=Lift ROBOT=Sawyer EPISODES=1000 STEPS=200 \
  OUT=$T/outputs/body_sawyer_rollouts.npz $RV $T/collect_rollouts.py

# 2. design matrix
NPZ=$T/outputs/body_sawyer_rollouts.npz $PV $T/build_steps.py

# 3. discovery (BOSS + BIC; 15-40 min by column count)
#    ROUTE=recboss = the C BOSS (seconds; build ../causal-learn/upstream/causal-get/site first,
#    see its VENDORED.md; float32 scoring needs ROWS+DISCOUNT, e.g. ROWS=50000 DISCOUNT=4)
NPZ=$T/outputs/body_sawyer_steps.npz ROUTE=boss \
  OUT=$T/outputs/body_sawyer_discovered.json $PV $T/discover.py
DISC=$T/outputs/body_sawyer_discovered.json \
  OUT=$T/outputs/body_sawyer_boss_summary.json $PV $T/summarize_graph.py

# 4. truth and discovery score
MUJOCO_GL=egl ROBOT=Sawyer $RV $T/true_graph_any.py
TRUE=$T/outputs/body_sawyer_true.json \
  DISC=$T/outputs/body_sawyer_discovered.json $PV $T/evaluate_vs_truth.py

# 5. Task 1, linear arm (the untrained fallback; run this first on any new dataset)
cd $T/task3_pipeline_v1
env L2_ARM=frozen GENOP=0 RESIDUAL=1.0 LAM_RES=1.0 BRIDGE=pearson NLDEP=1 POLFIX=0 \
  RCHAN=hard CI_MODE=marginal_shrink CUDA_VISIBLE_DEVICES=1 TORCH_THREADS=8 K=400 \
  TASK=1 DATASET=bodysawyer $PV main.py

# 6. training (dev robots only; ~4 min/epoch; canonical = best dev fold-4 match)
env DEV_SETS=liftbody,bodysawyer,bodyiiwa EPOCHS=20 K=400 K_GRAD=60 \
  CUDA_VISIBLE_DEVICES=1 DEVICE=cuda TORCH_THREADS=8 $PV train_body.py

# 7. Task 1 with the trained pair
env L2_ARM=mlp L2_CKPT=$T/task3_pipeline_v1/outputs/wn_body.pt GENOP=1 \
  GENOP_CKPT=$T/task3_pipeline_v1/outputs/gen_operator_body.pt \
  RESIDUAL=1.0 LAM_RES=1.0 BRIDGE=pearson NLDEP=1 POLFIX=0 RCHAN=hard \
  CI_MODE=marginal_shrink CUDA_VISIBLE_DEVICES=1 TORCH_THREADS=8 K=400 \
  GRAPHSEM_DICT=../../v6/outputs/concept_bank_l3_robot.npz \
  TASK=1 DATASET=bodyur5e $PV main.py

# 8. LLM naming baseline (needs judge env)
env DATASET=bodyur5e NAMING_MODEL=openai/gpt-4o-mini \
  OPENAI_API_KEY=$OPENROUTER_API_KEY JUDGE_BASE_URL=https://openrouter.ai/api/v1 \
  JUDGE_MODEL=openai/gpt-5.5 $PV $T/t1_naming_baseline.py
```

Judged runs: keys live in `~/.secrets/env.sh`; judge is gpt-5.5, naming gpt-4o-mini, both via
OpenRouter. `T3_GRAPH=<file in outputs/>` switches the graph a dataset loads (BOSS summary by
default; `*_true_summary.json` for the physics graph).

## task3_pipeline_v1

Full copy of `v6/`, modified for latent-free graphs; the certified v6 is untouched. Changes:
absolute DATA path, `liftbody`/`bodysawyer`/`bodyiiwa`/`bodyur5e` loaders (one mid-episode row
per episode; graph via `T3_GRAPH`; `edge_type` lag/contemp attached and fed to the operator's
type slot), `K_GRAD` truncated backprop in `core.py` (full 400-step backprop NaN-ed; last 60
steps carry gradient), non-finite guard in `train_body.py`, `outputs/` symlinked to
`../../v6/outputs`.

## Results (2026-08-06, match; judge is off-scale strict here, use match)

liftbody (dev, Panda, BOSS graph, chance .14):

| rawcorr | uniform | linear | naming | trained |
|---|---|---|---|---|
| .152 | .257 | .248 | .248 | **.281** |

UR5e (held-out, 6 joints, chance .11):

| uniform | linear | rawcorr | trained | naming |
|---|---|---|---|---|
| .217 | .242 | .328 | **.358** | .361 |

Training: 20 epochs, canonical = epoch 14, dev fold-4 match .3175 against the .2341 zero-init
start. Discovery vs truth: BOSS recall .56-.71, precision .12-.16 across the four robots; the
wrist-roll column of the positional Jacobian is exactly zero on all four, and BOSS-fitted weights
correlate .901 (Panda) / .999 (Sawyer, 3 edges) with the analytic Jacobian.

Known limits: the trained edge over the naming baseline is +.033 on dev and -.003 on held-out;
the questionnaire-trained v6 checkpoints must not be loaded on robot graphs (measured harmful);
the cube and its contact edge are out of scope here (state-dependent edge, deferred).

## Artifacts

| file | content |
|---|---|
| `outputs/body_<r>_steps.npz` | lag-1 design matrix, names, labels |
| `outputs/body_<r>_boss_summary.json` | channel-level graph, signs, edge_types |
| `outputs/body_<r>_true.json` | verified physics graph |
| `outputs/lift_body_*` | Panda equivalents |
| `task3_pipeline_v1/outputs/{gen_operator_body,wn_body}.pt` | canonical trained pair (ep14) |
| `task3_pipeline_v1/outputs/*_ep<k>.pt` | every epoch, for post-hoc screening |
| `outputs/t1_naming_<ds>.json` | naming baseline records |
| `../v6/outputs/concept_bank_l3_robot.npz` | dictionary + 35 index/axis atoms |
