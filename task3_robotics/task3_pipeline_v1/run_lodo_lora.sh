#!/bin/bash
# Relay: wait for the base-E5 lane, then run robot-LoRA LODO (TRAIN_LORA=1, scalar negation):
# per target, train pair+adapter on the source robots, re-encode the bank in the run's space,
# judged eval on the target. Usage: run_lodo_lora.sh <wait_pid>
set -u
cd "$(dirname "$0")"
while ps -p "$1" --no-headers >/dev/null 2>&1; do sleep 60; done
source ~/.secrets/env.sh
PV=/data2/shuhao/venv/bin/python
export CUDA_VISIBLE_DEVICES=1 TORCH_THREADS=8 DEVICE=cuda
export OPENAI_API_KEY="$OPENROUTER_API_KEY"
export JUDGE_BASE_URL=https://openrouter.ai/api/v1
export JUDGE_MODEL=openai/gpt-5.5
declare -A SRC=( [liftbody]=bodysawyer,bodyiiwa [bodysawyer]=liftbody,bodyiiwa \
                 [bodyiiwa]=liftbody,bodysawyer [bodyur5e]=liftbody,bodysawyer,bodyiiwa )
for tgt in liftbody bodysawyer bodyiiwa bodyur5e; do
  DIR=outputs_lodo/${tgt}_rlora
  mkdir -p $DIR
  ln -sf ../../outputs/l2_mlp.pt $DIR/l2_mlp.pt
  ln -sf ../../outputs/l3_lora.pt $DIR/l3_lora.pt      # start point: questionnaire adapter
  echo "=== train $tgt (rlora) $(date +%H:%M:%S)"
  env DEV_SETS=${SRC[$tgt]} EPOCHS=20 K=400 K_GRAD=60 GENOP_NEGATIVE_MODE=scalar \
    TRAIN_LORA=1 GENOP_CKPT=$DIR/gen_operator_body.pt \
    CORE_OUTPUT_DIR=$DIR $PV train_body.py > $DIR/train.log 2>&1
  echo "    rc=$?"
  echo "=== reencode bank $tgt $(date +%H:%M:%S)"
  env LORA_CKPT=$DIR/lora_body.pt OUT=$DIR/bank_rlora.npz $PV reencode_bank.py \
    > $DIR/reencode.log 2>&1
  echo "    rc=$?"
  echo "=== eval $tgt (rlora) $(date +%H:%M:%S)"
  env L2_ARM=mlp L2_CKPT=$DIR/wn_body.pt GENOP=1 GENOP_CKPT=$DIR/gen_operator_body.pt \
    GENOP_NEGATIVE_MODE=scalar LORA_CKPT=$DIR/lora_body.pt RESIDUAL=1.0 LAM_RES=1.0 \
    BRIDGE=pearson NLDEP=1 POLFIX=0 RCHAN=hard CI_MODE=marginal_shrink K=400 \
    GRAPHSEM_DICT=$DIR/bank_rlora.npz \
    TASK=1 DATASET=$tgt $PV main.py > $DIR/eval.log 2>&1
  echo "    rc=$?"
done
echo "LODO_RLORA_DONE"
