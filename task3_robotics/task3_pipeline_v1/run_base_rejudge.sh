#!/bin/bash
# Complete the base lane's judge column: one base-space bank, four judged re-evals.
set -u
cd "$(dirname "$0")"
source ~/.secrets/env.sh
PV=/data2/shuhao/venv/bin/python
export CUDA_VISIBLE_DEVICES=1 TORCH_THREADS=8
export OPENAI_API_KEY="$OPENROUTER_API_KEY"
export JUDGE_BASE_URL=https://openrouter.ai/api/v1
export JUDGE_MODEL=openai/gpt-5.5
BANK=outputs_lodo/bank_base.npz
echo "=== reencode base bank $(date +%H:%M:%S)"
env ENCODER=base OUT=$BANK $PV reencode_bank.py > outputs_lodo/reencode_base.log 2>&1
echo "    rc=$?"
for tgt in liftbody bodysawyer bodyiiwa bodyur5e; do
  DIR=outputs_lodo/${tgt}_base
  echo "=== rejudge $tgt (base) $(date +%H:%M:%S)"
  env L2_ARM=mlp L2_CKPT=$DIR/wn_body.pt GENOP=1 GENOP_CKPT=$DIR/gen_operator_body.pt \
    GENOP_NEGATIVE_MODE=scalar CORE_ENCODER_MODE=base RESIDUAL=1.0 LAM_RES=1.0 \
    BRIDGE=pearson NLDEP=1 POLFIX=0 RCHAN=hard CI_MODE=marginal_shrink K=400 \
    GRAPHSEM_DICT=$BANK \
    TASK=1 DATASET=$tgt $PV main.py > $DIR/eval_judged.log 2>&1
  echo "    rc=$?"
done
echo "BASE_REJUDGE_DONE"
