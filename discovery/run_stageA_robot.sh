#!/bin/bash
# Stage A3: robot LODO evals, all arms x both graph conditions, judge OFF, K=400.
# Usage: run_stageA_robot.sh <targets...>  (waits for A1/A2 lanes to clear first)
set -u
SD=$(cd "$(dirname "$0")" && pwd)
P=$SD/../task3_robotics/task3_pipeline_v1
cd "$P"
while pgrep -f "run_stageA_given.sh|run_stageA_disc.sh" >/dev/null 2>&1; do sleep 120; done
PV=/data2/shuhao/venv/bin/python
export RESIDUAL=1.0 LAM_RES=1.0 BRIDGE=pearson NLDEP=1 POLFIX=0 RCHAN=hard
export CI_MODE=marginal_shrink K=400 FORCE_DECODE=1
export OPENAI_API_KEY= JUDGE_MODEL=
export CUDA_VISIBLE_DEVICES=1 TORCH_THREADS=8
mkdir -p outputs/rec_v2
declare -A TS=( [liftbody]=lift_body_true_summary_v2.json [bodysawyer]=body_sawyer_true_summary_v2.json \
                [bodyiiwa]=body_iiwa_true_summary_v2.json [bodyur5e]=body_ur5e_true_summary_v2.json )
RB=../../v6/outputs/concept_bank_l3_robot.npz
for tgt in "$@"; do
  for graph in boss true; do
    G=""
    [ "$graph" = "true" ] && G="${TS[$tgt]}"
    for stack in base scalar semantic rlora linear; do
      DIR=outputs_lodo/${tgt}_${stack}
      case $stack in
        base)     EX="L2_ARM=mlp GENOP=1 L2_CKPT=$DIR/wn_body.pt GENOP_CKPT=$DIR/gen_operator_body.pt GENOP_NEGATIVE_MODE=scalar CORE_ENCODER_MODE=base GRAPHSEM_DICT=outputs_lodo/bank_base.npz";;
        scalar)   EX="L2_ARM=mlp GENOP=1 L2_CKPT=$DIR/wn_body.pt GENOP_CKPT=$DIR/gen_operator_body.pt GENOP_NEGATIVE_MODE=scalar GRAPHSEM_DICT=$RB";;
        semantic) EX="L2_ARM=mlp GENOP=1 L2_CKPT=$DIR/wn_body.pt GENOP_CKPT=$DIR/gen_operator_body.pt GENOP_NEGATIVE_MODE=semantic GRAPHSEM_DICT=$RB";;
        rlora)    EX="L2_ARM=mlp GENOP=1 L2_CKPT=$DIR/wn_body.pt GENOP_CKPT=$DIR/gen_operator_body.pt GENOP_NEGATIVE_MODE=scalar LORA_CKPT=$DIR/lora_body.pt GRAPHSEM_DICT=$DIR/bank_rlora.npz";;
        linear)   EX="L2_ARM=frozen GENOP=0 GRAPHSEM_DICT=$RB";;
      esac
      echo "=== $tgt $stack $graph $(date +%H:%M:%S)"
      env $EX T3_GRAPH="$G" TASK=1 DATASET=$tgt \
        RECORDS_OUT=outputs/rec_v2/t1_${tgt}_${stack}_${graph}.json \
        $PV main.py > outputs/rec_v2/log_${tgt}_${stack}_${graph}.txt 2>&1
      echo "    rc=$?"
    done
  done
done
echo "ROBOT_LANE_DONE $*"
