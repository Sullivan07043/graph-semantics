#!/bin/bash
# Robot Track B on GPU0: wait for the vLLM T3 filler, dump robot embeddings (with a
# reproduction check against the existing records), then leave-one-robot-out
# mapper training + eval.
set -u
CP=/data2/shuhao/semantic_interpretation/causal_llm_plugin
GS=/data2/shuhao/semantic_interpretation/graph_semantics
T3P=$GS/task3_robotics/task3_pipeline_v1
E=$GS/discovery/outputs/emb_v2
PV=/data2/shuhao/venv/bin/python
until [ "$(ls $GS/discovery/outputs/rec_v2_llm/qwen_t3_*.json 2>/dev/null | wc -l)" -eq 8 ]; do
  grep -q Traceback $CP/outputs/vllm_fill_t3.log 2>/dev/null && { echo T3_FILLER_FAILED; exit 1; }
  sleep 120
done
echo "qwen t3 complete $(date +%H:%M:%S)"
cd $T3P
export RESIDUAL=1.0 LAM_RES=1.0 BRIDGE=pearson NLDEP=1 POLFIX=0 RCHAN=hard
export CI_MODE=marginal_shrink K=400 FORCE_DECODE=1
export OPENAI_API_KEY= JUDGE_MODEL=
export CUDA_VISIBLE_DEVICES=0 TORCH_THREADS=8
declare -A TS=( [liftbody]=lift_body_true_summary_v3.json [bodysawyer]=body_sawyer_true_summary_v3.json \
                [bodyiiwa]=body_iiwa_true_summary_v3.json [bodyur5e]=body_ur5e_true_summary_v3.json )
declare -A BS=( [liftbody]=lift_body_summary.json [bodysawyer]=body_sawyer_boss_summary.json \
                [bodyiiwa]=body_iiwa_boss_summary.json [bodyur5e]=body_ur5e_boss_summary.json )
for tgt in liftbody bodysawyer bodyiiwa bodyur5e; do
  DIR=outputs_lodo/${tgt}_base
  for src in boss truev3; do
    NPZ=$E/robot_${tgt}_${src}_emb.npz
    [ -s "$NPZ" ] && { echo "dump SKIP $tgt $src"; continue; }
    G=${BS[$tgt]}; [ $src = truev3 ] && G=${TS[$tgt]}
    echo "=== dump $tgt $src $(date +%H:%M:%S)"
    env L2_ARM=mlp GENOP=1 L2_CKPT=$DIR/wn_body.pt GENOP_CKPT=$DIR/gen_operator_body.pt \
      GENOP_NEGATIVE_MODE=scalar CORE_ENCODER_MODE=base GRAPHSEM_DICT=outputs_lodo/bank_base.npz \
      EMB_DUMP=$NPZ T3_GRAPH=$G TASK=1 DATASET=$tgt \
      RECORDS_OUT=/tmp/rdump_${tgt}_${src}.json \
      $PV main.py > $E/log_robot_${tgt}_${src}.txt 2>&1
    echo "    rc=$?"
    $PV - <<PYEOF
import json
a=json.load(open("/tmp/rdump_${tgt}_${src}.json"))
b=json.load(open("$T3P/outputs/rec_v2/t1_${tgt}_base_${src}.json"))
ca={(r["fold"],r["var"]):r["decoded_words"] for r in a["records"] if r["arm"]=="core"}
cb={(r["fold"],r["var"]):r["decoded_words"] for r in b["records"] if r["arm"]=="core"}
same=sum(1 for k in cb if ca.get(k)==cb[k])
print(f"    GATE {'ok' if same==len(cb) else 'FAIL'} ({same}/{len(cb)})")
PYEOF
  done
done
CK=$CP/outputs/mappers
for tgt in liftbody bodysawyer bodyiiwa bodyur5e; do
  CKPT=$CK/mapper_robot_${tgt}.pt
  if [ ! -s "$CKPT" ]; then
    echo "=== train robot $tgt $(date +%H:%M:%S)"
    DOMAIN=robot MODE=train EVAL_DS=$tgt EPOCHS=8 OUT=$CKPT CUDA_VISIBLE_DEVICES=0 \
      $PV $CP/prefix/prefix_mapper.py > $CK/log_train_robot_${tgt}.txt 2>&1
    echo "    rc=$?"
  fi
  for src in boss truev3; do
    OUTF=$GS/discovery/outputs/rec_v2_llm/pfx_t3_${tgt}_${src}.json
    [ -s "$OUTF" ] && continue
    echo "=== eval robot $tgt $src $(date +%H:%M:%S)"
    DOMAIN=robot MODE=eval EVAL_DS=$tgt SRC=$src CKPT=$CKPT OUT=$OUTF CUDA_VISIBLE_DEVICES=0 \
      $PV $CP/prefix/prefix_mapper.py > $CK/log_eval_robot_${tgt}_${src}.txt 2>&1
    echo "    rc=$?"
  done
done
echo ROBOT_PREFIX_DONE
