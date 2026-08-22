#!/bin/bash
# Merged T1+2 records, given graphs. Usage: run_joint_given.sh <datasets...>
set -u
GS=/data2/shuhao/semantic_interpretation/graph_semantics
source $GS/discovery/stageA_env.sh
export JOINT_LATENTS=1
export CUDA_VISIBLE_DEVICES=${GPU:-0}
cd $GS/v6
PV=/data2/shuhao/venv/bin/python
for ds in "$@"; do
  REC=$GS/discovery/outputs/rec_v2_joint/t12_${ds}_given.json
  LOG=$GS/discovery/outputs/rec_v2_joint/log_${ds}_given.txt
  if [ -s "$REC" ] && grep -q "\[saved" "$LOG" 2>/dev/null; then echo "=== $ds SKIP"; continue; fi
  echo "=== $ds given $(date +%H:%M:%S)"
  TASK=1 DATASET=$ds RECORDS_OUT=$REC $PV main.py > "$LOG" 2>&1
  echo "    rc=$?"
done
echo JOINT_GIVEN_DONE $*
