#!/bin/bash
# Merged T1+2 records, discovered graphs. Usage: run_joint_disc.sh <datasets...>
set -u
GS=/data2/shuhao/semantic_interpretation/graph_semantics
source $GS/discovery/stageA_env.sh
export JOINT_LATENTS=1
export CUDA_VISIBLE_DEVICES=${GPU:-0}
cd $GS/discovery
PV=/data2/shuhao/venv/bin/python
for ds in "$@"; do
  REC=$GS/discovery/outputs/rec_v2_joint/t12_${ds}_disc.json
  LOG=$GS/discovery/outputs/rec_v2_joint/log_${ds}_disc.txt
  if [ -s "$REC" ] && grep -q "\[saved" "$LOG" 2>/dev/null; then echo "=== $ds SKIP"; continue; fi
  echo "=== $ds disc $(date +%H:%M:%S)"
  BASE=$ds TASK=1 GRAPH_JSON=${ds}_gpurlcd.json TAG=gpurlcd RECORDS_OUT=$REC $PV run_downstream.py > "$LOG" 2>&1
  echo "    rc=$?"
done
echo JOINT_DISC_DONE $*
