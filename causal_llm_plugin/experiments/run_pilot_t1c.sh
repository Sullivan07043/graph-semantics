#!/bin/bash
# Parallel T1 sub-lane: the datasets the main t1 lane has not reached yet.
set -u
SD=$(cd "$(dirname "$0")" && pwd)
cd "$SD"
source ~/.secrets/env.sh
export OPENAI_API_KEY=$OPENROUTER_API_KEY
export JUDGE_BASE_URL=https://openrouter.ai/api/v1
export CUDA_VISIBLE_DEVICES=
PV=/data2/shuhao/venv/bin/python
OUTD=../outputs/rec_v2_llm
for ds in dass rse; do
  for src in given disc; do
    R=../../v6/outputs/rec_v2/t1_${ds}_given.json
    [ $src = disc ] && R=../outputs/rec_v2/t1_${ds}_disc.json
    O=$OUTD/llm_t1_${ds}_${src}.json
    [ -s "$O" ] && { echo "SKIP $O"; continue; }
    echo "=== t1 $ds $src $(date +%H:%M:%S)"
    SURFACE=t1 BASE=$ds SOURCE=$src RECORDS=$R OUT=$O $PV run_plugin.py
    echo "    rc=$?"
  done
done
echo PILOT_LANE_DONE t1c
