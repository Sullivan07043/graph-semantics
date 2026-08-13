#!/bin/bash
# Closed loop for the post-freeze testbeds: T1+T2 on GPU-RLCD discovered graphs, judged.
# Given-graph references live in v6/outputs/t{1,2}_{dass,wvs}_v6cert.json.
set -u
cd "$(dirname "$0")"
source ~/.secrets/env.sh
PV=/data2/shuhao/venv/bin/python
export OPENAI_API_KEY="$OPENROUTER_API_KEY"
export JUDGE_BASE_URL=https://openrouter.ai/api/v1
export JUDGE_MODEL=openai/gpt-5.5
export CUDA_VISIBLE_DEVICES=1
export TORCH_THREADS=8
for ds in dass wvs; do
  for task in 1 2; do
    echo "=== $ds task$task $(date +%H:%M:%S)"
    BASE=$ds TASK=$task GRAPH_JSON=${ds}_gpurlcd.json TAG=gpurlcd \
      $PV run_downstream.py > outputs/closedloop_${ds}_t${task}.log 2>&1
    echo "    rc=$?"
  done
done
echo CLOSEDLOOP_DONE
