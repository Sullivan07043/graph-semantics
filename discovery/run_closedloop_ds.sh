#!/bin/bash
# One dataset's closed-loop chain: T1 then T2 (same-dataset nldep cache forbids overlap).
# Usage: run_closedloop_ds.sh <dataset> [wait_pid]
set -u
cd "$(dirname "$0")"
source ~/.secrets/env.sh
PV=/data2/shuhao/venv/bin/python
export OPENAI_API_KEY="$OPENROUTER_API_KEY"
export JUDGE_BASE_URL=https://openrouter.ai/api/v1
export JUDGE_MODEL=openai/gpt-5.5
export CUDA_VISIBLE_DEVICES=1
export TORCH_THREADS=8
ds=$1
if [ "${2:-}" ]; then
  while ps -p "$2" --no-headers >/dev/null 2>&1; do sleep 10; done
else
  echo "=== $ds task1 $(date +%H:%M:%S)"
  BASE=$ds TASK=1 GRAPH_JSON=${ds}_gpurlcd.json TAG=gpurlcd \
    $PV run_downstream.py > outputs/closedloop_${ds}_t1.log 2>&1
  echo "    rc=$?"
fi
echo "=== $ds task2 $(date +%H:%M:%S)"
BASE=$ds TASK=2 GRAPH_JSON=${ds}_gpurlcd.json TAG=gpurlcd \
  $PV run_downstream.py > outputs/closedloop_${ds}_t2.log 2>&1
echo "    rc=$?"
echo "${ds}_CHAIN_DONE"
