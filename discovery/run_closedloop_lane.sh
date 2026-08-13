#!/bin/bash
# One lane: for each dataset, T1 then T2 (T2 skipped when the discovered graph has no latents).
set -u
cd "$(dirname "$0")"
source ~/.secrets/env.sh
PV=/data2/shuhao/venv/bin/python
export OPENAI_API_KEY="$OPENROUTER_API_KEY"
export JUDGE_BASE_URL=https://openrouter.ai/api/v1
export JUDGE_MODEL=openai/gpt-5.5
export CUDA_VISIBLE_DEVICES=1
export TORCH_THREADS=8
for ds in "$@"; do
  echo "=== $ds t1 $(date +%H:%M:%S)"
  BASE=$ds TASK=1 GRAPH_JSON=${ds}_gpurlcd.json TAG=gpurlcd \
    $PV run_downstream.py > outputs/closedloop_${ds}_t1.log 2>&1
  echo "    rc=$?"
  nlat=$($PV -c "import json; d=json.load(open('outputs/${ds}_gpurlcd.json')); print(len({x for e in d['rlcd_directed'] for x in e if x.startswith('L') and x[1:].isdigit()}))")
  if [ "$nlat" -gt 0 ]; then
    echo "=== $ds t2 $(date +%H:%M:%S)"
    BASE=$ds TASK=2 GRAPH_JSON=${ds}_gpurlcd.json TAG=gpurlcd \
      $PV run_downstream.py > outputs/closedloop_${ds}_t2.log 2>&1
    echo "    rc=$?"
  else
    echo "=== $ds t2 SKIPPED (0 latents)"
  fi
done
echo LANE_DONE
