#!/bin/bash
# Stage A1: given-graph questionnaires, judge OFF, records per run. Usage: <lane datasets...>
set -u
cd "$(dirname "$0")/../v6"
PV=/data2/shuhao/venv/bin/python
source "$(dirname "$0")/stageA_env.sh"
for ds in "$@"; do
  for task in 1 2; do
    echo "=== $ds t$task $(date +%H:%M:%S)"
    TASK=$task DATASET=$ds RECORDS_OUT=outputs/rec_v2/t${task}_${ds}_given.json \
      $PV main.py > outputs/rec_v2/log_t${task}_${ds}_given.txt 2>&1
    echo "    rc=$?"
  done
done
echo "GIVEN_LANE_DONE $*"
