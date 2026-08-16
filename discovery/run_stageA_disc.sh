#!/bin/bash
# Stage A2: discovered-graph questionnaires via run_downstream, judge OFF. Usage: <lane datasets...>
set -u
cd "$(dirname "$0")"
PV=/data2/shuhao/venv/bin/python
source "$(dirname "$0")/stageA_env.sh"
NOLAT=" sd3 tlvd "
for ds in "$@"; do
  for task in 1 2; do
    if [ "$task" = "2" ] && [[ "$NOLAT" == *" $ds "* ]]; then
      echo "=== $ds t2 SKIPPED (0 latents)"; continue
    fi
    echo "=== $ds t$task $(date +%H:%M:%S)"
    BASE=$ds TASK=$task GRAPH_JSON=${ds}_gpurlcd.json TAG=gpurlcd \
      RECORDS_OUT=outputs/rec_v2/t${task}_${ds}_disc.json \
      $PV run_downstream.py > outputs/rec_v2/log_t${task}_${ds}_disc.txt 2>&1
    echo "    rc=$?"
  done
done
echo "DISC_LANE_DONE $*"
