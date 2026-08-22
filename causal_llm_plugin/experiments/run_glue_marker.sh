#!/bin/bash
# Session-independent glue: when all four API slices are done, append the fleet
# marker that wakes the judge/tables/PDF final chain. Exits after appending.
set -u
CP=/data2/shuhao/semantic_interpretation/causal_llm_plugin
while true; do
  N=$(grep -h SLICE_DONE $CP/outputs/joint_slice_*.log 2>/dev/null | wc -l)
  if [ "$N" -eq 4 ]; then
    grep -q JOINT_PLUGIN_DONE $CP/outputs/joint_plugin_fleet.log 2>/dev/null || \
      echo JOINT_PLUGIN_DONE >> $CP/outputs/joint_plugin_fleet.log
    echo "glue: marker ensured $(date +%H:%M:%S)"
    exit 0
  fi
  sleep 300
done
