#!/bin/bash
# Embedding dump for the prefix variant: rerun the frozen solve per dataset x source,
# saving each node's masked-fold embedding (+ latents per fold). Records go to /tmp.
set -u
GS=/data2/shuhao/semantic_interpretation/graph_semantics
source $GS/discovery/stageA_env.sh
export JOINT_LATENTS=1
export CUDA_VISIBLE_DEVICES=${GPU:-0}
PV=/data2/shuhao/venv/bin/python
E=$GS/discovery/outputs/emb_v2
SRC=$1; shift
for ds in "$@"; do
  NPZ=$E/${ds}_${SRC}_emb.npz
  [ -s "$NPZ" ] && { echo "=== $ds $SRC SKIP"; continue; }
  echo "=== $ds $SRC $(date +%H:%M:%S)"
  if [ "$SRC" = given ]; then
    cd $GS/v6
    EMB_DUMP=$NPZ TASK=1 DATASET=$ds RECORDS_OUT=/tmp/edump_${ds}_given.json $PV main.py > $E/log_${ds}_given.txt 2>&1
  else
    cd $GS/discovery
    EMB_DUMP=$NPZ BASE=$ds TASK=1 GRAPH_JSON=${ds}_gpurlcd.json TAG=gpurlcd RECORDS_OUT=/tmp/edump_${ds}_disc.json $PV run_downstream.py > $E/log_${ds}_disc.txt 2>&1
  fi
  echo "    rc=$?"
done
echo EMBDUMP_DONE $SRC $*
