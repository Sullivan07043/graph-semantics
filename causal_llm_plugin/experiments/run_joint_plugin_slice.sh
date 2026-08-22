#!/bin/bash
# Dataset-sliced parallel lane of the joint plugin fleet. Usage: <lane-id> <ds...>
set -u
CP=/data2/shuhao/semantic_interpretation/causal_llm_plugin
GS=/data2/shuhao/semantic_interpretation/graph_semantics
source ~/.secrets/env.sh
export OPENAI_API_KEY=$OPENROUTER_API_KEY
export JUDGE_BASE_URL=https://openrouter.ai/api/v1
export CUDA_VISIBLE_DEVICES=
PV=/data2/shuhao/venv/bin/python
J=$GS/discovery/outputs/rec_v2_joint
O=$GS/discovery/outputs/rec_v2_llm
LANE=$1; shift
for arm in llmfull llmgraph llmphrase llmplacebo; do
  for ds in "$@"; do
    for src in given disc; do
      R=$J/t12_${ds}_${src}.json
      [ -s "$R" ] || continue
      OUTF=$O/llm_t12_${ds}_${src}_${arm}.json
      [ -s "$OUTF" ] && continue
      echo "=== t12 $ds $src $arm $(date +%H:%M:%S)"
      SURFACE=t12 BASE=$ds SOURCE=$src RECORDS=$R OUT=$OUTF ARMS=$arm $PV $CP/plugin/run_plugin.py
      echo "    rc=$?"
    done
  done
done
echo SLICE_DONE $LANE
