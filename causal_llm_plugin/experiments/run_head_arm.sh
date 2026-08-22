#!/bin/bash
# T3 llmhead arm (gpt-5.5): phrases + graph + dt fact + the deterministic naming
# head's evidence-backed proposal. 4 robots x 2 graph modes, ARMS=llmhead only.
# Output: llm_t3h_<robot>_<mode>.json in the canonical rec_v2_llm dir.
# Marker: HEAD_ARM_DONE.
set -u
CP=/data2/shuhao/semantic_interpretation/causal_llm_plugin
GS=/data2/shuhao/semantic_interpretation/graph_semantics
source ~/.secrets/env.sh
export OPENAI_API_KEY=$OPENROUTER_API_KEY
export JUDGE_BASE_URL=https://openrouter.ai/api/v1
export CUDA_VISIBLE_DEVICES=
PV=/data2/shuhao/venv/bin/python
OUTD=$GS/discovery/outputs/rec_v2_llm
for tgt in liftbody bodysawyer bodyiiwa bodyur5e; do
  for g in boss truev3; do
    O=$OUTD/llm_t3h_${tgt}_${g}.json
    [ -s "$O" ] && { echo "SKIP $O"; continue; }
    echo "=== t3h $tgt $g $(date +%H:%M:%S)"
    SURFACE=t3 BASE=$tgt SOURCE=$g ARMS=llmhead \
      RECORDS=$GS/task3_robotics/task3_pipeline_v1/outputs/rec_v2/t1_${tgt}_base_${g}.json \
      OUT=$O $PV $CP/plugin/run_plugin.py
    echo "    rc=$?"
  done
done
echo HEAD_ARM_DONE
