#!/bin/bash
set -u
CP=/data2/shuhao/semantic_interpretation/causal_llm_plugin
GS=/data2/shuhao/semantic_interpretation/graph_semantics
until grep -q JOINT_PLUGIN_DONE $CP/outputs/joint_plugin_fleet.log 2>/dev/null; do sleep 300; done
source ~/.secrets/env.sh
export OPENAI_API_KEY=$OPENROUTER_API_KEY
export JUDGE_BASE_URL=https://openrouter.ai/api/v1
export JUDGE_MODEL=openai/gpt-5.5
/data2/shuhao/venv/bin/python $GS/discovery/stageB_judge.py; echo "judge rc=$?"
cd $CP/scoring
CUDA_VISIBLE_DEVICES=0 /data2/shuhao/venv/bin/python gen_llm_report.py; echo "llmrep rc=$?"
CUDA_VISIBLE_DEVICES=0 /data2/shuhao/venv/bin/python gen_plugin_tables.py; echo "tables rc=$?"
cd $GS/../report/plugin_experiment && /data2/shuhao/venv/bin/tectonic plugin_experiment.tex >/dev/null 2>&1; echo "pdf rc=$?"
echo FINAL_CHAIN_DONE
