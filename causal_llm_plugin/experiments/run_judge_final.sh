#!/bin/bash
set -u
CP=/data2/shuhao/semantic_interpretation/causal_llm_plugin
GS=/data2/shuhao/semantic_interpretation/graph_semantics
source ~/.secrets/env.sh
export OPENAI_API_KEY=$OPENROUTER_API_KEY
export JUDGE_BASE_URL=https://openrouter.ai/api/v1
export JUDGE_MODEL=openai/gpt-5.5
/data2/shuhao/venv/bin/python $GS/discovery/stageB_judge.py; echo "judge rc=$?"
cd $CP/scoring
CUDA_VISIBLE_DEVICES=0 /data2/shuhao/venv/bin/python gen_llm_report.py; echo "rep rc=$?"
CUDA_VISIBLE_DEVICES=0 /data2/shuhao/venv/bin/python gen_plugin_tables.py; echo "tab rc=$?"
cd $GS/../report/plugin_experiment && /data2/shuhao/venv/bin/tectonic plugin_experiment.tex >/dev/null 2>&1; echo "pdf rc=$?"
echo JUDGE_FINAL_DONE
