#!/bin/bash
# Rerun of the final-chain judge step after the mixed-mode batching fix
# in stageB_judge.py (merged T1+2 files hold item and latent rows in one
# file; batches are now grouped by mode). Then regenerate the report,
# the tables, and the PDF. Marker: JUDGE_REDO_DONE.
set -u
CP=/data2/shuhao/semantic_interpretation/causal_llm_plugin
GS=/data2/shuhao/semantic_interpretation/graph_semantics
source ~/.secrets/env.sh
export OPENAI_API_KEY=$OPENROUTER_API_KEY
export JUDGE_BASE_URL=https://openrouter.ai/api/v1
export JUDGE_MODEL=openai/gpt-5.5
/data2/shuhao/venv/bin/python $GS/discovery/stageB_judge.py; rc=$?
echo "judge rc=$rc"
if [ $rc -ne 0 ]; then echo "JUDGE_REDO_FAILED"; exit 1; fi
cd $CP/scoring
CUDA_VISIBLE_DEVICES=0 /data2/shuhao/venv/bin/python gen_llm_report.py; echo "llmrep rc=$?"
CUDA_VISIBLE_DEVICES=0 /data2/shuhao/venv/bin/python gen_plugin_tables.py; echo "tables rc=$?"
cd $GS/../report/plugin_experiment && /data2/shuhao/venv/bin/tectonic plugin_experiment.tex >/dev/null 2>&1; echo "pdf rc=$?"
echo JUDGE_REDO_DONE
