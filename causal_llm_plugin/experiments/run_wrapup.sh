#!/bin/bash
# Final wrap-up, single writer. Waits for the three producers (open-weight judge,
# gpt-5.5 head arm, Qwen head arm), then: one judge pass for the head-arm files,
# report + tables regeneration, PDF compile. Fail-closed on the judge step.
# Marker: WRAPUP_DONE / WRAPUP_FAILED.
set -u
CP=/data2/shuhao/semantic_interpretation/causal_llm_plugin
GS=/data2/shuhao/semantic_interpretation/graph_semantics
until grep -q JUDGE_OW_DONE $CP/outputs/judge_ow.log 2>/dev/null \
   && grep -q HEAD_ARM_DONE $CP/outputs/head_arm.log 2>/dev/null \
   && grep -q "QWEN_HEAD_DONE rc=0" $CP/outputs/qwen_head.log 2>/dev/null; do sleep 120; done
source ~/.secrets/env.sh
export OPENAI_API_KEY=$OPENROUTER_API_KEY
export JUDGE_BASE_URL=https://openrouter.ai/api/v1
export JUDGE_MODEL=openai/gpt-5.5
/data2/shuhao/venv/bin/python $GS/discovery/stageB_judge.py; rc=$?
echo "judge rc=$rc"
if [ $rc -ne 0 ]; then echo "WRAPUP_FAILED"; exit 1; fi
cd $CP/scoring
CUDA_VISIBLE_DEVICES=0 /data2/shuhao/venv/bin/python gen_llm_report.py; echo "llmrep rc=$?"
CUDA_VISIBLE_DEVICES=0 /data2/shuhao/venv/bin/python gen_plugin_tables.py; rc=$?
echo "tables rc=$rc"
if [ $rc -ne 0 ]; then echo "WRAPUP_FAILED"; exit 1; fi
cd $GS/../report/plugin_experiment && /data2/shuhao/venv/bin/tectonic plugin_experiment.tex >/dev/null 2>&1; rc=$?
echo "pdf rc=$rc"
if [ $rc -ne 0 ]; then echo "WRAPUP_FAILED"; exit 1; fi
echo WRAPUP_DONE
