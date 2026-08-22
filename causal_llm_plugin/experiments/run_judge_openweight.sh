#!/bin/bash
# Judge fill for the open-weight arms (qwen_* discrete, pfx_* prefix records),
# enabled by the qwen_*/pfx_* globs added to stageB_judge.py. Judge only; the
# tables and the PDF are regenerated later in one pass. Marker: JUDGE_OW_DONE.
set -u
GS=/data2/shuhao/semantic_interpretation/graph_semantics
source ~/.secrets/env.sh
export OPENAI_API_KEY=$OPENROUTER_API_KEY
export JUDGE_BASE_URL=https://openrouter.ai/api/v1
export JUDGE_MODEL=openai/gpt-5.5
/data2/shuhao/venv/bin/python $GS/discovery/stageB_judge.py; rc=$?
echo "judge rc=$rc"
if [ $rc -ne 0 ]; then echo "JUDGE_OW_FAILED"; exit 1; fi
echo JUDGE_OW_DONE
