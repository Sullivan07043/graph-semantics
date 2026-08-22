#!/bin/bash
# Judge gate: after the funded fleet finishes, fill judge scores for every record
# family (plugin, stress, joint, plus any unfilled stage A rows) when credit allows.
set -u
CP=/data2/shuhao/semantic_interpretation/causal_llm_plugin
GS=/data2/shuhao/semantic_interpretation/graph_semantics
source ~/.secrets/env.sh
avail() {
  curl -s https://openrouter.ai/api/v1/credits -H "Authorization: Bearer $OPENROUTER_API_KEY" \
   | /data2/shuhao/venv/bin/python -c "import json,sys;d=json.load(sys.stdin)['data'];print(d['total_credits']-d['total_usage'])"
}
until grep -q FUNDED_FLEET_DONE $CP/outputs/funded_gate.log 2>/dev/null; do sleep 600; done
until A=$(avail) && /data2/shuhao/venv/bin/python -c "exit(0 if float('$A')>=20 else 1)"; do
  echo "judge waiting for funds: \$$A"; sleep 600
done
echo "judge starting (\$$A)"
export OPENAI_API_KEY=$OPENROUTER_API_KEY
export JUDGE_BASE_URL=https://openrouter.ai/api/v1
export JUDGE_MODEL=openai/gpt-5.5
/data2/shuhao/venv/bin/python $GS/discovery/stageB_judge.py
echo "judge rc=$?"
echo JUDGE_ALL_DONE
