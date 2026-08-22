#!/bin/bash
# Funding gate: launch the joint plugin fleet when the account can afford it, then
# the judge chain at a lower threshold. Thresholds in dollars of available credit.
set -u
CP=/data2/shuhao/semantic_interpretation/causal_llm_plugin
GS=/data2/shuhao/semantic_interpretation/graph_semantics
source ~/.secrets/env.sh
avail() {
  curl -s https://openrouter.ai/api/v1/credits -H "Authorization: Bearer $OPENROUTER_API_KEY" \
   | /data2/shuhao/venv/bin/python -c "import json,sys;d=json.load(sys.stdin)['data'];print(d['total_credits']-d['total_usage'])"
}
J=$GS/discovery/outputs/rec_v2_joint
until [ "$(grep -l 'DONE' $J/lane_*.log 2>/dev/null | wc -l)" -eq 5 ]; do sleep 300; done
echo "joint records complete $(date +%H:%M:%S)"
until A=$(avail) && /data2/shuhao/venv/bin/python -c "exit(0 if float('$A')>=150 else 1)"; do
  echo "waiting for funds: available \$$A $(date +%H:%M)"; sleep 600
done
echo "funds ok (\$$A), launching joint plugin fleet"
bash $CP/experiments/run_joint_plugin.sh
echo FUNDED_FLEET_DONE
