#!/bin/bash
set -u
SD=$(cd "$(dirname "$0")" && pwd)
cd "$SD"
source ~/.secrets/env.sh
export OPENAI_API_KEY=$OPENROUTER_API_KEY
export JUDGE_BASE_URL=https://openrouter.ai/api/v1
export CUDA_VISIBLE_DEVICES=
export ARMS=llmfull,llmphrase,llmgraph
PV=/data2/shuhao/venv/bin/python
OUTD=../outputs/rec_v2_llm
for spec in "rse given ../../v6/outputs/rec_v2/t1_rse_given.json" \
            "rse disc ../outputs/rec_v2/t1_rse_disc.json" \
            "wpi disc ../outputs/rec_v2/t1_wpi_disc.json"; do
  set -- $spec
  O=$OUTD/llm_t1_$1_$2.json
  [ -s "$O" ] && { echo "SKIP $O"; continue; }
  echo "=== t1 $1 $2 $(date +%H:%M:%S)"
  SURFACE=t1 BASE=$1 SOURCE=$2 RECORDS=$3 OUT=$O $PV run_plugin.py
  echo "    rc=$?"
done
echo PILOT_FILL_DONE
