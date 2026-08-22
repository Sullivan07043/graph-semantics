#!/bin/bash
# Open-weight discrete plugin pilot: Qwen3-4B-Instruct-2507, same template and arms.
set -u
CP=/data2/shuhao/semantic_interpretation/causal_llm_plugin
GS=/data2/shuhao/semantic_interpretation/graph_semantics
export LLM_BACKEND=local
export CUDA_VISIBLE_DEVICES=${GPU:-1}
export JUDGE_BASE_URL=
PV=/data2/shuhao/venv/bin/python
O=$GS/discovery/outputs/rec_v2_llm
run() {  # surface base source records
  OUTF=$O/qwen_${1}_${2}_${3}.json
  [ -s "$OUTF" ] && { echo "SKIP $OUTF"; return; }
  echo "=== qwen $1 $2 $3 $(date +%H:%M:%S)"
  SURFACE=$1 BASE=$2 SOURCE=$3 RECORDS=$4 OUT=$OUTF $PV $CP/plugin/run_plugin.py
  echo "    rc=$?"
}
for ds in bigfive cfcs darktriad dass gcbs hexaco himi hs hsq kims mach npas riasec rse scs sd3 sixteenpf tlvd tma wpi wvs; do
  run t2 $ds given $GS/v6/outputs/rec_v2/t2_${ds}_given.json
  [ -f $GS/discovery/outputs/rec_v2/t2_${ds}_disc.json ] && run t2 $ds disc $GS/discovery/outputs/rec_v2/t2_${ds}_disc.json
done
for ds in bigfive hexaco rse wpi dass; do
  run t1 $ds given $GS/v6/outputs/rec_v2/t1_${ds}_given.json
  run t1 $ds disc $GS/discovery/outputs/rec_v2/t1_${ds}_disc.json
done
echo QWEN_PILOT_DONE
