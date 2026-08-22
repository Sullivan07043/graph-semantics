#!/bin/bash
# Plugin arms on the merged T1+2 records, all 21 datasets x given/disc.
# Arms in priority order so a funding cut preserves the substance arms first.
set -u
CP=/data2/shuhao/semantic_interpretation/causal_llm_plugin
GS=/data2/shuhao/semantic_interpretation/graph_semantics
source ~/.secrets/env.sh
export OPENAI_API_KEY=$OPENROUTER_API_KEY
export JUDGE_BASE_URL=https://openrouter.ai/api/v1
export CUDA_VISIBLE_DEVICES=
PV=/data2/shuhao/venv/bin/python
J=$GS/discovery/outputs/rec_v2_joint
O=$GS/discovery/outputs/rec_v2_llm
for arm in llmfull llmgraph llmphrase llmplacebo; do
  for ds in bigfive cfcs darktriad dass gcbs hexaco himi hs hsq kims mach npas riasec rse scs sd3 sixteenpf tlvd tma wpi wvs; do
    for src in given disc; do
      R=$J/t12_${ds}_${src}.json
      [ -s "$R" ] || continue
      OUTF=$O/llm_t12_${ds}_${src}_${arm}.json
      [ -s "$OUTF" ] && continue
      echo "=== t12 $ds $src $arm $(date +%H:%M:%S)"
      SURFACE=t12 BASE=$ds SOURCE=$src RECORDS=$R OUT=$OUTF ARMS=$arm $PV $CP/plugin/run_plugin.py
      echo "    rc=$?"
    done
  done
  echo "ARM_PASS_DONE $arm"
done
echo JOINT_PLUGIN_DONE
