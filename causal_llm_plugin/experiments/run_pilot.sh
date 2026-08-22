#!/bin/bash
# Track A pilot: T2 all datasets (given+disc), T1 five datasets (given+disc),
# T3 base stack (boss+truev3). All arms. Sequential within lane.
set -u
SD=$(cd "$(dirname "$0")" && pwd)
cd "$SD"
source ~/.secrets/env.sh
export OPENAI_API_KEY=$OPENROUTER_API_KEY
export JUDGE_BASE_URL=https://openrouter.ai/api/v1
export CUDA_VISIBLE_DEVICES=
PV=/data2/shuhao/venv/bin/python
OUTD=../outputs/rec_v2_llm
mkdir -p $OUTD
LANE=$1
run() {  # surface base source records
  O=$OUTD/llm_${1}_${2}_${3}.json
  if [ -s "$O" ] && grep -q "\[saved" <(tail -c 200 "$O") 2>/dev/null; then echo "SKIP $O"; return; fi
  [ -s "$O" ] && { echo "SKIP $O"; return; }
  echo "=== $1 $2 $3 $(date +%H:%M:%S)"
  SURFACE=$1 BASE=$2 SOURCE=$3 RECORDS=$4 OUT=$O $PV run_plugin.py
  echo "    rc=$?"
}
DS_ALL="bigfive cfcs darktriad dass gcbs hexaco himi hs hsq kims mach npas riasec rse scs sd3 sixteenpf tlvd tma wpi wvs"
DS_T1="bigfive hexaco rse wpi dass"
case $LANE in
  t2)
    for ds in $DS_ALL; do
      run t2 $ds given ../../v6/outputs/rec_v2/t2_${ds}_given.json
      [ -f ../outputs/rec_v2/t2_${ds}_disc.json ] && run t2 $ds disc ../outputs/rec_v2/t2_${ds}_disc.json
    done;;
  t1)
    for ds in $DS_T1; do
      run t1 $ds given ../../v6/outputs/rec_v2/t1_${ds}_given.json
      run t1 $ds disc ../outputs/rec_v2/t1_${ds}_disc.json
    done;;
  t3)
    for tgt in liftbody bodysawyer bodyiiwa bodyur5e; do
      for g in boss truev3; do
        run t3 $tgt $g ../../task3_robotics/task3_pipeline_v1/outputs/rec_v2/t1_${tgt}_base_${g}.json
      done
    done;;
esac
echo PILOT_LANE_DONE $LANE
