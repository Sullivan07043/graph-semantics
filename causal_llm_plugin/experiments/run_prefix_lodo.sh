#!/bin/bash
# Track B full run: leave-one-dataset-out mapper training + eval on both graph
# sources, per dataset. Waits for the embedding dumps and the Qwen pilot lane.
set -u
CP=/data2/shuhao/semantic_interpretation/causal_llm_plugin
GS=/data2/shuhao/semantic_interpretation/graph_semantics
E=$GS/discovery/outputs/emb_v2
until grep -q "EMBDUMP_DONE given" $E/lane_given.log 2>/dev/null \
   && grep -q "EMBDUMP_DONE disc" $E/lane_disc.log 2>/dev/null \
   && grep -q QWEN_PILOT_DONE $CP/outputs/qwen_pilot.log 2>/dev/null; do sleep 300; done
export CUDA_VISIBLE_DEVICES=${GPU:-1}
PV=/data2/shuhao/venv/bin/python
O=$GS/discovery/outputs/rec_v2_llm
CK=$CP/outputs/mappers
mkdir -p $CK
for ds in bigfive cfcs darktriad dass gcbs hexaco himi hs hsq kims mach npas riasec rse scs sd3 sixteenpf tlvd tma wpi wvs; do
  CKPT=$CK/mapper_lodo_${ds}.pt
  if [ ! -s "$CKPT" ]; then
    echo "=== train $ds $(date +%H:%M:%S)"
    MODE=train EVAL_DS=$ds EPOCHS=4 OUT=$CKPT $PV $CP/prefix/prefix_mapper.py > $CK/log_train_${ds}.txt 2>&1
    echo "    rc=$?"
  fi
  for src in given disc; do
    OUTF=$O/pfx_t12_${ds}_${src}.json
    [ -s "$OUTF" ] && continue
    echo "=== eval $ds $src $(date +%H:%M:%S)"
    MODE=eval EVAL_DS=$ds SRC=$src CKPT=$CKPT OUT=$OUTF $PV $CP/prefix/prefix_mapper.py > $CK/log_eval_${ds}_${src}.txt 2>&1
    echo "    rc=$?"
  done
done
echo PREFIX_LODO_DONE
