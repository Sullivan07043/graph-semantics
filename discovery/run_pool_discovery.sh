#!/bin/bash
# GPU-RLCD discovery over the certified pool (19 sets). Free (no judge). Sequential.
set -u
cd "$(dirname "$0")"
PV=/data2/shuhao/venv/bin/python
export CUDA_VISIBLE_DEVICES=1
for ds in bigfive himi tlvd cfcs darktriad gcbs hexaco hs hsq kims mach npas riasec rse scs sd3 sixteenpf tma wpi; do
  echo "=== $ds $(date +%H:%M:%S)"
  DATASET=$ds STAGE1_DISCOUNT=auto timeout 1200 $PV run_latent_discovery.py > outputs/pooldisc_${ds}.log 2>&1
  echo "    rc=$?"
done
echo POOL_DISCOVERY_DONE
