#!/bin/bash
# GNN causal-constraint ablation: retrain per arm on the dev pool, eval dev (geometric).
# Baseline "none" = frozen GNN v3 numbers (outputs/gnn_eval_dev.json from the 15:56 run).
cd /data2/shuhao/semantic_interpretation/graph_semantics
export GRAPHSEM_ENCODER=e5-large GNN_STEPS=8000
run_arm () {
  name=$1; shift
  echo "=== ARM $name ($*) $(date +%H:%M:%S)"
  env "$@" GNN_CKPT=outputs/gnn_c_${name}.pt /data2/shuhao/venv/bin/python gnn.py train \
    && env "$@" GNN_CKPT=outputs/gnn_c_${name}.pt /data2/shuhao/venv/bin/python gnn.py eval dev \
    && mv outputs/gnn_eval_dev.json outputs/gnn_c_${name}_dev.json
}
run_arm indep GNN_LAM_INDEP=0.3
run_arm res   GNN_LAM_RES=0.3
run_arm coll  GNN_LAM_COLL=0.3
run_arm dep   GNN_LAM_DEP=0.3
run_arm mb    GNN_LAM_MB=0.1
echo "=== ALL ARMS DONE $(date +%H:%M:%S)"
