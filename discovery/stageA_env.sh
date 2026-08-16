# ONE env block for BOTH graph conditions: A1 and A2 differ only in the graph source.
# Frozen v6 knobs + K=100 (user decision 2026-08-14; supersedes the K=60 certified setting).
export L2_ARM=mlp GENOP=1 RESIDUAL=1.0 LAM_RES=1.0 BRIDGE=pearson NLDEP=1 POLFIX=0
export RCHAN=hard CI_MODE=marginal_shrink K=100 FORCE_DECODE=1
export OPENAI_API_KEY= JUDGE_MODEL=
export CUDA_VISIBLE_DEVICES=1 TORCH_THREADS=8
