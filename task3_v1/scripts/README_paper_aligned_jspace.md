# Paper-aligned Qwen J-space replication

This directory contains a small substrate check called **a paper-aligned
J-space replication on Qwen3.5-4B**. It is not an exact reproduction of the
Anthropic paper's Claude experiments and is separate from the main Task 3
Stage 1 results, thresholds, features, and ridge-dual writer.

## Data and provenance

The primary data is Anthropic's official `probe-swap.json`, pinned to
`anthropics/jacobian-lens` commit
`581d398613e5602a5af361e1c34d3a92ea82ba8e` and distributed under
Apache-2.0. The unmodified source, upstream README, LICENSE, hashes, and source
manifest are in `task3_v1/data/prompts/official_anthropic/`. Processed Qwen
metadata is written separately to `task3_v1/data/prompts/paper_aligned_qwen/`.

Filtering uses only tokenizer compatibility and clean model behavior. Primary
examples require clean top-1 correctness and single-token Qwen representations
for both intermediate concepts and both scored answers. Clean top-5-but-not-
top-1 examples are retained as a secondary diagnostic and never enter the
primary swap-success denominator. Categories are kept intact across the
calibration/held-out split.

## Layer and intervention protocol

The calibration scan covers every fitted Qwen J-lens residual layer (native
layers 0–30). Depth is reported as `100 * layer / 30`. A contiguous six-layer
band is selected using calibration-set mean reciprocal rank for the known
unspoken intermediate, with the earliest band breaking exact ties. The band is
then frozen before any held-out swaps are evaluated.

The strict swap is separate from Task 3's ridge-dual write:

```text
V = [v_source, v_target]
c = pinv(V) h
h_patched = h + V (swap(c) - c)
```

It is applied at every prompt token position and every layer in the frozen
band. The component orthogonal to the pairwise span is preserved. The primary
behavioral result uses the full band; single-layer swaps and source-to-source
no-op swaps are implementation diagnostics only.

## Commands

Run from the repository root with the existing J-lens environment:

```powershell
& '..\.venv-jlens\Scripts\python.exe' task3_v1\scripts\prepare_paper_aligned_prompts.py
& '..\.venv-jlens\Scripts\python.exe' task3_v1\scripts\run_paper_aligned_jspace.py
```

CPU-only structural smoke test:

```powershell
..\.venv-jlens\Scripts\python.exe task3_v1\scripts\run_paper_aligned_jspace.py `
  --synthetic-smoke `
  --output-dir task3_v1\outputs\paper_aligned\smoke
```

Before launching a full model run, use `--max-calibration 2 --max-heldout 2`
for a small end-to-end sample. Runtime logs go to
`task3_v1/logs/paper_aligned/`; machine-readable results go to
`task3_v1/outputs/paper_aligned/`.
