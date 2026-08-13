# GPU-RLCD

RLCD latent discovery with the rank-test sweep batched on the GPU and the stage-1 partition
computed by the C BOSS. Input: a data matrix. Output: a latent graph json. The upstream RLCD
implementation is never modified; `rlcd_gpu.py` patches its sweep functions for one call and
restores them.

## Files

| file | what |
|---|---|
| `gpu_ranktest.py` | `GpuRankTest`: Wilks chi-square decisions on batched float64 whitened-SVD canonical correlations. Duck-types `Chi2RankTest`, adds `prime()` batch API. |
| `rlcd_gpu.py` | `RLCD_gpu(X, ...)` entry, `RLCD_serial` deterministic reference, recboss stage-1 injection |
| `certify.py` | the gate: decision parity + end-to-end graph identity vs the CPU reference |
| `run.py` | runner: `NPZ=` / `CSV=` / `SYNTH=` in, graph json out |

## Use

```bash
PV=/data2/shuhao/venv/bin/python
CUDA_VISIBLE_DEVICES=1 $PV certify.py                 # must print CERTIFY PASS first
CUDA_VISIBLE_DEVICES=1 NPZ=data.npz STAGE1=recboss $PV run.py
```

Datasets enter through the experiment layer (`discovery/run_latent_discovery.py`), never
through this package.

## Certified numbers (2026-08-12)

| scale | data | end to end | quality |
|---|---|---|---|
| p=15 | certification synthetic | seconds | graph identical to CPU reference |
| p=42 | DASS (real) | 164 s | scale purity .88, coverage 38/42 |
| p=150 | synthetic, 25 latents | 107 s | purity .78, coverage 145/150 |

Decision parity 800/800 at two seeds. The GES stage 1 it replaces did not finish p=42 in
35 minutes.

## Limits

- float64 is required (float32 scoring has a ~1e-3 decision noise floor, measured on recboss).
- The GPU removes per-test cost only. Subset counts still grow as $\binom{p}{k+1}$: exact
  sweeps are practical to about 150 active variables at k<=3. Larger inputs rely on the
  stage-1 partition keeping each group small.
- The upstream multiprocessing path is not reproducible run to run (per-worker hash seeds).
  This package's serial sweep is deterministic.
- Wilks over-rejects on skewed Likert data. The planned fix is a GPU permutation null.
