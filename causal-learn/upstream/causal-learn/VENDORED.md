# Vendored causal-learn

Upstream: https://github.com/py-why/causal-learn, MIT licence (see `LICENSE`).
Pinned at commit `dd1378775f24f8cc6c2c4a2bb90b4423953930f1` (2026-05-24), version 0.1.4.7.

Vendored so the discovery track pins one version and so `gpu/`, which is ours, lives under version
control. `tests/`, `docs/` and the upstream `.git` are excluded: 69 MB of test fixtures and 26 MB of
history that we do not use.

`causallearn/` is upstream code, unmodified. RLCD and GIN are called as published, and the
`polychoric` rank test is passed in through RLCD's own `ranktest_method` argument rather than by
patching anything here.

## `gpu/` — ours, not upstream

- `batched_hsic.py` — HSIC as batched GEMMs on the GPU. Certified to 1e-14 against
  `hsic_test_gamma`, about 300x faster.
- `gin_gpu.py` — GIN driver with the indexing bug below corrected, two interchangeable backends
  (reference loops the official test, gpu uses the batched one) so certification isolates the swap.
- `certify.py` — equivalence checks. Run before trusting a pilot.
- `run_pilot.py` — dataset runner.

## Upstream bug found here

`causallearn/search/HiddenCausal/GIN/GIN.py` lines 67-68 and 98-99 use the loop counter as a column
index:

```python
for z in range(len(remain_var_set)):
    pvals.append(indep_test(data[:, [z]], e[:, None], method=indep_test_method))
```

`z` runs over positions, so the GIN residual is tested against the first `len(remain_var_set)`
columns of `data`, which include the candidate cluster's own columns, instead of against the
remaining variables. The result depends on column order, which the GIN condition forbids. On a
two-factor synthetic set, permuting the two blocks changes the returned clusters from none to one.
`GIN_MI` in the same file does not use positional indexing. Reported upstream; `gpu/gin_gpu.py`
carries the correction as a declared deviation.
