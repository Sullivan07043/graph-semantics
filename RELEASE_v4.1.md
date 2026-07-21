# graph-semantics v4.1

Version 4.1 formally designates the README-TODO repair candidate as the current main line:

```text
E5-large-v2 + L3 LoRA
    -> re-encode the concept bank with the final L3 encoder
    -> train L2 WeightNet in the final L3 embedding space
    -> run the K=120 structured solver
```

The repository release version is `4.1`; it is intentionally distinct from serialization
schema versions. The L3 LoRA checkpoint and L3 dictionary use schema 3, while the L2 WeightNet
checkpoint uses schema 4. The authoritative machine-readable contract is
[`release_v4_1.json`](release_v4_1.json). Full formulas, tests, diagnostics, and limitations are
recorded in [`fix_todo.md`](fix_todo.md).

## Release artifacts

| Role | Path | SHA-256 |
|---|---|---|
| L3 LoRA | `outputs/l3_lora_rel.pt` | `7f7c1b9c96b8fbfa467854327324601fd50ac50b74c166f4fcaf00fb55bdf232` |
| L3 concept bank | `outputs/concept_bank_l3_rel.npz` | `87c58e49f93d77874e9da14d77f88f8560b43ad923cf6d9cafa96e26f4850603` |
| L2 WeightNet | `outputs/l2_mlp_v4_1.pt` | `d346442b16b6bfebb3eee18f95156fb227184fafc33dc8381e96f9881ff93f87` |
| L2 training log | `outputs/l2_mlp_v4_1_trainlog.json` | `b3d88cc52472fe69a76d7ec5f44dc5b06b6ee2a2ea30e3ee5a3c0564112aa8be` |
| Targeted Task 1 evaluation | `outputs/v4_1_targeted_task1.json` | `3f5e6e4cc83f75ccfb4e54a0973c653115633d8c2f010a66b8363bcae8bb9c5c` |
| All-13 Task 1 evaluation | `outputs/v4_1_task1_all13_api_free.json` | `6418c13b330ba536fca3284c819f998167c8107c5d2b1957a76c4fa82d0f13d8` |
| All-13 Task 2 API-free records | `outputs/v4_1_task2_all13_api_free.json` | `e0b1f26476ce5da7fc86949531aa1f1b9c6deadcd126a281c9ffa794de2df596` |

The L2 checkpoint, L2 log, and Task 1 evaluation records are byte-for-byte release-named copies
of the validated `_todo4` candidate; the original files remain intact. The Task 2 file is the
subsequent final v4.1 API-free run. The 2.23 GB concept bank was not copied or rewritten, avoiding
unnecessary storage and preserving its existing L3 SHA binding.

The main evaluator validates the manifest, the L3 and L2 file hashes, dictionary metadata, and
the fact that both the dictionary and L2 checkpoint depend on the same L3 checkpoint. Set
`VERIFY_RELEASE_SHA256=1` to hash the complete dictionary during evaluation. The same checks can
be run independently with:

```powershell
.\.venv\Scripts\python.exe -B pipeline_v4\verify_release.py --full-dictionary-sha
```

## Validation summary

- All 22 mechanism and release-contract tests passed. They cover masked-label non-leakage,
  graph/data-gated independence, L3 identity preservation, K=120 chunked-forward equivalence,
  checkpoint compatibility, and the release contract.
- The all-13 Task 1 API-free result is mean match `0.808126`, mean exact `0.007507`, and mean
  true-target cosine `0.907113`, across 2,079 records. No judge call was made.
- The all-13 Task 2 API-free run completed all 13 datasets, 90 unique dataset-latent pairs, and
  five folds, producing the expected 450 core records with no empty decoded-word lists. All judge
  fields are null. The current Task 2 protocol defines only judge-ACC, so this run validates the
  complete solve/decode path but does not provide an API-free accuracy number.
- RIASEC independent pairs changed from `1215` raw to `134` retained, with `1081` graph/data
  conflicts. The Himi frozen/L3 cosine gap changed from `0.116241` to `0.116350`. The 16PF
  match/exact/cosine result is `0.720833/0.012121/0.907536`.
- WeightNet's best DEV outer loss is `0.875472`, compared with `0.880056` for unit multipliers.
  The near-bound fraction is zero for all six multiplier types.
- A default-path RSE smoke run completed with match `1.000`, exact `0.000`, cosine `0.925377`,
  and no judge call.

## Running v4.1

```powershell
$env:OPENAI_API_KEY = ""
$env:TASK = "1"
$env:L2_ARM = "mlp"
$env:K = "120"
$env:DATASET = "tlvd,himi,bigfive,hs,rse,mach,gcbs,sixteenpf,hsq,sd3,hexaco,riasec,kims"
$env:RECORDS_OUT = "outputs/v4_1_task1_all13_api_free.json"
.\.venv\Scripts\python.exe -B pipeline_L3_v1\run_eval_l3.py
```

For the corresponding all-13 Task 2 API-free solve/decode regression, keep the same environment
and change:

```powershell
$env:TASK = "2"
$env:RECORDS_OUT = "outputs/v4_1_task2_all13_api_free.json"
.\.venv\Scripts\python.exe -B pipeline_L3_v1\run_eval_l3.py
```

Retraining must run in this order:

1. `pipeline_L3_v1/l3_train.py`
2. `pipeline_L3_v1/reencode_dict.py`
3. `pipeline_v4/l2_train.py`

CUDA is selected automatically when available.

## Known limitations

- RIASEC's incorrect orthogonality constraints are removed, but a generic circumplex is not
  explicitly represented by the structured core.
- MACH and RSE remain subject to an exchangeability limit when the data contain no stable signal
  that distinguishes individual items.
- The Himi guard prevents regression, but the selected LoRA update is small.
- Task 2 completed its final API-free solve/decode run, but its current protocol has no API-free
  latent accuracy metric. Quantitative judge-ACC remains unavailable when the judge is disabled.
- Some multi-factor datasets exhibit real match trade-offs. No judge evaluation and no K=60
  versus K=120 ablation were run.
- `outputs/` is gitignored. Source code plus the manifest alone does not include the binary
  artifacts listed above.
- The source release is committed on `xuran_v4` (initial release commit `83879b4`) and pushed to
  `origin/xuran_v4`. No Git tag was created.
