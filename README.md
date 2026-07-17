# graph-semantics

Complete the semantics of unlabeled variables on a **given causal graph**.

Setting (agreed 2026-07-02): a causal graph is GIVEN (latent and observed nodes, typed edges), together with
labels (short texts) for a SUBSET of the observed variables. The tasks:

- **Task 1** — complete the semantics of the unlabeled OBSERVED variables.
- **Task 2** — Task 1 + translate the LATENT variables.
- (Task 3, later — extend to LLM-scale settings such as agent communication; not in this repo yet.)

**Core problem: optimize latent (and unlabeled-observed) semantic EMBEDDINGS under the causal constraints
derived from the given graph.** Decoding embeddings to natural language uses an existing method (SpLiCE onto
a fixed dictionary) and is treated as downstream, not the research core.

## Current main line (adopted 2026-07-16)

```
labels --> e5-large-v2 + L3 LoRA  --> joint embedding optimization --> SpLiCE decode --> judge
           (pipeline_L3_v1)           L2 WeightNet-weighted unrolled     (L3-re-encoded
                                      solver over the graph constraints   521k dictionary)
                                      (pipeline_v4)
```

Three trained components sit on the structured-optimization core, each adopted with an
identity-at-init discipline (zero-initialized so training starts EXACTLY at the previous method)
and mechanism controls:

1. **L1 — `f_neg` semantic negation operator** (`negop.py`). A reverse-keyed item means the
   semantic opposite of its factor, which is NOT the negated vector (−u has cos −.6 to true
   reverse labels). Trained on WordNet antonym pairs + dev factor pole pairs.
2. **L2 — WeightNet learned solver** (`pipeline_v4/`). A 12-feature node-context MLP outputs
   per-node multipliers for each constraint term; the solve is unrolled K=60 differentiable
   functional-Adam steps and the weights are trained end-to-end on masked-label recovery
   (16 dev sets, folds 0-3 train / fold 4 validate). Controls: mult=1 (same dynamics, no
   learning) and 5-scalar static weights.
3. **L3 — LoRA-calibrated encoder space** (`pipeline_L3_v1/`). LoRA (r=8, zero-init B) in the
   q/v projections of e5's last 2 layers, trained so the space itself satisfies the bridge
   axiom: strong-dependence pairs keep high |cos| (upper tail), d-separated pairs decorrelate,
   reverse items align with `f_neg`(factor), while a 20k-word anchor loss pins general
   semantics (drift check: 2k held-back words cos ≥ .985; full 521k dictionary re-encode shift
   max .0199). The decode dictionary is re-encoded through the SAME LoRA (version-asserted).

Constraint set (all read off the given graph + data): signed generation equations (with `f_neg`
on negative edges), residual alignment to data partial correlations, independence decorrelation,
Pearson similarity lower bound on strongly dependent pairs, unit norm.

## Results snapshot (mask-20%, 5 folds, judge = gpt-5.5; 13 datasets)

Task 1 judge / match, means:

| | frozen space + 400-step solver | + WeightNet (L2) | + LoRA space (L3) — **main** |
|---|---|---|---|
| dev (10) | .643 / .793 | .696 / .784 | **.722** / .783 |
| held-out (3) | .680 / .569 | .697 / .619 | .698 / **.642** |
| all (13) | .652 / .741 | .696 / .746 | **.717** / **.751** |

Task 2 (latent translation, judge): .916 → .909 → **.930**. Fold-aligned LLM-naming baseline: .695.

Swap intervention (exchange two latents' embeddings → masked children must switch families):
geometric .789, judged .741 for the structured optimization vs .391 / .206 for a trained GNN —
the latents are causally load-bearing, not decorative. (`experiments/intervene*.py`)

## Pending fixes (known, prioritized)

1. **riasec regression (.431 judge under the main line, was .516).** Root cause: Holland's six
   types form a circumplex, not a hierarchy; the bridge/hierarchy constraints mis-shape it.
   Needs a circumplex-aware constraint (or exempting circumplex graphs from the upper tail).
2. **K=60 unroll budget binds on the deepest graphs** (hexaco/tlvd judge below the frozen-space
   column). Retrain WeightNet with larger K via truncated backpropagation; verification is
   API-free (match/embedding metrics). Evidence: mult=1 control loses .05 overall vs 400 steps.
3. **himi regression under L3** (.817 → .717 judge): not yet diagnosed; suspect the anchor set
   under-covers cognitive-task vocabulary (same family as the tlvd .500 ceiling).
4. **mach/rse single-factor scales**: graph constraints have nothing to use; known limitation,
   not a bug.
5. **week6\_report**: L3 section not yet added (L2 section is in).

Process rules in force: no pre-set pass/fail thresholds (adoption is the user's call on the full
comparison table); mechanism controls required for every adoption; held-out label texts never
enter any training; API spend only on final candidates (judge verdicts disk-cached in
`outputs/judge_cache.jsonl`); ≤15 concurrent eval processes (each loads ~4GB: encoder + dictionary).

## Evaluation protocol

- **Dev pool** (all fitting happens here): tlvd, himi, bigfive, hs, rse, mach, gcbs, sixteenpf,
  hsq, sd3 (+ cfcs, npas, scs, tma, darktriad, wpi for training breadth).
- **Held-out** (never used for any design or training decision): hexaco, riasec, kims.
- Within each dataset: 5-fold masking over observed labels — every variable is masked exactly
  once; the data matrix X stays visible. Both tasks share the folds.
- Metrics: judge-ACC (primary; gpt-5.5 dominant-meaning prompt; API failures recorded as missing,
  never wrong), matching-ACC (Hungarian, LLM-free), exact top-1. Dev iteration uses the free
  geometric metrics; the judge is spent on final candidates only.
- Baselines: uniform, raw correlation (no graph), fold-aligned LLM-naming (Task 2; single-agent
  version of TLVD's naming stage, sees only the fold's visible children).

## Testbeds (13 evaluated + 6 training-pool; real data; graphs = published keying / released files)

| dataset | role | observed | latents | graph |
|---|---|---|---|---|
| TLVD Multitasking | dev | 9 | 4 | TLVD's released RLCD .dot (latent-latent edges) |
| Himi | dev | 17 | 6 | study-design bipartite |
| Big Five IPIP | dev | 50 | 5 | design bipartite |
| Holzinger-Swineford 1939 | dev | 24 | 5 | classic 5-factor battery |
| RSE | dev | 10 | 1 | single factor |
| MACH-IV | dev | 20 | 1 | single factor |
| GCBS | dev | 15 | 5 | Brotherton 2013 Table A1 keying |
| 16PF (IPIP analogs) | dev | 162 | 16 | design bipartite |
| HSQ | dev | 32 | 4 | humor styles (mod-4 keying) |
| SD3 | dev | 27 | 3 | dark triad |
| HEXACO (240 items) | **held-out** | 240 | 6+24 | **two-level** factor→facet→item |
| RIASEC | **held-out** | 48 | 6 | Holland types (circumplex — see pending fix 1) |
| KIMS | **held-out** | 39 | 4 | keying from the codebook's scoring code |

(+ cfcs, npas, scs, tma, darktriad, wpi as additional dev-training pool, loaded in `pool.py`.)
Data: openpsychometrics.org `_rawdata` zips under `$GRAPHSEM_DATA/pool/`; nothing under `data/`
is committed.

## Run

```
# main line (LoRA space + WeightNet solver); API-free unless OPENAI_API_KEY is set
TASK=1 DATASET=heldout L2_ARM=mlp python pipeline_L3_v1/run_eval_l3.py

# retrain the pieces
python pipeline_v4/l2_train.py            # ARM=mlp|static, K, EPOCHS
python pipeline_L3_v1/l3_train.py         # then: python pipeline_L3_v1/reencode_dict.py
python negop.py train

# frozen-space reference runs
python run_task1.py                       # DATASET=dev|heldout|all|<csv>
python run_task2.py
python pipeline_v4/run_eval.py            # L2_ARM=mult1|static|mlp on the frozen space

# interventions
python experiments/intervene.py           # geometric swap
python experiments/intervene_judge.py     # judged swap
```

Env knobs: `DATASET`, `FOLDS`, `NEGOP`, `BRIDGE`, `RESIDUAL`, `LAM_RES`, `GRAPHSEM_ENCODER`,
`GRAPHSEM_DICT`, `JUDGE_MODEL`, `JUDGE_CACHE`, `RECORDS_OUT`, `L2_ARM`, `K`, `TORCH_THREADS`
(pin it — unbounded OpenMP oversubscribes 10x), `CUDA_VISIBLE_DEVICES` (always set explicitly).

## Honest notes

- The judge and the matching metric measure different things (semantic correctness vs individual
  identity); we report both. raw-correlation wins match while losing judge: copying the most
  correlated neighbour lands on surface-similar wording that is frequently the wrong meaning.
- The solver is deterministic within a process; across processes, float summation order differs
  and 400 nonconvex steps amplify it in weakly-constrained directions (task metrics unaffected).
- LoRA space WITHOUT the WeightNet solver is a net regression (match .729 vs .741) — the space
  calibration pays off only jointly with the learned solver.
- On TLVD's graph the latent ground truth uses TLVD's own released construct descriptions; KIMS
  item 21 is keyed to Observe per the published instrument; GCBS item texts come from the
  published scale. Raw cosine is not comparable across encoders.
