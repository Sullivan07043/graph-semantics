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

## Current main line — v4.1 (formalized 2026-07-21)

The release contract is [release_v4_1.json](release_v4_1.json); implementation formulas,
mechanism tests, API-free results, and unresolved limits are recorded in
[fix_todo.md](fix_todo.md) and summarized in [RELEASE_v4.1.md](RELEASE_v4.1.md).

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
2. **L2 — WeightNet learned solver** (`pipeline_v4/`). A 22-feature node-context MLP outputs
   per-node multipliers for six constraint terms; inference uses K=120 differentiable
   functional-Adam steps, while training uses 2×60 truncated BPTT on masked-label recovery
   (16 dev sets, folds 0-3 train / fold 4 validate). Controls: mult=1 (same dynamics, no
   learning) and six-scalar static weights.
3. **L3 — LoRA-calibrated encoder space** (`pipeline_L3_v1/`). LoRA (r=8, zero-init B) in the
   q/v projections of e5's last 2 layers, trained so the space itself satisfies the bridge
   axiom: strong-dependence pairs keep high |cos| (upper tail), d-separated pairs decorrelate,
   reverse items align with `f_neg`(factor), while a 20k-word anchor loss pins general
   semantics. Dev-label identity/relative-geometry losses and a checkpoint geometry guard prevent
   Himi's same-factor/cross-factor gap from collapsing. The decode dictionary is re-encoded
   through the SAME LoRA and is SHA-bound to that checkpoint.

Constraint set (all read off the given graph + data): signed generation equations (with `f_neg`
on negative edges), residual alignment to data partial correlations, independence decorrelation,
Pearson similarity lower bound on strongly dependent pairs, masked-item semantic identity, and
unit norm. Graph-independent pairs are retained as zero constraints only when data do not
significantly contradict them; no dataset-name special cases or new causal edges are introduced.

## Metrics — what each number means

- **judge-ACC (primary).** The masked variable's solved embedding is decoded into ~6 dictionary
  words; gpt-5.5 judges whether the words' DOMINANT meaning correctly describes the true
  variable (synonyms count; a few spurious words don't disqualify). Measures semantic
  correctness. API failures are recorded as missing, never as wrong.
- **match-ACC.** LLM-free identity test. Within a fold, m variables are masked and the method
  produces m predicted embeddings. Build the m×m cosine matrix between predictions and the m
  TRUE label embeddings, solve the optimal one-to-one assignment (Hungarian, maximizing total
  similarity); match-ACC = fraction of variables assigned to their OWN label. Chance ≈ 1/m.
  Measures whether each prediction can be told apart as ITS specific item — not whether its
  meaning is right, which is why raw-correlation scores high match (it copies a nearby
  neighbour's position) while losing judge (the copied meaning is often wrong).
- **exact.** Stricter variant: the prediction's nearest neighbour among ALL the dataset's label
  embeddings must be exactly itself (reported in records; near zero for all methods on large
  scales).
- **true-target cosine.** API-free cosine between a masked prediction and its own held-out target
  embedding. It is an evaluation metric only; target label text never enters solver input.

## Results (mask-20%, 5 folds, judge = gpt-5.5; every cell judge / match)

This table is the historical pre-v4.1 judge snapshot and is intentionally unchanged. The final
v4.1 API-free 13-dataset results are reported in `fix_todo.md` and the release manifest.

Task 1 — complete masked observed variables. ⭐ = held-out. Columns left to right: no-graph
baselines, then the method evolution (frozen solver → +L2 WeightNet → +L3 LoRA space = main).

| dataset | uniform | rawcorr | frozen+400 | +WeightNet | **LoRA+WeightNet (main)** |
|---|---|---|---|---|---|
| tlvd | .500 / .600 | .500 / 1.00 | .600 / 1.00 | .500 / 1.00 | .500 / 1.00 |
| himi | .483 / .567 | .550 / .767 | .717 / .800 | .817 / 1.00 | .717 / .900 |
| bigfive | .060 / .140 | .720 / .780 | .720 / .620 | .780 / .640 | **.900** / .720 |
| hs | .120 / .160 | .160 / .720 | .660 / .840 | .660 / .740 | **.760** / .660 |
| rse | 1.00 / .800 | 1.00 / 1.00 | .700 / 1.00 | .800 / 1.00 | **1.00** / .800 |
| mach | .550 / .250 | .500 / .600 | .450 / .600 | .450 / .450 | .350 / .650 |
| gcbs | .867 / .400 | .733 / .867 | .733 / 1.00 | .733 / .867 | .733 / 1.00 |
| 16PF | .192 / .019 | .673 / .648 | .622 / .592 | .679 / .617 | .672 / **.678** |
| hsq | .848 / .133 | .681 / .943 | .638 / .629 | **.843** / .752 | .810 / .724 |
| sd3 | .447 / .200 | .627 / .920 | .593 / .853 | .700 / .773 | **.780** / .700 |
| hexaco ⭐ | .196 / .029 | .733 / .700 | .779 / .375 | .725 / .525 | .762 / .496 |
| riasec ⭐ | .144 / .062 | .458 / 1.00 | .516 / .756 | .493 / .707 | .431 / .756 |
| kims ⭐ | .354 / .154 | .593 / .850 | .746 / .575 | .871 / .625 | **.900** / .675 |
| dev (10) | .507 / .327 | .614 / .824 | .643 / .793 | .696 / .784 | **.722** / .783 |
| held-out (3) | .231 / .082 | .595 / .850 | .680 / .569 | .697 / .619 | .698 / **.642** |
| all (13) | .443 / .270 | .610 / .830 | .652 / .741 | .696 / .746 | **.717** / **.751** |

(uniform judge is inflated on narrow single-domain scales — rse/hsq/gcbs — where the mean of all
visible labels already sounds topical; its near-zero match exposes it.)

Task 2 — translate latent variables (judge-ACC; LLM-naming is fold-aligned: it names each latent
from the fold's VISIBLE children only, max 6):

| dataset | LLM-naming | frozen+400 | +WeightNet | **LoRA+WeightNet (main)** |
|---|---|---|---|---|
| himi | .733 | .833 | .867 | **.867** |
| bigfive | .800 | 1.00 | 1.00 | 1.00 |
| gcbs | .280 | 1.00 | 1.00 | 1.00 |
| sd3 | .667 | 1.00 | .933 | 1.00 |
| hexaco ⭐ | .887 | .927 | .913 | **.947** |
| riasec ⭐ | 1.00 | 1.00 | 1.00 | 1.00 |
| kims ⭐ | .500 | .650 | .650 | **.700** |
| mean | .695 | .916 | .909 | **.930** |

Swap intervention (exchange two latents' embeddings → masked children's recovered meanings must
switch families): geometric .789, judged .741 for the structured optimization vs .391 / .206 for
a trained GNN — the latents are causally load-bearing, not decorative. (`experiments/intervene*.py`)

## v4.1 TODO status and remaining limits

1. **RIASEC:** data-gated independence removes 1081 graph/data-conflicting zero constraints
   (1215→134 retained). This fixes forced cross-type orthogonality, but the solver still has no
   explicit generic circumplex representation; API-free match 0.811 remains below rawcorr 1.000.
2. **K=60:** resolved by K=120 inference and 2×60 truncated BPTT training; chunked and unchunked
   K=120 forward values are tested identical.
3. **Himi L3 identity:** resolved for the adopted checkpoint by dev-only identity/geometry
   preservation; frozen/L3 same-vs-cross gap is 0.116241→0.116350 and match is 0.900.
4. **MACH/RSE single-factor identity:** reliable signed local data signals are now used without
   masked text. Completely exchangeable items remain information-theoretically unidentifiable.
5. **week6\_report:** intentionally outside v4.1 scope and not created.

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
- Metrics: defined in the "Metrics" section above. Dev iteration uses the free geometric
  metrics; the judge is spent on final candidates only.
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
| RIASEC | **held-out** | 48 | 6 | Holland types (circumplex — see v4.1 limits) |
| KIMS | **held-out** | 39 | 4 | keying from the codebook's scoring code |

(+ cfcs, npas, scs, tma, darktriad, wpi as additional dev-training pool, loaded in `pool.py`.)
Data: openpsychometrics.org `_rawdata` zips under `$GRAPHSEM_DATA/pool/`; nothing under `data/`
is committed.

## Run

```
# main line (LoRA space + WeightNet solver); API-free unless OPENAI_API_KEY is set
TASK=1 DATASET=heldout L2_ARM=mlp python pipeline_L3_v1/run_eval_l3.py

# retrain the pieces in the required order
python pipeline_L3_v1/l3_train.py
python pipeline_L3_v1/reencode_dict.py
python pipeline_v4/l2_train.py            # ARM=mlp|static; main line fixes K=120
python negop.py train

# frozen-space reference runs
python run_task1.py                       # DATASET=dev|heldout|all|<csv>
python run_task2.py
python pipeline_v4/run_eval.py            # frozen reference; learned arms require explicit L2_CKPT

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
- The solver is deterministic within a process; across processes, float summation order can
  differ and 120 nonconvex steps can amplify it in weakly-constrained directions.
- LoRA space WITHOUT the WeightNet solver is a net regression (match .729 vs .741) — the space
  calibration pays off only jointly with the learned solver.
- On TLVD's graph the latent ground truth uses TLVD's own released construct descriptions; KIMS
  item 21 is keyed to Observe per the published instrument; GCBS item texts come from the
  published scale. Raw cosine is not comparable across encoders.
