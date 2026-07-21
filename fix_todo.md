# graph-semantics v4.1: README TODO Repairs and Validation Record

Repairs completed: 2026-07-19. Formalized as v4.1: 2026-07-21.

## Executive summary

The existing main-line architecture was preserved:

~~~text
E5-large-v2 + L3 LoRA
    -> re-encode the concept bank with the final L3 encoder
    -> train L2 WeightNet in the final L3 embedding space
    -> run the WeightNet-weighted structured solver
~~~

The final L3 encoder and dictionary retain the _rel artifacts that passed the identity guard.
Only one final L2 schema-v4 candidate was trained. No combination ablation was performed, no
held-out label text was used for tuning, no judge or OpenAI API was called, and the README
historical result table was not modified.

The version numbers have different meanings and must not be conflated:

| Versioned object | Version |
|---|---:|
| Repository release | v4.1 |
| L3 LoRA checkpoint schema | 3 |
| L3 dictionary schema | 3 |
| L2 WeightNet checkpoint schema | 4 |
| Solver budget | K=120 |
| Training truncation interval | 60 steps |

The machine-readable release contract is release_v4_1.json.

| README TODO or targeted issue | Mechanism status | Final assessment |
|---|---|---|
| Generation loss increased with graph width | Fixed | Squared error is formed per generated node and then averaged across nodes. |
| K=60 was too short | Fixed | Training and inference use K=120; training uses 2 x 60 truncated BPTT. |
| Masked items lacked an identity signal | Main mechanism fixed | Local leakage-free targets and balanced polarity views raised all-13 mean match from 0.779 to 0.808. |
| RIASEC cross-type nodes were pushed toward orthogonality | Incorrect constraint fixed | Independent pairs changed from 1215 to 134; core match rose from 0.711 to 0.811. |
| Himi L3 identity regressed | Fixed for the adopted checkpoint | Frozen/L3 gap changed from 0.116241 to 0.116350; Himi match is 0.900. |
| WeightNet multipliers appeared boundary-pinned | Not present in the final run | Near-bound fractions are zero for all six multiplier types. |
| Learned WeightNet needed to beat unit multipliers | Improved | Best DEV outer loss is 0.875472 versus 0.880056 for unit multipliers. |
| MACH/RSE single-factor graphs lacked identity | Recoverable component fixed | MACH improved from 0.450 to 0.600 and RSE from 0.600 to 1.000; exact exchangeability remains unidentifiable. |

## Repository and artifact protection

The untracked data/ and experiment_logs/ directories existed before the repair and were
preserved. No reset, clean, deletion, or unintended staging of user files was performed. Existing
output artifacts were not overwritten. The scoped v4.1 source changes were later committed and
pushed on the dedicated xuran_v4 branch.

Formal v4.1 artifacts:

- outputs/l3_lora_rel.pt
- outputs/l3_rel_trainlog.json
- outputs/concept_bank_l3_rel.npz
- outputs/l2_mlp_v4_1.pt
- outputs/l2_mlp_v4_1_trainlog.json
- outputs/v4_1_targeted_task1.json
- outputs/v4_1_task1_all13_api_free.json
- outputs/v4_1_task2_all13_api_free.json

The release-named L2 checkpoint, L2 log, and Task 1 result files are byte-for-byte copies of the
final _todo4 artifacts. Their originals remain intact. The Task 2 result is the later final v4.1
API-free execution. The 2.23 GB dictionary was neither copied nor rewritten; the manifest
designates the existing SHA-bound file as the v4.1 dictionary. Earlier _rel L2 and evaluation
files remain available as the pre-v4.1 baseline.

~~~text
L3 checkpoint SHA-256
7f7c1b9c96b8fbfa467854327324601fd50ac50b74c166f4fcaf00fb55bdf232

L3 dictionary SHA-256
87c58e49f93d77874e9da14d77f88f8560b43ad923cf6d9cafa96e26f4850603

L2 checkpoint SHA-256
d346442b16b6bfebb3eee18f95156fb227184fafc33dc8381e96f9881ff93f87

Task 2 all-13 API-free records SHA-256
e0b1f26476ce5da7fc86949531aa1f1b9c6deadcd126a281c9ffa794de2df596
~~~

The dictionary and L2 checkpoint both record and validate the same L3 SHA.

## v4.1 formalization

- Root VERSION contains 4.1; pipeline_v4/release.py is the code-level release contract.
- release_v4_1.json freezes architecture, schemas, training order, artifact paths, sizes,
  hashes, API-free metrics, targeted diagnostics, and remaining limitations.
- outputs/l2_mlp_v4_1.pt is byte-identical to the validated candidate. State keys, tensor values,
  size, and SHA are unchanged. The log and evaluation records are also byte-identical copies.
- pipeline_L3_v1/run_eval_l3.py uses v4.1 paths by default and validates the manifest, L3/L2
  hashes, dictionary metadata, and both L3 dependency links.
- pipeline_v4/verify_release.py --full-dictionary-sha passed, including the complete dictionary
  hash and exact L2 tensor comparison.
- A default-path API-free RSE smoke run passed without artifact overrides. It produced 30 records
  with core match/exact/cosine 1.000000/0.000000/0.925377.
- The final all-13 Task 2 API-free run completed 13 datasets, 90 unique dataset-latent pairs, five
  folds, and all 450 expected core records. No decoded-word list was empty and no judge was called.
- No L3 or dictionary archive was rewritten, so the validated cryptographic binding is intact.
- The source was committed on xuran_v4 (initial release commit
  83879b4714004bdb05dd2eaa6ce63f326791f590) and pushed to origin/xuran_v4. No Git tag was created.

## 1. Normalized structured objective

### Original problem

The old objective summed generation error over generated nodes, while residual norm,
partial-correlation, independence, bridge, and norm terms were mostly means over applicable nodes
or pairs. Wider graphs therefore increased generation loss relative to every other term.

### Final repair and full formula

Let \(\mathcal G=\{i:\operatorname{pa}(i)\ne\varnothing\}\). A negative edge uses semantic
negation when available. Otherwise it retains the original signed linear edge; simple vector
negation is not substituted:

\[
T_{pi}(e_p)=
\begin{cases}
f_{\rm neg}(e_p), & w_{pi}<0\text{ and }f_{\rm neg}\text{ is available},\\
e_p, & \text{otherwise}.
\end{cases}
\]

\[
\widehat e_i=
\sum_{p\in\operatorname{pa}(i)}
\widetilde w_{pi}T_{pi}(e_p)+r_i,
\qquad
\widetilde w_{pi}=
\begin{cases}
|w_{pi}|,&w_{pi}<0\text{ and }f_{\rm neg}\text{ is used},\\
w_{pi},&\text{otherwise}.
\end{cases}
\]

WeightNet multipliers are applied before the node mean:

\[
L_{\rm gen}=
\frac1{|\mathcal G|}
\sum_{i\in\mathcal G}
m_i^{\rm gen}\|e_i-\widehat e_i\|_2^2.
\]

\[
L_{\rm resnorm}=
\mu\frac1{|\mathcal G|}
\sum_{i\in\mathcal G}m_i^{\rm res}\|r_i\|_2^2.
\]

The original PC1 residual contained part-whole leakage because the latent PC1 included the two
items being residualized. The final residual channel reconstructs a leave-pair-out parent proxy
for each local shared-parent pair and obtains \(\rho_{ij}^{(-ij)}\). It retains only reliable
relations and the symmetric union of each node's eight strongest local neighbors:

\[
\tau={2\over\sqrt n},\qquad
q_{ij}=(|\rho_{ij}^{(-ij)}|-\tau)_+,\qquad
m_{ij}^{\rm anchor}={m_i^{\rm anchor}+m_j^{\rm anchor}\over2},
\]

\[
L_{\rm residual-pair}
=\lambda_{\rm res}
{\sum_{(i,j)\in\mathcal R}
q_{ij}m_{ij}^{\rm anchor}
\bigl(\cos(r_i,r_j)-\rho_{ij}^{(-ij)}\bigr)^2
\over
\sum_{(i,j)\in\mathcal R}q_{ij}}.
\]

For single-factor scales with more than 32 indicators, full PC1 loadings are computed once and
the two item contributions are removed from score values for each pair. This avoids repeated SVDs
on WPI while preserving value-level leave-two-out behavior.

Let \(\bar a\) be the visible-label mean and
\(\tilde e_i=\operatorname{normalize}(e_i-\bar a)\). Fixed-fixed independent pairs calibrate the
semantic zero point but do not dilute the mean over active pairs:

\[
\eta_0=
\operatorname{median}_{(i,j)\in\mathcal I^\star_{\rm fixed}}
\cos(\tilde a_i,\tilde a_j),
\]

\[
L_{\rm zero}=
\lambda_0\operatorname{mean}_{(i,j)\in\mathcal I^\star_{\rm active}}
m_{ij}^{\rm node}
\bigl(\cos(\tilde e_i,\tilde e_j)-\eta_0\bigr)^2.
\]

Fixed-fixed bridge pairs similarly calibrate an achievable semantic floor:

\[
\beta_0=
\operatorname{median}_{(i,j)\in\mathcal B_{\rm fixed}}
|\cos(\tilde a_i,\tilde a_j)|,
\]

\[
L_{\rm bridge}=
\lambda_{\rm upper}
\operatorname{mean}_{(i,j)\in\mathcal B_{\rm active}}
m_{ij}^{\rm node}
\operatorname{relu}\bigl(\beta_0-|\cos(\tilde e_i,\tilde e_j)|\bigr)^2.
\]

The remaining dependence, collider, and norm terms are:

\[
L_{\rm dep}=
\lambda_{\rm dep}\operatorname{mean}_{(i,j)}
m_{ij}^{\rm node}
\operatorname{relu}
\bigl(\kappa|\rho_{ij}|-|\cos(\tilde e_i,\tilde e_j)|\bigr)^2,
\]

\[
L_{\rm coll}=
\lambda_{\rm coll}\operatorname{mean}_{(p_1,p_2,c)}
\operatorname{relu}
\left[
\cos\left(\Pi^\perp_{\tilde e_c}\tilde e_{p_1},
\Pi^\perp_{\tilde e_c}\tilde e_{p_2}\right)
\right]^2,
\]

\[
L_{\rm norm}=
\lambda_{\rm norm}\operatorname{mean}_{i\in\mathcal F}
m_i^{\rm norm}(\|e_i\|_2-1)^2.
\]

The complete objective is:

\[
L_{\rm structured}=
L_{\rm gen}+L_{\rm resnorm}+L_{\rm residual-pair}
+L_{\rm zero}+L_{\rm dep}+L_{\rm bridge}
+L_{\rm coll}+L_{\rm item}+L_{\rm profile}+L_{\rm norm}.
\]

solve_unrolled(..., weight_module=None) sets all multipliers to one and is covered by a test.

Key locations:

- optimize.py:23: observed marginal correlation.
- optimize.py:59: leave-pair-out residual pairs.
- pipeline_v4/core.py:186: shared context and zero/bridge calibration.
- pipeline_v4/core.py:370: normalized objective.
- pipeline_v4/core.py:567: K=120 functional-Adam solver.

## 2. Masked-item semantic identity

### Original problem

Parent generation recovers an item's broad factor but can confuse siblings. The first
partial_residual_corr target also inherited PC1 part-whole artifacts. The final design separates
whole-item marginal correlation \(C_{ij}\) from leave-pair-out residual relation
\(\rho_{ij}^{(-ij)}\). partial_residual_corr remains only for compatibility and diagnostics.

### Local leakage-free target

Candidates for masked observed node \(i\) are restricted to visible members of
g.mb_observed(i), including shared-parent siblings. Masked target label text or embedding never
enters the target, features, or solver:

\[
\mathcal V_i=
\{j\in\operatorname{MB}_{\rm obs}(i):
j\text{ is visible},\ |C_{ij}|>\tau\},
\qquad \tau={2\over\sqrt n}.
\]

\[
s(a_j,C_{ij})=
\begin{cases}
a_j,&C_{ij}\ge0,\\
f_{\rm neg}(a_j),&C_{ij}<0\text{ and }f_{\rm neg}\text{ is available},\\
\text{ignore},&C_{ij}<0\text{ and no }f_{\rm neg}\text{ is available}.
\end{cases}
\]

There is no \(-a_j\) branch. Reliable weights use only correlation above the noise floor:

\[
\omega_{ij}=(|C_{ij}|-\tau)_+,\qquad
\mathcal V_i^+=\{j\in\mathcal V_i:C_{ij}\ge0\},\qquad
\mathcal V_i^-=\{j\in\mathcal V_i:C_{ij}<0\}.
\]

\[
\begin{aligned}
u_i^+
&=\operatorname{normalize}
\left(\sum_{j\in\mathcal V_i^+}\omega_{ij}a_j\right),\\
u_i^-
&=\operatorname{normalize}
\left(\sum_{j\in\mathcal V_i^-}\omega_{ij}f_{\rm neg}(a_j)\right),\\
z_i
&=\operatorname{normalize}
\left(
\mathbf 1[\mathcal V_i^+\ne\varnothing]u_i^+
+\mathbf 1[\mathcal V_i^-\ne\varnothing]u_i^-
\right).
\end{aligned}
\]

Positive and negative candidates are aggregated and normalized separately, then combined as two
equally weighted semantic views. This prevents a larger positive-keyed group from overwhelming a
smaller reverse-keyed group by count. The negative view still requires neg_op.

\[
c_i=\max_{j\in\mathcal V_i}|C_{ij}|.
\]

The initially requested confidence multiplier was revised after mechanism diagnosis. Reliability
is already hard-gated by \(|C_{ij}|>\tau\), while \(\omega_{ij}\) encodes strength inside the
target. Multiplying by \(c_i\) again caused triple attenuation. The adopted objective therefore
uses unit base strength after reliability gating:

\[
L_{\rm item}=
\operatorname{mean}_{i:\mathcal V_i\ne\varnothing}
m_i^{\rm item}\bigl(1-\cos(e_i,z_i)\bigr).
\]

Confidence remains a WeightNet feature and diagnostic. No seventh constraint type was introduced.

The non-causal profile signal is enabled only when reliable visible relations exist outside the
graph-local neighborhood. If all reliable nodes are already local, as in a single-parent star,
the sibling set is not reused for a second near-duplicate profile. With reliable nonlocal evidence:

\[
L_{\rm profile}=
\operatorname{mean}_i
m_i^{\rm node}d_i
\operatorname{relu}
\left[
-\bigl(\cos(\tilde e_i,\tilde z_i^{\rm hi})
-\cos(\tilde e_i,\tilde z_i^{\rm lo})\bigr)
\right]^2.
\]

The profile adds no graph edge and uses no masked label.

### WeightNet

item is the sixth multiplier type. WeightNet has 22 features:

- 12 existing graph, node, and edge features.
- Seven item features: reliable-candidate count, maximum and mean \(|C_{ij}|\), visible-local
  count, confidence, normalized candidate entropy, and effective candidate count.
- Three independence features: retained-zero degree, conflict degree, and mean conflict strength.

The checkpoint format is graph-semantics-weightnet schema 4, with feature_dim=22 and six terms.
Old or mismatched checkpoints fail explicitly rather than loading silently.

Key locations:

- pipeline_v4/core.py:24: leakage-free target/profile construction.
- pipeline_v4/core.py:475: item and profile losses.
- pipeline_v4/l2_modules.py:13: 22-feature, six-term schema.
- pipeline_v4/l2_modules.py:144: strict checkpoint loading.

## 3. Graph/data reconciliation for independence

### Original problem

g.independent_pairs() used only empty-set d-separation. RIASEC is encoded as a bipartite graph,
but its measured structure is circumplex-like, so the old objective incorrectly pushed many
cross-type nodes toward orthogonality.

### Final repair

\[
x_i=
\begin{cases}
X_{:i},&i\text{ is observed},\\
\operatorname{PC1}_i,&i\text{ is latent and }estimate\_weights\text{ provides a score}.
\end{cases}
\]

For graph-independent set \(\mathcal I_G\), retain:

\[
\mathcal I^\star=
\left\{
(i,j)\in\mathcal I_G:
x_i\text{ or }x_j\text{ is unavailable}
\ \lor\
|\operatorname{corr}(x_i,x_j)|\le {2\over\sqrt n}
\right\}.
\]

A significant graph/data conflict removes only the zero constraint. It does not add an edge,
alter the DAG, hard-code RIASEC, or construct a directed cycle. graph.py:141 is the single shared
implementation used by L2 training, L2 inference, L3 training, and both task runners.

~~~text
RIASEC
n=5000
tau=0.028284
raw independent pairs=1215
retained=134
graph/data conflicts=1081
~~~

This fixes forced cross-type orthogonality. It does not learn a complete circumplex: final core
match is 0.811, below rawcorr 1.000. Removing a wrong zero constraint is not equivalent to adding
a correct circular generator.

## 4. L3 identity preservation

### Original problem

The old L3 objective retained bridge, independence, negation, and dictionary drift anchors but did
not directly preserve item identity or relative same-parent/cross-parent geometry. LoRA could
compress Himi's cosine gap.

### Final formulas

Before LoRA injection, frozen E5 embeddings \(h_i^0\) are computed for labels from all 16 DEV
datasets. Held-out label texts are not loaded. For every visible same-parent pair:

\[
b_{ij}=1-\cos(h_i^0,h_j^0),
\]

\[
a_{ij}=\cos(h_i,h_i^0)-\cos(h_i,h_j^0),
\]

\[
L_{\rm identity}=
\operatorname{mean}_{(i,j)}
\operatorname{relu}(b_{ij}-a_{ij})^2.
\]

Its fixed weight is one; no grid search was run. A graph-balanced safeguard protects geometry near
frozen initialization. Let \(c_{ij}=\cos(h_i,h_j)\),
\(c^0_{ij}=\cos(h_i^0,h_j^0)\), and \(s_{ij}=\max(1-c^0_{ij},0.05)\):

\[
L_{\rm same}=
\operatorname{mean}_{\rm parent\ groups}
\operatorname{mean}_{(i,j)}
\left({c_{ij}-c^0_{ij}\over s_{ij}}\right)^2,
\]

\[
L_{\rm cross}=
\operatorname{mean}_{\rm parent\text{-}pair\ groups}
\operatorname{mean}_{(i,j)}
\operatorname{relu}
\left({c_{ij}-c^0_{ij}\over s_{ij}}\right)^2,
\]

\[
L_{\rm relational}={1\over2}(L_{\rm same}+L_{\rm cross})
\quad\text{when both classes exist}.
\]

Same-parent cosine stays near frozen geometry. Cross-parent pairs are only prevented from becoming
more similar, so independence may still move them apart. Graph groups are averaged before
combination, and each step deterministically replays up to four groups of each type per DEV
dataset. Only DEV labels are used.

\[
L_{\rm indep}^{\rm L3}=
\operatorname{mean}_{(i,j)\in\mathcal I^\star}
\operatorname{relu}
\bigl(\cos(h_i,h_j)-\cos(h_i^0,h_j^0)\bigr)^2.
\]

The retained task loss is:

\[
L_{\rm task}=
10L_{\rm anchor}+L_{\rm bridge}
+0.3L_{\rm indep}+L_{\rm neg}.
\]

The dictionary anchor retains its internal factor of 100 and 0.99 cosine hinge. Identity,
relational, and replay terms form the safeguard. Gradient-balanced PCGrad scales the safeguard to
the task-gradient norm, projects away task conflict, and combines them. No extra weight search was
performed.

Checkpoint selection requires every multi-factor DEV gap to remain within \(10^{-4}\) of frozen.
Zero LoRA is always an available fallback; a failing epoch cannot overwrite the checkpoint.

~~~text
anchor=10, bridge=1, independence=0.3, negation=1,
identity=1, relational=1, replay=1
~~~

Key locations:

- pipeline_L3_v1/l3_train.py:110: identity hinge.
- pipeline_L3_v1/l3_train.py:121: relational safeguard.
- pipeline_L3_v1/l3_train.py:199: gradient-balanced PCGrad.
- pipeline_L3_v1/l3_train.py:239: retained task bundle.
- pipeline_L3_v1/l3_train.py:326: DEV-only training and geometry guard.
- pipeline_L3_v1/lora.py:17: strict L3 schema-v3 loader.

### L3 diagnostics

With a fixed seed and three epochs, epoch 0 was the final checkpoint satisfying every geometry
guard. Epochs 1 and 2 were rejected because at least one multi-factor DEV gap regressed.

| Metric | Frozen E5 | Selected L3 |
|---|---:|---:|
| Himi same-parent cosine | 0.896492 | 0.896362 |
| Himi cross-parent cosine | 0.780251 | 0.780012 |
| Himi gap | 0.116241 | 0.116350 |
| Gap change | — | +0.000110 |

~~~text
Selected epoch 0
train total=0.091665
validation total=0.083061
identity train=0.000000144
identity validation=0.000000273
held-back dictionary-anchor drift mean cosine=0.998102
held-back dictionary-anchor drift min cosine=0.997453

Final concept bank
shape=542705 x 1024
shift mean=0.00189960
shift p99=0.00230628
shift max=0.00276244
format/version=graph-semantics-l3-dictionary/3
~~~

## 5. Training order, K=120, and CUDA

Required order:

1. Train outputs/l3_lora_rel.pt.
2. Re-encode outputs/concept_bank_l3_rel.npz with that checkpoint.
3. Install the same final L3 encoder, encode DEV labels, and train
   outputs/l2_mlp_v4_1.pt.

pipeline_v4/l2_train.py installs L3 before computing any DEV label embedding. The dictionary, L2
checkpoint, and evaluator all verify the L3 SHA and stop clearly on mismatch.

~~~text
K=120
training truncation=60
inference truncation=None (all 120 steps execute)
device=cuda if torch.cuda.is_available() else cpu
~~~

At step 60, histories of embeddings, residuals, and functional-Adam m/v state are detached while
values are preserved. Steps 61–120 still use multipliers from the same WeightNet parameters.
Tests show bit-identical chunked/full K=120 forwards, different backward graphs, and nonzero
second-segment gradients. K=240 was neither tested nor adopted.

The DEV-only L2 outer objective is:

\[
L_{\rm own}=\operatorname{mean}_i(1-\cos(e_i,t_i)),
\]

\[
L_{\rm outer-id}=
\operatorname{mean}_{i\ne j}
\operatorname{relu}
\left(
{(1-\cos(t_i,t_j))-
[\cos(e_i,t_i)-\cos(e_i,t_j)]
\over \max(1-\cos(t_i,t_j),0.05)}
\right)^2,
\]

\[
L_{\rm outer}=L_{\rm own}+L_{\rm outer-id}+0.5L_{\rm latent}.
\]

## 6. L2 training and multiplier distributions

Training used a fixed seed, four epochs, folds 0–3 for training, and fold 4 for validation:

| Epoch | Training outer loss | Validation outer loss |
|---:|---:|---:|
| 0 | 0.892772 | 0.883691 |
| 1 | 0.901607 | 0.880386 |
| 2 | 0.900137 | 0.878927 |
| 3 | 0.892260 | **0.875472** |

Unit multipliers obtain 0.880056 with the same K=120 solver. The learned checkpoint is better by
0.004584. This is a real but small gain.

| Type | Count | Mean | Std | Min | Max |
|---|---:|---:|---:|---:|---:|
| gen | 605 | 0.787906 | 0.174039 | 0.576088 | 1.121809 |
| resnorm | 605 | 0.735094 | 0.112397 | 0.578348 | 1.020065 |
| anchor | 595 | 0.939167 | 0.053215 | 0.781258 | 1.002004 |
| node | 661 | 1.174037 | 0.129651 | 0.964594 | 1.638004 |
| norm | 175 | 1.001817 | 0.050755 | 0.923119 | 1.090799 |
| item | 115 | 1.416684 | 0.131929 | 1.172586 | 1.673284 |

The interval is \([\exp(-1.5),\exp(1.5)]\approx[0.2231,4.4817]\). Near-lower and near-upper
fractions are zero for all types. Generation and residual norm are generally below one, and item
is above one, but no type is boundary-saturated.

## 7. Tests

~~~powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
./.venv/Scripts/python.exe -B -m unittest discover -s tests -v
~~~

Result: **22/22 passed**.

Coverage includes generation normalization; no vector-negation fallback; polarity normalization;
profile de-duplication; unit item strength after gating; masked-label non-leakage;
weight_module=None; leave-pair-out residuals; graph/data conflict behavior; zero calibration;
identity initialization; relational geometry; PCGrad; K=120 chunking; legacy L2/L3 errors;
release/schema separation; schema-v4 round trip; manifest contract; and clear wrong-release errors.

Test files: tests/test_todo_fixes.py and tests/test_release_contract.py.

## 8. Targeted API-free diagnostics

Every run explicitly set OPENAI_API_KEY to an empty string; every judge field is null.

| Dataset | Independent raw→retained (conflicts) | Pre-v4.1 match | v4.1 match | v4.1 exact | v4.1 true-cosine |
|---|---:|---:|---:|---:|---:|
| RSE | 0→0 (0) | 0.600000 | 1.000000 | 0.000000 | 0.925377 |
| MACH | 0→0 (0) | 0.450000 | 0.600000 | 0.000000 | 0.881929 |
| Himi | 220→60 (160) | 0.900000 | 0.900000 | 0.000000 | 0.890545 |
| RIASEC | 1215→134 (1081) | 0.711111 | 0.811111 | 0.000000 | 0.877019 |
| 16PF | 14847→2389 (12458) | 0.740530 | 0.720833 | 0.012121 | 0.907536 |

- RSE and MACH gain 0.400 and 0.150 match. RSE reaches 1.000 and MACH reaches rawcorr's 0.600.
- Himi's gap does not regress, and match remains 0.900.
- RIASEC removes most contradicted zeros and rises from 0.711 to 0.811, but rawcorr is 1.000.
- 16PF match declines from 0.741 to 0.721 while cosine rises from 0.907 to 0.908. This is a real
  ranking trade-off, not a renewed collapse.

## 9. All 13 Task 1 datasets, API-free

| Dataset | Core match | Core exact | Core true-cosine |
|---|---:|---:|---:|
| tlvd | 1.000000 | 0.000000 | 0.949535 |
| himi | 0.900000 | 0.000000 | 0.890545 |
| bigfive | 0.660000 | 0.020000 | 0.910975 |
| hs | 0.840000 | 0.000000 | 0.909076 |
| rse | 1.000000 | 0.000000 | 0.925377 |
| mach | 0.600000 | 0.000000 | 0.881929 |
| gcbs | 1.000000 | 0.000000 | 0.890224 |
| sixteenpf | 0.720833 | 0.012121 | 0.907536 |
| hsq | 0.876190 | 0.028571 | 0.928708 |
| sd3 | 0.660000 | 0.000000 | 0.893887 |
| hexaco | 0.587500 | 0.008333 | 0.905299 |
| riasec | 0.811111 | 0.000000 | 0.877019 |
| kims | 0.850000 | 0.028571 | 0.922355 |
| DEV 10 mean | 0.825702 | 0.006069 | 0.908779 |
| Held-out 3 mean | 0.749537 | 0.012302 | 0.901558 |
| All-13 mean | 0.808126 | 0.007507 | 0.907113 |

Relative to pre-v4.1, all-13 mean match changes from 0.779165 to 0.808126 (+0.028961), and mean
true-target cosine changes from 0.906680 to 0.907113. Per-dataset changes are not uniformly
positive: Big Five -0.100, HS -0.080, HSQ -0.057, SD3 -0.100, HEXACO -0.050, 16PF -0.020;
RSE +0.400, MACH +0.150, GCBS +0.133, RIASEC +0.100; TLVD, Himi, and KIMS unchanged. No further
tuning was performed on these final results.

outputs/v4_1_task1_all13_api_free.json contains 2,079 arm/item records.

## 10. All 13 Task 2 datasets, API-free

The final Task 2 run used the same formal L3 checkpoint, L3 dictionary, schema-v4 WeightNet, and
K=120 solver as Task 1. It completed all 13 datasets and all five folds:

| Check | Result |
|---|---:|
| Datasets completed | 13 |
| Unique dataset-latent pairs | 90 |
| Folds | 5 |
| Expected core records | 450 |
| Written core records | 450 |
| Empty decoded-word lists | 0 |
| Non-null judge fields | 0 |

The output is outputs/v4_1_task2_all13_api_free.json (169,146 bytes; SHA-256
e0b1f26476ce5da7fc86949531aa1f1b9c6deadcd126a281c9ffa794de2df596).

This is a complete API-free solve/decode regression, not an accuracy claim. The current Task 2
runner defines latent accuracy only through judge-ACC. With the judge disabled, each dataset's
summary values are null; there is no Task 2 analogue of Task 1 match/exact/true-cosine yet. No
OpenAI API or other judge was invoked, and held-out latent descriptions were not used for tuning.

## 11. Single-factor theoretical limit

A single-factor bipartite graph gives every item the same parent. If the data distribution is
unchanged under exchange of two items, the graph, parent score, edge support, and visible labels
cannot distinguish the permutation. Masked-item identity is statistically unidentifiable.

v4.1 uses only stable information already present:

- Marginal-correlation fingerprints select different visible siblings.
- Leave-pair-out residuals represent local structure beyond the common factor.
- Polarity views are normalized separately so a smaller reverse-keyed group is not overwhelmed.

Under the official five-fold mask, \(n=5000\), and \(\tau=0.028284\):

| Dataset | Items with both reliable polarity views | Positive candidates mean/range | Negative candidates mean/range |
|---|---:|---:|---:|
| RSE | 10/10 | 3.6 / 3–4 | 4.4 / 4–5 |
| MACH | 20/20 | 7.6 / 7–9 | 8.4 / 7–9 |

The improvement is supported by visible data and does not hard-code dataset names. Candidate
statistics never read masked labels. Stronger ridge/Nyström fingerprint predictors were examined
but not adopted because they performed worse on DEV checks and would create a parallel identity
model. No new constraint type, network, or graph edge was added.

When correlations are within sampling noise, items are approximately equally related, or a fold
masks the only reliable neighbor, \(z_i\) lacks identifying information. High cosine can then mean
recovery of a common topic rather than the exact item. Increasing K, LoRA, or WeightNet capacity
cannot remove this information-theoretic limit.

## 12. Next methods for remaining issues

### RIASEC

A principled next step is a generic signed relation kernel from observed correlations, followed by
a low-rank spectral embedding that can retain two-dimensional circular structure. Its coordinates
should be used only as a non-causal residual/profile soft target, with rank selected by DEV-split
stability and without held-out label text.

A lighter alternative is rawcorr barycentric initialization for the masked residual, followed by
structured correction of reliable graph-constraint violations. RIASEC rawcorr match 1.000 shows
that the missing information already exists in the data kernel and does not require new directed
causal edges.

### WeightNet

The learned DEV advantage is only 0.004584 and cross-dataset trade-offs remain. Possible future
changes are zero-centered shrinkage on \(\log m\), dataset-equal outer-loss averaging, DEV-only
cross-fitting between unit and learned gates, and more fold averaging of meta-gradients. These
require retraining and were excluded from the single-final-candidate scope.

### Himi and L3

The guard fixes regression, but epoch 0 has very small drift. A future simple alternative is a
near-isometric low-rank rotation or a projection of the DEV Gram matrix after each step. This is
more direct than adding scalar identity weights and still requires no held-out labels.

## 13. Commands

~~~powershell
$env:OPENAI_API_KEY = ""
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

# 1. Train L3 LoRA.
./.venv/Scripts/python.exe -B pipeline_L3_v1/l3_train.py

# 2. Re-encode the concept bank with the final L3 checkpoint.
./.venv/Scripts/python.exe -B pipeline_L3_v1/reencode_dict.py

# 3. Train K=120, 2x60 WeightNet in the final L3 space.
$env:ARM = "mlp"
$env:K = "120"
./.venv/Scripts/python.exe -B pipeline_v4/l2_train.py

# Tests.
$env:PYTHONDONTWRITEBYTECODE = "1"
./.venv/Scripts/python.exe -B -m unittest discover -s tests -v

# Targeted API-free evaluation.
$env:TASK = "1"
$env:L2_ARM = "mlp"
$env:DATASET = "himi,riasec,sixteenpf,rse,mach"
$env:RECORDS_OUT = "outputs/v4_1_targeted_task1.json"
./.venv/Scripts/python.exe -B pipeline_L3_v1/run_eval_l3.py

# All 13 Task 1 datasets, API-free.
$env:DATASET = "tlvd,himi,bigfive,hs,rse,mach,gcbs,sixteenpf,hsq,sd3,hexaco,riasec,kims"
$env:RECORDS_OUT = "outputs/v4_1_task1_all13_api_free.json"
./.venv/Scripts/python.exe -B pipeline_L3_v1/run_eval_l3.py

# All 13 Task 2 datasets, API-free solve/decode regression.
$env:TASK = "2"
$env:RECORDS_OUT = "outputs/v4_1_task2_all13_api_free.json"
./.venv/Scripts/python.exe -B pipeline_L3_v1/run_eval_l3.py

# Full release verification.
./.venv/Scripts/python.exe -B pipeline_v4/verify_release.py --full-dictionary-sha
~~~

The evaluator enforces K=120 and verifies L3 format/schema, concept-bank format/schema/L3 SHA,
L2 format/schema/L3 SHA, and v4.1 artifact paths, sizes, and hashes.

## 14. Modified and added files

- graph.py: shared graph/data independence reconciliation.
- metrics.py: API-free true-target cosine.
- optimize.py: marginal relations and leave-pair-out residuals.
- pipeline_v4/core.py: normalized objective, item/profile targets, calibrated zero/bridge terms,
  and the K=120/2 x 60 solver.
- pipeline_v4/l2_modules.py: six multipliers, 22 features, and schema-v4 checkpoints.
- pipeline_v4/l2_train.py: final-L3 installation and multiplier diagnostics.
- pipeline_v4/release.py, VERSION, and release_v4_1.json: release contract and manifest.
- pipeline_v4/verify_release.py: artifact, byte-copy, tensor, and binding-chain checks.
- pipeline_v4/run_eval.py: solver inputs and explicit frozen-space checkpoint requirements.
- pipeline_L3_v1/l3_train.py: identity, replay, relative independence, PCGrad, and geometry guard.
- pipeline_L3_v1/lora.py: separate checkpoint/dictionary schemas and strict loading.
- pipeline_L3_v1/reencode_dict.py: dictionary encoding and SHA metadata.
- pipeline_L3_v1/run_eval_l3.py: release and L3/dictionary/L2 consistency checks.
- run_task1.py and run_task2.py: shared data-relation and gate inputs.
- tests/test_todo_fixes.py: 17 mechanism tests.
- tests/test_release_contract.py: five release-contract tests.
- README.md: current-main-line prose; historical result numbers unchanged.
- RELEASE_v4.1.md: release summary, artifacts, commands, and limitations.

## 15. Incomplete work and retained risks

- RIASEC's wrong independence constraints are fixed, but the core still lacks an explicit generic
  two-dimensional circumplex. Match 0.811 remains below rawcorr 1.000.
- WeightNet slightly improves DEV outer loss and no output is boundary-pinned, but Big Five, HS,
  HSQ, SD3, HEXACO, and 16PF have real match regressions.
- MACH/RSE identity remains unidentifiable without a stable data fingerprint.
- The Himi guard selects a very small epoch-0 update. It prevents regression but adds limited new
  semantic benefit.
- K=120 is implemented and tested, but no K=60/K=120 ablation was run, so metric changes cannot be
  attributed solely to unroll depth.
- Task 2 completed its final API-free solve/decode regression, but the runner exposes no API-free
  latent accuracy metric; only judge-ACC can score the decoded latent words today.
- L3 max lengths 64 and 128 are not fully unified across encoding paths.
- No judge or OpenAI API was called.
- No held-out labels were used for tuning. K=240, large ablations, and week6_report were explicit
  exclusions.
- outputs/ is gitignored, so source plus manifest does not distribute the binary artifacts.
- The source is committed and pushed on xuran_v4, but no Git tag was created. Because outputs/ is
  gitignored, the Task 2 JSON and other binary artifacts are not transported by the source branch.
