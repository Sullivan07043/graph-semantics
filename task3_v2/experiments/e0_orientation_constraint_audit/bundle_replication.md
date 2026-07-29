# E0-double-prime Bundle Replication

Behavioral trend reproduced: **True**. `checkpoint_or_bundle_drift=False`.

The selected probes reuse the exact E0-prime LoRA, WeightNet, negation operator, objective coefficients, candidate dictionary, and decoder. No artifact was retrained. Judge is pending; no missing verdict was treated as zero.

| dataset | role | arm | items | Match | Week-6 Match | gold cos | centered cos | MRR | R@5 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bigfive | hierarchy | hierarchy_without_latent_constraints | 50 | 0.540 | 0.660 | 0.919 | 0.177 | 0.265 | 0.580 |
| bigfive | hierarchy | hierarchy_with_latent_constraints | 50 | 0.620 | 0.740 | 0.924 | 0.215 | 0.239 | 0.540 |
| himi | dev | core | 17 | 0.800 | 0.900 | 0.881 | 0.457 | 0.381 | 1.000 |
| himi | dev | raw_correlation | 17 | 0.767 | 0.767 | 0.914 | -0.085 | 0.125 | 0.117 |
| himi | dev | uniform | 17 | 0.383 | 0.567 | 0.908 | -0.506 | 0.079 | 0.000 |
| kims | heldout | core | 39 | 0.693 | 0.675 | 0.806 | 0.258 | 0.159 | 0.232 |
| kims | heldout | raw_correlation | 39 | 0.900 | 0.850 | 0.940 | 0.231 | 0.160 | 0.232 |
| kims | heldout | uniform | 39 | 0.154 | 0.154 | 0.934 | -0.324 | 0.082 | 0.054 |

Qualitative checks:

- himi `core > raw correlation > uniform`: **True**
- kims `raw correlation > core > uniform`: **True**
- BigFive2 latent constraints on minus off Match: **+0.0800**

The hierarchy implementation ID is `bigfive2`; the canonical dataset is `bigfive`. Its missing Pearson cache was generated deterministically from the same observed X inside the audit results directory, with zero diagonal and recorded SHA-256.

The original reported release artifacts are absent and the local bundle had been retrained before E0-prime; this is a provenance limitation. It becomes a behavioral drift confound here only if the selected qualitative orderings or hierarchy direction fail.

Command:

```powershell
.\.venv\Scripts\python.exe task3_v2\scripts\run_e0_bundle_replication.py
```
