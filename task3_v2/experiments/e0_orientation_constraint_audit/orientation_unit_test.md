# Task 3 E0″ — Orientation Unit Test

## Verdict

The E0′ source/target orientation is correct end to end. The audit found no
deterministic JSON, adjacency, adapter, `Graph`, or solver orientation bug.
Therefore, there is no orientation repair to apply and E0′ must **not** be
rerun under the `orientation_interface_bug` branch.

The diagnostic finding is instead that the frozen generation equation is a
directed equation represented by a bidirectional quadratic compatibility
factor during joint embedding completion. A free node receives gradients both
as a generated child and as the parent of another generated node.

## Source trace

1. `task3_v2/scripts/e0_core.py:333-355` assigns JSON `source` to `parent`
   and JSON `target` to `child`.
2. `task3_v2/scripts/e0_core.py:366-375` requires every parent to precede its
   child in the declared topological order.
3. `task3_v2/scripts/e0_core.py:608-628` generates SCM values from incoming
   edges:

   `x_child = sum(coefficient[source, child] * x_source) + noise`.

4. `task3_v2/scripts/e0_core.py:660-682` defines adjacency as
   `A[row=source, column=target]`.
5. `task3_v2/scripts/run_e0_bridge.py:471-472` converts JSON directly to
   ordered `(source, target)` tuples.
6. `task3_v2/scripts/run_e0_bridge.py:666-687` passes those tuples to the
   frozen `Graph`; the solver does not receive an adjacency matrix, so there is
   no hidden matrix-transpose boundary in this adapter.
7. `v5/graph.py:18-34` records `_pa[target].append(source)` and
   `_ch[source].append(target)`.
8. `v5/l2_modules.py:16-40` independently reads incoming and outgoing weights
   as `(parent, node)` and `(node, child)`.
9. `v5/optimize.py:55-82` constructs ALS equations
   `z_node - sum(W[parent,node] * z_parent) = 0`.
10. `v5/l2_solver.py:25-39` builds generated nodes and parent maps from
    `Graph.parents`.
11. `v5/l2_solver.py:106-119` evaluates each generation term using incoming
    parents.
12. `v5/l2_solver.py:213-234` differentiates that complete loss with respect
    to every free embedding, including its appearances in downstream
    equations.

## Minimal chain

The executable audit uses:

```text
A --0.7--> B --0.4--> C
```

and verifies:

```text
parents(A) = {}
parents(B) = {A}
parents(C) = {B}

children(A) = {B}
children(B) = {C}
children(C) = {}
```

The source-by-target weighted adjacency is:

```text
[[0.0, 0.7, 0.0],
 [0.0, 0.0, 0.4],
 [0.0, 0.0, 0.0]]
```

A deliberately transposed `A <- B <- C` fixture is included as a negative
control. The orientation guard must raise `OrientationAuditError`; silent
acceptance is a test failure.

## Exact generation gradient

With `z_A=(1,0)`, `z_C=(0,1)`, and free `z_B=(0,0)`, the exact frozen
generation loss is:

```text
L_gen(z_B) = ||z_B - 0.7 z_A||² + ||z_C - 0.4 z_B||²
```

Consequently:

```text
incoming contribution to dL/dz_B = (-1.4,  0.0)
outgoing contribution to dL/dz_B = ( 0.0, -0.8)
total dL/dz_B                    = (-1.4, -0.8)
gradient-descent direction       = ( 1.4,  0.8)
loss at z_B=0                    = 1.49000001
```

Thus `z_B` is pulled toward **both** its causal parent `z_A` and its causal
child `z_C`. This is the expected derivative of the implemented objective,
not an edge-parsing defect.

The frozen ALS initialization also matches its ridge-aware analytic solution:

```text
z_B = (0.7 z_A + 0.4 z_C) / (1 + 0.4² + 1e-6)
    = (0.6034477556484864, 0.3448272889419923)
```

At unit weights, the forward and completely reversed chain have identical
generation loss and `z_B` gradient. This formally demonstrates why generation
alone has weak orientation identifiability even though every interface
preserves the declared direction.

## Executable artifacts

- Audit: `task3_v2/scripts/e0_orientation_audit.py`
- Tests: `task3_v2/tests/test_e0_orientation.py`

Commands:

```powershell
.\.venv\Scripts\python.exe task3_v2\scripts\e0_orientation_audit.py
.\.venv\Scripts\python.exe -m unittest discover -s task3_v2\tests -p test_e0_orientation.py -v
```

The audit emits structured JSON and exits nonzero if any orientation
assertion fails. No E0′ labels, graphs, folds, seeds, checkpoints, losses, or
results are modified.
