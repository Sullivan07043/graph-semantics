# causal-learn: the causal discovery home

One place for every discovery method in this project. Layout rule: `upstream/` is third-party
and read-only, everything else here is ours.

| path | what | status |
|---|---|---|
| `upstream/causal-learn/` | causal-learn @ dd13787 + RLCD (pinned, never edit) | read-only |
| `upstream/causal-get/` | C BOSS "recboss" @ 8b8508a (pinned, never edit) | read-only, `bash build.sh` builds `site/` |
| `gpu_rlcd/` | GPU-RLCD: batched float64 rank test + recboss stage 1 | active, certified |
| `gin_gpu/` | GIN with batched GPU HSIC | dormant |
| `polychoric.py` | polychoric correlation + rank test (point estimate certified) | paused (Wilks null over-rejects) |

Import discipline: the certified venv also carries a pip causal-learn (0.1.4.7, older). Every
script must pin `upstream/causal-learn` on `sys.path` and never rely on the bare import.

Experiment scripts (dataset loading, sweeps, downstream evaluation) do not live here. They live
in `discovery/` and `task3_robotics/` and import the methods from this directory.
