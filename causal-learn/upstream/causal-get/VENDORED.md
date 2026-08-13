# Vendored: causal-get (C BOSS, "recboss")

- Upstream: https://github.com/bja43/causal-get (Bryan Andrews)
- Vendored commit: `8b8508a` (2025-07-23, "removing compiled binary")
- Why this commit: upstream HEAD is in a broken interim state (tiered-knowledge
  refactor pushed half-done; it does not compile and the fixed build spins
  forever on a 20-variable input). `8b8508a` is the last commit before that
  refactor and matches the era of the prebuilt wheels J. Ramsey circulated by
  email (2026-08-08, forwarded by Yujia).
- Local changes: two `printf` debug lines in `causalget/boss.h` (marked
  "added for debugging" upstream) are commented out. Nothing else.

## Build

```bash
bash build.sh          # builds with /data2/shuhao/venv's Python (3.12), installs to site/
```

`site/` is untracked (platform-specific binary). `discover.py` adds `site/` to
`sys.path`, the certified venv itself stays untouched.

## API

```python
import causalget as cg
dag = cg.boss(R, n=n, discount=2, seed=1)   # R = correlation matrix (ndarray)
# dag[i, j] = 1 means j -> i
```

`discount=1` equals standard BIC (= causal-learn `local_score_BIC` default
lambda 0.5). Knowledge/tiers are not supported at this commit; time and tier
constraints are applied as a post-filter in `task3_robotics/discover.py`.

Measured on this box: p=20 instant, p=1000/n=1000 in 7 s,
Panda lag-1 matrix (223500 x 57) in 5 s.
