"""FCI compromise run on 16PF (the >25-item track's first testbed).

Declared compromise parameters (recorded in the output artifact): row subsample 2000,
alpha 0.001, conditioning sets capped at depth 3. Rationale in week9_report Part III:
full FCI does not terminate on measurement-model data (within-factor pairs have no
observed separating set), the alpha matches the pipeline's own multiple-testing levels,
and the depth cap errs conservatively (keeps edges, never fabricates independence).
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "v6"))

import numpy as np                                       # noqa: E402
import pool                                              # noqa: E402
from causallearn.search.ConstraintBased.FCI import fci   # noqa: E402

ROWS = int(os.environ.get("FCI_ROWS", 2000))
ALPHA = float(os.environ.get("FCI_ALPHA", 0.001))
DEPTH = int(os.environ.get("FCI_DEPTH", 3))
OUT = os.environ.get("FCI_OUT", "sixteenpf_fci_pag.npz")

if __name__ == "__main__":
    ds = pool.LOADERS["sixteenpf"]()
    g, X = ds["graph"], ds["X"]
    obs = list(g.observed)
    rng = np.random.default_rng(0)
    Xs = np.asarray(X, float)[rng.choice(X.shape[0], ROWS, replace=False)]
    print(f"fci compromise config: rows={ROWS} alpha={ALPHA} depth={DEPTH} "
          f"({len(obs)} nodes)", flush=True)
    t0 = time.time()
    gg, edges = fci(Xs, alpha=ALPHA, depth=DEPTH, node_names=obs,
                    show_progress=False, verbose=False)
    dt = time.time() - t0
    print(f"fci done {dt:.1f}s, {len(edges)} edges", flush=True)
    names = [n.get_name() for n in gg.get_nodes()]
    np.savez(os.path.join(HERE, "outputs", OUT),
             A=gg.graph, names=np.array(names), seconds=dt,
             rows=ROWS, alpha=ALPHA, depth=DEPTH)
    print(f"saved outputs/{OUT}", flush=True)
