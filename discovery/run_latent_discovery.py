"""Experiment layer for GPU-RLCD: load a project dataset, hand the matrix to the method.

The method package (causal-learn/gpu_rlcd) is dataset-agnostic. This script owns the dataset
side: it pulls X and variable names from the v6 loaders, writes a matrix npz, and calls the
package runner.

Env: DATASET=dass|wvs|... (a v6 pool_ext loader)  plus any run.py env (STAGE1, MAXK, OUT).
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "v6"))

import numpy as np

import pool  # noqa: E402
import pool_ext  # noqa: E402
import testbeds  # noqa: E402

LOADERS = {**testbeds.LOADERS, **pool.LOADERS, **pool_ext.LOADERS}
DATASET = os.environ.get("DATASET", "dass")


SAFE = "x:"     # RLCD names its latents L1, L2, ...; 16PF items are literally named L1..L10.
                # Prefix observed names on the way in, strip on the way out.


def main():
    ds = LOADERS[DATASET]()
    X = np.asarray(ds["X"], float)
    names = list(ds["graph"].observed)
    npz = os.path.join(HERE, "outputs", f"{DATASET}_matrix.npz")
    os.makedirs(os.path.dirname(npz), exist_ok=True)
    np.savez(npz, X=X, names=np.array([SAFE + n for n in names]))
    env = dict(os.environ)
    env["NPZ"] = npz
    out = env.setdefault("OUT", os.path.join(HERE, "outputs", f"{DATASET}_rlcd_gpu.json"))
    runner = os.path.join(ROOT, "causal-learn", "gpu_rlcd", "run.py")
    rc = subprocess.call([sys.executable, runner], env=env)
    if rc != 0:
        sys.exit(rc)
    # convert to the run_downstream graph format (rlcd_directed / rlcd_undirected)
    import json
    d = json.load(open(out))
    strip = lambda n: n[len(SAFE):] if n.startswith(SAFE) else n
    d["edges"] = [[strip(a), strip(b)] for a, b in d["edges"]]
    # break cycles among latents (RLCD covers can emit mutual latent edges; the v6 graph
    # traversals assume an acyclic latent hierarchy): keep an L->L edge only if the reverse
    # is not already reachable through kept L->L edges
    is_lat = lambda n: n.startswith("L") and n[1:].isdigit()
    kept_ll, adj = [], {}

    def reaches(a, b, seen=None):
        seen = seen or set()
        if a == b:
            return True
        seen.add(a)
        return any(reaches(c, b, seen) for c in adj.get(a, ()) if c not in seen)

    clean = []
    for a, b in d["edges"]:
        if is_lat(a) and is_lat(b):
            if reaches(b, a):
                continue
            adj.setdefault(a, []).append(b)
        clean.append([a, b])
    if len(clean) < len(d["edges"]):
        print(f"[{DATASET}] dropped {len(d['edges']) - len(clean)} cycle-closing latent edges",
              flush=True)
    d["edges"] = clean
    down = os.path.join(HERE, "outputs", f"{DATASET}_gpurlcd.json")
    json.dump({"rlcd_directed": d["edges"], "rlcd_undirected": [],
               "params": {"source": os.path.basename(out), "stage1": d["stage1"],
                          "maxk": d["maxk"], "seconds": d["seconds"]}},
              open(down, "w"), indent=1)
    print(f"[{DATASET}] downstream graph -> {down}", flush=True)


if __name__ == "__main__":
    main()
