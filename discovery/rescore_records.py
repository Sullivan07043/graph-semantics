"""Rescore saved run records with the referee-space metrics (NRR, SDA) plus stored match.

Input: a records json written by run_task1 (list under 'records' or a bare list) whose entries
carry {dataset, arm, var, true_label, decoded_words, judge}. The sibling reference for SDA is
the PUBLISHED structure (questionnaire construct key, or the channel family on robots), which
is an evaluation reference only and never an input to any method.

Env: RECORDS=<json> BASE=<loader name> OUT=<json>
"""
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "v6"))

import numpy as np

import metrics2  # noqa: E402

RECORDS = os.environ["RECORDS"]
BASE = os.environ["BASE"]
OUT = os.environ.get("OUT", RECORDS.replace(".json", "_rescored.json"))


def published_siblings(base):
    """var -> sibling var set, from the published structure of the BASE loader."""
    import pool
    import pool_ext
    import testbeds
    loaders = {**testbeds.LOADERS, **pool.LOADERS, **pool_ext.LOADERS}
    if base not in loaders:                           # robot loaders live in the task3 copy
        import importlib.util
        p3 = os.path.join(ROOT, "task3_robotics", "task3_pipeline_v1")
        sys.path.insert(0, p3)
        spec = importlib.util.spec_from_file_location("pool_ext_t3",
                                                      os.path.join(p3, "pool_ext.py"))
        pe3 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pe3)
        loaders.update(pe3.LOADERS)
    ds = loaders[base]()
    g, labels = ds["graph"], ds["labels"]
    fam = defaultdict(set)
    if g.latents:                                     # questionnaire: same published construct
        for L in g.latents:
            kids = [c for c in g.children(L) if not g.is_latent(c)]
            for c in kids:
                fam[c] |= set(kids)
    else:                                             # robot: same channel family (name stem)
        for o in g.observed:
            stem = o.rsplit(".", 1)[0]
            fam[stem].add(o)
        fam = {o: fam[o.rsplit(".", 1)[0]] for o in g.observed}
    return {o: sorted(fam.get(o, {o}) - {o}) for o in g.observed}, labels, list(g.observed)


def main():
    d = json.load(open(RECORDS))
    recs = d["records"] if isinstance(d, dict) and "records" in d else d
    recs = [r for r in recs if r.get("dataset", "").startswith(BASE)]
    if not recs:
        raise SystemExit(f"no records for BASE={BASE} in {RECORDS}")

    sib, labels, obs = published_siblings(BASE)
    cand = [labels[o] for o in obs]
    idx_of = {o: i for i, o in enumerate(obs)}
    cand_emb = metrics2.embed(cand)

    out = {}
    for arm in sorted({r["arm"] for r in recs}):
        rows = [r for r in recs if r["arm"] == arm and r["var"] in idx_of]
        preds = [r.get("decoded_words") for r in rows]
        tidx = [idx_of[r["var"]] for r in rows]
        sidx = [[idx_of[s] for s in sib[r["var"]] if s in idx_of] for r in rows]
        top1, mrr, _ = metrics2.nrr(preds, tidx, cand, cand_emb=cand_emb)
        sda, n_sda = metrics2.sda(preds, tidx, sidx, cand, cand_emb=cand_emb)
        judged = [r["judge"] for r in rows if r.get("judge") is not None]
        chance = 1.0 / len(cand)
        out[arm] = {
            "n": len(rows),
            "nrr_top1": round(top1, 4), "nrr_top1_norm": round(metrics2.chance_norm(top1, chance), 4),
            "mrr": round(mrr, 4),
            "sda": (round(sda, 4) if sda is not None else None), "sda_n": n_sda,
            "judge": (round(float(np.mean(judged)), 4) if judged else None),
        }
    json.dump({"base": BASE, "records": os.path.basename(RECORDS),
               "n_candidates": len(cand), "chance_top1": round(1.0 / len(cand), 4),
               "sda_chance_note": "SDA chance depends on family size; report raw",
               "arms": out}, open(OUT, "w"), indent=1)
    print(f"[{BASE}] {len(recs)} records, {len(cand)} candidates -> {OUT}")
    for a, m in out.items():
        print(f"  {a:10s} nrr@1={m['nrr_top1']:.3f} mrr={m['mrr']:.3f} "
              f"sda={m['sda'] if m['sda'] is not None else '--'} judge={m['judge']}")


if __name__ == "__main__":
    main()
