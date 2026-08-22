"""Score any records file (items, latents, or the merged protocol) with the
referee-space metrics, per arm, and write a PER-SET detail json.

Items: NRR@1/@5/MRR vs the dataset's full item-label set + SDA vs published siblings.
Latents: NRR@1/@5/MRR vs the unique latent gt texts (deduplicated; degenerate sets
report None). Merged files score both parts and a size-weighted combined mean.

Env: RECORDS=<json> BASE=<loader> OUT=<json into outputs/scores/> ARMS (optional).
"""
import json
import os
import sys

import numpy as np

_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PKG)
import config  # noqa: E402

sys.path.insert(0, config.V6)
sys.path.insert(0, config.DISC)

import metrics2  # noqa: E402

os.environ.setdefault("RECORDS", "unused")
os.environ.setdefault("BASE", "unused")
import rescore_records  # noqa: E402

_SIB = {}


def sib_of(base):
    if base not in _SIB:
        sib, labels, obs = rescore_records.published_siblings(base)
        cand = [labels[o] for o in obs]
        _SIB[base] = (sib, obs, cand, metrics2.embed(cand))
    return _SIB[base]


def text_of(r):
    w = r.get("decoded_words")
    if not w:
        return None
    return ", ".join(w) if isinstance(w, (list, tuple)) else str(w)


def score_items(rows, base):
    sib, obs, cand, ce = sib_of(base)
    idx = {o: i for i, o in enumerate(obs)}
    rows = [r for r in rows if r["var"] in idx]
    if not rows:
        return None
    preds = [text_of(r) for r in rows]
    tidx = [idx[r["var"]] for r in rows]
    sidx = [[idx[s] for s in sib[r["var"]] if s in idx] for r in rows]
    t1, mrr, rk = metrics2.nrr(preds, tidx, cand, cand_emb=ce)
    sda, n_sda = metrics2.sda(preds, tidx, sidx, cand, cand_emb=ce)
    return {"n": len(rows), "n_candidates": len(cand),
            "nrr_top1": round(t1, 4),
            "nrr_top5": round(sum(1 for x in rk if x <= 5) / len(rk), 4),
            "mrr": round(mrr, 4),
            "sda": (round(sda, 4) if sda is not None else None), "sda_n": n_sda}


def score_latents(rows):
    cand, of, tof = [], {}, {}
    for r in rows:
        t = r["gt"]
        if t not in of:
            of[t] = len(cand)
            cand.append(t)
        tof[(r["fold"], r["latent"])] = of[t]
    if len(cand) < 2:
        return {"n": len(rows), "n_candidates": len(cand), "nrr_top1": None,
                "nrr_top5": None, "mrr": None, "degenerate": True}
    ce = metrics2.embed(cand)
    preds = [text_of(r) for r in rows]
    tidx = [tof[(r["fold"], r["latent"])] for r in rows]
    t1, mrr, rk = metrics2.nrr(preds, tidx, cand, cand_emb=ce)
    return {"n": len(rows), "n_candidates": len(cand),
            "nrr_top1": round(t1, 4),
            "nrr_top5": round(sum(1 for x in rk if x <= 5) / len(rk), 4),
            "mrr": round(mrr, 4)}


def main():
    RECORDS, BASE = os.environ["RECORDS"], os.environ["BASE"]
    OUT = os.environ.get("OUT", os.path.join(
        config.SCORES, os.path.basename(RECORDS).replace(".json", "_scores.json")))
    d = json.load(open(RECORDS))
    recs = d["records"] if isinstance(d, dict) and "records" in d else d
    arms = (os.environ["ARMS"].split(",") if os.environ.get("ARMS")
            else sorted({r["arm"] for r in recs}))
    out = {}
    for arm in arms:
        rows = [r for r in recs if r["arm"] == arm]
        items = [r for r in rows if "var" in r]
        lats = [r for r in rows if "latent" in r]
        e = {}
        if items:
            e["items"] = score_items(items, BASE)
        if lats:
            e["latents"] = score_latents(lats)
        if e.get("items") and e.get("latents") and e["latents"].get("nrr_top1") is not None:
            wi, wl = e["items"]["n"], e["latents"]["n"]
            e["combined_nrr_top1"] = round(
                (e["items"]["nrr_top1"] * wi + e["latents"]["nrr_top1"] * wl) / (wi + wl), 4)
        if e:
            out[arm] = e
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"records": os.path.basename(RECORDS), "base": BASE, "arms": out},
              open(OUT, "w"), indent=1)
    print(f"[scored {os.path.basename(RECORDS)}] arms={list(out)} -> {OUT}")


if __name__ == "__main__":
    main()
