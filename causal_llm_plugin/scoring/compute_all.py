"""One unified result set: every (backend, surface, source, arm) with the FULL
metric family - NRR@1, NRR@5, MRR, SDA (item surfaces), judge accept rate, n.

Judge is a metric column, not a separate table. Per-dataset scores average
unweighted over datasets, matching the published tables. Output:
outputs/scores/all_results_unified.json plus a flat print.
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict

_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PKG)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
import gen_llm_report as G  # noqa: E402
import score_records as SR  # noqa: E402

ARMS = G.ARMS
ACC = defaultdict(lambda: defaultdict(list))   # (table, src, arm) -> per-set dicts


def jmean(rows):
    jv = [r["judge"] for r in rows if r.get("judge") is not None]
    return sum(jv) / len(jv) if jv else None


def put(table, src, arm, entry, judge):
    if entry:
        e = dict(entry)
        if judge is not None:
            e["judge"] = round(judge, 4)
        ACC[(table, src)][arm].append(e)


def rows_of(path):
    d = json.load(open(path))
    return d["records"] if isinstance(d, dict) and "records" in d else d


def latent_table(table, pattern, rex, armset):
    for p in sorted(glob.glob(os.path.join(config.REC_LLM, pattern))):
        m = re.match(rex, os.path.basename(p))
        if not m:
            continue
        recs = rows_of(p)
        for arm in armset:
            rws = [r for r in recs if r["arm"] == arm]
            lats = [r for r in rws if "latent" in r]
            if lats:
                put(table, m.group(2), arm, SR.score_latents(lats), jmean(lats))


def item_table(table, pattern, rex, armset, base_of=None):
    for p in sorted(glob.glob(os.path.join(config.REC_LLM, pattern))):
        m = re.match(rex, os.path.basename(p))
        if not m:
            continue
        recs = rows_of(p)
        base = base_of(m) if base_of else m.group(1)
        for arm in armset:
            rws = [r for r in recs if r["arm"] == arm and "var" in r]
            if rws:
                put(table, m.group(2), arm, SR.score_items(rws, base), jmean(rws))


def main():
    QD = r"_(\w+?)_(given|disc)\.json"
    RB = r"_(\w+?)_(boss|truev3)\.json"

    # gpt-5.5 surfaces
    latent_table("t2", "llm_t2_*.json", "llm_t2" + QD, ARMS)
    item_table("t1", "llm_t1_*.json", "llm_t1" + QD, ARMS)
    item_table("t3", "llm_t3_*.json", "llm_t3" + RB, ARMS + ["llmfact"])
    item_table("t3", "llm_t3h_*.json", "llm_t3h" + RB, ["llmhead"])
    # stress (gpt-5.5, T2 label-poor)
    for p in sorted(glob.glob(os.path.join(config.REC_LLM, "stress_t2_*_[369]0.json"))):
        m = re.match(r"stress_t2_(\w+)_(given|disc)_(\d+)\.json", os.path.basename(p))
        recs = rows_of(p)
        for arm in ("llmfull", "llmgraph"):
            rws = [r for r in recs if r["arm"] == arm]
            if rws:
                put(f"stress{m.group(3)}", m.group(2), arm,
                    SR.score_latents(rws), jmean(rws))

    # merged protocol: core from rec_v2_joint, llm arms from llm_t12_*
    for p in sorted(glob.glob(os.path.join(config.REC_JOINT, "t12_*.json"))):
        m = re.match(r"t12_(\w+)_(given|disc)\.json", os.path.basename(p))
        recs = [r for r in rows_of(p) if r["arm"] == "core"]
        items = [r for r in recs if "var" in r]
        lats = [r for r in recs if "latent" in r]
        put("joint_items", m.group(2), "core", SR.score_items(items, m.group(1)),
            jmean(items))
        put("joint_latents", m.group(2), "core", SR.score_latents(lats), jmean(lats))
    for p in sorted(glob.glob(os.path.join(config.REC_LLM, "llm_t12_*.json"))):
        m = re.match(r"llm_t12_(\w+)_(given|disc)_(\w+)\.json", os.path.basename(p))
        recs = rows_of(p)
        arm = m.group(3)
        items = [r for r in recs if r["arm"] == arm and "var" in r]
        lats = [r for r in recs if r["arm"] == arm and "latent" in r]
        if items:
            put("joint_items", m.group(2), arm, SR.score_items(items, m.group(1)),
                jmean(items))
        if lats:
            put("joint_latents", m.group(2), arm, SR.score_latents(lats), jmean(lats))

    # open weight: Qwen discrete
    latent_table("qwen_t2", "qwen_t2_*.json", "qwen_t2" + QD, ARMS)
    item_table("qwen_t1", "qwen_t1_*.json", "qwen_t1" + QD, ARMS)
    item_table("qwen_t3", "qwen_t3_*.json", "qwen_t3" + RB, ARMS + ["llmfact"])
    item_table("qwen_t3", "qwen_t3h_*.json", "qwen_t3h" + RB, ["llmhead"])
    # open weight: prefix
    for p in sorted(glob.glob(os.path.join(config.REC_LLM, "pfx_t12_*.json"))):
        m = re.match(r"pfx_t12_(\w+)_(given|disc)\.json", os.path.basename(p))
        recs = rows_of(p)
        for arm in ("prefixonly", "prefixgraph"):
            items = [r for r in recs if r["arm"] == arm and "var" in r]
            lats = [r for r in recs if r["arm"] == arm and "latent" in r]
            if items:
                put("pfx_items", m.group(2), arm, SR.score_items(items, m.group(1)),
                    jmean(items))
            if lats:
                put("pfx_latents", m.group(2), arm, SR.score_latents(lats),
                    jmean(lats))
    item_table("pfx_t3", "pfx_t3_*.json", "pfx_t3" + RB,
               ["prefixonly", "prefixgraph"])

    # core rows for the split surfaces, from the campaign aggregate
    T1_DS = ["bigfive", "dass", "hexaco", "rse", "wpi"]
    for src in ("given", "disc"):
        for surf, pool in (("t2", list(G.BASE_AGG[src])), ("t1", T1_DS)):
            for ds in pool:
                e = G.BASE_AGG[src].get(ds, {}).get(surf, {}).get("arms", {}).get("core")
                if e:
                    ACC[(surf, src)]["core"].append(e)
    eR = G.BASE_AGG["robot"]
    for key, src in (("base_boss", "boss"), ("base_truev3", "truev3")):
        for t in eR:
            if key in eR[t]:
                ACC[("t3", src)]["core"].append(eR[t][key]["arms"]["core"])

    # pool: unweighted mean over datasets per metric
    out = {}
    for (table, src), arms in sorted(ACC.items()):
        for arm, entries in arms.items():
            row = {"n_sets": len(entries)}
            for k in ("nrr_top1", "nrr_top5", "mrr", "sda", "judge"):
                vals = [e[k] for e in entries if e.get(k) is not None]
                row[k] = round(sum(vals) / len(vals), 3) if vals else None
            out.setdefault(table, {}).setdefault(src, {})[arm] = row
    dst = os.path.join(config.SCORES, "all_results_unified.json")
    json.dump(out, open(dst, "w"), indent=1)
    for table in out:
        for src in out[table]:
            for arm, r in out[table][src].items():
                print(table, src, arm, r)
    print("[saved]", dst)


if __name__ == "__main__":
    main()
