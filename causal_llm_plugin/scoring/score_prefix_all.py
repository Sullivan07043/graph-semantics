"""Score every questionnaire prefix-LODO record file (pfx_t12_<ds>_<src>.json)
with the referee-space metrics and write a pooled summary.

Reuses score_records.main() per file in one process (embedding model and
sibling caches load once). Per-set detail jsons land in outputs/scores/;
the pooled means land in outputs/scores/prefix_lodo_summary.json.
"""
import glob
import json
import os
import re
import sys

_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PKG)
import config  # noqa: E402
import score_records  # noqa: E402

PAT = re.compile(r"pfx_t12_([a-z0-9]+)_(given|disc)\.json$")


def main():
    files = sorted(glob.glob(os.path.join(
        config.DISC, "outputs", "rec_v2_llm", "pfx_t12_*.json")))
    for p in files:
        m = PAT.search(p)
        if not m:
            continue
        os.environ["RECORDS"] = p
        os.environ["BASE"] = m.group(1)
        os.environ["OUT"] = os.path.join(
            config.SCORES, os.path.basename(p).replace(".json", "_scores.json"))
        os.environ.pop("ARMS", None)
        score_records.main()

    # Pool per arm x graph source: unweighted mean over datasets, as in the
    # main tables.
    pool = {}
    for p in sorted(glob.glob(os.path.join(config.SCORES, "pfx_t12_*_scores.json"))):
        m = PAT.search(p.replace("_scores.json", ".json"))
        d = json.load(open(p))
        for arm, e in d["arms"].items():
            slot = pool.setdefault(arm, {}).setdefault(m.group(2), {})
            slot.setdefault("sets", []).append(m.group(1))
            for part in ("items", "latents"):
                if e.get(part):
                    for k in ("nrr_top1", "nrr_top5", "mrr", "sda"):
                        if e[part].get(k) is not None:
                            slot.setdefault(part, {}).setdefault(k, []).append(e[part][k])
            if e.get("combined_nrr_top1") is not None:
                slot.setdefault("combined", []).append(e["combined_nrr_top1"])
    summary = {}
    for arm, srcs in pool.items():
        for src, slot in srcs.items():
            row = {"n_sets": len(slot["sets"])}
            for part in ("items", "latents"):
                if part in slot:
                    row[part] = {k: round(sum(v) / len(v), 4)
                                 for k, v in slot[part].items()}
            if "combined" in slot:
                row["combined_nrr_top1"] = round(
                    sum(slot["combined"]) / len(slot["combined"]), 4)
            summary.setdefault(arm, {})[src] = row
    outp = os.path.join(config.SCORES, "prefix_lodo_summary.json")
    json.dump(summary, open(outp, "w"), indent=1)
    print(json.dumps(summary, indent=1))
    print(f"[summary] {outp}")


if __name__ == "__main__":
    main()
