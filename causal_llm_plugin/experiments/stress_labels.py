"""Label-poor stress test on T2: mask a fraction of the children item labels in the
evidence and measure how llmgraph degrades and how much the phrases add back.

For each non-degenerate T2 pilot file, each ratio in (0.3, 0.6, 0.9), and each arm in
(llmgraph, llmfull): the masked child lines read `the item "(label hidden)"`. Masking
is deterministic per (dataset, source, ratio). Phrases come from the same pilot
records. Ratio 0.0 needs no calls: the pilot arms are the 0% points.

Output: outputs/rec_v2_llm/stress_t2_<ds>_<src>_<pct>.json (rec_v2 schema, arms
llmgraph/llmfull), shared cache jsonl. Env: OPENAI_API_KEY, JUDGE_BASE_URL.
"""
import glob
import json
import os
import re
import sys

import numpy as np

import os
import sys
_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PKG)
sys.path.insert(0, os.path.join(_PKG, "plugin"))
import config
sys.path.insert(0, config.V6)
sys.path.insert(0, config.DISC)

import evidence  # noqa: E402
import prompts  # noqa: E402

for _k, _v in (("SURFACE", "t2"), ("BASE", "stress"), ("SOURCE", "stress"),
               ("RECORDS", "unused"),
               ("OUT", os.path.join(config.REC_LLM, "stress_t2.json"))):
    os.environ.setdefault(_k, _v)
import run_plugin as rp  # noqa: E402  (reuses ask + the shared stress cache)

RATIOS = (0.3, 0.6, 0.9)
OUTD = config.REC_LLM


def main():
    files = sorted(glob.glob(os.path.join(OUTD, "llm_t2_*.json")))
    for p in files:
        m = re.match(r"llm_t2_(\w+)_(given|disc)\.json", os.path.basename(p))
        ds, src = m.groups()
        d = json.load(open(p))
        rows = [r for r in d["records"] if r["arm"] == "llmfull"]
        if len({r["gt"] for r in rows}) < 2:
            continue
        g, labels, _ = evidence.load_questionnaire(ds, src)
        for ratio in RATIOS:
            out_p = os.path.join(OUTD, f"stress_t2_{ds}_{src}_{int(ratio * 100)}.json")
            if os.path.exists(out_p):
                print(f"SKIP {out_p}", flush=True)
                continue
            rng = np.random.default_rng(abs(hash((ds, src, ratio))) % 2**32)
            out_rows = []
            for r in rows:
                kids = [c for c in g.children(r["latent"]) if not g.is_latent(c)]
                n_mask = int(round(ratio * len(kids)))
                hidden = set(rng.permutation(len(kids))[:n_mask].tolist())
                lines = [f'- EFFECT: it causes the item '
                         f'"{"(label hidden)" if i in hidden else labels[c]}"'
                         for i, c in enumerate(kids[:evidence.TOP_K])]
                if len(kids) > evidence.TOP_K:
                    lines.append(f"  (and {len(kids) - evidence.TOP_K} more items omitted)")
                out_rows.append((r, lines))
            # load pipeline phrases once
            src_rec = (os.path.join(config.REC_V6, f"t2_{ds}_given.json")
                       if src == "given" else
                       os.path.join(config.REC_DISC, f"t2_{ds}_disc.json"))
            sd = json.load(open(src_rec))
            srecs = sd["records"] if isinstance(sd, dict) and "records" in sd else sd
            phrases_of = {}
            for sr in srecs:
                if sr["arm"] == "core" and sr["latent"] not in phrases_of:
                    phrases_of[sr["latent"]] = sr.get("decoded_words")
            final = []
            for r, lines in out_rows:
                for arm in ("llmgraph", "llmfull"):
                    ph = phrases_of.get(r["latent"]) if arm == "llmfull" else None
                    if not ph and not lines:
                        name = None            # latent with no observed children
                    else:
                        prompt = prompts.build("t2", ph or None, lines or None)
                        key = f'{r["latent"]}:{int(ratio * 100)}'
                        name = rp.call(arm, key, prompt)
                    nr = dict(r)
                    nr["arm"] = arm
                    nr["decoded_words"] = [name] if name else None
                    nr["judge"] = None
                    nr["mask_ratio"] = ratio
                    final.append(nr)
            json.dump({"records": final}, open(out_p, "w"), indent=1)
            print(f"[saved {out_p} ({len(final)} rows)]", flush=True)


if __name__ == "__main__":
    main()
