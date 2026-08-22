"""Stage B: fill judge scores into every Stage A record file (gpt-5.5 via OpenRouter).

Idempotent: rows with a non-null judge are skipped; each file is rewritten atomically
after its batch completes, so a killed run resumes where it stopped. T1 rows use the
"completion" judge mode (target = true_label); T2 rows use "latent" (target = gt).
API failures leave judge null (missing, never wrong) and are retried on rerun.

Env: needs OPENAI_API_KEY (OpenRouter key), JUDGE_BASE_URL, JUDGE_MODEL=openai/gpt-5.5.
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "v6"))

import judge as judge_mod  # noqa: E402

CHUNK = 40

assert judge_mod.available(), "OPENAI_API_KEY not set"
assert "openrouter" in os.environ.get("JUDGE_BASE_URL", ""), "judge must go via OpenRouter"
assert os.environ.get("JUDGE_MODEL") == "openai/gpt-5.5", "judge model must be openai/gpt-5.5"


def fill(path):
    d = json.load(open(path))
    recs = d["records"] if isinstance(d, dict) and "records" in d else d
    todo = [i for i, r in enumerate(recs)
            if r.get("judge") is None and r.get("decoded_words")]
    if not todo:
        print(f"[skip] {os.path.basename(path)} (all judged)", flush=True)
        return 0
    # Merged T1+2 files mix item rows and latent rows; group by judge mode
    # first, then chunk within each group (each API batch is single-mode).
    by_mode = {"completion": [], "latent": []}
    for i in todo:
        by_mode["latent" if "latent" in recs[i] else "completion"].append(i)
    n_new = 0
    for mode, idxs in by_mode.items():
        for c0 in range(0, len(idxs), CHUNK):
            chunk = idxs[c0:c0 + CHUNK]
            items = [(recs[i]["decoded_words"],
                      recs[i]["gt"] if mode == "latent" else recs[i]["true_label"])
                     for i in chunk]
            verdicts = judge_mod.judge_batch(items, mode)
            for i, v in zip(chunk, verdicts):
                if v is not None:
                    recs[i]["judge"] = bool(v)
                    n_new += 1
            tmp = path + ".tmp"
            json.dump(d, open(tmp, "w"), indent=1)
            os.replace(tmp, path)
    print(f"[judged] {os.path.basename(path)}: +{n_new}/{len(todo)}", flush=True)
    return n_new


def main():
    files = (sorted(glob.glob(os.path.join(ROOT, "v6", "outputs", "rec_v2", "t[12]_*_given.json")))
             + sorted(glob.glob(os.path.join(HERE, "outputs", "rec_v2", "t[12]_*_disc.json")))
             + [p for p in sorted(glob.glob(os.path.join(ROOT, "v6", "outputs", "rec_v2",
                                                         "t1_*_*_*.json")))
                if not p.endswith("_rescored.json")]
             + sorted(glob.glob(os.path.join(HERE, "outputs", "rec_v2_naming",
                                             "naming_t1_*.json")))
             + sorted(glob.glob(os.path.join(HERE, "outputs", "rec_v2_llm", "llm_*.json")))
             + sorted(glob.glob(os.path.join(HERE, "outputs", "rec_v2_llm", "stress_t2_*.json")))
             + sorted(glob.glob(os.path.join(HERE, "outputs", "rec_v2_llm", "qwen_*.json")))
             + sorted(glob.glob(os.path.join(HERE, "outputs", "rec_v2_llm", "pfx_*.json")))
             + sorted(glob.glob(os.path.join(HERE, "outputs", "rec_v2_joint", "t12_*.json"))))
    total = 0
    for p in files:
        total += fill(p)
    print(f"[done] {len(files)} files, {total} new verdicts", flush=True)


if __name__ == "__main__":
    main()
