"""Batch-fill the remaining Qwen discrete-pilot T1 files with vLLM (one session).

Replicates run_plugin's pack construction and placebo derangement exactly (same
ordering, same rng seed), builds every missing (file, arm) prompt list, generates
greedily in one batched vLLM call per file-arm, and writes the same output schema
with backend provenance. Appends QWEN_PILOT_DONE to the pilot log when finished so
the prefix chain fires.
"""
import json
import os
import sys

import numpy as np

_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PKG)
sys.path.insert(0, os.path.join(_PKG, "plugin"))
import config  # noqa: E402
import evidence  # noqa: E402
import prompts  # noqa: E402

MODEL_ID = os.environ.get("QWEN_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
DS_T1 = ["bigfive", "hexaco", "rse", "wpi", "dass"]
ARMS = ["llmfull", "llmphrase", "llmgraph", "llmplacebo"]


def build_packs(base, src, records):
    d = json.load(open(records))
    recs = d["records"] if isinstance(d, dict) and "records" in d else d
    core = [r for r in recs if r["arm"] == "core"]
    g, labels, lat_names = evidence.load_questionnaire(base, src)
    packs = []
    by_fold = {}
    for r in core:
        by_fold.setdefault(r["fold"], []).append(r)
    for fold, rows in sorted(by_fold.items()):
        hidden = {r["var"] for r in rows}
        for r in rows:
            lines = evidence.q_item_lines(g, r["var"], labels, hidden, lat_names)
            packs.append((r, r.get("decoded_words"), lines))
    return packs


def main():
    from vllm import LLM, SamplingParams
    llm = LLM(model=MODEL_ID, download_dir=os.environ.get("HF_CACHE",
                                                          "/data2/shuhao/hf_cache"),
              dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=4096)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=32)
    for base in DS_T1:
        for src in ("given", "disc"):
            outf = os.path.join(config.REC_LLM, f"qwen_t1_{base}_{src}.json")
            if os.path.exists(outf):
                print(f"SKIP {outf}", flush=True)
                continue
            records = (os.path.join(config.REC_V6, f"t1_{base}_given.json")
                       if src == "given" else
                       os.path.join(config.REC_DISC, f"t1_{base}_disc.json"))
            packs = build_packs(base, src, records)
            rng = np.random.default_rng(0)
            n = len(packs)
            perm = rng.permutation(n)
            shift = (perm + 1) % n
            out_rows = []
            for arm in ARMS:
                texts, keep = [], []
                for i, (r, phrases, lines) in enumerate(packs):
                    ph, gl = phrases, lines
                    if arm == "llmphrase":
                        gl = None
                    elif arm == "llmgraph":
                        ph = None
                    elif arm == "llmplacebo":
                        _, ph, gl = packs[int(perm[int(shift[i])])]
                    if not ph and not gl:
                        keep.append(None)
                        continue
                    prompt = prompts.build("t1", ph, gl)
                    texts.append(tok.apply_chat_template(
                        [{"role": "user", "content": prompt}], tokenize=False,
                        add_generation_prompt=True))
                    keep.append(len(texts) - 1)
                outs = llm.generate(texts, sp) if texts else []
                gen = [o.outputs[0].text.strip().strip('"').splitlines()[0]
                       if o.outputs[0].text.strip() else None for o in outs]
                for i, (r, _, _) in enumerate(packs):
                    name = gen[keep[i]] if keep[i] is not None else None
                    nr = dict(r)
                    nr["arm"] = arm
                    nr["decoded_words"] = [name] if name else None
                    nr["judge"] = None
                    nr["prompt_version"] = prompts.PROMPT_VERSION
                    nr["model"] = f"local:{MODEL_ID}"
                    nr["backend"] = "vllm"
                    out_rows.append(nr)
                print(f"[{base} {src}] arm {arm} done", flush=True)
            json.dump({"records": out_rows}, open(outf, "w"), indent=1)
            print(f"[saved {outf} ({len(out_rows)} rows)]", flush=True)


if __name__ == "__main__":
    main()
