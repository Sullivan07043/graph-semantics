"""Batch-fill the Qwen discrete-pilot T3 (robot) files with vLLM (one session).

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
TARGETS = ["liftbody", "bodysawyer", "bodyiiwa", "bodyur5e"]
ARMS = (os.environ["ARMS"].split(",") if os.environ.get("ARMS")
        else ["llmfull", "llmphrase", "llmgraph", "llmplacebo", "llmfact"])
OUT_STEM = os.environ.get("OUT_STEM", "qwen_t3")
ROBOT_GRAPHS = {
    ("liftbody", "boss"): "lift_body_summary.json",
    ("liftbody", "truev3"): "lift_body_true_summary_v3.json",
    ("bodysawyer", "boss"): "body_sawyer_boss_summary.json",
    ("bodysawyer", "truev3"): "body_sawyer_true_summary_v3.json",
    ("bodyiiwa", "boss"): "body_iiwa_boss_summary.json",
    ("bodyiiwa", "truev3"): "body_iiwa_true_summary_v3.json",
    ("bodyur5e", "boss"): "body_ur5e_boss_summary.json",
    ("bodyur5e", "truev3"): "body_ur5e_true_summary_v3.json",
}


def build_packs(base, src, records):
    d = json.load(open(records))
    recs = d["records"] if isinstance(d, dict) and "records" in d else d
    core = [r for r in recs if r["arm"] == "core"]
    labels, W, T = evidence.load_robot(base, ROBOT_GRAPHS[(base, src)])
    packs = []
    by_fold = {}
    for r in core:
        by_fold.setdefault(r["fold"], []).append(r)
    for fold, rows in sorted(by_fold.items()):
        hidden = {r["var"] for r in rows}
        for r in rows:
            lines = evidence.r_channel_lines(r["var"], labels, W, T, hidden)
            packs.append((r, r.get("decoded_words"), lines))
    # llmhead: the naming head's evidence-backed proposals (openvocab rows that
    # rewrote the decode), same rule as run_plugin.py
    core_words = {(r["fold"], r["var"]): r.get("decoded_words") for r in core}
    head = {}
    for r in recs:
        if r["arm"] == "openvocab" and r.get("decoded_words"):
            k = (r["fold"], r["var"])
            if r["decoded_words"] != core_words.get(k):
                head[k] = ", ".join(r["decoded_words"])
    return packs, head


def main():
    from vllm import LLM, SamplingParams
    llm = LLM(model=MODEL_ID, download_dir=os.environ.get("HF_CACHE",
                                                          "/data2/shuhao/hf_cache"),
              dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=4096)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=32)
    for base in TARGETS:
        for src in ("boss", "truev3"):
            outf = os.path.join(config.REC_LLM, f"{OUT_STEM}_{base}_{src}.json")
            if os.path.exists(outf):
                print(f"SKIP {outf}", flush=True)
                continue
            records = os.path.join(config.REC_T3, f"t1_{base}_base_{src}.json")
            packs, head = build_packs(base, src, records)
            rng = np.random.default_rng(0)
            n = len(packs)
            perm = rng.permutation(n)
            shift = (perm + 1) % n
            out_rows = []
            for arm in ARMS:
                texts, keep = [], []
                for i, (r, phrases, lines) in enumerate(packs):
                    ph, gl = phrases, lines
                    fact, prop = False, None
                    if arm == "llmphrase":
                        gl = None
                    elif arm == "llmgraph":
                        ph = None
                    elif arm == "llmplacebo":
                        _, ph, gl = packs[int(perm[int(shift[i])])]
                    elif arm == "llmfact":
                        fact = True
                    elif arm == "llmhead":
                        fact = True
                        prop = head.get((r["fold"], r["var"]))
                    if not ph and not gl:
                        keep.append(None)
                        continue
                    prompt = prompts.build("t3", ph, gl, with_context_fact=fact,
                                           head_proposal=prop)
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
