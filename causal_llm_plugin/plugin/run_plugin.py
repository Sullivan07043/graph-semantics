"""Batch driver for the LLM-steering plugin (Track A).

Reads the Stage A records (core arm) for one dataset x graph source, builds the
evidence package per masked variable, calls the naming LLM once per arm, and writes
rec_v2-schema records so the standard judge fill and rescore machinery applies.

Arms (block-level ablations of ONE frozen template):
  llmfull      phrases + causal neighbors
  llmphrase    phrases only
  llmgraph     causal neighbors only
  llmplacebo   full template, but the evidence of a DIFFERENT masked variable
               (deterministic derangement) - exposes prior leakage
  llmfact      t3 only: llmfull plus the dt context fact line (measures the fact)

Env: SURFACE=t1|t2|t3  BASE=<loader>  SOURCE=given|disc|boss|truev3
     RECORDS=<stage A records json>  OUT=<json>
     NAMING_MODEL (default openai/gpt-5.5 via OpenRouter), OPENAI_API_KEY,
     JUDGE_BASE_URL, ARMS (comma list, default all applicable).
Cache: one jsonl per file next to OUT, keyed by (prompt_version, arm, var).
"""
import json
import os
import sys
import urllib.request

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

SURFACE = os.environ["SURFACE"]
BASE = os.environ["BASE"]
SOURCE = os.environ["SOURCE"]
RECORDS = os.environ["RECORDS"]
OUT = os.environ["OUT"]
MODEL = os.environ.get("NAMING_MODEL", config.NAMING_MODEL)
API = os.environ.get("JUDGE_BASE_URL", config.API_URL)
KEY = os.environ.get("OPENAI_API_KEY", "")
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

_cache_path = OUT.replace(".json", "_cache.jsonl")
_cache = {}
if os.path.exists(_cache_path):
    for line in open(_cache_path):
        r = json.loads(line)
        _cache[(r["v"], r["arm"], r["key"])] = r["out"]


if os.environ.get("LLM_BACKEND") == "local":
    import local_llm

    def ask(prompt):
        return local_llm.ask(prompt)
else:
    def ask(prompt):
        return _ask_api(prompt)


def _ask_api(prompt):
    req = urllib.request.Request(
        f"{API}/chat/completions",
        data=json.dumps({"model": MODEL, "temperature": 0, "max_tokens": 8000,
                         "messages": [{"role": "user", "content": prompt}]}).encode(),
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=120).read())
    return r["choices"][0]["message"]["content"].strip().strip('"').splitlines()[0]


def call(arm, key, prompt):
    ck = (prompts.PROMPT_VERSION, arm, key)
    if ck in _cache:
        return _cache[ck]
    try:
        out = ask(prompt)
    except Exception as e:
        print(f"  [api error {type(e).__name__}] {arm} {key}", flush=True)
        return None
    _cache[ck] = out
    with open(_cache_path, "a") as f:
        f.write(json.dumps({"v": prompts.PROMPT_VERSION, "arm": arm,
                            "key": key, "out": out}, ensure_ascii=False) + "\n")
    return out


def main():
    d = json.load(open(RECORDS))
    recs = d["records"] if isinstance(d, dict) and "records" in d else d
    core = [r for r in recs if r["arm"] == "core"]

    # llmhead arm (t3): the deterministic head's proposal per (fold, var), taken
    # from the openvocab rows already present in RECORDS. Only evidence-backed
    # rewrites count: rows where the head kept the raw decode propose nothing.
    head = {}
    if SURFACE == "t3":
        core_words = {(r["fold"], r["var"]): r.get("decoded_words") for r in core}
        for r in recs:
            if r["arm"] == "openvocab" and r.get("decoded_words"):
                k = (r["fold"], r["var"])
                if r["decoded_words"] != core_words.get(k):
                    head[k] = ", ".join(r["decoded_words"])
    arms = os.environ.get("ARMS", "").split(",") if os.environ.get("ARMS") else \
        (["llmfull", "llmphrase", "llmgraph", "llmplacebo"]
         + (["llmfact"] if SURFACE == "t3" else []))

    # evidence per row
    if SURFACE in ("t1", "t2", "t12"):
        g, labels, lat_names = evidence.load_questionnaire(BASE, SOURCE)
    else:
        labels, W, T = evidence.load_robot(BASE, ROBOT_GRAPHS[(BASE, SOURCE)])

    packs = []                                    # (row, phrases, graph_lines)
    if SURFACE == "t12":
        # merged T1+2: per fold, translate masked items AND every latent. Latent names
        # are hidden from ALL evidence (lat_names withheld); a latent's children lines
        # hide the fold's masked item labels.
        by_fold = {}
        for r in core:
            by_fold.setdefault(r["fold"], []).append(r)
        for fold, rows in sorted(by_fold.items()):
            hidden = {r["var"] for r in rows if "var" in r}
            for r in rows:
                if "var" in r:
                    lines = evidence.q_item_lines(g, r["var"], labels, hidden, {})
                else:
                    kids = [c for c in g.children(r["latent"]) if not g.is_latent(c)]
                    lines = [f'- EFFECT: it causes the item '
                             f'"{"(label hidden)" if c in hidden else labels[c]}"'
                             for c in kids[:evidence.TOP_K]]
                    if len(kids) > evidence.TOP_K:
                        lines.append(f"  (and {len(kids) - evidence.TOP_K} more items omitted)")
                packs.append((r, r.get("decoded_words"), lines))
    elif SURFACE == "t2":
        seen = set()
        for r in core:
            if r["latent"] in seen:
                continue
            seen.add(r["latent"])
            lines = evidence.q_latent_lines(g, r["latent"], labels) \
                if r["latent"] in (g.latents or []) else []
            packs.append((r, r.get("decoded_words"), lines))
    else:
        by_fold = {}
        for r in core:
            by_fold.setdefault(r["fold"], []).append(r)
        for fold, rows in sorted(by_fold.items()):
            hidden = {r["var"] for r in rows}
            for r in rows:
                if SURFACE == "t1":
                    lines = evidence.q_item_lines(g, r["var"], labels, hidden, lat_names)
                else:
                    lines = evidence.r_channel_lines(r["var"], labels, W, T, hidden)
                packs.append((r, r.get("decoded_words"), lines))

    # deterministic derangement for the placebo arm
    rng = np.random.default_rng(0)
    n = len(packs)
    perm = rng.permutation(n)
    shift = (perm + 1) % n                       # index i takes evidence of perm-mate

    out_rows = []
    for arm in arms:
        for i, (r, phrases, lines) in enumerate(packs):
            ph, gl, fact, prop = phrases, lines, False, None
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
                name = None
            else:
                surf = ("t2" if "latent" in r else "t1") if SURFACE == "t12" else SURFACE
                prompt = prompts.build(surf, ph, gl, with_context_fact=fact,
                                       head_proposal=prop)
                key = (f'{r["fold"]}:L:{r["latent"]}' if "latent" in r
                       else f'{r["fold"]}:{r["var"]}') if SURFACE == "t12" else \
                    (r["latent"] if SURFACE == "t2" else f'{r["fold"]}:{r["var"]}')
                name = call(arm, key, prompt)
            nr = dict(r)
            nr["arm"] = arm
            nr["decoded_words"] = [name] if name else None
            nr["judge"] = None
            nr["prompt_version"] = prompts.PROMPT_VERSION
            nr["model"] = ("local:" + os.environ.get("QWEN_MODEL", "Qwen3-4B-Instruct-2507")
                           if os.environ.get("LLM_BACKEND") == "local" else MODEL)
            out_rows.append(nr)
        print(f"[{BASE} {SOURCE} {SURFACE}] arm {arm} done "
              f"({sum(1 for x in out_rows if x['arm'] == arm and x['decoded_words'])}"
              f"/{len(packs)} named)", flush=True)

    json.dump({"records": out_rows}, open(OUT, "w"), indent=1)
    print(f"[saved {OUT} ({len(out_rows)} rows)]", flush=True)


if __name__ == "__main__":
    main()
