"""
LLM judge (TLVD-style yes/no), shared by BOTH paths and BOTH deliverables.
  mode="latent"     : do the candidate words/name correctly denote the target construct?
  mode="completion" : do the recovered meaning words identify this SPECIFIC measured variable
                      (not merely its general family)?
ACC = fraction judged yes, mirroring TLVD's "proportion of latent variables correctly predicted".
Key from env OPENAI_API_KEY ONLY (never written to disk). Batched (one call judges a numbered list),
temperature 0, in-process cache keyed by (mode, recovered, target).
"""
import os, json, socket, time
import ssl
import urllib.error
import urllib.request

_CACHE = {}
MODEL = os.environ.get("JUDGE_MODEL", "gpt-4o-mini")
RETRIES = int(os.environ.get("JUDGE_RETRIES", 5))
RETRY_BASE = float(os.environ.get("JUDGE_RETRY_BASE", 1.0))


def available():
    return bool(os.environ.get("OPENAI_API_KEY"))


def _chat(prompt, model=None):
    key = os.environ["OPENAI_API_KEY"]
    payload = {"model": model or MODEL, "temperature": 0,
               "messages": [{"role": "user", "content": prompt}]}
    last_err = None
    max_attempts = max(RETRIES, 1)
    for attempt in range(max_attempts):
        req = urllib.request.Request("https://api.openai.com/v1/chat/completions",
                                     data=json.dumps(payload).encode(),
                                     headers={"Authorization": f"Bearer {key}",
                                              "Content-Type": "application/json"})
        try:
            r = json.loads(urllib.request.urlopen(req, timeout=120).read())
            return r["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")
            if "temperature" in msg and "temperature" in payload:
                payload.pop("temperature")             # some newer models reject explicit temperature
                continue
            if e.code not in (408, 409, 429, 500, 502, 503, 504):
                raise RuntimeError(f"judge API error: HTTP {e.code}: {msg[:200]}")
            last_err = RuntimeError(f"judge API error: HTTP {e.code}: {msg[:200]}")
        except (urllib.error.URLError, TimeoutError, socket.timeout, ssl.SSLError) as e:
            last_err = e
        if attempt < max_attempts - 1:
            wait = RETRY_BASE * (2 ** attempt)
            print(f"  [judge retry {attempt + 1}/{max_attempts} after {last_err}; sleeping {wait:.1f}s]",
                  flush=True)
            time.sleep(wait)
    raise RuntimeError(f"judge API unreachable after {max_attempts} attempts: {last_err}")


def _k(mode, rec, tgt):
    rec_s = ", ".join(rec) if isinstance(rec, (list, tuple)) else str(rec)
    return (mode, rec_s, str(tgt))


def judge_batch(items, mode):
    """items = list of (recovered, target); recovered = word list or name string. -> list[bool] | None."""
    if not available():
        return None
    out = [None] * len(items)
    todo = []
    for i, it in enumerate(items):
        v = _CACHE.get(_k(mode, *it))
        if v is None:
            todo.append(i)
        else:
            out[i] = v
    failed = False
    if todo:
        lines = []
        for r, i in enumerate(todo):
            rec, tgt = items[i]
            rec_s = ", ".join(rec) if isinstance(rec, (list, tuple)) else str(rec)
            if mode == "latent":
                lines.append(f'{r + 1}. candidate: [{rec_s}] ; target construct: "{tgt}"')
            else:
                lines.append(f'{r + 1}. recovered meaning: [{rec_s}] ; true variable: "{tgt}"')
        if mode == "latent":
            head = ("You are judging translations of hidden factors. Each candidate is a set of words from a "
                    "sparse dictionary decomposition, so a few words may be spurious noise. For each numbered "
                    "item, judge by the DOMINANT meaning of the candidate words: taken together, do they "
                    "correctly refer to the target construct? Synonyms and close paraphrases count as correct. "
                    "Do NOT reject an item merely because it contains some unrelated noise words; answer no "
                    "only if the dominant meaning points to a different construct, or is too generic to "
                    "identify the target.")
        else:
            head = ("You are judging recovered descriptions of measured variables. Each recovered meaning is a "
                    "set of words from a sparse dictionary decomposition, so a few words may be spurious noise. "
                    "For each numbered item, judge by the DOMINANT meaning: do the recovered words correctly "
                    "describe what the true variable measures? Synonyms and close paraphrases count as correct; "
                    "the words need not repeat the exact label. Answer no only if the dominant meaning "
                    "describes something else, or is too generic to relate to this variable.")
        prompt = (head + '\nRespond with ONLY a JSON array of "yes"/"no", one per item, in order.\n\n'
                  + "\n".join(lines))
        try:
            txt = _chat(prompt)
            arr = json.loads(txt[txt.find("["): txt.rfind("]") + 1])
            assert len(arr) == len(todo)
            for r, i in enumerate(todo):
                v = str(arr[r]).strip().lower().startswith("y")
                _CACHE[_k(mode, *items[i])] = v
                out[i] = v
        except Exception as e:
            print(f"  [judge batch failed ({e}); per-item fallback]", flush=True)
            for i in todo:
                rec, tgt = items[i]
                rec_s = ", ".join(rec) if isinstance(rec, (list, tuple)) else str(rec)
                q = (f'Candidate words (sparse decomposition; a few may be spurious noise): [{rec_s}]. Judging '
                     f'by their DOMINANT meaning, do they correctly refer to the construct "{tgt}"? Synonyms '
                     f'count as correct; do not reject merely for noise words.' if mode == "latent"
                     else f'Recovered meaning words (sparse decomposition; a few may be spurious): [{rec_s}]. '
                          f'Judging by their DOMINANT meaning, do they correctly describe what this variable '
                          f'measures: "{tgt}"? Synonyms count; reject only if the dominant meaning is about '
                          f'something else or too generic.')
                try:
                    v = _chat(q + " Answer yes or no only.").strip().lower().startswith("y")
                    _CACHE[_k(mode, *items[i])] = v
                    out[i] = v
                except Exception as item_e:
                    print(f"  [judge item failed ({item_e}); marking judge unavailable]", flush=True)
                    failed = True
    return None if failed or any(v is None for v in out) else out
