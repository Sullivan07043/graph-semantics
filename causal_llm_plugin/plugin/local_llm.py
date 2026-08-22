"""Local open-weight backend for the plugin (Qwen3-4B-Instruct-2507 by default).

Same contract as the API path: ask(prompt) -> first line of the answer. Greedy
decoding, chat template, model loaded once per process. Select the GPU with
CUDA_VISIBLE_DEVICES; override the checkpoint with QWEN_MODEL.
"""
import os

_MODEL_ID = os.environ.get("QWEN_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
_M = None
_T = None


def _load():
    global _M, _T
    if _M is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        cache = os.environ.get("HF_CACHE", "/data2/shuhao/hf_cache")
        _T = AutoTokenizer.from_pretrained(_MODEL_ID, cache_dir=cache)
        _M = AutoModelForCausalLM.from_pretrained(
            _MODEL_ID, cache_dir=cache, torch_dtype=torch.bfloat16, device_map="cuda:0")
        _M.eval()
    return _M, _T


def ask(prompt):
    import torch
    m, t = _load()
    msgs = [{"role": "user", "content": prompt}]
    text = t.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = t(text, return_tensors="pt").to(m.device)
    with torch.no_grad():
        out = m.generate(**ids, max_new_tokens=32, do_sample=False,
                         pad_token_id=t.eos_token_id)
    ans = t.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    return ans.strip().strip('"').splitlines()[0] if ans.strip() else None
