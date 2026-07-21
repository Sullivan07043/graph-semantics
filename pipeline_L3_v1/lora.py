"""LoRA adaptation of the last E5-large-v2 attention layers.

The query/value projections in the final layers are wrapped with rank-8 LoRA modules.  Their B
matrices are zero-initialized, so injection preserves the frozen encoder exactly at step zero.
"""
import hashlib
import os

import torch
import torch.nn as nn

R = int(os.environ.get("LORA_R", 8))
ALPHA = float(os.environ.get("LORA_ALPHA", 16))
N_LAYERS = int(os.environ.get("LORA_LAYERS", 2))
ENCODER_NAME = "intfloat/e5-large-v2"
CHECKPOINT_FORMAT = "graph-semantics-l3-lora"
CHECKPOINT_VERSION = 3
DICTIONARY_FORMAT = "graph-semantics-l3-dictionary"
DICTIONARY_VERSION = 3


class LoRALinear(nn.Module):
    def __init__(self, orig, r=R, alpha=ALPHA):
        super().__init__()
        self.orig = orig
        for p in self.orig.parameters():
            p.requires_grad_(False)
        self.A = nn.Linear(orig.in_features, r, bias=False)
        self.B = nn.Linear(r, orig.out_features, bias=False)
        nn.init.normal_(self.A.weight, std=0.01)
        nn.init.zeros_(self.B.weight)
        self.scale = alpha / r

    def forward(self, x):
        return self.orig(x) + self.scale * self.B(self.A(x))


def load_st(device=None):
    """Load the frozen E5-large-v2 SentenceTransformer on the requested/available device."""
    from sentence_transformers import SentenceTransformer
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    cache = os.environ.get("HF_CACHE")
    if not cache:
        local_cache = os.path.abspath(os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".hf_cache"))
        if os.path.isdir(local_cache):
            cache = local_cache
    kwargs = {"device": device}
    if cache:
        kwargs["cache_folder"] = cache
        # A complete repository-local cache should not trigger network metadata probes.
        kwargs["local_files_only"] = True
    st = SentenceTransformer(ENCODER_NAME, **kwargs)
    for p in st.parameters():
        p.requires_grad_(False)
    return st


def inject(st):
    """Wrap q/v projections of the last ``N_LAYERS`` and return trainable parameters."""
    bert = st._first_module().auto_model
    params = []
    for layer in bert.encoder.layer[-N_LAYERS:]:
        att = layer.attention.self
        att.query = LoRALinear(att.query).to(next(att.query.parameters()).device)
        att.value = LoRALinear(att.value).to(next(att.value.parameters()).device)
        params += list(att.query.A.parameters()) + list(att.query.B.parameters())
        params += list(att.value.A.parameters()) + list(att.value.B.parameters())
    return params


def lora_state(st):
    bert = st._first_module().auto_model
    out = {}
    for i, layer in enumerate(bert.encoder.layer[-N_LAYERS:]):
        att = layer.attention.self
        for name in ("query", "value"):
            m = getattr(att, name)
            out[f"{i}.{name}.A"] = m.A.weight.detach().cpu()
            out[f"{i}.{name}.B"] = m.B.weight.detach().cpu()
    return out


def checkpoint_payload(st, metadata=None):
    return {"format": CHECKPOINT_FORMAT, "version": CHECKPOINT_VERSION,
            "encoder": ENCODER_NAME, "r": R, "alpha": ALPHA, "layers": N_LAYERS,
            "state": lora_state(st), "metadata": dict(metadata or {})}


def _validated_checkpoint(path):
    ck = torch.load(path, map_location="cpu")
    if not isinstance(ck, dict) or ck.get("format") != CHECKPOINT_FORMAT \
            or ck.get("version") != CHECKPOINT_VERSION:
        got = ck.get("version", "legacy/unversioned") if isinstance(ck, dict) else type(ck).__name__
        raise RuntimeError(
            f"Incompatible L3 checkpoint {path!r}: got {got}, expected format "
            f"{CHECKPOINT_FORMAT!r} version {CHECKPOINT_VERSION}. Retrain "
            "pipeline_L3_v1/l3_train.py before re-encoding or L2 training.")
    expected = {"encoder": ENCODER_NAME, "r": R, "alpha": ALPHA, "layers": N_LAYERS}
    bad = {k: (ck.get(k), v) for k, v in expected.items() if ck.get(k) != v}
    if bad:
        raise RuntimeError(f"L3 checkpoint configuration is incompatible: {bad}")
    return ck


def load_lora(st, path):
    ck = _validated_checkpoint(path)
    bert = st._first_module().auto_model
    try:
        for i, layer in enumerate(bert.encoder.layer[-N_LAYERS:]):
            att = layer.attention.self
            for name in ("query", "value"):
                m = getattr(att, name)
                dev = m.A.weight.device
                m.A.weight.data.copy_(ck["state"][f"{i}.{name}.A"].to(dev))
                m.B.weight.data.copy_(ck["state"][f"{i}.{name}.B"].to(dev))
    except (KeyError, RuntimeError) as exc:
        raise RuntimeError(f"L3 checkpoint state in {path!r} is incompatible: {exc}") from exc
    return st


def checkpoint_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def encode_grad(st, texts, device, max_len=64):
    """Encode with gradients through LoRA, using E5's symmetric ``query:`` prefix."""
    feats = st.tokenize(["query: " + t for t in texts])
    feats = {k: v.to(device) for k, v in feats.items()}
    if feats["input_ids"].shape[1] > max_len:
        feats = {k: v[:, :max_len] for k, v in feats.items()}
    out = st._first_module()(feats)
    tok, mask = out["token_embeddings"], out["attention_mask"].unsqueeze(-1).float()
    emb = (tok * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    return torch.nn.functional.normalize(emb, dim=1)


class AdaptedSentenceEncoder:
    """Small SentenceTransformer-compatible wrapper used by ``encode.embed``."""

    def __init__(self, st, device):
        self.st = st
        self.device = device

    def encode(self, texts, batch_size=1024, normalize_embeddings=True):
        import numpy as np
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), min(int(batch_size), 256)):
                batch = [t[len("query: "):] if t.startswith("query: ") else t
                         for t in texts[i:i + min(int(batch_size), 256)]]
                out.append(encode_grad(self.st, batch, self.device, max_len=128).cpu().numpy())
        return np.concatenate(out)


def install_as_encode_model(encode_module, checkpoint_path, device=None):
    """Install the final L3 encoder behind ``encode.embed`` and return its SHA-256 identity."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    st = load_st(device)
    inject(st)
    load_lora(st, checkpoint_path)
    st.eval()
    encode_module._MODEL = AdaptedSentenceEncoder(st, device)
    return st, checkpoint_sha256(checkpoint_path)
