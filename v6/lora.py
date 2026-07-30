"""L3: LoRA injection into the frozen e5-large-v2 encoder (plan B, user-selected 2026-07-16).

LoRA wraps the query/value projections of the LAST 2 transformer layers. B is zero-initialized,
so the adapted encoder is EXACTLY the frozen encoder at step 0 (same discipline as f_neg/WeightNet).
Only LoRA matrices train; the e5 body stays frozen. Rollback = don't load the checkpoint.
"""
import os
import torch
import torch.nn as nn

R = int(os.environ.get("LORA_R", 8))
ALPHA = float(os.environ.get("LORA_ALPHA", 16))
N_LAYERS = int(os.environ.get("LORA_LAYERS", 2))          # how many last layers to wrap


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


def load_st(device="cuda"):
    """Frozen e5-large-v2 SentenceTransformer (Transformer + mean pooling)."""
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer("intfloat/e5-large-v2", device=device,
                             cache_folder=os.environ.get("HF_CACHE", "/data2/shuhao/hf_cache"))
    for p in st.parameters():
        p.requires_grad_(False)
    return st


def inject(st):
    """Wrap q/v projections of the last N_LAYERS encoder layers. Returns trainable params."""
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


def load_lora(st, path):
    ck = torch.load(path, map_location="cpu")
    bert = st._first_module().auto_model
    for i, layer in enumerate(bert.encoder.layer[-N_LAYERS:]):
        att = layer.attention.self
        for name in ("query", "value"):
            m = getattr(att, name)
            dev = m.A.weight.device
            m.A.weight.data = ck["state"][f"{i}.{name}.A"].to(dev)
            m.B.weight.data = ck["state"][f"{i}.{name}.B"].to(dev)
    return st


def encode_grad(st, texts, device, max_len=64):
    """Encode WITH gradients through LoRA: tokenize -> transformer -> mean pool -> L2 normalize.
    e5 requires the 'query: ' prefix (matches encode.py)."""
    feats = st.tokenize(["query: " + t for t in texts])
    feats = {k: v.to(device) for k, v in feats.items()}
    if feats["input_ids"].shape[1] > max_len:
        feats = {k: v[:, :max_len] for k, v in feats.items()}
    out = st._first_module()(feats)
    tok, mask = out["token_embeddings"], out["attention_mask"].unsqueeze(-1).float()
    emb = (tok * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    return torch.nn.functional.normalize(emb, dim=1)
