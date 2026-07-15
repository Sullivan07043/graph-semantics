"""SEMANTIC NEGATION OPERATOR f_neg (step A of the 2026-07-15 plan).

Mechanism hypothesis: a reverse-keyed item is generated from the semantic NEGATION of its factor's
meaning, not from the vector negation -u ("a negative embedding is not an antonym embedding" — shown
independently by our oracle-free polarity analysis and the xuran-branch oracle). f_neg is a GLOBAL
semantic component (same status as the frozen encoder and the dictionary): a map in e5 space taking a
meaning direction to its opposite pole.

Training data (LODO-safe: WordNet + DEV pool only, held-out untouched):
  - WordNet antonym lemma pairs (~3.5k), both directions;
  - dev-pool factor pole pairs: for each latent with >=2 positively and >=2 negatively loaded items,
    (normalized mean of positive-item labels <-> normalized mean of negative-item labels), both
    directions, upweighted (in-domain psychometric register).
Losses: mapping cos loss + involution f(f(x)) ~ x.

Usage: python negop.py train  -> outputs/negop.pt (+ prints WordNet val cos and dev-pair cos)
       python negop.py probe  -> qualitative decode of f_neg on a few probes
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pool, encode
# NOTE: run_task1 is imported LAZILY inside dev_pole_pairs() — the runners import this module at
# startup (NEGOP=1) and a top-level import here would be circular.

DEVICE = "cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else "cpu"
CKPT = os.environ.get("NEGOP_CKPT", os.path.join(HERE, "outputs", "negop.pt"))
STEPS = int(os.environ.get("NEGOP_STEPS", 3000))
DEV_WEIGHT = float(os.environ.get("NEGOP_DEV_WEIGHT", 5.0))


class NegOp(nn.Module):
    def __init__(self, d, hid=1024):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hid), nn.GELU(), nn.Linear(hid, d))

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


def wordnet_pairs():
    import nltk
    try:
        from nltk.corpus import wordnet as wn
        wn.all_synsets
        list(wn.all_synsets("a"))[:1]
    except LookupError:
        nltk.download("wordnet", quiet=True)
        from nltk.corpus import wordnet as wn
    from nltk.corpus import wordnet as wn
    pairs = set()
    for syn in wn.all_synsets():
        for l in syn.lemmas():
            for a in l.antonyms():
                pairs.add(tuple(sorted((l.name().replace("_", " "), a.name().replace("_", " ")))))
    return sorted(pairs)


def dev_pole_pairs():
    from run_task1 import ALL_LOADERS
    out = []
    for name in pool.DEV:
        ds = ALL_LOADERS[name]()
        g, X, labels = ds["graph"], ds["X"], ds["labels"]
        oi = {o: k for k, o in enumerate(g.observed)}
        W, _ = g.estimate_weights(X, oi)
        T = encode.embed([labels[o] for o in g.observed])
        for L in g.latents:
            ch = [c for c in g.children(L) if not g.is_latent(c)]
            pos = [oi[c] for c in ch if W.get((L, c), 0.0) > 0.05]
            neg = [oi[c] for c in ch if W.get((L, c), 0.0) < -0.05]
            if len(pos) >= 2 and len(neg) >= 2:
                vp = T[pos].mean(0); vp /= np.linalg.norm(vp) + 1e-9
                vn = T[neg].mean(0); vn /= np.linalg.norm(vn) + 1e-9
                out.append((vp, vn))
    return out


def train():
    torch.manual_seed(0)
    wpairs = wordnet_pairs()
    words = sorted({w for p in wpairs for w in p})
    widx = {w: i for i, w in enumerate(words)}
    E = encode.embed(words)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(wpairs))
    val_n = max(1, len(wpairs) // 10)
    val_idx, tr_idx = perm[:val_n], perm[val_n:]
    A = np.stack([E[widx[a]] for a, b in wpairs]); B = np.stack([E[widx[b]] for a, b in wpairs])
    dev = dev_pole_pairs()
    print(f"[{time.strftime('%H:%M:%S')}] wordnet pairs={len(wpairs)} (val {val_n}), "
          f"dev pole pairs={len(dev)}, d={E.shape[1]}", flush=True)
    Dp = torch.tensor(np.stack([p for p, q in dev] + [q for p, q in dev]), dtype=torch.float32,
                      device=DEVICE)
    Dq = torch.tensor(np.stack([q for p, q in dev] + [p for p, q in dev]), dtype=torch.float32,
                      device=DEVICE)
    At = torch.tensor(np.concatenate([A[tr_idx], B[tr_idx]]), dtype=torch.float32, device=DEVICE)
    Bt = torch.tensor(np.concatenate([B[tr_idx], A[tr_idx]]), dtype=torch.float32, device=DEVICE)
    Av = torch.tensor(np.concatenate([A[val_idx], B[val_idx]]), dtype=torch.float32, device=DEVICE)
    Bv = torch.tensor(np.concatenate([B[val_idx], A[val_idx]]), dtype=torch.float32, device=DEVICE)

    m = NegOp(E.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=3e-4)
    g = torch.Generator().manual_seed(0)
    for step in range(STEPS):
        idx = torch.randint(len(At), (256,), generator=g).to(DEVICE)
        x, y = At[idx], Bt[idx]
        loss = (1 - F.cosine_similarity(m(x), y, dim=1)).mean()
        loss = loss + 0.3 * (1 - F.cosine_similarity(m(m(x)), x, dim=1)).mean()   # involution
        loss = loss + DEV_WEIGHT * (1 - F.cosine_similarity(m(Dp), Dq, dim=1)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0 or step == STEPS - 1:
            with torch.no_grad():
                vc = F.cosine_similarity(m(Av), Bv, dim=1).mean()
                dc = F.cosine_similarity(m(Dp), Dq, dim=1).mean()
                base = F.cosine_similarity(Av, Bv, dim=1).mean()
            print(f"[{time.strftime('%H:%M:%S')}] step {step} loss={float(loss):.4f} "
                  f"val_cos={float(vc):.3f} (raw antonym cos={float(base):.3f}) "
                  f"dev_pole_cos={float(dc):.3f}", flush=True)
    torch.save({"state": m.state_dict(), "d": E.shape[1]}, CKPT)
    print(f"saved {CKPT}", flush=True)


def load(d=None):
    ck = torch.load(CKPT, map_location="cpu")
    m = NegOp(ck["d"])
    m.load_state_dict(ck["state"]); m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def probe():
    import metrics
    C, cwords = encode.load_dictionary()
    m = load()
    texts = ["talkative, outgoing, life of the party", "calm and emotionally stable",
             "organized, disciplined, careful", "I trust other people easily"]
    E = encode.embed(texts)
    alpha = metrics.pick_alpha(E, C)
    with torch.no_grad():
        N = m(torch.tensor(E, dtype=torch.float32)).numpy().astype(np.float64)
    for t, w0, w1 in zip(texts, metrics.decode_words(E, C, cwords, alpha),
                         metrics.decode_words(N, C, cwords, alpha)):
        print(f"  {t!r}\n    原方向 -> {w0[:5]}\n    f_neg  -> {w1[:5]}", flush=True)


if __name__ == "__main__":
    (train if (len(sys.argv) < 2 or sys.argv[1] == "train") else probe)()
