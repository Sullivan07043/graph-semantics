"""LADDER L1 (main line): g_phi — edge-conditioned NONLINEAR generation transform.

Per-edge generation contribution becomes |w| * g_phi(e_parent, cond), cond = [sign(w), |w|, is_ll]
(latent->latent vs latent->observed). Architecture guarantees INIT == the current frozen method:
    g_phi(x, cond) = normalize( base(x, sign) + Delta_phi(x, cond) ),  Delta zero-initialized,
    base(x, sign) = x if sign>=0 else f_neg(x)   (f_neg frozen inside).
So step-0 behavior is exactly generation-with-f_neg; training only adds expressiveness (degree,
concretization, contextualization) on top.

Training pairs (LODO-safe: WordNet + DEV pool only):
  - WordNet SYNONYM lemma pairs (same synset), cond=(+1, 1, 0)
  - WordNet ANTONYM pairs, cond=(-1, 1, 0)
  - DEV leave-one-out generation pairs: for each dev latent p and child c, x = leave-one-out pole
    vector of p (positive children mean; negative children folded in via frozen f_neg), cond =
    (sign(w_c), |w_c|, 0); target = a_c. Latent->latent pairs from hierarchy: none in dev (TLVD's
    are latent GT-free) — the ll flag exists for inference completeness, trained only via WordNet.
Loss: cosine to target + 0.1*||Delta|| (stay near base unless data demands).

Usage: python pipeline_v3/genphi.py train | probe     -> outputs/genphi.pt
"""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import pool, encode, negop

DEVICE = "cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else "cpu"
CKPT = os.environ.get("GENPHI_CKPT", os.path.join(ROOT, "outputs", "genphi.pt"))
STEPS = int(os.environ.get("GENPHI_STEPS", 4000))


class GPhi(nn.Module):
    def __init__(self, d, fneg, hid=1024, cdim=3):
        super().__init__()
        self.fneg = fneg                                  # frozen inside
        for p in self.fneg.parameters():
            p.requires_grad_(False)
        self.inp = nn.Linear(d, hid)
        self.gamma = nn.Linear(cdim, hid)
        self.beta = nn.Linear(cdim, hid)
        self.out = nn.Linear(hid, d)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)   # Delta == 0 at init

    def forward(self, x, cond):
        """x: [n,d]; cond: [n,3] = (sign, |w|, is_ll)."""
        sign = cond[:, :1]
        base = torch.where(sign >= 0, x, self.fneg(x))
        h = F.gelu(self.inp(x)) * (1 + self.gamma(cond)) + self.beta(cond)
        return F.normalize(base + self.out(F.gelu(h)), dim=-1)


def wordnet_pairs():
    import nltk
    try:
        from nltk.corpus import wordnet as wn
        list(wn.all_synsets("a"))[:1]
    except LookupError:
        nltk.download("wordnet", quiet=True)
        from nltk.corpus import wordnet as wn
    from nltk.corpus import wordnet as wn
    syn, ant = set(), set()
    for s in wn.all_synsets():
        lemmas = [l.name().replace("_", " ") for l in s.lemmas()]
        for i in range(len(lemmas)):
            for j in range(i + 1, min(len(lemmas), i + 3)):
                if lemmas[i].lower() != lemmas[j].lower():
                    syn.add((lemmas[i], lemmas[j]))
        for l in s.lemmas():
            for a in l.antonyms():
                ant.add((l.name().replace("_", " "), a.name().replace("_", " ")))
    return sorted(syn), sorted(ant)


def dev_gt_pairs(neg):
    """(x=embed(latent GT text), cond, y=child label emb): teaches CONCRETIZATION — abstract
    construct name -> concrete item phrasing (meeting-note pt.3 sanctions dev latent-GT use;
    held-out untouched). Negative children get sign=-1 so the f_neg base handles polarity."""
    from run_task1 import ALL_LOADERS
    xs, cs, ys = [], [], []
    for name in pool.DEV:
        ds = ALL_LOADERS[name]()
        g, X, labels, gt = ds["graph"], ds["X"], ds["labels"], ds["latent_gt"]
        oi = {o: k for k, o in enumerate(g.observed)}
        W, _ = g.estimate_weights(X, oi)
        T = encode.embed([labels[o] for o in g.observed])
        lat = [L for L in g.latents if L in gt]
        if not lat:
            continue
        G = encode.embed([gt[L] for L in lat]).astype(np.float32)
        for gi, L in enumerate(lat):
            for c in g.children(L):
                if g.is_latent(c):
                    continue
                w = W.get((L, c), 0.0)
                if abs(w) < 0.05:
                    continue
                xs.append(G[gi]); cs.append([1.0 if w >= 0 else -1.0, abs(w), 0.0])
                ys.append(T[oi[c]])
    return np.array(xs, np.float32), np.array(cs, np.float32), np.array(ys, np.float32)


def dev_gen_pairs(neg):
    """(x=leave-one-out pole vector, cond, y=child label emb) from the dev pool."""
    from run_task1 import ALL_LOADERS
    xs, cs, ys = [], [], []
    for name in pool.DEV:
        ds = ALL_LOADERS[name]()
        g, X, labels = ds["graph"], ds["X"], ds["labels"]
        oi = {o: k for k, o in enumerate(g.observed)}
        W, _ = g.estimate_weights(X, oi)
        T = encode.embed([labels[o] for o in g.observed])
        with torch.no_grad():
            Tneg = neg(torch.tensor(T, dtype=torch.float32)).numpy()
        for L in g.latents:
            ch = [(c, W.get((L, c), 0.0)) for c in g.children(L) if not g.is_latent(c)]
            ch = [(c, w) for c, w in ch if abs(w) > 0.05]
            if len(ch) < 3:
                continue
            for c, w in ch:
                rest = [(T[oi[o]] if wo >= 0 else Tneg[oi[o]]) for o, wo in ch if o != c]
                x = np.mean(rest, 0)
                x /= np.linalg.norm(x) + 1e-9
                xs.append(x); cs.append([1.0 if w >= 0 else -1.0, abs(w), 0.0]); ys.append(T[oi[c]])
    return np.array(xs, np.float32), np.array(cs, np.float32), np.array(ys, np.float32)


def train():
    torch.manual_seed(0)
    neg = negop.load()
    syn, ant = wordnet_pairs()
    words = sorted({w for p in syn + ant for w in p})
    widx = {w: i for i, w in enumerate(words)}
    E = encode.embed(words).astype(np.float32)
    def pairs_to_xy(pairs, sign):
        a = np.stack([E[widx[p]] for p, q in pairs] + [E[widx[q]] for p, q in pairs])
        b = np.stack([E[widx[q]] for p, q in pairs] + [E[widx[p]] for p, q in pairs])
        c = np.tile([[sign, 1.0, 0.0]], (len(a), 1)).astype(np.float32)
        return a, c, b
    xs_s, cs_s, ys_s = pairs_to_xy(syn, +1.0)
    xs_a, cs_a, ys_a = pairs_to_xy(ant, -1.0)
    xs_d, cs_d, ys_d = dev_gen_pairs(neg)
    xs_g, cs_g, ys_g = dev_gt_pairs(neg)
    print(f"[{time.strftime('%H:%M:%S')}] pairs: syn={len(xs_s)} ant={len(xs_a)} dev={len(xs_d)} "
          f"gt-concretize={len(xs_g)}", flush=True)
    rng = np.random.default_rng(0)
    val = rng.permutation(len(xs_d))[:max(1, len(xs_d) // 10)]
    trn = np.setdiff1d(np.arange(len(xs_d)), val)
    m = GPhi(E.shape[1], neg).to(DEVICE)
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=3e-4)
    g = torch.Generator().manual_seed(0)
    T = lambda a: torch.tensor(a, device=DEVICE)
    for step in range(STEPS):
        loss = 0.0
        for X_, C_, Y_, k in [(xs_s, cs_s, ys_s, 256), (xs_a, cs_a, ys_a, 128),
                              (xs_d[trn], cs_d[trn], ys_d[trn], 256), (xs_g, cs_g, ys_g, 256)]:
            idx = torch.randint(len(X_), (min(k, len(X_)),), generator=g).numpy()
            out = m(T(X_[idx]), T(C_[idx]))
            loss = loss + (1 - F.cosine_similarity(out, T(Y_[idx]), dim=1)).mean()
        loss = loss + 0.03 * (m.out.weight.norm() + m.out.bias.norm())
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0 or step == STEPS - 1:
            with torch.no_grad():
                vout = m(T(xs_d[val]), T(cs_d[val]))
                vc = F.cosine_similarity(vout, T(ys_d[val]), dim=1).mean()
                b0 = F.cosine_similarity(
                    torch.where(T(cs_d[val])[:, :1] >= 0, T(xs_d[val]), m.fneg(T(xs_d[val]))),
                    T(ys_d[val]), dim=1).mean()
            print(f"[{time.strftime('%H:%M:%S')}] step {step} loss={float(loss):.3f} "
                  f"dev-val cos={float(vc):.3f} (base={float(b0):.3f})", flush=True)
    torch.save({"state": m.state_dict(), "d": E.shape[1]}, CKPT)
    print(f"saved {CKPT}", flush=True)


def load():
    ck = torch.load(CKPT, map_location="cpu")
    m = GPhi(ck["d"], negop.load())
    m.load_state_dict(ck["state"]); m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


if __name__ == "__main__":
    train() if (len(sys.argv) < 2 or sys.argv[1] == "train") else None
