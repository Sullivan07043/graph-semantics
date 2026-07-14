"""LINE B: a completion operator TRAINED across datasets (the 'train on broad data' answer).

Model: typed message-passing network g_theta over the GIVEN graph. Input per node: [frozen label
embedding (zeros if unlabeled/latent), is_labeled, is_latent]. Relations: 4 edge types (latent->latent,
latent->obs, obs->obs, obs->latent) x 2 directions, each with its own message MLP; data-estimated edge
weight w rides along as a message feature. Output head: a d-dim embedding for EVERY node.

Training: masked-label reconstruction across ALL dev-pool datasets — sample a dataset, mask a random
subset of observed labels, reconstruct the held-back label embeddings (cosine loss). Auxiliary
generation-consistency term ties every generated node's output to the weighted combination of its
parents' outputs (same constraint as line A, here as a soft architectural prior so latent outputs are
meaningful for Task 2). The frozen encoder is NEVER fine-tuned: inputs and targets both live in its
fixed space, so the v3 semantic-drift failure mode does not exist here.

Held-out datasets are ZERO-SHOT: theta is frozen, the graph is new.

Usage:
  python gnn.py train            -> trains on pool.DEV, saves outputs/gnn.pt (+ loss curve log)
  python gnn.py eval [group]     -> 5-fold masked completion, arm 'gnn', geometric metrics
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import testbeds, pool, encode, metrics
from run_task1 import ALL_LOADERS

DEVICE = os.environ.get("GNN_DEVICE", "cuda:1" if torch.cuda.is_available()
                        and torch.cuda.device_count() > 1 else "cpu")
STEPS = int(os.environ.get("GNN_STEPS", 4000))
HID = int(os.environ.get("GNN_HID", 256))
LAYERS = int(os.environ.get("GNN_LAYERS", 4))
CKPT = os.path.join(HERE, "outputs", "gnn.pt")


def ts():
    return time.strftime("%H:%M:%S")


# --------------------------------------------------------------------------- graph tensors
def graph_tensors(ds):
    """Precompute per-dataset tensors: node order = graph.nodes; 8 relations of (src, dst, w)."""
    g, X = ds["graph"], ds["X"]
    obs = g.observed
    oi = {o: k for k, o in enumerate(obs)}
    T = encode.embed([ds["labels"][o] for o in obs])
    W, _ = g.estimate_weights(X, oi)
    nidx = {n: i for i, n in enumerate(g.nodes)}
    lat = set(g.latents)
    rel_src, rel_dst, rel_w = [[] for _ in range(8)], [[] for _ in range(8)], [[] for _ in range(8)]
    for a, b in g.edges:
        r = (2 if a in lat else 0) + (1 if b in lat else 0)          # ll=3, lo=2, ol=1, oo=0
        w = float(W.get((a, b), 0.0))
        rel_src[r].append(nidx[a]); rel_dst[r].append(nidx[b]); rel_w[r].append(w)
        rel_src[r + 4].append(nidx[b]); rel_dst[r + 4].append(nidx[a]); rel_w[r + 4].append(w)
    rels = []
    for r in range(8):
        rels.append((torch.tensor(rel_src[r], dtype=torch.long),
                     torch.tensor(rel_dst[r], dtype=torch.long),
                     torch.tensor(rel_w[r], dtype=torch.float32)))
    obs_pos = torch.tensor([nidx[o] for o in obs], dtype=torch.long)
    is_lat = torch.tensor([1.0 if n in lat else 0.0 for n in g.nodes])
    gen_pa = [(nidx[n], [(nidx[p], float(W.get((p, n), 0.0))) for p in g.parents(n)])
              for n in g.nodes if g.parents(n)]
    return dict(name=ds["name"], g=g, T=torch.tensor(T, dtype=torch.float32), rels=rels,
                obs_pos=obs_pos, is_lat=is_lat, n=len(g.nodes), gen_pa=gen_pa)


class CompletionGNN(nn.Module):
    def __init__(self, d, hid=HID, layers=LAYERS):
        super().__init__()
        self.inp = nn.Linear(d + 2, hid)
        self.msg = nn.ModuleList([nn.ModuleList(
            [nn.Sequential(nn.Linear(hid + 1, hid), nn.GELU(), nn.Linear(hid, hid))
             for _ in range(8)]) for _ in range(layers)])
        self.upd = nn.ModuleList([nn.Sequential(nn.Linear(2 * hid, hid), nn.GELU(),
                                                nn.Linear(hid, hid)) for _ in range(layers)])
        self.ln = nn.ModuleList([nn.LayerNorm(hid) for _ in range(layers)])
        self.head = nn.Linear(hid, d)

    def forward(self, gt, node_emb, labeled_mask):
        """node_emb: [n, d] label embeddings with zeros for unlabeled; labeled_mask: [n] float."""
        x = torch.cat([node_emb, labeled_mask[:, None], gt["is_lat"].to(node_emb.device)[:, None]], 1)
        h = self.inp(x)
        for L in range(len(self.upd)):
            agg = torch.zeros_like(h)
            cnt = torch.zeros(h.shape[0], 1, device=h.device)
            for r, (src, dst, w) in enumerate(gt["rels"]):
                if len(src) == 0:
                    continue
                src, dst, w = src.to(h.device), dst.to(h.device), w.to(h.device)
                m = self.msg[L][r](torch.cat([h[src], w[:, None]], 1))
                agg = agg.index_add(0, dst, m)
                cnt = cnt.index_add(0, dst, torch.ones(len(dst), 1, device=h.device))
            agg = agg / cnt.clamp(min=1.0)
            h = self.ln[L](h + self.upd[L](torch.cat([h, agg], 1)))
        return F.normalize(self.head(h), dim=1)


def masked_forward(model, gt, mask_obs_idx):
    """mask_obs_idx: indices INTO gt.obs_pos of observed nodes whose labels are hidden."""
    n, d = gt["n"], gt["T"].shape[1]
    node_emb = torch.zeros(n, d, device=DEVICE)
    labeled = torch.zeros(n, device=DEVICE)
    keep = [k for k in range(len(gt["obs_pos"])) if k not in set(mask_obs_idx)]
    pos = gt["obs_pos"][keep].to(DEVICE)
    node_emb[pos] = gt["T"][keep].to(DEVICE)
    labeled[pos] = 1.0
    return model(gt, node_emb, labeled)


def gen_consistency(out, gt):
    tgt, mix = [], []
    for nid, ps in gt["gen_pa"]:
        v = sum(w * out[p] for p, w in ps)
        tgt.append(out[nid]); mix.append(v)
    if not tgt:
        return out.sum() * 0.0
    Tg, Mx = torch.stack(tgt), torch.stack(mix)
    return (1 - F.cosine_similarity(Tg, Mx, dim=1)).mean()


def train():
    torch.manual_seed(0)
    data = [graph_tensors(ALL_LOADERS[n]()) for n in pool.DEV]
    d = data[0]["T"].shape[1]
    model = CompletionGNN(d).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    rng = np.random.default_rng(0)
    print(f"[{ts()}] train: {len(data)} dev datasets, d={d}, device={DEVICE}, steps={STEPS}", flush=True)
    for step in range(STEPS):
        gt = data[rng.integers(len(data))]
        n_obs = len(gt["obs_pos"])
        k = max(1, int(n_obs * rng.uniform(0.1, 0.45)))
        mask = rng.choice(n_obs, k, replace=False).tolist()
        out = masked_forward(model, gt, mask)
        pos = gt["obs_pos"][mask].to(DEVICE)
        target = gt["T"][mask].to(DEVICE)
        loss = (1 - F.cosine_similarity(out[pos], target, dim=1)).mean()
        loss = loss + 0.1 * gen_consistency(out, gt)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 200 == 0 or step == STEPS - 1:
            print(f"[{ts()}]   step {step}/{STEPS} loss={float(loss):.4f} ({gt['name']})", flush=True)
    os.makedirs(os.path.dirname(CKPT), exist_ok=True)
    torch.save({"state": model.state_dict(), "d": d, "hid": HID, "layers": LAYERS}, CKPT)
    print(f"[{ts()}] saved {CKPT}", flush=True)


def evaluate(group="heldout", folds=5):
    ck = torch.load(CKPT, map_location=DEVICE)
    model = CompletionGNN(ck["d"], ck["hid"], ck["layers"]).to(DEVICE)
    model.load_state_dict(ck["state"]); model.eval()
    names = pool.HELDOUT if group == "heldout" else (pool.DEV if group == "dev" else [group])
    out = {}
    for n in names:
        gt = graph_tensors(ALL_LOADERS[n]())
        T = gt["T"].numpy().astype(np.float64)
        Tn = metrics.norm_rows(T)
        n_obs = len(gt["obs_pos"])
        rng = np.random.default_rng(0)
        perm = rng.permutation(n_obs)
        cs, ms = [], []
        for f in range(folds):
            mask = sorted(int(i) for i in perm[f::folds])
            with torch.no_grad():
                o = masked_forward(model, gt, mask)
            P = o[gt["obs_pos"][mask].to(DEVICE)].cpu().numpy().astype(np.float64)
            cs.append(float(np.mean((metrics.norm_rows(P) * Tn[mask]).sum(1))))
            ms.append(metrics.match_acc(P, mask, T))
        out[n] = {"cos": float(np.mean(cs)), "match": float(np.mean(ms))}
        print(f"[{ts()}] gnn {n:10s} cos={out[n]['cos']:.3f} match={out[n]['match']:.3f}", flush=True)
    json.dump(out, open(os.path.join(HERE, "outputs", f"gnn_eval_{group}.json"), "w"), indent=1)
    return out


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"
    if cmd == "train":
        train()
    else:
        evaluate(sys.argv[2] if len(sys.argv) > 2 else "heldout")
