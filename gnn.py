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
import testbeds, pool, encode, metrics, optimize
from run_task1 import ALL_LOADERS

DEVICE = os.environ.get("GNN_DEVICE", "cuda:1" if torch.cuda.is_available()
                        and torch.cuda.device_count() > 1 else "cpu")
STEPS = int(os.environ.get("GNN_STEPS", 4000))
HID = int(os.environ.get("GNN_HID", 256))
LAYERS = int(os.environ.get("GNN_LAYERS", 4))
CKPT = os.environ.get("GNN_CKPT", os.path.join(HERE, "outputs", "gnn.pt"))
# causal-constraint losses on the GNN's OUTPUT embeddings (all trained across the dev pool):
LAM_INDEP = float(os.environ.get("GNN_LAM_INDEP", 0.0))  # d-separated pairs -> decorrelate outputs
LAM_RES = float(os.environ.get("GNN_LAM_RES", 0.0))      # residual (out - gen) cos ~ shrunk partial corr
LAM_COLL = float(os.environ.get("GNN_LAM_COLL", 0.0))    # explaining away at v-structures
LAM_DEP = float(os.environ.get("GNN_LAM_DEP", 0.0))      # faithfulness floor on trek pairs
LAM_MB = float(os.environ.get("GNN_LAM_MB", 0.0))        # Markov-blanket locality consistency
LAM_JAC = float(os.environ.get("GNN_LAM_JAC", 0.0))      # Jacobian locality: d(pred_i)/d(label_j) ~ 0
                                                         # for j marginally independent of i (soft MB)
LAM_GEN = float(os.environ.get("GNN_LAM_GEN", 0.0))      # latent generation head: masked child ~
                                                         # sign * gen(h_parent) (latents must carry
                                                         # their children's semantics)


def ts():
    return time.strftime("%H:%M:%S")


# --------------------------------------------------------------------------- graph tensors
def graph_tensors(ds):
    """Precompute per-dataset tensors: node order = graph.nodes; 8 relations of (src, dst, w)."""
    g, X = ds["graph"], ds["X"]
    obs = g.observed
    oi = {o: k for k, o in enumerate(obs)}
    T = encode.embed([ds["labels"][o] for o in obs])
    W, score = g.estimate_weights(X, oi)
    nidx = {n: i for i, n in enumerate(g.nodes)}
    lat = set(g.latents)
    NREL = 9
    rel_src, rel_dst, rel_w = ([[] for _ in range(NREL)] for _ in range(3))
    for a, b in g.edges:
        r = (2 if a in lat else 0) + (1 if b in lat else 0)          # ll=3, lo=2, ol=1, oo=0
        w = float(W.get((a, b), 0.0))
        rel_src[r].append(nidx[a]); rel_dst[r].append(nidx[b]); rel_w[r].append(w)
        rel_src[r + 4].append(nidx[b]); rel_dst[r + 4].append(nidx[a]); rel_w[r + 4].append(w)
    # relation 8: data-correlation top-k neighbours among observed (signed corr as weight) — injects the
    # data signal beyond the design edges (what the raw-correlation baseline exploits), no label leakage
    Cr = np.corrcoef(X.T); np.fill_diagonal(Cr, 0.0)
    K = min(5, len(obs) - 1)
    for i, o in enumerate(obs):
        for j in np.argsort(-np.abs(Cr[i]))[:K]:
            rel_src[8].append(nidx[obs[j]]); rel_dst[8].append(nidx[o]); rel_w[8].append(float(Cr[i, j]))
    rels = []
    for r in range(NREL):
        rels.append((torch.tensor(rel_src[r], dtype=torch.long),
                     torch.tensor(rel_dst[r], dtype=torch.long),
                     torch.tensor(rel_w[r], dtype=torch.float32)))
    obs_pos = torch.tensor([nidx[o] for o in obs], dtype=torch.long)
    is_lat = torch.tensor([1.0 if n in lat else 0.0 for n in g.nodes])
    gen_pa = [(nidx[n], [(nidx[p], float(W.get((p, n), 0.0))) for p in g.parents(n)])
              for n in g.nodes if g.parents(n)]
    obs_i = {o: k for k, o in enumerate(obs)}
    mb_obs = [[obs_i[x] for x in g.mb_observed(o)] for o in obs]     # observed MB-closure per item
    # constraint tensors (all on the GIVEN graph + data, no label leakage)
    zp = g.independent_pairs()
    zp_a = torch.tensor([nidx[a] for a, b in zp], dtype=torch.long)
    zp_b = torch.tensor([nidx[b] for a, b in zp], dtype=torch.long)
    vst = [(nidx[p1], nidx[p2], nidx[c]) for p1, p2, c in g.v_structures()]
    obs_set = set(obs)
    tp = [(a, b) for a, b in g.trek_pairs() if a in obs_set and b in obs_set]
    tp_a = torch.tensor([nidx[a] for a, b in tp], dtype=torch.long)
    tp_b = torch.tensor([nidx[b] for a, b in tp], dtype=torch.long)
    tp_floor = torch.tensor([0.5 * abs(float(Cr[obs_i[a], obs_i[b]])) for a, b in tp],
                            dtype=torch.float32)
    _, Ppc = optimize.partial_residual_corr(g, X, oi, score)
    Ppc = optimize.shrink_corr(Ppc, X.shape[0])
    indep_of = {}
    for a, b in zp:
        indep_of.setdefault(a, []).append(b)
        indep_of.setdefault(b, []).append(a)
    Wmat = torch.zeros(len(g.nodes), len(g.nodes))
    for a, b in g.edges:
        Wmat[nidx[b], nidx[a]] = float(W.get((a, b), 0.0))
    return dict(name=ds["name"], g=g, T=torch.tensor(T, dtype=torch.float32), rels=rels,
                obs_pos=obs_pos, is_lat=is_lat, n=len(g.nodes), gen_pa=gen_pa,
                Craw=torch.tensor(np.clip(Cr, 0, None), dtype=torch.float32), mb_obs=mb_obs,
                zp=(zp_a, zp_b), vst=vst, tp=(tp_a, tp_b, tp_floor),
                Ppc=torch.tensor(Ppc, dtype=torch.float32), Wmat=Wmat, indep_of=indep_of)


class CompletionGNN(nn.Module):
    def __init__(self, d, hid=HID, layers=LAYERS):
        super().__init__()
        # input: [own label emb (0 if hidden), rawcorr-weighted mean of visible label embs, flags]
        self.inp = nn.Linear(2 * d + 2, hid)
        self.msg = nn.ModuleList([nn.ModuleList(
            [nn.Sequential(nn.Linear(hid + 1, hid), nn.GELU(), nn.Linear(hid, hid))
             for _ in range(9)]) for _ in range(layers)])
        self.upd = nn.ModuleList([nn.Sequential(nn.Linear(2 * hid, hid), nn.GELU(),
                                                nn.Linear(hid, hid)) for _ in range(layers)])
        self.ln = nn.ModuleList([nn.LayerNorm(hid) for _ in range(layers)])
        self.head = nn.Linear(hid, d)
        # latent GENERATION head: decodes a LATENT node's hidden state into the embedding-space
        # direction its children are generated from. Trained so a masked child is reconstructable
        # from sign * gen(h_parent) ALONE -> forces the latent pathway to carry child semantics
        # (without it the rawcorr input feature lets predictions bypass latents entirely).
        self.gen = nn.Linear(hid, d)

    def forward(self, gt, node_emb, labeled_mask, corr_emb, h_delta=None, return_hidden=False):
        """node_emb: [n, d] label embeddings with zeros for unlabeled; labeled_mask: [n] float;
        corr_emb: [n, d] rawcorr-weighted mean of the VISIBLE label embeddings (the strong no-graph
        baseline, provided as an input feature so the network learns the structure delta on top).
        h_delta: optional [n, hid] perturbation added after the input layer (Jacobian read-off);
        return_hidden: also return the final hidden states (latent generation head reads them)."""
        x = torch.cat([node_emb, corr_emb, labeled_mask[:, None],
                       gt["is_lat"].to(node_emb.device)[:, None]], 1)
        h = self.inp(x)
        if h_delta is not None:
            h = h + h_delta
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
        out = F.normalize(self.head(h), dim=1)
        if return_hidden:
            return out, h
        return out


def masked_inputs(gt, mask_obs_idx):
    n, d = gt["n"], gt["T"].shape[1]
    node_emb = torch.zeros(n, d, device=DEVICE)
    labeled = torch.zeros(n, device=DEVICE)
    keep = [k for k in range(len(gt["obs_pos"])) if k not in set(mask_obs_idx)]
    pos = gt["obs_pos"][keep].to(DEVICE)
    node_emb[pos] = gt["T"][keep].to(DEVICE)
    labeled[pos] = 1.0
    # rawcorr-weighted mean of visible labels, for every OBSERVED node (zeros for latents)
    corr_emb = torch.zeros(n, d, device=DEVICE)
    Wc = gt["Craw"][:, keep].to(DEVICE)                              # [n_obs, n_keep]
    Wc = Wc / Wc.sum(1, keepdim=True).clamp(min=1e-9)
    corr_emb[gt["obs_pos"].to(DEVICE)] = Wc @ gt["T"][keep].to(DEVICE)
    return node_emb, labeled, corr_emb


def masked_forward(model, gt, mask_obs_idx):
    """mask_obs_idx: indices INTO gt.obs_pos of observed nodes whose labels are hidden."""
    node_emb, labeled, corr_emb = masked_inputs(gt, mask_obs_idx)
    return model(gt, node_emb, labeled, corr_emb)


def genhead_readoff(model, gt, mask_obs_idx, lat_names):
    """Latent translation via the trained GENERATION head: u_L = gen(h_L) (the direction this
    latent generates its children from). Requires a checkpoint trained with GNN_LAM_GEN>0."""
    g = gt["g"]
    nidx = {n_: i for i, n_ in enumerate(g.nodes)}
    node_emb, labeled, corr_emb = masked_inputs(gt, mask_obs_idx)
    with torch.no_grad():
        _, hid = model(gt, node_emb, labeled, corr_emb, return_hidden=True)
        U = model.gen(hid[torch.tensor([nidx[L] for L in lat_names], device=DEVICE)])
    return U.cpu().numpy().astype(np.float64)


def jacobian_readoff(model, gt, mask_obs_idx, lat_names):
    """JACOBIAN READ-OFF (derivative-side latent translation): a latent's meaning is the input
    direction whose perturbation most increases its VISIBLE observed descendants' alignment with
    their own labels, sign-weighted by the data:  v_L = d/d(delta_L) sum_c sign_c (out_c . a_c).
    Uses only visible labels (no leakage). Returns [n_lat, d] numpy."""
    import torch
    g = gt["g"]
    nidx = {n_: i for i, n_ in enumerate(g.nodes)}
    obs_i = {o: k for k, o in enumerate(g.observed)}
    node_emb, labeled, corr_emb = masked_inputs(gt, mask_obs_idx)
    delta = torch.zeros(gt["n"], model.head.in_features, device=DEVICE, requires_grad=True)
    out = model(gt, node_emb, labeled, corr_emb, h_delta=delta)
    Tn = torch.nn.functional.normalize(gt["T"], dim=1).to(DEVICE)
    masked = set(mask_obs_idx)
    vs = []
    for L in lat_names:
        terms = []
        for c in g.observed_descendants(L):
            k = obs_i[c]
            if k in masked:
                continue                                             # visible labels only
            # path sign = product of edge-weight signs climbing the (tree) parent chain c -> L
            sgn, node = 1.0, c
            while node != L:
                ps = g.parents(node)
                if not ps:
                    break
                p = ps[0]
                w = float(gt["Wmat"][nidx[node], nidx[p]])
                sgn *= 1.0 if w >= 0 else -1.0
                node = p
            terms.append(sgn * (out[nidx[c]] @ Tn[k]))
        if not terms:
            vs.append(np.zeros(gt["T"].shape[1]))
            continue
        score = torch.stack(terms).sum()
        ghid = torch.autograd.grad(score, delta, retain_graph=True)[0][nidx[L]]     # [hid]
        gvec = model.head.weight @ ghid                  # push hidden sensitivity into embedding space
        vs.append(gvec.detach().cpu().numpy().astype(np.float64))
    return np.stack(vs)


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
        if LAM_GEN > 0:
            ne_, lb_, ce_ = masked_inputs(gt, mask)
            out, hid = model(gt, ne_, lb_, ce_, return_hidden=True)
        else:
            out = masked_forward(model, gt, mask)
        pos = gt["obs_pos"][mask].to(DEVICE)
        target = gt["T"][mask].to(DEVICE)
        loss = (1 - F.cosine_similarity(out[pos], target, dim=1)).mean()
        loss = loss + 0.1 * gen_consistency(out, gt)
        if LAM_GEN > 0:                                              # latent generation head
            g_ = gt["g"]
            nidx = {n_: i2 for i2, n_ in enumerate(g_.nodes)}
            preds, tgts = [], []
            for k in mask:
                c = g_.observed[k]
                ps = [p for p in g_.parents(c) if g_.is_latent(p)]
                if not ps:
                    continue
                p = ps[0]
                w = float(gt["Wmat"][nidx[c], nidx[p]])
                sgn = 1.0 if w >= 0 else -1.0
                preds.append(sgn * model.gen(hid[nidx[p]]))
                tgts.append(gt["T"][k].to(DEVICE))
            if preds:
                Pd, Tg = torch.stack(preds), torch.stack(tgts)
                loss = loss + LAM_GEN * (1 - F.cosine_similarity(Pd, Tg, dim=1)).mean()
        On = F.normalize(out, dim=1)
        if LAM_INDEP > 0 and len(gt["zp"][0]):                       # d-separated pairs decorrelate
            za, zb = gt["zp"][0].to(DEVICE), gt["zp"][1].to(DEVICE)
            loss = loss + LAM_INDEP * (((On[za] * On[zb]).sum(1)) ** 2).mean()
        if LAM_RES > 0:                                              # residual cos ~ shrunk partial corr
            gen_all = gt["Wmat"].to(DEVICE) @ out
            op = gt["obs_pos"].to(DEVICE)
            Rn = F.normalize(out[op] - gen_all[op], dim=1)
            Pt = gt["Ppc"].to(DEVICE)
            off = ~torch.eye(len(op), dtype=torch.bool, device=DEVICE)
            loss = loss + LAM_RES * (((Rn @ Rn.T) - Pt)[off] ** 2).mean()
        if LAM_COLL > 0 and gt["vst"]:                               # explaining away at v-structures
            cl = 0.0
            for i1, i2, ic in gt["vst"]:
                u1 = On[i1] - (On[i1] @ On[ic]) * On[ic]
                u2 = On[i2] - (On[i2] @ On[ic]) * On[ic]
                cl = cl + torch.relu((u1 @ u2) / (u1.norm() * u2.norm() + 1e-9)) ** 2
            loss = loss + LAM_COLL * cl / len(gt["vst"])
        if LAM_DEP > 0 and len(gt["tp"][0]):                         # faithfulness dependence floor
            ta, tb = gt["tp"][0].to(DEVICE), gt["tp"][1].to(DEVICE)
            fl = gt["tp"][2].to(DEVICE)
            loss = loss + LAM_DEP * (torch.relu(fl - (On[ta] * On[tb]).sum(1).abs()) ** 2).mean()
        if LAM_MB > 0:                                               # Markov-blanket locality consistency
            k = int(mask[int(rng.integers(len(mask)))])
            outside = [j for j in range(n_obs) if j not in set(gt["mb_obs"][k])]
            out_mb = masked_forward(model, gt, sorted(set(mask) | set(outside)))
            pk = gt["obs_pos"][k].to(DEVICE)
            loss = loss + LAM_MB * (1 - F.cosine_similarity(out[pk], out_mb[pk], dim=0))
        if LAM_JAC > 0:                                              # Jacobian locality (soft MB):
            # for one sampled masked target i, the gradient of its prediction wrt the INPUT labels of
            # nodes marginally independent of i should vanish (double backprop; probe direction random)
            k = int(mask[int(rng.integers(len(mask)))])
            g_ = gt["g"]
            nidx = {n_: i2 for i2, n_ in enumerate(g_.nodes)}
            tgt_name = g_.observed[k]
            indep = gt["indep_of"].get(tgt_name, [])
            if indep:
                node_emb2, labeled2, corr_emb2 = masked_inputs(gt, mask)
                node_emb2 = node_emb2.detach().requires_grad_(True)
                out2 = model(gt, node_emb2, labeled2, corr_emb2)
                probe = torch.randn(out2.shape[1], device=DEVICE)
                probe = probe / probe.norm()
                s = out2[gt["obs_pos"][k].to(DEVICE)] @ probe
                Jrow = torch.autograd.grad(s, node_emb2, create_graph=True)[0]
                ipos = torch.tensor([nidx[n_] for n_ in indep], dtype=torch.long, device=DEVICE)
                loss = loss + LAM_JAC * (Jrow[ipos] ** 2).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 200 == 0 or step == STEPS - 1:
            print(f"[{ts()}]   step {step}/{STEPS} loss={float(loss):.4f} ({gt['name']})", flush=True)
    os.makedirs(os.path.dirname(CKPT), exist_ok=True)
    torch.save({"state": model.state_dict(), "d": d, "hid": HID, "layers": LAYERS}, CKPT)
    print(f"[{ts()}] saved {CKPT}", flush=True)


def family_subds(ds, root):
    """Sub-testbed induced by one root latent's family: the root, every latent below it, and all their
    observed descendants. Brings an unseen big graph back into the size range the operator was trained
    on — Markov-blanket-style LOCAL INFERENCE (decomposition at eval time, not a training loss)."""
    import graph as G
    g = ds["graph"]
    lats = [L for L in g.latents if L == root or root in g.ancestors(L)]
    obs = [o for o in g.observed if set(g.ancestors(o)) & set(lats)]
    keep = set(lats) | set(obs)
    edges = [(a, b) for a, b in g.edges if a in keep and b in keep]
    oi_full = {o: k for k, o in enumerate(g.observed)}
    X = ds["X"][:, [oi_full[o] for o in obs]]
    sub = dict(ds)
    sub["graph"] = G.Graph(lats, obs, edges)
    sub["X"] = X
    return sub


def evaluate_local(group="heldout", folds=5):
    """Zero-shot eval with per-family local inference: predictions for each masked item come from the
    forward pass on its root-family subgraph only. Same folds/metrics as evaluate()."""
    ck = torch.load(CKPT, map_location=DEVICE)
    model = CompletionGNN(ck["d"], ck["hid"], ck["layers"]).to(DEVICE)
    model.load_state_dict(ck["state"], strict=False); model.eval()
    names = pool.HELDOUT if group == "heldout" else (pool.DEV if group == "dev" else [group])
    out = {}
    for n in names:
        ds = ALL_LOADERS[n]()
        g = ds["graph"]
        obs_full = g.observed
        roots = [L for L in g.latents if not g.parents(L)]
        fams = []
        for r in roots:
            sub = family_subds(ds, r)
            fams.append((set(sub["graph"].observed), sub, graph_tensors(sub)))
        T = encode.embed([ds["labels"][o] for o in obs_full])
        Tn = metrics.norm_rows(T)
        rng = np.random.default_rng(0)
        perm = rng.permutation(len(obs_full))
        cs, ms = [], []
        for f in range(folds):
            mask_full = sorted(int(i) for i in perm[f::folds])
            masked_names = {obs_full[i] for i in mask_full}
            pred = {}
            for fam_obs, sub, gt in fams:
                fam_masked = [k for k, o in enumerate(sub["graph"].observed) if o in masked_names]
                if not fam_masked:
                    continue
                with torch.no_grad():
                    o_ = masked_forward(model, gt, fam_masked)
                P_ = o_[gt["obs_pos"][fam_masked].to(DEVICE)].cpu().numpy().astype(np.float64)
                for r_, k in enumerate(fam_masked):
                    pred[sub["graph"].observed[k]] = P_[r_]
            P = np.stack([pred[obs_full[i]] for i in mask_full])
            cs.append(float(np.mean((metrics.norm_rows(P) * Tn[mask_full]).sum(1))))
            ms.append(metrics.match_acc(P, mask_full, T))
        out[n] = {"cos": float(np.mean(cs)), "match": float(np.mean(ms))}
        print(f"[{ts()}] gnn-local {n:10s} cos={out[n]['cos']:.3f} match={out[n]['match']:.3f}",
              flush=True)
    dst = os.environ.get("GNN_EVAL_OUT", os.path.join(HERE, "outputs", f"gnn_local_{group}.json"))
    json.dump(out, open(dst, "w"), indent=1)
    return out


def evaluate(group="heldout", folds=5):
    ck = torch.load(CKPT, map_location=DEVICE)
    model = CompletionGNN(ck["d"], ck["hid"], ck["layers"]).to(DEVICE)
    model.load_state_dict(ck["state"], strict=False); model.eval()
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
    dst = os.environ.get("GNN_EVAL_OUT", os.path.join(HERE, "outputs", f"gnn_eval_{group}.json"))
    json.dump(out, open(dst, "w"), indent=1)
    return out


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"
    if cmd == "train":
        train()
    elif cmd == "eval_local":
        evaluate_local(sys.argv[2] if len(sys.argv) > 2 else "heldout")
    else:
        evaluate(sys.argv[2] if len(sys.argv) > 2 else "heldout")
