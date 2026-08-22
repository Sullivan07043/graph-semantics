"""Continuous-prefix variant (Track B): map the solved causal embedding to soft
prefix tokens of a frozen Qwen3-4B-Instruct-2507 (ClipCap-style; the direct analog
of prefix-tuning with the prefix conditioned on our embedding).

Pairs come from the merged-protocol dumps: every node's embedding is taken from the
fold where its label was hidden (emb_v2/<ds>_<src>_emb.npz), labels from the joint
records. Discipline is leave-one-dataset-out: the mapper never sees the eval
dataset (either source).

Modes (env MODE):
  train  EVAL_DS=<ds> OUT=<ckpt>          train on all datasets except EVAL_DS
  eval   EVAL_DS=<ds> CKPT=<ckpt> SRC=given|disc OUT=<records json>
         arms: prefixonly (soft prefix + bare instruction) and
               prefixgraph (soft prefix + the same CAUSAL NEIGHBORS text block)
K soft tokens (default 8), mapper = MLP 1024 -> 2048 -> K*hidden, frozen LM.
"""
import glob
import json
import os
import re
import sys

import numpy as np
import torch
import torch.nn as nn

_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PKG)
sys.path.insert(0, os.path.join(_PKG, "plugin"))
import config  # noqa: E402

MODEL_ID = os.environ.get("QWEN_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
K = int(os.environ.get("K_SOFT", 8))
DEV = "cuda:0"
INSTR = ("The prefix encodes evidence about one masked variable of a dataset. "
         "Name what the variable measures. Reply with one short label only.")


def load_lm():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cache = os.environ.get("HF_CACHE", "/data2/shuhao/hf_cache")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=cache)
    lm = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, cache_dir=cache, torch_dtype=torch.bfloat16, device_map=DEV)
    lm.eval()
    for p in lm.parameters():
        p.requires_grad_(False)
    return lm, tok


class Mapper(nn.Module):
    def __init__(self, d_in, d_lm, k):
        super().__init__()
        self.k = k
        self.net = nn.Sequential(nn.Linear(d_in, 2048), nn.GELU(),
                                 nn.Linear(2048, k * d_lm))

    def forward(self, x):                       # [B, d_in] -> [B, k, d_lm]
        return self.net(x).view(x.shape[0], self.k, -1)


ROBOTS = ["liftbody", "bodysawyer", "bodyiiwa", "bodyur5e"]
ROBOT_GRAPHS = {("liftbody", "boss"): "lift_body_summary.json",
                ("liftbody", "truev3"): "lift_body_true_summary_v3.json",
                ("bodysawyer", "boss"): "body_sawyer_boss_summary.json",
                ("bodysawyer", "truev3"): "body_sawyer_true_summary_v3.json",
                ("bodyiiwa", "boss"): "body_iiwa_boss_summary.json",
                ("bodyiiwa", "truev3"): "body_iiwa_true_summary_v3.json",
                ("bodyur5e", "boss"): "body_ur5e_boss_summary.json",
                ("bodyur5e", "truev3"): "body_ur5e_true_summary_v3.json"}


def robot_label_maps():
    """(tgt, src) -> {channel -> label} from the robot base-stack records."""
    out = {}
    for tgt in ROBOTS:
        for src in ("boss", "truev3"):
            p = os.path.join(config.REC_T3, f"t1_{tgt}_base_{src}.json")
            if not os.path.exists(p):
                continue
            recs = json.load(open(p))["records"]
            out[(tgt, src)] = {r["var"]: r["true_label"]
                               for r in recs if r["arm"] == "core"}
    return out


def label_maps():
    """(ds, src) -> {npz key -> label text} from the joint records."""
    out = {}
    for p in sorted(glob.glob(os.path.join(config.REC_JOINT, "t12_*.json"))):
        m = re.match(r"t12_(\w+)_(given|disc)\.json", os.path.basename(p))
        ds, src = m.groups()
        d = json.load(open(p))
        recs = d["records"] if isinstance(d, dict) and "records" in d else d
        lab = {}
        for r in recs:
            if r["arm"] != "core":
                continue
            if "var" in r:
                lab[r["var"]] = r["true_label"]
            else:
                for f_ in range(5):
                    if r.get("gt"):
                        lab[f'{r["latent"]}@f{f_}'] = r["gt"]
        out[(ds, src)] = lab
    return out


def pairs_for(ds_list, labmaps):
    X, Y = [], []
    for (ds, src), lab in labmaps.items():
        if ds not in ds_list:
            continue
        pref = "robot_" if os.environ.get("DOMAIN") == "robot" else ""
        npz = os.path.join(config.DISC, "outputs", "emb_v2", f"{pref}{ds}_{src}_emb.npz")
        if not os.path.exists(npz):
            continue
        d = np.load(npz)
        for n, v in zip(d["names"], d["vecs"]):
            n = str(n)
            if n in lab and lab[n]:
                X.append(v)
                Y.append(lab[n])
    return np.stack(X), Y


def splice(tok, lm, soft, suffix_text, device):
    """inputs_embeds for: <im_start>user\\n INSTR [SOFT] suffix <im_end> assistant."""
    pre = tok.apply_chat_template([{"role": "user", "content": INSTR}],
                                  tokenize=False, add_generation_prompt=False)
    pre = pre.rsplit("<|im_end|>", 1)[0]                     # keep user turn open
    post = (suffix_text or "") + "<|im_end|>\n<|im_start|>assistant\n"
    emb_layer = lm.get_input_embeddings()
    ia = tok(pre, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    ib = tok(post, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    ea, eb = emb_layer(ia), emb_layer(ib)
    return torch.cat([ea, soft.unsqueeze(0).to(ea.dtype), eb], dim=1), ia.shape[1]


def train():
    eval_ds = os.environ["EVAL_DS"]
    out = os.environ["OUT"]
    lm, tok = load_lm()
    labmaps = robot_label_maps() if os.environ.get("DOMAIN") == "robot" else label_maps()
    all_ds = sorted({ds for ds, _ in labmaps})
    X, Y = pairs_for([d for d in all_ds if d != eval_ds], labmaps)
    print(f"[train] eval_ds={eval_ds} pairs={len(Y)}")
    d_lm = lm.get_input_embeddings().weight.shape[1]
    mp = Mapper(1024, d_lm, K).to(DEV).to(torch.float32)
    opt = torch.optim.AdamW(mp.parameters(), lr=1e-4)
    emb_layer = lm.get_input_embeddings()
    EPOCHS = int(os.environ.get("EPOCHS", 4))
    B = 8
    rng = np.random.default_rng(0)
    for ep in range(EPOCHS):
        order = rng.permutation(len(Y))
        tot, nb = 0.0, 0
        for s in range(0, len(order), B):
            idx = order[s:s + B]
            opt.zero_grad()
            loss_acc = 0.0
            for i in idx:
                soft = mp(torch.tensor(X[i], device=DEV).unsqueeze(0))[0]
                ie, _ = splice(tok, lm, soft, "", DEV)
                yt = tok(Y[i] + "<|im_end|>", return_tensors="pt",
                         add_special_tokens=False).input_ids.to(DEV)
                ye = emb_layer(yt)
                full = torch.cat([ie, ye], dim=1)
                out_ = lm(inputs_embeds=full).logits
                lg = out_[0, ie.shape[1] - 1:-1]
                loss = nn.functional.cross_entropy(lg.float(), yt[0])
                (loss / len(idx)).backward()
                loss_acc += float(loss)
            opt.step()
            tot += loss_acc / len(idx)
            nb += 1
        print(f"[train] epoch {ep + 1}/{EPOCHS} loss {tot / nb:.3f}", flush=True)
    torch.save({"state": mp.state_dict(), "k": K, "d_lm": d_lm}, out)
    print(f"[saved {out}]")


def evaluate():
    eval_ds = os.environ["EVAL_DS"]
    src = os.environ["SRC"]
    out = os.environ["OUT"]
    lm, tok = load_lm()
    ck = torch.load(os.environ["CKPT"], map_location=DEV)
    mp = Mapper(1024, ck["d_lm"], ck["k"]).to(DEV).to(torch.float32)
    mp.load_state_dict(ck["state"])
    mp.eval()
    robot = os.environ.get("DOMAIN") == "robot"
    labmaps = robot_label_maps() if robot else label_maps()
    lab = labmaps[(eval_ds, src)]
    npz = np.load(os.path.join(config.DISC, "outputs", "emb_v2",
                               (f"robot_{eval_ds}_{src}_emb.npz"
                                if os.environ.get("DOMAIN") == "robot"
                                else f"{eval_ds}_{src}_emb.npz")))
    # graph text block per node, reused from the discrete plugin's evidence
    import evidence
    if robot:
        rlabels, W, T = evidence.load_robot(eval_ds, ROBOT_GRAPHS[(eval_ds, src)])
        recs_r = json.load(open(os.path.join(config.REC_T3,
                                             f"t1_{eval_ds}_base_{src}.json")))["records"]
        fold_of_r = {r["var"]: r["fold"] for r in recs_r if r["arm"] == "core"}
        masked_r = {}
        for r in recs_r:
            if r["arm"] == "core":
                masked_r.setdefault(r["fold"], set()).add(r["var"])
        rows = []
        with torch.no_grad():
            for n, v in zip(npz["names"], npz["vecs"]):
                n = str(n)
                if n not in lab:
                    continue
                fno = fold_of_r.get(n, 0)
                lines = evidence.r_channel_lines(n, rlabels, W, T,
                                                 masked_r.get(fno, set()))
                soft = mp(torch.tensor(v, device=DEV).unsqueeze(0))[0]
                for arm, suffix in (("prefixonly", ""),
                                    ("prefixgraph",
                                     "\nCAUSAL NEIGHBORS:\n" + "\n".join(lines)
                                     if lines else "")):
                    ie, _ = splice(tok, lm, soft, suffix, DEV)
                    am = torch.ones(ie.shape[:2], dtype=torch.long, device=ie.device)
                    gen = lm.generate(inputs_embeds=ie, attention_mask=am,
                                      max_new_tokens=24, do_sample=False,
                                      pad_token_id=tok.eos_token_id)
                    txt = tok.decode(gen[0], skip_special_tokens=True).strip()
                    txt = txt.splitlines()[0].strip().strip('"') if txt else None
                    rows.append({"task": 1, "dataset": f"{eval_ds}_pfx_{src}",
                                 "fold": fno, "arm": arm, "var": n,
                                 "true_label": lab[n],
                                 "decoded_words": ([txt] if txt else None),
                                 "judge": None, "model": f"prefix:{MODEL_ID}"})
        json.dump({"records": rows}, open(out, "w"), indent=1)
        print(f"[saved {out} ({len(rows)} rows)]")
        return
    g, labels, _ = evidence.load_questionnaire(eval_ds, src)
    jrec = json.load(open(os.path.join(config.REC_JOINT,
                                       f"t12_{eval_ds}_{src}.json")))["records"]
    fold_of = {r["var"]: r["fold"] for r in jrec if r["arm"] == "core" and "var" in r}
    masked_of = {}
    for r in jrec:
        if r["arm"] == "core" and "var" in r:
            masked_of.setdefault(r["fold"], set()).add(r["var"])
    rows = []
    with torch.no_grad():
        for n, v in zip(npz["names"], npz["vecs"]):
            n = str(n)
            if n not in lab or not lab[n]:
                continue
            is_lat = "@f" in n
            if is_lat:
                base_n, fno = n.split("@f")[0], int(n.split("@f")[1])
                lines = [f'- EFFECT: it causes the item '
                         f'"{"(label hidden)" if c in masked_of.get(fno, set()) else labels[c]}"'
                         for c in g.children(base_n) if not g.is_latent(c)][:20] \
                    if base_n in (g.latents or []) else []
            else:
                fno = fold_of.get(n, 0)
                lines = evidence.q_item_lines(g, n, labels, masked_of.get(fno, set()), {})
            soft = mp(torch.tensor(v, device=DEV).unsqueeze(0))[0]
            for arm, suffix in (("prefixonly", ""),
                                ("prefixgraph",
                                 "\nCAUSAL NEIGHBORS:\n" + "\n".join(lines) if lines else "")):
                ie, _ = splice(tok, lm, soft, suffix, DEV)
                am = torch.ones(ie.shape[:2], dtype=torch.long, device=ie.device)
                gen = lm.generate(inputs_embeds=ie, attention_mask=am,
                                  max_new_tokens=24, do_sample=False,
                                  pad_token_id=tok.eos_token_id)
                txt = tok.decode(gen[0], skip_special_tokens=True).strip()
                txt = txt.splitlines()[0].strip().strip('"') if txt else None
                row = {"task": 1, "dataset": f"{eval_ds}_pfx_{src}", "fold": fno,
                       "arm": arm, "decoded_words": ([txt] if txt else None),
                       "judge": None, "model": f"prefix:{MODEL_ID}"}
                if is_lat:
                    row["latent"] = f"{base_n}@f{fno}"
                    row["gt"] = lab[n]
                else:
                    row["var"] = n
                    row["true_label"] = lab[n]
                rows.append(row)
    json.dump({"records": rows}, open(out, "w"), indent=1)
    print(f"[saved {out} ({len(rows)} rows)]")


if __name__ == "__main__":
    (train if os.environ.get("MODE", "train") == "train" else evaluate)()
