"""Permuted-target control for the LLM judge.

The judge is a VERIFICATION task: it is shown the decoded words and one target, and asked whether
they refer to it. A verification judge can score high without discriminating, because any plausible
connection supports a yes. The prompt also leans lenient on purpose ("synonyms count", "do not
reject merely for noise words", "answer no ONLY if ...").

This measures the resulting false-positive rate directly. For each Task 2 record, the same decoded
words are re-judged against a DIFFERENT latent of the same dataset (the mismatched target is drawn
deterministically, one shift along the dataset's own latents). A judge that discriminates should
say yes on the true pairing and no on the shifted one. Reported per dataset:

  true-pair ACC   = the number the pipeline reports
  shifted ACC     = false-positive rate; near zero is what a usable metric looks like
  discrimination  = true-pair ACC minus shifted ACC

Task 1 gets the harder version of the same control. Its failures are sibling swaps, so the shifted
target is another observed variable UNDER THE SAME LATENT. A judge that cannot separate siblings
would score high there, and Task 1 accuracy would then be measuring family membership rather than
identity. The uniform arm is reported next to core, because a high uniform score is the other
symptom of a metric that does not discriminate.

Env: DATASETS=dass,wvs,nhanes  (records read from v6/outputs/t{1,2}_<name>_v6cert.json)
"""
import ast
import random
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import judge as judge_mod                                              # noqa: E402

OUT = os.path.join(HERE, "outputs", "judge_control.json")


def load(name):
    path = os.path.join(HERE, "outputs", f"t2_{name}_v6cert.json")
    recs = [r for r in json.load(open(path))["records"] if r.get("arm") == "core"]
    for r in recs:
        w = r["decoded_words"]
        r["decoded_words"] = ast.literal_eval(w) if isinstance(w, str) else w
        r["judge"] = str(r["judge"]).lower().startswith("t")
    return recs


def load_t1(name):
    path = os.path.join(HERE, "outputs", f"t1_{name}_v6cert.json")
    recs = json.load(open(path))["records"]
    for r in recs:
        w = r["decoded_words"]
        r["decoded_words"] = ast.literal_eval(w) if isinstance(w, str) else w
        r["judge"] = str(r["judge"]).lower().startswith("t")
    return recs


def t1_control(name, report):
    """Sibling-shifted control: same decoded words, but the target is another item of the SAME
    latent. This is the hard confusion Task 1 actually makes."""
    import pool_ext
    recs = load_t1(name)
    core = [r for r in recs if r.get("arm") == "core"]
    unif = [r for r in recs if r.get("arm") == "uniform"]
    if not core:
        return
    ds = pool_ext.LOADERS[name]()
    g, labels = ds["graph"], ds["labels"]
    parent = {c: L for L in g.latents for c in g.children(L) if not g.is_latent(c)}
    text_to_var = {v: k for k, v in labels.items()}
    rng = random.Random(0)
    # a fixed sibling would bias the result: whichever item is semantically most central in a
    # family would match every family-level decode. Draw one at random per record instead, and add
    # a cross-latent arm to separate "judges the family" from "judges nothing".
    sib_items, cross_items, kept = [], [], []
    all_obs = [o for o in g.observed if o in parent]
    for r in core:
        var = text_to_var.get(r["true_label"])
        if var is None or var not in parent:
            continue
        sibs = [s for s in g.children(parent[var]) if s != var]
        outs = [o for o in all_obs if parent[o] != parent[var]]
        if not sibs or not outs:
            continue
        sib_items.append((r["decoded_words"], labels[rng.choice(sorted(sibs))]))
        cross_items.append((r["decoded_words"], labels[rng.choice(sorted(outs))]))
        kept.append(r)
    if not kept:
        return
    vs = [v for v in judge_mod.judge_batch(sib_items, "completion") if v is not None]
    vx = [v for v in judge_mod.judge_batch(cross_items, "completion") if v is not None]
    acc_c = sum(r["judge"] for r in core) / len(core)
    acc_s = sum(vs) / max(len(vs), 1)
    acc_x = sum(vx) / max(len(vx), 1)
    acc_u = (sum(r["judge"] for r in unif) / len(unif)) if unif else float("nan")
    report[f"{name}_t1"] = {"n": len(kept), "core_acc": acc_c, "sibling_shifted_acc": acc_s,
                            "cross_latent_shifted_acc": acc_x, "uniform_acc": acc_u,
                            "sibling_discrimination": acc_c - acc_s,
                            "family_discrimination": acc_c - acc_x}
    print(f"[{name} T1] n={len(kept)}  core {acc_c:.3f} | sibling {acc_s:.3f} | "
          f"cross-latent {acc_x:.3f} | uniform {acc_u:.3f}  ==> sibling-disc {acc_c - acc_s:+.3f}, "
          f"family-disc {acc_c - acc_x:+.3f}", flush=True)


def main():
    names = os.environ.get("DATASETS", "dass,wvs,nhanes").split(",")
    report = {}
    for name in names:
        recs = load(name)
        gt_of = {}
        for r in recs:
            gt_of.setdefault(r["latent"], r["gt"])
        lats = sorted(gt_of)
        if len(lats) < 2:
            print(f"[{name}] only one latent, no shifted target available", flush=True)
            continue
        shift = {L: lats[(i + 1) % len(lats)] for i, L in enumerate(lats)}

        true_items = [(r["decoded_words"], r["gt"]) for r in recs]
        wrong_items = [(r["decoded_words"], gt_of[shift[r["latent"]]]) for r in recs]
        vt = judge_mod.judge_batch(true_items, "latent")
        vw = judge_mod.judge_batch(wrong_items, "latent")
        vt = [v for v in vt if v is not None]
        vw = [v for v in vw if v is not None]
        acc_t = sum(vt) / max(len(vt), 1)
        acc_w = sum(vw) / max(len(vw), 1)
        report[name] = {"n": len(recs), "latents": len(lats), "true_acc": acc_t,
                        "shifted_acc": acc_w, "discrimination": acc_t - acc_w}
        print(f"[{name} T2] n={len(recs)} latents={len(lats)}  true {acc_t:.3f} | "
              f"shifted {acc_w:.3f} | discrimination {acc_t - acc_w:+.3f}", flush=True)
    for name in names:
        try:
            t1_control(name, report)
        except Exception as e:
            print(f"[{name} T1] control failed: {type(e).__name__}: {e}", flush=True)
    json.dump(report, open(OUT, "w"), indent=1)
    print(f"[saved {OUT}]", flush=True)


if __name__ == "__main__":
    main()
