"""LLM naming baseline for Task 1, the arm the protocol never had.

Task 2 has always carried an LLM-naming baseline; Task 1 never did, and with the robot task
being Task-1-only the comparison "causal-constraint optimization against just asking an LLM"
was silently missing. This fills it.

Same information as core, different inference: for each masked channel the model sees its GRAPH
NEIGHBOURHOOD (parent and child labels, edge direction, sign, lag or contemporaneous type) and
writes one label. The guess is embedded with the same encoder and scored with the same fold-wise
Hungarian match and the same judge. Folds replicate run_task1 exactly (rng(0) permutation,
5 interleaved folds).

Env: DATASET=liftbody T3_GRAPH=... NAMING_MODEL=openai/gpt-4o-mini OUT=<json>
Judge env as usual. Runs in the certified venv from task3_pipeline_v1.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P1 = os.path.join(HERE, "task3_pipeline_v1")
sys.path.insert(0, P1)

DATASET = os.environ.get("DATASET", "liftbody")
MODEL = os.environ.get("NAMING_MODEL", "openai/gpt-4o-mini")
OUT = os.environ.get("OUT", os.path.join(HERE, "outputs", f"t1_naming_{DATASET}.json"))
FOLDS = 5

os.environ.setdefault("GRAPHSEM_DICT",
                      os.path.join(P1, "outputs", "concept_bank_l3_robot.npz"))


def neighbourhood(g, node, labels, hidden, signs, etypes):
    """Visible neighbour descriptions with edge direction, sign and time type."""
    lines = []
    for p in g.parents(node):
        if p in hidden:
            continue
        w = signs.get(f"{p}->{node}")
        t = etypes.get(f"{p}->{node}", "lag")
        s = "positively" if (w or 0) >= 0 else "negatively"
        lines.append(f"- CAUSE ({t}): \"{labels[p]}\" influences it {s}")
    for c in g.children(node):
        if c in hidden:
            continue
        w = signs.get(f"{node}->{c}")
        t = etypes.get(f"{node}->{c}", "lag")
        s = "positively" if (w or 0) >= 0 else "negatively"
        lines.append(f"- EFFECT ({t}): it influences \"{labels[c]}\" {s}")
    return lines


def main():
    import urllib.request

    import encode
    import metrics
    import pool_ext

    ds = pool_ext.LOADERS[DATASET]()
    g, X, labels = ds["graph"], ds["X"], ds["labels"]
    obs = list(g.observed)
    # the summary json carries signs and edge types; reload it for the prompt
    graph_file = os.environ.get("T3_GRAPH", f"body_{DATASET.replace('body','')}_boss_summary.json"
                                if DATASET.startswith("body") else "lift_body_summary.json")
    summ = json.load(open(os.path.join(HERE, "outputs", graph_file)))
    signs, etypes = summ.get("signs", {}), summ.get("edge_types", {})

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(obs))
    folds = [perm[i::FOLDS] for i in range(FOLDS)]

    key = os.environ["OPENAI_API_KEY"]
    base = os.environ.get("JUDGE_BASE_URL", "https://api.openai.com/v1")

    def ask(prompt):
        req = urllib.request.Request(
            f"{base}/chat/completions",
            data=json.dumps({"model": MODEL, "temperature": 0,
                             "messages": [{"role": "user", "content": prompt}]}).encode(),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        r = json.loads(urllib.request.urlopen(req, timeout=120).read())
        return r["choices"][0]["message"]["content"].strip().strip('"')

    T = encode.embed([labels[o] for o in obs])
    fold_match, records, guesses_all, targets_all = [], [], [], []
    for fno, fold in enumerate(folds):
        masked = sorted(int(i) for i in fold)
        hidden = {obs[i] for i in masked}
        guesses = []
        for i in masked:
            n = obs[i]
            lines = neighbourhood(g, n, labels, hidden, signs, etypes)
            ctx = "\n".join(lines) if lines else "(no visible neighbours in the causal graph)"
            prompt = (
                "A robot manipulation log has one unlabeled channel. Its causal relations to "
                "labeled channels are:\n" + ctx + "\n\n"
                "Every channel is a scalar quantity of a robot arm or its environment. Reply with "
                "ONE short label naming what this channel measures, in the same style as the "
                "labels above. Reply with the label only.")
            try:
                guesses.append(ask(prompt))
            except Exception as e:
                print(f"  [naming] fold {fno} {n}: API error {type(e).__name__}", flush=True)
                guesses.append("")
        P = encode.embed(guesses)
        m = metrics.match_acc(P, masked, T)
        fold_match.append(m)
        guesses_all += guesses
        targets_all += [labels[obs[i]] for i in masked]
        for i, gtxt in zip(masked, guesses):
            records.append({"fold": fno, "var": obs[i], "true": labels[obs[i]], "guess": gtxt})
        print(f"[naming] fold {fno + 1}/5 match={m:.3f}", flush=True)

    import judge as judge_mod
    verd = judge_mod.judge_batch(list(zip(guesses_all, targets_all)), "completion")
    ok = [v for v in (verd or []) if v is not None]
    jacc = float(np.mean(ok)) if ok else None
    res = {"dataset": DATASET, "model": MODEL, "match": float(np.mean(fold_match)),
           "match_per_fold": [round(m, 4) for m in fold_match],
           "judge": jacc, "records": records}
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"[naming] {DATASET}: match={np.mean(fold_match):.3f} "
          f"judge={jacc if jacc is None else round(jacc, 3)} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
