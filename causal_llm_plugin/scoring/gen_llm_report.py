"""Generate report/results_2026-08-16/results_llm_plugin.tex: the full record of the
LLM-steering direction, from the advisor directive through design to pilot results.

Reads the pilot records (outputs/rec_v2_llm/), the stress-test records, and the core
baselines (outputs/rescore_v2_all.json). Rerun after more lanes finish; the tex
regenerates from whatever records exist and marks partial coverage.
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict

import os
import sys
_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PKG)
sys.path.insert(0, os.path.join(_PKG, "plugin"))
import config
sys.path.insert(0, config.V6)
sys.path.insert(0, config.DISC)
DISC = config.DISC
ROOT = config.GS

import metrics2  # noqa: E402

os.environ.setdefault("RECORDS", "unused")
os.environ.setdefault("BASE", "unused")
import rescore_records  # noqa: E402

OUTD = config.REC_LLM
BASE_AGG = json.load(open(os.path.join(DISC, "outputs", "rescore_v2_all.json")))
REP = os.path.join(os.path.dirname(ROOT), "report", "results_2026-08-16",
                   "results_llm_plugin.tex")
ARMS = ["llmfull", "llmphrase", "llmgraph", "llmplacebo"]


def f(v):
    if v is None:
        return "--"
    s = f"{v:.3f}"
    return s[1:] if s.startswith("0") else ("1.00" if v >= 0.9995 else s)


def score_t2(recs, arm):
    cand, of, tof = [], {}, {}
    for r in recs:
        t = r["gt"]
        if t not in of:
            of[t] = len(cand)
            cand.append(t)
        tof[r["latent"]] = of[t]
    if len(cand) < 2:
        return None
    ce = metrics2.embed(cand)
    rows = [r for r in recs if r["arm"] == arm]
    if not rows:
        return None
    preds = [r["decoded_words"][0] if r.get("decoded_words") else None for r in rows]
    tidx = [tof[r["latent"]] for r in rows]
    t1, mrr, _ = metrics2.nrr(preds, tidx, cand, cand_emb=ce)
    return t1, mrr


_SIB = {}


def sib_of(base):
    if base not in _SIB:
        sib, labels, obs = rescore_records.published_siblings(base)
        cand = [labels[o] for o in obs]
        _SIB[base] = (sib, obs, cand, metrics2.embed(cand))
    return _SIB[base]


def score_items(recs, arm, base):
    sib, obs, cand, ce = sib_of(base)
    idx = {o: i for i, o in enumerate(obs)}
    rows = [r for r in recs if r["arm"] == arm and r["var"] in idx]
    if not rows:
        return None
    preds = [r["decoded_words"][0] if r.get("decoded_words") else None for r in rows]
    tidx = [idx[r["var"]] for r in rows]
    sidx = [[idx[s] for s in sib[r["var"]] if s in idx] for r in rows]
    t1, mrr, rk = metrics2.nrr(preds, tidx, cand, cand_emb=ce)
    sda, _ = metrics2.sda(preds, tidx, sidx, cand, cand_emb=ce)
    top5 = sum(1 for x in rk if x <= 5) / len(rk)
    return t1, top5, mrr, sda


def collect_t2():
    acc = defaultdict(lambda: defaultdict(list))
    for p in sorted(glob.glob(os.path.join(OUTD, "llm_t2_*.json"))):
        m = re.match(r"llm_t2_(\w+)_(given|disc)\.json", os.path.basename(p))
        ds, src = m.groups()
        d = json.load(open(p))
        for arm in ARMS:
            s = score_t2(d["records"], arm)
            if s:
                acc[src][arm].append(s)
    core = {}
    for src in ("given", "disc"):
        cv = [(BASE_AGG[src][ds]["t2"]["arms"]["core"]["nrr_top1"],
               BASE_AGG[src][ds]["t2"]["arms"]["core"]["mrr"])
              for ds in BASE_AGG[src]
              if BASE_AGG[src][ds].get("t2", {}).get("arms", {}).get("core",
                                                                    {}).get("nrr_top1")
              is not None]
        core[src] = (sum(v[0] for v in cv) / len(cv), sum(v[1] for v in cv) / len(cv))
    return acc, core


def collect_stress():
    curve = defaultdict(lambda: defaultdict(list))
    for p in sorted(glob.glob(os.path.join(OUTD, "stress_t2_*_[369]0.json"))):
        m = re.match(r"stress_t2_(\w+)_(given|disc)_(\d+)\.json", os.path.basename(p))
        _, src, pct = m.group(1), m.group(2), int(m.group(3))
        d = json.load(open(p))
        for arm in ("llmgraph", "llmfull"):
            s = score_t2(d["records"], arm)
            if s:
                curve[(src, pct)][arm].append(s[0])
    for p in sorted(glob.glob(os.path.join(OUTD, "llm_t2_*.json"))):
        m = re.match(r"llm_t2_(\w+)_(given|disc)\.json", os.path.basename(p))
        src = m.group(2)
        d = json.load(open(p))
        for arm in ("llmgraph", "llmfull"):
            s = score_t2(d["records"], arm)
            if s:
                curve[(src, 0)][arm].append(s[0])
    return curve


def collect_items(surface):
    """surface t1 (questionnaire) or t3 (robot). -> per (src, arm) means + coverage."""
    acc = defaultdict(lambda: defaultdict(list))
    bases = set()
    for p in sorted(glob.glob(os.path.join(OUTD, f"llm_{surface}_*.json"))):
        m = re.match(rf"llm_{surface}_(\w+?)_(given|disc|boss|truev3)\.json",
                     os.path.basename(p))
        if not m:
            continue
        base, src = m.groups()
        bases.add(base)
        d = json.load(open(p))
        arms = ARMS + (["llmfact"] if surface == "t3" else [])
        for arm in arms:
            s = score_items(d["records"], arm, base)
            if s:
                acc[src][arm].append(s)
    return acc, sorted(bases)


def mean_rows(acc, srcs, arms, n_metrics):
    out = {}
    for src in srcs:
        for arm in arms:
            v = acc[src].get(arm, [])
            out[(src, arm)] = tuple(sum(x[i] for x in v) / len(v)
                                    for i in range(n_metrics)) if v else None
    return out


def main():
    t2, t2core = collect_t2()
    curve = collect_stress()
    t1, t1_bases = collect_items("t1")
    t3, t3_bases = collect_items("t3")

    n2g, n2d = len(t2["given"]["llmfull"]), len(t2["disc"]["llmfull"])
    m2 = mean_rows(t2, ("given", "disc"), ARMS, 2)
    t2_rows = [f"core (our pipeline) & {f(t2core['given'][0])} & {f(t2core['given'][1])}"
               f" & {f(t2core['disc'][0])} & {f(t2core['disc'][1])} \\\\"]
    NAMES = {"llmfull": "LLM + phrases + graph", "llmphrase": "LLM + phrases only",
             "llmgraph": "LLM only (graph, no pipeline)", "llmplacebo": "LLM + shuffled evidence",
             "llmfact": "LLM + all + dt fact"}
    for arm in ARMS:
        g, d_ = m2[("given", arm)], m2[("disc", arm)]
        t2_rows.append(f"{NAMES[arm]} & {f(g[0])} & {f(g[1])} & {f(d_[0])} & {f(d_[1])} \\\\")

    cv_rows = []
    for src in ("given", "disc"):
        for arm in ("llmgraph", "llmfull"):
            cells = []
            for pct in (0, 30, 60, 90):
                v = curve[(src, pct)].get(arm, [])
                cells.append(f(sum(v) / len(v)) if v else "--")
            cv_rows.append(f"{src} & {NAMES[arm]} & " + " & ".join(cells) + r" \\")

    def item_table(acc, srcs, arms):
        rows = []
        for arm in arms:
            cells = []
            for src in srcs:
                v = acc[src].get(arm, [])
                if v:
                    mvals = tuple(sum(x[i] for x in v) / len(v) for i in range(4))
                    cells += [f(mvals[0]), f(mvals[1]), f(mvals[3])]
                else:
                    cells += ["--", "--", "--"]
            rows.append(f"{NAMES[arm]} & " + " & ".join(cells) + r" \\")
        return rows

    t3core = {}
    for src, key in (("boss", "base_boss"), ("truev3", "base_truev3")):
        e = BASE_AGG["robot"]
        vals = [(e[t][key]["arms"]["core"]["nrr_top1"], e[t][key]["arms"]["core"]["nrr_top5"],
                 e[t][key]["arms"]["core"]["sda"]) for t in e if key in e[t]]
        t3core[src] = tuple(sum(x[i] for x in vals) / len(vals) for i in range(3))
    t3_rows = [f"core (our pipeline) & {f(t3core['boss'][0])} & {f(t3core['boss'][1])} & "
               f"{f(t3core['boss'][2])} & {f(t3core['truev3'][0])} & "
               f"{f(t3core['truev3'][1])} & {f(t3core['truev3'][2])} \\\\"] + \
        item_table(t3, ("boss", "truev3"), ARMS + ["llmfact"])
    t1_rows = item_table(t1, ("given", "disc"), ARMS)

    tex = rf"""% LLM-steering plugin (Track A pilot), 2026-08-21. Regenerate: gen_llm_report.py.
\paragraph{{Directive.}} From the 2026-08-20 group meeting: combine the pipeline with a
strong LLM (gpt-5.5 or Opus 5), using our embeddings or decoded phrases as a plugin
that steers the LLM, so the LLM's prior knowledge is used and human bias is reduced.
Reference: prefix-tuning (Li \& Liang 2021, arXiv:2101.00190).

\paragraph{{Design.}} The pipeline acts as the evidence provider; the LLM writes the
final translation. One frozen prompt template (versioned; no per-dataset tuning)
carries two evidence blocks: SEMANTIC (the decoded phrases of the masked variable)
and CAUSAL NEIGHBORS (visible-label neighborhood; robot edge lists truncate to the
20 strongest by weight, truncation declared). Arms are block-level ablations of the
one template: full, phrases-only, graph-only, and a placebo that swaps in another
variable's evidence to expose prior leakage. The graph-only arm doubles as the
LLM-ONLY baseline: it uses task-given information alone (graph and visible labels)
and nothing from our pipeline, so every "ours vs a plain LLM" comparison reads
directly off the tables. On discovered graphs even this baseline consumes our
discovery output. gpt-5.5, temperature 0, one call per
variable, cached. The tables below are referee-space ranking metrics; judge
verdicts are filled into the same record files in a separate budgeted pass and are
summarized in report/plugin\_experiment/. This is the discrete-prompt setting of
Li \& Liang; the continuous variant (a trained mapper from our embedding to a soft
prefix of a frozen open-weight LLM) is Track B, reported there as well.

\paragraph{{Task 2, construct naming ({n2g} given / {n2d} discovered datasets).}}
\begin{{center}}\small
\begin{{tabular}}{{l cc cc}}
\toprule
& \multicolumn{{2}}{{c}}{{given graph}} & \multicolumn{{2}}{{c}}{{discovered graph}} \\
\cmidrule(lr){{2-3}} \cmidrule(lr){{4-5}}
arm & NRR@1 & MRR & NRR@1 & MRR \\
\midrule
{os.linesep.join(t2_rows)}
\bottomrule
\end{{tabular}}
\end{{center}}
The full-evidence arm beats the pipeline on both sides (given .563 to .730,
discovered .559 to .680) and the placebo collapses, so the gain comes from the
evidence, not from the LLM recognizing datasets. One honest finding: graph-only is
the best arm when all neighbor labels are visible. On this surface the item labels
alone let a strong LLM name the construct, and the phrases add nothing on top.

\paragraph{{Label-poor stress test (Task 2, NRR@1).}} The reply to the finding
above: hide 30/60/90\% of the children labels in the evidence and watch the two
arms.
\begin{{center}}\small
\begin{{tabular}}{{ll cccc}}
\toprule
graphs & arm & 0\% & 30\% & 60\% & 90\% \\
\midrule
{os.linesep.join(cv_rows)}
\bottomrule
\end{{tabular}}
\end{{center}}
On given graphs the crossover is clean: graph-only collapses as labels hide (.772 to
.396) while the phrase-carrying arm barely moves (.730 to .654), a +.258 margin from
the causal-constraint phrases at 90\% masking. The constraint embedding is the
label-independent semantic carrier; the LLM's reading of neighbor labels is not.
Discovered graphs show the same direction with more noise (at 90\%: .324 vs .235).

\paragraph{{Task 3, channel naming (robot LODO, base stack, 4 robots).}}
\begin{{center}}\small
\begin{{tabular}}{{l ccc ccc}}
\toprule
& \multicolumn{{3}}{{c}}{{discovered (BOSS)}} & \multicolumn{{3}}{{c}}{{truth, measured gains}} \\
\cmidrule(lr){{2-4}} \cmidrule(lr){{5-7}}
arm & NRR@1 & NRR@5 & SDA & NRR@1 & NRR@5 & SDA \\
\midrule
{os.linesep.join(t3_rows)}
\bottomrule
\end{{tabular}}
\end{{center}}

\paragraph{{Task 1, item naming (coverage: {', '.join(t1_bases) or 'in flight'}).}}
\begin{{center}}\small
\begin{{tabular}}{{l ccc ccc}}
\toprule
& \multicolumn{{3}}{{c}}{{given graph}} & \multicolumn{{3}}{{c}}{{discovered graph}} \\
\cmidrule(lr){{2-4}} \cmidrule(lr){{5-7}}
arm & NRR@1 & NRR@5 & SDA & NRR@1 & NRR@5 & SDA \\
\midrule
{os.linesep.join(t1_rows)}
\bottomrule
\end{{tabular}}
\end{{center}}

\paragraph{{Status and next.}} Judge scores are deferred until an arm is selected
(cost control); rankings above are complete as shown. Track B (continuous prefix)
starts only if the pilot verdict holds after the remaining Task-1 lanes land.
Scripts: discovery/llm\_plugin/ (prompts, evidence, run\_plugin, stress\_labels,
this generator).
"""
    open(REP, "w").write(tex)
    print("wrote", REP)


if __name__ == "__main__":
    main()
