"""Emit the table files for report/plugin_experiment/ (generated/*.tex).

Unified layout: judge is one metric column next to NRR@1/NRR@5/MRR/SDA in every
table, never a separate table. All numbers come from compute_all.py's
all_results_unified.json, which this script refreshes first. Prose in
plugin_experiment.tex is hand-written and inputs these files.
"""
import json
import os
import sys

_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PKG)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
import compute_all  # noqa: E402

compute_all.main()
U = json.load(open(os.path.join(config.SCORES, "all_results_unified.json")))

OUT = os.path.join(os.path.dirname(config.GS), "report", "plugin_experiment", "generated")
os.makedirs(OUT, exist_ok=True)

NAMES = {"core": "pipeline alone (no LLM)", "llmfull": "LLM + plugin (main)",
         "llmphrase": "LLM + phrases only", "llmgraph": "LLM only (no plugin)",
         "llmplacebo": "LLM + shuffled evidence", "llmfact": "LLM + plugin + dt fact",
         "llmhead": "LLM + plugin + dt + naming head (main)",
         "prefixonly": "prefix only (ablation)",
         "prefixgraph": "prefix + graph text (main)"}
# On T3 the main method carries dt fact + naming head; phrases+graph is an ablation.
T3NAMES = dict(NAMES)
T3NAMES["llmfull"] = "LLM + phrases + graph"
QNAMES = {k: v.replace("LLM", "Qwen") for k, v in T3NAMES.items()}
QNAMES["llmfull"] = "Qwen + plugin (main)"

M4 = ("nrr_top1", "nrr_top5", "mrr", "judge")               # latent surfaces
M5 = ("nrr_top1", "nrr_top5", "mrr", "sda", "judge")        # item surfaces

MHEAD = {"nrr_top1": "NRR@1", "nrr_top5": "NRR@5", "mrr": "MRR",
         "sda": "SDA", "judge": "judge"}
SRCHEAD = {"given": "given graphs", "disc": "discovered graphs",
           "boss": "discovered (BOSS)", "truev3": "truth, measured gains"}


def f(v):
    if v is None:
        return "--"
    if v >= 0.9995:
        return "1.00"
    return f"{v:.3f}"[1:]


def w(name, srcs, metrics, rows):
    k = len(metrics)
    spec = "l " + " ".join(["c" * k] * len(srcs))
    head = ("& " + " & ".join(f"\\multicolumn{{{k}}}{{c}}{{{SRCHEAD[s]}}}" for s in srcs)
            + " \\\\\n"
            + "".join(f"\\cmidrule(lr){{{2 + i * k}-{1 + (i + 1) * k}}}"
                      for i in range(len(srcs))) + "\n"
            + "arm & " + " & ".join([MHEAD[m] for m in metrics] * len(srcs)) + " \\\\")
    full = ("\\begin{tabular}{" + spec + "}\n\\toprule\n" + head +
            "\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    p = os.path.join(OUT, name)
    open(p, "w").write(full)
    print("wrote", p)


def cells(table, src, arm, metrics):
    e = U.get(table, {}).get(src, {}).get(arm)
    return [f(e.get(m)) if e else "--" for m in metrics]


def two_mode(name, table, srcs, arms, metrics, nm=NAMES):
    rows = [f"{nm[arm]} & " + " & ".join(
        cells(table, srcs[0], arm, metrics) + cells(table, srcs[1], arm, metrics))
        + r" \\" for arm in arms]
    w(name, srcs, metrics, rows)


def main():
    QD = ("given", "disc")
    RB = ("boss", "truev3")
    CORE5 = ["core", "llmfull", "llmphrase", "llmgraph", "llmplacebo"]

    two_mode("table_t2.tex", "t2", QD, CORE5, M4)
    two_mode("table_t1.tex", "t1", QD, CORE5, M5)
    two_mode("table_joint_items.tex", "joint_items", QD, CORE5, M5)
    two_mode("table_joint_latents.tex", "joint_latents", QD, CORE5, M4)
    two_mode("table_t3.tex", "t3", RB,
             CORE5 + ["llmfact", "llmhead"], M5, T3NAMES)

    # stress: NRR@1 and judge across masking levels; 0% column = the T2 numbers
    rows = []
    for src in QD:
        for arm in ("llmgraph", "llmfull"):
            c = [f(U["t2"][src][arm]["nrr_top1"])] + \
                [f(U[f"stress{p}"][src][arm]["nrr_top1"]) for p in (30, 60, 90)] + \
                [f(U["t2"][src][arm]["judge"])] + \
                [f(U[f"stress{p}"][src][arm]["judge"]) for p in (30, 60, 90)]
            rows.append(f"{src} & {NAMES[arm]} & " + " & ".join(c) + r" \\")
    spec = "ll cccc cccc"
    head = ("& & \\multicolumn{4}{c}{NRR@1} & \\multicolumn{4}{c}{judge} \\\\\n"
            "\\cmidrule(lr){3-6}\\cmidrule(lr){7-10}\n"
            "graphs & arm & 0\\% & 30\\% & 60\\% & 90\\% & 0\\% & 30\\% & 60\\% & 90\\% \\\\")
    full = ("\\begin{tabular}{" + spec + "}\n\\toprule\n" + head +
            "\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    open(os.path.join(OUT, "table_stress.tex"), "w").write(full)
    print("wrote", os.path.join(OUT, "table_stress.tex"))

    # open weight
    two_mode("table_qwen_t2.tex", "qwen_t2", QD,
             ["llmfull", "llmphrase", "llmgraph", "llmplacebo"], M4, QNAMES)
    two_mode("table_qwen_t1.tex", "qwen_t1", QD,
             ["llmfull", "llmphrase", "llmgraph", "llmplacebo"], M5, QNAMES)
    two_mode("table_pfx_items.tex", "pfx_items", QD,
             ["prefixonly", "prefixgraph"], M5, QNAMES)
    two_mode("table_pfx_latents.tex", "pfx_latents", QD,
             ["prefixonly", "prefixgraph"], M4, QNAMES)
    rows = [f"{QNAMES[arm]} & " + " & ".join(
        cells("qwen_t3", "boss", arm, M5) + cells("qwen_t3", "truev3", arm, M5))
        + r" \\" for arm in ["llmfull", "llmphrase", "llmgraph", "llmplacebo",
                             "llmfact", "llmhead"]]
    rows += [f"{QNAMES[arm]} & " + " & ".join(
        cells("pfx_t3", "boss", arm, M5) + cells("pfx_t3", "truev3", arm, M5))
        + r" \\" for arm in ["prefixonly", "prefixgraph"]]
    w("table_ow_t3.tex", RB, M5, rows)

    # retire the superseded split-judge tables so stale files cannot be input
    for old in ("table_joint.tex", "table_joint_judge.tex",
                "table_judge_surfaces.tex", "table_ow_quest.tex"):
        p = os.path.join(OUT, old)
        if os.path.exists(p):
            os.remove(p)
            print("removed", p)
    print("[done]")


if __name__ == "__main__":
    main()
