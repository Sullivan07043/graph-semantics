"""Emit the table files for report/plugin_experiment/ (generated/*.tex).

Reuses the collectors in gen_llm_report. The merged-task table renders from
llm_t12_* files plus the joint records' core arm; it degrades to '--' cells while
those runs are still in flight. Rerun any time; prose in plugin_experiment.tex is
hand-written and inputs these files.
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict

_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PKG)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
import gen_llm_report as G  # noqa: E402

OUT = os.path.join(os.path.dirname(config.GS), "report", "plugin_experiment", "generated")
os.makedirs(OUT, exist_ok=True)
f = G.f
NAMES = {"core": "pipeline alone (no LLM)", "llmfull": "LLM + plugin (main)",
         "llmphrase": "LLM + phrases only", "llmgraph": "LLM only (no plugin)",
         "llmplacebo": "LLM + shuffled evidence", "llmfact": "LLM + plugin + dt fact",
         "llmhead": "LLM + plugin + dt + naming head (main)"}
# On T3 the main method is the full plugin: dt fact and naming head included.
T3NAMES = dict(NAMES)
T3NAMES["llmfull"] = "LLM + phrases + graph"


HEADERS = {
    "table_t2.tex": ("l cc cc",
        "& \\multicolumn{2}{c}{given graphs} & \\multicolumn{2}{c}{discovered graphs} \\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
        "arm & NRR@1 & MRR & NRR@1 & MRR \\\\"),
    "table_stress.tex": ("ll cccc",
        "graphs & arm & 0\\% & 30\\% & 60\\% & 90\\% \\\\"),
    "table_t1.tex": ("l ccc ccc",
        "& \\multicolumn{3}{c}{given graphs} & \\multicolumn{3}{c}{discovered graphs} \\\\\n"
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n"
        "arm & NRR@1 & NRR@5 & SDA & NRR@1 & NRR@5 & SDA \\\\"),
    "table_t3.tex": ("l ccc ccc",
        "& \\multicolumn{3}{c}{discovered (BOSS)} & \\multicolumn{3}{c}{truth, measured gains} \\\\\n"
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n"
        "arm & NRR@1 & NRR@5 & SDA & NRR@1 & NRR@5 & SDA \\\\"),
    "table_joint.tex": ("l cc cc",
        "& \\multicolumn{2}{c}{given graphs} & \\multicolumn{2}{c}{discovered graphs} \\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
        "arm & items NRR@1 & latents NRR@1 & items NRR@1 & latents NRR@1 \\\\"),
    "table_ow_quest.tex": ("l cc cc",
        "& \\multicolumn{2}{c}{given graphs} & \\multicolumn{2}{c}{discovered graphs} \\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
        "arm & items NRR@1 & latents NRR@1 & items NRR@1 & latents NRR@1 \\\\"),
    "table_ow_t3.tex": ("l ccc ccc",
        "& \\multicolumn{3}{c}{discovered (BOSS)} & \\multicolumn{3}{c}{truth, measured gains} \\\\\n"
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n"
        "arm & NRR@1 & NRR@5 & SDA & NRR@1 & NRR@5 & SDA \\\\"),
    "table_joint_judge.tex": ("l cc cc",
        "& \\multicolumn{2}{c}{given graphs} & \\multicolumn{2}{c}{discovered graphs} \\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
        "arm & items judge & latents judge & items judge & latents judge \\\\"),
    "table_judge_surfaces.tex": ("ll cc",
        "surface & arm & given / BOSS & discovered / truth \\\\"),
}


def w(name, text):
    spec, head = HEADERS[name]
    full = ("\\begin{tabular}{" + spec + "}\n\\toprule\n" + head +
            "\n\\midrule\n" + text.rstrip() + "\n\\bottomrule\n\\end{tabular}\n")
    p = os.path.join(OUT, name)
    open(p, "w").write(full)
    print("wrote", p)


def main():
    # ---- T2 arms table (both modes) -------------------------------------------
    t2, t2core = G.collect_t2()
    rows = [f"{NAMES['core']} & {f(t2core['given'][0])} & {f(t2core['given'][1])} & "
            f"{f(t2core['disc'][0])} & {f(t2core['disc'][1])} \\\\"]
    for arm in G.ARMS:
        g_, d_ = G.mean_rows(t2, ("given", "disc"), [arm], 2)[("given", arm)], \
            G.mean_rows(t2, ("given", "disc"), [arm], 2)[("disc", arm)]
        rows.append(f"{NAMES[arm]} & {f(g_[0])} & {f(g_[1])} & {f(d_[0])} & {f(d_[1])} \\\\")
    w("table_t2.tex", "\n".join(rows) + "\n")

    # ---- stress curve ----------------------------------------------------------
    curve = G.collect_stress()
    rows = []
    for src in ("given", "disc"):
        for arm in ("llmgraph", "llmfull"):
            cells = []
            for pct in (0, 30, 60, 90):
                v = curve[(src, pct)].get(arm, [])
                cells.append(f(sum(v) / len(v)) if v else "--")
            rows.append(f"{src} & {NAMES[arm]} & " + " & ".join(cells) + r" \\")
    w("table_stress.tex", "\n".join(rows) + "\n")

    # ---- T1 and T3 arms tables -------------------------------------------------
    for surface, srcs, corekeys in (("t1", ("given", "disc"), None),
                                    ("t3", ("boss", "truev3"),
                                     ("base_boss", "base_truev3"))):
        acc, bases = G.collect_items(surface)
        arms = G.ARMS + (["llmfact", "llmhead"] if surface == "t3" else [])
        if surface == "t3":
            # llmhead rows live in their own llm_t3h_* files
            for p in sorted(glob.glob(os.path.join(config.REC_LLM, "llm_t3h_*.json"))):
                m = re.match(r"llm_t3h_(\w+?)_(boss|truev3)\.json", os.path.basename(p))
                if not m:
                    continue
                dh = json.load(open(p))
                rh = dh["records"] if isinstance(dh, dict) and "records" in dh else dh
                s = G.score_items(rh, "llmhead", m.group(1))
                if s:
                    acc[m.group(2)]["llmhead"].append(s)
        rows = []
        if surface == "t3":
            e = G.BASE_AGG["robot"]
            cells = []
            for key in corekeys:
                vals = [(e[t][key]["arms"]["core"]["nrr_top1"],
                         e[t][key]["arms"]["core"]["nrr_top5"],
                         e[t][key]["arms"]["core"]["sda"]) for t in e if key in e[t]]
                m = tuple(sum(x[i] for x in vals) / len(vals) for i in range(3))
                cells += [f(m[0]), f(m[1]), f(m[2])]
            rows.append(f"{NAMES['core']} & " + " & ".join(cells) + r" \\")
        else:
            DS = ["bigfive", "dass", "hexaco", "rse", "wpi"]
            cells = []
            for src in srcs:
                cv = [(G.BASE_AGG[src][d0]["t1"]["arms"]["core"]["nrr_top1"],
                       G.BASE_AGG[src][d0]["t1"]["arms"]["core"]["nrr_top5"],
                       G.BASE_AGG[src][d0]["t1"]["arms"]["core"]["sda"]) for d0 in DS]
                m = tuple(sum(x[i] for x in cv) / len(cv) for i in range(3))
                cells += [f(m[0]), f(m[1]), f(m[2])]
            rows.append(f"{NAMES['core']} & " + " & ".join(cells) + r" \\")
        nm = T3NAMES if surface == "t3" else NAMES
        for arm in arms:
            cells = []
            for src in srcs:
                v = acc[src].get(arm, [])
                if v:
                    m = tuple(sum(x[i] for x in v) / len(v) for i in range(4))
                    cells += [f(m[0]), f(m[1]), f(m[3])]
                else:
                    cells += ["--", "--", "--"]
            rows.append(f"{nm[arm]} & " + " & ".join(cells) + r" \\")
        w(f"table_{surface}.tex", "\n".join(rows) + "\n")

    # ---- merged T1+2 task ------------------------------------------------------
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import score_records as SR
    agg = defaultdict(lambda: defaultdict(
        lambda: {"i": [], "l": [], "c": [], "ij": [], "lj": []}))
    for p in sorted(glob.glob(os.path.join(config.REC_JOINT, "t12_*.json"))):
        m = re.match(r"t12_(\w+)_(given|disc)\.json", os.path.basename(p))
        ds, src = m.groups()
        d = json.load(open(p))
        recs = d["records"] if isinstance(d, dict) and "records" in d else d
        rows_ = [r for r in recs if r["arm"] == "core"]
        items = [r for r in rows_ if "var" in r]
        lats = [r for r in rows_ if "latent" in r]
        si = SR.score_items(items, ds) if items else None
        sl = SR.score_latents(lats) if lats else None
        if si:
            agg[src]["core"]["i"].append(si["nrr_top1"])
        if sl and sl.get("nrr_top1") is not None:
            agg[src]["core"]["l"].append(sl["nrr_top1"])
        for part, rs in (("ij", items), ("lj", lats)):
            jv = [r["judge"] for r in rs if r.get("judge") is not None]
            if jv:
                agg[src]["core"][part].append(sum(jv) / len(jv))
        for q in sorted(glob.glob(os.path.join(config.REC_LLM,
                                               f"llm_t12_{ds}_{src}_*.json"))):
            arm = re.match(rf"llm_t12_{ds}_{src}_(\w+)\.json", os.path.basename(q)).group(1)
            dq = json.load(open(q))
            rq = dq["records"] if isinstance(dq, dict) and "records" in dq else dq
            ai = [r for r in rq if r["arm"] == arm and "var" in r]
            al = [r for r in rq if r["arm"] == arm and "latent" in r]
            si = SR.score_items(ai, ds) if ai else None
            sl = SR.score_latents(al) if al else None
            if si:
                agg[src][arm]["i"].append(si["nrr_top1"])
            if sl and sl.get("nrr_top1") is not None:
                agg[src][arm]["l"].append(sl["nrr_top1"])
            for part, rs in (("ij", ai), ("lj", al)):
                jv = [r["judge"] for r in rs if r.get("judge") is not None]
                if jv:
                    agg[src][arm][part].append(sum(jv) / len(jv))
    rows = []
    for arm in ("core", "llmfull", "llmgraph", "llmphrase", "llmplacebo"):
        cells = []
        for src in ("given", "disc"):
            e = agg[src].get(arm)
            for part in ("i", "l"):
                v = e[part] if e else []
                cells.append(f(sum(v) / len(v)) if v else "--")
        rows.append(f"{NAMES[arm]} & " + " & ".join(cells) + r" \\")
    w("table_joint.tex", "\n".join(rows) + "\n")

    # ---- judge means on the merged task (only rows the judge has reached) ------
    rows = []
    for arm in ("core", "llmfull", "llmgraph", "llmphrase", "llmplacebo"):
        cells = []
        for src in ("given", "disc"):
            e = agg[src].get(arm)
            for part in ("ij", "lj"):
                v = e[part] if e else []
                cells.append(f(sum(v) / len(v)) if v else "--")
        rows.append(f"{NAMES[arm]} & " + " & ".join(cells) + r" \\")
    w("table_joint_judge.tex", "\n".join(rows) + "\n")

    # ---- judge means across the split surfaces ---------------------------------
    def _jmean(recs, arm):
        jv = [r["judge"] for r in recs
              if r.get("arm") == arm and r.get("judge") is not None]
        return sum(jv) / len(jv) if jv else None

    jt = defaultdict(lambda: defaultdict(list))
    SCANS = (
        ("t2", "llm_t2", r"llm_t2_(\w+)_(given|disc)\.json", G.ARMS),
        ("t1", "llm_t1", r"llm_t1_(\w+)_(given|disc)\.json", G.ARMS),
        ("t3", "llm_t3", r"llm_t3_(\w+?)_(boss|truev3)\.json", G.ARMS + ["llmfact"]),
        ("t3", "llm_t3h", r"llm_t3h_(\w+?)_(boss|truev3)\.json", ["llmhead"]),
        ("qt2", "qwen_t2", r"qwen_t2_(\w+)_(given|disc)\.json", G.ARMS),
        ("qt1", "qwen_t1", r"qwen_t1_(\w+)_(given|disc)\.json", G.ARMS),
        ("qt3", "qwen_t3", r"qwen_t3_(\w+?)_(boss|truev3)\.json", G.ARMS + ["llmfact"]),
        ("qt3", "qwen_t3h", r"qwen_t3h_(\w+?)_(boss|truev3)\.json", ["llmhead"]),
        ("pfx3", "pfx_t3", r"pfx_t3_(\w+?)_(boss|truev3)\.json",
         ["prefixonly", "prefixgraph"]),
    )
    for surf, stem, rex, armset in SCANS:
        for p in sorted(glob.glob(os.path.join(config.REC_LLM, f"{stem}_*.json"))):
            m = re.match(rex, os.path.basename(p))
            if not m:
                continue
            d = json.load(open(p))
            recs = d["records"] if isinstance(d, dict) and "records" in d else d
            for arm in armset:
                v = _jmean(recs, arm)
                if v is not None:
                    jt[(surf, m.group(2))][arm].append(v)
    # prefix merged records: judge means split into items and latents
    for p in sorted(glob.glob(os.path.join(config.REC_LLM, "pfx_t12_*.json"))):
        m = re.match(r"pfx_t12_(\w+)_(given|disc)\.json", os.path.basename(p))
        if not m:
            continue
        d = json.load(open(p))
        recs = d["records"] if isinstance(d, dict) and "records" in d else d
        for arm in ("prefixonly", "prefixgraph"):
            for key, want_item in (("pfxi", True), ("pfxl", False)):
                jv = [r["judge"] for r in recs
                      if r["arm"] == arm and (("var" in r) == want_item)
                      and r.get("judge") is not None]
                if jv:
                    jt[(key, m.group(2))][arm].append(sum(jv) / len(jv))
    for p in sorted(glob.glob(os.path.join(config.REC_LLM, "stress_t2_*_[369]0.json"))):
        m = re.match(r"stress_t2_(\w+)_(given|disc)_(\d+)\.json", os.path.basename(p))
        d = json.load(open(p))
        for arm in ("llmfull", "llmgraph"):
            v = _jmean(d["records"], arm)
            if v is not None:
                jt[(f"stress{m.group(3)}", m.group(2))][arm].append(v)

    # core rows from the campaign aggregate (same dataset pools as the tables)
    core = {}
    T1_DS = ["bigfive", "dass", "hexaco", "rse", "wpi"]
    for src in ("given", "disc"):
        for surf, pool in (("t2", list(G.BASE_AGG[src])), ("t1", T1_DS)):
            cv = [G.BASE_AGG[src][ds][surf]["arms"]["core"].get("judge")
                  for ds in pool
                  if G.BASE_AGG[src].get(ds, {}).get(surf, {}).get("arms", {}).get("core")]
            cv = [v for v in cv if v is not None]
            core[(surf, src)] = sum(cv) / len(cv) if cv else None
    eR = G.BASE_AGG["robot"]
    for key, src in (("base_boss", "boss"), ("base_truev3", "truev3")):
        cv = [eR[t][key]["arms"]["core"].get("judge") for t in eR if key in eR[t]]
        cv = [v for v in cv if v is not None]
        core[("t3", src)] = sum(cv) / len(cv) if cv else None

    JL = {**T3NAMES, "prefixonly": "prefix only",
          "prefixgraph": "prefix + graph text"}
    QD = ("given", "disc")
    RB = ("boss", "truev3")
    GROUPS = (
        ("t2", QD, "T2 latents", ["core"] + G.ARMS),
        ("t1", QD, "T1 items", ["core"] + G.ARMS),
        ("t3", RB, "T3 robots", ["core"] + G.ARMS + ["llmfact", "llmhead"]),
        ("stress30", QD, "T2 stress 30\\%", ["llmfull", "llmgraph"]),
        ("stress60", QD, "T2 stress 60\\%", ["llmfull", "llmgraph"]),
        ("stress90", QD, "T2 stress 90\\%", ["llmfull", "llmgraph"]),
        ("qt2", QD, "T2 latents (Qwen)", G.ARMS),
        ("qt1", QD, "T1 items (Qwen)", G.ARMS),
        ("qt3", RB, "T3 robots (Qwen)", G.ARMS + ["llmfact", "llmhead"]),
        ("pfxi", QD, "Merged items (Qwen prefix)", ["prefixonly", "prefixgraph"]),
        ("pfxl", QD, "Merged latents (Qwen prefix)", ["prefixonly", "prefixgraph"]),
        ("pfx3", RB, "T3 robots (Qwen prefix)", ["prefixonly", "prefixgraph"]),
    )
    groups = []
    ROBOT_SURFS = {"t3", "qt3", "pfx3", "pfxi", "pfxl"}
    for surf, srcs, label, arm_list in GROUPS:
        nm = JL if surf in ROBOT_SURFS else {**NAMES, **{k: JL[k] for k in
                                             ("prefixonly", "prefixgraph")}}
        grows = []
        for arm in arm_list:
            cells = []
            for src in srcs:
                if arm == "core":
                    v = core.get((surf, src))
                else:
                    vv = jt[(surf, src)].get(arm, [])
                    v = sum(vv) / len(vv) if vv else None
                cells.append(f(v) if v is not None else "--")
            grows.append(f"{label} & {nm[arm]} & " + " & ".join(cells) + r" \\")
        groups.append("\n".join(grows))
    w("table_judge_surfaces.tex", "\n\\addlinespace\n".join(groups) + "\n")

    # ---- open-weight backend: prefix on the merged task ------------------------
    OWNAMES = {"prefixonly": "Qwen + embedding prefix",
               "prefixgraph": "Qwen + prefix + graph text",
               "llmfull": "Qwen prompt + plugin",
               "llmphrase": "Qwen prompt + phrases only",
               "llmgraph": "Qwen prompt, graph only",
               "llmplacebo": "Qwen prompt + shuffled evidence",
               "llmfact": "Qwen prompt + plugin + dt fact",
               "llmhead": "Qwen prompt + plugin + dt + naming head"}
    sump = os.path.join(_PKG, "outputs", "scores", "prefix_lodo_summary.json")
    if os.path.exists(sump):
        S = json.load(open(sump))
        rows = []
        for arm in ("prefixonly", "prefixgraph"):
            cells = []
            for src in ("given", "disc"):
                e = S.get(arm, {}).get(src, {})
                cells.append(f(e["items"]["nrr_top1"]) if e.get("items") else "--")
                cells.append(f(e["latents"]["nrr_top1"]) if e.get("latents") else "--")
            rows.append(f"{OWNAMES[arm]} & " + " & ".join(cells) + r" \\")
        w("table_ow_quest.tex", "\n".join(rows) + "\n")

    # ---- open-weight backend: robots (discrete Qwen + prefix Qwen) -------------
    ow3 = defaultdict(lambda: defaultdict(list))
    for stem, armset in (("qwen_t3", G.ARMS + ["llmfact"]),
                         ("qwen_t3h", ["llmhead"]),
                         ("pfx_t3", ["prefixonly", "prefixgraph"])):
        for p in sorted(glob.glob(os.path.join(config.REC_LLM, f"{stem}_*.json"))):
            m = re.match(rf"{stem}_(\w+?)_(boss|truev3)\.json", os.path.basename(p))
            if not m:
                continue
            base, src = m.groups()
            d = json.load(open(p))
            recs = d["records"] if isinstance(d, dict) and "records" in d else d
            for arm in armset:
                s = G.score_items(recs, arm, base)
                if s:
                    ow3[src][arm].append(s)
    rows = []
    for arm in ("llmfull", "llmphrase", "llmgraph", "llmplacebo", "llmfact",
                "llmhead", "prefixonly", "prefixgraph"):
        label = OWNAMES[arm]
        cells = []
        for src in ("boss", "truev3"):
            v = ow3[src].get(arm, [])
            if v:
                mn = tuple(sum(x[i] for x in v) / len(v) for i in range(4))
                cells += [f(mn[0]), f(mn[1]), f(mn[3])]
            else:
                cells += ["--", "--", "--"]
        rows.append(f"{label} & " + " & ".join(cells) + r" \\")
    w("table_ow_t3.tex", "\n".join(rows) + "\n")
    print("[done]")


if __name__ == "__main__":
    main()
