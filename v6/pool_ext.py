"""Post-freeze evaluation testbeds (batch of 2026-08-02, user plan step 1).

Same loader contract as testbeds.py / pool.py. These sets were added AFTER the v6 method freeze and
are never used for tuning; role = EVAL_NEW. Domains deliberately leave personality questionnaires:
clinical scales first, then survey values / lab panels / gene pathways.

DASS-42: Depression Anxiety Stress Scales (Lovibond & Lovibond 1995), openpsychometrics raw data
(2017-2019, n=39775 after the site's own research-consent filter). Items rated 1-4 (past week).
Subscale key verified against two independent scoring sources (NovoPsych DASS-42 scoring page;
Temple RT assessment DASS form) and the official content-area descriptions at
www2.psy.unsw.edu.au/dass/over.htm; the three 14-item sets partition items 1..42 exactly.
Empirical check on the loaded matrix: within-scale mean r exceeds every between-scale mean r for
all three scales, and 40/42 items correlate most with their own published scale. The two exceptions
(Q9, Q30, both keyed Anxiety) are anxiety/stress cross-loaders with margins of .002 and .011; the
published key stays authoritative, so this graph is a mildly impure measurement model by design.
"""
import html as _html
import os
import re

import numpy as np
import pandas as pd

import graph as G
from pool import CAP, POOL, _norm
from testbeds import z

DASS_KEY = {
    "depression": [3, 5, 10, 13, 16, 17, 21, 24, 26, 31, 34, 37, 38, 42],
    "anxiety": [2, 4, 7, 9, 15, 19, 20, 23, 25, 28, 30, 36, 40, 41],
    "stress": [1, 6, 8, 11, 12, 14, 18, 22, 27, 29, 32, 33, 35, 39],
}


def dass():
    d = os.path.join(POOL, "DASS", "DASS_data_21.02.19")
    raw = open(os.path.join(d, "codebook.txt"), encoding="cp1252").read()
    items = {f"Q{k}": _norm(_html.unescape(v))
             for k, v in re.findall(r"^Q(\d+)\t(.+)$", raw, re.M)}
    construct_of = {f"Q{n}": s for s, ns in DASS_KEY.items() for n in ns}
    assert set(construct_of) == set(items) and len(items) == 42
    g = G.bipartite(construct_of)
    df = pd.read_csv(os.path.join(d, "data.csv"), sep="\t", low_memory=False)
    R = df[[q + "A" for q in g.observed]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    R[R < 1] = np.nan
    R = R[~np.isnan(R).any(1)]
    if len(R) > CAP:
        R = R[np.random.default_rng(0).choice(len(R), CAP, replace=False)]
    gt = {s: f"the DASS clinical scale: {s}" for s in DASS_KEY}
    return dict(name="dass", graph=g, X=z(R), labels=items, latent_gt=gt)


# --------------------------------------------------------------- NHANES clinical laboratory panels
# NHANES 2017 to March 2020 pre-pandemic cycle, public domain, no registration. Observed variables
# are serum and whole-blood analytes; label texts are the SAS variable labels shipped in the XPT
# files, verbatim.
#
# The grouping is standard clinical panel membership (NLM MedlinePlus panel definitions), not a key
# shipped with NHANES. This is a weaker provenance than a questionnaire scoring key and is declared
# as such. Analytes that belong to no multi-item panel (creatine phosphokinase, serum iron) are
# dropped rather than given a one-child latent.
#
# Excluded by construction: SI-unit duplicates (LBD*SI) and comment codes (LBD*LC), which are the
# same measurement twice; the differential absolute counts (LBD*NO), which are percent times white
# blood cell count exactly; osmolality, which NHANES computes from sodium, glucose and urea; and the
# serum cholesterol copy in the biochemistry file, which duplicates the total cholesterol file.
# Fasting-subsample files (plasma glucose, insulin, LDL) are excluded to keep n near 9000.
NHANES_PANELS = {
    "white blood cell differential": ["LBXWBCSI", "LBXLYPCT", "LBXMOPCT", "LBXNEPCT",
                                      "LBXEOPCT", "LBXBAPCT"],
    "red blood cell indices": ["LBXRBCSI", "LBXHGB", "LBXHCT", "LBXMCVSI", "LBXMC",
                               "LBXMCHSI", "LBXRDW"],
    "platelet indices": ["LBXPLTSI", "LBXMPSI"],
    "liver function": ["LBXSATSI", "LBXSASSI", "LBXSAPSI", "LBXSGTSI", "LBXSTB", "LBXSLDSI"],
    "serum proteins": ["LBXSAL", "LBXSGB", "LBXSTP"],
    "kidney function": ["LBXSBU", "LBXSCR", "LBXSUA"],
    "electrolytes": ["LBXSNASI", "LBXSKSI", "LBXSCLSI", "LBXSC3SI"],
    "bone minerals": ["LBXSCA", "LBXSPH"],
    "lipid panel": ["LBXTC", "LBDHDD", "LBXSTR"],
    "glycemic control": ["LBXGH", "LBXSGL"],
}
NHANES_FILES = ["P_CBC", "P_BIOPRO", "P_TCHOL", "P_HDL", "P_GHB"]
NHANES_DEDUP_R = 0.99      # drop one of any analyte pair this collinear (algebraically derived)


def _xpt(path):
    """Read an NHANES XPT file, returning (dataframe, {variable: SAS label})."""
    reader = pd.read_sas(path, format="xport", iterator=True)
    labels = {f["name"].decode(): _norm(f["label"].decode("latin1")) for f in reader.fields}
    return pd.read_sas(path, format="xport"), labels


def nhanes():
    d = os.path.join(POOL, "NHANES")
    df, labels = None, {}
    for f in NHANES_FILES:
        part, lab = _xpt(os.path.join(d, f + ".XPT"))
        labels.update(lab)
        df = part if df is None else df.merge(part, on="SEQN", how="inner")
    wanted = [v for vs in NHANES_PANELS.values() for v in vs]
    R = df[wanted].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    R = R[~np.isnan(R).any(1)]

    # drop the second member of any algebraically derived pair, keeping panel order stable
    C = np.corrcoef(R, rowvar=False)
    drop = set()
    for i in range(len(wanted)):
        for j in range(i + 1, len(wanted)):
            if j not in drop and abs(C[i, j]) > NHANES_DEDUP_R:
                drop.add(j)
    keep = [v for i, v in enumerate(wanted) if i not in drop]
    R = R[:, [i for i in range(len(wanted)) if i not in drop]]
    if len(R) > CAP:
        R = R[np.random.default_rng(0).choice(len(R), CAP, replace=False)]

    construct_of = {v: p for p, vs in NHANES_PANELS.items() for v in vs if v in keep}
    items = {v: labels[v] for v in keep}
    g = G.bipartite(construct_of)
    order = [keep.index(v) for v in g.observed]
    gt = {p: f"the clinical laboratory panel: {p}" for p in NHANES_PANELS}
    return dict(name="nhanes", graph=g, X=z(R[:, order]), labels=items, latent_gt=gt)


# --------------------------------------------------------------- WVS Wave 7 (Welzel value indices)
# World Values Survey Wave 7 (2017-2022), SPSS file, 97220 respondents, 613 variables. Downloaded
# under the WVS Conditions of Use. Three-layer published structure, the same shape as HEXACO:
# two overall indices -> sub-indices -> items, all defined by Welzel's recode syntax and shipped
# in the file as computed columns.
#
# X uses the official I_* recodes (0 to 1). The LABEL TEXTS do not: the shipped I_* labels name
# their own parent construct ("IMAGIN- Welzel autonomy-2: ...") and some carry no content at all
# ("Welzel voice-1"), so the label text is taken from the SOURCE question instead. The source of
# each I_* variable was identified empirically by rank correlation over 15000 respondents; every
# mapping below reached |rho| >= 0.933 and 16 of 21 reached 1.000. Two of them (I_INDEP = Q8,
# I_HOMOLIB = Q182) match the legacy codes A029 and F118 in Welzel's published syntax.
#
# DECLARED discrepancy: the shipped label calls I_DEVOUT "inverse devoutness", but the data is a
# rescaling of Q27 (rho 1.000, and its four levels map onto Q27's four categories), while every
# religiosity item correlates at most 0.39. The label text follows the data. Read as a group, the
# DEFIANCE items are then deference to three traditional authorities: the state, the nation and
# the parents.
#
# The VOICE sub-index is dropped: its three items have no substantive label in the file and no
# single source question, so no honest label text exists for them. RESEMAVAL therefore has three
# sub-indices here rather than four.
WVS_SOURCE = {
    "I_AUTHORITY": ("Q45", "Future changes: Greater respect for authority"),
    "I_NATIONALISM": ("Q254", "National pride"),
    "I_DEVOUT": ("Q27", "One of main goals in life has been to make my parents proud"),
    "I_RELIGIMP": ("Q6", "Important in life: Religion"),
    "I_RELIGBEL": ("Q173", "Religious person"),
    "I_RELIGPRAC": ("Q171", "How often do you attend religious services"),
    "I_NORM1": ("Q178", "Justifiable: Avoiding a fare on public transport"),
    "I_NORM2": ("Q180", "Justifiable: Cheating on taxes"),
    "I_NORM3": ("Q181", "Justifiable: Someone accepting a bribe in the course of their duties"),
    "I_TRUSTARMY": ("Q65", "Confidence: Armed Forces"),
    "I_TRUSTPOLICE": ("Q69", "Confidence: The Police"),
    "I_TRUSTCOURTS": ("Q70", "Confidence: Justice System/Courts"),
    "I_INDEP": ("Q8", "Important child qualities: independence"),
    "I_IMAGIN": ("Q11", "Important child qualities: imagination"),
    "I_NONOBED": ("Q17", "Important child qualities: obedience"),
    "I_WOMJOB": ("Q33", "Jobs scarce: Men should have more right to a job than women"),
    "I_WOMPOL": ("Q29", "Men make better political leaders than women do"),
    "I_WOMEDU": ("Q30", "University is more important for a boy than for a girl"),
    "I_HOMOLIB": ("Q182", "Justifiable: Homosexuality"),
    "I_ABORTLIB": ("Q184", "Justifiable: Abortion"),
    "I_DIVORLIB": ("Q185", "Justifiable: Divorce"),
}
WVS_SUBINDEX = {
    "defiance": ["I_AUTHORITY", "I_NATIONALISM", "I_DEVOUT"],
    "disbelief": ["I_RELIGIMP", "I_RELIGBEL", "I_RELIGPRAC"],
    "relativism": ["I_NORM1", "I_NORM2", "I_NORM3"],
    "scepticism": ["I_TRUSTARMY", "I_TRUSTPOLICE", "I_TRUSTCOURTS"],
    "autonomy": ["I_INDEP", "I_IMAGIN", "I_NONOBED"],
    "equality": ["I_WOMJOB", "I_WOMPOL", "I_WOMEDU"],
    "choice": ["I_HOMOLIB", "I_ABORTLIB", "I_DIVORLIB"],
}
WVS_TOP = {
    "secular values": ["defiance", "disbelief", "relativism", "scepticism"],
    "emancipative values": ["autonomy", "equality", "choice"],
}
WVS_GT = {
    "defiance": "the Welzel sub-index: defiance, rejection of deference to traditional authority",
    "disbelief": "the Welzel sub-index: disbelief, distance from religious belief and practice",
    "relativism": "the Welzel sub-index: relativism, refusal of absolute moral norms",
    "scepticism": "the Welzel sub-index: scepticism, low confidence in state institutions",
    "autonomy": "the Welzel sub-index: autonomy, valuing independence over obedience in children",
    "equality": "the Welzel sub-index: equality, support for equal standing of women and men",
    "choice": "the Welzel sub-index: choice, acceptance of personal lifestyle choices",
    "secular values": "the Welzel overall index: secular values, as opposed to sacred and"
                      " traditional values",
    "emancipative values": "the Welzel overall index: emancipative values, priority on freedom of"
                           " choice and equality of opportunity",
}


def wvs():
    import pyreadstat
    path = os.path.join(POOL, "WVS", "WVS_Cross-National_Wave_7_spss_v6_0.sav")
    items = list(WVS_SOURCE)
    df, _ = pyreadstat.read_sav(path, usecols=items)
    R = df[items].to_numpy(float)
    R[R < 0] = np.nan                                   # WVS missing codes are negative
    R = R[~np.isnan(R).any(1)]
    if len(R) > CAP:
        R = R[np.random.default_rng(0).choice(len(R), CAP, replace=False)]
    edges = [(top, sub) for top, subs in WVS_TOP.items() for sub in subs]
    edges += [(sub, it) for sub, its in WVS_SUBINDEX.items() for it in its]
    g = G.Graph(sorted(WVS_TOP) + sorted(WVS_SUBINDEX), items, edges)
    order = [items.index(v) for v in g.observed]
    labels = {v: _norm(WVS_SOURCE[v][1]) for v in items}
    return dict(name="wvs", graph=g, X=z(R[:, order]), labels=labels, latent_gt=dict(WVS_GT))


LOADERS = {"dass": dass, "nhanes": nhanes, "wvs": wvs}

EVAL_NEW = ["dass", "nhanes", "wvs"]
