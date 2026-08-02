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


LOADERS = {"dass": dass}

EVAL_NEW = ["dass"]
