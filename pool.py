"""Openpsychometrics dataset pool + Holzinger-Swineford: 10 additional given-graph testbeds.

Every loader returns the same dict as testbeds.py (graph, X, labels, latent_gt, name). Graphs are the
DESIGNED structures of the published questionnaires (item -> scale keying from each dataset's own
codebook; GCBS keying and item texts from Brotherton, French & Pickering 2013, Table A1). Item label
texts come VERBATIM from the codebooks (cp1252-decoded, typographic quotes normalized).

Cleaning rules, uniform across the pool: values < valid minimum are treated as missing; rows with any
missing item are dropped; datasets larger than CAP rows are subsampled with a fixed seed.

Dataset roles (LODO protocol, fixed 2026-07-14):
  DEV      - used to fit every global choice (lambdas, encoder, loss form). TLVD/Himi/BigFive live in
             testbeds.py and are dev too (they already influenced the v1 design).
  HELDOUT  - never touched during development; final frozen-method run only.
"""
import os, re, html as _html
import numpy as np
import pandas as pd
import graph as G
from testbeds import DATA, z

POOL = os.path.join(DATA, "pool")
CAP = 5000


def _norm(s):
    for a, b in [("’", "'"), ("‘", "'"), ("“", '"'), ("”", '"'),
                 ("–", "-"), ("—", "-"), ("…", "..."), ("", "'"),
                 ("", '"'), ("", '"')]:
        s = s.replace(a, b)
    return re.sub(r"\s+", " ", s).strip()


def _codebook_items(path, pattern):
    """Extract {item_code: text} from a codebook via a 2-group regex, cp1252-decoded."""
    out = {}
    for line in open(path, encoding="cp1252"):
        m = re.match(pattern, line.strip())
        if m:
            out[m.group(1)] = _norm(m.group(2))
    return out


def _load_matrix(path, cols, sep, vmin=1, cap=CAP, seed=0):
    df = pd.read_csv(path, sep=sep, low_memory=False)
    df.columns = [str(c).strip().strip('"') for c in df.columns]
    R = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    R[R < vmin] = np.nan
    R = R[~np.isnan(R).any(1)]
    if len(R) > cap:
        R = R[np.random.default_rng(seed).choice(len(R), cap, replace=False)]
    return z(R)


def _pack(name, construct_of, labels, X_path, sep, latent_gt, vmin=1, graph_override=None):
    g = graph_override or G.bipartite(construct_of)
    X = _load_matrix(X_path, g.observed, sep, vmin=vmin)
    return dict(name=name, graph=g, X=X, labels=labels, latent_gt=latent_gt)


# ---------------------------------------------------------------- HEXACO (3-layer: factor -> facet -> item)
HEX_FACTOR = {"H": "honesty-humility", "E": "emotionality", "X": "extraversion",
              "A": "agreeableness", "C": "conscientiousness", "O": "openness to experience"}
HEX_FACET = {"HSinc": "sincerity", "HFair": "fairness", "HGree": "greed avoidance", "HMode": "modesty",
             "EFear": "fearfulness", "EAnxi": "anxiety", "EDepe": "dependence", "ESent": "sentimentality",
             "XExpr": "expressiveness", "XSocB": "social boldness", "XSoci": "sociability",
             "XLive": "liveliness",
             "AForg": "forgiveness", "AGent": "gentleness", "AFlex": "flexibility", "APati": "patience",
             "COrga": "organization", "CDili": "diligence", "CPerf": "perfectionism", "CPrud": "prudence",
             "OAesA": "aesthetic appreciation", "OInqu": "inquisitiveness", "OCrea": "creativity",
             "OUnco": "unconventionality"}


def hexaco():
    d = os.path.join(POOL, "HEXACO", "HEXACO")
    items = _codebook_items(os.path.join(d, "codebook.txt"), r"^([HEXACO][A-Za-z]{4}\d+)\s+(.+)$")
    facets = sorted(set(HEX_FACET.values()))
    factors = sorted(set(HEX_FACTOR.values()))
    edges = [(HEX_FACTOR[k[0]], v) for k, v in HEX_FACET.items()]              # factor -> facet
    edges += [(HEX_FACET[re.match(r"[HEXACO][A-Za-z]{4}", it).group(0)], it) for it in items]
    g = G.Graph(factors + facets, sorted(items), edges)
    X = _load_matrix(os.path.join(d, "data.csv"), g.observed, "\t")
    gt = {v: f"the HEXACO personality factor: {v}" for v in factors}
    gt.update({HEX_FACET[k]: f"the {HEX_FACTOR[k[0]]} facet: {HEX_FACET[k]}" for k in HEX_FACET})
    return dict(name="hexaco", graph=g, X=X, labels=items, latent_gt=gt)


# ---------------------------------------------------------------- 16PF (IPIP analogs of Cattell's factors)
# The codebook re-letters the 16 scales contiguously A..P in Cattell's canonical order; mapping verified
# item-by-item against each scale's first item (e.g. D1 "I take charge" = dominance).
PF16_NAMES = ["warmth", "reasoning", "emotional stability", "dominance", "liveliness",
              "rule-consciousness", "social boldness", "sensitivity", "vigilance", "abstractedness",
              "privateness", "apprehension", "openness to change", "self-reliance", "perfectionism",
              "tension"]
PF16 = {L: n for L, n in zip("ABCDEFGHIJKLMNOP", PF16_NAMES)}


def sixteenpf():
    d = os.path.join(POOL, "16PF", "16PF")
    t = open(os.path.join(d, "codebook.html"), encoding="cp1252").read()
    items = {}
    for r in re.findall(r"<tr>(.*?)</tr>", t, re.S):
        cells = [_html.unescape(re.sub("<[^>]+>", "", c)).strip()
                 for c in re.findall(r"<td[^>]*>(.*?)</td>", r, re.S)]
        if len(cells) >= 3 and re.fullmatch(r"[A-P]\d+", cells[0]):
            m = re.match(r'"(.+?)" rated', cells[2])
            if m:
                items[cells[0]] = _norm(m.group(1))
    construct_of = {c: PF16[c[0]] for c in items}
    gt = {n: f"the 16PF personality factor: {n}" for n in PF16_NAMES}
    return _pack("sixteenpf", construct_of, items, os.path.join(d, "data.csv"), "\t", gt)


# ---------------------------------------------------------------- RIASEC (Holland occupational interests)
RIASEC = {"R": "realistic interests", "I": "investigative interests", "A": "artistic interests",
          "S": "social interests", "E": "enterprising interests", "C": "conventional interests"}


def riasec():
    d = os.path.join(POOL, "RIASEC", "RIASEC_data12Dec2018")
    items = _codebook_items(os.path.join(d, "codebook.txt"), r"^([RIASEC]\d)\t(.+)$")
    construct_of = {c: RIASEC[c[0]] for c in items}
    gt = {v: f"the Holland occupational interest type: {v}" for v in RIASEC.values()}
    return _pack("riasec", construct_of, items, os.path.join(d, "data.csv"), "\t", gt)


# ---------------------------------------------------------------- HSQ (humor styles; items cycle mod 4)
HSQ_STYLES = ["affiliative humor", "self-enhancing humor", "aggressive humor", "self-defeating humor"]


def hsq():
    d = os.path.join(POOL, "HSQ", "HSQ")
    items = _codebook_items(os.path.join(d, "codebook.txt"), r"^Q(\d+)\.\s*(.+)$")
    items = {f"Q{k}": v for k, v in items.items()}
    construct_of = {q: HSQ_STYLES[(int(q[1:]) - 1) % 4] for q in items}
    gt = {s: f"the humor style: {s}" for s in HSQ_STYLES}
    return _pack("hsq", construct_of, items, os.path.join(d, "data.csv"), ",", gt)


# ---------------------------------------------------------------- KIMS (mindfulness; keying from the
# scoring code shipped at the end of the codebook itself)
KIMS_SCALES = {"observing": "observing", "describing": "describing",
               "acting": "acting with awareness", "accepting": "accepting without judgment"}


def kims():
    d = os.path.join(POOL, "KIMS", "KIMS")
    raw = open(os.path.join(d, "codebook.txt"), encoding="cp1252").read()
    items = {f"Q{k}": _norm(v) for k, v in re.findall(r"^Q(\d+)\.\s*(.+)$", raw, re.M)}
    construct_of = {}
    for var, name in KIMS_SCALES.items():
        block = re.search(r"\$" + var + r"\s*=\s*round\(\((.*?)\)\s*/", raw, re.S)
        for q in re.findall(r"Q(\d+)", block.group(1)):
            construct_of[f"Q{q}"] = name
    # The site's scoring code omits Q21 ("I pay attention to sensations, such as the wind in my hair or
    # sun on my face"); the published instrument (Baer, Smith & Allen 2004) keys item 21 on Observe.
    construct_of.setdefault("Q21", "observing")
    assert set(construct_of) == set(items), "KIMS keying must cover all 39 items"
    gt = {n: f"the mindfulness skill: {n}" for n in KIMS_SCALES.values()}
    return _pack("kims", construct_of, items, os.path.join(d, "data.csv"), ",", gt)


# ---------------------------------------------------------------- SD3 (short dark triad)
SD3 = {"M": "machiavellianism", "N": "narcissism", "P": "psychopathy"}


def sd3():
    d = os.path.join(POOL, "SD3", "SD3")
    items = _codebook_items(os.path.join(d, "codebook.txt"), r"^([MNP]\d)\s+(.+)$")
    construct_of = {c: SD3[c[0]] for c in items}
    gt = {v: f"the dark-triad personality trait: {v}" for v in SD3.values()}
    return _pack("sd3", construct_of, items, os.path.join(d, "data.csv"), "\t", gt)


# ---------------------------------------------------------------- GCBS (generic conspiracist beliefs)
# Item texts and 5-facet keying from Brotherton, French & Pickering 2013 (Frontiers in Psychology),
# Table A1; the codebook states question numbers match that table (facets cycle mod 5).
GCBS_FACETS = ["government malfeasance", "malevolent global conspiracies", "extraterrestrial cover-up",
               "personal wellbeing", "control of information"]
GCBS_ITEMS = {
    "Q1": "The government is involved in the murder of innocent citizens and/or well-known public figures",
    "Q2": "The power held by heads of state is second to that of small unknown groups who really control"
          " world politics",
    "Q3": "Secret organizations communicate with extraterrestrials, but keep this fact from the public",
    "Q4": "The spread of certain viruses and/or diseases is the result of deliberate, concealed efforts"
          " of some organization",
    "Q5": "Groups of scientists manipulate, fabricate, or suppress evidence in order to deceive the public",
    "Q6": "The government permits or perpetrates acts of terrorism on its own soil, disguising its"
          " involvement",
    "Q7": "A small, secret group of people is responsible for making all major world decisions, such as"
          " going to war",
    "Q8": "Evidence of alien contact is being concealed from the public",
    "Q9": "Technology with mind-control capacities is used on people without their knowledge",
    "Q10": "New and advanced technology which would harm current industry is being suppressed",
    "Q11": "The government uses people as patsies to hide its involvement in criminal activity",
    "Q12": "Certain significant events have been the result of the activity of a small group who secretly"
           " manipulate world events",
    "Q13": "Some UFO sightings and rumors are planned or staged in order to distract the public from real"
           " alien contact",
    "Q14": "Experiments involving new drugs or technologies are routinely carried out on the public"
           " without consent",
    "Q15": "A lot of important information is deliberately concealed from the public out of self-interest"}


def gcbs():
    d = os.path.join(POOL, "GCBS", "data")
    construct_of = {q: GCBS_FACETS[(int(q[1:]) - 1) % 5] for q in GCBS_ITEMS}
    gt = {f: f"belief in conspiracies of the type: {f}" for f in GCBS_FACETS}
    return _pack("gcbs", construct_of, dict(GCBS_ITEMS), os.path.join(d, "data.csv"), ",", gt)


# ---------------------------------------------------------------- RSE (Rosenberg self-esteem; 1 factor)
def rse():
    d = os.path.join(POOL, "RSE", "RSE")
    items = _codebook_items(os.path.join(d, "codebook.txt"), r"^Q(\d+)\.\s*(.+)$")
    items = {f"Q{k}": v for k, v in items.items()}
    construct_of = {q: "self-esteem" for q in items}
    gt = {"self-esteem": "the construct: self-esteem (global evaluation of one's own worth)"}
    return _pack("rse", construct_of, items, os.path.join(d, "data.csv"), "\t", gt)


# ---------------------------------------------------------------- MACH-IV (machiavellianism; 1 factor)
def mach():
    d = os.path.join(POOL, "MACH", "MACH_data")
    raw = open(os.path.join(d, "codebook.txt"), encoding="cp1252").read()
    items = {f"Q{k}A": _norm(v) for k, v in re.findall(r'"Q(\d+)"\s*:\s*"(.+?)"', raw)}
    construct_of = {q: "machiavellianism" for q in items}
    gt = {"machiavellianism": "the personality trait: machiavellianism (manipulativeness, cynical view"
                              " of human nature, moral pragmatism)"}
    return _pack("mach", construct_of, items, os.path.join(d, "data.csv"), "\t", gt)


# ---------------------------------------------------------------- Holzinger-Swineford 1939 (24 ability
# tests, classic 5-factor battery; variable names are the test names from the published study)
HS_FACTOR = {
    "visual": "spatial ability", "cubes": "spatial ability", "paper": "spatial ability",
    "flags": "spatial ability",
    "general": "verbal ability", "paragrap": "verbal ability", "sentence": "verbal ability",
    "wordc": "verbal ability", "wordm": "verbal ability",
    "addition": "perceptual speed", "code": "perceptual speed", "counting": "perceptual speed",
    "straight": "perceptual speed",
    "wordr": "memory", "numberr": "memory", "figurer": "memory", "object": "memory",
    "numberf": "memory", "figurew": "memory",
    "deduct": "mathematical reasoning", "numeric": "mathematical reasoning",
    "problemr": "mathematical reasoning", "series": "mathematical reasoning",
    "arithmet": "mathematical reasoning"}
HS_DESC = {
    "visual": "visual perception test", "cubes": "cubes spatial test",
    "paper": "paper form board spatial test", "flags": "lozenges flags spatial test",
    "general": "general information verbal test", "paragrap": "paragraph comprehension test",
    "sentence": "sentence completion test", "wordc": "word classification test",
    "wordm": "word meaning test", "addition": "speeded addition test",
    "code": "speeded code perceptual test", "counting": "speeded counting of groups of dots",
    "straight": "speeded discrimination of straight and curved capitals",
    "wordr": "word recognition memory test", "numberr": "number recognition memory test",
    "figurer": "figure recognition memory test", "object": "object-number association memory test",
    "numberf": "number-figure association memory test", "figurew": "figure-word association memory test",
    "deduct": "deduction reasoning test", "numeric": "numerical puzzles test",
    "problemr": "problem reasoning test", "series": "series completion reasoning test",
    "arithmet": "arithmetic problems test"}


def hs():
    g = G.bipartite(HS_FACTOR)
    df = pd.read_csv(os.path.join(DATA, "HS.data.csv"))
    R = df[g.observed].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    R = R[~np.isnan(R).any(1)]
    gt = {f: f"the cognitive ability factor: {f}" for f in g.latents}
    return dict(name="hs", graph=g, X=z(R), labels=dict(HS_DESC), latent_gt=gt)


LOADERS = {"hexaco": hexaco, "sixteenpf": sixteenpf, "riasec": riasec, "hsq": hsq, "kims": kims,
           "sd3": sd3, "gcbs": gcbs, "rse": rse, "mach": mach, "hs": hs}

# LODO roles (testbeds.py's tlvd/himi/bigfive are DEV: they already shaped the v1 design)
DEV = ["tlvd", "himi", "bigfive", "hs", "rse", "mach", "gcbs"]
HELDOUT = ["hexaco", "sixteenpf", "riasec", "hsq", "kims", "sd3"]
