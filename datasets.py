"""Given-graph testbeds: three real-data studies plus generated oracle diagnostics.
Each loader returns a dict:
  graph      : graph.Graph
  X          : [n_samples, n_observed] z-scored, columns ordered as graph.observed
  labels     : dict observed_name -> label text (ALL observed have one; masking happens in the runner)
  latent_gt  : dict latent_name -> ground-truth description text (for Task 2 judging)
  name       : dataset name
Data locations: env GRAPHSEM_DATA (default ../data relative to this file's parent project);
TLVD graph/description files cached by fetch_tlvd.sh into data_cache/.
"""
import csv
import os, json
import numpy as np
import graph as G

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("GRAPHSEM_DATA", os.path.abspath(os.path.join(HERE, "..", "data")))
CACHE = os.path.join(HERE, "data_cache")
ORACLE_DATA = os.environ.get("GRAPHSEM_ORACLE_DATA", os.path.join(HERE, "data"))


def z(a):
    a = np.asarray(a, float)
    return (a - a.mean(0)) / (a.std(0) + 1e-9)


# --------------------------------------------------------------------------- TLVD Multitasking (given .dot)
def tlvd():
    import pyreadstat
    g = G.from_dot(os.path.join(CACHE, "multitasking_alpha0.05_rtscale1_N-1.dot"))
    desc = json.load(open(os.path.join(CACHE, "multitasking_description.json")))
    labels = {o: desc[o]["description"] for o in g.observed}
    df, _ = pyreadstat.read_sav(os.path.join(DATA, "TLVD", "Final_Multitasking_Data.sav"))
    cols = {o: (o[2:] if o.startswith("X_") else o) for o in g.observed}
    R = df[[cols[o] for o in g.observed]].to_numpy(float)
    R = R[~np.isnan(R).any(1)]
    # latent GT: the 4 construct descriptions shipped in TLVD's own description file. The latent->description
    # MAPPING below is a DERIVATION from the graph (stated, not a released artifact):
    #   L4 -> X_Speed (drives the speed tasks)      L3 -> X_Error (drives the error tasks)
    #   L2 -> Cognitive Ability ("answering questions correctly" per its description; drives Ques Part2/3)
    #   L1 -> Cognitive Processing Speed (higher-order; TLVD names it "Speed")
    dd = {k: v["description"] for k, v in desc.items()}
    latent_gt = {"L4": f"task completion speed ({dd['X_Speed']})",
                 "L3": f"error rate ({dd['X_Error']})",
                 "L2": f"cognitive ability ({dd['Cognitive Ability']})",
                 "L1": f"cognitive processing speed ({dd['Cognitive Processing Speed']})"}
    return dict(name="tlvd", graph=g, X=z(R), labels=labels, latent_gt=latent_gt)


# --------------------------------------------------------------------------- Himi (design bipartite graph)
HIMI_CONSTRUCT = {
    "Number_Letter": "shifting", "Category_Switch": "shifting", "Color_Shape": "shifting",
    "Keep_Track": "updating", "Letter_Memory": "updating", "N_Back": "updating",
    "Antisaccade": "inhibition", "Stop_Signal": "inhibition", "Stroop": "inhibition",
    "OSpan_PCU": "working memory", "RSpan_PCU": "working memory", "SSpan_PCU": "working memory",
    "RI_F": "relational integration", "RI_N": "relational integration", "RI_V": "relational integration",
    "DA_Uni": "divided attention", "DA_Cross": "divided attention"}
HIMI_DESC = {
    "Number_Letter": "number-letter task switching", "Category_Switch": "category switch task",
    "Color_Shape": "color-shape task switching", "Keep_Track": "keep track memory updating",
    "Letter_Memory": "letter memory updating", "N_Back": "n-back working memory updating",
    "Antisaccade": "antisaccade response inhibition", "Stop_Signal": "stop-signal response inhibition",
    "Stroop": "stroop color-word interference inhibition", "OSpan_PCU": "operation span working memory",
    "RSpan_PCU": "reading span working memory", "SSpan_PCU": "symmetry span working memory",
    "RI_F": "relational integration figural", "RI_N": "relational integration numerical",
    "RI_V": "relational integration verbal", "DA_Uni": "divided attention unimodal",
    "DA_Cross": "divided attention crossmodal"}


def himi():
    import pyreadstat
    g = G.bipartite(HIMI_CONSTRUCT)
    df, _ = pyreadstat.read_sav(os.path.join(DATA, "TLVD", "Final_Multitasking_Data.sav"))
    R = df[g.observed].to_numpy(float)
    R = R[~np.isnan(R).any(1)]
    latent_gt = {c: f"the executive-function construct: {c}" for c in g.latents}
    return dict(name="himi", graph=g, X=z(R), labels=dict(HIMI_DESC), latent_gt=latent_gt)


# --------------------------------------------------------------------------- Big Five (design bipartite)
BIG5_FACTOR = {"E": "extraversion", "N": "neuroticism", "A": "agreeableness",
               "C": "conscientiousness", "O": "openness"}


def bigfive(nsub=3000):
    import re, pandas as pd
    item_text = {}
    for line in open(os.path.join(DATA, "BIG5", "codebook.txt")):
        m = re.match(r"^([ENACO]\d{1,2})\t(.+)$", line.strip())
        if m:
            item_text[m.group(1)] = m.group(2)
    construct_of = {c: BIG5_FACTOR[c[0]] for c in item_text}
    g = G.bipartite(construct_of)
    df = pd.read_csv(os.path.join(DATA, "BIG5", "data.csv"), sep="\t")
    R = df[g.observed].to_numpy(float)
    R[R == 0] = np.nan
    R = R[~np.isnan(R).any(1)]
    if len(R) > nsub:
        R = R[np.random.default_rng(0).choice(len(R), nsub, replace=False)]
    latent_gt = {f: f"the personality factor: {f}" for f in g.latents}
    return dict(name="bigfive", graph=g, X=z(R), labels=item_text, latent_gt=latent_gt)


# --------------------------------------------------------------------------- Controlled oracle graphs
ORACLE_DATASETS = (
    "oracle_clean",
    "oracle_polarity",
    "oracle_mixed_parent",
    "oracle_sparse_sibling",
)


def _oracle_folder(name):
    roots = [ORACLE_DATA, os.path.join(HERE, "data"), DATA]
    seen = set()
    for root in roots:
        root = os.path.abspath(root)
        if root in seen:
            continue
        seen.add(root)
        folder = os.path.join(root, name)
        if os.path.isfile(os.path.join(folder, "graph.dot")):
            return folder
    raise FileNotFoundError(
        f"oracle dataset '{name}' is missing. Run scripts/make_oracle_datasets.py first; "
        f"searched: {', '.join(os.path.join(root, name) for root in roots)}"
    )


def oracle(name):
    folder = _oracle_folder(name)
    g = G.from_dot(os.path.join(folder, "graph.dot"))
    labels = {}
    with open(os.path.join(folder, "codebook.txt"), encoding="utf-8") as handle:
        for line in handle:
            node, text = line.rstrip("\n").split("\t", 1)
            labels[node] = text
    with open(os.path.join(folder, "data.csv"), newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [[float(row[node]) for node in g.observed] for row in reader]
    latent_gt = json.load(open(os.path.join(folder, "latent_labels.json"), encoding="utf-8"))
    metadata = json.load(open(os.path.join(folder, "oracle_metadata.json"), encoding="utf-8"))
    return dict(name=name, graph=g, X=z(np.asarray(rows, dtype=float)), labels=labels,
                latent_gt=latent_gt, oracle_metadata=metadata)


def oracle_clean():
    return oracle("oracle_clean")


def oracle_polarity():
    return oracle("oracle_polarity")


def oracle_mixed_parent():
    return oracle("oracle_mixed_parent")


def oracle_sparse_sibling():
    return oracle("oracle_sparse_sibling")


DEFAULT_TASK1_DATASETS = ("tlvd", "himi", "bigfive")
LOADERS = {
    "tlvd": tlvd,
    "himi": himi,
    "bigfive": bigfive,
    "oracle_clean": oracle_clean,
    "oracle_polarity": oracle_polarity,
    "oracle_mixed_parent": oracle_mixed_parent,
    "oracle_sparse_sibling": oracle_sparse_sibling,
}
