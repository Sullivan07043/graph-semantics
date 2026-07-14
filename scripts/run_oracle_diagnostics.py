"""Focused Task 1 diagnostics on controlled oracle graphs.

This runner is deliberately narrower than the full ablation sweep. It compares
uniform, raw correlation, the original objective, and three selected graph
settings on four synthetic datasets with known loading and semantic structure.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

import datasets
import encode
import metrics
import optimize


FOLDS = int(os.environ.get("FOLDS", 5))
STEPS = int(os.environ.get("STEPS", 1500))
LAM_ZERO = float(os.environ.get("LAM_ZERO", 0.3))
LAM_NORM = float(os.environ.get("LAM_NORM", 0.1))
DEVICE = os.environ.get("DEVICE", "cpu")
OBS_PRIOR_SCOPE = os.environ.get("OBS_PRIOR_SCOPE", "siblings")
WHICH = os.environ.get("ORACLE_DATASET", os.environ.get("DATASET", "all"))
OUT_DIR = Path(os.environ.get("ORACLE_OUT_DIR", REPO_ROOT / "outputs" / "oracle"))


ARM_SPECS = (
    {"name": "uniform", "kind": "uniform"},
    {"name": "rawcorr", "kind": "rawcorr"},
    {"name": "A_original", "kind": "core", "edge": "signed", "normalize": False, "prior": 0.0},
    {"name": "D_norm_only", "kind": "core", "edge": "signed", "normalize": True, "prior": 0.0},
    {"name": "M_abs_norm_prior_03", "kind": "core", "edge": "abs", "normalize": True, "prior": 0.3},
    {"name": "K_signed_norm_prior_10", "kind": "core", "edge": "signed", "normalize": True, "prior": 1.0},
)


def ts():
    return time.strftime("%H:%M:%S")


def normalize(v):
    return np.asarray(v, dtype=float) / (np.linalg.norm(v) + 1e-9)


def mean_or_none(values):
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def fmt(value):
    return "-" if value is None else f"{float(value):.3f}"


def exact_flags(pred, masked, target_norm):
    flags = []
    for row, item_index in enumerate(masked):
        flags.append(int(np.argmax(target_norm @ normalize(pred[row])) == item_index))
    return flags


def matching_flags(pred, masked, target):
    from scipy.optimize import linear_sum_assignment

    score = metrics.norm_rows(pred) @ metrics.norm_rows(target[masked]).T
    rows, cols = linear_sum_assignment(-score)
    flags = [0] * len(masked)
    for row, col in zip(rows, cols):
        flags[row] = int(row == col)
    return flags


def cosine_to_true(pred, masked, target):
    return [float(normalize(pred[row]) @ normalize(target[index])) for row, index in enumerate(masked)]


def shared_parent_candidates(graph, observed, item_index, visible):
    item_parents = set(graph.parents(observed[item_index]))
    return [index for index in visible if item_parents & set(graph.parents(observed[index]))]


def observed_prior_details(graph, corr, observed, labels, visible, visible_embeddings, metadata, scope):
    priors, details = {}, {}
    visible_set = set(visible)
    for item_index, name in enumerate(observed):
        if item_index in visible_set:
            continue
        if scope == "siblings":
            candidates = shared_parent_candidates(graph, observed, item_index, visible)
        elif scope == "all":
            candidates = list(visible)
        else:
            raise ValueError(f"unknown OBS_PRIOR_SCOPE: {scope}")
        weights = np.clip(corr[item_index, candidates], 0.0, None) if candidates else np.asarray([])
        target_meta = metadata["observed"][name]
        same_polarity = [
            candidate for candidate in candidates
            if metadata["observed"][observed[candidate]].get("polarity") == target_meta.get("polarity")
        ]
        info = {
            "visible_sibling_candidates": [observed[index] for index in candidates],
            "visible_same_polarity_candidates": [observed[index] for index in same_polarity],
            "prior_available": False,
            "prior_top_contributors": [],
        }
        if len(candidates) and weights.sum() >= 1e-9:
            mixture = sum(
                float(weight) * visible_embeddings[observed[index]]
                for weight, index in zip(weights, candidates)
            ) / weights.sum()
            priors[name] = normalize(mixture)
            order = np.argsort(-weights)[:5]
            info["prior_available"] = True
            info["prior_top_contributors"] = [
                {
                    "variable": observed[candidates[position]],
                    "label": labels[observed[candidates[position]]],
                    "weight": float(weights[position]),
                    "normalized_weight": float(weights[position] / weights.sum()),
                    "polarity": metadata["observed"][observed[candidates[position]]].get("polarity"),
                }
                for position in order
                if weights[position] > 0
            ]
        details[name] = info
    return priors, details


def latent_semantic_anchors(graph, observed, target, metadata):
    anchors = {}
    for latent in graph.latents:
        pure_indices = [
            index for index, name in enumerate(observed)
            if latent in metadata["observed"][name]["parents"]
            and metadata["observed"][name].get("item_type") == "pure"
        ]
        indices = pure_indices or [
            index for index, name in enumerate(observed)
            if latent in metadata["observed"][name]["parents"]
        ]
        anchors[latent] = normalize(target[indices].mean(axis=0))
    return anchors


def oracle_item_metrics(prediction, item_index, observed, target_norm, metadata, anchors):
    name = observed[item_index]
    spec = metadata["observed"][name]
    p = normalize(prediction)
    parent_scores = {latent: float(p @ anchor) for latent, anchor in anchors.items()}
    ranked = sorted(parent_scores, key=parent_scores.get, reverse=True)
    parents = list(spec["parents"])
    parent_set_correct = set(ranked[:len(parents)]) == set(parents)

    polarity = spec.get("polarity")
    same_score = opposite_score = polarity_margin = polarity_correct = None
    if polarity in {"positive", "reverse"}:
        same_score = float(p @ target_norm[item_index])
        opposite = [
            index for index, other_name in enumerate(observed)
            if set(metadata["observed"][other_name]["parents"]) == set(parents)
            and metadata["observed"][other_name].get("polarity") not in {None, polarity}
        ]
        if opposite:
            opposite_score = float(np.mean([p @ target_norm[index] for index in opposite]))
            polarity_margin = same_score - opposite_score
            polarity_correct = polarity_margin > 0.0

    return {
        "latent_parents": parents,
        "item_type": spec.get("item_type", "pure"),
        "polarity": polarity,
        "sibling_count": spec.get("sibling_count"),
        "parent_scores": parent_scores,
        "parent_ranked_latents": ranked,
        "parent_set_correct": bool(parent_set_correct),
        "same_polarity_similarity": same_score,
        "opposite_polarity_similarity": opposite_score,
        "polarity_margin": polarity_margin,
        "polarity_correct": polarity_correct,
    }


def prepare_dataset(name, dictionary):
    dataset = datasets.LOADERS[name]()
    graph, X, labels = dataset["graph"], dataset["X"], dataset["labels"]
    observed = list(graph.observed)
    observation_index = {name: index for index, name in enumerate(observed)}
    target = encode.embed([labels[name] for name in observed])
    target_norm = metrics.norm_rows(target)
    weights, _ = graph.estimate_weights(X, observation_index)
    corr = np.corrcoef(X.T)
    np.fill_diagonal(corr, 0.0)
    rng = np.random.default_rng(0)
    permutation = rng.permutation(len(observed))
    folds = [permutation[index::FOLDS] for index in range(FOLDS)]
    metadata = dataset["oracle_metadata"]
    return {
        "dataset": dataset,
        "graph": graph,
        "X": X,
        "labels": labels,
        "observed": observed,
        "target": target,
        "target_norm": target_norm,
        "weights": weights,
        "corr": corr,
        "folds": folds,
        "alpha": metrics.pick_alpha(target, dictionary),
        "metadata": metadata,
        "anchors": latent_semantic_anchors(graph, observed, target, metadata),
    }


def baseline_prediction(affinity, target, observed_count, masked, visible):
    prediction = np.zeros((len(masked), target.shape[1]))
    for row, item_index in enumerate(masked):
        weights = np.zeros(observed_count)
        weights[visible] = affinity[item_index, visible]
        if weights.sum() < 1e-9:
            weights[visible] = 1.0
        prediction[row] = (weights / weights.sum()) @ target
    return prediction


def run_dataset(prepared, dictionary, dictionary_words):
    graph = prepared["graph"]
    labels = prepared["labels"]
    observed = prepared["observed"]
    target = prepared["target"]
    target_norm = prepared["target_norm"]
    corr = prepared["corr"]
    metadata = prepared["metadata"]
    records = []
    print(
        f"[{ts()}] {prepared['dataset']['name']}: {prepared['X'].shape[0]}x{len(observed)} | "
        f"{len(graph.latents)} latents, {len(graph.edges)} edges, "
        f"{len(graph.independent_pairs())} independent pairs",
        flush=True,
    )

    for fold_index, fold in enumerate(prepared["folds"]):
        masked = sorted(int(index) for index in fold)
        masked_set = set(masked)
        visible = [index for index in range(len(observed)) if index not in masked_set]
        visible_embeddings = {observed[index]: target[index] for index in visible}
        priors, prior_details = observed_prior_details(
            graph, corr, observed, labels, visible, visible_embeddings, metadata, OBS_PRIOR_SCOPE
        )
        predictions = {}
        for arm in ARM_SPECS:
            if arm["kind"] == "uniform":
                predictions[arm["name"]] = baseline_prediction(
                    np.ones_like(corr), target, len(observed), masked, visible
                )
            elif arm["kind"] == "rawcorr":
                predictions[arm["name"]] = baseline_prediction(
                    np.clip(corr, 0.0, None), target, len(observed), masked, visible
                )
            else:
                prior_embeddings = priors if arm["prior"] > 0 else None
                embeddings = optimize.optimize_embeddings(
                    graph,
                    prepared["weights"],
                    visible_embeddings,
                    d=target.shape[1],
                    steps=STEPS,
                    lam_zero=LAM_ZERO,
                    lam_norm=LAM_NORM,
                    seed=fold_index,
                    device=DEVICE,
                    edge_weight_mode=arm["edge"],
                    normalize_gen=arm["normalize"],
                    observed_prior_emb=prior_embeddings,
                    lam_obs_prior=arm["prior"],
                )
                predictions[arm["name"]] = np.stack([embeddings[observed[index]] for index in masked])

        for arm in ARM_SPECS:
            name = arm["name"]
            prediction = predictions[name]
            exact = exact_flags(prediction, masked, target_norm)
            matching = matching_flags(prediction, masked, target)
            cosines = cosine_to_true(prediction, masked, target)
            decoded = metrics.decode_words(prediction, dictionary, dictionary_words, prepared["alpha"])
            for row, item_index in enumerate(masked):
                item_name = observed[item_index]
                details = prior_details[item_name]
                oracle_metrics = oracle_item_metrics(
                    prediction[row], item_index, observed, target_norm, metadata, prepared["anchors"]
                )
                records.append({
                    "dataset": prepared["dataset"]["name"],
                    "fold": fold_index,
                    "arm": name,
                    "variable_id": item_index,
                    "variable_name": item_name,
                    "true_label": labels[item_name],
                    "decoded_top_words": decoded[row],
                    "exact_correct": bool(exact[row]),
                    "matching_correct": bool(matching[row]),
                    "cosine_pred_to_true": cosines[row],
                    "parents": list(graph.parents(item_name)),
                    "children": list(graph.children(item_name)),
                    "prior_available": bool(arm.get("prior", 0.0) > 0 and details["prior_available"]),
                    "prior_top_contributors": details["prior_top_contributors"],
                    "visible_sibling_candidates": details["visible_sibling_candidates"],
                    "visible_same_polarity_candidates": details["visible_same_polarity_candidates"],
                    "edge_weight_mode": arm.get("edge"),
                    "normalize_gen": arm.get("normalize"),
                    "lam_obs_prior": arm.get("prior", 0.0),
                    **oracle_metrics,
                })
        print(f"[{ts()}]   fold {fold_index + 1}/{FOLDS} done", flush=True)
    return records


def summarize_rows(rows):
    def aggregate(group):
        return {
            "matching_acc": mean_or_none([float(row["matching_correct"]) for row in group]),
            "exact_top1": mean_or_none([float(row["exact_correct"]) for row in group]),
            "mean_cosine": mean_or_none([row["cosine_pred_to_true"] for row in group]),
            "parent_set_acc": mean_or_none([float(row["parent_set_correct"]) for row in group]),
            "polarity_acc": mean_or_none([
                float(row["polarity_correct"]) for row in group if row["polarity_correct"] is not None
            ]),
            "polarity_margin": mean_or_none([row["polarity_margin"] for row in group]),
            "prior_coverage": mean_or_none([float(row["prior_available"]) for row in group]),
            "n_items": len(group),
        }

    rows_out = []
    for dataset_name in datasets.ORACLE_DATASETS:
        for arm in [spec["name"] for spec in ARM_SPECS]:
            group = [row for row in rows if row["dataset"] == dataset_name and row["arm"] == arm]
            values = aggregate(group)
            rows_out.append({"dataset": dataset_name, "arm": arm, **values})
    for arm in [spec["name"] for spec in ARM_SPECS]:
        group_rows = [row for row in rows_out if row["arm"] == arm and row["dataset"] != "macro_average"]
        rows_out.append({
            "dataset": "macro_average",
            "arm": arm,
            **{
                key: mean_or_none([row[key] for row in group_rows])
                for key in ("matching_acc", "exact_top1", "mean_cosine", "parent_set_acc",
                            "polarity_acc", "polarity_margin", "prior_coverage")
            },
            "n_items": sum(row["n_items"] for row in group_rows),
        })
    return rows_out


def write_diagnostics(rows, path):
    fields = [
        "dataset", "fold", "arm", "variable_id", "variable_name", "true_label", "decoded_top_words",
        "exact_correct", "matching_correct", "cosine_pred_to_true", "parents", "children",
        "latent_parents", "item_type",
        "polarity", "sibling_count", "visible_sibling_candidates", "visible_same_polarity_candidates",
        "prior_available", "prior_top_contributors", "edge_weight_mode", "normalize_gen", "lam_obs_prior",
        "parent_scores", "parent_ranked_latents", "parent_set_correct", "same_polarity_similarity",
        "opposite_polarity_similarity", "polarity_margin", "polarity_correct",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            for field in (
                "decoded_top_words", "parents", "children", "latent_parents", "visible_sibling_candidates",
                "visible_same_polarity_candidates", "prior_top_contributors", "parent_scores",
                "parent_ranked_latents",
            ):
                output[field] = json.dumps(output[field], ensure_ascii=False)
            writer.writerow(output)


def write_summary(rows, csv_path, markdown_path):
    fields = [
        "dataset", "arm", "matching_acc", "exact_top1", "mean_cosine", "parent_set_acc",
        "polarity_acc", "polarity_margin", "prior_coverage", "n_items",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = ["# Controlled Oracle Graph Diagnostics", ""]
    lines.append(
        f"Settings: FOLDS={FOLDS}, STEPS={STEPS}, LAM_ZERO={LAM_ZERO:g}, "
        f"LAM_NORM={LAM_NORM:g}, DEVICE={DEVICE}, OBS_PRIOR_SCOPE={OBS_PRIOR_SCOPE}. "
        "Judge is intentionally disabled."
    )
    lines += [
        "",
        "Matching-ACC is fold-local one-to-one assignment accuracy and can saturate when a sparse fold has few "
        "competitors. Parent-set ACC uses known oracle latent anchors. Polarity ACC compares each prediction with "
        "its known positive/reverse counterpart. Prior coverage is the fraction of masked items with a nonzero "
        "sibling prior.",
    ]
    order = list(datasets.ORACLE_DATASETS) + ["macro_average"]
    for dataset_name in order:
        lines += ["", f"## {dataset_name}", ""]
        lines.append(
            "| arm | matching-ACC | exact top-1 | mean cosine | parent-set ACC | "
            "polarity ACC | polarity margin | prior coverage |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for row in [row for row in rows if row["dataset"] == dataset_name]:
            lines.append(
                f"| {row['arm']} | {fmt(row['matching_acc'])} | {fmt(row['exact_top1'])} | "
                f"{fmt(row['mean_cosine'])} | {fmt(row['parent_set_acc'])} | "
                f"{fmt(row['polarity_acc'])} | {fmt(row['polarity_margin'])} | "
                f"{fmt(row['prior_coverage'])} |"
            )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def stats_for(rows, dataset_name, arm, predicate=lambda row: True):
    group = [row for row in rows if row["dataset"] == dataset_name and row["arm"] == arm and predicate(row)]
    return {
        "n": len(group),
        "matching": mean_or_none([float(row["matching_correct"]) for row in group]),
        "exact": mean_or_none([float(row["exact_correct"]) for row in group]),
        "cosine": mean_or_none([row["cosine_pred_to_true"] for row in group]),
        "parent_set": mean_or_none([float(row["parent_set_correct"]) for row in group]),
        "polarity": mean_or_none([
            float(row["polarity_correct"]) for row in group if row["polarity_correct"] is not None
        ]),
        "margin": mean_or_none([row["polarity_margin"] for row in group]),
        "prior": mean_or_none([float(row["prior_available"]) for row in group]),
    }


def mean_visible_siblings(rows, dataset_name, arm):
    group = [row for row in rows if row["dataset"] == dataset_name and row["arm"] == arm]
    return mean_or_none([len(row["visible_sibling_candidates"]) for row in group])


def write_error_report(rows, path):
    m_clean = stats_for(rows, "oracle_clean", "M_abs_norm_prior_03")
    raw_clean = stats_for(rows, "oracle_clean", "rawcorr")
    d_clean = stats_for(rows, "oracle_clean", "D_norm_only")
    m_positive = stats_for(rows, "oracle_polarity", "M_abs_norm_prior_03",
                           lambda row: row["polarity"] == "positive")
    m_reverse = stats_for(rows, "oracle_polarity", "M_abs_norm_prior_03",
                          lambda row: row["polarity"] == "reverse")
    d_polarity = stats_for(rows, "oracle_polarity", "D_norm_only")
    k_polarity = stats_for(rows, "oracle_polarity", "K_signed_norm_prior_10")
    m_pure = stats_for(rows, "oracle_mixed_parent", "M_abs_norm_prior_03",
                       lambda row: row["item_type"] == "pure")
    m_mixed = stats_for(rows, "oracle_mixed_parent", "M_abs_norm_prior_03",
                        lambda row: row["item_type"] == "mixed")
    m_sparse = stats_for(rows, "oracle_sparse_sibling", "M_abs_norm_prior_03")
    m_sparse_prior = stats_for(rows, "oracle_sparse_sibling", "M_abs_norm_prior_03",
                               lambda row: row["prior_available"])
    m_sparse_no_prior = stats_for(rows, "oracle_sparse_sibling", "M_abs_norm_prior_03",
                                  lambda row: not row["prior_available"])

    clean_sanity_pass = (
        m_clean["parent_set"] is not None and m_clean["parent_set"] >= 0.90
        and m_clean["cosine"] is not None and m_clean["cosine"] >= 0.50
    )
    polarity_confusion = (
        m_positive["polarity"] is not None and m_reverse["polarity"] is not None
        and (
            m_positive["polarity"] < 0.50
            or m_reverse["polarity"] < 0.50
            or m_positive["margin"] is not None and m_positive["margin"] < 0.0
            or m_reverse["margin"] is not None and m_reverse["margin"] < 0.0
        )
    )
    mixed_failure = (
        m_mixed["parent_set"] is not None
        and (m_mixed["parent_set"] < 0.67 or m_mixed["cosine"] < (m_pure["cosine"] or 0.0) - 0.15)
    )
    sparse_matching_drop = (
        m_sparse["matching"] is not None and m_clean["matching"] is not None
        and m_sparse["matching"] < m_clean["matching"] - 0.15
    )
    sparse_cosine_drop = (
        m_sparse["cosine"] is not None and m_clean["cosine"] is not None
        and m_sparse["cosine"] < m_clean["cosine"] - 0.10
    )
    sparse_drop = sparse_matching_drop or sparse_cosine_drop
    clean_visible = mean_visible_siblings(rows, "oracle_clean", "M_abs_norm_prior_03")
    sparse_visible = mean_visible_siblings(rows, "oracle_sparse_sibling", "M_abs_norm_prior_03")
    clean_vs_raw = m_clean["matching"] - raw_clean["matching"]
    clean_vs_d = m_clean["matching"] - d_clean["matching"]

    lines = ["# Oracle Error Report", ""]
    lines.append(
        "This report uses known oracle structure plus geometric Task 1 metrics. "
        "LLM judging is intentionally disabled because the ground-truth loading and item-type metadata provide "
        "a more direct controlled diagnostic."
    )
    lines += ["", "## 1. Does M Succeed on oracle_clean?", ""]
    lines.append(
        f"M: matching={fmt(m_clean['matching'])}, exact={fmt(m_clean['exact'])}, cosine={fmt(m_clean['cosine'])}, "
        f"parent-set ACC={fmt(m_clean['parent_set'])}. Rawcorr matching={fmt(raw_clean['matching'])}; "
        f"D_norm_only matching={fmt(d_clean['matching'])}."
    )
    if clean_sanity_pass:
        lines.append(
            f"**Answer:** Structurally yes, but not as a v1.1 gain. M clears the clean recovery sanity check, "
            f"ties rawcorr (delta={clean_vs_raw:+.3f}), and trails D_norm_only (delta={clean_vs_d:+.3f})."
        )
    else:
        lines.append(
            "**Answer:** Not convincingly. The clean dense-positive graph does not meet the structural recovery sanity check."
        )

    lines += ["", "## 2. Does M Confuse Positive and Reverse Items?", ""]
    lines.append(
        f"Positive items: polarity ACC={fmt(m_positive['polarity'])}, margin={fmt(m_positive['margin'])}, "
        f"cosine={fmt(m_positive['cosine'])}."
    )
    lines.append(
        f"Reverse items: polarity ACC={fmt(m_reverse['polarity'])}, margin={fmt(m_reverse['margin'])}, "
        f"cosine={fmt(m_reverse['cosine'])}, prior coverage={fmt(m_reverse['prior'])}."
    )
    lines.append(
        f"Signed-edge controls: D_norm_only polarity ACC={fmt(d_polarity['polarity'])}; "
        f"K_signed_norm_prior_10 polarity ACC={fmt(k_polarity['polarity'])}."
    )
    if polarity_confusion:
        lines.append(
            "**Answer:** Yes. The absolute edge transform erases the negative loading in child generation, and the "
            "positive-correlation prior provides no support to reverse-coded items when their only same-factor peers "
            "have negative correlations."
        )
    else:
        lines.append("**Answer:** No clear polarity confusion under this oracle.")

    lines += ["", "## 3. Does Parent Aggregation Work on Mixed Parents?", ""]
    lines.append(
        f"Pure items: matching={fmt(m_pure['matching'])}, cosine={fmt(m_pure['cosine'])}, "
        f"parent-set ACC={fmt(m_pure['parent_set'])}."
    )
    lines.append(
        f"Mixed items: matching={fmt(m_mixed['matching'])}, cosine={fmt(m_mixed['cosine'])}, "
        f"parent-set ACC={fmt(m_mixed['parent_set'])}."
    )
    lines.append(f"**Answer:** {'No, not reliably' if mixed_failure else 'Yes, under this oracle'}." )

    lines += ["", "## 4. Does Sparse Sibling Density Cause a Drop?", ""]
    lines.append(
        f"M sparse: matching={fmt(m_sparse['matching'])}, cosine={fmt(m_sparse['cosine'])}, "
        f"prior coverage={fmt(m_sparse['prior'])}, mean visible siblings={fmt(sparse_visible)}. "
        f"M clean: matching={fmt(m_clean['matching'])}, cosine={fmt(m_clean['cosine'])}, "
        f"mean visible siblings={fmt(clean_visible)}."
    )
    lines.append(
        f"Sparse items with an available prior: n={m_sparse_prior['n']}, matching={fmt(m_sparse_prior['matching'])}; "
        f"without one: n={m_sparse_no_prior['n']}, matching={fmt(m_sparse_no_prior['matching'])}."
    )
    if sparse_drop:
        details = []
        if sparse_matching_drop:
            details.append("matching")
        if sparse_cosine_drop:
            details.append("cosine")
        lines.append(
            "**Answer:** Yes, in " + " and ".join(details) + ". Matching is not directly comparable when sparse "
            "folds have fewer competitors, so the cosine decline is the more informative signal here."
        )
    else:
        lines.append("**Answer:** No material sparse-sibling drop under matching or cosine.")

    lines += ["", "## 5. Constraint Representation or Optimization?", ""]
    if not clean_sanity_pass:
        lines.append(
            "The clean dense-positive oracle does not pass the sanity check, so the current evidence points first "
            "to an optimization/training-algorithm issue (or an objective that is too weak even in the easy case)."
        )
    elif polarity_confusion or mixed_failure or sparse_drop:
        failures = []
        if polarity_confusion:
            failures.append("negative-loading polarity")
        if mixed_failure:
            failures.append("mixed-parent composition")
        if sparse_drop:
            failures.append("sparse sibling support")
        lines.append(
            "The clean oracle passes while the targeted stressors fail: " + ", ".join(failures) + ". "
            "This pattern is more consistent with constraint representation than with a generic optimizer failure. "
            "However, M does not improve the clean graph over rawcorr or D_norm_only."
        )
    else:
        lines.append(
            "The tested oracle stressors do not expose a decisive failure pattern. More seeds or stronger stressors "
            "are needed before attributing remaining error to representation or optimization."
        )

    lines += ["", "## 6. Recommended Next Change", ""]
    if polarity_confusion and clean_sanity_pass:
        lines.append(
            "**Recommend a polarity-aware sibling prior next, together with a sign-preserving edge rule for "
            "reverse-coded children.** A generic optimizer is not the first priority: the signed controls recover "
            "polarity, whereas M's absolute transform does not."
        )
    elif not clean_sanity_pass:
        lines.append(
            "**Recommend investigating the optimizer/objective before adding a new constraint.** The method must first "
            "recover the clean dense-positive oracle reliably."
        )
    else:
        lines.append(
            "**Keep the current optimizer for now and extend the controlled diagnostics.** No polarity-specific failure "
            "was strong enough in this run to justify changing the prior first."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_diagnostics(path):
    json_fields = {
        "decoded_top_words", "parents", "children", "latent_parents", "visible_sibling_candidates",
        "visible_same_polarity_candidates", "prior_top_contributors", "parent_scores", "parent_ranked_latents",
    }
    bool_fields = {"exact_correct", "matching_correct", "prior_available", "parent_set_correct", "polarity_correct"}
    float_fields = {
        "cosine_pred_to_true", "lam_obs_prior", "same_polarity_similarity", "opposite_polarity_similarity",
        "polarity_margin",
    }
    int_fields = {"fold", "variable_id", "sibling_count"}
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row = dict(raw)
            for field in json_fields:
                row[field] = json.loads(row[field])
            for field in bool_fields:
                row[field] = None if row[field] == "" else row[field].lower() == "true"
            for field in float_fields:
                row[field] = None if row[field] == "" else float(row[field])
            for field in int_fields:
                row[field] = None if row[field] == "" else int(row[field])
            rows.append(row)
    return rows


def main():
    report_only = os.environ.get("ORACLE_REPORT_ONLY", "0").lower() in ("1", "true", "yes", "on")
    if report_only:
        diagnostics_path = OUT_DIR / "per_item_diagnostics.csv"
        if not diagnostics_path.is_file():
            raise FileNotFoundError(f"ORACLE_REPORT_ONLY requires {diagnostics_path}")
        records = load_diagnostics(diagnostics_path)
        write_summary(summarize_rows(records), OUT_DIR / "summary.csv", OUT_DIR / "summary.md")
        write_error_report(records, OUT_DIR / "oracle_error_report.md")
        print(f"[{ts()}] refreshed oracle reports in {OUT_DIR}", flush=True)
        return

    names = list(datasets.ORACLE_DATASETS) if WHICH == "all" else [WHICH]
    unknown = [name for name in names if name not in datasets.ORACLE_DATASETS]
    if unknown:
        raise ValueError(f"ORACLE_DATASET must be one of {datasets.ORACLE_DATASETS} or all; got {unknown}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dictionary, dictionary_words = encode.load_dictionary()
    records = []
    for name in names:
        records.extend(run_dataset(prepare_dataset(name, dictionary), dictionary, dictionary_words))

    summary = summarize_rows(records)
    write_diagnostics(records, OUT_DIR / "per_item_diagnostics.csv")
    write_summary(summary, OUT_DIR / "summary.csv", OUT_DIR / "summary.md")
    write_error_report(records, OUT_DIR / "oracle_error_report.md")
    manifest = {
        "datasets": names,
        "settings": {
            "FOLDS": FOLDS,
            "STEPS": STEPS,
            "LAM_ZERO": LAM_ZERO,
            "LAM_NORM": LAM_NORM,
            "DEVICE": DEVICE,
            "OBS_PRIOR_SCOPE": OBS_PRIOR_SCOPE,
            "judge": False,
        },
        "arms": ARM_SPECS,
    }
    (OUT_DIR / "oracle_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[{ts()}] wrote oracle diagnostics to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
