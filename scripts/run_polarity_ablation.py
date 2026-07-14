"""Small polarity-focused Task 1 ablation on oracle_polarity.

This runner separates edge-sign handling from sibling-prior construction. It
does not call an LLM judge and does not change the default Task 1 objective.
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
sys.path.insert(0, str(SCRIPT_DIR))

import encode
import metrics
import optimize
import run_oracle_diagnostics as oracle


OUT_DIR = Path(os.environ.get("POLARITY_OUT_DIR", REPO_ROOT / "outputs" / "polarity_ablation"))
FOLDS = int(os.environ.get("FOLDS", 5))
STEPS = int(os.environ.get("STEPS", 1500))
LAM_ZERO = float(os.environ.get("LAM_ZERO", 0.3))
LAM_NORM = float(os.environ.get("LAM_NORM", 0.1))
DEVICE = os.environ.get("DEVICE", "cpu")


ARM_SPECS = (
    {"name": "uniform", "kind": "uniform"},
    {"name": "rawcorr", "kind": "rawcorr"},
    {"name": "D_signed_no_prior", "kind": "core", "edge": "signed", "prior": 0.0, "prior_mode": "none"},
    {"name": "E_abs_no_prior", "kind": "core", "edge": "abs", "prior": 0.0, "prior_mode": "none"},
    {"name": "J_signed_corr_prior_03", "kind": "core", "edge": "signed", "prior": 0.3,
     "prior_mode": "corr_positive"},
    {"name": "M_abs_corr_prior_03", "kind": "core", "edge": "abs", "prior": 0.3,
     "prior_mode": "corr_positive"},
    {"name": "P_signed_loading_prior_03", "kind": "core", "edge": "signed", "prior": 0.3,
     "prior_mode": "loading_same"},
    {"name": "Q_abs_loading_prior_03", "kind": "core", "edge": "abs", "prior": 0.3,
     "prior_mode": "loading_same"},
)


def ts():
    return time.strftime("%H:%M:%S")


def mean(values):
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def fmt(value):
    return "-" if value is None else f"{float(value):.3f}"


def build_priors(prepared, visible, visible_embeddings, mode):
    return optimize.observed_label_priors(
        prepared["graph"],
        prepared["corr"],
        prepared["observed"],
        visible,
        visible_embeddings,
        scope="siblings",
        polarity_mode=mode,
        edge_weights=prepared["weights"],
        return_details=True,
    )


def run_ablation(prepared, dictionary, dictionary_words):
    graph = prepared["graph"]
    observed = prepared["observed"]
    labels = prepared["labels"]
    target = prepared["target"]
    target_norm = prepared["target_norm"]
    corr = prepared["corr"]
    metadata = prepared["metadata"]
    records = []
    pair_deltas = {"J_vs_P": [], "M_vs_Q": []}
    prior_deltas = []

    print(
        f"[{ts()}] oracle_polarity: {prepared['X'].shape[0]}x{len(observed)} | "
        f"folds={FOLDS}, steps={STEPS}, device={DEVICE}",
        flush=True,
    )

    for fold_index, fold in enumerate(prepared["folds"]):
        masked = sorted(int(index) for index in fold)
        masked_set = set(masked)
        visible = [index for index in range(len(observed)) if index not in masked_set]
        visible_embeddings = {observed[index]: target[index] for index in visible}
        corr_priors, corr_details = build_priors(
            prepared, visible, visible_embeddings, "corr_positive"
        )
        loading_priors, loading_details = build_priors(
            prepared, visible, visible_embeddings, "loading_same"
        )

        for item_index in masked:
            item = observed[item_index]
            if item in corr_priors and item in loading_priors:
                prior_deltas.append(float(np.linalg.norm(corr_priors[item] - loading_priors[item])))

        predictions = {
            "uniform": oracle.baseline_prediction(
                np.ones_like(corr), target, len(observed), masked, visible
            ),
            "rawcorr": oracle.baseline_prediction(
                np.clip(corr, 0.0, None), target, len(observed), masked, visible
            ),
        }
        for arm in ARM_SPECS:
            if arm["kind"] != "core":
                continue
            if arm["prior_mode"] == "corr_positive":
                priors = corr_priors
            elif arm["prior_mode"] == "loading_same":
                priors = loading_priors
            else:
                priors = None
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
                normalize_gen=True,
                observed_prior_emb=priors,
                lam_obs_prior=arm["prior"],
            )
            predictions[arm["name"]] = np.stack(
                [embeddings[observed[index]] for index in masked]
            )

        for left, right, key in (
            ("J_signed_corr_prior_03", "P_signed_loading_prior_03", "J_vs_P"),
            ("M_abs_corr_prior_03", "Q_abs_loading_prior_03", "M_vs_Q"),
        ):
            pair_deltas[key].extend(
                np.linalg.norm(predictions[left] - predictions[right], axis=1).tolist()
            )

        for arm in ARM_SPECS:
            arm_name = arm["name"]
            prediction = predictions[arm_name]
            exact = oracle.exact_flags(prediction, masked, target_norm)
            matching = oracle.matching_flags(prediction, masked, target)
            cosines = oracle.cosine_to_true(prediction, masked, target)
            decoded = metrics.decode_words(
                prediction, dictionary, dictionary_words, prepared["alpha"]
            )
            for row_index, item_index in enumerate(masked):
                item = observed[item_index]
                item_meta = metadata["observed"][item]
                loading_by_parent = {
                    parent: float(prepared["weights"].get((parent, item), 0.0))
                    for parent in graph.parents(item)
                }
                prior_mode = arm.get("prior_mode", "none")
                selected_details = (
                    corr_details[item] if prior_mode == "corr_positive"
                    else loading_details[item] if prior_mode == "loading_same"
                    else {
                        "scope_candidates": [],
                        "eligible_candidates": [],
                        "prior_available": False,
                        "contributors": [],
                    }
                )
                oracle_metrics = oracle.oracle_item_metrics(
                    prediction[row_index], item_index, observed, target_norm,
                    metadata, prepared["anchors"]
                )
                records.append({
                    "dataset": "oracle_polarity",
                    "fold": fold_index,
                    "arm": arm_name,
                    "variable_id": item_index,
                    "variable_name": item,
                    "true_label": labels[item],
                    "true_polarity": item_meta.get("polarity"),
                    "estimated_loading_by_parent": loading_by_parent,
                    "decoded_top_words": decoded[row_index],
                    "exact_correct": bool(exact[row_index]),
                    "matching_correct": bool(matching[row_index]),
                    "cosine_pred_to_true": cosines[row_index],
                    "edge_weight_mode": arm.get("edge"),
                    "normalize_gen": arm.get("kind") == "core",
                    "lam_obs_prior": arm.get("prior", 0.0),
                    "prior_mode": prior_mode,
                    "prior_available": bool(
                        arm.get("prior", 0.0) > 0 and selected_details["prior_available"]
                    ),
                    "prior_scope_candidates": selected_details["scope_candidates"],
                    "prior_eligible_candidates": selected_details["eligible_candidates"],
                    "prior_contributors": selected_details["contributors"],
                    **oracle_metrics,
                })
        print(f"[{ts()}]   fold {fold_index + 1}/{FOLDS} done", flush=True)

    comparisons = {
        "max_prior_l2_corr_vs_loading": max(prior_deltas, default=0.0),
        "mean_prior_l2_corr_vs_loading": mean(prior_deltas) or 0.0,
        "max_prediction_l2_J_vs_P": max(pair_deltas["J_vs_P"], default=0.0),
        "max_prediction_l2_M_vs_Q": max(pair_deltas["M_vs_Q"], default=0.0),
    }
    return records, comparisons


def summarize(records):
    rows = []
    for group in ("all", "positive", "reverse"):
        for arm in [spec["name"] for spec in ARM_SPECS]:
            selected = [
                row for row in records
                if row["arm"] == arm and (group == "all" or row["true_polarity"] == group)
            ]
            rows.append({
                "group": group,
                "arm": arm,
                "n_items": len(selected),
                "matching_acc": mean([float(row["matching_correct"]) for row in selected]),
                "exact_top1": mean([float(row["exact_correct"]) for row in selected]),
                "mean_cosine": mean([row["cosine_pred_to_true"] for row in selected]),
                "parent_set_acc": mean([float(row["parent_set_correct"]) for row in selected]),
                "polarity_acc": mean([
                    float(row["polarity_correct"]) for row in selected
                    if row["polarity_correct"] is not None
                ]),
                "polarity_margin": mean([row["polarity_margin"] for row in selected]),
                "prior_coverage": mean([float(row["prior_available"]) for row in selected]),
            })
    return rows


def write_csv(rows, path):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_diagnostics(records, path):
    fields = list(records[0])
    json_fields = {
        "estimated_loading_by_parent", "decoded_top_words", "prior_scope_candidates",
        "prior_eligible_candidates", "prior_contributors", "latent_parents",
        "parent_scores", "parent_ranked_latents",
    }
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = dict(record)
            for field in json_fields:
                row[field] = json.dumps(row[field], ensure_ascii=False)
            writer.writerow(row)


def row_for(rows, arm, group="all"):
    return next(row for row in rows if row["arm"] == arm and row["group"] == group)


def write_summary(rows, comparisons, path):
    lines = [
        "# Small Polarity-Aware Ablation",
        "",
        f"Settings: oracle_polarity only, FOLDS={FOLDS}, STEPS={STEPS}, "
        f"LAM_ZERO={LAM_ZERO:g}, LAM_NORM={LAM_NORM:g}, DEVICE={DEVICE}, judge disabled.",
        "",
        "The core design is edge mode (signed/abs) x prior mode "
        "(none/current positive-correlation/explicit same-loading).",
    ]
    for group in ("all", "positive", "reverse"):
        lines += [
            "",
            f"## {group}",
            "",
            "| arm | matching | exact | cosine | parent-set | polarity | margin | prior coverage |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in [row for row in rows if row["group"] == group]:
            lines.append(
                f"| {row['arm']} | {fmt(row['matching_acc'])} | {fmt(row['exact_top1'])} | "
                f"{fmt(row['mean_cosine'])} | {fmt(row['parent_set_acc'])} | "
                f"{fmt(row['polarity_acc'])} | {fmt(row['polarity_margin'])} | "
                f"{fmt(row['prior_coverage'])} |"
            )
    lines += [
        "",
        "## Equivalence Checks",
        "",
        f"- Max prior L2, current vs explicit loading-aware: "
        f"`{comparisons['max_prior_l2_corr_vs_loading']:.8f}`.",
        f"- Max prediction L2, signed current vs signed loading-aware: "
        f"`{comparisons['max_prediction_l2_J_vs_P']:.8f}`.",
        f"- Max prediction L2, abs current vs abs loading-aware: "
        f"`{comparisons['max_prediction_l2_M_vs_Q']:.8f}`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(rows, comparisons, records, path):
    d = row_for(rows, "D_signed_no_prior")
    e = row_for(rows, "E_abs_no_prior")
    j = row_for(rows, "J_signed_corr_prior_03")
    m = row_for(rows, "M_abs_corr_prior_03")
    p = row_for(rows, "P_signed_loading_prior_03")
    q = row_for(rows, "Q_abs_loading_prior_03")
    d_reverse = row_for(rows, "D_signed_no_prior", "reverse")
    e_reverse = row_for(rows, "E_abs_no_prior", "reverse")
    j_reverse = row_for(rows, "J_signed_corr_prior_03", "reverse")
    m_reverse = row_for(rows, "M_abs_corr_prior_03", "reverse")
    p_reverse = row_for(rows, "P_signed_loading_prior_03", "reverse")
    opposite_contributors = sum(
        contributor.get("loading_relation") == "opposite"
        for row in records if row["prior_mode"] == "corr_positive"
        for contributor in row["prior_contributors"]
    )
    equivalent = (
        comparisons["max_prior_l2_corr_vs_loading"] < 1e-7
        and comparisons["max_prediction_l2_J_vs_P"] < 1e-6
        and comparisons["max_prediction_l2_M_vs_Q"] < 1e-6
    )

    lines = [
        "# Polarity-Aware Ablation Report",
        "",
        "## 1. Does explicit loading-sign filtering change the sibling prior?",
        "",
        f"Max prior L2 difference is `{comparisons['max_prior_l2_corr_vs_loading']:.8f}`; "
        f"opposite-loading contributors admitted by the current prior: `{opposite_contributors}`.",
        f"**Answer:** {'No on this oracle' if equivalent else 'Yes'}. "
        + ("Positive-correlation clipping already selects the same-polarity siblings in the clean linear generator."
           if equivalent else "The loading gate changes at least one prior or prediction."),
        "",
        "## 2. Is the polarity failure caused by the prior or edge generation?",
        "",
        f"Without a prior, signed D has polarity={fmt(d['polarity_acc'])}, cosine={fmt(d['mean_cosine'])}; "
        f"abs E has polarity={fmt(e['polarity_acc'])}, cosine={fmt(e['mean_cosine'])}.",
        f"With the current 0.3 prior, signed J has polarity={fmt(j['polarity_acc'])}, "
        f"cosine={fmt(j['mean_cosine'])}; abs M has polarity={fmt(m['polarity_acc'])}, "
        f"cosine={fmt(m['mean_cosine'])}.",
        f"On reverse items specifically, signed D/J have cosine={fmt(d_reverse['mean_cosine'])}/"
        f"{fmt(j_reverse['mean_cosine'])}, while abs E/M have cosine={fmt(e_reverse['mean_cosine'])}/"
        f"{fmt(m_reverse['mean_cosine'])}.",
        "**Answer:** Edge generation causes the polarity-ACC collapse, not the prior. However, signed generation "
        "is not a semantic solution: it obtains the correct relative polarity by negating the latent vector, while "
        "the reverse-item prediction is strongly anti-aligned with its true text embedding.",
        "",
        "## 3. Does the explicit polarity-aware prior improve M?",
        "",
        f"M current: polarity={fmt(m['polarity_acc'])}, cosine={fmt(m['mean_cosine'])}. "
        f"Q loading-aware: polarity={fmt(q['polarity_acc'])}, cosine={fmt(q['mean_cosine'])}.",
        f"Reverse-item prior coverage under M is {fmt(m_reverse['prior_coverage'])}; "
        f"under signed loading-aware P it is {fmt(p_reverse['prior_coverage'])}.",
        f"**Answer:** {'No; the predictions are numerically identical.' if equivalent else 'The explicit gate changes the result.'}",
        "",
        "## 4. What should happen next?",
        "",
    ]
    if equivalent:
        lines.append(
            "Do not add a more elaborate sibling-filtering rule yet. On this oracle, the existing positive-correlation "
            "prior is already implicitly polarity-selective. The unresolved issue is how a negative-loading edge should "
            "act in semantic embedding space: absolute generation loses direction, while multiplying an embedding by "
            "a negative scalar does not reliably produce an antonym. The next smallest controlled test should use a "
            "separate balanced polarity oracle with at least two reverse items per latent and estimate sign-group semantic "
            "prototypes from visible labels. Only if that non-parametric relation representation is insufficient should "
            "the method learn a positive/reverse relation transform. This remains a constraint-representation problem "
            "before it is a general training-algorithm problem."
        )
    else:
        lines.append(
            "Retain the explicit loading gate for a multi-seed check before changing the optimizer."
        )
    lines += [
        "",
        f"For reference, signed loading-aware P has polarity={fmt(p['polarity_acc'])}, "
        f"cosine={fmt(p['mean_cosine'])}. A useful method must improve both semantic cosine and polarity rather "
        "than trading one for the other.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    if FOLDS != oracle.FOLDS:
        raise ValueError("FOLDS must be set before importing the runner")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dictionary, dictionary_words = encode.load_dictionary()
    prepared = oracle.prepare_dataset("oracle_polarity", dictionary)
    records, comparisons = run_ablation(prepared, dictionary, dictionary_words)
    summary = summarize(records)
    write_csv(summary, OUT_DIR / "summary.csv")
    write_diagnostics(records, OUT_DIR / "per_item_diagnostics.csv")
    write_summary(summary, comparisons, OUT_DIR / "summary.md")
    write_report(summary, comparisons, records, OUT_DIR / "report.md")
    manifest = {
        "dataset": "oracle_polarity",
        "settings": {
            "FOLDS": FOLDS,
            "STEPS": STEPS,
            "LAM_ZERO": LAM_ZERO,
            "LAM_NORM": LAM_NORM,
            "DEVICE": DEVICE,
            "NORMALIZE_GEN": True,
            "OBS_PRIOR_SCOPE": "siblings",
            "judge": False,
        },
        "arms": ARM_SPECS,
        "equivalence_checks": comparisons,
    }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[{ts()}] wrote polarity ablation to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
