"""Validate a CauScale graph with matched J-space interventions.

The unit of analysis is a *prompt pair*, never an individual row.  For each
``pair_id`` this script requires exactly one clean row and one single-node
intervention row.  Legacy additive pilots use the configured signed sigma as
an additive dose.  Absolute hard-set datasets do not: they use the target
shift realized in each matched pair, report setpoint-group paired ATEs, and
test a nonparametric group-ATE magnitude.  CauScale direct edges and reachable
descendants are compared with these paired effects, with a node-label permutation null that
preserves the graph's exact in/out-degree multiset and an edge-count-matched
clean-correlation DAG as controls.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


THIS_FILE = Path(__file__).resolve()
sys.path.insert(0, str(THIS_FILE.parent))

from contracts import load_json, save_json, sha256_file, validate_dataset  # noqa: E402
from graph_postprocess import build_dag  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
            if not isinstance(record, dict):
                raise TypeError(f"row metadata must be objects; got {type(record)!r}")
            records.append(record)
    return records


def resolve_intervention_mode(
    manifest: dict[str, Any], row_records: list[dict[str, Any]], requested: str
) -> dict[str, Any]:
    declared = None
    design = manifest.get("intervention_design")
    if isinstance(design, dict):
        declared = design.get("mode")
    if declared is None:
        declared = manifest.get("intervention_mode")
    row_modes = {
        str(record["intervention_mode"])
        for record in row_records
        if record.get("intervention_mode") is not None
    }
    if len(row_modes) > 1:
        raise ValueError(f"rows.jsonl declares inconsistent intervention modes: {row_modes}")
    if declared is None and row_modes:
        declared = next(iter(row_modes))

    def canonical(value: str) -> str:
        normalized = value.strip().lower().replace("-", "_")
        if normalized in {"hard_set_coordinate", "absolute_hard_set", "hard_set"}:
            return "hard_set_coordinate"
        if normalized in {
            "legacy_additive",
            "additive",
            "additive_direction",
            "single_coordinate_injection",
        }:
            return "legacy_additive"
        raise ValueError(f"unsupported intervention mode {value!r}")

    if requested != "auto":
        mode = canonical(requested)
        if declared is not None and canonical(str(declared)) != mode:
            raise ValueError(
                f"requested intervention mode {mode!r} conflicts with manifest mode {declared!r}"
            )
        source = "explicit_cli"
    elif declared is not None:
        mode = canonical(str(declared))
        source = "manifest_or_rows"
    else:
        # The first additive pilot predates the intervention_design.mode field.
        mode = "legacy_additive"
        source = "legacy_manifest_without_mode"
    return {"mode": mode, "declared_value": declared, "resolution_source": source}


def paired_responses(
    X: np.ndarray,
    interventions: np.ndarray,
    strengths: np.ndarray,
    row_records: list[dict[str, Any]],
    clean_std: np.ndarray,
) -> tuple[dict[int, dict[str, np.ndarray]], list[dict[str, Any]], np.ndarray]:
    """Return raw target-indexed paired deltas, doses, and pairing audit rows."""
    if X.shape != interventions.shape or X.shape != strengths.shape:
        raise ValueError("X, interventions, and intervention strengths must share [N,d] shape")
    if len(row_records) != len(X):
        raise ValueError(f"rows.jsonl has {len(row_records)} rows but X has {len(X)}")
    if clean_std.shape != (X.shape[1],) or np.any(~np.isfinite(clean_std)):
        raise ValueError("clean_std must be one finite value per coordinate")
    if np.any(clean_std <= 1e-8):
        raise ValueError("cannot standardize paired deltas: a clean coordinate has near-zero std")

    groups: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(row_records):
        pair_id = record.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id:
            raise ValueError(f"rows.jsonl row {index} has no nonempty pair_id")
        groups[pair_id].append(index)

    by_target: dict[int, dict[str, list[Any]]] = defaultdict(
        lambda: {"delta": [], "configured_condition": [], "realized_dose": []}
    )
    audit_rows: list[dict[str, Any]] = []
    clean_indices: list[int] = []
    for pair_id, indices in groups.items():
        if len(indices) != 2:
            raise ValueError(f"pair_id {pair_id!r} has {len(indices)} rows; expected exactly 2")
        prompt_ids = {
            row_records[index].get("prompt_id")
            for index in indices
            if row_records[index].get("prompt_id") is not None
        }
        prompts = {
            row_records[index].get("prompt")
            for index in indices
            if row_records[index].get("prompt") is not None
        }
        if len(prompt_ids) > 1 or (prompt_ids and prompt_ids != {pair_id}):
            raise ValueError(f"pair_id {pair_id!r} does not match a single prompt_id")
        if len(prompts) > 1:
            raise ValueError(f"pair_id {pair_id!r} joins rows from different prompt texts")
        clean = [index for index in indices if int(interventions[index].sum()) == 0]
        edited = [index for index in indices if int(interventions[index].sum()) == 1]
        if len(clean) != 1 or len(edited) != 1:
            raise ValueError(
                f"pair_id {pair_id!r} must have one clean and one single-node intervention row"
            )
        clean_index, edited_index = clean[0], edited[0]
        target = int(np.flatnonzero(interventions[edited_index])[0])
        signed_sigma = float(strengths[edited_index, target])
        if not math.isfinite(signed_sigma) or abs(signed_sigma) < 1e-8:
            raise ValueError(f"pair_id {pair_id!r} has a zero/nonfinite signed intervention strength")
        non_target_strengths = np.delete(strengths[edited_index], target)
        if np.any(np.abs(non_target_strengths) > 1e-8):
            raise ValueError(f"pair_id {pair_id!r} has nonzero strengths on multiple nodes")

        delta = (X[edited_index] - X[clean_index]) / clean_std
        realized_dose = float(delta[target])
        if not np.isfinite(delta).all() or not math.isfinite(realized_dose):
            raise ValueError(f"pair_id {pair_id!r} produced a nonfinite paired delta")
        by_target[target]["delta"].append(delta.astype(np.float64, copy=False))
        by_target[target]["configured_condition"].append(signed_sigma)
        by_target[target]["realized_dose"].append(realized_dose)
        clean_indices.append(clean_index)
        edited_record = row_records[edited_index]
        metadata_realized = edited_record.get("realized_target_shift")
        if metadata_realized is not None:
            metadata_realized = float(metadata_realized) / float(clean_std[target])
            if not math.isclose(metadata_realized, realized_dose, rel_tol=2e-3, abs_tol=2e-3):
                raise ValueError(
                    f"pair_id {pair_id!r} rows.realized_target_shift disagrees with paired X"
                )
        configured_setpoint = edited_record.get(
            "configured_setpoint", edited_record.get("hard_do_target_value")
        )
        configured_shift = edited_record.get(
            "configured_target_shift", edited_record.get("requested_coordinate_delta")
        )
        if configured_setpoint is not None and configured_shift is not None:
            expected_shift = float(configured_setpoint) - float(X[clean_index, target])
            if not math.isclose(
                float(configured_shift), expected_shift, rel_tol=2e-3, abs_tol=1e-5
            ):
                raise ValueError(
                    f"pair_id {pair_id!r} configured setpoint/target shift metadata disagree"
                )
        audit_rows.append(
            {
                "pair_id": pair_id,
                "clean_row": clean_index,
                "intervention_row": edited_index,
                "target": target,
                "configured_condition_sigma": signed_sigma,
                "configured_setpoint": (
                    None if configured_setpoint is None else float(configured_setpoint)
                ),
                "realized_target_shift_clean_sd": realized_dose,
            }
        )

    arrays = {
        target: {
            "delta": np.stack(payload["delta"]),
            "configured_condition": np.asarray(
                payload["configured_condition"], dtype=np.float64
            ),
            "realized_dose": np.asarray(payload["realized_dose"], dtype=np.float64),
        }
        for target, payload in by_target.items()
    }
    return arrays, audit_rows, np.asarray(clean_indices, dtype=np.int64)


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values, preserving input shape."""
    values = np.asarray(pvalues, dtype=np.float64)
    if values.ndim != 1 or np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("p-values must be a finite one-dimensional array in [0,1]")
    if len(values) == 0:
        return values.copy()
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.minimum(adjusted, 1.0)
    return output


def estimate_effects(
    paired: dict[int, dict[str, np.ndarray]],
    d: int,
    *,
    intervention_mode: str,
    permutations: int,
    seed: int,
    min_pairs: int,
    min_realized_dose: float,
    fdr: float,
    min_abs_effect: float,
) -> dict[str, np.ndarray]:
    """Estimate paired effects and target-wise sign-flip permutation q-values.

    Legacy additive injections estimate mean response per configured signed
    sigma.  Absolute hard sets instead use a nonparametric RMS of the paired
    ATEs across configured setpoint groups as the detection magnitude; the
    signed estimate is a through-origin slope against the *realized* target
    shift, never against the configured setpoint.
    """
    if permutations < 99:
        raise ValueError("at least 99 sign-flip permutations are required")
    rng = np.random.default_rng(seed)
    mean = np.full((d, d), np.nan, dtype=np.float64)
    magnitude = np.full((d, d), np.nan, dtype=np.float64)
    mean_abs = np.full((d, d), np.nan, dtype=np.float64)
    sign_consistency = np.full((d, d), np.nan, dtype=np.float64)
    sign_directionality = np.full((d, d), np.nan, dtype=np.float64)
    pvalue = np.full((d, d), np.nan, dtype=np.float64)
    qvalue = np.full((d, d), np.nan, dtype=np.float64)
    n_pairs = np.zeros(d, dtype=np.int64)
    n_pairs_total = np.zeros(d, dtype=np.int64)
    weak_dose_pairs_excluded = np.zeros(d, dtype=np.int64)

    for target, payload in paired.items():
        delta = payload["delta"]
        configured = payload["configured_condition"]
        realized_dose = payload["realized_dose"]
        if delta.ndim != 2 or delta.shape[1] != d:
            raise ValueError(f"paired deltas for target {target} have invalid shape {delta.shape}")
        if configured.shape != (len(delta),) or realized_dose.shape != (len(delta),):
            raise ValueError(f"paired conditions/doses for target {target} have invalid shape")
        n_pairs_total[target] = len(delta)
        if intervention_mode == "hard_set_coordinate":
            keep = np.abs(realized_dose) >= min_realized_dose
            weak_dose_pairs_excluded[target] = int((~keep).sum())
            delta = delta[keep]
            configured = configured[keep]
            realized_dose = realized_dose[keep]
        n = len(delta)
        n_pairs[target] = n
        if n == 0:
            continue

        if intervention_mode == "legacy_additive":
            aligned = delta / configured[:, None]
            mean[target] = aligned.mean(axis=0)
            magnitude[target] = np.abs(mean[target])
            mean_abs[target] = np.abs(aligned).mean(axis=0)
        elif intervention_mode == "hard_set_coordinate":
            dose_weighted_delta = realized_dose[:, None] * delta
            denominator = float(np.square(realized_dose).sum())
            if denominator <= 1e-12:
                continue
            mean[target] = dose_weighted_delta.sum(axis=0) / denominator
            mean_abs[target] = np.abs(delta).mean(axis=0)
            group_means = np.stack(
                [delta[configured == condition].mean(axis=0) for condition in np.unique(configured)]
            )
            magnitude[target] = np.sqrt(np.square(group_means).mean(axis=0))
            aligned = np.sign(realized_dose)[:, None] * delta
        else:
            raise ValueError(f"unknown intervention mode {intervention_mode!r}")

        positive = (aligned > 1e-10).sum(axis=0)
        negative = (aligned < -1e-10).sum(axis=0)
        nonzero = positive + negative
        sign_consistency[target] = np.divide(
            np.maximum(positive, negative),
            nonzero,
            out=np.full(d, np.nan, dtype=np.float64),
            where=nonzero > 0,
        )
        sign_directionality[target] = np.divide(
            np.abs(positive - negative),
            nonzero,
            out=np.full(d, np.nan, dtype=np.float64),
            where=nonzero > 0,
        )
        if n < min_pairs:
            continue
        signs = rng.choice(np.array([-1.0, 1.0]), size=(permutations, n))
        if intervention_mode == "legacy_additive":
            null_statistic = np.abs(signs @ aligned / n)
        else:
            null_squared = np.zeros((permutations, d), dtype=np.float64)
            conditions = np.unique(configured)
            for condition in conditions:
                group = configured == condition
                null_group_mean = signs[:, group] @ delta[group] / int(group.sum())
                null_squared += np.square(null_group_mean)
            null_statistic = np.sqrt(null_squared / len(conditions))
        observed = magnitude[target]
        pvalue[target] = (1.0 + (null_statistic >= observed).sum(axis=0)) / (
            permutations + 1.0
        )
        tested = np.array([node for node in range(d) if node != target], dtype=np.int64)
        qvalue[target, tested] = benjamini_hochberg(pvalue[target, tested])

    valid = np.isfinite(qvalue)
    detectable = valid & (qvalue <= fdr) & (magnitude >= min_abs_effect)
    np.fill_diagonal(valid, False)
    np.fill_diagonal(detectable, False)
    return {
        "mean": mean,
        "magnitude": magnitude,
        "mean_abs": mean_abs,
        "sign_consistency": sign_consistency,
        "sign_directionality": sign_directionality,
        "pvalue": pvalue,
        "qvalue": qvalue,
        "valid": valid,
        "detectable": detectable,
        "n_pairs": n_pairs,
        "n_pairs_total": n_pairs_total,
        "weak_dose_pairs_excluded": weak_dose_pairs_excluded,
    }


def transitive_closure(adjacency: np.ndarray) -> np.ndarray:
    closure = np.asarray(adjacency, dtype=bool).copy()
    for pivot in range(len(closure)):
        closure |= closure[:, pivot, None] & closure[pivot, None, :]
    np.fill_diagonal(closure, False)
    return closure


def is_dag(adjacency: np.ndarray) -> bool:
    adjacency = np.asarray(adjacency, dtype=bool)
    indegree = adjacency.sum(axis=0).astype(np.int64)
    stack = [int(node) for node in np.flatnonzero(indegree == 0)]
    visited = 0
    while stack:
        node = stack.pop()
        visited += 1
        for child in np.flatnonzero(adjacency[node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                stack.append(int(child))
    return visited == len(adjacency)


def max_product_path_scores(adjacency: np.ndarray, direct_scores: np.ndarray) -> np.ndarray:
    """Maximum product of retained-edge scores over all directed paths."""
    paths = np.where(adjacency, direct_scores, 0.0).astype(np.float64)
    for pivot in range(len(paths)):
        paths = np.maximum(paths, paths[:, pivot, None] * paths[pivot, None, :])
    np.fill_diagonal(paths, 0.0)
    return paths


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positive_rank_sum = float(ranks[labels].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """Threshold-grouped average precision, invariant to ordering within ties."""
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(labels.sum())
    if positives == 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    true_positive = 0
    false_positive = 0
    previous_recall = 0.0
    area = 0.0
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        true_positive += int(sorted_labels[start:end].sum())
        false_positive += int(end - start - sorted_labels[start:end].sum())
        recall = true_positive / positives
        precision = true_positive / max(true_positive + false_positive, 1)
        area += (recall - previous_recall) * precision
        previous_recall = recall
        start = end
    return area


def score_metrics(labels: np.ndarray, scores: np.ndarray, valid: np.ndarray, top_k: int) -> dict[str, Any]:
    mask = np.asarray(valid, dtype=bool)
    y = np.asarray(labels, dtype=bool)[mask]
    s = np.asarray(scores, dtype=np.float64)[mask]
    if len(y) == 0:
        return {"roc_auc": None, "average_precision": None, "reason": "no_valid_effect_tests"}
    k = min(max(int(top_k), 0), len(y))
    if k:
        selected = np.argsort(-s, kind="mergesort")[:k]
        precision_at_k = float(y[selected].mean())
    else:
        precision_at_k = None
    return {
        "roc_auc": roc_auc(y, s),
        "average_precision": average_precision(y, s),
        "positive_prevalence": float(y.mean()),
        "n_positive": int(y.sum()),
        "n_tested": int(len(y)),
        "top_k": k,
        "precision_at_k": precision_at_k,
    }


def finite_mean(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else None


def nullable_float(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def graph_group_summary(mask: np.ndarray, effects: dict[str, np.ndarray]) -> dict[str, Any]:
    selected = np.asarray(mask, dtype=bool) & effects["valid"]
    count = int(selected.sum())
    if count == 0:
        return {
            "n_pairs": 0,
            "detectable_effect_fraction": None,
            "mean_empirical_effect_magnitude": None,
            "mean_sign_consistency": None,
            "mean_sign_directionality": None,
        }
    return {
        "n_pairs": count,
        "detectable_effect_fraction": float(effects["detectable"][selected].mean()),
        "mean_empirical_effect_magnitude": finite_mean(effects["magnitude"][selected]),
        "mean_sign_consistency": finite_mean(effects["sign_consistency"][selected]),
        "mean_sign_directionality": finite_mean(effects["sign_directionality"][selected]),
    }


def graph_report(
    adjacency: np.ndarray,
    direct_scores: np.ndarray,
    effects: dict[str, np.ndarray],
) -> dict[str, Any]:
    closure = transitive_closure(adjacency)
    indirect = closure & ~np.asarray(adjacency, dtype=bool)
    unrelated = ~closure & ~np.eye(len(adjacency), dtype=bool)
    path_scores = max_product_path_scores(adjacency, direct_scores)
    n_edges = int(adjacency.sum())
    n_descendants = int(closure.sum())
    return {
        "n_direct_edges": n_edges,
        "n_reachable_ordered_pairs": n_descendants,
        "groups": {
            "direct_edges": graph_group_summary(adjacency, effects),
            "indirect_descendants": graph_group_summary(indirect, effects),
            "all_descendants": graph_group_summary(closure, effects),
            "unrelated_ordered_pairs": graph_group_summary(unrelated, effects),
        },
        "effect_label_prediction": {
            "direct_probability": score_metrics(
                effects["detectable"], direct_scores, effects["valid"], n_edges
            ),
            "max_product_reachability": score_metrics(
                effects["detectable"], path_scores, effects["valid"], n_descendants
            ),
        },
    }


def effect_records(
    mask: np.ndarray,
    effects: dict[str, np.ndarray],
    node_ids: list[str],
    direct_scores: np.ndarray,
    paired: dict[int, dict[str, np.ndarray]],
    intervention_mode: str,
    min_realized_dose: float,
) -> list[dict[str, Any]]:
    records = []
    for source, target in np.argwhere(mask):
        source_index, target_index = int(source), int(target)
        payload = paired.get(source_index)
        condition_effects = []
        if payload is not None:
            delta = payload["delta"]
            configured = payload["configured_condition"]
            realized = payload["realized_dose"]
            if intervention_mode == "hard_set_coordinate":
                keep = np.abs(realized) >= min_realized_dose
                delta, configured, realized = delta[keep], configured[keep], realized[keep]
            for condition in np.unique(configured):
                group = configured == condition
                group_delta = delta[group, target_index]
                group_realized = realized[group]
                condition_effects.append(
                    {
                        "configured_condition_sigma": float(condition),
                        "n_prompt_pairs": int(group.sum()),
                        "mean_realized_target_shift_clean_sd": float(group_realized.mean()),
                        "mean_paired_outcome_delta_clean_sd": float(group_delta.mean()),
                        "mean_absolute_paired_outcome_delta_clean_sd": float(
                            np.abs(group_delta).mean()
                        ),
                    }
                )
        records.append(
            {
                "source": node_ids[int(source)],
                "target": node_ids[int(target)],
                "graph_direct_probability": float(direct_scores[source, target]),
                "n_prompt_pairs": int(effects["n_pairs"][source]),
                "primary_signed_effect_estimate": nullable_float(
                    effects["mean"][source, target]
                ),
                "empirical_effect_magnitude": nullable_float(
                    effects["magnitude"][source, target]
                ),
                "mean_absolute_pairwise_response": nullable_float(
                    effects["mean_abs"][source, target]
                ),
                "sign_consistency": nullable_float(effects["sign_consistency"][source, target]),
                "sign_directionality": nullable_float(
                    effects["sign_directionality"][source, target]
                ),
                "permutation_p": nullable_float(effects["pvalue"][source, target]),
                "targetwise_bh_q": nullable_float(effects["qvalue"][source, target]),
                "detectable_effect": bool(effects["detectable"][source, target]),
                "configured_condition_effects": condition_effects,
            }
        )
    return records


def correlation_control(
    clean_X: np.ndarray,
    nodes: list[dict[str, Any]],
    edge_count: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Top clean-only correlations, directed by preregistered layer/column order."""
    d = clean_X.shape[1]
    with np.errstate(invalid="ignore", divide="ignore"):
        correlation = np.corrcoef(clean_X, rowvar=False)
    correlation = np.nan_to_num(np.abs(correlation), nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(correlation, 0.0)

    def order_key(index: int) -> tuple[int, int]:
        column = int(nodes[index].get("column", index))
        return int(nodes[index].get("layer", column)), column

    order = sorted(range(d), key=order_key)
    rank = {node: position for position, node in enumerate(order)}
    candidates = []
    score_matrix = np.zeros((d, d), dtype=np.float64)
    for left in range(d):
        for right in range(left + 1, d):
            source, target = (left, right) if rank[left] < rank[right] else (right, left)
            score = float(correlation[left, right])
            score_matrix[source, target] = score
            candidates.append((score, source, target))
    adjacency = np.zeros((d, d), dtype=bool)
    for _, source, target in sorted(candidates, reverse=True)[:edge_count]:
        adjacency[source, target] = True
    return adjacency, score_matrix, {
        "construction": "top_abs_Pearson_on_one_clean_row_per_prompt_oriented_by_(layer,column)",
        "uses_intervention_rows": False,
        "edge_count_matched": edge_count,
        "is_dag": is_dag(adjacency),
    }


def distribution_summary(values: list[float | None], observed: float | None) -> dict[str, Any]:
    finite = np.asarray([value for value in values if value is not None and math.isfinite(value)])
    if not len(finite):
        return {"n": 0, "mean": None, "std": None, "q05": None, "q50": None, "q95": None}
    summary: dict[str, Any] = {
        "n": int(len(finite)),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "q05": float(np.quantile(finite, 0.05)),
        "q50": float(np.quantile(finite, 0.50)),
        "q95": float(np.quantile(finite, 0.95)),
    }
    summary["causcale_empirical_p_ge"] = (
        None
        if observed is None
        else float((1 + np.sum(finite >= observed)) / (1 + len(finite)))
    )
    return summary


def shuffled_null(
    adjacency: np.ndarray,
    direct_scores: np.ndarray,
    effects: dict[str, np.ndarray],
    *,
    repetitions: int,
    seed: int,
    observed: dict[str, Any],
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    collected: dict[str, list[float | None]] = defaultdict(list)
    d = len(adjacency)
    for _ in range(repetitions):
        permutation = rng.permutation(d)
        shuffled_adjacency = adjacency[np.ix_(permutation, permutation)]
        shuffled_scores = direct_scores[np.ix_(permutation, permutation)]
        report = graph_report(shuffled_adjacency, shuffled_scores, effects)
        collected["direct_roc_auc"].append(
            report["effect_label_prediction"]["direct_probability"]["roc_auc"]
        )
        collected["direct_average_precision"].append(
            report["effect_label_prediction"]["direct_probability"]["average_precision"]
        )
        collected["reachability_roc_auc"].append(
            report["effect_label_prediction"]["max_product_reachability"]["roc_auc"]
        )
        collected["reachability_average_precision"].append(
            report["effect_label_prediction"]["max_product_reachability"]["average_precision"]
        )
        collected["direct_mean_effect_magnitude"].append(
            report["groups"]["direct_edges"]["mean_empirical_effect_magnitude"]
        )
        collected["descendant_mean_effect_magnitude"].append(
            report["groups"]["all_descendants"]["mean_empirical_effect_magnitude"]
        )

    observed_values = {
        "direct_roc_auc": observed["effect_label_prediction"]["direct_probability"]["roc_auc"],
        "direct_average_precision": observed["effect_label_prediction"]["direct_probability"][
            "average_precision"
        ],
        "reachability_roc_auc": observed["effect_label_prediction"]["max_product_reachability"][
            "roc_auc"
        ],
        "reachability_average_precision": observed["effect_label_prediction"]
        ["max_product_reachability"]["average_precision"],
        "direct_mean_effect_magnitude": observed["groups"]["direct_edges"]
        ["mean_empirical_effect_magnitude"],
        "descendant_mean_effect_magnitude": observed["groups"]["all_descendants"]
        ["mean_empirical_effect_magnitude"],
    }
    return {
        "construction": "uniform_node_label_permutation_preserving_exact_in/out_degree_multiset",
        "repetitions": repetitions,
        "seed": seed,
        "metrics": {
            name: distribution_summary(values, observed_values[name])
            for name, values in collected.items()
        },
    }


def clean_scale(manifest: dict[str, Any], clean_X: np.ndarray, d: int) -> tuple[np.ndarray, str]:
    stored = manifest.get("standardization", {}).get("std")
    if isinstance(stored, list) and len(stored) == d:
        scale = np.asarray(stored, dtype=np.float64)
        source = "manifest.standardization.std_fitted_on_clean_dev_rows"
    else:
        scale = clean_X.std(axis=0).astype(np.float64)
        source = "recomputed_from_one_clean_row_per_paired_prompt"
    if np.any(~np.isfinite(scale)) or np.any(scale <= 1e-8):
        raise ValueError("clean coordinate scale contains nonfinite or near-zero values")
    return scale, source


def paired_design_summary(
    paired: dict[int, dict[str, np.ndarray]], node_ids: list[str], min_realized_dose: float
) -> dict[str, Any]:
    summary = {}
    for target, payload in paired.items():
        configured = payload["configured_condition"]
        realized = payload["realized_dose"]
        conditions = []
        for condition in np.unique(configured):
            selected = configured == condition
            values = realized[selected]
            conditions.append(
                {
                    "configured_condition_sigma": float(condition),
                    "n_prompt_pairs": int(selected.sum()),
                    "mean_realized_target_shift_clean_sd": float(values.mean()),
                    "std_realized_target_shift_clean_sd": float(values.std()),
                    "median_absolute_realized_target_shift_clean_sd": float(
                        np.median(np.abs(values))
                    ),
                }
            )
        summary[node_ids[target]] = {
            "n_prompt_pairs": int(len(realized)),
            "n_below_minimum_absolute_realized_dose": int(
                (np.abs(realized) < min_realized_dose).sum()
            ),
            "conditions": conditions,
        }
    return summary


def load_causcale(path: Path, d: int, node_ids: list[str], threshold: float):
    payload = np.load(path, allow_pickle=False)
    if "node_names" in payload:
        names = payload["node_names"].astype(str).tolist()
        if names != node_ids:
            raise ValueError("CauScale node_names do not exactly match dataset node ordering")
    if "directed_probs" in payload:
        directed = payload["directed_probs"].astype(np.float64)
    elif "pair_index" in payload and "pair_probs" in payload:
        directed = np.zeros((d, d), dtype=np.float64)
        for (left, right), probabilities in zip(payload["pair_index"], payload["pair_probs"]):
            directed[int(left), int(right)] = float(probabilities[1])
            directed[int(right), int(left)] = float(probabilities[2])
    else:
        raise ValueError("CauScale NPZ needs directed_probs or pair_index plus pair_probs")
    if directed.shape != (d, d) or np.any(~np.isfinite(directed)):
        raise ValueError("directed CauScale probabilities have invalid shape or values")
    if np.any((directed < 0) | (directed > 1)):
        raise ValueError("directed CauScale probabilities must lie in [0,1]")

    if "adjacency" in payload:
        adjacency = payload["adjacency"].astype(bool)
    elif "pair_index" in payload and "pair_probs" in payload:
        adjacency, _, _, _ = build_dag(
            payload["pair_index"], payload["pair_probs"], d=d, threshold=threshold
        )
        adjacency = adjacency.astype(bool)
    else:
        raise ValueError("CauScale NPZ needs adjacency or pair probabilities for DAG construction")
    if adjacency.shape != (d, d) or np.any(np.diag(adjacency)):
        raise ValueError("CauScale adjacency must be dxd with a zero diagonal")
    if not is_dag(adjacency):
        raise ValueError("CauScale adjacency is not acyclic")
    return adjacency, directed


def validate_causcale_provenance(
    path: Path, current_manifest: dict[str, Any], *, allow_cross_dataset: bool
) -> dict[str, Any]:
    """Reject a stale CauScale sidecar after the dataset has been recollected."""
    sidecar = path.with_suffix(".json")
    if not sidecar.is_file():
        return {
            "sidecar": None,
            "dataset_hash_match_verified": False,
            "prediction_dataset_id": None,
            "validation_dataset_id": current_manifest.get("dataset_id"),
            "relationship": "external_graph_unverified" if allow_cross_dataset else "unknown",
            "warning": "CauScale sidecar is absent; dataset provenance could not be cross-checked",
        }
    metadata = load_json(sidecar)
    prediction_manifest = metadata.get("dataset", {}).get("manifest", {})
    prediction_dataset_id = prediction_manifest.get("dataset_id")
    validation_dataset_id = current_manifest.get("dataset_id")
    predicted_files = prediction_manifest.get("files", {})
    current_files = current_manifest.get("files", {})
    checked = []
    mismatches = []
    for name in ("X.npy", "interventions.npy", "intervention_strengths.npy", "rows.jsonl"):
        predicted = predicted_files.get(name)
        current = current_files.get(name)
        if predicted is not None and current is not None:
            checked.append(name)
            if predicted != current:
                mismatches.append(name)
    if mismatches:
        if not allow_cross_dataset:
            raise ValueError(
                "CauScale prediction is stale or comes from a different dataset; mismatched "
                f"files: {mismatches}. Rerun predict_causcale.py, or use "
                "--allow-external-graph only for an intentionally fixed discovery graph "
                "evaluated on disjoint heldout interventions."
            )
        return {
            "sidecar": str(sidecar.resolve()),
            "sidecar_sha256": sha256_file(sidecar),
            "dataset_hash_match_verified": False,
            "intentional_cross_dataset_graph": True,
            "prediction_dataset_id": prediction_dataset_id,
            "validation_dataset_id": validation_dataset_id,
            "relationship": "disjoint_external_graph",
            "mismatched_files": mismatches,
            "warning": "fixed graph was learned on a different dataset by explicit request",
        }
    return {
        "sidecar": str(sidecar.resolve()),
        "sidecar_sha256": sha256_file(sidecar),
        "dataset_hash_match_verified": bool(checked),
        "intentional_cross_dataset_graph": False,
        "prediction_dataset_id": prediction_dataset_id,
        "validation_dataset_id": validation_dataset_id,
        "relationship": "same_dataset",
        "checked_files": checked,
        "warning": None if checked else "sidecar had no comparable dataset file hashes",
    }


def self_test() -> None:
    d = 4
    clean_std = np.ones(d)
    rows = []
    X_rows = []
    intervention_rows = []
    strength_rows = []
    for pair in range(12):
        sign = -1.0 if pair % 2 else 1.0
        clean = np.array([0.1 * pair, -0.05 * pair, 0.02 * pair, 0.0])
        edited = clean.copy()
        edited[0] += sign
        edited[1] += sign * 0.8
        edited[2] += sign * 0.5
        X_rows.extend([clean, edited])
        intervention_rows.extend([np.zeros(d), np.array([1, 0, 0, 0])])
        strength_rows.extend([np.zeros(d), np.array([sign, 0, 0, 0])])
        rows.extend(
            [
                {"pair_id": f"p{pair}", "regime": "clean"},
                {
                    "pair_id": f"p{pair}",
                    "regime": "single_coordinate_injection",
                    "realized_target_shift": sign,
                },
            ]
        )
    paired, audit, clean_indices = paired_responses(
        np.asarray(X_rows, dtype=np.float32),
        np.asarray(intervention_rows, dtype=np.uint8),
        np.asarray(strength_rows, dtype=np.float32),
        rows,
        clean_std,
    )
    assert len(audit) == 12 and len(np.unique(clean_indices)) == 12
    legacy_response = paired[0]["delta"] / paired[0]["configured_condition"][:, None]
    assert np.allclose(legacy_response[:, 1], 0.8) and np.allclose(legacy_response[:, 2], 0.5)
    adjacency = np.zeros((d, d), dtype=bool)
    adjacency[0, 1] = True
    adjacency[1, 2] = True
    closure = transitive_closure(adjacency)
    assert closure[0, 2] and not closure[2, 0] and is_dag(adjacency)
    assert roc_auc(np.array([1, 1, 0, 0]), np.array([0.9, 0.8, 0.2, 0.1])) == 1.0
    assert average_precision(np.array([1, 1, 0, 0]), np.array([0.9, 0.8, 0.2, 0.1])) == 1.0
    effects = estimate_effects(
        paired,
        d,
        intervention_mode="legacy_additive",
        permutations=1023,
        seed=7,
        min_pairs=5,
        min_realized_dose=0.05,
        fdr=0.05,
        min_abs_effect=0.1,
    )
    assert effects["detectable"][0, 1] and effects["detectable"][0, 2]
    hard_set_effects = estimate_effects(
        paired,
        d,
        intervention_mode="hard_set_coordinate",
        permutations=1023,
        seed=7,
        min_pairs=5,
        min_realized_dose=0.05,
        fdr=0.05,
        min_abs_effect=0.1,
    )
    assert np.allclose(hard_set_effects["mean"][0, 1:3], [0.8, 0.5])
    assert hard_set_effects["detectable"][0, 1] and hard_set_effects["detectable"][0, 2]
    protocol = resolve_intervention_mode(
        {"intervention_design": {"mode": "hard_set_coordinate"}}, rows, "auto"
    )
    assert protocol["mode"] == "hard_set_coordinate"
    scores = np.zeros((d, d), dtype=np.float64)
    scores[0, 1] = scores[1, 2] = 0.9
    report = graph_report(adjacency, scores, effects)
    assert report["groups"]["direct_edges"]["n_pairs"] == 1
    corr_adjacency, _, corr_meta = correlation_control(
        np.asarray(X_rows)[clean_indices],
        [{"column": index, "layer": index} for index in range(d)],
        edge_count=2,
    )
    assert corr_adjacency.sum() == 2 and corr_meta["is_dag"]
    with tempfile.TemporaryDirectory(prefix="jspace_intervention_validation_") as directory:
        graph_path = Path(directory) / "graph.npz"
        save_json(
            graph_path.with_suffix(".json"),
            {
                "dataset": {
                    "manifest": {
                        "dataset_id": "discovery_fixture",
                        "files": {"X.npy": "discovery_hash"},
                    }
                }
            },
        )
        heldout_manifest = {
            "dataset_id": "heldout_fixture",
            "files": {"X.npy": "heldout_hash"},
        }
        try:
            validate_causcale_provenance(
                graph_path, heldout_manifest, allow_cross_dataset=False
            )
        except ValueError:
            pass
        else:
            raise AssertionError("cross-dataset graph must require an explicit opt-in")
        external = validate_causcale_provenance(
            graph_path, heldout_manifest, allow_cross_dataset=True
        )
        assert external["relationship"] == "disjoint_external_graph"
        assert external["prediction_dataset_id"] == "discovery_fixture"
        assert external["validation_dataset_id"] == "heldout_fixture"
    print(
        json.dumps(
            {
                "self_test": "passed",
                "paired_prompt_pairs": len(audit),
                "detectable_effects": int(effects["detectable"].sum()),
                "correlation_control_edges": int(corr_adjacency.sum()),
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--causcale", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--effect-fdr", type=float, default=0.05)
    parser.add_argument("--min-abs-effect", type=float, default=0.10)
    parser.add_argument("--min-pairs-per-target", type=int, default=5)
    parser.add_argument("--effect-permutations", type=int, default=4096)
    parser.add_argument("--graph-shuffles", type=int, default=200)
    parser.add_argument(
        "--allow-external-graph",
        "--allow-cross-dataset-graph",
        dest="allow_cross_dataset_graph",
        action="store_true",
        help="allow a fixed discovery graph to be evaluated on a disjoint heldout dataset",
    )
    parser.add_argument(
        "--intervention-mode",
        choices=["auto", "legacy_additive", "hard_set_coordinate"],
        default="auto",
    )
    parser.add_argument(
        "--min-realized-dose",
        type=float,
        default=0.05,
        help="hard-set pairs below this absolute realized target shift (clean SD) are excluded",
    )
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.dataset is None or args.causcale is None or args.output is None:
        parser.error("--dataset, --causcale, and --output are required unless --self-test is used")
    if not 0 < args.effect_fdr < 1 or args.min_abs_effect < 0:
        raise ValueError("effect FDR must be in (0,1) and minimum effect must be nonnegative")
    if args.graph_shuffles < 1:
        raise ValueError("at least one degree-matched graph shuffle is required")
    if args.min_realized_dose < 0:
        raise ValueError("minimum realized dose must be nonnegative")

    dataset_report = validate_dataset(args.dataset)
    rows_path = args.dataset / "rows.jsonl"
    strengths_path = args.dataset / "intervention_strengths.npy"
    if not rows_path.is_file() or not strengths_path.is_file():
        raise FileNotFoundError(
            "paired validation requires rows.jsonl and intervention_strengths.npy; "
            "the unpaired engineering fixture is intentionally not accepted"
        )
    X = np.load(args.dataset / "X.npy").astype(np.float64)
    interventions = np.load(args.dataset / "interventions.npy")
    strengths = np.load(strengths_path).astype(np.float64)
    row_records = read_jsonl(rows_path)
    nodes = load_json(args.dataset / "nodes.json")
    manifest = load_json(args.dataset / "manifest.json")
    intervention_protocol = resolve_intervention_mode(
        manifest, row_records, args.intervention_mode
    )
    intervention_mode = intervention_protocol["mode"]
    node_ids = [str(node["node_id"]) for node in nodes]
    d = len(node_ids)

    # First identify clean rows by pairing metadata; scale estimation and the
    # correlation control then use exactly one clean row per prompt pair.
    provisional_groups: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(row_records):
        provisional_groups[str(record.get("pair_id"))].append(index)
    provisional_clean = []
    for pair_id, indices in provisional_groups.items():
        candidates = [index for index in indices if interventions[index].sum() == 0]
        if len(indices) != 2 or len(candidates) != 1:
            raise ValueError(f"invalid clean/intervention pairing for pair_id {pair_id!r}")
        provisional_clean.append(candidates[0])
    clean_X = X[np.asarray(provisional_clean, dtype=np.int64)]
    scale, scale_source = clean_scale(manifest, clean_X, d)
    paired, pairing_audit, clean_indices = paired_responses(
        X, interventions, strengths, row_records, scale
    )
    clean_X = X[clean_indices]
    effects = estimate_effects(
        paired,
        d,
        intervention_mode=intervention_mode,
        permutations=args.effect_permutations,
        seed=args.seed,
        min_pairs=args.min_pairs_per_target,
        min_realized_dose=args.min_realized_dose,
        fdr=args.effect_fdr,
        min_abs_effect=args.min_abs_effect,
    )
    causcale_provenance = validate_causcale_provenance(
        args.causcale,
        manifest,
        allow_cross_dataset=args.allow_cross_dataset_graph,
    )
    adjacency, directed_scores = load_causcale(
        args.causcale, d, node_ids, threshold=args.threshold
    )
    causcale_report = graph_report(adjacency, directed_scores, effects)
    closure = transitive_closure(adjacency)

    corr_adjacency, corr_scores, corr_metadata = correlation_control(
        clean_X, nodes, edge_count=int(adjacency.sum())
    )
    corr_report = graph_report(corr_adjacency, corr_scores, effects)
    corr_report["metadata"] = corr_metadata
    shuffle_report = shuffled_null(
        adjacency,
        directed_scores,
        effects,
        repetitions=args.graph_shuffles,
        seed=args.seed + 1,
        observed=causcale_report,
    )

    direct_records = effect_records(
        adjacency,
        effects,
        node_ids,
        directed_scores,
        paired,
        intervention_mode,
        args.min_realized_dose,
    )
    indirect_records = effect_records(
        closure & ~adjacency,
        effects,
        node_ids,
        directed_scores,
        paired,
        intervention_mode,
        args.min_realized_dose,
    )
    if intervention_mode == "hard_set_coordinate":
        effect_formula = {
            "paired_delta": "(X_intervention-X_clean)/clean_coordinate_std",
            "configured_strength_semantics": (
                "absolute setpoint sigma relative to calibration mean/std; not a dose"
            ),
            "primary_signed_estimate": (
                "through-origin slope of standardized paired outcome delta on the "
                "realized standardized target shift"
            ),
            "detection_magnitude": (
                "root-mean-square of paired outcome ATEs across configured setpoint groups"
            ),
            "null_test": "pair-level sign flips of raw paired deltas, recomputing setpoint-group ATE RMS",
            "sign_alignment": "sign(realized_target_shift)*standardized_paired_outcome_delta",
        }
    else:
        effect_formula = {
            "paired_delta": "(X_intervention-X_clean)/clean_coordinate_std",
            "configured_strength_semantics": "legacy additive signed-sigma shift",
            "primary_signed_estimate": "mean(paired_delta/configured_signed_sigma)",
            "detection_magnitude": "absolute_value_of_primary_signed_estimate",
            "null_test": "pair-level sign flips of per-configured-sigma responses",
            "sign_alignment": "paired_delta/configured_signed_sigma",
        }
    report = {
        "schema_version": 1,
        "evaluation": "matched_prompt_paired_intervention_validation",
        "evidence_status": "interventional_validation_of_fixed_causcale_prediction",
        "inputs": {
            "dataset": str(args.dataset.resolve()),
            "dataset_id": manifest.get("dataset_id"),
            "dataset_manifest_sha256": sha256_file(args.dataset / "manifest.json"),
            "causcale": str(args.causcale.resolve()),
            "causcale_sha256": sha256_file(args.causcale),
            "causcale_dataset_provenance": causcale_provenance,
            "dag_threshold_if_adjacency_missing": args.threshold,
            "n_nodes": d,
            "validated_dataset_contract": dataset_report,
        },
        "pairing": {
            "unit_of_analysis": "within_prompt_intervention_minus_clean_pair",
            "clean_and_intervention_rows_treated_as_independent": False,
            "n_prompt_pairs": len(pairing_audit),
            "pairs_per_intervention_target": {
                node_ids[index]: int(count)
                for index, count in enumerate(effects["n_pairs"])
            },
            "total_pairs_per_intervention_target_before_dose_filter": {
                node_ids[index]: int(count)
                for index, count in enumerate(effects["n_pairs_total"])
            },
            "weak_realized_dose_pairs_excluded": {
                node_ids[index]: int(count)
                for index, count in enumerate(effects["weak_dose_pairs_excluded"])
            },
            "pairing_checks": "exactly_one_clean_and_one_single_node_intervention_per_pair_id",
        },
        "intervention_protocol": {
            **intervention_protocol,
            "intervention_strengths_semantics_from_manifest": (
                manifest.get("intervention_design", {}).get(
                    "intervention_strengths_semantics"
                )
                if isinstance(manifest.get("intervention_design"), dict)
                else None
            ),
            "minimum_absolute_realized_target_shift_clean_sd": (
                args.min_realized_dose if intervention_mode == "hard_set_coordinate" else None
            ),
            "realized_design_by_target": paired_design_summary(
                paired, node_ids, args.min_realized_dose
            ),
        },
        "effect_definition": {
            **effect_formula,
            "clean_scale_source": scale_source,
            "detectable_effect": (
                f"target-wise BH q<={args.effect_fdr} from {args.effect_permutations} "
                f"paired sign-flip permutations and empirical magnitude>={args.min_abs_effect}"
            ),
            "minimum_pairs_per_target": args.min_pairs_per_target,
            "multiple_testing_scope": "within_each_intervention_target_across_off_diagonal_outcomes",
            "sign_consistency": (
                "max(fraction_positive,fraction_negative) after the mode-specific alignment"
            ),
            "n_valid_ordered_pair_tests": int(effects["valid"].sum()),
            "n_detectable_ordered_pair_effects": int(effects["detectable"].sum()),
        },
        "causcale": causcale_report,
        "causcale_effect_details": {
            "direct_edges": direct_records,
            "indirect_descendants": indirect_records,
        },
        "controls": {
            "clean_correlation_graph": corr_report,
            "degree_matched_node_label_shuffle": shuffle_report,
        },
        "interpretation_guardrails": [
            "Paired deltas, not pooled clean/intervention rows, define every empirical effect.",
            "Interventions identify total downstream response; they do not by themselves prove which retained edge is direct.",
            "CauScale edge probabilities are unsigned, so sign consistency tests stability of the observed response, not a predicted sign.",
            "The correlation control uses only one clean row per prompt and never sees intervention outcomes.",
            "The node-label shuffle preserves the exact directed graph up to relabeling and therefore its in/out-degree multiset.",
            "When discovery data generated the CauScale graph, this is an in-sample diagnostic; use --allow-external-graph to evaluate that fixed graph against the disjoint heldout paired dataset.",
        ],
    }
    save_json(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "paired_prompts": len(pairing_audit),
                "valid_effect_tests": int(effects["valid"].sum()),
                "detectable_effects": int(effects["detectable"].sum()),
                "causcale_direct_edges": int(adjacency.sum()),
                "causcale_reachable_pairs": int(closure.sum()),
                "causcale_reachability_auc": causcale_report["effect_label_prediction"]
                ["max_product_reachability"]["roc_auc"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
