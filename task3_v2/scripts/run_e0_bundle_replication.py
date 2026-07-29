"""API-free selected-dataset replication of the frozen E0-prime Stage-3 bundle.

Follows v5/run_task1.py folds and frozen objective, while persisting semantic
metrics and pending Judge requests. No artifact is trained or mutated.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

SCRIPT_PATH = Path(__file__).resolve()
TASK_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = TASK_ROOT.parent
V5_ROOT = REPO_ROOT / "v5"
EXPERIMENT_DIR = TASK_ROOT / "experiments" / "e0_orientation_constraint_audit"
RESULTS_DIR = EXPERIMENT_DIR / "results"
DEFAULT_CONFIG = EXPERIMENT_DIR / "config.yaml"
HF_CACHE = REPO_ROOT.parent / ".hf_cache"
os.environ["HF_CACHE"] = str(HF_CACHE)
os.environ.setdefault("HF_HOME", str(HF_CACHE))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ["OPENAI_API_KEY"] = ""
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(V5_ROOT) not in sys.path:
    sys.path.insert(0, str(V5_ROOT))

import numpy as np
from task3_v2.scripts import e0_core as core
from task3_v2.scripts import run_e0_audit as audit
from task3_v2.scripts import run_e0_bridge as prime


class BundleReplicationError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BundleReplicationError(message)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _sha256_dataset(dataset: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(dataset["X"], dtype=np.float64).tobytes())
    digest.update(json.dumps(dataset["labels"], ensure_ascii=False, sort_keys=True).encode("utf-8"))
    digest.update(json.dumps(list(dataset["graph"].edges), separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_datasets(runtime: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    import run_task1
    himi = run_task1.ALL_LOADERS["himi"]()
    kims = run_task1.ALL_LOADERS["kims"]()
    base = run_task1.ALL_LOADERS["bigfive"]()
    edges = list(base["graph"].edges)
    for upper, lowers in {
        "stability": ["agreeableness", "conscientiousness", "neuroticism"],
        "plasticity": ["extraversion", "openness"],
    }.items():
        edges.extend((upper, lower) for lower in lowers)
    edges.extend((("GFP", "stability"), ("GFP", "plasticity")))
    hierarchy = runtime["graph"].Graph(
        list(base["graph"].latents) + ["stability", "plasticity", "GFP"],
        list(base["graph"].observed),
        edges,
    )
    bigfive2 = {
        "name": "bigfive2", "canonical_name": "bigfive", "graph": hierarchy,
        "X": np.asarray(base["X"], dtype=np.float64), "labels": dict(base["labels"]),
    }
    return {"himi": himi, "kims": kims, "bigfive2": bigfive2}


def dependence_matrix(dataset: Mapping[str, Any]) -> tuple[np.ndarray, Path, str, str]:
    name = str(dataset["name"])
    if name == "bigfive2":
        path = RESULTS_DIR / "bigfive2_marginal_pearson.npy"
        matrix = np.abs(np.corrcoef(np.asarray(dataset["X"]).T))
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(matrix, 0.0)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        np.save(path, np.asarray(matrix, dtype=np.float32))
        source = "generated_in_audit_results_from_full_bigfive_observed_X"
    else:
        path = V5_ROOT / "outputs" / "dependence" / f"{name}_marginal_pearson.npy"
        _require(path.is_file(), f"formal dependence cache missing: {path}")
        matrix = np.asarray(np.load(path), dtype=np.float32)
        source = "existing_v5_formal_dependence_cache"
    observed = len(dataset["graph"].observed)
    _require(matrix.shape == (observed, observed), f"{name}: dependence shape mismatch")
    _require(np.isfinite(matrix).all(), f"{name}: non-finite dependence")
    _require(np.allclose(matrix, matrix.T, atol=1e-6), f"{name}: asymmetric dependence")
    _require(np.max(np.abs(np.diag(matrix))) <= 1e-7, f"{name}: nonzero diagonal")
    return matrix, path, _sha256_file(path), source


def prepare_objective(runtime: Mapping[str, Any], dataset: Mapping[str, Any], *, latcon: bool, base_dependence: np.ndarray) -> dict[str, Any]:
    import latent_constraints as lc
    graph = dataset["graph"]
    X = np.asarray(dataset["X"], dtype=np.float64)
    observed = list(graph.observed)
    oi = {node: index for index, node in enumerate(observed)}
    weights, score = graph.estimate_weights(X, oi)
    if latcon:
        weights, score = lc.sign_fix(graph, weights, score)
    partial = runtime["optimize"].partial_residual_corr(graph, X, oi, score)
    if latcon:
        partial = lc.augmented_partial_corr(graph, X, oi, score, partial)
    names, matrix = list(observed), np.asarray(base_dependence, dtype=np.float64)
    if latcon:
        names, matrix = lc.augmented_bridge(graph, observed, oi, X, score, matrix)
    return {
        "weights": weights,
        "partial_corr": partial,
        "bridge": {"obs": names, "dep_marg": matrix, "lam_upper": 0.3, "kappa": 0.5, "q": 0.7},
    }


def _normalized_rows(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    return matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)


def fold_metrics(predictions: np.ndarray, targets: np.ndarray, masked: Sequence[int]) -> dict[str, Any]:
    import metrics as formal_metrics
    masked_np = np.asarray(masked, dtype=np.int64)
    ranked = core.rank_metrics(predictions, targets, masked_np)
    similarity = _normalized_rows(predictions) @ _normalized_rows(targets).T
    margins = np.asarray([
        similarity[row, truth] - np.max(np.delete(similarity[row], truth))
        for row, truth in enumerate(masked_np)
    ])
    center = targets.mean(axis=0)
    centered = np.sum(_normalized_rows(predictions - center) * _normalized_rows(targets[masked_np] - center), axis=1)
    return {
        "match_acc": float(formal_metrics.match_acc(predictions, masked_np, targets)),
        "exact_acc": float(formal_metrics.exact_acc(predictions, masked_np, _normalized_rows(targets))),
        "gold_cosine": float(np.mean(ranked["gold_embedding_cosine"])),
        "centered_cosine": float(np.mean(centered)),
        "prediction_margin": float(np.mean(margins)),
        "mrr": float(np.mean(ranked["reciprocal_rank"])),
        "recall_at_1": float(np.mean(ranked["recall_at_1"])),
        "recall_at_5": float(np.mean(ranked["recall_at_5"])),
        "per_node": {
            "gold_cosine": np.asarray(ranked["gold_embedding_cosine"]),
            "centered_cosine": centered, "prediction_margin": margins,
            "rank": np.asarray(ranked["rank"]), "mrr": np.asarray(ranked["reciprocal_rank"]),
            "recall_at_1": np.asarray(ranked["recall_at_1"]),
            "recall_at_5": np.asarray(ranked["recall_at_5"]),
            "exact": np.asarray(ranked["exact_decode"]),
            "match": core.hungarian_match_hits(predictions, targets[masked_np]),
        },
    }

def run_dataset_probe(
    runtime: Mapping[str, Any], config: Mapping[str, Any], dataset: Mapping[str, Any],
    *, role: str, core_arm: str, latcon: bool, include_baselines: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[tuple[str, int], dict[str, Any]], dict[str, Any]]:
    started = time.time()
    graph = dataset["graph"]
    observed = list(graph.observed)
    labels = [str(dataset["labels"][node]) for node in observed]
    targets = prime.encode_texts(runtime, labels)
    dependence, cache_path, cache_hash, cache_source = dependence_matrix(dataset)
    objective = prepare_objective(runtime, dataset, latcon=latcon, base_dependence=dependence)
    raw_corr = np.nan_to_num(np.corrcoef(np.asarray(dataset["X"]).T), nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(raw_corr, 0.0)
    permutation = np.random.default_rng(0).permutation(len(observed))
    folds = [permutation[index::5] for index in range(5)]
    records: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    decode_context: dict[tuple[str, int], dict[str, Any]] = {}
    probe_id = f"{dataset['name']}__latcon{int(latcon)}" if role == "hierarchy" else str(dataset["name"])
    for fold, raw_fold in enumerate(folds):
        masked = sorted(int(index) for index in raw_fold)
        visible = [index for index in range(len(observed)) if index not in set(masked)]
        visible_embeddings = {observed[index]: targets[index] for index in visible}
        features = runtime["torch"].tensor(
            runtime["l2_modules"].node_features(graph, objective["weights"], set(visible_embeddings)),
            dtype=runtime["torch"].float32, device="cpu",
        )
        embeddings, _ = runtime["l2_solver"].solve_unrolled(
            graph, objective["weights"], visible_embeddings, d=1024,
            weight_module=runtime["weightnet"], K=int(config["frozen_stage3"]["solver"]["unroll_steps"]),
            inner_lr=float(config["frozen_stage3"]["solver"]["inner_learning_rate"]),
            lam_zero=float(config["frozen_stage3"]["objective"]["lam_zero"]),
            lam_norm=float(config["frozen_stage3"]["objective"]["lam_norm"]),
            seed=fold, device="cpu", residual=float(config["frozen_stage3"]["objective"]["residual"]),
            lam_res=float(config["frozen_stage3"]["objective"]["lam_res"]),
            partial_corr=objective["partial_corr"], neg_op=runtime["negation"],
            bridge=objective["bridge"], train=False, feats=features,
        )
        predictions: dict[str, np.ndarray] = {
            core_arm: np.stack([embeddings[observed[index]] for index in masked])
        }
        if include_baselines:
            predictions["raw_correlation"] = prime._baseline_predictions(np.clip(raw_corr, 0.0, None), targets, masked, visible)
            predictions["uniform"] = prime._baseline_predictions(np.ones_like(raw_corr), targets, masked, visible)
        decode_context[(probe_id, fold)] = {"visible_embeddings": targets[visible], "targets": targets}
        for arm, predicted in predictions.items():
            metrics = fold_metrics(predicted, targets, masked)
            fold_rows.append({
                "probe_id": probe_id, "dataset": dataset.get("canonical_name", dataset["name"]),
                "implementation_dataset_id": dataset["name"], "role": role, "fold": fold,
                "arm": arm, "n_items": len(masked), "latent_constraints": latcon,
                **{key: value for key, value in metrics.items() if key != "per_node"},
            })
            per_node = metrics["per_node"]
            for position, item_index in enumerate(masked):
                records.append({
                    "probe_id": probe_id, "dataset": dataset.get("canonical_name", dataset["name"]),
                    "implementation_dataset_id": dataset["name"], "role": role, "fold": fold,
                    "arm": arm, "latent_constraints": latcon, "node_id": observed[item_index],
                    "true_label": labels[item_index],
                    "gold_cosine": float(per_node["gold_cosine"][position]),
                    "centered_cosine": float(per_node["centered_cosine"][position]),
                    "prediction_margin": float(per_node["prediction_margin"][position]),
                    "rank": int(per_node["rank"][position]), "mrr": float(per_node["mrr"][position]),
                    "recall_at_1": int(per_node["recall_at_1"][position]),
                    "recall_at_5": int(per_node["recall_at_5"][position]),
                    "match_acc": int(per_node["match"][position]), "exact": int(per_node["exact"][position]),
                    "judge_acc": None, "judge_status": "pending", "decoded_words": None,
                    "decoder_alpha": None, "_prediction": np.asarray(predicted[position], dtype=np.float64),
                })
        print(f"[{time.strftime('%H:%M:%S')}] bundle {probe_id} fold {fold + 1}/5", flush=True)
    metadata = {
        "probe_id": probe_id, "dataset_sha256": _sha256_dataset(dataset),
        "data_shape": list(np.asarray(dataset["X"]).shape), "observed_nodes": len(observed),
        "latent_nodes": len(graph.latents), "edge_count": len(graph.edges),
        "dependence_cache": str(cache_path.relative_to(REPO_ROOT)),
        "dependence_cache_sha256": cache_hash, "dependence_cache_source": cache_source,
        "runtime_seconds": time.time() - started,
    }
    return records, fold_rows, decode_context, metadata


def decode_bundle_records(
    prime_config: Mapping[str, Any], records: list[dict[str, Any]],
    contexts: Mapping[tuple[str, int], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    concepts, words = prime.load_normalized_dictionary(prime_config)
    evaluation = prime_config["evaluation"]
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[(str(row["probe_id"]), int(row["fold"]))].append(row)
    requests: list[dict[str, Any]] = []
    for key in sorted(grouped):
        group = grouped[key]
        alpha = prime.pick_alpha_visible(
            np.asarray(contexts[key]["visible_embeddings"]), concepts,
            screen_k=int(evaluation["decoder_screen_k"]), target_l0=int(evaluation["decoder_target_l0"]),
        )
        decoded = prime._decode_group(
            np.stack([row["_prediction"] for row in group]), concepts, words, alpha,
            screen_k=int(evaluation["decoder_screen_k"]), top_k=int(evaluation["decoder_top_k"]),
        )
        for row, concept_words in zip(group, decoded):
            row["decoded_words"] = concept_words
            row["decoder_alpha"] = float(alpha)
            requests.append({
                "request_id": f"bundle_{len(requests):04d}", "model": str(evaluation["judge"]["model"]),
                "mode": "completion", "rec": ", ".join(concept_words), "tgt": row["true_label"],
                "probe_id": row["probe_id"], "dataset": row["dataset"], "fold": row["fold"],
                "node_id": row["node_id"], "arm": row["arm"], "decoded_words": concept_words,
                "status": "pending",
            })
    _require(len(requests) == len(records), "bundle Judge request completeness failed")
    return requests

def aggregate_summary(fold_rows: Sequence[Mapping[str, Any]], references: Mapping[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in fold_rows:
        grouped[(str(row["probe_id"]), str(row["arm"]))].append(row)
    summary: list[dict[str, Any]] = []
    for (probe_id, arm), rows in sorted(grouped.items()):
        first = rows[0]
        dataset = str(first["dataset"])
        reference_key = "himi_match" if dataset == "himi" else "kims_match" if dataset == "kims" else "hierarchy_bigfive_match"
        reference_arm = {
            "core": "ours", "raw_correlation": "raw_correlation", "uniform": "uniform",
            "hierarchy_without_latent_constraints": "hierarchy_only",
            "hierarchy_with_latent_constraints": "hierarchy_latent_constraints",
        }.get(arm)
        reference_map = references.get(reference_key, {})
        summary.append({
            "probe_id": probe_id, "dataset": dataset,
            "implementation_dataset_id": first["implementation_dataset_id"], "role": first["role"],
            "arm": arm, "latent_constraints": first["latent_constraints"],
            "n_items": int(sum(int(row["n_items"]) for row in rows)), "n_folds": len(rows),
            "judge_status": "pending", "judge_acc": None,
            "match_acc": float(np.mean([row["match_acc"] for row in rows])),
            "exact_acc": float(np.mean([row["exact_acc"] for row in rows])),
            "gold_cosine": float(np.mean([row["gold_cosine"] for row in rows])),
            "centered_cosine": float(np.mean([row["centered_cosine"] for row in rows])),
            "prediction_margin": float(np.mean([row["prediction_margin"] for row in rows])),
            "mrr": float(np.mean([row["mrr"] for row in rows])),
            "recall_at_1": float(np.mean([row["recall_at_1"] for row in rows])),
            "recall_at_5": float(np.mean([row["recall_at_5"] for row in rows])),
            "reference_match_acc": float(reference_map[reference_arm]) if reference_arm in reference_map else None,
            "reference_source": "Week-6 report",
        })
    return summary


def classify_trends(summary: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = {(str(row["dataset"]), str(row["arm"])): float(row["match_acc"]) for row in summary}
    himi = values[("himi", "core")] > values[("himi", "raw_correlation")] > values[("himi", "uniform")]
    kims = values[("kims", "raw_correlation")] > values[("kims", "core")] > values[("kims", "uniform")]
    hierarchy_delta = values[("bigfive", "hierarchy_with_latent_constraints")] - values[("bigfive", "hierarchy_without_latent_constraints")]
    hierarchy = hierarchy_delta > 0.0
    passed = himi and kims and hierarchy
    return {
        "behavioral_trend_reproduced": passed, "checkpoint_or_bundle_drift": not passed,
        "himi_expected_order_core_gt_raw_gt_uniform": himi,
        "kims_expected_order_raw_gt_core_gt_uniform": kims,
        "hierarchy_latent_constraints_effective": hierarchy,
        "hierarchy_match_delta_on_minus_off": hierarchy_delta,
        "judge_trend_status": "not_evaluable_without_api",
    }


def write_markdown(summary: Sequence[Mapping[str, Any]], decision: Mapping[str, Any]) -> None:
    lines = [
        "# E0-double-prime Bundle Replication", "",
        f"Behavioral trend reproduced: **{decision['behavioral_trend_reproduced']}**. "
        f"`checkpoint_or_bundle_drift={decision['checkpoint_or_bundle_drift']}`.", "",
        "The selected probes reuse the exact E0-prime LoRA, WeightNet, negation operator, objective "
        "coefficients, candidate dictionary, and decoder. No artifact was retrained. Judge is pending; "
        "no missing verdict was treated as zero.", "",
        "| dataset | role | arm | items | Match | Week-6 Match | gold cos | centered cos | MRR | R@5 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        reference = "—" if row["reference_match_acc"] is None else f"{row['reference_match_acc']:.3f}"
        lines.append(
            f"| {row['dataset']} | {row['role']} | {row['arm']} | {row['n_items']} | "
            f"{row['match_acc']:.3f} | {reference} | {row['gold_cosine']:.3f} | "
            f"{row['centered_cosine']:.3f} | {row['mrr']:.3f} | {row['recall_at_5']:.3f} |"
        )
    lines.extend([
        "", "Qualitative checks:", "",
        f"- himi `core > raw correlation > uniform`: **{decision['himi_expected_order_core_gt_raw_gt_uniform']}**",
        f"- kims `raw correlation > core > uniform`: **{decision['kims_expected_order_raw_gt_core_gt_uniform']}**",
        f"- BigFive2 latent constraints on minus off Match: **{decision['hierarchy_match_delta_on_minus_off']:+.4f}**",
        "", "The hierarchy implementation ID is `bigfive2`; the canonical dataset is `bigfive`. Its "
        "missing Pearson cache was generated deterministically from the same observed X inside the audit "
        "results directory, with zero diagonal and recorded SHA-256.", "",
        "The original reported release artifacts are absent and the local bundle had been retrained before "
        "E0-prime; this is a provenance limitation. It becomes a behavioral drift confound here only if "
        "the selected qualitative orderings or hierarchy direction fail.", "", "Command:", "",
        "```powershell", ".\\.venv\\Scripts\\python.exe task3_v2\\scripts\\run_e0_bundle_replication.py", "```", "",
    ])
    (EXPERIMENT_DIR / "bundle_replication.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--skip-decode", action="store_true", help="debug only")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.time()
    config, prime_config, _specs, artifact_report = audit.validate_and_load_config(args.config.resolve(), hash_artifacts=True)
    runtime = prime.load_frozen_runtime(prime_config)
    datasets = load_datasets(runtime)
    _require(
        config["bundle_replication"]["datasets"] == {"dev": "himi", "heldout": "kims", "hierarchy": "bigfive2"},
        "bundle dataset freeze changed",
    )
    all_records: list[dict[str, Any]] = []
    all_folds: list[dict[str, Any]] = []
    all_contexts: dict[tuple[str, int], dict[str, Any]] = {}
    metadata: list[dict[str, Any]] = []
    probes = [
        (datasets["himi"], "dev", "core", True, True),
        (datasets["kims"], "heldout", "core", True, True),
        (datasets["bigfive2"], "hierarchy", "hierarchy_without_latent_constraints", False, False),
        (datasets["bigfive2"], "hierarchy", "hierarchy_with_latent_constraints", True, False),
    ]
    for dataset, role, arm, latcon, baselines in probes:
        records, folds, contexts, probe_metadata = run_dataset_probe(
            runtime, config, dataset, role=role, core_arm=arm, latcon=latcon, include_baselines=baselines
        )
        all_records.extend(records)
        all_folds.extend(folds)
        all_contexts.update(contexts)
        metadata.append(probe_metadata)
    if args.skip_decode:
        requests = []
        for index, row in enumerate(all_records):
            row["judge_status"] = "debug_decode_skipped"
            row["decoded_words"] = []
            requests.append({
                "request_id": f"bundle_{index:04d}", "model": "gpt-5.5", "mode": "completion",
                "rec": "", "tgt": row["true_label"], "probe_id": row["probe_id"],
                "dataset": row["dataset"], "fold": row["fold"], "node_id": row["node_id"],
                "arm": row["arm"], "status": "debug_decode_skipped",
            })
    else:
        requests = decode_bundle_records(prime_config, all_records, all_contexts)
    summary = aggregate_summary(all_folds, config["bundle_replication"]["week6_reference"])
    decision = classify_trends(summary)
    public_records = [{key: value for key, value in row.items() if not key.startswith("_")} for row in all_records]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_fields = (
        "probe_id", "dataset", "implementation_dataset_id", "role", "arm", "latent_constraints",
        "n_items", "n_folds", "judge_status", "judge_acc", "match_acc", "exact_acc", "gold_cosine",
        "centered_cosine", "prediction_margin", "mrr", "recall_at_1", "recall_at_5",
        "reference_match_acc", "reference_source",
    )
    node_fields = (
        "probe_id", "dataset", "implementation_dataset_id", "role", "fold", "arm",
        "latent_constraints", "node_id", "true_label", "gold_cosine", "centered_cosine",
        "prediction_margin", "rank", "mrr", "recall_at_1", "recall_at_5", "match_acc", "exact",
        "judge_acc", "judge_status", "decoder_alpha", "decoded_words",
    )
    _write_csv(RESULTS_DIR / "bundle_replication.csv", summary, summary_fields)
    csv_records = []
    for row in public_records:
        value = dict(row)
        value["decoded_words"] = json.dumps(value["decoded_words"], ensure_ascii=False)
        csv_records.append(value)
    _write_csv(RESULTS_DIR / "bundle_per_node.csv", csv_records, node_fields)
    _write_jsonl(RESULTS_DIR / "bundle_judge_requests.jsonl", requests)
    run_command = ".\\.venv\\Scripts\\python.exe task3_v2\\scripts\\run_e0_bundle_replication.py"
    output = {
        "schema_version": "task3.e0_double_prime.bundle_replication.v1",
        "created_at_utc": _utc_now(),
        "status": "formal_local_api_free" if not args.skip_decode else "debug_skip_decode",
        **decision,
        "summary": summary,
        "probe_metadata": metadata,
        "provenance": {
            "runner": "audit adapter over v5/run_task1.py protocol and frozen Stage-3 libraries",
            "source_bindings": {
                "audit_config_path": str(args.config.resolve().relative_to(REPO_ROOT)),
                "audit_config_sha256": _sha256_file(args.config.resolve()),
                "source_e0_prime_config_path": str(
                    (REPO_ROOT / config["source_e0_prime"]["config"])
                    .resolve()
                    .relative_to(REPO_ROOT)
                ),
                "source_e0_prime_config_sha256": config["source_e0_prime"][
                    "config_sha256"
                ],
                "bundle_runner_path": str(SCRIPT_PATH.relative_to(REPO_ROOT)),
                "bundle_runner_sha256": _sha256_file(SCRIPT_PATH),
            },
            "official_entrypoints_audited": ["v5/main.py", "v5/run_bigfive_hier.py"],
            "bundle_artifacts": artifact_report,
            "retrained_in_replication": False,
            "judge_api_called": False,
            "judge_request_count": len(requests),
            "decoder_run": not args.skip_decode,
            "original_release_bundle_present": False,
            "local_bundle_origin": "locally retrained before E0-prime; see v5/outputs/reproduction_report.md",
            "runtime_seconds": time.time() - started,
        },
        "api_free_limitations": {
            "judge_acc": None, "judge_trend": "not_evaluable_without_api",
            "missing_judge_not_scored_as_zero": True,
        },
        "run_commands": [run_command],
    }
    _write_json(RESULTS_DIR / "bundle_replication.json", output)
    write_markdown(summary, decision)
    print(json.dumps({
        "status": output["status"],
        "behavioral_trend_reproduced": decision["behavioral_trend_reproduced"],
        "checkpoint_or_bundle_drift": decision["checkpoint_or_bundle_drift"],
        "hierarchy_match_delta": decision["hierarchy_match_delta_on_minus_off"],
        "rows": len(public_records), "judge_requests": len(requests),
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())