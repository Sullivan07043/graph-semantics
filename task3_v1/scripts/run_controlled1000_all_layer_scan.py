"""Exploratory all-layer Stage 1 scan for controlled-current32 v1.

This does not feed a 992-node matrix to CauScale.  It extracts every fitted
J-lens layer once, scans all ordered layer pairs as 64-node problems, removes
same-concept edges from candidate ranking, and bootstraps only the strongest
innovation-residualized layer pairs.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.covariance import LedoitWolf
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
TASK3 = ROOT / "task3_v1"
SCRIPTS = TASK3 / "scripts"
sys.path[:0] = [str(TASK3), str(SCRIPTS)]

import run_official90_stage1_exploratory as common  # noqa: E402
from build_discovery_matrix import CONCEPTS  # noqa: E402
from build_innovation_matrix import fit_ridge  # noqa: E402
from paper_aligned_core import load_config, sha256_file  # noqa: E402
from run_controlled1000_stage1_exploratory import load_prompts  # noqa: E402
from run_causcale_smoke import directed_probabilities  # noqa: E402
from run_paper_aligned_jspace import load_components  # noqa: E402


@torch.no_grad()
def extract_all_layers(
    model,
    tokenizer,
    lens,
    prompts: list[dict[str, Any]],
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    from jlens.hooks import ActivationRecorder
    from torch.nn import functional

    layers = sorted(map(int, lens.jacobians))
    token_ids = []
    for concept in CONCEPTS:
        encoded = tokenizer.encode(concept, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"Concept is not one token: {concept!r} -> {encoded}")
        token_ids.append(int(encoded[0]))
    vectors = {}
    for layer in layers:
        jacobian = lens.jacobians[layer].to(device="cuda", dtype=torch.float32)
        projected = jacobian.T @ model._lm_head.weight[token_ids].float().T
        vectors[layer] = functional.normalize(projected, dim=0)

    values: dict[int, list[list[float]]] = {layer: [] for layer in layers}
    timings = []
    token_counts = []
    for index, item in enumerate(prompts):
        tick = time.time()
        input_ids = model.encode(item["prompt"])
        with ActivationRecorder(model.layers, at=layers) as recorder:
            model.forward(input_ids)
            for layer in layers:
                residual = recorder.activations[layer][0, -1].detach().float()
                values[layer].append((residual @ vectors[layer]).cpu().tolist())
        torch.cuda.synchronize()
        timings.append(time.time() - tick)
        token_counts.append(int(input_ids.shape[-1]))
        if (index + 1) % 50 == 0:
            logging.info("All-layer projection %d/%d", index + 1, len(prompts))
    arrays = {
        layer: np.asarray(layer_values, dtype=np.float32)
        for layer, layer_values in values.items()
    }
    return arrays, {
        "layers": layers,
        "layer_count": len(layers),
        "prompt_seconds_mean": float(np.mean(timings)),
        "prompt_seconds_total": float(np.sum(timings)),
        "prompt_token_count_min": min(token_counts),
        "prompt_token_count_median": float(np.median(token_counts)),
        "prompt_token_count_max": max(token_counts),
    }


def residualize_pair(
    source: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    splitter = GroupKFold(n_splits=len(np.unique(groups)))
    residual = target.astype(np.float64).copy()
    audits = []
    for concept_index, concept in enumerate(CONCEPTS):
        x = source[:, concept_index : concept_index + 1].astype(np.float64)
        y = target[:, concept_index].astype(np.float64)
        oof = np.full(len(y), np.nan, dtype=np.float64)
        for train, test in splitter.split(x, y, groups=groups):
            coefficient, intercept = fit_ridge(x[train], y[train], 1.0)
            oof[test] = x[test] @ coefficient + intercept
        coefficient, intercept = fit_ridge(x, y, 1.0)
        residual[:, concept_index] = y - (x @ coefficient + intercept)
        total = float(np.sum((y - y.mean()) ** 2))
        oof_r2 = (
            1.0 - float(np.sum((y - oof) ** 2)) / total if total > 0 else 0.0
        )
        audits.append({"concept": concept, "oof_r2": oof_r2})
    return residual.astype(np.float32), {
        "method": "target same-concept source-layer ridge residualization",
        "ridge_alpha": 1.0,
        "group_folds": int(len(np.unique(groups))),
        "oof_r2_min": float(min(row["oof_r2"] for row in audits)),
        "oof_r2_median": float(np.median([row["oof_r2"] for row in audits])),
        "oof_r2_max": float(max(row["oof_r2"] for row in audits)),
    }


def scan_one(
    causcale,
    source_layer: int,
    target_layer: int,
    matrix: np.ndarray,
    token_ids: list[int],
) -> tuple[dict[str, Any], np.ndarray]:
    columns = common.columns_for([source_layer, target_layer], token_ids)
    inference, scores = common.infer_causcale(causcale, matrix, columns)
    block = scores[: len(CONCEPTS), len(CONCEPTS) :]
    cross = ~np.eye(len(CONCEPTS), dtype=bool)
    cross_values = block[cross]
    selected = cross & (block >= 0.5)
    selected_indices = np.argwhere(selected)
    order = sorted(
        range(len(selected_indices)),
        key=lambda index: block[tuple(selected_indices[index])],
        reverse=True,
    )
    top = []
    for rank, order_index in enumerate(order[:10], start=1):
        source, target = selected_indices[order_index]
        top.append(
            {
                "rank": rank,
                "source_concept": CONCEPTS[int(source)],
                "target_concept": CONCEPTS[int(target)],
                "probability": float(block[source, target]),
            }
        )
    result = {
        "source_layer": source_layer,
        "target_layer": target_layer,
        "node_count": int(matrix.shape[1]),
        "same_concept_edges_masked_from_candidates": True,
        "selected_cross_edges_ge_0_5": int(selected.sum()),
        "cross_probability_median": float(np.median(cross_values)),
        "cross_probability_p95": float(np.percentile(cross_values, 95)),
        "cross_probability_p99": float(np.percentile(cross_values, 99)),
        "cross_probability_max": float(np.max(cross_values)),
        "cross_excess_mass_above_0_5": float(
            np.maximum(cross_values - 0.5, 0.0).sum()
        ),
        "same_concept_positive_control": {
            "causcale_auprc": inference["causcale_same_concept_auprc"],
            "absolute_correlation_auprc": (
                inference["absolute_correlation_same_concept_auprc"]
            ),
        },
        "top_cross_edges": top,
    }
    return result, block


@torch.no_grad()
def bootstrap_pair(
    causcale,
    matrix: np.ndarray,
    source_layer: int,
    target_layer: int,
    *,
    runs: int,
) -> dict[str, Any]:
    concept_count = len(CONCEPTS)
    concepts_per_run = round(0.8 * concept_count)
    availability = np.zeros((concept_count, concept_count), dtype=np.int32)
    selected_count = np.zeros((concept_count, concept_count), dtype=np.int32)
    probability_sum = np.zeros((concept_count, concept_count), dtype=np.float64)
    edge_sets: list[set[int]] = []
    available_sets: list[set[int]] = []
    run_records = []
    for run in range(runs):
        rng = np.random.RandomState(20260723 + run)
        chosen = np.sort(
            rng.choice(concept_count, size=concepts_per_run, replace=False)
        )
        nodes = np.concatenate([chosen, concept_count + chosen])
        row_indices = rng.choice(len(matrix), size=len(matrix), replace=True)
        sampled = matrix[row_indices][:, nodes]
        prior = LedoitWolf().fit(sampled).get_precision().astype(np.float32)
        mean = sampled.mean(axis=0, keepdims=True)
        std = sampled.std(axis=0, keepdims=True)
        standardized = (sampled - mean) / np.where(std < 1e-8, 1.0, std)
        batch = {
            "data": torch.from_numpy(standardized).unsqueeze(0).cuda(),
            "interv": torch.zeros(
                1, len(sampled), len(nodes), device="cuda"
            ),
            "feats": torch.from_numpy(prior).unsqueeze(0).cuda(),
        }
        encoded = causcale.encoder(batch)
        directed, _ = directed_probabilities(causcale, encoded, len(nodes))
        local = directed.cpu().numpy()[:concepts_per_run, concepts_per_run:]
        global_rows, global_cols = np.meshgrid(chosen, chosen, indexing="ij")
        cross = global_rows != global_cols
        global_rows = global_rows[cross]
        global_cols = global_cols[cross]
        values = local[cross]
        availability[global_rows, global_cols] += 1
        selected_count[global_rows, global_cols] += (values >= 0.5).astype(np.int32)
        probability_sum[global_rows, global_cols] += values
        selected_flat = set(
            (global_rows[values >= 0.5] * concept_count + global_cols[values >= 0.5])
            .astype(int)
            .tolist()
        )
        available_flat = set(
            (global_rows * concept_count + global_cols).astype(int).tolist()
        )
        edge_sets.append(selected_flat)
        available_sets.append(available_flat)
        run_records.append(
            {"run": run, "selected_cross_edge_count": len(selected_flat)}
        )

    observed = availability > 0
    frequency = np.zeros_like(probability_sum, dtype=np.float32)
    mean_probability = np.zeros_like(probability_sum, dtype=np.float32)
    frequency[observed] = selected_count[observed] / availability[observed]
    mean_probability[observed] = probability_sum[observed] / availability[observed]
    stable = observed & (frequency >= 0.8) & (mean_probability >= 0.5)
    stable_indices = np.argwhere(stable)
    stable_order = sorted(
        range(len(stable_indices)),
        key=lambda index: (
            frequency[tuple(stable_indices[index])],
            mean_probability[tuple(stable_indices[index])],
        ),
        reverse=True,
    )
    stable_edges = []
    for rank, order_index in enumerate(stable_order, start=1):
        source, target = stable_indices[order_index]
        stable_edges.append(
            {
                "rank": rank,
                "source_layer": source_layer,
                "target_layer": target_layer,
                "source_concept": CONCEPTS[int(source)],
                "target_concept": CONCEPTS[int(target)],
                "selection_frequency": float(frequency[source, target]),
                "mean_probability": float(mean_probability[source, target]),
            }
        )
    jaccards = []
    for left, right in itertools.combinations(range(runs), 2):
        common_available = available_sets[left] & available_sets[right]
        left_edges = edge_sets[left] & common_available
        right_edges = edge_sets[right] & common_available
        union = left_edges | right_edges
        jaccards.append(
            len(left_edges & right_edges) / len(union) if union else 1.0
        )
    return {
        "runs": runs,
        "sample_bootstrap": f"{len(matrix)} rows with replacement",
        "feature_bootstrap": f"{concepts_per_run}/32 concept groups",
        "same_concept_edges_excluded": True,
        "stable_cross_edge_count": len(stable_edges),
        "pairwise_cross_edge_jaccard_min": float(np.min(jaccards)),
        "pairwise_cross_edge_jaccard_median": float(np.median(jaccards)),
        "pairwise_cross_edge_jaccard_max": float(np.max(jaccards)),
        "run_records": run_records,
        "stable_cross_edges": stable_edges,
    }


def make_summary(result: dict[str, Any]) -> str:
    lines = [
        "# Controlled-current32 all-layer pair scan",
        "",
        "All fitted layers were extracted, but CauScale was run on 64-node layer "
        "pairs rather than one underdetermined 992-node graph. Same-concept edges "
        "were excluded from ranking and bootstrap stability.",
        "",
        "| Rank | Layers | Innovation cross edges >=.5 | Stable cross | "
        "Median Jaccard |",
        "|---:|---:|---:|---:|---:|",
    ]
    for rank, item in enumerate(result["bootstrapped_pairs"], start=1):
        scan = item["scan"]
        bootstrap = item["bootstrap"]
        lines.append(
            f"| {rank} | {scan['source_layer']}->{scan['target_layer']} | "
            f"{scan['selected_cross_edges_ge_0_5']} | "
            f"{bootstrap['stable_cross_edge_count']} | "
            f"{bootstrap['pairwise_cross_edge_jaccard_median']:.3f} |"
        )
    lines.extend(
        [
            "",
            "This is a data-driven exploratory scan with multiple comparisons. "
            "Stable edges remain candidates until replicated on behaviorally "
            "validated prompts and tested by held-out interventions.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=TASK3 / "configs/paper_aligned_jspace.yaml",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=TASK3 / "data/prompts/controlled_current32/v1/prompts.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=TASK3 / "outputs/stage1_controlled1000_all_layer_scan",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=TASK3 / "logs/stage1_controlled1000_all_layer_scan",
    )
    parser.add_argument("--bootstrap-runs", type=int, default=20)
    parser.add_argument("--top-pairs", type=int, default=20)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(args.log_dir / "run.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    prompts = load_prompts(args.prompts)
    groups = np.asarray([str(row["fold"]) for row in prompts], dtype=str)
    started = time.time()
    torch.cuda.reset_peak_memory_stats()

    config = load_config(args.config)
    logging.info("Loading Qwen and the frozen n1000 J-lens")
    model, tokenizer, lens, loading = load_components(config, ROOT)
    arrays, extraction = extract_all_layers(model, tokenizer, lens, prompts)
    token_ids = [
        int(tokenizer.encode(concept, add_special_tokens=False)[0])
        for concept in CONCEPTS
    ]
    del model, tokenizer, lens
    torch.cuda.empty_cache()
    np.savez_compressed(
        args.output_dir / "all_layer_features_1000x32.npz",
        **{f"layer_{layer}": array for layer, array in arrays.items()},
    )

    sys.path.insert(0, str(ROOT.parent / ".deps/CauScale-main/src"))
    logging.info("Loading released CauScale checkpoint")
    causcale, causcale_src, checkpoint = common.load_causcale()
    pair_results = []
    score_blocks: dict[str, np.ndarray] = {}
    pairs = list(itertools.combinations(sorted(arrays), 2))
    for pair_index, (source_layer, target_layer) in enumerate(pairs, start=1):
        source = arrays[source_layer]
        target = arrays[target_layer]
        target_innovation, audit = residualize_pair(source, target, groups)
        raw_matrix = np.concatenate([source, target], axis=1)
        innovation_matrix = np.concatenate([source, target_innovation], axis=1)
        raw_result, raw_scores = scan_one(
            causcale, source_layer, target_layer, raw_matrix, token_ids
        )
        innovation_result, innovation_scores = scan_one(
            causcale, source_layer, target_layer, innovation_matrix, token_ids
        )
        innovation_result["innovation_audit"] = audit
        key = f"{source_layer}_{target_layer}"
        score_blocks[f"raw_{key}"] = raw_scores.astype(np.float16)
        score_blocks[f"innovation_{key}"] = innovation_scores.astype(np.float16)
        pair_results.append(
            {
                "source_layer": source_layer,
                "target_layer": target_layer,
                "raw": raw_result,
                "innovation": innovation_result,
            }
        )
        if pair_index % 25 == 0 or pair_index == len(pairs):
            logging.info("Scanned %d/%d layer pairs", pair_index, len(pairs))

    ranked = sorted(
        pair_results,
        key=lambda row: (
            row["innovation"]["selected_cross_edges_ge_0_5"],
            row["innovation"]["cross_excess_mass_above_0_5"],
            row["innovation"]["cross_probability_max"],
        ),
        reverse=True,
    )
    bootstrapped = []
    for rank, row in enumerate(ranked[: args.top_pairs], start=1):
        source_layer = row["source_layer"]
        target_layer = row["target_layer"]
        logging.info(
            "Bootstrap rank %d: layers %d->%d",
            rank,
            source_layer,
            target_layer,
        )
        target_innovation, _ = residualize_pair(
            arrays[source_layer], arrays[target_layer], groups
        )
        matrix = np.concatenate(
            [arrays[source_layer], target_innovation], axis=1
        )
        bootstrapped.append(
            {
                "scan": row["innovation"],
                "bootstrap": bootstrap_pair(
                    causcale,
                    matrix,
                    source_layer,
                    target_layer,
                    runs=args.bootstrap_runs,
                ),
            }
        )

    aggregate: dict[tuple[str, str], dict[str, Any]] = {}
    for item in pair_results:
        for edge in item["innovation"]["top_cross_edges"]:
            key = (edge["source_concept"], edge["target_concept"])
            record = aggregate.setdefault(
                key,
                {
                    "source_concept": key[0],
                    "target_concept": key[1],
                    "selected_layer_pair_count": 0,
                    "maximum_probability": 0.0,
                    "layer_pairs": [],
                },
            )
            record["selected_layer_pair_count"] += 1
            record["maximum_probability"] = max(
                record["maximum_probability"], edge["probability"]
            )
            record["layer_pairs"].append(
                {
                    "source_layer": item["source_layer"],
                    "target_layer": item["target_layer"],
                    "probability": edge["probability"],
                }
            )
    aggregate_edges = sorted(
        aggregate.values(),
        key=lambda row: (
            row["selected_layer_pair_count"],
            row["maximum_probability"],
        ),
        reverse=True,
    )
    result = {
        "status": (
            "exploratory_all_layer_pair_scan_candidate_not_causal_graph"
        ),
        "prompt_file": str(args.prompts),
        "prompt_sha256": sha256_file(args.prompts),
        "prompt_count": len(prompts),
        "concept_count": len(CONCEPTS),
        "all_layer_node_count_if_concatenated": len(arrays) * len(CONCEPTS),
        "concatenated_992_node_graph_was_not_run": True,
        "scan_design": {
            "layer_count": len(arrays),
            "layer_pair_count": len(pairs),
            "nodes_per_causcale_run": 64,
            "same_concept_edges_excluded_from_candidate_ranking": True,
            "innovation_before_candidate_ranking": True,
            "top_pairs_bootstrapped": args.top_pairs,
            "bootstrap_runs": args.bootstrap_runs,
        },
        "loading": loading,
        "extraction": extraction,
        "causcale_source": str(causcale_src),
        "causcale_checkpoint": str(checkpoint),
        "causcale_checkpoint_sha256": sha256_file(checkpoint),
        "pair_results": pair_results,
        "bootstrapped_pairs": bootstrapped,
        "aggregate_cross_concept_edges": aggregate_edges,
        "limitations": [
            "candidate prompts are not behaviorally validated or frozen",
            "layer pairs omit other-layer confounding and mediation",
            "top pairs are selected on the same data used for bootstrap",
            "465-pair scan incurs substantial multiple comparisons",
            "no independent held-out intervention validation",
        ],
        "total_seconds": time.time() - started,
        "peak_cuda_memory_gib": (
            torch.cuda.max_memory_allocated() / 1024**3
        ),
        "environment": common.environment(),
    }
    common.write_json(args.output_dir / "all_layer_pair_scan.json", result)
    np.savez_compressed(
        args.output_dir / "pair_cross_probability_blocks_float16.npz",
        **score_blocks,
    )
    (args.output_dir / "SUMMARY.md").write_text(
        make_summary(result), encoding="utf-8"
    )
    logging.info("Completed in %.1f seconds", result["total_seconds"])


if __name__ == "__main__":
    main()
