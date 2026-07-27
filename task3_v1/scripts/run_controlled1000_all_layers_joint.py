"""Run one joint CauScale graph over all fitted J-space layers.

The fitted lens currently exposes layers 0..30.  With 32 concepts this creates
992 nodes.  Same-concept cross-layer edges are retained only as diagnostics and
are masked from the reported candidate set.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
TASK3 = ROOT / "task3_v1"
SCRIPTS = TASK3 / "scripts"
sys.path[:0] = [str(TASK3), str(SCRIPTS)]

import run_official90_stage1_exploratory as common  # noqa: E402
from build_discovery_matrix import CONCEPTS  # noqa: E402
from build_innovation_matrix import fit_ridge  # noqa: E402
from run_controlled1000_stage1_exploratory import load_prompts  # noqa: E402


def innovation_all_layers(
    matrix: np.ndarray,
    layers: list[int],
    groups: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    concept_count = len(CONCEPTS)
    innovation = matrix.astype(np.float64).copy()
    splitter = GroupKFold(n_splits=len(np.unique(groups)))
    audits = []
    for concept_index, concept in enumerate(CONCEPTS):
        for layer_position in range(1, len(layers)):
            target_index = layer_position * concept_count + concept_index
            predictor_indices = np.asarray(
                [
                    earlier * concept_count + concept_index
                    for earlier in range(layer_position)
                ],
                dtype=np.int64,
            )
            predictors = matrix[:, predictor_indices].astype(np.float64)
            target = matrix[:, target_index].astype(np.float64)
            oof = np.full(len(matrix), np.nan, dtype=np.float64)
            for train, test in splitter.split(
                predictors, target, groups=groups
            ):
                coefficient, intercept = fit_ridge(
                    predictors[train], target[train], 1.0
                )
                oof[test] = predictors[test] @ coefficient + intercept
            coefficient, intercept = fit_ridge(predictors, target, 1.0)
            fitted = predictors @ coefficient + intercept
            innovation[:, target_index] = target - fitted
            total = float(np.sum((target - target.mean()) ** 2))
            oof_r2 = (
                1.0 - float(np.sum((target - oof) ** 2)) / total
                if total > 0
                else 0.0
            )
            audits.append(
                {
                    "concept": concept,
                    "layer": layers[layer_position],
                    "predictor_count": int(len(predictor_indices)),
                    "oof_r2": oof_r2,
                }
            )
        logging.info(
            "Innovation residualization %d/%d concepts",
            concept_index + 1,
            concept_count,
        )
    values = [row["oof_r2"] for row in audits]
    return innovation.astype(np.float32), {
        "method": "same-concept all-earlier-layer ridge residualization",
        "ridge_alpha": 1.0,
        "group_folds": int(len(np.unique(groups))),
        "regression_count": len(audits),
        "oof_r2_min": float(np.min(values)),
        "oof_r2_median": float(np.median(values)),
        "oof_r2_max": float(np.max(values)),
        "audits": audits,
    }


def cross_concept_summary(
    scores: np.ndarray,
    columns: list[dict[str, Any]],
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    allowed, same = common.allowed_and_same(columns)
    candidates = allowed & ~same
    values = scores[candidates]
    selected = candidates & (scores >= threshold)
    indices = np.argwhere(selected)
    order = sorted(
        range(len(indices)),
        key=lambda index: scores[tuple(indices[index])],
        reverse=True,
    )
    top = []
    for rank, order_index in enumerate(order[:100], start=1):
        source, target = indices[order_index]
        top.append(
            {
                "rank": rank,
                "source_index": int(source),
                "target_index": int(target),
                "source_layer": int(columns[source]["layer"]),
                "target_layer": int(columns[target]["layer"]),
                "source_concept": columns[source]["concept"],
                "target_concept": columns[target]["concept"],
                "probability": float(scores[source, target]),
            }
        )
    return {
        "same_concept_edges_masked": True,
        "allowed_cross_concept_edge_count": int(candidates.sum()),
        "selected_cross_concept_edges_ge_0_5": int(selected.sum()),
        "selected_cross_concept_fraction": float(selected.sum() / candidates.sum()),
        "cross_probability_median": float(np.median(values)),
        "cross_probability_p95": float(np.percentile(values, 95)),
        "cross_probability_p99": float(np.percentile(values, 99)),
        "cross_probability_max": float(np.max(values)),
        "top_cross_concept_edges": top,
    }


def make_summary(result: dict[str, Any]) -> str:
    lines = [
        "# Controlled-current32 joint all-layer CauScale",
        "",
        f"Layers: {result['layers'][0]}..{result['layers'][-1]} "
        f"({len(result['layers'])} fitted layers); nodes: {result['node_count']}.",
        "",
        "All same-concept cross-layer edges are excluded from the candidate "
        "counts and rankings below.",
        "",
        "| Matrix | Status | Cross candidates >=.5 | Fraction | Cross p99 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("raw", "innovation"):
        item = result["matrices"][name]
        if item["status"] != "completed":
            lines.append(
                f"| {name} | {item['status']} | n/a | n/a | n/a |"
            )
            continue
        summary = item["cross_concept_summary"]
        lines.append(
            f"| {name} | completed | "
            f"{summary['selected_cross_concept_edges_ge_0_5']} | "
            f"{summary['selected_cross_concept_fraction']:.4f} | "
            f"{summary['cross_probability_p99']:.3f} |"
        )
    lines.extend(
        [
            "",
            "This 992-node run is outside the released CauScale checkpoint's "
            "demonstrated approximately 500-node regime. The output is therefore "
            "an exploratory stress test, not a validated causal graph.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        type=Path,
        default=TASK3
        / "outputs/stage1_controlled1000_all_layer_scan"
        / "all_layer_features_1000x32.npz",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=TASK3 / "data/prompts/controlled_current32/v1/prompts.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=TASK3 / "outputs/stage1_controlled1000_all_layers_joint",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=TASK3 / "logs/stage1_controlled1000_all_layers_joint",
    )
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
    with np.load(args.features) as payload:
        layers = sorted(
            int(key.removeprefix("layer_"))
            for key in payload.files
            if key.startswith("layer_")
        )
        layer_arrays = {
            layer: payload[f"layer_{layer}"].astype(np.float32)
            for layer in layers
        }
    if layers != list(range(31)):
        raise RuntimeError(f"Expected fitted layers 0..30, got {layers}")
    raw = np.concatenate([layer_arrays[layer] for layer in layers], axis=1)
    if raw.shape != (1000, 992):
        raise RuntimeError(f"Expected (1000, 992), got {raw.shape}")
    np.save(args.output_dir / "raw_matrix_1000x992.npy", raw)

    logging.info("Building generalized all-layer innovation matrix")
    innovation, innovation_audit = innovation_all_layers(raw, layers, groups)
    np.save(args.output_dir / "innovation_matrix_1000x992.npy", innovation)

    columns = common.columns_for(layers, list(range(len(CONCEPTS))))
    sys.path.insert(0, str(ROOT.parent / ".deps/CauScale-main/src"))
    logging.info("Loading released CauScale checkpoint")
    causcale, causcale_src, checkpoint = common.load_causcale()
    result: dict[str, Any] = {
        "status": "exploratory_joint_992_node_stress_test",
        "prompt_count": len(prompts),
        "features": str(args.features),
        "layers": layers,
        "concept_count": len(CONCEPTS),
        "node_count": raw.shape[1],
        "sample_to_node_ratio": float(len(raw) / raw.shape[1]),
        "allowed_cross_concept_edge_count": int(
            len(list(__import__("itertools").combinations(layers, 2)))
            * len(CONCEPTS)
            * (len(CONCEPTS) - 1)
        ),
        "same_concept_edges_excluded_from_candidates": True,
        "innovation_audit": innovation_audit,
        "causcale_source": str(causcale_src),
        "causcale_checkpoint": str(checkpoint),
        "checkpoint_regime_warning": (
            "992 nodes exceed the approximately 500-node demonstrated regime"
        ),
        "matrices": {},
    }
    started = time.time()
    for name, matrix in (("raw", raw), ("innovation", innovation)):
        logging.info("Joint 992-node CauScale inference: %s", name)
        tick = time.time()
        try:
            inference, scores = common.infer_causcale(
                causcale, matrix, columns
            )
            cross_summary = cross_concept_summary(scores, columns)
            np.save(
                args.output_dir / f"{name}_directed_probabilities_float16.npy",
                scores.astype(np.float16),
            )
            result["matrices"][name] = {
                "status": "completed",
                "seconds": time.time() - tick,
                "positive_control_inference": inference,
                "cross_concept_summary": cross_summary,
            }
        except RuntimeError as error:
            if "out of memory" in str(error).lower():
                torch.cuda.empty_cache()
                result["matrices"][name] = {
                    "status": "cuda_out_of_memory",
                    "seconds": time.time() - tick,
                    "error": str(error),
                }
                logging.exception("Joint %s inference ran out of memory", name)
            else:
                raise
    result["inference_seconds_total"] = time.time() - started
    result["peak_cuda_memory_gib"] = (
        torch.cuda.max_memory_allocated() / 1024**3
    )
    result["environment"] = common.environment()
    common.write_json(args.output_dir / "joint_all_layers_result.json", result)
    (args.output_dir / "SUMMARY.md").write_text(
        make_summary(result), encoding="utf-8"
    )
    logging.info("Joint all-layer run finished")


if __name__ == "__main__":
    main()
