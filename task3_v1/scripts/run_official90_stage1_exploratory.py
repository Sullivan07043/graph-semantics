"""Exploratory Stage 1 comparison on the 90 official Anthropic prompts.

This deliberately does not modify or replace the frozen paper-aligned
diagnostic.  It projects the current 32 Task 3 concepts at two four-layer
choices, applies the existing innovation residualization recipe, and runs the
released CauScale checkpoint plus the existing 20-run bootstrap criterion.

The result is a pipeline diagnostic only: 90 prompts for 128 nodes is
underdetermined, the prompts are not controlled-current32 data, and there is
no independent intervention validation for the discovered edges.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.covariance import LedoitWolf
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
TASK3 = ROOT / "task3_v1"
SCRIPTS = TASK3 / "scripts"
sys.path.insert(0, str(TASK3))
sys.path.insert(0, str(SCRIPTS))

from build_discovery_matrix import CONCEPTS  # noqa: E402
from build_innovation_matrix import fit_ridge  # noqa: E402
from paper_aligned_core import load_config, sha256_file  # noqa: E402
from run_causcale_smoke import directed_probabilities, model_args  # noqa: E402
from run_paper_aligned_jspace import load_components  # noqa: E402


LAYER_SETS = {
    "workspace_23_25_26_28": [23, 25, 26, 28],
    # The requested layer 32 is not fitted by the current n1000 J-lens.
    # Layer 30 is the highest available fitted layer and is used explicitly as
    # a documented substitution rather than silently pretending 32 exists.
    "legacy_4_8_16_30": [4, 8, 16, 30],
}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
    }


def load_official_prompts(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["items"]
    if len(rows) != 90:
        raise ValueError(f"Expected 90 official prompts, got {len(rows)}")
    required = {"name", "category", "prompt"}
    for index, row in enumerate(rows):
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"Official row {index} is missing {missing}")
    return rows


@torch.no_grad()
def extract_feature_matrices(
    model,
    tokenizer,
    lens,
    prompts: list[dict[str, Any]],
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    from jlens.hooks import ActivationRecorder
    from torch.nn import functional

    all_layers = sorted({layer for layers in LAYER_SETS.values() for layer in layers})
    available_layers = sorted(map(int, lens.jacobians))
    missing = sorted(set(all_layers) - set(available_layers))
    if missing:
        raise ValueError(
            f"Requested layers {missing} are absent; available range is "
            f"{available_layers[0]}..{available_layers[-1]}"
        )

    token_ids: list[int] = []
    for concept in CONCEPTS:
        ids = tokenizer.encode(concept, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Current concept is not one token: {concept!r} -> {ids}")
        token_ids.append(int(ids[0]))

    vectors: dict[int, torch.Tensor] = {}
    for layer in all_layers:
        jacobian = lens.jacobians[layer].to(device="cuda", dtype=torch.float32)
        raw = jacobian.T @ model._lm_head.weight[token_ids].float().T
        vectors[layer] = functional.normalize(raw, dim=0)

    rows_by_layer: dict[int, list[list[float]]] = {
        layer: [] for layer in all_layers
    }
    prompt_records: list[dict[str, Any]] = []
    timings: list[float] = []
    for index, item in enumerate(prompts):
        tick = time.time()
        input_ids = model.encode(item["prompt"])
        with ActivationRecorder(model.layers, at=all_layers) as recorder:
            model.forward(input_ids)
            for layer in all_layers:
                residual = recorder.activations[layer][0, -1].detach().float()
                rows_by_layer[layer].append(
                    (residual @ vectors[layer]).cpu().tolist()
                )
        torch.cuda.synchronize()
        timings.append(time.time() - tick)
        prompt_records.append(
            {
                "row_index": index,
                "example_id": item["name"],
                "category": item["category"],
                "prompt": item["prompt"],
                "token_count": int(input_ids.shape[-1]),
                "measurement_position": -1,
            }
        )
        if (index + 1) % 10 == 0:
            logging.info("Projected %d/%d official prompts", index + 1, len(prompts))

    matrices: dict[str, np.ndarray] = {}
    for name, layers in LAYER_SETS.items():
        matrices[name] = np.concatenate(
            [
                np.asarray(rows_by_layer[layer], dtype=np.float32)
                for layer in layers
            ],
            axis=1,
        )
        if matrices[name].shape != (90, 128):
            raise RuntimeError(
                f"{name}: expected (90, 128), got {matrices[name].shape}"
            )

    extraction = {
        "all_layers_captured_once": all_layers,
        "available_lens_layer_min": available_layers[0],
        "available_lens_layer_max": available_layers[-1],
        "prompt_seconds_mean": float(np.mean(timings)),
        "prompt_seconds_total": float(np.sum(timings)),
        "prompt_token_count_min": min(row["token_count"] for row in prompt_records),
        "prompt_token_count_median": float(
            np.median([row["token_count"] for row in prompt_records])
        ),
        "prompt_token_count_max": max(row["token_count"] for row in prompt_records),
    }
    return matrices, prompt_records, extraction


def columns_for(layers: list[int], token_ids: list[int]) -> list[dict[str, Any]]:
    return [
        {
            "index": index,
            "layer": int(layer),
            "concept": concept,
            "token_id": int(token_id),
        }
        for index, (layer, concept, token_id) in enumerate(
            (layer, concept, token_id)
            for layer in layers
            for concept, token_id in zip(CONCEPTS, token_ids)
        )
    ]


def innovation_residualize(
    matrix: np.ndarray,
    groups: np.ndarray,
    *,
    alpha: float = 1.0,
    folds: int = 5,
) -> tuple[np.ndarray, dict[str, Any]]:
    layers_count = 4
    concepts_count = len(CONCEPTS)
    innovation = matrix.astype(np.float64).copy()
    unique_groups = np.unique(groups)
    folds = min(folds, len(unique_groups))
    if folds < 2:
        raise ValueError("At least two prompt groups are required")
    splitter = GroupKFold(n_splits=folds)
    audits = []
    for concept_index, concept in enumerate(CONCEPTS):
        for layer_index in range(1, layers_count):
            target_index = layer_index * concepts_count + concept_index
            predictor_indices = np.asarray(
                [
                    earlier * concepts_count + concept_index
                    for earlier in range(layer_index)
                ],
                dtype=np.int64,
            )
            predictors = matrix[:, predictor_indices].astype(np.float64)
            target = matrix[:, target_index].astype(np.float64)
            oof = np.full(len(matrix), np.nan, dtype=np.float64)
            for train, test in splitter.split(predictors, target, groups=groups):
                coefficient, intercept = fit_ridge(
                    predictors[train], target[train], alpha
                )
                oof[test] = predictors[test] @ coefficient + intercept
            coefficient, intercept = fit_ridge(predictors, target, alpha)
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
                    "layer_position": layer_index,
                    "predictor_count": int(len(predictor_indices)),
                    "oof_r2": oof_r2,
                }
            )
    result = innovation.astype(np.float32)
    return result, {
        "method": "same-concept earlier-layer ridge residualization",
        "ridge_alpha": alpha,
        "group_folds": folds,
        "group_count": int(len(unique_groups)),
        "oof_r2_min": float(min(row["oof_r2"] for row in audits)),
        "oof_r2_median": float(np.median([row["oof_r2"] for row in audits])),
        "oof_r2_max": float(max(row["oof_r2"] for row in audits)),
        "audits": audits,
    }


def load_causcale():
    causcale_src = ROOT.parent / ".deps" / "CauScale-main"
    checkpoint_path = (
        causcale_src
        / "checkpoints"
        / "synthetic"
        / "auprc=0.905_migrated.ckpt"
    )
    if not causcale_src.is_dir() or not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"CauScale source/checkpoint missing: {causcale_src}, {checkpoint_path}"
        )
    sys.path.insert(0, str(causcale_src))
    from model import CauScale

    model = CauScale(model_args())
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.eval().cuda(), causcale_src, checkpoint_path


def allowed_and_same(columns: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    layers = np.asarray([column["layer"] for column in columns])
    concepts = np.asarray([column["concept"] for column in columns])
    allowed = layers[:, None] < layers[None, :]
    same = allowed & (concepts[:, None] == concepts[None, :])
    return allowed, same


@torch.no_grad()
def infer_causcale(
    causcale,
    matrix: np.ndarray,
    columns: list[dict[str, Any]],
) -> tuple[dict[str, Any], np.ndarray]:
    prior = LedoitWolf().fit(matrix).get_precision().astype(np.float32)
    mean = matrix.mean(axis=0, keepdims=True)
    std = matrix.std(axis=0, keepdims=True)
    standardized = (matrix - mean) / np.where(std < 1e-8, 1.0, std)
    batch = {
        "data": torch.from_numpy(standardized).unsqueeze(0).cuda(),
        "interv": torch.zeros(
            1, len(matrix), matrix.shape[1], device="cuda"
        ),
        "feats": torch.from_numpy(prior).unsqueeze(0).cuda(),
    }
    encoded = causcale.encoder(batch)
    directed, _ = directed_probabilities(causcale, encoded, matrix.shape[1])
    scores = directed.cpu().numpy()
    allowed, same = allowed_and_same(columns)
    labels = same[allowed].astype(np.int32)
    causcale_auprc = float(average_precision_score(labels, scores[allowed]))
    correlations = np.abs(np.corrcoef(matrix, rowvar=False))
    correlations = np.nan_to_num(correlations, nan=0.0)
    corr_auprc = float(
        average_precision_score(labels, correlations[allowed])
    )
    selected = allowed & (scores >= 0.5)
    selected_indices = np.argwhere(selected)
    order = sorted(
        range(len(selected_indices)),
        key=lambda i: scores[tuple(selected_indices[i])],
        reverse=True,
    )
    top_edges = []
    for rank, order_index in enumerate(order[:30], start=1):
        source, target = selected_indices[order_index]
        top_edges.append(
            {
                "rank": rank,
                "source_layer": int(columns[source]["layer"]),
                "target_layer": int(columns[target]["layer"]),
                "source_concept": columns[source]["concept"],
                "target_concept": columns[target]["concept"],
                "same_concept": bool(
                    columns[source]["concept"] == columns[target]["concept"]
                ),
                "probability": float(scores[source, target]),
            }
        )
    return {
        "sample_count": int(len(matrix)),
        "node_count": int(matrix.shape[1]),
        "sample_to_node_ratio": float(len(matrix) / matrix.shape[1]),
        "same_concept_positive_count": int(labels.sum()),
        "allowed_edge_count": int(allowed.sum()),
        "causcale_same_concept_auprc": causcale_auprc,
        "absolute_correlation_same_concept_auprc": corr_auprc,
        "allowed_probability_median": float(np.median(scores[allowed])),
        "allowed_probability_p95": float(np.percentile(scores[allowed], 95)),
        "allowed_probability_max": float(np.max(scores[allowed])),
        "selected_edges_ge_0_5": int(selected.sum()),
        "selected_same_concept_edges": int((selected & same).sum()),
        "selected_cross_concept_edges": int((selected & ~same).sum()),
        "top_selected_edges": top_edges,
    }, scores


@torch.no_grad()
def bootstrap_causcale(
    causcale,
    matrix: np.ndarray,
    columns: list[dict[str, Any]],
    *,
    runs: int,
    threshold: float = 0.5,
    concept_fraction: float = 0.8,
) -> dict[str, Any]:
    n = matrix.shape[1]
    concept_count = len(CONCEPTS)
    concepts_per_run = round(concept_count * concept_fraction)
    availability = np.zeros((n, n), dtype=np.int32)
    selected_count = np.zeros((n, n), dtype=np.int32)
    probability_sum = np.zeros((n, n), dtype=np.float64)
    probability_sq_sum = np.zeros((n, n), dtype=np.float64)
    edge_sets: list[set[int]] = []
    available_edge_sets: list[set[int]] = []
    run_records = []

    for run in range(runs):
        seed = 20260723 + run
        rng = np.random.RandomState(seed)
        chosen = np.sort(
            rng.choice(concept_count, size=concepts_per_run, replace=False)
        )
        nodes = np.asarray(
            [
                layer_index * concept_count + concept_index
                for layer_index in range(4)
                for concept_index in chosen
            ],
            dtype=np.int64,
        )
        row_indices = rng.choice(len(matrix), size=len(matrix), replace=True)
        sampled = matrix[row_indices][:, nodes]
        prior = LedoitWolf().fit(sampled).get_precision().astype(np.float32)
        mean = sampled.mean(axis=0, keepdims=True)
        std = sampled.std(axis=0, keepdims=True)
        standardized = (sampled - mean) / np.where(std < 1e-8, 1.0, std)
        local_layers = np.asarray([columns[index]["layer"] for index in nodes])
        local_allowed = local_layers[:, None] < local_layers[None, :]
        batch = {
            "data": torch.from_numpy(standardized).unsqueeze(0).cuda(),
            "interv": torch.zeros(
                1, len(sampled), len(nodes), device="cuda"
            ),
            "feats": torch.from_numpy(prior).unsqueeze(0).cuda(),
        }
        encoded = causcale.encoder(batch)
        directed, _ = directed_probabilities(causcale, encoded, len(nodes))
        scores = directed.cpu().numpy()
        global_rows, global_cols = np.meshgrid(nodes, nodes, indexing="ij")
        global_rows = global_rows[local_allowed]
        global_cols = global_cols[local_allowed]
        values = scores[local_allowed]
        availability[global_rows, global_cols] += 1
        selected_count[global_rows, global_cols] += (
            values >= threshold
        ).astype(np.int32)
        probability_sum[global_rows, global_cols] += values
        probability_sq_sum[global_rows, global_cols] += values**2
        selected_flat = set(
            (
                global_rows[values >= threshold] * n
                + global_cols[values >= threshold]
            ).tolist()
        )
        available_flat = set((global_rows * n + global_cols).tolist())
        edge_sets.append(selected_flat)
        available_edge_sets.append(available_flat)
        run_records.append(
            {
                "run": run,
                "seed": seed,
                "selected_edge_count": len(selected_flat),
            }
        )

    observed = availability > 0
    frequency = np.zeros((n, n), dtype=np.float32)
    mean_probability = np.zeros((n, n), dtype=np.float32)
    frequency[observed] = selected_count[observed] / availability[observed]
    mean_probability[observed] = probability_sum[observed] / availability[observed]
    globally_allowed, same = allowed_and_same(columns)
    stable = (
        observed
        & globally_allowed
        & (frequency >= 0.8)
        & (mean_probability >= threshold)
    )
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
                "source_index": int(source),
                "target_index": int(target),
                "source_layer": int(columns[source]["layer"]),
                "target_layer": int(columns[target]["layer"]),
                "source_concept": columns[source]["concept"],
                "target_concept": columns[target]["concept"],
                "same_concept": bool(same[source, target]),
                "availability": int(availability[source, target]),
                "selection_frequency": float(frequency[source, target]),
                "mean_probability": float(mean_probability[source, target]),
            }
        )
    jaccards = []
    for left, right in itertools.combinations(range(runs), 2):
        common = available_edge_sets[left] & available_edge_sets[right]
        left_edges = edge_sets[left] & common
        right_edges = edge_sets[right] & common
        union = left_edges | right_edges
        jaccards.append(
            len(left_edges & right_edges) / len(union) if union else 1.0
        )
    same_count = sum(edge["same_concept"] for edge in stable_edges)
    return {
        "runs": runs,
        "sample_bootstrap": f"{len(matrix)} rows with replacement",
        "feature_bootstrap": (
            f"{concepts_per_run}/32 concept groups; all four layers retained"
        ),
        "edge_probability_threshold": threshold,
        "stable_selection_frequency_threshold": 0.8,
        "stable_edge_count": len(stable_edges),
        "same_concept_stable_edge_count": int(same_count),
        "cross_concept_stable_edge_count": int(len(stable_edges) - same_count),
        "same_concept_stable_edge_fraction": (
            float(same_count / len(stable_edges)) if stable_edges else 0.0
        ),
        "pairwise_edge_set_jaccard_min": float(np.min(jaccards)),
        "pairwise_edge_set_jaccard_median": float(np.median(jaccards)),
        "pairwise_edge_set_jaccard_max": float(np.max(jaccards)),
        "run_records": run_records,
        "stable_edges": stable_edges,
    }


def summary_markdown(comparison: dict[str, Any]) -> str:
    lines = [
        "# Official-90 exploratory Stage 1 comparison",
        "",
        "This is a pipeline diagnostic, not a causal graph result. The experiment "
        "uses 90 prompts for 128 nodes and has no independent edge-level "
        "intervention validation.",
        "",
        "The requested legacy layer 32 is absent from the current fitted lens; "
        "layer 30 was substituted and recorded explicitly.",
        "",
        "| Layer set | Matrix | CauScale AUPRC | Abs-corr AUPRC | Stable | "
        "Same | Cross | Median Jaccard |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for layer_name, layer_result in comparison["layer_sets"].items():
        for matrix_name in ("raw", "innovation"):
            result = layer_result[matrix_name]
            inference = result["inference"]
            bootstrap = result["bootstrap"]
            lines.append(
                f"| {layer_name} | {matrix_name} | "
                f"{inference['causcale_same_concept_auprc']:.3f} | "
                f"{inference['absolute_correlation_same_concept_auprc']:.3f} | "
                f"{bootstrap['stable_edge_count']} | "
                f"{bootstrap['same_concept_stable_edge_count']} | "
                f"{bootstrap['cross_concept_stable_edge_count']} | "
                f"{bootstrap['pairwise_edge_set_jaccard_median']:.3f} |"
            )
    lines.extend(
        [
            "",
            "Interpretation must remain exploratory because N/P = 90/128 = "
            "0.703, the official prompt set contains duplicate/template "
            "structure, and the 32 current concepts were not used to construct "
            "these prompts.",
            "",
        ]
    )
    return "\n".join(lines)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=TASK3 / "configs" / "paper_aligned_jspace.yaml",
    )
    parser.add_argument(
        "--official-prompts",
        type=Path,
        default=(
            TASK3
            / "data"
            / "prompts"
            / "official_anthropic"
            / "probe-swap.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=TASK3 / "outputs" / "stage1_official90_exploratory",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=TASK3 / "logs" / "stage1_official90_exploratory",
    )
    parser.add_argument("--bootstrap-runs", type=int, default=20)
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
    for path in (args.config, args.official_prompts):
        if not path.is_file():
            raise FileNotFoundError(path)

    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    config = load_config(args.config)
    prompts = load_official_prompts(args.official_prompts)
    logging.info("Loading Qwen and the frozen n1000 J-lens")
    model, tokenizer, lens, loading = load_components(config, ROOT)
    matrices, prompt_records, extraction = extract_feature_matrices(
        model, tokenizer, lens, prompts
    )
    token_ids = [
        int(tokenizer.encode(concept, add_special_tokens=False)[0])
        for concept in CONCEPTS
    ]
    del model, tokenizer, lens
    torch.cuda.empty_cache()

    groups = np.asarray([row["category"] for row in prompt_records], dtype=str)
    prompt_ids = np.asarray(
        [row["example_id"] for row in prompt_records], dtype=str
    )
    comparison: dict[str, Any] = {
        "status": "exploratory_stage1_pipeline_diagnostic_not_causal_graph",
        "official_prompt_count": len(prompts),
        "official_prompt_file": str(args.official_prompts),
        "official_prompt_sha256": sha256_file(args.official_prompts),
        "layer32_substitution": {
            "requested": 32,
            "used": 30,
            "reason": "current n1000 J-lens fitted layers end at 30",
        },
        "concept_count": len(CONCEPTS),
        "node_count": 128,
        "sample_to_node_ratio": 90 / 128,
        "loading": loading,
        "extraction": extraction,
        "layer_sets": {},
        "limitations": [
            "90 samples for 128 nodes",
            "official probe-swap prompts are not controlled-current32 prompts",
            "known duplicate, near-duplicate, and entity-overlap structure",
            "no independent held-out interventions for discovered 32-concept edges",
            "CauScale checkpoint is zero-shot out of distribution",
        ],
    }

    logging.info("Loading released CauScale checkpoint")
    causcale, causcale_src, checkpoint_path = load_causcale()
    comparison["causcale_source"] = str(causcale_src)
    comparison["causcale_checkpoint"] = str(checkpoint_path)
    comparison["causcale_checkpoint_sha256"] = sha256_file(checkpoint_path)

    for layer_name, layers in LAYER_SETS.items():
        logging.info("Running exploratory Stage 1 for %s", layer_name)
        layer_dir = args.output_dir / layer_name
        layer_dir.mkdir(parents=True, exist_ok=True)
        raw = matrices[layer_name]
        innovation, innovation_audit = innovation_residualize(raw, groups)
        columns = columns_for(layers, token_ids)
        np.save(layer_dir / "raw_matrix_90x128.npy", raw)
        np.save(layer_dir / "innovation_matrix_90x128.npy", innovation)
        np.savez_compressed(
            layer_dir / "matrix_rows_and_groups.npz",
            prompt_ids=prompt_ids,
            group_ids=groups,
        )
        metadata = {
            "status": "exploratory_matrix_not_causal_graph",
            "shape": list(raw.shape),
            "layers": layers,
            "concepts": CONCEPTS,
            "columns": columns,
            "prompt_records": prompt_records,
            "raw_matrix_sha256_memory": sha256_array(raw),
            "innovation_matrix_sha256_memory": sha256_array(innovation),
            "raw_zero_variance_columns": int(
                np.sum(raw.std(axis=0, ddof=1) < 1e-8)
            ),
            "innovation_zero_variance_columns": int(
                np.sum(innovation.std(axis=0, ddof=1) < 1e-8)
            ),
            "innovation_audit": innovation_audit,
        }
        write_json(layer_dir / "matrix_metadata.json", metadata)
        comparison["layer_sets"][layer_name] = {
            "layers": layers,
            "raw": {},
            "innovation": {},
        }
        for matrix_name, matrix in (
            ("raw", raw),
            ("innovation", innovation),
        ):
            logging.info("%s / %s: CauScale inference", layer_name, matrix_name)
            inference, scores = infer_causcale(causcale, matrix, columns)
            logging.info("%s / %s: bootstrap", layer_name, matrix_name)
            bootstrap = bootstrap_causcale(
                causcale,
                matrix,
                columns,
                runs=args.bootstrap_runs,
            )
            result = {
                "status": "exploratory_candidate_dependencies_not_causal_edges",
                "layer_set": layer_name,
                "layers": layers,
                "matrix": matrix_name,
                "inference": inference,
                "bootstrap": bootstrap,
            }
            comparison["layer_sets"][layer_name][matrix_name] = result
            write_json(layer_dir / f"{matrix_name}_stage1_result.json", result)
            np.save(
                layer_dir / f"{matrix_name}_directed_probabilities.npy",
                scores,
            )

    comparison["total_seconds"] = time.time() - started
    comparison["peak_cuda_memory_gib"] = (
        torch.cuda.max_memory_allocated() / 1024**3
    )
    comparison["environment"] = environment()
    write_json(args.output_dir / "comparison.json", comparison)
    (args.output_dir / "SUMMARY.md").write_text(
        summary_markdown(comparison), encoding="utf-8"
    )
    logging.info("Completed in %.1f seconds", comparison["total_seconds"])
    logging.info("Summary: %s", args.output_dir / "SUMMARY.md")


if __name__ == "__main__":
    main()
