"""Compare two four-layer Stage 1 settings on controlled-current32 v1.

The input passed static validation but is not behaviorally validated or frozen,
so every output from this runner is explicitly exploratory.
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

ROOT = Path(__file__).resolve().parents[2]
TASK3 = ROOT / "task3_v1"
SCRIPTS = TASK3 / "scripts"
sys.path[:0] = [str(TASK3), str(SCRIPTS)]

import run_official90_stage1_exploratory as common  # noqa: E402
from build_discovery_matrix import CONCEPTS  # noqa: E402
from paper_aligned_core import load_config, sha256_file  # noqa: E402
from run_paper_aligned_jspace import load_components  # noqa: E402

LAYER_SETS = {
    "workspace_23_25_26_28": [23, 25, 26, 28],
    "legacy_4_8_16_30": [4, 8, 16, 30],
}


def load_prompts(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 1000:
        raise ValueError(f"Expected 1000 prompts, got {len(rows)}")
    required = {"id", "prompt", "primary_concept", "condition", "fold"}
    for index, row in enumerate(rows):
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"Row {index} is missing {missing}")
        if not row["prompt"].endswith("Answer:"):
            raise ValueError(f"Row {index} lacks the Answer: anchor")
    if len({row["id"] for row in rows}) != 1000:
        raise ValueError("Prompt IDs are not unique")
    if sorted({int(row["fold"]) for row in rows}) != list(range(5)):
        raise ValueError("Expected folds 0..4")
    return rows


@torch.no_grad()
def extract_features(
    model,
    tokenizer,
    lens,
    prompts: list[dict[str, Any]],
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    from jlens.hooks import ActivationRecorder
    from torch.nn import functional

    layers = sorted({layer for values in LAYER_SETS.values() for layer in values})
    available = sorted(map(int, lens.jacobians))
    missing = sorted(set(layers) - set(available))
    if missing:
        raise ValueError(f"Missing fitted layers {missing}; available={available}")
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
    records = []
    timings = []
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
        records.append(
            {
                "row_index": index,
                "example_id": item["id"],
                "fold": int(item["fold"]),
                "condition": item["condition"],
                "primary_concept": item["primary_concept"],
                "secondary_concept": item.get("secondary_concept"),
                "template_family": item.get("template_family"),
                "token_count": int(input_ids.shape[-1]),
                "measurement_position": -1,
            }
        )
        if (index + 1) % 50 == 0:
            logging.info("Projected %d/%d prompts", index + 1, len(prompts))

    arrays = {
        layer: np.asarray(layer_values, dtype=np.float32)
        for layer, layer_values in values.items()
    }
    matrices = {
        name: np.concatenate([arrays[layer] for layer in layer_set], axis=1)
        for name, layer_set in LAYER_SETS.items()
    }
    for name, matrix in matrices.items():
        if matrix.shape != (1000, 128):
            raise RuntimeError(f"{name}: expected (1000, 128), got {matrix.shape}")
    extraction = {
        "all_layers_captured_once": layers,
        "available_lens_layer_min": available[0],
        "available_lens_layer_max": available[-1],
        "prompt_seconds_mean": float(np.mean(timings)),
        "prompt_seconds_total": float(np.sum(timings)),
        "prompt_token_count_min": min(row["token_count"] for row in records),
        "prompt_token_count_median": float(
            np.median([row["token_count"] for row in records])
        ),
        "prompt_token_count_max": max(row["token_count"] for row in records),
    }
    return matrices, records, extraction


def make_summary(result: dict[str, Any]) -> str:
    lines = [
        "# Controlled-current32 candidate Stage 1",
        "",
        "Exploratory only: static validation passed, but behavioral validation "
        "and dataset freezing have not been completed.",
        "",
        "Layer 30 substitutes for the unavailable fitted layer 32.",
        "",
        "| Layer set | Matrix | CauScale AUPRC | Abs-corr AUPRC | Stable | "
        "Same | Cross | Median Jaccard |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for layer_name, layer_result in result["layer_sets"].items():
        for matrix_name in ("raw", "innovation"):
            inference = layer_result[matrix_name]["inference"]
            bootstrap = layer_result[matrix_name]["bootstrap"]
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
            "Same-concept AUPRC is a positive-control diagnostic. Cross-concept "
            "edges are candidates only until held-out intervention validation.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=TASK3 / "configs" / "paper_aligned_jspace.yaml",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=TASK3
        / "data/prompts/controlled_current32/v1/prompts.jsonl",
    )
    parser.add_argument(
        "--validation-report",
        type=Path,
        default=TASK3
        / "data/prompts/controlled_current32/v1/validation_report.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=TASK3 / "outputs/stage1_controlled1000_layer_sets",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=TASK3 / "logs/stage1_controlled1000_layer_sets",
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
    for path in (args.config, args.prompts, args.validation_report):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    validation = json.loads(args.validation_report.read_text(encoding="utf-8"))
    if not validation.get("hard_checks_passed"):
        raise RuntimeError("Static hard checks did not pass")

    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    prompts = load_prompts(args.prompts)
    config = load_config(args.config)
    logging.info("Loading Qwen and the frozen n1000 J-lens")
    model, tokenizer, lens, loading = load_components(config, ROOT)
    matrices, prompt_records, extraction = extract_features(
        model, tokenizer, lens, prompts
    )
    token_ids = [
        int(tokenizer.encode(concept, add_special_tokens=False)[0])
        for concept in CONCEPTS
    ]
    del model, tokenizer, lens
    torch.cuda.empty_cache()

    result: dict[str, Any] = {
        "status": "candidate_static_validated_not_behaviorally_frozen",
        "prompt_count": len(prompts),
        "prompt_file": str(args.prompts),
        "prompt_sha256": sha256_file(args.prompts),
        "validation_status": validation.get("status"),
        "behavioral_validation_performed": validation.get(
            "behavioral_validation_performed", False
        ),
        "concept_count": len(CONCEPTS),
        "node_count": 128,
        "sample_to_node_ratio": len(prompts) / 128,
        "layer32_substitution": {"requested": 32, "used": 30},
        "loading": loading,
        "extraction": extraction,
        "layer_sets": {},
        "limitations": [
            "candidate prompts are not behaviorally validated or frozen",
            "no independent held-out intervention validation",
            "released CauScale checkpoint is used zero-shot",
        ],
    }
    groups = np.asarray([str(row["fold"]) for row in prompt_records], dtype=str)
    prompt_ids = np.asarray(
        [row["example_id"] for row in prompt_records], dtype=str
    )

    sys.path.insert(0, str(ROOT.parent / ".deps/CauScale-main/src"))
    logging.info("Loading released CauScale checkpoint")
    causcale, causcale_src, checkpoint = common.load_causcale()
    result["causcale_source"] = str(causcale_src)
    result["causcale_checkpoint"] = str(checkpoint)
    result["causcale_checkpoint_sha256"] = sha256_file(checkpoint)

    for layer_name, layers in LAYER_SETS.items():
        logging.info("Running %s", layer_name)
        output = args.output_dir / layer_name
        output.mkdir(parents=True, exist_ok=True)
        raw = matrices[layer_name]
        innovation, audit = common.innovation_residualize(raw, groups)
        columns = common.columns_for(layers, token_ids)
        np.save(output / "raw_matrix_1000x128.npy", raw)
        np.save(output / "innovation_matrix_1000x128.npy", innovation)
        np.savez_compressed(
            output / "matrix_rows_and_groups.npz",
            prompt_ids=prompt_ids,
            group_ids=groups,
        )
        common.write_json(
            output / "matrix_metadata.json",
            {
                "status": result["status"],
                "shape": list(raw.shape),
                "layers": layers,
                "concepts": CONCEPTS,
                "columns": columns,
                "prompt_records": prompt_records,
                "raw_matrix_sha256_memory": common.sha256_array(raw),
                "innovation_matrix_sha256_memory": common.sha256_array(innovation),
                "raw_zero_variance_columns": int(
                    np.sum(raw.std(axis=0, ddof=1) < 1e-8)
                ),
                "innovation_zero_variance_columns": int(
                    np.sum(innovation.std(axis=0, ddof=1) < 1e-8)
                ),
                "innovation_audit": audit,
            },
        )
        result["layer_sets"][layer_name] = {"layers": layers}
        for matrix_name, matrix in (("raw", raw), ("innovation", innovation)):
            logging.info("%s / %s inference", layer_name, matrix_name)
            inference, scores = common.infer_causcale(causcale, matrix, columns)
            logging.info("%s / %s bootstrap", layer_name, matrix_name)
            bootstrap = common.bootstrap_causcale(
                causcale, matrix, columns, runs=args.bootstrap_runs
            )
            payload = {
                "status": "candidate_dependencies_not_causal_edges",
                "layer_set": layer_name,
                "layers": layers,
                "matrix": matrix_name,
                "inference": inference,
                "bootstrap": bootstrap,
            }
            result["layer_sets"][layer_name][matrix_name] = payload
            common.write_json(
                output / f"{matrix_name}_stage1_result.json", payload
            )
            np.save(output / f"{matrix_name}_directed_probabilities.npy", scores)

    result["total_seconds"] = time.time() - started
    result["peak_cuda_memory_gib"] = (
        torch.cuda.max_memory_allocated() / 1024**3
    )
    result["environment"] = common.environment()
    common.write_json(args.output_dir / "comparison.json", result)
    (args.output_dir / "SUMMARY.md").write_text(
        make_summary(result), encoding="utf-8"
    )
    logging.info("Completed in %.1f seconds", result["total_seconds"])


if __name__ == "__main__":
    main()
