"""Confirm the all-layer scan's selected 18/19/24/25 band as one graph."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TASK3 = ROOT / "task3_v1"
SCRIPTS = TASK3 / "scripts"
sys.path[:0] = [str(TASK3), str(SCRIPTS)]

import run_official90_stage1_exploratory as common  # noqa: E402
from build_discovery_matrix import CONCEPTS  # noqa: E402
from run_controlled1000_stage1_exploratory import load_prompts  # noqa: E402

LAYERS = [18, 19, 24, 25]


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
        default=TASK3 / "outputs/stage1_controlled1000_selected_band",
    )
    parser.add_argument("--bootstrap-runs", type=int, default=20)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    prompts = load_prompts(args.prompts)
    groups = np.asarray([str(row["fold"]) for row in prompts], dtype=str)
    with np.load(args.features) as payload:
        matrix = np.concatenate(
            [payload[f"layer_{layer}"].astype(np.float32) for layer in LAYERS],
            axis=1,
        )
    if matrix.shape != (1000, 128):
        raise RuntimeError(f"Expected (1000, 128), got {matrix.shape}")
    innovation, audit = common.innovation_residualize(matrix, groups)

    sys.path.insert(0, str(ROOT.parent / ".deps/CauScale-main/src"))
    causcale, causcale_src, checkpoint = common.load_causcale()
    token_ids = list(range(len(CONCEPTS)))
    columns = common.columns_for(LAYERS, token_ids)
    result = {
        "status": "data_selected_exploratory_band_not_causal_graph",
        "selection_source": (
            "top late-band pairs from the same controlled1000 all-layer scan"
        ),
        "selection_bias_warning": True,
        "layers": LAYERS,
        "shape": list(matrix.shape),
        "same_concept_edges_excluded_from_final_candidate_interpretation": True,
        "causcale_source": str(causcale_src),
        "causcale_checkpoint": str(checkpoint),
        "innovation_audit": audit,
        "matrices": {},
    }
    for name, values in (("raw", matrix), ("innovation", innovation)):
        inference, scores = common.infer_causcale(causcale, values, columns)
        bootstrap = common.bootstrap_causcale(
            causcale, values, columns, runs=args.bootstrap_runs
        )
        result["matrices"][name] = {
            "inference": inference,
            "bootstrap": bootstrap,
        }
        np.save(args.output_dir / f"{name}_matrix_1000x128.npy", values)
        np.save(
            args.output_dir / f"{name}_directed_probabilities.npy", scores
        )
    result["total_seconds"] = time.time() - started
    result["peak_cuda_memory_gib"] = (
        torch.cuda.max_memory_allocated() / 1024**3
    )
    result["environment"] = common.environment()
    common.write_json(args.output_dir / "selected_band_result.json", result)

    lines = [
        "# Selected 18/19/24/25 band follow-up",
        "",
        "This layer set was selected using the same data, so the result is an "
        "exploratory confirmation and not an independent replication.",
        "",
        "| Matrix | Stable | Same | Cross | Median Jaccard |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("raw", "innovation"):
        bootstrap = result["matrices"][name]["bootstrap"]
        lines.append(
            f"| {name} | {bootstrap['stable_edge_count']} | "
            f"{bootstrap['same_concept_stable_edge_count']} | "
            f"{bootstrap['cross_concept_stable_edge_count']} | "
            f"{bootstrap['pairwise_edge_set_jaccard_median']:.3f} |"
        )
    (args.output_dir / "SUMMARY.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
