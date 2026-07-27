"""Within-fold layer-row permutation null for the selected 18/19/24/25 band."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

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
        "--observed-result",
        type=Path,
        default=TASK3
        / "outputs/stage1_controlled1000_selected_band"
        / "selected_band_result.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=TASK3
        / "outputs/stage1_controlled1000_selected_band"
        / "within_fold_permutation_null.json",
    )
    parser.add_argument("--permutations", type=int, default=5)
    parser.add_argument("--bootstrap-runs", type=int, default=20)
    args = parser.parse_args()

    prompts = load_prompts(args.prompts)
    groups = np.asarray([str(row["fold"]) for row in prompts], dtype=str)
    with np.load(args.features) as payload:
        arrays = {
            layer: payload[f"layer_{layer}"].astype(np.float32)
            for layer in LAYERS
        }
    observed = json.loads(args.observed_result.read_text(encoding="utf-8"))
    observed_innovation = observed["matrices"]["innovation"]

    sys.path.insert(0, str(ROOT.parent / ".deps/CauScale-main/src"))
    causcale, causcale_src, checkpoint = common.load_causcale()
    columns = common.columns_for(LAYERS, list(range(len(CONCEPTS))))
    records = []
    for permutation in range(args.permutations):
        rng = np.random.RandomState(20260725 + permutation)
        shuffled_layers = []
        for layer in LAYERS:
            shuffled = arrays[layer].copy()
            for fold in np.unique(groups):
                indices = np.flatnonzero(groups == fold)
                shuffled[indices] = arrays[layer][rng.permutation(indices)]
            shuffled_layers.append(shuffled)
        raw = np.concatenate(shuffled_layers, axis=1)
        innovation, _ = common.innovation_residualize(raw, groups)
        inference, _ = common.infer_causcale(causcale, innovation, columns)
        bootstrap = common.bootstrap_causcale(
            causcale,
            innovation,
            columns,
            runs=args.bootstrap_runs,
        )
        records.append(
            {
                "permutation": permutation,
                "seed": 20260725 + permutation,
                "selected_cross_edges_ge_0_5": (
                    inference["selected_cross_concept_edges"]
                ),
                "stable_cross_edge_count": (
                    bootstrap["cross_concept_stable_edge_count"]
                ),
                "median_jaccard": (
                    bootstrap["pairwise_edge_set_jaccard_median"]
                ),
            }
        )
    null_stable = np.asarray(
        [row["stable_cross_edge_count"] for row in records], dtype=float
    )
    null_selected = np.asarray(
        [row["selected_cross_edges_ge_0_5"] for row in records], dtype=float
    )
    result = {
        "status": "within_fold_layer_row_permutation_negative_control",
        "layers": LAYERS,
        "null_hypothesis": (
            "layer marginals and fold composition preserved; cross-layer row "
            "alignment destroyed independently for every layer"
        ),
        "same_concept_edges_excluded_from_reported_counts": True,
        "observed": {
            "selected_cross_edges_ge_0_5": (
                observed_innovation["inference"]["selected_cross_concept_edges"]
            ),
            "stable_cross_edge_count": (
                observed_innovation["bootstrap"]["cross_concept_stable_edge_count"]
            ),
            "median_jaccard": (
                observed_innovation["bootstrap"][
                    "pairwise_edge_set_jaccard_median"
                ]
            ),
        },
        "permutations": records,
        "null_summary": {
            "selected_cross_edges_min": int(null_selected.min()),
            "selected_cross_edges_median": float(np.median(null_selected)),
            "selected_cross_edges_max": int(null_selected.max()),
            "stable_cross_edges_min": int(null_stable.min()),
            "stable_cross_edges_median": float(np.median(null_stable)),
            "stable_cross_edges_max": int(null_stable.max()),
        },
        "causcale_source": str(causcale_src),
        "causcale_checkpoint": str(checkpoint),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    common.write_json(args.output, result)


if __name__ == "__main__":
    main()
