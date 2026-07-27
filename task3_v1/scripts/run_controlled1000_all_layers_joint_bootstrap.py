"""Sample-bootstrap the 992-node joint all-layer CauScale graph."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.covariance import LedoitWolf

ROOT = Path(__file__).resolve().parents[2]
TASK3 = ROOT / "task3_v1"
SCRIPTS = TASK3 / "scripts"
sys.path[:0] = [str(TASK3), str(SCRIPTS)]

import run_official90_stage1_exploratory as common  # noqa: E402
from build_discovery_matrix import CONCEPTS  # noqa: E402
from run_causcale_smoke import directed_probabilities  # noqa: E402


@torch.no_grad()
def bootstrap_joint(
    causcale,
    matrix: np.ndarray,
    columns: list[dict[str, Any]],
    *,
    runs: int,
) -> dict[str, Any]:
    node_count = matrix.shape[1]
    allowed, same = common.allowed_and_same(columns)
    candidates = allowed & ~same
    availability = np.zeros((node_count, node_count), dtype=np.int16)
    selected_count = np.zeros((node_count, node_count), dtype=np.int16)
    probability_sum = np.zeros((node_count, node_count), dtype=np.float32)
    edge_sets: list[set[int]] = []
    records = []
    for run in range(runs):
        rng = np.random.RandomState(20260723 + run)
        rows = rng.choice(len(matrix), size=len(matrix), replace=True)
        sampled = matrix[rows]
        prior = LedoitWolf().fit(sampled).get_precision().astype(np.float32)
        mean = sampled.mean(axis=0, keepdims=True)
        std = sampled.std(axis=0, keepdims=True)
        standardized = (sampled - mean) / np.where(std < 1e-8, 1.0, std)
        batch = {
            "data": torch.from_numpy(standardized).unsqueeze(0).cuda(),
            "interv": torch.zeros(
                1, len(sampled), node_count, device="cuda"
            ),
            "feats": torch.from_numpy(prior).unsqueeze(0).cuda(),
        }
        encoded = causcale.encoder(batch)
        directed, _ = directed_probabilities(causcale, encoded, node_count)
        scores = directed.cpu().numpy()
        selected = candidates & (scores >= 0.5)
        availability[candidates] += 1
        selected_count[selected] += 1
        probability_sum[candidates] += scores[candidates]
        selected_flat = set(np.flatnonzero(selected).astype(int).tolist())
        edge_sets.append(selected_flat)
        records.append(
            {
                "run": run,
                "seed": 20260723 + run,
                "selected_cross_concept_edge_count": len(selected_flat),
            }
        )

    frequency = selected_count.astype(np.float32) / runs
    mean_probability = probability_sum / runs
    stable = (
        candidates
        & (frequency >= 0.8)
        & (mean_probability >= 0.5)
    )
    stable_indices = np.argwhere(stable)
    order = sorted(
        range(len(stable_indices)),
        key=lambda index: (
            frequency[tuple(stable_indices[index])],
            mean_probability[tuple(stable_indices[index])],
        ),
        reverse=True,
    )
    top = []
    for rank, order_index in enumerate(order[:200], start=1):
        source, target = stable_indices[order_index]
        top.append(
            {
                "rank": rank,
                "source_index": int(source),
                "target_index": int(target),
                "source_layer": int(columns[source]["layer"]),
                "target_layer": int(columns[target]["layer"]),
                "source_concept": columns[source]["concept"],
                "target_concept": columns[target]["concept"],
                "selection_frequency": float(frequency[source, target]),
                "mean_probability": float(mean_probability[source, target]),
            }
        )
    jaccards = []
    for left, right in itertools.combinations(range(runs), 2):
        union = edge_sets[left] | edge_sets[right]
        jaccards.append(
            len(edge_sets[left] & edge_sets[right]) / len(union)
            if union
            else 1.0
        )
    return {
        "runs": runs,
        "sample_bootstrap": f"{len(matrix)} rows with replacement",
        "all_992_nodes_retained_each_run": True,
        "same_concept_edges_excluded": True,
        "allowed_cross_concept_edge_count": int(candidates.sum()),
        "stable_cross_concept_edge_count": int(stable.sum()),
        "stable_cross_concept_edge_fraction": float(
            stable.sum() / candidates.sum()
        ),
        "pairwise_cross_edge_jaccard_min": float(np.min(jaccards)),
        "pairwise_cross_edge_jaccard_median": float(np.median(jaccards)),
        "pairwise_cross_edge_jaccard_max": float(np.max(jaccards)),
        "run_records": records,
        "top_stable_cross_concept_edges": top,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--joint-dir",
        type=Path,
        default=TASK3 / "outputs/stage1_controlled1000_all_layers_joint",
    )
    parser.add_argument("--runs", type=int, default=20)
    args = parser.parse_args()
    raw = np.load(args.joint_dir / "raw_matrix_1000x992.npy")
    innovation = np.load(args.joint_dir / "innovation_matrix_1000x992.npy")
    result = json.loads(
        (args.joint_dir / "joint_all_layers_result.json").read_text(
            encoding="utf-8"
        )
    )
    layers = result["layers"]
    columns = common.columns_for(layers, list(range(len(CONCEPTS))))
    sys.path.insert(0, str(ROOT.parent / ".deps/CauScale-main/src"))
    causcale, causcale_src, checkpoint = common.load_causcale()
    payload = {
        "status": "exploratory_joint_992_node_sample_bootstrap",
        "layers": layers,
        "node_count": raw.shape[1],
        "causcale_source": str(causcale_src),
        "causcale_checkpoint": str(checkpoint),
        "checkpoint_regime_warning": (
            "992 nodes exceed the approximately 500-node demonstrated regime"
        ),
        "matrices": {},
    }
    for name, matrix in (("raw", raw), ("innovation", innovation)):
        payload["matrices"][name] = bootstrap_joint(
            causcale, matrix, columns, runs=args.runs
        )
    common.write_json(
        args.joint_dir / "joint_all_layers_bootstrap.json", payload
    )


if __name__ == "__main__":
    main()
