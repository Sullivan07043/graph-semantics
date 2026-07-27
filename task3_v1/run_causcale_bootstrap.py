"""Grouped feature/sample bootstrap for the 128-node CauScale pilot.

The resulting stable edges are intervention candidates, not causal claims.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.covariance import LedoitWolf

from run_causcale_smoke import directed_probabilities, model_args, sha256


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--causcale-src", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--concept-fraction", type=float, default=0.8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("task3") / "outputs" / "causcale",
    )
    parser.add_argument("--output-stem", default="causcale_bootstrap_20")
    args = parser.parse_args()
    for path in [args.matrix, args.metadata, args.checkpoint]:
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.causcale_src.is_dir():
        raise FileNotFoundError(args.causcale_src)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    sys.path.insert(0, str(args.causcale_src))
    from model import CauScale

    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    matrix = np.load(args.matrix).astype(np.float32)
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    columns = metadata["columns"]
    layers = sorted({int(column["layer"]) for column in columns})
    concepts = []
    for column in columns:
        if column["concept"] not in concepts:
            concepts.append(column["concept"])
    if matrix.shape != (1000, 128) or len(layers) != 4 or len(concepts) != 32:
        raise ValueError("Expected the 1,000 x (4 layers x 32 concepts) pilot")

    model = CauScale(model_args())
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model = model.eval().cuda()

    n = matrix.shape[1]
    concepts_per_run = round(len(concepts) * args.concept_fraction)
    availability = np.zeros((n, n), dtype=np.int32)
    selected_count = np.zeros((n, n), dtype=np.int32)
    probability_sum = np.zeros((n, n), dtype=np.float64)
    probability_sq_sum = np.zeros((n, n), dtype=np.float64)
    edge_sets: list[set[int]] = []
    available_edge_sets: list[set[int]] = []
    run_records = []

    print(
        f"Running {args.runs} bootstraps with {concepts_per_run}/32 "
        "concept groups...",
        flush=True,
    )
    for run in range(args.runs):
        seed = 20260723 + run
        rng = np.random.RandomState(seed)
        chosen_concepts = np.sort(
            rng.choice(len(concepts), size=concepts_per_run, replace=False)
        )
        nodes = np.asarray(
            [
                layer_index * len(concepts) + concept_index
                for layer_index in range(len(layers))
                for concept_index in chosen_concepts
            ],
            dtype=np.int64,
        )
        row_indices = rng.choice(len(matrix), size=len(matrix), replace=True)
        sampled = matrix[row_indices][:, nodes]
        prior_indices = rng.choice(len(sampled), size=500, replace=False)
        precision = (
            LedoitWolf().fit(sampled[prior_indices]).get_precision().astype(np.float32)
        )
        mean = sampled.mean(axis=0, keepdims=True)
        std = sampled.std(axis=0, keepdims=True)
        standardized = (sampled - mean) / np.where(std == 0, 1.0, std)
        local_layers = np.asarray([columns[index]["layer"] for index in nodes])
        local_allowed = local_layers[:, None] < local_layers[None, :]

        batch = {
            "data": torch.from_numpy(standardized).unsqueeze(0).cuda(),
            "interv": torch.zeros(
                1, len(sampled), len(nodes), device="cuda"
            ),
            "feats": torch.from_numpy(precision).unsqueeze(0).cuda(),
        }
        tick = time.time()
        encoded = model.encoder(batch)
        directed, _ = directed_probabilities(model, encoded, len(nodes))
        torch.cuda.synchronize()
        elapsed = time.time() - tick
        local_scores = directed.cpu().numpy()
        local_selected = local_allowed & (local_scores >= args.threshold)

        global_rows, global_cols = np.meshgrid(nodes, nodes, indexing="ij")
        global_rows = global_rows[local_allowed]
        global_cols = global_cols[local_allowed]
        local_values = local_scores[local_allowed]
        availability[global_rows, global_cols] += 1
        selected_count[global_rows, global_cols] += (
            local_values >= args.threshold
        ).astype(np.int32)
        probability_sum[global_rows, global_cols] += local_values
        probability_sq_sum[global_rows, global_cols] += local_values**2
        selected_flat = set(
            (global_rows[local_values >= args.threshold] * n
             + global_cols[local_values >= args.threshold]).tolist()
        )
        available_flat = set((global_rows * n + global_cols).tolist())
        edge_sets.append(selected_flat)
        available_edge_sets.append(available_flat)
        run_records.append(
            {
                "run": run,
                "seed": seed,
                "concepts": [concepts[index] for index in chosen_concepts],
                "node_count": len(nodes),
                "selected_edge_count": len(selected_flat),
                "inference_seconds": elapsed,
            }
        )
        print(
            f"  bootstrap {run + 1}/{args.runs}: "
            f"{len(selected_flat)} edges in {elapsed:.3f}s",
            flush=True,
        )

    observed = availability > 0
    selection_frequency = np.zeros((n, n), dtype=np.float32)
    mean_probability = np.zeros((n, n), dtype=np.float32)
    probability_std = np.zeros((n, n), dtype=np.float32)
    selection_frequency[observed] = (
        selected_count[observed] / availability[observed]
    )
    mean_probability[observed] = (
        probability_sum[observed] / availability[observed]
    )
    variance = np.zeros((n, n), dtype=np.float64)
    variance[observed] = (
        probability_sq_sum[observed] / availability[observed]
        - mean_probability[observed].astype(np.float64) ** 2
    )
    probability_std[observed] = np.sqrt(np.maximum(variance[observed], 0))

    global_layers = np.asarray([column["layer"] for column in columns])
    globally_allowed = global_layers[:, None] < global_layers[None, :]
    stable = (
        observed
        & globally_allowed
        & (selection_frequency >= 0.8)
        & (mean_probability >= args.threshold)
    )
    stable_indices = np.argwhere(stable)
    stable_order = sorted(
        range(len(stable_indices)),
        key=lambda index: (
            selection_frequency[tuple(stable_indices[index])],
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
                "availability": int(availability[source, target]),
                "selection_frequency": float(
                    selection_frequency[source, target]
                ),
                "mean_probability": float(mean_probability[source, target]),
                "probability_std": float(probability_std[source, target]),
            }
        )

    pairwise_jaccard = []
    for left_index, right_index in itertools.combinations(
        range(len(edge_sets)), 2
    ):
        common = (
            available_edge_sets[left_index]
            & available_edge_sets[right_index]
        )
        left = edge_sets[left_index] & common
        right = edge_sets[right_index] & common
        union = left | right
        pairwise_jaccard.append(len(left & right) / len(union) if union else 1.0)
    same_concept_stable = sum(
        edge["source_concept"] == edge["target_concept"] for edge in stable_edges
    )
    record = {
        "status": "bootstrap_candidate_graph_not_causal_graph",
        "matrix": str(args.matrix),
        "matrix_sha256": sha256(args.matrix),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "bootstrap_runs": args.runs,
        "sample_bootstrap": "1000 rows with replacement",
        "feature_bootstrap": (
            f"{concepts_per_run}/32 concept groups without replacement; "
            "all four layer-nodes retained together"
        ),
        "edge_probability_threshold": args.threshold,
        "stable_selection_frequency_threshold": 0.8,
        "stable_edge_count": len(stable_edges),
        "same_concept_stable_edge_count": same_concept_stable,
        "same_concept_stable_edge_fraction": (
            same_concept_stable / len(stable_edges) if stable_edges else 0.0
        ),
        "pairwise_edge_set_jaccard_min": float(np.min(pairwise_jaccard)),
        "pairwise_edge_set_jaccard_median": float(
            np.median(pairwise_jaccard)
        ),
        "pairwise_edge_set_jaccard_max": float(np.max(pairwise_jaccard)),
        "availability_min_for_allowed_edges": int(
            availability[globally_allowed].min()
        ),
        "availability_median_for_allowed_edges": float(
            np.median(availability[globally_allowed])
        ),
        "availability_max_for_allowed_edges": int(
            availability[globally_allowed].max()
        ),
        "run_records": run_records,
        "stable_edges": stable_edges,
        "total_seconds": time.time() - started,
        "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "interpretation": (
            "Stability only prioritizes candidate edges. The checkpoint is "
            "zero-shot OOD and stable edges still require held-out activation "
            "intervention validation."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{args.output_stem}.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        args.output_dir / f"{args.output_stem}.npz",
        availability=availability,
        selected_count=selected_count,
        selection_frequency=selection_frequency,
        mean_probability=mean_probability,
        probability_std=probability_std,
        stable_mask=stable,
    )
    print(json.dumps(record, ensure_ascii=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
