"""Aggregate CauScale feature-bootstrap predictions into a stable consensus DAG."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from contracts import load_json, save_json, sha256_file
from graph_postprocess import build_dag


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--pattern", default="causcale_seed*.npz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-frequency", type=float, default=0.8)
    args = parser.parse_args()
    if not 0 < args.min_frequency <= 1:
        raise ValueError("min-frequency must be in (0,1]")

    paths = sorted(args.dataset.glob(args.pattern))
    if len(paths) < 2:
        raise ValueError(f"need at least two predictions matching {args.pattern!r}")
    payloads = [np.load(path, allow_pickle=False) for path in paths]
    pair_index = payloads[0]["pair_index"]
    node_names = payloads[0]["node_names"].astype(str)
    for path, payload in zip(paths[1:], payloads[1:]):
        if not np.array_equal(payload["pair_index"], pair_index):
            raise ValueError(f"pair ordering differs in {path}")
        if not np.array_equal(payload["node_names"].astype(str), node_names):
            raise ValueError(f"node ordering differs in {path}")

    pair_probs = np.stack([payload["pair_probs"] for payload in payloads])
    adjacencies = np.stack([payload["adjacency"].astype(bool) for payload in payloads])
    mean_pair_probs = pair_probs.mean(axis=0)
    d = len(node_names)
    mean_dag, directed_probs, no_edge_probs, _ = build_dag(
        pair_index, mean_pair_probs, d=d, threshold=args.threshold
    )
    edge_frequency = adjacencies.mean(axis=0)
    stable_before_filter = edge_frequency >= args.min_frequency
    consensus = mean_dag.astype(bool) & stable_before_filter
    jaccards = []
    for left, right in itertools.combinations(adjacencies, 2):
        union = int((left | right).sum())
        jaccards.append(float((left & right).sum() / max(union, 1)))
    seeds = []
    for path in paths:
        digits = "".join(character for character in path.stem if character.isdigit())
        seeds.append(int(digits) if digits else path.stem)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        pair_index=pair_index,
        pair_probs=mean_pair_probs.astype(np.float32),
        directed_probs=directed_probs.astype(np.float32),
        no_edge_probs=no_edge_probs.astype(np.float32),
        edge_frequency=edge_frequency.astype(np.float32),
        adjacency=consensus.astype(np.uint8),
        node_names=node_names,
        threshold=np.float32(args.threshold),
        min_frequency=np.float32(args.min_frequency),
        feature_seeds=np.asarray(seeds),
    )
    manifest = load_json(args.dataset / "manifest.json")
    edge_counts = adjacencies.sum(axis=(1, 2))
    metadata = {
        "mode": "feature_bootstrap_consensus",
        "dataset": {
            "directory": str(args.dataset.resolve()),
            "dataset_id": manifest.get("dataset_id"),
            "manifest": manifest,
        },
        "inputs": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)} for path in paths
        ],
        "feature_seeds": seeds,
        "n_predictions": len(paths),
        "threshold": args.threshold,
        "min_frequency": args.min_frequency,
        "edge_count": {
            "minimum": int(edge_counts.min()),
            "mean": float(edge_counts.mean()),
            "maximum": int(edge_counts.max()),
        },
        "pairwise_jaccard": {
            "minimum": float(np.min(jaccards)),
            "mean": float(np.mean(jaccards)),
            "maximum": float(np.max(jaccards)),
        },
        "stable_edges_before_mean_dag_filter": int(stable_before_filter.sum()),
        "mean_probability_dag_edges": int(mean_dag.sum()),
        "consensus_edges": int(consensus.sum()),
    }
    save_json(args.output.with_suffix(".json"), metadata)
    print(json.dumps({"output": str(args.output), **metadata["edge_count"],
                      "jaccard_mean": metadata["pairwise_jaccard"]["mean"],
                      "consensus_edges": metadata["consensus_edges"]}, indent=2))


if __name__ == "__main__":
    main()
