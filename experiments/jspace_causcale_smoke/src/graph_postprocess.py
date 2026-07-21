"""Turn CauScale three-class pair probabilities into an auditable DAG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def creates_cycle(adjacency: np.ndarray, source: int, target: int) -> bool:
    """Adding source->target cycles iff target already reaches source."""
    stack = [target]
    seen: set[int] = set()
    while stack:
        node = stack.pop()
        if node == source:
            return True
        if node in seen:
            continue
        seen.add(node)
        stack.extend(int(child) for child in np.flatnonzero(adjacency[node]))
    return False


def build_dag(
    pair_index: np.ndarray,
    pair_probs: np.ndarray,
    d: int,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    if pair_index.shape != (d * (d - 1) // 2, 2):
        raise ValueError(f"pair_index has wrong shape {pair_index.shape} for d={d}")
    if pair_probs.shape != (len(pair_index), 3):
        raise ValueError("pair_probs must be [n_pairs,3]")
    if not np.isfinite(pair_probs).all() or np.any(pair_probs < 0):
        raise ValueError("pair probabilities must be finite and nonnegative")
    if not np.allclose(pair_probs.sum(axis=1), 1.0, atol=1e-4):
        raise ValueError("each probability triple must sum to one")

    directed = np.zeros((d, d), dtype=np.float32)
    no_edge = np.zeros((d, d), dtype=np.float32)
    candidates: list[tuple[float, int, int, float]] = []
    for (i, j), probabilities in zip(pair_index, pair_probs):
        i, j = int(i), int(j)
        if not (0 <= i < j < d):
            raise ValueError(f"pair_index must contain ordered i<j pairs; got {(i, j)}")
        p0, pij, pji = map(float, probabilities)
        no_edge[i, j] = no_edge[j, i] = p0
        directed[i, j], directed[j, i] = pij, pji
        best_class = int(np.argmax(probabilities))
        if best_class == 1 and pij >= threshold:
            candidates.append((pij, i, j, p0))
        elif best_class == 2 and pji >= threshold:
            candidates.append((pji, j, i, p0))

    adjacency = np.zeros((d, d), dtype=np.uint8)
    decisions: list[dict] = []
    for probability, source, target, p0 in sorted(candidates, reverse=True):
        cycle = creates_cycle(adjacency, source, target)
        if not cycle:
            adjacency[source, target] = 1
        decisions.append(
            {
                "source": source,
                "target": target,
                "probability": probability,
                "no_edge_probability": p0,
                "kept": not cycle,
                "reason": "kept" if not cycle else "removed_to_break_cycle",
            }
        )
    return adjacency, directed, no_edge, decisions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probabilities", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--oracle", type=Path)
    args = parser.parse_args()
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("threshold must lie strictly between zero and one")

    payload = np.load(args.probabilities, allow_pickle=False)
    pair_index = payload["pair_index"]
    pair_probs = payload["pair_probs"]
    if "node_names" in payload:
        node_names = payload["node_names"].astype(str)
        d = len(node_names)
    else:
        d = int(pair_index.max()) + 1
        node_names = np.array([f"z{i:02d}" for i in range(d)])

    adjacency, directed, no_edge, decisions = build_dag(
        pair_index, pair_probs, d=d, threshold=args.threshold
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        adjacency=adjacency,
        directed_probs=directed,
        no_edge_probs=no_edge,
        node_names=node_names,
        threshold=np.array(args.threshold, dtype=np.float32),
    )

    summary = {
        "probabilities": str(args.probabilities.resolve()),
        "output": str(args.output.resolve()),
        "threshold": args.threshold,
        "n_nodes": d,
        "candidate_edges": len(decisions),
        "kept_edges": int(adjacency.sum()),
        "cycle_edges_removed": sum(not decision["kept"] for decision in decisions),
        "decisions": decisions,
    }
    if args.oracle:
        oracle = np.load(args.oracle)
        if oracle.shape != adjacency.shape:
            raise ValueError("oracle and predicted adjacency shapes differ")
        tp = int(np.logical_and(adjacency, oracle).sum())
        fp = int(np.logical_and(adjacency, 1 - oracle).sum())
        fn = int(np.logical_and(1 - adjacency, oracle).sum())
        summary["oracle_fixture_metrics"] = {
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "precision": tp / max(tp + fp, 1),
            "recall": tp / max(tp + fn, 1),
            "shd": fp + fn,
            "warning": "fixture plumbing metric; not J-space evidence",
        }

    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
