"""Generate a 20-node SCM fixture for adapter tests.

The generated values are explicitly *not* J-space measurements and therefore
must never be reported as evidence that CauScale or v4.1 works on a model.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np


THIS_FILE = Path(__file__).resolve()
EXPERIMENT_ROOT = THIS_FILE.parents[1]
sys.path.insert(0, str(THIS_FILE.parent))

from contracts import SCHEMA_VERSION, directed_to_pair_probabilities, save_json, sha256_file, validate_dataset  # noqa: E402


FAMILIES = {
    "geography": ["geography", "country", "France", "Paris", "Europe"],
    "animals": ["animal", "bird", "eagle", "feathers", "flight"],
    "chemistry": ["chemistry", "element", "oxygen", "gas", "respiration"],
    "astronomy": ["astronomy", "planet", "Mars", "red planet", "orbit"],
}


def build_graph() -> tuple[list[dict], dict[str, str], np.ndarray, list[tuple[int, int, float]]]:
    nodes: list[dict] = []
    labels: dict[str, str] = {}
    weighted_edges: list[tuple[int, int, float]] = []
    for family_no, (family, meanings) in enumerate(FAMILIES.items()):
        offset = family_no * len(meanings)
        for level, meaning in enumerate(meanings):
            node_id = f"z{offset + level:02d}"
            nodes.append(
                {
                    "node_id": node_id,
                    "column": offset + level,
                    "kind": "fixture_coordinate",
                    "family": family,
                    "level": level,
                }
            )
            labels[node_id] = meaning
        for level in range(len(meanings) - 1):
            sign = -1.0 if (family_no == 2 and level == 2) else 1.0
            weighted_edges.append((offset + level, offset + level + 1, sign * (0.72 + 0.04 * level)))

    d = len(nodes)
    graph = np.zeros((d, d), dtype=np.uint8)
    for source, target, _ in weighted_edges:
        graph[source, target] = 1
    return nodes, labels, graph, weighted_edges


def simulate(
    rng: np.random.Generator,
    n_samples: int,
    d: int,
    weighted_edges: list[tuple[int, int, float]],
) -> tuple[np.ndarray, np.ndarray]:
    interventions = np.zeros((n_samples, d), dtype=np.uint8)
    n_observational = n_samples // 2
    targets = np.repeat(np.arange(d), (n_samples - n_observational) // d)
    if len(targets) != n_samples - n_observational:
        raise ValueError("interventional rows must divide evenly across nodes")
    rng.shuffle(targets)
    interventions[n_observational + np.arange(len(targets)), targets] = 1

    parents: dict[int, list[tuple[int, float]]] = {node: [] for node in range(d)}
    for source, target, weight in weighted_edges:
        parents[target].append((source, weight))

    X = np.zeros((n_samples, d), dtype=np.float32)
    for row in range(n_samples):
        target = int(np.argmax(interventions[row])) if interventions[row].any() else None
        for node in range(d):
            if node == target:
                X[row, node] = rng.choice([-1.0, 1.0]) * rng.normal(2.5, 0.2)
                continue
            structural = sum(weight * float(X[row, source]) for source, weight in parents[node])
            family_context = 0.15 * np.sin((row + 1) * (node // 5 + 1) / 17.0)
            X[row, node] = np.tanh(structural + family_context) + rng.normal(0.0, 0.28)
    return X, interventions


def main() -> None:
    config_path = EXPERIMENT_ROOT / "config" / "smoke.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    rng = np.random.default_rng(seed)
    nodes, labels, oracle_graph, weighted_edges = build_graph()
    X, interventions = simulate(rng, n_samples=1000, d=len(nodes), weighted_edges=weighted_edges)

    output = EXPERIMENT_ROOT / "runs" / "fixture"
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "X.npy", X)
    np.save(output / "interventions.npy", interventions)
    np.save(output / "oracle_graph.npy", oracle_graph)
    save_json(output / "nodes.json", nodes)
    save_json(output / "labels.json", labels)

    clean = interventions.sum(axis=1) == 0
    dev_rows = np.flatnonzero(clean)[:400]
    mean = X[dev_rows].mean(axis=0)
    std = X[dev_rows].std(axis=0)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": "fixture_scm_20x1000",
        "evidence_status": "engineering_fixture_only_not_jspace_evidence",
        "seed": seed,
        "n_samples": int(X.shape[0]),
        "n_nodes": int(X.shape[1]),
        "dev_rows": dev_rows.tolist(),
        "standardization": {
            "fit_split": "first_400_observational_rows",
            "mean": mean.astype(float).tolist(),
            "std": std.astype(float).tolist(),
        },
        "edge_convention": "adjacency[source,target] = 1",
        "files": {},
    }
    for name in ["X.npy", "interventions.npy", "oracle_graph.npy", "nodes.json", "labels.json"]:
        manifest["files"][name] = sha256_file(output / name)
    save_json(output / "manifest.json", manifest)

    # Near-perfect probabilities exercise the exact CauScale [no-edge,i->j,j->i] contract.
    directed = np.full(oracle_graph.shape, 0.01, dtype=np.float32)
    directed[oracle_graph.astype(bool)] = 0.97
    np.fill_diagonal(directed, 0.0)
    no_edge = np.full(oracle_graph.shape, 0.98, dtype=np.float32)
    no_edge[(oracle_graph + oracle_graph.T).astype(bool)] = 0.02
    np.fill_diagonal(no_edge, 0.0)
    pairs, pair_probs = directed_to_pair_probabilities(directed, no_edge)
    np.savez_compressed(
        output / "oracle_pair_probs.npz",
        pair_index=pairs,
        pair_probs=pair_probs,
        node_names=np.array([entry["node_id"] for entry in nodes]),
        source=np.array("fixture_oracle_with_0.97_confidence"),
    )

    # The official loader needs this CSV. The prediction-only wrapper ignores graph labels.
    with (output / "causcale_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["fp_data", "fp_graph", "fp_regime", "split"])
        writer.writeheader()
        writer.writerow(
            {
                "fp_data": (output / "X.npy").as_posix(),
                "fp_graph": (output / "oracle_graph.npy").as_posix(),
                "fp_regime": (output / "interventions.npy").as_posix(),
                "split": "test",
            }
        )

    report = validate_dataset(output, require_oracle=True)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"fixture saved: {output}")


if __name__ == "__main__":
    main()
