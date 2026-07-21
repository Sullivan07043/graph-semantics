"""Shared on-disk contract for the J-space/CauScale smoke experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_dataset(directory: Path, *, require_oracle: bool = False) -> dict[str, Any]:
    """Validate arrays and metadata without modifying or standardizing the data."""
    directory = directory.resolve()
    required = ["X.npy", "interventions.npy", "nodes.json", "labels.json", "manifest.json"]
    if require_oracle:
        required.append("oracle_graph.npy")
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing dataset files in {directory}: {missing}")

    X = np.load(directory / "X.npy", mmap_mode="r")
    interventions = np.load(directory / "interventions.npy", mmap_mode="r")
    nodes = load_json(directory / "nodes.json")
    labels = load_json(directory / "labels.json")
    manifest = load_json(directory / "manifest.json")

    if X.ndim != 2 or X.shape[0] < 3 or X.shape[1] < 2:
        raise ValueError(f"X must be [N,d] with N>=3,d>=2; got {X.shape}")
    if X.dtype != np.float32:
        raise TypeError(f"X must be float32; got {X.dtype}")
    if interventions.shape != X.shape:
        raise ValueError(
            f"interventions must match X shape {X.shape}; got {interventions.shape}"
        )
    if not np.issubdtype(interventions.dtype, np.integer) and interventions.dtype != np.bool_:
        raise TypeError(f"interventions must be integer/bool; got {interventions.dtype}")
    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or infinity")
    if not np.isin(interventions, [0, 1]).all():
        raise ValueError("interventions must contain only 0/1")
    per_row = np.asarray(interventions).sum(axis=1)
    if np.any(per_row > 1):
        raise ValueError("smoke protocol permits at most one intervention target per row")

    node_ids = [entry["node_id"] for entry in nodes]
    if len(nodes) != X.shape[1] or len(set(node_ids)) != len(node_ids):
        raise ValueError("nodes.json must contain one unique ordered node_id per X column")
    if set(labels) != set(node_ids):
        raise ValueError("labels.json keys must exactly match nodes.json node_id values")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported manifest schema {manifest.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION}"
        )

    oracle_path = directory / "oracle_graph.npy"
    if oracle_path.is_file():
        oracle = np.load(oracle_path, mmap_mode="r")
        if oracle.shape != (X.shape[1], X.shape[1]):
            raise ValueError(f"oracle graph must be {(X.shape[1], X.shape[1])}; got {oracle.shape}")
        if not np.isin(oracle, [0, 1]).all() or np.any(np.diag(oracle)):
            raise ValueError("oracle graph must be binary with a zero diagonal")

    return {
        "directory": str(directory),
        "n_samples": int(X.shape[0]),
        "n_nodes": int(X.shape[1]),
        "observational_rows": int(np.sum(per_row == 0)),
        "interventional_rows": int(np.sum(per_row == 1)),
        "node_ids": node_ids,
        "manifest": manifest,
    }


def directed_to_pair_probabilities(
    directed: np.ndarray, no_edge: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Convert directed dxd probabilities to CauScale's i<j three-class layout."""
    directed = np.asarray(directed, dtype=np.float64)
    if directed.ndim != 2 or directed.shape[0] != directed.shape[1]:
        raise ValueError("directed probabilities must be a square matrix")
    d = directed.shape[0]
    pairs = np.array([(i, j) for i in range(d) for j in range(i + 1, d)], dtype=np.int64)
    probs = np.empty((len(pairs), 3), dtype=np.float64)
    for row, (i, j) in enumerate(pairs):
        pij = float(directed[i, j])
        pji = float(directed[j, i])
        p0 = float(no_edge[i, j]) if no_edge is not None else max(0.0, 1.0 - pij - pji)
        total = p0 + pij + pji
        if total <= 0 or min(p0, pij, pji) < 0:
            raise ValueError(f"invalid probability triple for pair {(i, j)}")
        probs[row] = (p0 / total, pij / total, pji / total)
    return pairs, probs.astype(np.float32)
