"""Feature Propagation baseline shared by graph-semantic completion tasks.

The propagation rule follows Rossi et al. (2022), *On the Unreasonable
Effectiveness of Feature Propagation in Learning on Graphs with Missing Node
Features* (arXiv:2111.12128): diffuse with a symmetrically normalised
adjacency and clamp every known feature after every iteration.

The benchmark graph is directed, whereas the original method operates on a
homogeneous undirected graph.  We therefore use the binary undirected
projection of the given graph.  This adaptation never estimates an edge from
the response matrix and never consumes a latent description.  For Task 2,
only fold-visible observed-label embeddings may be supplied as anchors.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


FEATURE_PROP_VERSION = "rossi22-fp-undirected-causal-projection-v1"


def _embedding_dict(
    embeddings: Mapping[str, Any],
    valid_nodes: Sequence[str],
    name: str,
) -> tuple[dict[str, np.ndarray], int]:
    """Validate and copy a sparse node-to-vector mapping."""

    if not isinstance(embeddings, Mapping):
        raise TypeError(f"{name} must be a node -> embedding mapping")

    valid = set(valid_nodes)
    unknown = set(embeddings) - valid
    if unknown:
        raise KeyError(f"{name} contains nodes outside the graph: {sorted(unknown)}")
    if not embeddings:
        raise ValueError(f"{name} must contain at least one embedding")

    output: dict[str, np.ndarray] = {}
    dimension: int | None = None
    for node, value in embeddings.items():
        vector = np.asarray(value, dtype=float)
        if vector.ndim != 1 or vector.size == 0:
            raise ValueError(f"embedding for {node!r} must be a non-empty 1-D vector")
        if not np.all(np.isfinite(vector)):
            raise ValueError(f"embedding for {node!r} contains a non-finite value")
        if dimension is None:
            dimension = int(vector.size)
        elif vector.size != dimension:
            raise ValueError(
                f"embedding for {node!r} has dimension {vector.size}; "
                f"expected {dimension}"
            )
        output[node] = vector.copy()

    assert dimension is not None
    return output, dimension


def _fallback_vector(
    embeddings: Mapping[str, np.ndarray],
    dimension: int,
    fallback: str | np.ndarray,
) -> np.ndarray:
    """Resolve the value for a component with no visible semantic anchor."""

    if isinstance(fallback, str):
        if fallback == "mean":
            return np.mean(np.stack(list(embeddings.values())), axis=0)
        if fallback == "zeros":
            return np.zeros(dimension, dtype=float)
        raise ValueError("fallback must be 'mean', 'zeros', or a 1-D vector")

    vector = np.asarray(fallback, dtype=float)
    if vector.ndim != 1 or vector.size != dimension:
        raise ValueError(f"fallback vector must have shape ({dimension},)")
    if not np.all(np.isfinite(vector)):
        raise ValueError("fallback vector contains a non-finite value")
    return vector.copy()


def feature_propagation(
    graph: Any,
    known_embeddings: Mapping[str, Any],
    *,
    max_iter: int = 40,
    tol: float = 1e-8,
    fallback: str | np.ndarray = "zeros",
) -> dict[str, np.ndarray]:
    """Recover every missing node vector with clamped Feature Propagation.

    Reciprocal directed edges count once in the binary undirected projection.
    Components without a known node are not identifiable by propagation and
    receive ``fallback`` after the iterations.
    """

    nodes = list(graph.nodes)
    if len(set(nodes)) != len(nodes):
        raise ValueError("graph.nodes must contain unique node names")
    if not nodes:
        if known_embeddings:
            raise KeyError("known_embeddings contains nodes but the graph is empty")
        return {}
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be a finite non-negative number")

    known, dimension = _embedding_dict(
        known_embeddings, nodes, "known_embeddings"
    )
    fill = _fallback_vector(known, dimension, fallback)
    index = {node: position for position, node in enumerate(nodes)}
    node_count = len(nodes)

    adjacency = np.zeros((node_count, node_count), dtype=float)
    for source, target in graph.edges:
        if source not in index or target not in index:
            raise KeyError(f"edge ({source!r}, {target!r}) contains an unknown node")
        source_index, target_index = index[source], index[target]
        adjacency[source_index, target_index] = 1.0
        adjacency[target_index, source_index] = 1.0

    degree = adjacency.sum(axis=1)
    inverse_sqrt_degree = np.zeros_like(degree)
    nonisolated = degree > 0
    inverse_sqrt_degree[nonisolated] = 1.0 / np.sqrt(degree[nonisolated])
    normalised = (
        inverse_sqrt_degree[:, None]
        * adjacency
        * inverse_sqrt_degree[None, :]
    )

    known_indices = np.asarray([index[node] for node in known], dtype=int)
    known_values = np.stack([known[node] for node in known])
    values = np.zeros((node_count, dimension), dtype=float)
    values[known_indices] = known_values

    # Determine which undirected components contain at least one anchor.
    anchored = np.zeros(node_count, dtype=bool)
    anchored[known_indices] = True
    frontier = anchored.copy()
    while frontier.any():
        reached = (adjacency @ frontier.astype(float)) > 0
        new_frontier = reached & ~anchored
        if not new_frontier.any():
            break
        anchored |= new_frontier
        frontier = new_frontier

    for _ in range(int(max_iter)):
        updated = normalised @ values
        updated[known_indices] = known_values
        if np.max(np.abs(updated - values)) <= tol:
            values = updated
            break
        values = updated

    values[~anchored] = fill
    values[known_indices] = known_values
    return {node: values[position].copy() for position, node in enumerate(nodes)}


def predict_task2_latent_embeddings(
    graph: Any,
    visible_observed_embeddings: Mapping[str, Any],
    *,
    latent_nodes: Sequence[str] | None = None,
    max_iter: int = 40,
    tol: float = 1e-8,
    fallback: str | np.ndarray = "zeros",
) -> dict[str, np.ndarray]:
    """Predict Task 2 latent vectors from fold-visible observed anchors only.

    The narrow signature is a leakage guard: callers cannot provide latent
    embeddings, response data, hidden observed-label embeddings, or latent
    ground-truth descriptions through this adapter.
    """

    if not isinstance(visible_observed_embeddings, Mapping):
        raise TypeError(
            "visible_observed_embeddings must be a node -> embedding mapping"
        )
    observed = set(graph.observed)
    invalid_anchors = set(visible_observed_embeddings) - observed
    if invalid_anchors:
        raise ValueError(
            "Task 2 Feature Propagation anchors must all be observed nodes; "
            f"got {sorted(invalid_anchors)}"
        )

    requested = list(graph.latents if latent_nodes is None else latent_nodes)
    invalid_targets = set(requested) - set(graph.latents)
    if invalid_targets:
        raise ValueError(
            f"Task 2 targets must be latent graph nodes; got {sorted(invalid_targets)}"
        )
    if len(set(requested)) != len(requested):
        raise ValueError("latent_nodes must be unique")

    propagated = feature_propagation(
        graph,
        visible_observed_embeddings,
        max_iter=max_iter,
        tol=tol,
        fallback=fallback,
    )
    return {node: propagated[node] for node in requested}


__all__ = [
    "FEATURE_PROP_VERSION",
    "feature_propagation",
    "predict_task2_latent_embeddings",
]
