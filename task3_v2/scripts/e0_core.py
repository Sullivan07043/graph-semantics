"""Deterministic, model-free core utilities for Task 3 E0'.

This module deliberately has no dependency on the Stage-3 encoder, checkpoints,
or decode dictionary.  It owns the auditable experiment mechanics around that
frozen solver: graph validation, SCM generation, mask/permutation validation,
local ranking metrics, and paired hierarchical bootstrap statistics.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


GRAPH_SCHEMA_VERSION = "task3.e0_prime.graph.v1"
EXPECTED_GRAPH_IDS = ("graph_00", "graph_01", "graph_02")
EXPECTED_WORLDS = (
    "industrial_cooling_system",
    "logistics_and_delivery_system",
    "water_treatment_system",
)
EXPECTED_NODE_IDS = tuple(f"node_{i:03d}" for i in range(20))
EXPECTED_MODULES = 4
MIN_EDGES = 24
MAX_EDGES = 32
MAX_INDEGREE = 3
MIN_COEFFICIENT = 0.4
MAX_COEFFICIENT = 0.9
EXPECTED_FOLDS = 5
MASKED_PER_FOLD = 4
EXPECTED_PERMUTATIONS = 20
DEFAULT_SPLIT_SIZES = (1200, 400, 400)


class ValidationError(ValueError):
    """Raised when a frozen E0' fixture violates the declared protocol."""


def _fail(message: str, source: str | os.PathLike[str] | None = None) -> None:
    prefix = f"{source}: " if source is not None else ""
    raise ValidationError(prefix + message)


def _is_real_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and math.isfinite(float(value))
    )


def _stable_key(value: Any) -> tuple[str, str]:
    return type(value).__name__, repr(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"JSON object keys must be strings, got {type(key).__name__}")
            out[key] = _jsonable(item)
        return out
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_jsonable(item) for item in sorted(value, key=_stable_key)]
    return value


def canonical_json_dumps(value: Any) -> str:
    """Return the unique compact JSON representation used for content hashes."""

    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_json(value: Any) -> str:
    """SHA256 of :func:`canonical_json_dumps`, as a lowercase hex digest."""

    return hashlib.sha256(canonical_json_dumps(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 8 << 20) -> str:
    """Stream a file into SHA256 without loading it into memory."""

    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: str | os.PathLike[str]) -> Any:
    """Load UTF-8 JSON with errors annotated by its source path."""

    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"{source}: could not load JSON: {exc}") from exc


def write_json(
    path: str | os.PathLike[str],
    value: Any,
    *,
    indent: int = 2,
) -> Path:
    """Atomically write deterministic, human-readable UTF-8 JSON."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    text = json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        indent=indent,
        allow_nan=False,
    )
    temporary.write_text(text + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    return destination


# Backwards-friendly aliases for callers that use noun-first names.
json_sha256 = sha256_json
dump_json = write_json


def _require_mapping(value: Any, name: str, source: Any = None) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{name} must be a JSON object", source)
    return value


def _require_list(value: Any, name: str, source: Any = None) -> list[Any]:
    if not isinstance(value, list):
        _fail(f"{name} must be a JSON array", source)
    return value


def _require_nonempty_string(value: Any, name: str, source: Any = None) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(f"{name} must be a non-empty string", source)
    return value


def _graph_components(
    spec: Mapping[str, Any],
) -> tuple[
    list[str],
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
    dict[str, list[str]],
    dict[str, list[str]],
]:
    nodes = spec["nodes"]
    node_ids = [node["id"] for node in nodes]
    node_by_id = {node["id"]: node for node in nodes}
    edges = spec["edges"]
    parents = {node_id: [] for node_id in node_ids}
    children = {node_id: [] for node_id in node_ids}
    for edge in edges:
        parents[edge["target"]].append(edge["source"])
        children[edge["source"]].append(edge["target"])
    return node_ids, node_by_id, edges, parents, children


def _kahn_topological_order(
    node_ids: Sequence[str],
    children: Mapping[str, Sequence[str]],
    parents: Mapping[str, Sequence[str]],
) -> list[str]:
    position = {node_id: i for i, node_id in enumerate(node_ids)}
    indegree = {node_id: len(parents[node_id]) for node_id in node_ids}
    ready = sorted(
        (node_id for node_id in node_ids if indegree[node_id] == 0),
        key=position.__getitem__,
    )
    order: list[str] = []
    while ready:
        node = ready.pop(0)
        order.append(node)
        for child in sorted(children[node], key=position.__getitem__):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
                ready.sort(key=position.__getitem__)
    return order


def validate_graph_spec(
    spec: Mapping[str, Any],
    source: str | os.PathLike[str] | None = "<memory>",
) -> dict[str, Any]:
    """Strictly validate one frozen 20-node natural-semantic causal DAG.

    The returned dictionary contains structural statistics and explicit motif
    witnesses.  The input is never mutated.
    """

    spec = _require_mapping(spec, "graph specification", source)
    required = {
        "schema_version",
        "graph_id",
        "world",
        "graph_seed",
        "observed_only",
        "hidden_confounders",
        "effect_sign",
        "modules",
        "nodes",
        "topological_order",
        "edges",
    }
    missing = sorted(required - set(spec))
    if missing:
        _fail(f"missing required fields: {', '.join(missing)}", source)

    if spec["schema_version"] != GRAPH_SCHEMA_VERSION:
        _fail(
            f"schema_version must be {GRAPH_SCHEMA_VERSION!r}, "
            f"got {spec['schema_version']!r}",
            source,
        )
    _require_nonempty_string(spec["graph_id"], "graph_id", source)
    _require_nonempty_string(spec["world"], "world", source)
    if not isinstance(spec["graph_seed"], int) or isinstance(spec["graph_seed"], bool):
        _fail("graph_seed must be an integer", source)
    if spec["observed_only"] is not True:
        _fail("observed_only must be true", source)
    if spec["hidden_confounders"] is not False:
        _fail("hidden_confounders must be false", source)
    if spec["effect_sign"] != "positive_monotone":
        _fail("effect_sign must be 'positive_monotone'", source)

    modules = _require_list(spec["modules"], "modules", source)
    if len(modules) != EXPECTED_MODULES:
        _fail(f"modules must contain exactly {EXPECTED_MODULES} entries", source)
    for i, module in enumerate(modules):
        _require_nonempty_string(module, f"modules[{i}]", source)
    if len(set(modules)) != len(modules):
        _fail("module names must be unique", source)

    nodes = _require_list(spec["nodes"], "nodes", source)
    if len(nodes) != len(EXPECTED_NODE_IDS):
        _fail(f"nodes must contain exactly {len(EXPECTED_NODE_IDS)} entries", source)
    node_ids: list[str] = []
    labels: list[str] = []
    module_counts: Counter[str] = Counter()
    for i, raw_node in enumerate(nodes):
        node = _require_mapping(raw_node, f"nodes[{i}]", source)
        node_missing = {"id", "gold_label", "causal_description", "module"} - set(node)
        if node_missing:
            _fail(
                f"nodes[{i}] missing fields: {', '.join(sorted(node_missing))}",
                source,
            )
        node_id = _require_nonempty_string(node["id"], f"nodes[{i}].id", source)
        label = _require_nonempty_string(
            node["gold_label"], f"nodes[{i}].gold_label", source
        )
        _require_nonempty_string(
            node["causal_description"], f"nodes[{i}].causal_description", source
        )
        module = _require_nonempty_string(
            node["module"], f"nodes[{i}].module", source
        )
        if module not in modules:
            _fail(f"nodes[{i}].module {module!r} is not declared in modules", source)
        node_ids.append(node_id)
        labels.append(label.casefold())
        module_counts[module] += 1
    if tuple(node_ids) != EXPECTED_NODE_IDS:
        _fail(
            "node IDs and order must be exactly node_000 through node_019",
            source,
        )
    if len(set(labels)) != len(labels):
        _fail("gold_label values must be unique (case-insensitive)", source)
    if set(module_counts) != set(modules) or any(module_counts[m] == 0 for m in modules):
        _fail("every declared module must contain at least one node", source)

    topological_order = _require_list(
        spec["topological_order"], "topological_order", source
    )
    if len(topological_order) != len(node_ids):
        _fail("topological_order must contain every node exactly once", source)
    if len(set(topological_order)) != len(topological_order):
        _fail("topological_order contains duplicates", source)
    if set(topological_order) != set(node_ids):
        _fail("topological_order must be a permutation of node IDs", source)

    edges = _require_list(spec["edges"], "edges", source)
    if not MIN_EDGES <= len(edges) <= MAX_EDGES:
        _fail(f"edge count must be in [{MIN_EDGES}, {MAX_EDGES}]", source)
    node_set = set(node_ids)
    edge_pairs: list[tuple[str, str]] = []
    coefficients: list[float] = []
    for i, raw_edge in enumerate(edges):
        edge = _require_mapping(raw_edge, f"edges[{i}]", source)
        edge_missing = {"source", "target", "coefficient"} - set(edge)
        if edge_missing:
            _fail(
                f"edges[{i}] missing fields: {', '.join(sorted(edge_missing))}",
                source,
            )
        parent = _require_nonempty_string(
            edge["source"], f"edges[{i}].source", source
        )
        child = _require_nonempty_string(
            edge["target"], f"edges[{i}].target", source
        )
        if parent not in node_set or child not in node_set:
            _fail(f"edges[{i}] references an unknown node", source)
        if parent == child:
            _fail(f"edges[{i}] is a self-loop", source)
        coefficient = edge["coefficient"]
        if not _is_real_number(coefficient):
            _fail(f"edges[{i}].coefficient must be a finite real number", source)
        coefficient = float(coefficient)
        if not MIN_COEFFICIENT <= coefficient <= MAX_COEFFICIENT:
            _fail(
                f"edges[{i}].coefficient must be in "
                f"[{MIN_COEFFICIENT}, {MAX_COEFFICIENT}]",
                source,
            )
        if coefficient <= 0.0:
            _fail(f"edges[{i}].coefficient must be positive", source)
        edge_pairs.append((parent, child))
        coefficients.append(coefficient)
    if len(set(edge_pairs)) != len(edge_pairs):
        _fail("directed edges must be unique", source)

    _, _, _, parents, children = _graph_components(spec)
    indegrees = {node_id: len(parents[node_id]) for node_id in node_ids}
    outdegrees = {node_id: len(children[node_id]) for node_id in node_ids}
    if max(indegrees.values()) > MAX_INDEGREE:
        _fail(f"maximum indegree must not exceed {MAX_INDEGREE}", source)

    topo_position = {node_id: i for i, node_id in enumerate(topological_order)}
    invalid_topology = [
        (parent, child)
        for parent, child in edge_pairs
        if topo_position[parent] >= topo_position[child]
    ]
    if invalid_topology:
        _fail(
            f"topological_order violates edge {invalid_topology[0][0]} -> "
            f"{invalid_topology[0][1]}",
            source,
        )
    kahn_order = _kahn_topological_order(node_ids, children, parents)
    if len(kahn_order) != len(node_ids):
        _fail("edges do not form a DAG", source)

    chains = [
        (parent, middle, child)
        for parent, middle in edge_pairs
        for child in children[middle]
        if child != parent
    ]
    fork_nodes = [node_id for node_id in node_ids if outdegrees[node_id] >= 2]
    collider_nodes = [node_id for node_id in node_ids if indegrees[node_id] >= 2]
    mediator_nodes = [
        node_id
        for node_id in node_ids
        if indegrees[node_id] >= 1 and outdegrees[node_id] >= 1
    ]
    if not chains:
        _fail("graph must contain a directed chain", source)
    if not fork_nodes:
        _fail("graph must contain a fork", source)
    if not collider_nodes:
        _fail("graph must contain a collider", source)
    if not mediator_nodes:
        _fail("graph must contain a mediator", source)

    roots = [node_id for node_id in node_ids if indegrees[node_id] == 0]
    leaves = [node_id for node_id in node_ids if outdegrees[node_id] == 0]
    return {
        "graph_id": spec["graph_id"],
        "world": spec["world"],
        "nodes": len(node_ids),
        "edges": len(edges),
        "modules": len(modules),
        "module_sizes": {module: module_counts[module] for module in modules},
        "roots": roots,
        "leaves": leaves,
        "max_indegree": max(indegrees.values()),
        "max_outdegree": max(outdegrees.values()),
        "chain_count": len(chains),
        "fork_count": len(fork_nodes),
        "collider_count": len(collider_nodes),
        "mediator_count": len(mediator_nodes),
        "chain_example": list(chains[0]),
        "fork_nodes": fork_nodes,
        "collider_nodes": collider_nodes,
        "mediator_nodes": mediator_nodes,
        "coefficient_min": min(coefficients),
        "coefficient_max": max(coefficients),
        "is_dag": True,
    }


def graph_structure_stats(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return the report-ready structural statistics."""

    return validate_graph_spec(spec)


def load_graph_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load and strictly validate one graph fixture."""

    source = Path(path)
    spec = load_json(source)
    validate_graph_spec(spec, source)
    if source.stem != spec["world"]:
        _fail(
            f"filename stem {source.stem!r} must match world {spec['world']!r}",
            source,
        )
    return spec


def load_graph_specs(
    path_or_paths: str | os.PathLike[str] | Iterable[str | os.PathLike[str]],
    *,
    expected_count: int = 3,
) -> list[dict[str, Any]]:
    """Load the complete fixed three-graph collection.

    A directory is expanded as sorted ``*.json`` paths.  Collection-level IDs,
    worlds, graph seeds, and filenames must also be unique.
    """

    if isinstance(path_or_paths, (str, os.PathLike)):
        candidate = Path(path_or_paths)
        paths = sorted(candidate.glob("*.json")) if candidate.is_dir() else [candidate]
    else:
        paths = sorted((Path(item) for item in path_or_paths), key=lambda p: str(p))
    if len(paths) != expected_count:
        _fail(f"expected exactly {expected_count} graph JSON files, found {len(paths)}")
    specs = [load_graph_json(path) for path in paths]
    graph_ids = [spec["graph_id"] for spec in specs]
    worlds = [spec["world"] for spec in specs]
    graph_seeds = [spec["graph_seed"] for spec in specs]
    if len(set(graph_ids)) != len(graph_ids):
        _fail("graph_id values must be unique across graph fixtures")
    if len(set(worlds)) != len(worlds):
        _fail("world values must be unique across graph fixtures")
    if len(set(graph_seeds)) != len(graph_seeds):
        _fail("graph_seed values must be unique across graph fixtures")
    if expected_count == 3:
        if set(graph_ids) != set(EXPECTED_GRAPH_IDS):
            _fail(f"graph IDs must be exactly {list(EXPECTED_GRAPH_IDS)}")
        if set(worlds) != set(EXPECTED_WORLDS):
            _fail(f"worlds must be exactly {list(EXPECTED_WORLDS)}")
    return sorted(specs, key=lambda spec: spec["graph_id"])


def validate_folds(
    folds: Sequence[Sequence[str]],
    node_ids: Sequence[str] = EXPECTED_NODE_IDS,
    *,
    expected_folds: int = EXPECTED_FOLDS,
    masked_per_fold: int = MASKED_PER_FOLD,
) -> tuple[tuple[str, ...], ...]:
    """Validate the fixed 20%-by-five masking partition."""

    if isinstance(folds, (str, bytes)) or not isinstance(folds, Sequence):
        _fail("folds must be a sequence of fold sequences")
    nodes = tuple(node_ids)
    if len(nodes) != len(set(nodes)):
        _fail("node_ids must be unique")
    if len(folds) != expected_folds:
        _fail(f"expected {expected_folds} folds, found {len(folds)}")
    normalized: list[tuple[str, ...]] = []
    seen: list[str] = []
    node_set = set(nodes)
    for fold_index, fold in enumerate(folds):
        if isinstance(fold, (str, bytes)) or not isinstance(fold, Sequence):
            _fail(f"fold {fold_index} must be a sequence")
        current = tuple(fold)
        if len(current) != masked_per_fold:
            _fail(
                f"fold {fold_index} must mask exactly {masked_per_fold} nodes, "
                f"found {len(current)}"
            )
        if len(set(current)) != len(current):
            _fail(f"fold {fold_index} contains duplicate nodes")
        unknown = sorted(set(current) - node_set)
        if unknown:
            _fail(f"fold {fold_index} contains unknown nodes: {unknown}")
        normalized.append(current)
        seen.extend(current)
    counts = Counter(seen)
    missing = sorted(node_set - set(counts))
    repeated = sorted(node for node, count in counts.items() if count != 1)
    if missing or repeated or len(seen) != len(nodes):
        _fail(
            "folds must mask every node exactly once; "
            f"missing={missing}, non_unit_counts={repeated}"
        )
    return tuple(normalized)


def validate_fold_assignments(
    assignments: Mapping[str, Sequence[Sequence[str]]],
    graph_specs: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[tuple[str, ...], ...]]:
    """Validate graph-keyed folds against a loaded graph collection."""

    assignments = _require_mapping(assignments, "masking assignments")
    expected_ids = {spec["graph_id"] for spec in graph_specs}
    if set(assignments) != expected_ids:
        _fail(
            "masking assignment keys must equal graph IDs; "
            f"expected={sorted(expected_ids)}, got={sorted(assignments)}"
        )
    out: dict[str, tuple[tuple[str, ...], ...]] = {}
    for spec in sorted(graph_specs, key=lambda item: item["graph_id"]):
        node_ids = tuple(node["id"] for node in spec["nodes"])
        out[spec["graph_id"]] = validate_folds(assignments[spec["graph_id"]], node_ids)
    return out


def train_only_zscore(
    matrix: np.ndarray,
    train_size: int = DEFAULT_SPLIT_SIZES[0],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit population z-score statistics on train rows and transform all rows."""

    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        _fail("matrix must be a non-empty two-dimensional array")
    if not np.isfinite(values).all():
        _fail("matrix contains non-finite values")
    if not isinstance(train_size, int) or not 0 < train_size <= values.shape[0]:
        _fail("train_size must be a positive integer no larger than row count")
    train = values[:train_size]
    mean = train.mean(axis=0)
    scale = train.std(axis=0, ddof=0)
    if not np.isfinite(scale).all() or np.any(scale <= 0.0):
        _fail("at least one training column has zero or invalid variance")
    standardized = (values - mean) / scale
    return standardized, mean, scale


def generate_scm(
    graph_spec: Mapping[str, Any],
    *,
    data_seed: int,
    n_samples: int = sum(DEFAULT_SPLIT_SIZES),
    root_std: float = 1.0,
    noise_std: float = 0.65,
    split_sizes: Sequence[int] = DEFAULT_SPLIT_SIZES,
) -> dict[str, Any]:
    """Generate the fixed linear additive Gaussian SCM and train-fit z-scores."""

    validate_graph_spec(graph_spec)
    if not isinstance(data_seed, int) or isinstance(data_seed, bool):
        _fail("data_seed must be an integer")
    if not isinstance(n_samples, int) or n_samples <= 0:
        _fail("n_samples must be a positive integer")
    if not _is_real_number(root_std) or float(root_std) <= 0.0:
        _fail("root_std must be a positive finite number")
    if not _is_real_number(noise_std) or float(noise_std) <= 0.0:
        _fail("noise_std must be a positive finite number")
    if (
        isinstance(split_sizes, (str, bytes))
        or len(split_sizes) != 3
        or any(not isinstance(size, int) or size <= 0 for size in split_sizes)
    ):
        _fail("split_sizes must contain three positive integers")
    train_size, dev_size, test_size = (int(size) for size in split_sizes)
    if train_size + dev_size + test_size != n_samples:
        _fail("split_sizes must sum exactly to n_samples")

    node_ids = tuple(node["id"] for node in graph_spec["nodes"])
    node_index = {node_id: i for i, node_id in enumerate(node_ids)}
    topological_order = tuple(graph_spec["topological_order"])
    incoming: dict[str, list[tuple[str, float]]] = {node_id: [] for node_id in node_ids}
    true_weights: dict[tuple[str, str], float] = {}
    for edge in graph_spec["edges"]:
        pair = (edge["source"], edge["target"])
        coefficient = float(edge["coefficient"])
        incoming[edge["target"]].append((edge["source"], coefficient))
        true_weights[pair] = coefficient

    rng = np.random.default_rng(data_seed)
    raw = np.empty((n_samples, len(node_ids)), dtype=np.float64)
    for node_id in topological_order:
        column = node_index[node_id]
        parents = incoming[node_id]
        if not parents:
            raw[:, column] = rng.normal(0.0, float(root_std), size=n_samples)
            continue
        generated = np.zeros(n_samples, dtype=np.float64)
        for parent, coefficient in parents:
            generated += coefficient * raw[:, node_index[parent]]
        generated += rng.normal(0.0, float(noise_std), size=n_samples)
        raw[:, column] = generated
    if not np.isfinite(raw).all():
        _fail("SCM generation produced non-finite values")

    standardized, train_mean, train_std = train_only_zscore(raw, train_size)
    train_stop = train_size
    dev_stop = train_stop + dev_size
    split_indices = {
        "train": (0, train_stop),
        "dev": (train_stop, dev_stop),
        "test": (dev_stop, n_samples),
    }
    return {
        "node_ids": node_ids,
        "raw": raw,
        "standardized": standardized,
        "train": standardized[:train_stop],
        "dev": standardized[train_stop:dev_stop],
        "test": standardized[dev_stop:],
        "train_mean": train_mean,
        "train_std": train_std,
        "split_indices": split_indices,
        "data_seed": data_seed,
        "root_std": float(root_std),
        "noise_std": float(noise_std),
        "true_weights": true_weights,
    }


generate_linear_gaussian_scm = generate_scm


def adjacency_matrix(
    graph_spec: Mapping[str, Any],
    *,
    weighted: bool = False,
    validate_design_constraints: bool = True,
) -> np.ndarray:
    """Return source-by-target adjacency in the frozen node order.

    ``validate_design_constraints=False`` is reserved for derived arms such as
    the fully reversed graph: reversal preserves acyclicity but can turn an
    allowed oracle outdegree of four into an indegree of four.
    """

    if validate_design_constraints:
        validate_graph_spec(graph_spec)
    node_ids = [node["id"] for node in graph_spec["nodes"]]
    node_index = {node_id: i for i, node_id in enumerate(node_ids)}
    dtype = np.float64 if weighted else np.int8
    adjacency = np.zeros((len(node_ids), len(node_ids)), dtype=dtype)
    for edge in graph_spec["edges"]:
        value = float(edge["coefficient"]) if weighted else 1
        adjacency[node_index[edge["source"]], node_index[edge["target"]]] = value
    return adjacency


def validate_permutation(permutation: Sequence[int], n_nodes: int) -> np.ndarray:
    """Validate and return an old-index -> new-index permutation."""

    values = np.asarray(permutation)
    if values.ndim != 1 or len(values) != n_nodes:
        _fail(f"permutation must have shape ({n_nodes},)")
    if not np.issubdtype(values.dtype, np.integer):
        _fail("permutation entries must be integers")
    values = values.astype(np.int64, copy=False)
    if not np.array_equal(np.sort(values), np.arange(n_nodes, dtype=np.int64)):
        _fail("permutation must contain every index exactly once")
    return values.copy()


def permutation_matrix(permutation: Sequence[int]) -> np.ndarray:
    """Return P with ``P[old, new] = 1``."""

    values = validate_permutation(permutation, len(permutation))
    matrix = np.zeros((len(values), len(values)), dtype=np.int8)
    matrix[np.arange(len(values)), values] = 1
    return matrix


def generate_permutations(
    n_nodes: int,
    seeds: Sequence[int],
    *,
    expected_count: int | None = EXPECTED_PERMUTATIONS,
    require_non_identity: bool = True,
) -> list[np.ndarray]:
    """Generate deterministic old->new permutations from the frozen seeds."""

    if not isinstance(n_nodes, int) or n_nodes < 2:
        _fail("n_nodes must be an integer of at least two")
    if isinstance(seeds, (str, bytes)) or not isinstance(seeds, Sequence):
        _fail("seeds must be a sequence of integers")
    if expected_count is not None and len(seeds) != expected_count:
        _fail(f"expected {expected_count} permutation seeds, found {len(seeds)}")
    if any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds):
        _fail("every permutation seed must be an integer")
    if len(set(seeds)) != len(seeds):
        _fail("permutation seeds must be unique")
    identity = np.arange(n_nodes, dtype=np.int64)
    out: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()
    for seed in seeds:
        permutation = np.random.default_rng(seed).permutation(n_nodes).astype(np.int64)
        permutation = validate_permutation(permutation, n_nodes)
        key = tuple(int(item) for item in permutation)
        if require_non_identity and np.array_equal(permutation, identity):
            _fail(f"seed {seed} generated the identity permutation")
        if key in seen:
            _fail(f"seed {seed} generated a duplicate permutation")
        seen.add(key)
        out.append(permutation)
    return out


def permute_adjacency(adjacency: np.ndarray, permutation: Sequence[int]) -> np.ndarray:
    """Compute ``P.T @ A @ P`` for an old-index -> new-index permutation."""

    original = np.asarray(adjacency)
    if original.ndim != 2 or original.shape[0] != original.shape[1]:
        _fail("adjacency must be a square matrix")
    values = validate_permutation(permutation, original.shape[0])
    permuted = np.zeros_like(original)
    permuted[np.ix_(values, values)] = original
    return permuted


def validate_shuffled_adjacency(
    original: np.ndarray,
    shuffled: np.ndarray,
    permutation: Sequence[int],
) -> bool:
    """Strictly verify relabeling equality and directed degree preservation."""

    original = np.asarray(original)
    shuffled = np.asarray(shuffled)
    if original.shape != shuffled.shape:
        _fail("original and shuffled adjacency shapes differ")
    if original.ndim != 2 or original.shape[0] != original.shape[1]:
        _fail("adjacency matrices must be square")
    values = validate_permutation(permutation, original.shape[0])
    expected = permute_adjacency(original, values)
    if not np.array_equal(shuffled, expected):
        _fail("shuffled adjacency is not exactly P.T @ A @ P")
    old_binary = original != 0
    new_binary = shuffled != 0
    old_indegree = old_binary.sum(axis=0)
    old_outdegree = old_binary.sum(axis=1)
    new_indegree = new_binary.sum(axis=0)
    new_outdegree = new_binary.sum(axis=1)
    if not np.array_equal(new_indegree[values], old_indegree):
        _fail("nodewise indegree was not preserved under relabeling")
    if not np.array_equal(new_outdegree[values], old_outdegree):
        _fail("nodewise outdegree was not preserved under relabeling")
    if sorted(new_indegree.tolist()) != sorted(old_indegree.tolist()):
        _fail("indegree distribution was not preserved")
    if sorted(new_outdegree.tolist()) != sorted(old_outdegree.tolist()):
        _fail("outdegree distribution was not preserved")
    return True


def permute_graph(
    graph_spec: Mapping[str, Any],
    permutation: Sequence[int],
) -> dict[str, Any]:
    """Relabel graph support while leaving node/data/semantic identities fixed."""

    validate_graph_spec(graph_spec)
    node_ids = [node["id"] for node in graph_spec["nodes"]]
    node_index = {node_id: i for i, node_id in enumerate(node_ids)}
    values = validate_permutation(permutation, len(node_ids))

    def relabel(node_id: str) -> str:
        return node_ids[int(values[node_index[node_id]])]

    shuffled = copy.deepcopy(dict(graph_spec))
    shuffled["edges"] = [
        {
            **dict(edge),
            "source": relabel(edge["source"]),
            "target": relabel(edge["target"]),
        }
        for edge in graph_spec["edges"]
    ]
    shuffled["topological_order"] = [
        relabel(node_id) for node_id in graph_spec["topological_order"]
    ]
    validate_graph_spec(shuffled)
    validate_shuffled_adjacency(
        adjacency_matrix(graph_spec, weighted=True),
        adjacency_matrix(shuffled, weighted=True),
        values,
    )
    return shuffled


def reverse_graph(graph_spec: Mapping[str, Any]) -> dict[str, Any]:
    """Reverse every oracle edge, retaining its positive coefficient."""

    validate_graph_spec(graph_spec)
    reversed_spec = copy.deepcopy(dict(graph_spec))
    reversed_spec["edges"] = [
        {
            **dict(edge),
            "source": edge["target"],
            "target": edge["source"],
        }
        for edge in graph_spec["edges"]
    ]
    reversed_spec["topological_order"] = list(reversed(graph_spec["topological_order"]))
    original = adjacency_matrix(graph_spec, weighted=True)
    # Reversal can turn an allowed oracle outdegree into an indegree above the
    # fixture-only cap of three.  The reversed arm requires exact reversal and
    # acyclicity, not requalification as a new oracle fixture.
    topo_position = {
        node_id: i for i, node_id in enumerate(reversed_spec["topological_order"])
    }
    if any(
        topo_position[edge["source"]] >= topo_position[edge["target"]]
        for edge in reversed_spec["edges"]
    ):
        _fail("reversed graph is not a DAG")
    reversed_adjacency = adjacency_matrix(
        reversed_spec,
        weighted=True,
        validate_design_constraints=False,
    )
    if not np.array_equal(reversed_adjacency, original.T):
        _fail("reversed graph adjacency is not exactly A.T")
    return reversed_spec


def _normalized_rows(matrix: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        _fail(f"{name} must be a non-empty two-dimensional array")
    if not np.isfinite(values).all():
        _fail(f"{name} contains non-finite values")
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms <= 0.0):
        _fail(f"{name} contains a zero-norm row")
    return values / norms[:, None]


def rank_metrics(
    predictions: np.ndarray,
    candidates: np.ndarray,
    true_indices: Sequence[int],
) -> dict[str, Any]:
    """Per-node cosine/rank metrics against a fixed candidate embedding set."""

    predicted = _normalized_rows(predictions, "predictions")
    candidate = _normalized_rows(candidates, "candidates")
    if predicted.shape[1] != candidate.shape[1]:
        _fail("prediction and candidate embedding dimensions differ")
    truth = np.asarray(true_indices)
    if truth.ndim != 1 or len(truth) != len(predicted):
        _fail("true_indices must contain one index per prediction")
    if not np.issubdtype(truth.dtype, np.integer):
        _fail("true_indices must be integers")
    truth = truth.astype(np.int64, copy=False)
    if np.any(truth < 0) or np.any(truth >= len(candidate)):
        _fail("true_indices contains an out-of-range candidate index")

    similarities = predicted @ candidate.T
    order = np.argsort(-similarities, axis=1, kind="stable")
    ranks = np.empty(len(predicted), dtype=np.int64)
    for row, true_index in enumerate(truth):
        ranks[row] = int(np.flatnonzero(order[row] == true_index)[0]) + 1
    cosine = similarities[np.arange(len(predicted)), truth]
    reciprocal_rank = 1.0 / ranks.astype(np.float64)
    recall_at_1 = (ranks <= 1).astype(np.int8)
    recall_at_5 = (ranks <= 5).astype(np.int8)
    exact_decode = recall_at_1.copy()
    summary = {
        "gold_embedding_cosine": float(cosine.mean()),
        "mrr": float(reciprocal_rank.mean()),
        "recall_at_1": float(recall_at_1.mean()),
        "recall_at_5": float(recall_at_5.mean()),
        "exact_decode": float(exact_decode.mean()),
    }
    return {
        "similarities": similarities,
        "top_candidate_index": order[:, 0].copy(),
        "gold_embedding_cosine": cosine,
        "cosine": cosine,
        "rank": ranks,
        "reciprocal_rank": reciprocal_rank,
        "recall_at_1": recall_at_1,
        "recall_at_5": recall_at_5,
        "exact_decode": exact_decode,
        "mean_gold_embedding_cosine": summary["gold_embedding_cosine"],
        "mrr": summary["mrr"],
        "mean_recall_at_1": summary["recall_at_1"],
        "mean_recall_at_5": summary["recall_at_5"],
        "mean_exact_decode": summary["exact_decode"],
        "summary": summary,
    }


embedding_rank_metrics = rank_metrics


def hungarian_match(
    predictions: np.ndarray,
    true_embeddings: np.ndarray,
) -> dict[str, Any]:
    """Optimal one-to-one masked-node matching with per-prediction hit flags."""

    predicted = _normalized_rows(predictions, "predictions")
    truth = _normalized_rows(true_embeddings, "true_embeddings")
    if predicted.shape != truth.shape:
        _fail("predictions and true_embeddings must have the same shape")
    from scipy.optimize import linear_sum_assignment

    similarities = predicted @ truth.T
    row_ind, column_ind = linear_sum_assignment(-similarities)
    assignment = np.empty(len(predicted), dtype=np.int64)
    assignment[row_ind] = column_ind
    hits = (assignment == np.arange(len(predicted))).astype(np.int8)
    return {
        "similarities": similarities,
        "row_ind": row_ind,
        "column_ind": column_ind,
        "assignment": assignment,
        "hits": hits,
        "match_acc": float(hits.mean()),
    }


def hungarian_match_hits(
    predictions: np.ndarray,
    true_embeddings: np.ndarray,
) -> np.ndarray:
    """Return one 0/1 Hungarian identity hit for each masked node."""

    return hungarian_match(predictions, true_embeddings)["hits"]


def _record_field(
    record: Mapping[str, Any],
    primary: str,
    aliases: Sequence[str],
) -> Any:
    if primary in record:
        return record[primary]
    for alias in aliases:
        if alias in record:
            return record[alias]
    raise ValidationError(
        f"bootstrap record is missing {primary!r} "
        f"(accepted aliases: {', '.join(repr(alias) for alias in aliases)})"
    )


def hierarchical_bootstrap(
    records: Iterable[Mapping[str, Any]],
    *,
    value_key: str = "delta",
    graph_key: str = "graph_id",
    fold_key: str = "fold",
    node_key: str = "node_id",
    draws: int = 10_000,
    seed: int = 88_173,
    confidence: float = 0.95,
    return_draws: bool = False,
) -> dict[str, Any]:
    """Paired hierarchical bootstrap in graph -> fold -> masked-node order."""

    if not isinstance(draws, int) or draws <= 0:
        _fail("draws must be a positive integer")
    if not isinstance(seed, int) or isinstance(seed, bool):
        _fail("seed must be an integer")
    if not _is_real_number(confidence) or not 0.0 < float(confidence) < 1.0:
        _fail("confidence must be strictly between zero and one")

    parsed: list[tuple[Any, Any, Any, float]] = []
    seen: set[tuple[Any, Any, Any]] = set()
    for index, raw_record in enumerate(records):
        record = _require_mapping(raw_record, f"records[{index}]")
        graph = _record_field(record, graph_key, ("graph",))
        fold = _record_field(record, fold_key, ("fold_id",))
        node = _record_field(record, node_key, ("masked_node", "node", "var"))
        value = _record_field(record, value_key, ("value",))
        if not _is_real_number(value):
            _fail(f"records[{index}] bootstrap value must be finite")
        identity = (graph, fold, node)
        try:
            duplicate = identity in seen
        except TypeError as exc:
            raise ValidationError(
                f"records[{index}] hierarchy keys must be hashable"
            ) from exc
        if duplicate:
            _fail(f"duplicate bootstrap leaf {identity!r}")
        seen.add(identity)
        parsed.append((graph, fold, node, float(value)))
    if not parsed:
        _fail("records must contain at least one paired delta")

    parsed.sort(key=lambda row: tuple(_stable_key(item) for item in row[:3]))
    hierarchy: dict[Any, dict[Any, np.ndarray]] = {}
    grouped: dict[Any, dict[Any, list[tuple[Any, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for graph, fold, node, value in parsed:
        grouped[graph][fold].append((node, value))
    for graph in sorted(grouped, key=_stable_key):
        hierarchy[graph] = {}
        for fold in sorted(grouped[graph], key=_stable_key):
            leaves = sorted(grouped[graph][fold], key=lambda item: _stable_key(item[0]))
            hierarchy[graph][fold] = np.asarray(
                [value for _, value in leaves], dtype=np.float64
            )

    graphs = list(hierarchy)
    rng = np.random.default_rng(seed)
    bootstrap_draws = np.empty(draws, dtype=np.float64)
    for draw_index in range(draws):
        sampled_values: list[float] = []
        for graph_index in rng.integers(0, len(graphs), size=len(graphs)):
            graph = graphs[int(graph_index)]
            folds = list(hierarchy[graph])
            for fold_index in rng.integers(0, len(folds), size=len(folds)):
                fold = folds[int(fold_index)]
                leaves = hierarchy[graph][fold]
                sampled_leaf_indices = rng.integers(0, len(leaves), size=len(leaves))
                sampled_values.extend(leaves[sampled_leaf_indices].tolist())
        bootstrap_draws[draw_index] = float(np.mean(sampled_values))

    values = np.asarray([row[3] for row in parsed], dtype=np.float64)
    tail = (1.0 - float(confidence)) / 2.0
    low, high = np.quantile(bootstrap_draws, [tail, 1.0 - tail])
    result: dict[str, Any] = {
        "mean": float(values.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "confidence_level": float(confidence),
        "paired_win_rate": float(np.mean(values > 0.0)),
        "bootstrap_positive_rate": float(np.mean(bootstrap_draws > 0.0)),
        "draws": draws,
        "seed": seed,
        "n_graphs": len(graphs),
        "n_folds": sum(len(folds) for folds in hierarchy.values()),
        "n_nodes": len(values),
    }
    if return_draws:
        result["bootstrap_draws"] = bootstrap_draws
    return result


def bootstrap_by_graph(
    records: Iterable[Mapping[str, Any]],
    *,
    graph_key: str = "graph_id",
    seed: int = 88_173,
    **kwargs: Any,
) -> dict[Any, dict[str, Any]]:
    """Run the fold -> node bootstrap independently for each graph."""

    materialized = list(records)
    grouped: dict[Any, list[Mapping[str, Any]]] = defaultdict(list)
    for record in materialized:
        graph = _record_field(record, graph_key, ("graph",))
        grouped[graph].append(record)
    out: dict[Any, dict[str, Any]] = {}
    for offset, graph in enumerate(sorted(grouped, key=_stable_key)):
        out[graph] = hierarchical_bootstrap(
            grouped[graph],
            graph_key=graph_key,
            seed=seed + offset,
            **kwargs,
        )
    return out


__all__ = [
    "DEFAULT_SPLIT_SIZES",
    "EXPECTED_FOLDS",
    "EXPECTED_GRAPH_IDS",
    "EXPECTED_MODULES",
    "EXPECTED_NODE_IDS",
    "EXPECTED_PERMUTATIONS",
    "EXPECTED_WORLDS",
    "GRAPH_SCHEMA_VERSION",
    "MASKED_PER_FOLD",
    "ValidationError",
    "adjacency_matrix",
    "bootstrap_by_graph",
    "canonical_json_dumps",
    "dump_json",
    "embedding_rank_metrics",
    "generate_linear_gaussian_scm",
    "generate_permutations",
    "generate_scm",
    "graph_structure_stats",
    "hierarchical_bootstrap",
    "hungarian_match",
    "hungarian_match_hits",
    "json_sha256",
    "load_graph_json",
    "load_graph_specs",
    "load_json",
    "permutation_matrix",
    "permute_adjacency",
    "permute_graph",
    "rank_metrics",
    "reverse_graph",
    "sha256_file",
    "sha256_json",
    "train_only_zscore",
    "validate_fold_assignments",
    "validate_folds",
    "validate_graph_spec",
    "validate_permutation",
    "validate_shuffled_adjacency",
    "write_json",
]
