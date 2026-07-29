#!/usr/bin/env python3
"""Run Task 3 E0-prime without importing the side-effectful ``v5/main.py``."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import shlex
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCRIPT_PATH = Path(__file__).resolve()
TASK_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[2]
V5_ROOT = REPO_ROOT / "v5"
DEFAULT_CONFIG = TASK_ROOT / "experiments" / "e0_oracle_bridge" / "config.yaml"
HF_CACHE = REPO_ROOT.parent / ".hf_cache"
E5_REVISION = "f169b11e22de13617baa190a028a32f3493550b6"
E5_SNAPSHOT = HF_CACHE / "models--intfloat--e5-large-v2" / "snapshots" / E5_REVISION

# Set offline/cache state before sentence-transformers, transformers, or torch is imported.
os.environ["HF_CACHE"] = str(HF_CACHE)
os.environ.setdefault("HF_HOME", str(HF_CACHE))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("JUDGE_MODEL", "gpt-5.5")

import numpy as np
import yaml

if str(SCRIPT_PATH.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PATH.parent))
import e0_core as core


EXPECTED_ARTIFACTS = {
    "lora": (
        "v5/outputs/l3_lora.pt",
        "d90b024e7fb030e3ee1545c8d19606cf032cf810fc7f4a758749dfefc95a49d5",
    ),
    "dictionary": (
        "v5/outputs/concept_bank_l3.npz",
        "6da2de255dcb2fa559fa1c2a8bfba25fd0e4fcfecb5c768ecf229b6ce4e7bb9e",
    ),
    "negop": (
        "v5/outputs/negop.pt",
        "6f30f0d68ee653d52aef93bdae97feeac8abd17189e688ef49f594e18574690e",
    ),
    "weightnet": (
        "v5/outputs/l2_mlp.pt",
        "70ffc4fcf668b57d943240fde67a8b339c702fa0093a5594f34e3297a5c50bfd",
    ),
}
REQUIRED_ARMS = (
    "core_oracle_estimated_weights",
    "core_oracle_true_weights",
    "core_shuffled_graph",
    "core_reversed_graph",
    "raw_correlation",
    "uniform",
)
LOCAL_METRICS = (
    "match_acc",
    "gold_embedding_cosine",
    "mrr",
    "recall_at_1",
    "recall_at_5",
    "exact_decode",
)
COMPARATORS = {
    "oracle_vs_shuffle": "core_shuffled_graph",
    "oracle_vs_reverse": "core_reversed_graph",
    "oracle_vs_no_graph": "uniform",
    "oracle_vs_rawcorr": "raw_correlation",
}
PRIMARY_ARM = "core_oracle_estimated_weights"
PRIMARY_METRIC = "gold_embedding_cosine"


class ValidationError(RuntimeError):
    """Raised when a frozen preregistration or artifact invariant is violated."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, obj: Any, *, indent: int = 2) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        _jsonable(obj), ensure_ascii=False, allow_nan=False, indent=indent, sort_keys=True
    )
    path.write_text(text + "\n", encoding="utf-8", newline="\n")
    return path


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    _jsonable(dict(row)), ensure_ascii=False, allow_nan=False, sort_keys=True
                )
                + "\n"
            )
    return path


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for source in rows:
            row: dict[str, Any] = {}
            for field in fields:
                value = source.get(field, "")
                if value is None or (isinstance(value, float) and not math.isfinite(value)):
                    value = ""
                elif isinstance(value, bool):
                    value = int(value)
                elif isinstance(value, (list, tuple, dict)):
                    value = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)
                row[field] = value
            writer.writerow(row)
    return path


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(obj: Any) -> str:
    payload = json.dumps(
        _jsonable(obj), ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValidationError(f"{label}: expected {expected!r}, found {actual!r}")


def _resolve_repo_path(value: str, label: str) -> Path:
    path = (REPO_ROOT / value).resolve()
    try:
        path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ValidationError(f"{label} escapes repository root: {value}") from exc
    return path


def load_config(path: Path) -> dict[str, Any]:
    path = path.resolve()
    _require(path.is_file(), f"config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    _require(isinstance(data, dict), "config must decode to a mapping")
    return data


def _artifact_entries(config: Mapping[str, Any]) -> dict[str, tuple[str, str]]:
    frozen = config["frozen_stage3"]
    return {
        "lora": (frozen["calibrated_space"]["checkpoint"], frozen["calibrated_space"]["sha256"]),
        "dictionary": (frozen["decode_dictionary"]["path"], frozen["decode_dictionary"]["sha256"]),
        "negop": (frozen["semantic_negation"]["checkpoint"], frozen["semantic_negation"]["sha256"]),
        "weightnet": (frozen["solver"]["checkpoint"], frozen["solver"]["sha256"]),
    }


def _graph_nodes(spec: Mapping[str, Any]) -> list[str]:
    return [str(node["id"]) for node in spec["nodes"]]


def validate_graph_spec(spec: Mapping[str, Any], source: Path) -> dict[str, Any]:
    _require(isinstance(spec, Mapping), f"{source}: graph must be an object")
    for key in ("graph_id", "world", "nodes", "edges", "topological_order", "modules"):
        _require(key in spec, f"{source}: missing {key}")
    nodes = _graph_nodes(spec)
    _require_equal(len(nodes), 20, f"{source}: node count")
    _require_equal(len(set(nodes)), 20, f"{source}: unique node count")
    order = [str(x) for x in spec["topological_order"]]
    _require_equal(set(order), set(nodes), f"{source}: topological node set")
    _require_equal(len(order), len(nodes), f"{source}: topological order length")
    pos = {node: i for i, node in enumerate(order)}
    seen_edges: set[tuple[str, str]] = set()
    indegree = {node: 0 for node in nodes}
    outdegree = {node: 0 for node in nodes}
    coefficients: list[float] = []
    for index, edge in enumerate(spec["edges"]):
        a, b = str(edge["source"]), str(edge["target"])
        coeff = float(edge["coefficient"])
        _require(a in pos and b in pos, f"{source}: edge {index} has unknown endpoint")
        _require(a != b, f"{source}: self-loop {a}->{b}")
        _require(pos[a] < pos[b], f"{source}: edge {a}->{b} violates topological order")
        _require((a, b) not in seen_edges, f"{source}: duplicate edge {a}->{b}")
        _require(0.4 <= coeff <= 0.9, f"{source}: coefficient out of bounds on {a}->{b}")
        _require(coeff > 0, f"{source}: non-positive coefficient on {a}->{b}")
        seen_edges.add((a, b))
        indegree[b] += 1
        outdegree[a] += 1
        coefficients.append(coeff)
    _require(max(indegree.values()) <= 3, f"{source}: maximum indegree exceeds three")
    _require_equal(len(spec["modules"]), 4, f"{source}: module count")
    node_modules = {str(node["module"]) for node in spec["nodes"]}
    module_ids = {
        str(module["id"]) if isinstance(module, Mapping) else str(module)
        for module in spec["modules"]
    }
    _require_equal(node_modules, module_ids, f"{source}: referenced module IDs")
    _require(bool(spec.get("observed_only")), f"{source}: observed_only must be true")
    _require_equal(spec.get("hidden_confounders"), False, f"{source}: hidden_confounders")
    for node in spec["nodes"]:
        _require(str(node.get("gold_label", "")).strip() != "", f"{source}: empty gold label")
        _require(
            str(node.get("causal_description", "")).strip() != "",
            f"{source}: empty causal description",
        )
    return {
        "node_count": len(nodes),
        "edge_count": len(seen_edges),
        "root_count": sum(value == 0 for value in indegree.values()),
        "max_indegree": max(indegree.values()),
        "max_outdegree": max(outdegree.values()),
        "coefficient_min": min(coefficients),
        "coefficient_max": max(coefficients),
    }


def load_graph_specs(config: Mapping[str, Any], config_path: Path) -> list[dict[str, Any]]:
    entries = config["graphs"]
    _require_equal(len(entries), 3, "number of graph entries")
    specs: list[dict[str, Any]] = []
    graph_ids: set[str] = set()
    for entry in entries:
        spec_path = (config_path.parent / str(entry["spec"])).resolve()
        _require(spec_path.is_file(), f"graph spec not found: {spec_path}")
        with spec_path.open("r", encoding="utf-8") as handle:
            spec = json.load(handle)
        _require_equal(spec["graph_id"], entry["graph_id"], f"{spec_path}: graph_id")
        _require_equal(spec["world"], entry["world"], f"{spec_path}: world")
        _require_equal(int(spec["graph_seed"]), int(entry["graph_seed"]), f"{spec_path}: graph_seed")
        spec["_source_path"] = str(spec_path)
        spec["_source_sha256"] = sha256_file(spec_path)
        spec["_data_seed"] = int(entry["data_seed"])
        spec["_stats"] = core.validate_graph_spec(spec, spec_path)
        validate_graph_spec(spec, spec_path)
        _require(spec["graph_id"] not in graph_ids, f"duplicate graph_id {spec['graph_id']}")
        graph_ids.add(spec["graph_id"])
        specs.append(spec)
    return specs


def validate_config(
    config: Mapping[str, Any], config_path: Path, *, hash_artifacts: bool = True
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Strictly validate every frozen protocol value and artifact identity."""
    _require_equal(config.get("schema_version"), "task3.e0_prime.v1", "schema_version")
    _require_equal(config.get("status"), "frozen_before_execution", "status")
    frozen = config["frozen_stage3"]
    _require_equal(frozen["source_entrypoint"], "v5/main.py", "formal source entrypoint")
    _require_equal(frozen["adapter_entrypoint"], "task3_v2/scripts/run_e0_bridge.py", "adapter")
    _require_equal(frozen["semantic_encoder"], "intfloat/e5-large-v2", "semantic encoder")
    calibrated = frozen["calibrated_space"]
    _require_equal(
        (calibrated["rank"], calibrated["alpha"], calibrated["wrapped_last_layers"]),
        (8, 16, 2),
        "LoRA configuration",
    )
    solver = frozen["solver"]
    _require_equal(solver["module"], "WeightNet", "solver module")
    _require_equal(solver["unroll_steps"], 60, "solver K")
    _require_equal(float(solver["inner_learning_rate"]), 0.02, "inner learning rate")
    objective = frozen["objective"]
    for key, value in (
        ("lam_zero", 0.3),
        ("lam_norm", 0.1),
        ("residual", 1.0),
        ("lam_res", 1.0),
    ):
        _require_equal(float(objective[key]), value, f"objective.{key}")
    _require_equal(float(objective["lam_dep"]), 0.0, "objective.lam_dep")
    _require_equal(float(objective["lam_coll"]), 0.0, "objective.lam_coll")
    bridge = objective["bridge"]
    for key, value in (("lam_upper", 0.3), ("kappa", 0.5), ("q", 0.7)):
        _require_equal(float(bridge[key]), value, f"bridge.{key}")
    _require_equal(bridge["measure"], "pearson", "bridge measure")
    _require_equal(
        bridge["dependence_matrix"],
        "train_dev_absolute_pearson_adapter",
        "bridge dependence adapter",
    )
    _require_equal(bridge["test_used_for_estimation"], False, "bridge test leakage flag")
    _require_equal(objective["negation_operator"], "frozen_f_neg", "negation operator")
    _require_equal(objective["generative_operator"], None, "generative operator")
    _require_equal(objective["latent_constraints_observed_only_effect"], "no_op", "latent no-op")
    _require_equal(frozen["training"]["retrain_in_e0_prime"], False, "retrain flag")
    _require_equal(
        frozen["training"]["artifact_training_commit"],
        "70efde7ed488229667ae7958237116c7bdb40e45",
        "artifact training commit",
    )

    scm = config["scm"]
    _require_equal(scm["samples_per_graph"], 2000, "SCM sample count")
    _require_equal(float(scm["root_distribution"]["standard_deviation"]), 1.0, "root SD")
    _require_equal(float(scm["non_root_noise"]["standard_deviation"]), 0.65, "noise SD")
    _require_equal(
        scm["split"],
        {"train": 1200, "dev": 400, "test": 400, "shuffle_before_split": False},
        "SCM split",
    )
    _require_equal(scm["standardization"]["fit_split"], "train", "z-score fit split")
    _require_equal(
        scm["edge_weight_estimation"]["fit_splits"], ["train", "dev"], "edge fit splits"
    )
    _require_equal(
        scm["edge_weight_estimation"]["test_used_for_selection"],
        False,
        "edge test leakage flag",
    )
    masking = config["masking"]
    _require_equal(
        (masking["folds"], masking["masked_nodes_per_fold"]), (5, 4), "masking dimensions"
    )
    _require_equal(tuple(config["arms"]), REQUIRED_ARMS, "arm order")
    null = config["shuffle_null"]
    _require_equal(null["permutations_per_graph"], 20, "shuffle permutations")
    evaluation = config["evaluation"]
    _require_equal(evaluation["decoder_screen_k"], 2000, "decoder screen_k")
    _require_equal(evaluation["decoder_top_k"], 6, "decoder top_k")
    _require_equal(evaluation["decoder_target_l0"], 8, "decoder target_l0")
    _require_equal(
        evaluation["decoder_alpha_fit"], "visible_labels_only_per_fold", "decoder alpha fit"
    )
    _require_equal(evaluation["judge"]["mode"], "requests_only", "judge mode")
    _require_equal(evaluation["judge"]["model"], "gpt-5.5", "judge model")
    _require_equal(config["bootstrap"]["draws"], 10000, "bootstrap draws")
    _require_equal(config["bootstrap"]["seed"], 88173, "bootstrap seed")

    specs = load_graph_specs(config, config_path)
    assignments = masking["assignments"]
    seeds_by_graph = null["permutation_seeds"]
    for spec in specs:
        graph_id = spec["graph_id"]
        nodes = _graph_nodes(spec)
        folds = assignments[graph_id]
        _require_equal(len(folds), 5, f"{graph_id}: fold count")
        flat: list[str] = []
        for fold_index, fold in enumerate(folds):
            _require_equal(len(fold), 4, f"{graph_id}: fold {fold_index} size")
            _require_equal(len(set(fold)), 4, f"{graph_id}: fold {fold_index} unique nodes")
            _require(set(fold) <= set(nodes), f"{graph_id}: fold {fold_index} has unknown node")
            flat.extend(str(node) for node in fold)
        _require_equal(sorted(flat), sorted(nodes), f"{graph_id}: folds must partition nodes")
        seeds = [int(seed) for seed in seeds_by_graph[graph_id]]
        _require_equal(len(seeds), 20, f"{graph_id}: shuffle seed count")
        _require_equal(len(set(seeds)), 20, f"{graph_id}: unique shuffle seeds")

    artifact_report: dict[str, Any] = {}
    entries = _artifact_entries(config)
    for name, expected in EXPECTED_ARTIFACTS.items():
        configured_path, configured_sha = entries[name]
        _require_equal(
            (configured_path, configured_sha.lower()), expected, f"{name} artifact declaration"
        )
        path = _resolve_repo_path(configured_path, f"{name} artifact")
        _require(path.is_file(), f"{name} artifact missing: {path}")
        actual_sha = sha256_file(path) if hash_artifacts else None
        if hash_artifacts:
            _require_equal(actual_sha, configured_sha.lower(), f"{name} artifact SHA-256")
        artifact_report[name] = {
            "path": str(path),
            "sha256": actual_sha or configured_sha.lower(),
            "size_bytes": path.stat().st_size,
            "mtime": path.stat().st_mtime,
        }

    dictionary_path = Path(artifact_report["dictionary"]["path"])
    with np.load(dictionary_path, allow_pickle=True) as dictionary:
        _require("emb" in dictionary and "names" in dictionary, "dictionary lacks emb/names")
        dictionary_shape = tuple(int(x) for x in dictionary["emb"].shape)
        _require_equal(dictionary_shape[1], 1024, "dictionary embedding width")
        _require_equal(dictionary_shape[0], len(dictionary["names"]), "dictionary names")
        _require("lora_version" in dictionary, "dictionary lacks lora_version")
        dict_lora_version = float(np.asarray(dictionary["lora_version"]).reshape(-1)[0])
    lora_mtime = float(artifact_report["lora"]["mtime"])
    _require(
        abs(dict_lora_version - lora_mtime) < 1.0,
        f"dictionary lora_version {dict_lora_version} does not match LoRA mtime {lora_mtime}",
    )
    artifact_report["dictionary"]["lora_version"] = dict_lora_version
    artifact_report["dictionary"]["embedding_shape"] = list(dictionary_shape)
    _require(E5_SNAPSHOT.is_dir(), f"frozen e5 snapshot missing: {E5_SNAPSHOT}")
    artifact_report["encoder"] = {
        "model": "intfloat/e5-large-v2",
        "revision": E5_REVISION,
        "snapshot": str(E5_SNAPSHOT),
        "offline": True,
    }
    return specs, artifact_report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate config, graphs, artifact hashes, and versions without loading models",
    )
    parser.add_argument(
        "--skip-decode",
        action="store_true",
        help="debug only: skip the frozen dictionary/SpLiCE decoding stage",
    )
    return parser.parse_args(argv)


def _clean_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in spec.items() if not str(key).startswith("_")}


def _edge_pairs(spec: Mapping[str, Any]) -> list[tuple[str, str]]:
    return [(str(edge["source"]), str(edge["target"])) for edge in spec["edges"]]


def _parent_child_maps(spec: Mapping[str, Any]) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    nodes = _graph_nodes(spec)
    parents = {node: set() for node in nodes}
    children = {node: set() for node in nodes}
    for parent, child in _edge_pairs(spec):
        parents[child].add(parent)
        children[parent].add(child)
    return parents, children


def _scm_qa(scm_data: Mapping[str, Any], spec: Mapping[str, Any]) -> dict[str, Any]:
    raw = np.asarray(scm_data["raw"], dtype=np.float64)
    standardized = np.asarray(scm_data["standardized"], dtype=np.float64)
    train = np.asarray(scm_data["train"], dtype=np.float64)
    _require_equal(raw.shape, (2000, 20), f"{spec['graph_id']}: raw SCM shape")
    _require_equal(standardized.shape, raw.shape, f"{spec['graph_id']}: z-scored SCM shape")
    _require(np.isfinite(raw).all(), f"{spec['graph_id']}: non-finite raw SCM values")
    _require(np.isfinite(standardized).all(), f"{spec['graph_id']}: non-finite standardized values")
    mean_error = float(np.max(np.abs(train.mean(axis=0))))
    std_error = float(np.max(np.abs(train.std(axis=0) - 1.0)))
    _require(mean_error < 1e-10, f"{spec['graph_id']}: train z-score mean error {mean_error}")
    _require(std_error < 1e-8, f"{spec['graph_id']}: train z-score SD error {std_error}")
    index = {node: i for i, node in enumerate(scm_data["node_ids"])}
    edge_correlations = [
        float(np.corrcoef(raw[:, index[parent]], raw[:, index[child]])[0, 1])
        for parent, child in _edge_pairs(spec)
    ]
    return {
        "finite_raw": True,
        "finite_standardized": True,
        "train_zscore_max_abs_mean": mean_error,
        "train_zscore_max_abs_std_error": std_error,
        "raw_column_std_min": float(raw.std(axis=0).min()),
        "raw_column_std_max": float(raw.std(axis=0).max()),
        "edge_parent_child_corr_min": min(edge_correlations),
        "edge_parent_child_corr_max": max(edge_correlations),
        "all_edge_parent_child_corr_positive": all(value > 0.0 for value in edge_correlations),
        "test_rows_excluded_from_estimation": True,
    }


def generate_datasets(
    config: Mapping[str, Any],
    config_path: Path,
    specs: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    experiment_dir = config_path.parent
    data_dir = experiment_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    scm_cfg = config["scm"]
    split = scm_cfg["split"]
    split_sizes = (int(split["train"]), int(split["dev"]), int(split["test"]))
    datasets: dict[str, dict[str, Any]] = {}
    manifest_graphs: list[dict[str, Any]] = []
    for spec_with_meta in specs:
        spec = _clean_spec(spec_with_meta)
        graph_id = str(spec["graph_id"])
        scm_data = core.generate_scm(
            spec,
            data_seed=int(spec_with_meta["_data_seed"]),
            n_samples=int(scm_cfg["samples_per_graph"]),
            root_std=float(scm_cfg["root_distribution"]["standard_deviation"]),
            noise_std=float(scm_cfg["non_root_noise"]["standard_deviation"]),
            split_sizes=split_sizes,
        )
        qa = _scm_qa(scm_data, spec)
        true_weight_matrix = core.adjacency_matrix(spec, weighted=True)
        metadata = {
            "schema_version": "task3.e0_prime.scm.v1",
            "graph_id": graph_id,
            "world": spec["world"],
            "data_seed": int(spec_with_meta["_data_seed"]),
            "root_std": float(scm_data["root_std"]),
            "noise_std": float(scm_data["noise_std"]),
            "split_indices": scm_data["split_indices"],
            "standardization_fit_split": "train",
            "test_used_for_estimation": False,
        }
        data_path = data_dir / f"{graph_id}.npz"
        np.savez_compressed(
            data_path,
            node_ids=np.asarray(scm_data["node_ids"], dtype="<U16"),
            raw=np.asarray(scm_data["raw"], dtype=np.float64),
            standardized=np.asarray(scm_data["standardized"], dtype=np.float64),
            train=np.asarray(scm_data["train"], dtype=np.float64),
            dev=np.asarray(scm_data["dev"], dtype=np.float64),
            test=np.asarray(scm_data["test"], dtype=np.float64),
            train_mean=np.asarray(scm_data["train_mean"], dtype=np.float64),
            train_std=np.asarray(scm_data["train_std"], dtype=np.float64),
            true_weight_matrix=np.asarray(true_weight_matrix, dtype=np.float64),
            metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True)),
        )
        data_sha = sha256_file(data_path)
        datasets[graph_id] = dict(scm_data)
        datasets[graph_id]["spec"] = spec
        manifest_graphs.append(
            {
                **metadata,
                "spec_path": str(Path(spec_with_meta["_source_path"]).relative_to(REPO_ROOT)),
                "spec_sha256": spec_with_meta["_source_sha256"],
                "data_path": str(data_path.relative_to(REPO_ROOT)),
                "data_sha256": data_sha,
                "data_size_bytes": data_path.stat().st_size,
                "shape": [2000, 20],
                "qa": qa,
            }
        )
        print(f"[{time.strftime('%H:%M:%S')}] generated {graph_id}: {data_path.name}", flush=True)
    manifest = {
        "schema_version": "task3.e0_prime.data_manifest.v1",
        "created_at_utc": _utc_now(),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "scm": scm_cfg,
        "graphs": manifest_graphs,
        "qa_passed": True,
    }
    write_json(experiment_dir / "data_manifest.json", manifest)
    return datasets, manifest


def _import_v5(config: Mapping[str, Any]) -> dict[str, Any]:
    """Load only frozen library modules; deliberately never import v5/main.py."""
    if str(V5_ROOT) not in sys.path:
        sys.path.insert(0, str(V5_ROOT))
    negop_path = _resolve_repo_path(
        config["frozen_stage3"]["semantic_negation"]["checkpoint"], "negop checkpoint"
    )
    os.environ["NEGOP_CKPT"] = str(negop_path)
    names = ("graph", "optimize", "lora", "l2_modules", "l2_solver", "negop")
    return {name: importlib.import_module(name) for name in names}


def load_frozen_runtime(config: Mapping[str, Any]) -> dict[str, Any]:
    modules = _import_v5(config)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("E0-prime requires CUDA for the frozen e5+LoRA encoder")
    torch.set_num_threads(int(config["execution"]["torch_threads"]))
    seed = int(config["execution"]["deterministic_seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

    lora_mod = modules["lora"]
    encoder = lora_mod.load_st(device="cuda")
    lora_mod.inject(encoder)
    lora_mod.load_lora(
        encoder,
        str(_resolve_repo_path(config["frozen_stage3"]["calibrated_space"]["checkpoint"], "LoRA")),
    )
    encoder.eval()
    weightnet = modules["l2_modules"].load(
        str(_resolve_repo_path(config["frozen_stage3"]["solver"]["checkpoint"], "WeightNet")),
        device="cpu",
    )
    weightnet.eval()
    negation = modules["negop"].load().to("cpu").eval()
    for parameter in weightnet.parameters():
        parameter.requires_grad_(False)
    for parameter in negation.parameters():
        parameter.requires_grad_(False)
    return {
        **modules,
        "torch": torch,
        "encoder": encoder,
        "weightnet": weightnet,
        "negation": negation,
        "encoder_device": "cuda",
        "solver_device": "cpu",
    }


def encode_texts(runtime: Mapping[str, Any], texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
    torch = runtime["torch"]
    batches: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            encoded = runtime["lora"].encode_grad(
                runtime["encoder"], list(texts[start : start + batch_size]), "cuda", max_len=128
            )
            batches.append(encoded.detach().cpu().numpy().astype(np.float64))
    values = np.concatenate(batches, axis=0)
    return values / (np.linalg.norm(values, axis=1, keepdims=True) + 1e-9)


def _make_graph_context(
    runtime: Mapping[str, Any],
    spec: Mapping[str, Any],
    x_fit: np.ndarray,
    bridge: Mapping[str, Any],
    *,
    true_weights: Mapping[tuple[str, str], float] | None = None,
) -> dict[str, Any]:
    node_ids = _graph_nodes(spec)
    graph_obj = runtime["graph"].Graph([], node_ids, _edge_pairs(spec))
    observed_index = {node: index for index, node in enumerate(node_ids)}
    estimated_weights, _ = graph_obj.estimate_weights(x_fit, observed_index)
    weights = (
        {edge: float(true_weights[edge]) for edge in graph_obj.edges}
        if true_weights is not None
        else estimated_weights
    )
    # The fixture has no latents. The empty score mapping is the observed-only adapter;
    # latent_constraints augmentation is intentionally neither imported nor called.
    partial_corr = runtime["optimize"].partial_residual_corr(
        graph_obj, x_fit, observed_index, {}
    )
    return {
        "graph": graph_obj,
        "weights": weights,
        "estimated_weights": estimated_weights,
        "partial_corr": partial_corr,
        "bridge": dict(bridge),
        "spec": spec,
    }


def _solve_core(
    runtime: Mapping[str, Any],
    config: Mapping[str, Any],
    context: Mapping[str, Any],
    visible_embeddings: Mapping[str, np.ndarray],
    masked_nodes: Sequence[str],
    fold_index: int,
) -> np.ndarray:
    torch = runtime["torch"]
    graph_obj = context["graph"]
    weights = context["weights"]
    features = torch.tensor(
        runtime["l2_modules"].node_features(graph_obj, weights, set(visible_embeddings)),
        dtype=torch.float32,
        device="cpu",
    )
    solver = config["frozen_stage3"]["solver"]
    objective = config["frozen_stage3"]["objective"]
    embeddings, _ = runtime["l2_solver"].solve_unrolled(
        graph_obj,
        weights,
        dict(visible_embeddings),
        d=1024,
        weight_module=runtime["weightnet"],
        K=int(solver["unroll_steps"]),
        inner_lr=float(solver["inner_learning_rate"]),
        lam_zero=float(objective["lam_zero"]),
        lam_norm=float(objective["lam_norm"]),
        seed=int(fold_index),
        device="cpu",
        residual=float(objective["residual"]),
        lam_res=float(objective["lam_res"]),
        partial_corr=context["partial_corr"],
        neg_op=runtime["negation"],
        bridge=context["bridge"],
        train=False,
        feats=features,
    )
    predictions = np.stack([np.asarray(embeddings[node], dtype=np.float64) for node in masked_nodes])
    _require(np.isfinite(predictions).all(), "solver produced non-finite embeddings")
    _require(np.all(np.linalg.norm(predictions, axis=1) > 1e-12), "solver produced zero embeddings")
    return predictions


def _baseline_predictions(
    affinity: np.ndarray,
    target_embeddings: np.ndarray,
    masked_indices: Sequence[int],
    visible_indices: Sequence[int],
) -> np.ndarray:
    output = np.zeros((len(masked_indices), target_embeddings.shape[1]), dtype=np.float64)
    visible = np.asarray(visible_indices, dtype=np.int64)
    for row, masked_index in enumerate(masked_indices):
        weights = np.zeros(len(target_embeddings), dtype=np.float64)
        weights[visible] = affinity[int(masked_index), visible]
        if float(weights.sum()) < 1e-9:
            weights[visible] = 1.0
        output[row] = (weights / weights.sum()) @ target_embeddings
    return output


def _structural_flags(
    spec: Mapping[str, Any], masked_node: str, visible_nodes: set[str]
) -> dict[str, Any]:
    parents, children = _parent_child_maps(spec)
    modules = {str(node["id"]): str(node["module"]) for node in spec["nodes"]}
    visible_parents = parents[masked_node] & visible_nodes
    visible_children = children[masked_node] & visible_nodes
    visible_same_module = {
        node for node in visible_nodes if modules[node] == modules[masked_node]
    }
    return {
        "is_root": len(parents[masked_node]) == 0,
        "has_visible_parent": bool(visible_parents),
        "visible_parent_count": len(visible_parents),
        "has_visible_child": bool(visible_children),
        "visible_child_count": len(visible_children),
        "has_visible_same_module": bool(visible_same_module),
        "visible_same_module_count": len(visible_same_module),
        "module": modules[masked_node],
    }


def _append_prediction_records(
    records: list[dict[str, Any]],
    *,
    spec: Mapping[str, Any],
    fold_index: int,
    arm: str,
    shuffle_id: str | None,
    masked_nodes: Sequence[str],
    visible_nodes: set[str],
    predictions: np.ndarray,
    targets: np.ndarray,
) -> None:
    node_ids = _graph_nodes(spec)
    node_index = {node: index for index, node in enumerate(node_ids)}
    masked_indices = [node_index[node] for node in masked_nodes]
    ranks = core.rank_metrics(predictions, targets, masked_indices)
    match_hits = core.hungarian_match_hits(predictions, targets[masked_indices])
    node_meta = {str(node["id"]): node for node in spec["nodes"]}
    for row, node in enumerate(masked_nodes):
        records.append(
            {
                "graph_id": spec["graph_id"],
                "world": spec["world"],
                "fold": int(fold_index),
                "node_id": node,
                "arm": arm,
                "shuffle_id": shuffle_id,
                "true_label": str(node_meta[node]["gold_label"]),
                "causal_description": str(node_meta[node]["causal_description"]),
                **_structural_flags(spec, node, visible_nodes),
                "gold_embedding_cosine": float(ranks["gold_embedding_cosine"][row]),
                "rank": int(ranks["rank"][row]),
                "mrr": float(ranks["reciprocal_rank"][row]),
                "recall_at_1": int(ranks["recall_at_1"][row]),
                "recall_at_5": int(ranks["recall_at_5"][row]),
                "exact_decode": int(ranks["exact_decode"][row]),
                "match_acc": int(match_hits[row]),
                "judge_acc": None,
                "judge_status": "pending",
                "decoded_words": None,
                "decoder_alpha": None,
                "_prediction": np.asarray(predictions[row], dtype=np.float64),
            }
        )


def evaluate_all_arms(
    config: Mapping[str, Any],
    specs: Sequence[Mapping[str, Any]],
    datasets: Mapping[str, Mapping[str, Any]],
    runtime: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[tuple[str, int], dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    shuffle_metadata: list[dict[str, Any]] = []
    fold_decode_context: dict[tuple[str, int], dict[str, Any]] = {}
    assignments = config["masking"]["assignments"]
    permutation_seeds = config["shuffle_null"]["permutation_seeds"]

    for spec_with_meta in specs:
        spec = _clean_spec(spec_with_meta)
        graph_id = str(spec["graph_id"])
        data = datasets[graph_id]
        node_ids = _graph_nodes(spec)
        node_index = {node: index for index, node in enumerate(node_ids)}
        labels = [str(node["gold_label"]) for node in spec["nodes"]]
        targets = encode_texts(runtime, labels)
        x_fit = np.concatenate([np.asarray(data["train"]), np.asarray(data["dev"])], axis=0)
        raw_corr = np.corrcoef(x_fit.T)
        np.fill_diagonal(raw_corr, 0.0)
        raw_corr = np.nan_to_num(raw_corr, nan=0.0, posinf=0.0, neginf=0.0)
        dependence_matrix = np.abs(np.corrcoef(x_fit.T))
        np.fill_diagonal(dependence_matrix, 0.0)
        bridge_cfg = config["frozen_stage3"]["objective"]["bridge"]
        bridge = {
            "obs": list(node_ids),
            "dep_marg": dependence_matrix,
            "lam_upper": float(bridge_cfg["lam_upper"]),
            "kappa": float(bridge_cfg["kappa"]),
            "q": float(bridge_cfg["q"]),
        }

        oracle_context = _make_graph_context(runtime, spec, x_fit, bridge)
        true_context = _make_graph_context(
            runtime, spec, x_fit, bridge, true_weights=data["true_weights"]
        )
        reversed_spec = core.reverse_graph(spec)
        reversed_context = _make_graph_context(runtime, reversed_spec, x_fit, bridge)
        seeds = [int(value) for value in permutation_seeds[graph_id]]
        permutations = core.generate_permutations(len(node_ids), seeds, expected_count=20)
        shuffled_contexts: list[tuple[str, dict[str, Any]]] = []
        original_adjacency = core.adjacency_matrix(spec, weighted=True)
        for permutation_index, (seed, permutation) in enumerate(zip(seeds, permutations)):
            shuffle_id = f"shuffle_{permutation_index:02d}"
            shuffled_spec = core.permute_graph(spec, permutation)
            shuffled_adjacency = core.adjacency_matrix(shuffled_spec, weighted=True)
            core.validate_shuffled_adjacency(original_adjacency, shuffled_adjacency, permutation)
            shuffled_contexts.append(
                (shuffle_id, _make_graph_context(runtime, shuffled_spec, x_fit, bridge))
            )
            shuffle_metadata.append(
                {
                    "graph_id": graph_id,
                    "world": spec["world"],
                    "shuffle_id": shuffle_id,
                    "permutation_index": permutation_index,
                    "permutation_seed": seed,
                    "permutation": [int(value) for value in permutation],
                    "permutation_sha256": sha256_json([int(value) for value in permutation]),
                    "adjacency_sha256": sha256_json(shuffled_adjacency.tolist()),
                    "edge_count": len(shuffled_spec["edges"]),
                    "support_validated": True,
                }
            )

        for fold_index, raw_masked in enumerate(assignments[graph_id]):
            masked_nodes = [str(node) for node in raw_masked]
            masked_set = set(masked_nodes)
            visible_nodes = set(node_ids) - masked_set
            masked_indices = [node_index[node] for node in masked_nodes]
            visible_indices = [node_index[node] for node in node_ids if node in visible_nodes]
            visible_embeddings = {node: targets[node_index[node]] for node in visible_nodes}
            fold_decode_context[(graph_id, fold_index)] = {
                "visible_embeddings": targets[visible_indices],
                "targets": targets,
            }
            predictions: list[tuple[str, str | None, np.ndarray]] = [
                (
                    "uniform",
                    None,
                    _baseline_predictions(
                        np.ones_like(raw_corr), targets, masked_indices, visible_indices
                    ),
                ),
                (
                    "raw_correlation",
                    None,
                    _baseline_predictions(
                        np.clip(raw_corr, 0.0, None), targets, masked_indices, visible_indices
                    ),
                ),
                (
                    "core_oracle_estimated_weights",
                    None,
                    _solve_core(
                        runtime, config, oracle_context, visible_embeddings, masked_nodes, fold_index
                    ),
                ),
                (
                    "core_oracle_true_weights",
                    None,
                    _solve_core(
                        runtime, config, true_context, visible_embeddings, masked_nodes, fold_index
                    ),
                ),
                (
                    "core_reversed_graph",
                    None,
                    _solve_core(
                        runtime, config, reversed_context, visible_embeddings, masked_nodes, fold_index
                    ),
                ),
            ]
            for shuffle_id, shuffled_context in shuffled_contexts:
                predictions.append(
                    (
                        "core_shuffled_graph",
                        shuffle_id,
                        _solve_core(
                            runtime,
                            config,
                            shuffled_context,
                            visible_embeddings,
                            masked_nodes,
                            fold_index,
                        ),
                    )
                )
            for arm, shuffle_id, predicted in predictions:
                _append_prediction_records(
                    records,
                    spec=spec,
                    fold_index=fold_index,
                    arm=arm,
                    shuffle_id=shuffle_id,
                    masked_nodes=masked_nodes,
                    visible_nodes=visible_nodes,
                    predictions=predicted,
                    targets=targets,
                )
            print(
                f"[{time.strftime('%H:%M:%S')}] {graph_id} fold {fold_index + 1}/5 "
                f"complete ({len(predictions)} arm instances)",
                flush=True,
            )
    _require_equal(len(shuffle_metadata), 60, "shuffle metadata row count")
    _require_equal(len(records), 1500, "per-node arm-instance row count")
    return records, shuffle_metadata, fold_decode_context


def load_normalized_dictionary(config: Mapping[str, Any]) -> tuple[np.ndarray, list[str]]:
    path = _resolve_repo_path(
        config["frozen_stage3"]["decode_dictionary"]["path"], "decode dictionary"
    )
    with np.load(path, allow_pickle=True) as archive:
        concepts = np.asarray(archive["emb"], dtype=np.float32)
        words = [str(value) for value in archive["names"]]
    _require_equal(concepts.shape, (len(words), 1024), "dictionary shape")
    norms = np.linalg.norm(concepts, axis=1, keepdims=True)
    _require(np.all(norms > 0), "dictionary contains zero-norm concepts")
    concepts /= norms + 1e-9
    return concepts, words


def _normalized_rows(values: np.ndarray) -> np.ndarray:
    matrix = np.atleast_2d(np.asarray(values, dtype=np.float32))
    return matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)


def _batch_cosine_screen(
    embeddings: np.ndarray, concepts: np.ndarray, screen_k: int, *, batch_size: int = 16
) -> list[np.ndarray]:
    normalized = _normalized_rows(embeddings)
    keep = min(int(screen_k), len(concepts))
    screens: list[np.ndarray] = []
    for start in range(0, len(normalized), batch_size):
        batch = normalized[start : start + batch_size]
        similarities = concepts @ batch.T
        if keep == len(concepts):
            indices = np.broadcast_to(np.arange(len(concepts))[:, None], similarities.shape).T
        else:
            indices = np.argpartition(similarities, len(concepts) - keep, axis=0)[-keep:].T
        screens.extend(np.asarray(row, dtype=np.int64) for row in indices)
    return screens


def _positive_lasso_coefficients(
    embedding: np.ndarray,
    concepts: np.ndarray,
    screened_indices: np.ndarray,
    alpha: float,
) -> np.ndarray:
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import Lasso
    import warnings

    normalized = _normalized_rows(np.asarray(embedding))[0]
    model = Lasso(alpha=float(alpha), positive=True, fit_intercept=False, max_iter=3000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(
            np.asarray(concepts[screened_indices].T, dtype=np.float64),
            np.asarray(normalized, dtype=np.float64),
        )
    return np.asarray(model.coef_, dtype=np.float32)


def pick_alpha_visible(
    visible_embeddings: np.ndarray,
    concepts: np.ndarray,
    *,
    screen_k: int,
    target_l0: int,
) -> float:
    selected = np.asarray(visible_embeddings)
    if len(selected) > 8:
        selected = selected[np.linspace(0, len(selected) - 1, 8).astype(int)]
    screens = _batch_cosine_screen(selected, concepts, screen_k)
    low, high = 1e-5, 2e-2
    for _ in range(12):
        alpha = math.sqrt(low * high)
        support_sizes = [
            int(
                np.sum(
                    _positive_lasso_coefficients(vector, concepts, screen, alpha) > 1e-6
                )
            )
            for vector, screen in zip(selected, screens)
        ]
        if float(np.mean(support_sizes)) > float(target_l0):
            low = alpha
        else:
            high = alpha
    return math.sqrt(low * high)


def _decode_group(
    embeddings: np.ndarray,
    concepts: np.ndarray,
    words: Sequence[str],
    alpha: float,
    *,
    screen_k: int,
    top_k: int,
) -> list[list[str]]:
    normalized = _normalized_rows(embeddings)
    positive_screens = _batch_cosine_screen(normalized, concepts, screen_k)
    decoded: list[list[str] | None] = [None] * len(normalized)
    fallback_rows: list[int] = []
    for row, (vector, screen) in enumerate(zip(normalized, positive_screens)):
        coefficients = _positive_lasso_coefficients(vector, concepts, screen, alpha)
        order = np.argsort(coefficients)[::-1]
        selected = [str(words[int(screen[index])]) for index in order[:top_k] if coefficients[index] > 1e-6]
        if selected:
            decoded[row] = selected
        else:
            fallback_rows.append(row)
    if fallback_rows:
        negative = -normalized[fallback_rows]
        negative_screens = _batch_cosine_screen(negative, concepts, screen_k)
        for row, vector, screen in zip(fallback_rows, negative, negative_screens):
            coefficients = _positive_lasso_coefficients(vector, concepts, screen, alpha)
            order = np.argsort(coefficients)[::-1]
            decoded[row] = [
                "low " + str(words[int(screen[index])])
                for index in order[:top_k]
                if coefficients[index] > 1e-6
            ]
    return [value if value is not None else [] for value in decoded]


def decode_predictions(
    config: Mapping[str, Any],
    records: list[dict[str, Any]],
    fold_context: Mapping[tuple[str, int], Mapping[str, Any]],
    *,
    skip_decode: bool,
) -> list[dict[str, Any]]:
    judge_requests: list[dict[str, Any]] = []
    evaluation = config["evaluation"]
    judge_model = str(evaluation["judge"]["model"])
    if skip_decode:
        for record in records:
            record["judge_status"] = "debug_decode_skipped"
            record["decoded_words"] = []
            judge_requests.append(
                {
                    "model": judge_model,
                    "mode": "completion",
                    "rec": "",
                    "tgt": record["true_label"],
                    "graph_id": record["graph_id"],
                    "fold": record["fold"],
                    "node_id": record["node_id"],
                    "arm": record["arm"],
                    "shuffle_id": record["shuffle_id"],
                    "status": "debug_decode_skipped",
                }
            )
        return judge_requests

    concepts, words = load_normalized_dictionary(config)
    screen_k = int(evaluation["decoder_screen_k"])
    top_k = int(evaluation["decoder_top_k"])
    target_l0 = int(evaluation["decoder_target_l0"])
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(str(record["graph_id"]), int(record["fold"]))].append(record)
    for key in sorted(grouped):
        group = grouped[key]
        alpha = pick_alpha_visible(
            np.asarray(fold_context[key]["visible_embeddings"]),
            concepts,
            screen_k=screen_k,
            target_l0=target_l0,
        )
        predicted = np.stack([np.asarray(record["_prediction"]) for record in group])
        decoded = _decode_group(
            predicted, concepts, words, alpha, screen_k=screen_k, top_k=top_k
        )
        for record, concept_words in zip(group, decoded):
            record["decoded_words"] = concept_words
            record["decoder_alpha"] = float(alpha)
            record["judge_status"] = "pending"
            rec_string = ", ".join(concept_words)
            judge_requests.append(
                {
                    "model": judge_model,
                    "mode": "completion",
                    "rec": rec_string,
                    "tgt": record["true_label"],
                    "graph_id": record["graph_id"],
                    "world": record["world"],
                    "fold": record["fold"],
                    "node_id": record["node_id"],
                    "arm": record["arm"],
                    "shuffle_id": record["shuffle_id"],
                    "decoded_words": concept_words,
                    "status": "pending",
                }
            )
        print(
            f"[{time.strftime('%H:%M:%S')}] decoded {key[0]} fold {key[1] + 1}/5 "
            f"alpha={alpha:.3e}",
            flush=True,
        )
    _require_equal(len(judge_requests), len(records), "Judge request completeness")
    _require(
        all(request["status"] == "pending" for request in judge_requests),
        "full decode must leave every Judge request pending",
    )
    return judge_requests


def _mean_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for metric in LOCAL_METRICS:
        values = [float(row[metric]) for row in rows if row.get(metric) is not None]
        output[metric] = float(np.mean(values)) if values else None
    judge_values = [float(row["judge_acc"]) for row in rows if row.get("judge_acc") is not None]
    output["judge_acc"] = float(np.mean(judge_values)) if judge_values else None
    output["n"] = len(rows)
    return output


def _aggregate_records(
    records: Sequence[Mapping[str, Any]], keys: Sequence[str]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[tuple(record.get(key) for key in keys)].append(record)
    output: list[dict[str, Any]] = []
    sort_key = lambda key: tuple("" if value is None else str(value) for value in key)
    for identity in sorted(grouped, key=sort_key):
        rows = grouped[identity]
        output.append(
            {
                **{key: value for key, value in zip(keys, identity)},
                **_mean_metrics(rows),
                "judge_status": (
                    "pending" if not any(row.get("judge_acc") is not None for row in rows) else "complete"
                ),
            }
        )
    return output


def build_paired_deltas(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    leaves: dict[tuple[str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        leaves[(str(record["graph_id"]), int(record["fold"]), str(record["node_id"]))].append(record)
    rows: list[dict[str, Any]] = []
    for (graph_id, fold, node_id), leaf_rows in sorted(leaves.items()):
        by_arm: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for record in leaf_rows:
            by_arm[str(record["arm"])].append(record)
        _require_equal(len(by_arm[PRIMARY_ARM]), 1, f"{graph_id}/{fold}/{node_id}: primary rows")
        primary = by_arm[PRIMARY_ARM][0]
        for comparison, comparator_arm in COMPARATORS.items():
            comparator_rows = by_arm[comparator_arm]
            expected = 20 if comparator_arm == "core_shuffled_graph" else 1
            _require_equal(
                len(comparator_rows), expected, f"{graph_id}/{fold}/{node_id}: {comparison} rows"
            )
            for metric in LOCAL_METRICS:
                primary_value = float(primary[metric])
                comparator_value = float(np.mean([float(row[metric]) for row in comparator_rows]))
                rows.append(
                    {
                        "graph_id": graph_id,
                        "world": primary["world"],
                        "fold": fold,
                        "node_id": node_id,
                        "comparison": comparison,
                        "primary_arm": PRIMARY_ARM,
                        "comparator_arm": comparator_arm,
                        "metric": metric,
                        "primary_value": primary_value,
                        "comparator_value": comparator_value,
                        "comparator_instances": expected,
                        "delta": primary_value - comparator_value,
                    }
                )
    _require_equal(len(rows), 60 * len(COMPARATORS) * len(LOCAL_METRICS), "paired delta rows")
    return rows


def bootstrap_deltas(
    config: Mapping[str, Any], paired_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    bootstrap = config["bootstrap"]
    draws = int(bootstrap["draws"])
    seed = int(bootstrap["seed"])
    confidence = float(bootstrap["confidence_level"])
    output: list[dict[str, Any]] = []
    for comparison in COMPARATORS:
        for metric in LOCAL_METRICS:
            subset = [
                row
                for row in paired_rows
                if row["comparison"] == comparison and row["metric"] == metric
            ]
            stats = core.hierarchical_bootstrap(
                subset, draws=draws, seed=seed, confidence=confidence
            )
            output.append(
                {
                    "row_type": "comparison",
                    "scope": "aggregate",
                    "graph_id": "aggregate",
                    "comparison": comparison,
                    "metric": metric,
                    **stats,
                }
            )
            by_graph = core.bootstrap_by_graph(
                subset, draws=draws, seed=seed, confidence=confidence
            )
            for graph_id, graph_stats in by_graph.items():
                output.append(
                    {
                        "row_type": "comparison",
                        "scope": "graph",
                        "graph_id": graph_id,
                        "comparison": comparison,
                        "metric": metric,
                        **graph_stats,
                    }
                )
    return output


def build_shuffle_null(
    records: Sequence[Mapping[str, Any]], metadata: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    shuffled = [record for record in records if record["arm"] == "core_shuffled_graph"]
    shuffled_means = {
        (row["graph_id"], row["shuffle_id"]): row
        for row in _aggregate_records(shuffled, ("graph_id", "world", "shuffle_id"))
    }
    oracle_means = {
        row["graph_id"]: row
        for row in _aggregate_records(
            [record for record in records if record["arm"] == PRIMARY_ARM],
            ("graph_id", "world", "arm"),
        )
    }
    output: list[dict[str, Any]] = []
    for item in metadata:
        key = (item["graph_id"], item["shuffle_id"])
        mean_row = shuffled_means[key]
        oracle = oracle_means[item["graph_id"]]
        row = {**item, **{metric: mean_row[metric] for metric in LOCAL_METRICS}, "n": mean_row["n"]}
        for metric in LOCAL_METRICS:
            row[f"oracle_minus_shuffle_{metric}"] = float(oracle[metric]) - float(mean_row[metric])
        output.append(row)
    _require_equal(len(output), 60, "shuffle_null.csv row count")
    return output


def build_structural_diagnostics(
    records: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    properties = (
        ("is_root", "root_status"),
        ("has_visible_parent", "visible_parent"),
        ("has_visible_child", "visible_child"),
        ("has_visible_same_module", "visible_same_module"),
    )
    output: list[dict[str, Any]] = []
    graph_ids = sorted({str(row["graph_id"]) for row in records})
    for property_key, diagnostic in properties:
        scopes = [("aggregate", None)] + [("graph", value) for value in graph_ids]
        for scope, graph_id in scopes:
            scoped = (
                list(records)
                if graph_id is None
                else [row for row in records if row["graph_id"] == graph_id]
            )
            for arm in REQUIRED_ARMS:
                arm_rows = [row for row in scoped if row["arm"] == arm]
                for value in (False, True):
                    stratum = [row for row in arm_rows if bool(row[property_key]) is value]
                    if not stratum:
                        continue
                    if property_key == "is_root":
                        stratum_name = "root" if value else "non_root"
                    else:
                        stratum_name = "yes" if value else "no"
                    output.append(
                        {
                            "scope": scope,
                            "graph_id": graph_id or "aggregate",
                            "arm": arm,
                            "diagnostic": diagnostic,
                            "stratum": stratum_name,
                            **_mean_metrics(stratum),
                            "judge_status": "pending",
                        }
                    )
    return output

def classify_decision(
    *,
    shuffle_mean: float,
    shuffle_ci_low: float,
    no_graph_mean: float,
    no_graph_ci_low: float,
    reverse_mean: float,
    reverse_ci_low: float,
    consistent_graph_count: int,
) -> str:
    """Apply the preregistered three-way E0-prime decision rule to summary statistics."""
    shuffle_supported_positive = shuffle_mean > 0.0 and shuffle_ci_low > 0.0
    no_graph_supported_positive = no_graph_mean > 0.0 and no_graph_ci_low > 0.0
    reverse_supported_positive = reverse_mean > 0.0 and reverse_ci_low > 0.0
    directionally_positive = shuffle_mean > 0.0 and no_graph_mean > 0.0

    go = (
        shuffle_supported_positive
        and no_graph_supported_positive
        and consistent_graph_count >= 2
        and reverse_supported_positive
    )
    no_go = (
        not shuffle_supported_positive
        and not no_graph_supported_positive
        and reverse_mean <= 0.0
        and not directionally_positive
    )
    if go:
        return "GO"
    if no_go:
        return "NO-GO"
    return "INCONCLUSIVE"

def make_decision(
    records: Sequence[Mapping[str, Any]], bootstrap_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    aggregate = {
        row["comparison"]: row
        for row in bootstrap_rows
        if row["scope"] == "aggregate" and row["metric"] == PRIMARY_METRIC
    }
    graph_rows: dict[tuple[str, str], Mapping[str, Any]] = {
        (row["comparison"], row["graph_id"]): row
        for row in bootstrap_rows
        if row["scope"] == "graph" and row["metric"] == PRIMARY_METRIC
    }
    shuffle = aggregate["oracle_vs_shuffle"]
    no_graph = aggregate["oracle_vs_no_graph"]
    reverse = aggregate["oracle_vs_reverse"]
    graph_ids = sorted({str(row["graph_id"]) for row in records})
    consistent_graphs = [
        graph_id
        for graph_id in graph_ids
        if graph_rows[("oracle_vs_shuffle", graph_id)]["mean"] > 0.0
        and graph_rows[("oracle_vs_no_graph", graph_id)]["mean"] > 0.0
    ]
    decision = classify_decision(
        shuffle_mean=float(shuffle["mean"]),
        shuffle_ci_low=float(shuffle["ci_low"]),
        no_graph_mean=float(no_graph["mean"]),
        no_graph_ci_low=float(no_graph["ci_low"]),
        reverse_mean=float(reverse["mean"]),
        reverse_ci_low=float(reverse["ci_low"]),
        consistent_graph_count=len(consistent_graphs),
    )

    arm_means = {row["arm"]: row for row in _aggregate_records(records, ("arm",))}
    estimated = float(arm_means[PRIMARY_ARM][PRIMARY_METRIC])
    true_weight = float(arm_means["core_oracle_true_weights"][PRIMARY_METRIC])
    shuffled_mean = float(arm_means["core_shuffled_graph"][PRIMARY_METRIC])
    uniform = float(arm_means["uniform"][PRIMARY_METRIC])
    if decision == "GO":
        failure_source = "none_detected"
        next_step = "E1 oracle graph plus J-space values is allowed"
        e1_allowed = True
    elif true_weight > estimated and true_weight > shuffled_mean and true_weight > uniform:
        failure_source = "edge_weight_estimation"
        next_step = "audit or improve edge-weight estimation before any E1"
        e1_allowed = False
    elif max(estimated, true_weight) <= max(shuffled_mean, uniform):
        failure_source = "causal_graph_semantic_constraint_mismatch"
        next_step = "diagnose graph-to-semantic constraint transfer; E1 is not allowed"
        e1_allowed = False
    else:
        failure_source = "other_engineering_or_statistical_uncertainty"
        next_step = "resolve statistical or interface uncertainty; E1 is not allowed"
        e1_allowed = False
    return {
        "decision": decision,
        "primary_metric": PRIMARY_METRIC,
        "judge_acc_status": "pending",
        "consistent_positive_graphs": consistent_graphs,
        "consistent_positive_graph_count": len(consistent_graphs),
        "criteria": {
            "oracle_vs_shuffle_mean_positive": shuffle["mean"] > 0.0,
            "oracle_vs_shuffle_ci_excludes_zero": shuffle["ci_low"] > 0.0,
            "oracle_vs_shuffle_supported_positive": (
                shuffle["mean"] > 0.0 and shuffle["ci_low"] > 0.0
            ),
            "oracle_vs_no_graph_mean_positive": no_graph["mean"] > 0.0,
            "oracle_vs_no_graph_ci_excludes_zero": no_graph["ci_low"] > 0.0,
            "oracle_vs_no_graph_supported_positive": (
                no_graph["mean"] > 0.0 and no_graph["ci_low"] > 0.0
            ),
            "core_directionally_positive": (
                shuffle["mean"] > 0.0 and no_graph["mean"] > 0.0
            ),
            "both_core_comparisons_unsupported": (
                not (shuffle["mean"] > 0.0 and shuffle["ci_low"] > 0.0)
                and not (no_graph["mean"] > 0.0 and no_graph["ci_low"] > 0.0)
            ),
            "reverse_mean_nonpositive": reverse["mean"] <= 0.0,
            "at_least_two_graphs_consistent": len(consistent_graphs) >= 2,
            "stable_oracle_advantage_over_reverse": (
                reverse["mean"] > 0.0 and reverse["ci_low"] > 0.0
            ),
        },
        "failure_source": failure_source,
        "e1_allowed": e1_allowed,
        "next_step": next_step,
        "operationalization": (
            "Supported-positive means mean >0 and aggregate 95% CI lower bound >0. GO requires "
            "supported-positive oracle-vs-shuffle, oracle-vs-uniform, and oracle-vs-reverse effects, "
            "plus joint positive direction in >=2 graphs. NO-GO requires both core comparisons "
            "(oracle-vs-shuffle and oracle-vs-uniform) to be unsupported, reverse mean <=0, and the "
            "two core means not both positive. Directionally positive core means that miss CI or "
            "cross-graph requirements, and all other conflicting patterns, are INCONCLUSIVE."
        ),
    }


def _capture(command: Sequence[str]) -> str:
    completed = subprocess.run(
        list(command),
        cwd=REPO_ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.stdout.strip()


def collect_provenance(
    config: Mapping[str, Any],
    config_path: Path,
    artifacts: Mapping[str, Any],
    runtime: Mapping[str, Any],
    *,
    started_at: str,
    elapsed_seconds: float,
    skip_decode: bool,
) -> dict[str, Any]:
    torch = runtime["torch"]
    status = _capture(("git", "status", "--porcelain=v1"))
    package_names = (
        "numpy",
        "scipy",
        "scikit-learn",
        "torch",
        "sentence-transformers",
        "transformers",
        "PyYAML",
    )
    packages: dict[str, str | None] = {}
    for name in package_names:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    gpu = {
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "device_count": int(torch.cuda.device_count()),
        "devices": [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
            }
            for index in range(torch.cuda.device_count())
        ],
        "encoder_device": "cuda",
        "weightnet_device": "cpu",
        "solver_device": "cpu",
        "negop_device": "cpu",
    }
    return {
        "schema_version": "task3.e0_prime.run_manifest.v1",
        "experiment_id": config["experiment_id"],
        "started_at_utc": started_at,
        "finished_at_utc": _utc_now(),
        "elapsed_seconds": elapsed_seconds,
        "formal_run": not skip_decode,
        "debug_skip_decode": skip_decode,
        "actual_command": subprocess.list2cmdline([sys.executable, *sys.argv]),
        "working_directory": str(Path.cwd()),
        "runner_path": str(SCRIPT_PATH),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "current_git_commit": _capture(("git", "rev-parse", "HEAD")),
        "git_dirty": bool(status),
        "git_status_porcelain": status.splitlines(),
        "artifact_training_commit": config["frozen_stage3"]["training"]["artifact_training_commit"],
        "artifacts": artifacts,
        "e5_snapshot_revision": E5_REVISION,
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "packages": packages,
            "gpu": gpu,
            "offline_environment": {
                key: os.environ.get(key)
                for key in (
                    "HF_CACHE",
                    "HF_HOME",
                    "HF_HUB_OFFLINE",
                    "TRANSFORMERS_OFFLINE",
                    "HF_DATASETS_OFFLINE",
                )
            },
        },
        "frozen_training_audit": {
            "weightnet": {
                "outer_optimizer": "Adam",
                "outer_learning_rate": 0.001,
                "epochs": 4,
                "procedure_total_updates": 256,
                "selected_checkpoint_epoch": 3,
                "selected_checkpoint_updates": 192,
                "inner_unroll_steps": 60,
            },
            "lora": {
                "optimizer": "Adam",
                "learning_rate": 0.0001,
                "epochs": 3,
                "updates": 192,
                "loss_weights": {
                    "anchor": 10.0,
                    "bridge": 1.0,
                    "independence": 0.3,
                    "negation": 1.0,
                },
            },
            "retrained_in_e0_prime": False,
        },
        "artifact_caveat": (
            "The v5-compatible WeightNet and LoRA checkpoints were retrained locally because the "
            "original ignored release bundle is absent. They are frozen for E0-prime, but the local "
            "reproduction did not exactly reproduce the README metrics."
        ),
        "adapter_contract": {
            "solver_input": "anonymous IDs, numeric X/W, graph support, visible-label embeddings only",
            "weight_estimation_rows": "train+dev only",
            "bridge_rows": "train+dev only",
            "residual_correlation_rows": "train+dev only",
            "raw_correlation_rows": "train+dev only",
            "test_rows_used_for_estimation_or_selection": False,
            "latent_augmentation": "skipped because latent set is empty",
            "judge_api_called": False,
        },
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "pending"
    return f"{float(value):.{digits}f}"


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_report(
    path: Path,
    config: Mapping[str, Any],
    specs: Sequence[Mapping[str, Any]],
    data_manifest: Mapping[str, Any],
    arm_summary: Sequence[Mapping[str, Any]],
    per_graph_rows: Sequence[Mapping[str, Any]],
    structural_rows: Sequence[Mapping[str, Any]],
    bootstrap_rows: Sequence[Mapping[str, Any]],
    decision: Mapping[str, Any],
    provenance: Mapping[str, Any],
    *,
    judge_request_count: int,
    skip_decode: bool,
) -> Path:
    arm_by_name = {str(row["arm"]): row for row in arm_summary}
    arm_table = _markdown_table(
        ("Arm", "Judge", "Match", "Cosine", "MRR", "R@1", "R@5", "Exact"),
        [
            (
                arm,
                "pending",
                _fmt(arm_by_name[arm]["match_acc"]),
                _fmt(arm_by_name[arm]["gold_embedding_cosine"]),
                _fmt(arm_by_name[arm]["mrr"]),
                _fmt(arm_by_name[arm]["recall_at_1"]),
                _fmt(arm_by_name[arm]["recall_at_5"]),
                _fmt(arm_by_name[arm]["exact_decode"]),
            )
            for arm in REQUIRED_ARMS
        ],
    )
    graph_table = _markdown_table(
        ("Graph", "World", "Nodes", "Edges", "Roots", "Max in", "Chain", "Fork", "Collider", "Mediator"),
        [
            (
                spec["graph_id"], spec["world"], spec["_stats"]["nodes"],
                spec["_stats"]["edges"], len(spec["_stats"]["roots"]),
                spec["_stats"]["max_indegree"], spec["_stats"]["chain_count"],
                spec["_stats"]["fork_count"], spec["_stats"]["collider_count"],
                spec["_stats"]["mediator_count"],
            )
            for spec in specs
        ],
    )
    qa_table = _markdown_table(
        ("Graph", "Data SHA-256", "Train mean err", "Train SD err", "Edge corr min", "Finite"),
        [
            (
                row["graph_id"], row["data_sha256"],
                f"{row['qa']['train_zscore_max_abs_mean']:.2e}",
                f"{row['qa']['train_zscore_max_abs_std_error']:.2e}",
                _fmt(row["qa"]["edge_parent_child_corr_min"]),
                "yes" if row["qa"]["finite_raw"] and row["qa"]["finite_standardized"] else "no",
            )
            for row in data_manifest["graphs"]
        ],
    )
    arm_order = {arm: index for index, arm in enumerate(REQUIRED_ARMS)}
    per_graph_table = _markdown_table(
        ("Graph", "Arm", "Match", "Cosine", "MRR", "R@1", "R@5", "Exact"),
        [
            (
                row["graph_id"],
                row["arm"],
                _fmt(row["match_acc"]),
                _fmt(row["gold_embedding_cosine"]),
                _fmt(row["mrr"]),
                _fmt(row["recall_at_1"]),
                _fmt(row["recall_at_5"]),
                _fmt(row["exact_decode"]),
            )
            for row in sorted(
                per_graph_rows,
                key=lambda value: (str(value["graph_id"]), arm_order[str(value["arm"])]),
            )
        ],
    )
    primary_bootstrap = [row for row in bootstrap_rows if row["metric"] == PRIMARY_METRIC]
    comparison_table = _markdown_table(
        ("Scope", "Graph", "Comparison", "Mean delta", "95% CI", "Paired win"),
        [
            (
                row["scope"], row["graph_id"], row["comparison"], _fmt(row["mean"]),
                f"[{_fmt(row['ci_low'])}, {_fmt(row['ci_high'])}]",
                _fmt(row["paired_win_rate"]),
            )
            for row in primary_bootstrap
        ],
    )
    primary_structural = [
        row for row in structural_rows
        if row["scope"] == "aggregate" and row["arm"] == PRIMARY_ARM
    ]
    structural_table = _markdown_table(
        ("Diagnostic", "Stratum", "n", "Match", "Cosine", "MRR"),
        [
            (
                row["diagnostic"], row["stratum"], row["n"], _fmt(row["match_acc"]),
                _fmt(row["gold_embedding_cosine"]), _fmt(row["mrr"]),
            )
            for row in primary_structural
        ],
    )
    artifact_table = _markdown_table(
        ("Artifact", "Path", "SHA-256"),
        [
            (name, provenance["artifacts"][name]["path"], provenance["artifacts"][name]["sha256"])
            for name in ("lora", "dictionary", "weightnet", "negop")
        ],
    )
    frozen = config["frozen_stage3"]
    objective = frozen["objective"]
    report_status = "DEBUG / NON-FORMAL (--skip-decode)" if skip_decode else "FORMAL LOCAL RUN"
    lines = [
        "# Task 3 E0-prime -- Oracle Causal-Graph Bridge Test: Results",
        "",
        f"Run status: **{report_status}**  ",
        f"Decision: **{decision['decision']}**  ",
        "Judge-ACC: **pending** (requests written; no API was called)",
        "",
        "## Frozen Stage-3 method and audit",
        "",
        f"Formal source entrypoint: `{frozen['source_entrypoint']}`. Actual E0 adapter: "
        f"`{frozen['adapter_entrypoint']}`. The adapter imports library modules directly and never "
        "imports or executes the side-effectful formal entrypoint.",
        "",
        f"The encoder is `{frozen['semantic_encoder']}` at local snapshot `{E5_REVISION}`, with the "
        "frozen last-two-layer rank-8, alpha-16 LoRA. The frozen WeightNet runs K=60 functional-Adam "
        "inner steps at lr=0.02. WeightNet, negop, and solver run on CPU; only the encoder uses CUDA.",
        "",
        f"Frozen objective: lam_zero={objective['lam_zero']}, lam_norm={objective['lam_norm']}, "
        f"residual={objective['residual']}, lam_res={objective['lam_res']}, Pearson bridge "
        f"lambda={objective['bridge']['lam_upper']}, kappa={objective['bridge']['kappa']}, "
        f"q={objective['bridge']['q']}. This adapter changes interface shape only, not the loss.",
        "",
        "The frozen residual term is the solver's embedding residual plus a train+dev parent-regressed "
        "partial-correlation anchor. It is **not** the forbidden J-space innovation-residual preprocessing.",
        "",
        "Training audit (recorded, not rerun): the WeightNet procedure ran outer Adam lr=0.001 "
        "for 4 epochs / 256 total updates; best-checkpoint selection retained epoch 3 / update 192. "
        "It used K=60 inner steps. LoRA Adam ran lr=1e-4 for 3 epochs / 192 updates, with anchor=10, "
        "bridge=1, independence=0.3, negation=1 loss weights. No component was retrained in E0-prime.",
        "",
        artifact_table,
        "",
        "Artifact limitation: the original ignored v5-compatible release bundle is absent. These "
        "checkpoints were retrained locally at the recorded artifact commit and frozen. The local "
        "reproduction did not exactly reproduce README metrics; E0-prime tests this local frozen bundle.",
        "",
        "## Graph fixtures",
        "",
        graph_table,
        "",
        "Solver nodes are anonymous IDs. Each masked solve receives support, numeric X/W, and only "
        "the 16 visible-label embeddings; gold metadata remains in the fixture/evaluation layer.",
        "",
        "## SCM data QA",
        "",
        qa_table,
        "",
        "Each graph has 2,000 rows split 1,200/400/400. Z-score statistics use train only. W, bridge, "
        "partial residual correlation, and rawcorr use train+dev only; test is excluded from selection.",
        "",
        "## All-arm local metrics",
        "",
        arm_table,
        "",
        "The shuffled arm is averaged across 20 permutations per graph; `shuffle_null.csv` contains "
        "the full exactly-60-permutation null distribution.",
        "",
        "## Per-graph all-arm results",
        "",
        per_graph_table,
        "",
        "Every graph retains all 20 nodes and all six arms; shuffled entries average 20 permutations.",
        "",
        "## Structural diagnostics",
        "",
        structural_table,
        "",
        "No difficult node was excluded. Per-graph/all-arm strata are in `structural_diagnostics.csv`.",
        "",
        "## Paired hierarchical bootstrap",
        "",
        comparison_table,
        "",
        "For oracle-vs-shuffle, 20 shuffle values are averaged within each masked node first. All "
        "comparisons then resample graph -> fold -> masked node for exactly 10,000 fixed-seed draws. "
        "`paired_deltas.csv` includes all metrics, aggregate rows, and graph-specific rows.",
        "",
        "## Judge requests",
        "",
        f"`judge_requests.jsonl` contains {judge_request_count} cache-compatible requests frozen to "
        "model `gpt-5.5`, mode `completion`, with `rec`, `tgt`, arm, and shuffle provenance. Formal-run "
        "requests are all pending; no network/API call or fabricated verdict occurs.",
        "",
        "## Decision and attribution",
        "",
        f"**{decision['decision']}** on `{decision['primary_metric']}`. Consistent positive graphs: "
        f"{decision['consistent_positive_graph_count']}/3 "
        f"({', '.join(decision['consistent_positive_graphs']) or 'none'}).",
        "",
        decision["operationalization"],
        "",
        f"Failure-source label: **{decision['failure_source']}**. This is a diagnostic attribution "
        "heuristic from preregistered arm point estimates, not a proven causal attribution.",
        "",
        f"E1 allowed: **{'yes' if decision['e1_allowed'] else 'no'}**. {decision['next_step']}.",
        "",
        "## Reproducibility",
        "",
        f"- Current git commit: `{provenance['current_git_commit']}`",
        f"- Working tree dirty: `{provenance['git_dirty']}`",
        f"- Artifact training commit: `{provenance['artifact_training_commit']}`",
        f"- Config: `{provenance['config_path']}`",
        f"- Config SHA-256: `{provenance['config_sha256']}`",
        f"- Actual command: `{provenance['actual_command']}`",
        f"- e5 snapshot revision: `{E5_REVISION}`",
        "- Packages, GPU inventory, git status, checkpoint metadata, and hashes: `run_manifest.json`.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return path


def write_outputs(
    config: Mapping[str, Any],
    config_path: Path,
    specs: Sequence[Mapping[str, Any]],
    data_manifest: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    runtime: Mapping[str, Any],
    records: list[dict[str, Any]],
    shuffle_metadata: Sequence[Mapping[str, Any]],
    judge_requests: Sequence[Mapping[str, Any]],
    *,
    started_at: str,
    elapsed_seconds: float,
    skip_decode: bool,
) -> dict[str, Any]:
    results_dir = config_path.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    public_records = [
        {key: value for key, value in record.items() if not key.startswith("_")}
        for record in records
    ]
    per_fold = _aggregate_records(public_records, ("graph_id", "world", "fold", "arm"))
    per_graph = _aggregate_records(public_records, ("graph_id", "world", "arm"))
    arm_summary = _aggregate_records(public_records, ("arm",))
    paired_nodes = build_paired_deltas(public_records)
    paired_stats = bootstrap_deltas(config, paired_nodes)
    shuffle_null = build_shuffle_null(public_records, shuffle_metadata)
    structural = build_structural_diagnostics(public_records)
    decision = make_decision(public_records, paired_stats)
    decision["formal_run"] = not skip_decode
    decision["debug_skip_decode"] = skip_decode
    provenance = collect_provenance(
        config,
        config_path,
        artifacts,
        runtime,
        started_at=started_at,
        elapsed_seconds=elapsed_seconds,
        skip_decode=skip_decode,
    )

    metric_fields = list(LOCAL_METRICS)
    per_node_fields = (
        "graph_id", "world", "fold", "node_id", "arm", "shuffle_id", "true_label",
        "causal_description", "module", "is_root", "has_visible_parent", "visible_parent_count",
        "has_visible_child", "visible_child_count", "has_visible_same_module",
        "visible_same_module_count", "judge_acc", "judge_status", "match_acc",
        "gold_embedding_cosine", "rank", "mrr", "recall_at_1", "recall_at_5",
        "exact_decode", "decoder_alpha", "decoded_words",
    )
    aggregate_fields = (
        "graph_id", "world", "fold", "arm", "n", "judge_acc", "judge_status",
        *metric_fields,
    )
    per_graph_fields = (
        "graph_id", "world", "arm", "n", "judge_acc", "judge_status", *metric_fields,
    )
    shuffle_fields = (
        "graph_id", "world", "shuffle_id", "permutation_index", "permutation_seed",
        "permutation", "permutation_sha256", "adjacency_sha256", "edge_count",
        "support_validated", "n", *metric_fields,
        *(f"oracle_minus_shuffle_{metric}" for metric in LOCAL_METRICS),
    )
    paired_fields = (
        "row_type", "scope", "graph_id", "comparison", "metric", "mean", "ci_low",
        "ci_high", "confidence_level", "paired_win_rate", "bootstrap_positive_rate", "draws",
        "seed", "n_graphs", "n_folds", "n_nodes",
    )
    paired_node_fields = (
        "graph_id", "world", "fold", "node_id", "comparison", "primary_arm",
        "comparator_arm", "metric", "primary_value", "comparator_value",
        "comparator_instances", "delta",
    )
    structural_fields = (
        "scope", "graph_id", "arm", "diagnostic", "stratum", "n", "judge_acc",
        "judge_status", *metric_fields,
    )
    summary_fields = (
        "row_type", "scope", "graph_id", "arm", "comparison", "metric", "n",
        "judge_acc", "judge_status", *metric_fields, "mean", "ci_low", "ci_high",
        "confidence_level", "paired_win_rate", "bootstrap_positive_rate", "draws", "seed",
        "n_graphs", "n_folds", "n_nodes",
    )
    summary_rows = [
        {"row_type": "arm", "scope": "aggregate", "graph_id": "aggregate", **row}
        for row in arm_summary
    ] + list(paired_stats)

    written = [
        write_csv(results_dir / "per_node.csv", public_records, per_node_fields),
        write_csv(results_dir / "per_fold.csv", per_fold, aggregate_fields),
        write_csv(results_dir / "per_graph.csv", per_graph, per_graph_fields),
        write_csv(results_dir / "shuffle_null.csv", shuffle_null, shuffle_fields),
        write_csv(results_dir / "paired_deltas.csv", paired_stats, paired_fields),
        write_csv(results_dir / "paired_node_deltas.csv", paired_nodes, paired_node_fields),
        write_csv(results_dir / "summary.csv", summary_rows, summary_fields),
        write_csv(results_dir / "structural_diagnostics.csv", structural, structural_fields),
        write_jsonl(results_dir / "judge_requests.jsonl", judge_requests),
        write_json(results_dir / "decision.json", decision),
        write_json(results_dir / "config_resolved.json", config),
    ]
    report_path = write_report(
        results_dir / "report.md",
        config,
        specs,
        data_manifest,
        arm_summary,
        per_graph,
        structural,
        paired_stats,
        decision,
        provenance,
        judge_request_count=len(judge_requests),
        skip_decode=skip_decode,
    )
    written.append(report_path)
    output_files = {
        str(path.relative_to(config_path.parent)): {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in written
    }
    provenance["decision"] = decision
    provenance["output_files"] = output_files
    provenance["data_manifest_path"] = str((config_path.parent / "data_manifest.json").resolve())
    provenance["data_manifest_sha256"] = sha256_file(config_path.parent / "data_manifest.json")
    write_json(results_dir / "provenance.json", provenance)
    write_json(results_dir / "run_manifest.json", provenance)
    return {
        "results_dir": results_dir,
        "decision": decision,
        "output_files": output_files,
        "per_node_rows": len(public_records),
        "judge_request_rows": len(judge_requests),
        "shuffle_null_rows": len(shuffle_null),
        "paired_stats_rows": len(paired_stats),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config.resolve()
    config = load_config(config_path)
    specs, artifacts = validate_config(config, config_path, hash_artifacts=True)
    preflight = {
        "status": "validated",
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "graph_ids": [spec["graph_id"] for spec in specs],
        "graph_stats": {spec["graph_id"]: spec["_stats"] for spec in specs},
        "artifacts": artifacts,
        "e5_snapshot_revision": E5_REVISION,
        "models_loaded": False,
        "experiment_executed": False,
    }
    if args.validate_only:
        print(json.dumps(_jsonable(preflight), ensure_ascii=False, allow_nan=False, indent=2))
        return 0
    if args.skip_decode:
        print(
            "WARNING: --skip-decode is a debug-only path; outputs are marked non-formal.",
            file=sys.stderr,
            flush=True,
        )

    started_at = _utc_now()
    started_clock = time.perf_counter()
    datasets, data_manifest = generate_datasets(config, config_path, specs)
    runtime = load_frozen_runtime(config)
    records, shuffle_metadata, fold_context = evaluate_all_arms(
        config, specs, datasets, runtime
    )
    judge_requests = decode_predictions(
        config, records, fold_context, skip_decode=bool(args.skip_decode)
    )
    elapsed = time.perf_counter() - started_clock
    output = write_outputs(
        config,
        config_path,
        specs,
        data_manifest,
        artifacts,
        runtime,
        records,
        shuffle_metadata,
        judge_requests,
        started_at=started_at,
        elapsed_seconds=elapsed,
        skip_decode=bool(args.skip_decode),
    )
    print(
        json.dumps(
            _jsonable(
                {
                    "status": "complete",
                    "formal_run": not args.skip_decode,
                    "results_dir": str(output["results_dir"]),
                    "decision": output["decision"],
                    "per_node_rows": output["per_node_rows"],
                    "judge_request_rows": output["judge_request_rows"],
                    "shuffle_null_rows": output["shuffle_null_rows"],
                    "elapsed_seconds": elapsed,
                }
            ),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValidationError, core.ValidationError) as error:
        print(f"E0-prime validation failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
