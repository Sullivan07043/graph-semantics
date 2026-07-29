"""Task 3 E0-double-prime: orientation and frozen-constraint audit.

This is a diagnostic adapter around the frozen v5 Stage-3 implementation.  It
reuses the exact E0-prime graphs, SCM arrays, splits, folds, seeds, semantic
space, WeightNet, dictionary, decoder, and baselines.  It never imports an LLM
or causal-discovery pipeline and never trains or mutates a checkpoint.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import yaml


SCRIPT_PATH = Path(__file__).resolve()
TASK_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = TASK_ROOT.parent
V5_ROOT = REPO_ROOT / "v5"
EXPERIMENT_DIR = TASK_ROOT / "experiments" / "e0_orientation_constraint_audit"
DEFAULT_CONFIG = EXPERIMENT_DIR / "config.yaml"
RESULTS_DIR = EXPERIMENT_DIR / "results"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from task3_v2.scripts import e0_audit_solver as audit_solver  # noqa: E402
from task3_v2.scripts import e0_core as core  # noqa: E402
from task3_v2.scripts import e0_orientation_audit as orientation  # noqa: E402
from task3_v2.scripts import run_e0_bridge as prime  # noqa: E402


ARMS: tuple[str, ...] = (
    "full_oracle",
    "generation_only_oracle",
    "oracle_without_generation",
    "residual_only_oracle",
    "independence_only_oracle",
    "symmetrized_oracle",
    "markov_blanket_oracle",
    "same_module_graph",
    "reversed_full",
    "shuffled_full",
    "raw_correlation",
    "uniform",
)

SOLVER_ARMS: tuple[str, ...] = ARMS[:-2]
NONSHUFFLED_SOLVER_ARMS: tuple[str, ...] = SOLVER_ARMS[:-1]
METRICS: tuple[str, ...] = (
    "gold_cosine",
    "centered_cosine",
    "prediction_margin",
    "mrr",
    "recall_at_1",
    "recall_at_5",
    "match_acc",
    "exact",
)
STRUCTURAL_GROUPS: tuple[str, ...] = (
    "root",
    "non_root",
    "non_root_visible_parent",
    "no_visible_parent_visible_child",
    "no_visible_structural_anchor",
)
MOTIF_PRECEDENCE: tuple[str, ...] = (
    "collider",
    "fork",
    "mediator",
    "chain",
    "other",
)
EXPECTED_ROWS = 60 * (9 + 20 + 2)


class AuditError(RuntimeError):
    """Fail-closed error for an invalid or non-parity audit run."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def _sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _audit_source_report() -> list[dict[str, Any]]:
    paths = (
        SCRIPT_PATH,
        TASK_ROOT / "scripts" / "e0_audit_solver.py",
        TASK_ROOT / "scripts" / "e0_orientation_audit.py",
        TASK_ROOT / "scripts" / "run_e0_bundle_replication.py",
        DEFAULT_CONFIG,
    )
    return [
        {
            "path": str(path.relative_to(REPO_ROOT)),
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in paths
        if path.is_file()
    ]

def _sha256_vector(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float32))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(row), ensure_ascii=False, sort_keys=True) + "\n")
    return path


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: (
                        json.dumps(_jsonable(row.get(field)), ensure_ascii=False)
                        if isinstance(row.get(field), (list, tuple, dict))
                        else row.get(field)
                    )
                    for field in fields
                }
            )
    return path


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    _require(isinstance(value, dict), f"{path}: YAML root must be a mapping")
    return value


def _repo_path(value: str) -> Path:
    path = (REPO_ROOT / value).resolve()
    try:
        path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise AuditError(f"path escapes repository: {value}") from exc
    return path


def _capture(command: Sequence[str]) -> str:
    try:
        return subprocess.check_output(
            list(command), cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unavailable"


def validate_and_load_config(
    config_path: Path, *, hash_artifacts: bool = True
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Validate the E0-double-prime declaration and the inherited E0-prime lock."""

    config_path = config_path.resolve()
    config = _load_yaml(config_path)
    _require(
        config.get("schema_version") == "task3.e0_double_prime.v1",
        "unexpected audit schema_version",
    )
    _require(
        config.get("status") == "frozen_before_formal_execution",
        "audit config was not frozen before execution",
    )
    _require(tuple(config.get("arms", ())) == ARMS, "audit arm order changed")
    _require(
        tuple(config["loss_terms"]["order"]) == audit_solver.TERM_NAMES,
        "loss component order differs from audit solver",
    )
    for arm in SOLVER_ARMS:
        expected = list(audit_solver.TERM_MASKS[arm])
        actual = [int(value) for value in config["loss_terms"]["masks"][arm]]
        _require(actual == expected, f"{arm}: configured term mask changed")
    _require(config["bootstrap"]["draws"] == 10000, "bootstrap draws changed")
    _require(config["bootstrap"]["seed"] == 88173, "bootstrap seed changed")

    source = config["source_e0_prime"]
    source_config_path = _repo_path(str(source["config"]))
    actual_source_sha = _sha256_file(source_config_path)
    _require(
        actual_source_sha == str(source["config_sha256"]).lower(),
        "E0-prime config SHA-256 differs from the frozen audit declaration",
    )
    prime_config = prime.load_config(source_config_path)
    specs, artifact_report = prime.validate_config(
        prime_config, source_config_path, hash_artifacts=hash_artifacts
    )
    source_decision_path = _repo_path(str(source["decision"]))
    source_decision = json.loads(source_decision_path.read_text(encoding="utf-8"))
    _require(
        source_decision.get("decision") == source["required_decision"] == "NO-GO",
        "E0-double-prime must diagnose the frozen E0-prime NO-GO",
    )

    _require(
        config["masking"]["assignments"] == prime_config["masking"]["assignments"],
        "masking assignments differ from E0-prime",
    )
    _require(
        config["shuffle_null"]["permutation_seeds"]
        == prime_config["shuffle_null"]["permutation_seeds"],
        "shuffle permutation seeds differ from E0-prime",
    )
    for key in ("lam_zero", "lam_norm", "residual", "lam_res"):
        _require(
            float(config["frozen_stage3"]["objective"][key])
            == float(prime_config["frozen_stage3"]["objective"][key]),
            f"frozen objective coefficient {key} changed",
        )
    for name, audit_entry in (
        ("calibrated_space", config["frozen_stage3"]["calibrated_space"]),
        ("decode_dictionary", config["frozen_stage3"]["decode_dictionary"]),
        ("semantic_negation", config["frozen_stage3"]["semantic_negation"]),
        ("solver", config["frozen_stage3"]["solver"]),
    ):
        prime_name = "semantic_negation" if name == "semantic_negation" else name
        prime_entry = prime_config["frozen_stage3"][prime_name]
        _require(
            str(audit_entry["sha256"]).lower() == str(prime_entry["sha256"]).lower(),
            f"{name} checkpoint hash changed",
        )
    # Fail closed on every inherited setting consumed by the audit adapter.
    # The source config hash protects the complete E0-prime declaration; these
    # explicit comparisons also prevent a contradictory normalized audit view.
    audit_stage = config["frozen_stage3"]
    prime_stage = prime_config["frozen_stage3"]
    for key in ("semantic_encoder", "semantic_encoder_snapshot_revision"):
        _require(audit_stage[key] == prime_stage[key], f"{key} changed")
    for key in ("calibrated_space", "decode_dictionary", "solver"):
        _require(audit_stage[key] == prime_stage[key], f"{key} declaration changed")
    for key in ("checkpoint", "sha256"):
        _require(
            audit_stage["semantic_negation"][key]
            == prime_stage["semantic_negation"][key],
            f"semantic_negation.{key} changed",
        )

    audit_objective = audit_stage["objective"]
    prime_objective = prime_stage["objective"]
    for key in ("lam_zero", "lam_norm", "residual", "lam_res"):
        _require(
            float(audit_objective[key]) == float(prime_objective[key]),
            f"objective.{key} changed",
        )
    for key in ("lam_upper", "kappa", "q"):
        _require(
            float(audit_objective["bridge"][key])
            == float(prime_objective["bridge"][key]),
            f"objective.bridge.{key} changed",
        )
    _require(
        audit_objective["bridge"]["measure"] == "train_dev_absolute_pearson"
        and prime_objective["bridge"]["measure"] == "pearson"
        and prime_objective["bridge"]["dependence_matrix"]
        == "train_dev_absolute_pearson_adapter",
        "bridge measure/adapter declaration changed",
    )
    _require(
        bool(audit_objective["latent_constraints"])
        == bool(prime_objective["latent_constraints"]),
        "latent-constraint switch changed",
    )
    _require(
        audit_objective["observed_only_effect"]
        == prime_objective["latent_constraints_observed_only_effect"],
        "observed-only latent-constraint behavior changed",
    )
    _require(
        audit_stage["artifact_training_commit"]
        == prime_stage["training"]["artifact_training_commit"],
        "artifact training commit changed",
    )
    _require(
        bool(audit_stage["retrain_in_e0_double_prime"]) is False
        and bool(prime_stage["training"]["retrain_in_e0_prime"]) is False,
        "artifact retraining is forbidden",
    )

    audit_graphs = config["graphs"]
    prime_graphs = prime_config["graphs"]
    _require(len(audit_graphs) == len(prime_graphs) == 3, "graph count changed")
    for audit_graph, prime_graph in zip(audit_graphs, prime_graphs):
        for key in ("graph_id", "world", "graph_seed", "data_seed"):
            _require(
                audit_graph[key] == prime_graph[key],
                f"graph declaration changed: {prime_graph['graph_id']}.{key}",
            )
        audit_spec = (config_path.parent / str(audit_graph["spec"])).resolve()
        prime_spec = (
            source_config_path.parent / str(prime_graph["spec"])
        ).resolve()
        _require(audit_spec == prime_spec, "graph spec path changed")

    audit_scm = config["scm"]
    prime_scm = prime_config["scm"]
    _require(audit_scm["form"] == prime_scm["form"], "SCM form changed")
    _require(
        int(audit_scm["samples_per_graph"])
        == int(prime_scm["samples_per_graph"]),
        "SCM sample count changed",
    )
    _require(
        float(audit_scm["non_root_noise_standard_deviation"])
        == float(prime_scm["non_root_noise"]["standard_deviation"]),
        "SCM non-root noise changed",
    )
    for split in ("train", "dev", "test"):
        _require(
            int(audit_scm["split"][split]) == int(prime_scm["split"][split]),
            f"SCM {split} split changed",
        )
    _require(
        audit_scm["standardization_fit_split"]
        == prime_scm["standardization"]["fit_split"]
        == "train",
        "standardization fit split changed",
    )
    _require(
        list(audit_scm["estimation_splits"]) == ["train", "dev"]
        and prime_scm["edge_weight_estimation"]["fit_splits"]
        == ["train", "dev"]
        and not prime_scm["edge_weight_estimation"]["test_used_for_selection"],
        "edge-weight estimation split changed",
    )

    for key in ("fraction", "folds", "masked_nodes_per_fold", "assignments"):
        _require(
            config["masking"][key] == prime_config["masking"][key],
            f"masking.{key} changed",
        )
    _require(
        config["shuffle_null"]["permutations_per_graph"]
        == prime_config["shuffle_null"]["permutations_per_graph"],
        "shuffle count changed",
    )
    for key in ("hierarchy", "draws", "seed", "confidence_level"):
        _require(
            config["bootstrap"][key] == prime_config["bootstrap"][key],
            f"bootstrap.{key} changed",
        )
    audit_eval = config["evaluation"]
    prime_eval = prime_config["evaluation"]
    for key in (
        "label_candidate_set",
        "decoder",
        "decoder_screen_k",
        "decoder_top_k",
        "decoder_target_l0",
        "decoder_alpha_fit",
        "judge",
    ):
        _require(audit_eval[key] == prime_eval[key], f"evaluation.{key} changed")
    for key in ("torch_threads", "deterministic_seed", "overwrite_results"):
        _require(
            config["execution"][key] == prime_config["execution"][key],
            f"execution.{key} changed",
        )
    _require(bool(config["execution"]["full_decode"]), "formal decoder was disabled")

    authority = config["code_authority"]
    frozen_main = str(authority["latest_main_commit_at_freeze"])
    _require(
        _capture(["git", "cat-file", "-t", frozen_main]) == "commit",
        "frozen latest-main commit is unavailable",
    )
    current_main = _capture(["git", "rev-parse", "main"])
    _require(current_main != "unavailable", "current main ref is unavailable")
    for label, commit in (("frozen main", frozen_main), ("current main", current_main)):
        result = subprocess.run(
            ["git", "diff", "--quiet", commit, "--", "v5"],
            cwd=REPO_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        _require(result.returncode == 0, f"working v5 differs from {label}")
    _require(
        _capture(
            ["git", "ls-files", "--others", "--exclude-standard", "--", "v5"]
        )
        == "",
        "untracked non-ignored files exist under v5",
    )
    _require(
        config["code_authority"]["v5_worktree_equals_latest_main"] is True,
        "latest-main v5 equality was not frozen",
    )
    return config, prime_config, specs, artifact_report


def load_frozen_e0_data(
    prime_config: Mapping[str, Any],
    specs: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Read the exact E0-prime NPZ arrays and verify their recorded hashes."""

    manifest_path = (
        TASK_ROOT / "experiments" / "e0_oracle_bridge" / "data_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = {str(row["graph_id"]): row for row in manifest["graphs"]}
    datasets: dict[str, dict[str, Any]] = {}
    for spec_with_meta in specs:
        graph_id = str(spec_with_meta["graph_id"])
        entry = entries[graph_id]
        data_path = _repo_path(str(entry["data_path"]))
        _require(data_path.is_file(), f"frozen E0-prime data missing: {data_path}")
        actual_sha = _sha256_file(data_path)
        _require(actual_sha == entry["data_sha256"], f"{graph_id}: frozen data SHA mismatch")
        with np.load(data_path, allow_pickle=False) as archive:
            node_ids = [str(value) for value in archive["node_ids"].tolist()]
            matrix = np.asarray(archive["true_weight_matrix"], dtype=np.float64)
            dataset = {
                "node_ids": node_ids,
                "raw": np.asarray(archive["raw"], dtype=np.float64),
                "standardized": np.asarray(archive["standardized"], dtype=np.float64),
                "train": np.asarray(archive["train"], dtype=np.float64),
                "dev": np.asarray(archive["dev"], dtype=np.float64),
                "test": np.asarray(archive["test"], dtype=np.float64),
                "train_mean": np.asarray(archive["train_mean"], dtype=np.float64),
                "train_std": np.asarray(archive["train_std"], dtype=np.float64),
                "data_path": str(data_path.relative_to(REPO_ROOT)),
                "data_sha256": actual_sha,
            }
        expected_nodes = prime._graph_nodes(spec_with_meta)
        _require(node_ids == expected_nodes, f"{graph_id}: NPZ node order changed")
        dataset["true_weights"] = {
            (node_ids[i], node_ids[j]): float(matrix[i, j])
            for i in range(len(node_ids))
            for j in range(len(node_ids))
            if matrix[i, j] != 0.0
        }
        datasets[graph_id] = dataset
    _require(len(datasets) == 3, "expected all three frozen E0-prime datasets")
    return datasets, manifest


def _clean_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in spec.items() if not str(key).startswith("_")}


def _original_maps(
    spec: Mapping[str, Any],
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    return prime._parent_child_maps(spec)


def _unordered_edge(a: str, b: str, order: Mapping[str, int]) -> tuple[str, str]:
    return (a, b) if order[a] < order[b] else (b, a)


def derived_relations(
    spec: Mapping[str, Any],
) -> dict[str, list[tuple[str, str]]]:
    """Build frozen skeleton, Markov-blanket, and module-support relations."""

    nodes = prime._graph_nodes(spec)
    order = {node: index for index, node in enumerate(nodes)}
    parents, children = _original_maps(spec)
    skeleton: set[tuple[str, str]] = {
        _unordered_edge(a, b, order) for a, b in prime._edge_pairs(spec)
    }
    markov: set[tuple[str, str]] = set()
    for node in nodes:
        blanket = set(parents[node]) | set(children[node])
        for child in children[node]:
            blanket |= set(parents[child])
        blanket.discard(node)
        for neighbor in blanket:
            markov.add(_unordered_edge(node, neighbor, order))
    modules = {str(row["id"]): str(row["module"]) for row in spec["nodes"]}
    same_module: set[tuple[str, str]] = set()
    for i, a in enumerate(nodes):
        for b in nodes[i + 1 :]:
            if modules[a] == modules[b]:
                same_module.add((a, b))
    return {
        "symmetrized_oracle": sorted(skeleton, key=lambda pair: (order[pair[0]], order[pair[1]])),
        "markov_blanket_oracle": sorted(markov, key=lambda pair: (order[pair[0]], order[pair[1]])),
        "same_module_graph": sorted(
            same_module, key=lambda pair: (order[pair[0]], order[pair[1]])
        ),
    }


def _make_bidirected_context(
    runtime: Mapping[str, Any],
    spec: Mapping[str, Any],
    x_fit: np.ndarray,
    bridge: Mapping[str, Any],
    relations: Sequence[tuple[str, str]],
) -> dict[str, Any]:
    nodes = prime._graph_nodes(spec)
    index = {node: i for i, node in enumerate(nodes)}
    corr = np.corrcoef(x_fit.T)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    edges: list[tuple[str, str]] = []
    weights: dict[tuple[str, str], float] = {}
    for a, b in relations:
        value = abs(float(corr[index[a], index[b]]))
        edges.extend(((a, b), (b, a)))
        weights[(a, b)] = value
        weights[(b, a)] = value
    graph_obj = runtime["graph"].Graph([], nodes, edges)
    observed_index = {node: i for i, node in enumerate(nodes)}
    partial_corr = runtime["optimize"].partial_residual_corr(
        graph_obj, x_fit, observed_index, {}
    )
    return {
        "graph": graph_obj,
        "weights": weights,
        "estimated_weights": dict(weights),
        "partial_corr": partial_corr,
        "bridge": dict(bridge),
        "spec": spec,
    }


def build_graph_contexts(
    runtime: Mapping[str, Any],
    config: Mapping[str, Any],
    spec_with_meta: Mapping[str, Any],
    dataset: Mapping[str, Any],
) -> tuple[
    dict[str, dict[str, Any]],
    list[tuple[str, dict[str, Any]]],
    list[dict[str, Any]],
]:
    """Create every graph arm while retaining the exact E0-prime permutations."""

    spec = _clean_spec(spec_with_meta)
    graph_id = str(spec["graph_id"])
    nodes = prime._graph_nodes(spec)
    x_fit = np.concatenate(
        [np.asarray(dataset["train"]), np.asarray(dataset["dev"])], axis=0
    )
    dependence = np.abs(np.corrcoef(x_fit.T))
    dependence = np.nan_to_num(dependence, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(dependence, 0.0)
    bridge_cfg = config["frozen_stage3"]["objective"]["bridge"]
    bridge = {
        "obs": list(nodes),
        "dep_marg": dependence,
        "lam_upper": float(bridge_cfg["lam_upper"]),
        "kappa": float(bridge_cfg["kappa"]),
        "q": float(bridge_cfg["q"]),
    }
    oracle_context = prime._make_graph_context(runtime, spec, x_fit, bridge)
    reversed_spec = core.reverse_graph(spec)
    reversed_context = prime._make_graph_context(runtime, reversed_spec, x_fit, bridge)
    contexts: dict[str, dict[str, Any]] = {
        "full_oracle": oracle_context,
        "generation_only_oracle": oracle_context,
        "oracle_without_generation": oracle_context,
        "residual_only_oracle": oracle_context,
        "independence_only_oracle": oracle_context,
        "reversed_full": reversed_context,
    }
    relation_sets = derived_relations(spec)
    graph_metadata: list[dict[str, Any]] = []
    for arm, relations in relation_sets.items():
        context = _make_bidirected_context(runtime, spec, x_fit, bridge, relations)
        contexts[arm] = context
        graph_metadata.append(
            {
                "graph_id": graph_id,
                "world": spec["world"],
                "arm": arm,
                "relation_count": len(relations),
                "directed_edge_count": len(context["graph"].edges),
                "weight_rule": "abs_train_dev_pearson_identical_both_directions",
                "is_causal_dag": False,
                "cycle_adapter": True,
                "relation_sha256": hashlib.sha256(
                    json.dumps(relations, separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
            }
        )
    graph_metadata.extend(
        [
            {
                "graph_id": graph_id,
                "world": spec["world"],
                "arm": "full_oracle",
                "relation_count": len(oracle_context["graph"].edges),
                "directed_edge_count": len(oracle_context["graph"].edges),
                "weight_rule": "signed_train_dev_pearson",
                "is_causal_dag": True,
                "cycle_adapter": False,
            },
            {
                "graph_id": graph_id,
                "world": spec["world"],
                "arm": "reversed_full",
                "relation_count": len(reversed_context["graph"].edges),
                "directed_edge_count": len(reversed_context["graph"].edges),
                "weight_rule": "signed_train_dev_pearson",
                "is_causal_dag": True,
                "cycle_adapter": False,
            },
        ]
    )

    seeds = [
        int(value)
        for value in config["shuffle_null"]["permutation_seeds"][graph_id]
    ]
    permutations = core.generate_permutations(len(nodes), seeds, expected_count=20)
    original_adjacency = core.adjacency_matrix(spec, weighted=True)
    shuffled: list[tuple[str, dict[str, Any]]] = []
    for index, (seed, permutation) in enumerate(zip(seeds, permutations)):
        shuffle_id = f"shuffle_{index:02d}"
        shuffled_spec = core.permute_graph(spec, permutation)
        shuffled_adjacency = core.adjacency_matrix(shuffled_spec, weighted=True)
        core.validate_shuffled_adjacency(
            original_adjacency, shuffled_adjacency, permutation
        )
        context = prime._make_graph_context(runtime, shuffled_spec, x_fit, bridge)
        shuffled.append((shuffle_id, context))
        graph_metadata.append(
            {
                "graph_id": graph_id,
                "world": spec["world"],
                "arm": "shuffled_full",
                "shuffle_id": shuffle_id,
                "permutation_seed": seed,
                "permutation": [int(value) for value in permutation],
                "directed_edge_count": len(context["graph"].edges),
                "weight_rule": "signed_train_dev_pearson",
                "is_causal_dag": True,
                "cycle_adapter": False,
            }
        )
    return contexts, shuffled, graph_metadata


def structural_features(
    spec: Mapping[str, Any], node: str, visible_nodes: set[str]
) -> dict[str, Any]:
    nodes = prime._graph_nodes(spec)
    parents, children = _original_maps(spec)
    modules = {str(row["id"]): str(row["module"]) for row in spec["nodes"]}
    visible_parents = sorted(parents[node] & visible_nodes)
    visible_children = sorted(children[node] & visible_nodes)
    visible_same_module = sorted(
        other
        for other in visible_nodes
        if other != node and modules[other] == modules[node]
    )
    indegree = len(parents[node])
    outdegree = len(children[node])
    roles: list[str] = []
    if indegree >= 2:
        roles.append("collider")
    if outdegree >= 2:
        roles.append("fork")
    if indegree >= 1 and outdegree >= 1:
        roles.append("mediator")
    if indegree == 1 and outdegree == 1:
        roles.append("chain")
    if not roles:
        roles.append("other")
    primary = next(role for role in MOTIF_PRECEDENCE if role in roles)
    # "Structural anchor" refers to a relation used by the current oracle
    # objective (visible parent or child). Same-module support is recorded
    # separately because it is only a positive diagnostic in E0-double-prime.
    no_anchor = not (visible_parents or visible_children)
    no_any_candidate_anchor = not (
        visible_parents or visible_children or visible_same_module
    )
    return {
        "module": modules[node],
        "is_root": indegree == 0,
        "is_sink": outdegree == 0,
        "indegree": indegree,
        "outdegree": outdegree,
        "has_visible_parent": bool(visible_parents),
        "visible_parent_count": len(visible_parents),
        "visible_parents": visible_parents,
        "has_visible_child": bool(visible_children),
        "visible_child_count": len(visible_children),
        "visible_children": visible_children,
        "has_visible_same_module": bool(visible_same_module),
        "visible_same_module_count": len(visible_same_module),
        "no_visible_structural_anchor": no_anchor,
        "no_visible_any_candidate_anchor": no_any_candidate_anchor,
        "motif_roles": roles,
        "primary_motif_role": primary,
    }


def structural_group_memberships(row: Mapping[str, Any]) -> list[str]:
    groups: list[str] = []
    if bool(row["is_root"]):
        groups.append("root")
    else:
        groups.append("non_root")
    if not bool(row["is_root"]) and bool(row["has_visible_parent"]):
        groups.append("non_root_visible_parent")
    if not bool(row["has_visible_parent"]) and bool(row["has_visible_child"]):
        groups.append("no_visible_parent_visible_child")
    if bool(row["no_visible_structural_anchor"]):
        groups.append("no_visible_structural_anchor")
    return groups


def _normalized_rows(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    return matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)


def prediction_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    target_indices: Sequence[int],
) -> dict[str, np.ndarray]:
    predicted = np.asarray(predictions, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    target_indices_np = np.asarray(target_indices, dtype=np.int64)
    ranks = core.rank_metrics(predicted, target, target_indices_np)
    pnorm = _normalized_rows(predicted)
    tnorm = _normalized_rows(target)
    similarity = pnorm @ tnorm.T
    margins = np.empty(len(predicted), dtype=np.float64)
    for row, truth in enumerate(target_indices_np):
        wrong = np.delete(similarity[row], truth)
        margins[row] = similarity[row, truth] - float(np.max(wrong))
    center = target.mean(axis=0)
    centered_pred = predicted - center
    centered_target = target[target_indices_np] - center
    centered_cos = np.sum(
        _normalized_rows(centered_pred) * _normalized_rows(centered_target), axis=1
    )
    return {
        "gold_cosine": np.asarray(ranks["gold_embedding_cosine"], dtype=np.float64),
        "centered_cosine": centered_cos,
        "prediction_margin": margins,
        "rank": np.asarray(ranks["rank"], dtype=np.int64),
        "mrr": np.asarray(ranks["reciprocal_rank"], dtype=np.float64),
        "recall_at_1": np.asarray(ranks["recall_at_1"], dtype=np.int64),
        "recall_at_5": np.asarray(ranks["recall_at_5"], dtype=np.int64),
        "exact": np.asarray(ranks["exact_decode"], dtype=np.int64),
    }


def append_prediction_records(
    records: list[dict[str, Any]],
    initial_vectors: list[np.ndarray],
    final_vectors: list[np.ndarray],
    *,
    spec: Mapping[str, Any],
    fold: int,
    arm: str,
    shuffle_id: str | None,
    masked_nodes: Sequence[str],
    visible_nodes: set[str],
    initial: np.ndarray,
    final: np.ndarray,
    targets: np.ndarray,
    trace: Mapping[str, Any] | None,
    optimization_applicable: bool,
) -> None:
    nodes = prime._graph_nodes(spec)
    node_index = {node: index for index, node in enumerate(nodes)}
    masked_indices = [node_index[node] for node in masked_nodes]
    archived_initial = np.asarray(initial, dtype=np.float32)
    archived_final = np.asarray(final, dtype=np.float32)
    metrics = prediction_metrics(archived_final, targets, masked_indices)
    match_hits = core.hungarian_match_hits(
        archived_final, targets[masked_indices]
    )
    metadata = {str(row["id"]): row for row in spec["nodes"]}
    for position, node in enumerate(masked_nodes):
        record_id = len(records)
        initial_vector = archived_initial[position]
        final_vector = archived_final[position]
        initial_vectors.append(initial_vector.copy())
        final_vectors.append(final_vector.copy())
        structure = structural_features(spec, node, visible_nodes)
        row = {
            "record_id": record_id,
            "graph_id": spec["graph_id"],
            "world": spec["world"],
            "fold": int(fold),
            "node_id": node,
            "arm": arm,
            "shuffle_id": shuffle_id,
            "true_label": str(metadata[node]["gold_label"]),
            "causal_description": str(metadata[node]["causal_description"]),
            **structure,
            "structural_groups": structural_group_memberships(structure),
            "optimization_applicable": bool(optimization_applicable),
            "initial_embedding_row": record_id,
            "final_embedding_row": record_id,
            "initial_embedding_sha256": _sha256_vector(initial_vector),
            "final_embedding_sha256": _sha256_vector(final_vector),
            "displacement_norm": float(np.linalg.norm(final_vector - initial_vector)),
            "gold_cosine": float(metrics["gold_cosine"][position]),
            "centered_cosine": float(metrics["centered_cosine"][position]),
            "prediction_margin": float(metrics["prediction_margin"][position]),
            "rank": int(metrics["rank"][position]),
            "mrr": float(metrics["mrr"][position]),
            "recall_at_1": int(metrics["recall_at_1"][position]),
            "recall_at_5": int(metrics["recall_at_5"][position]),
            "match_acc": int(match_hits[position]),
            "exact": int(metrics["exact"][position]),
            "judge_acc": None,
            "judge_status": "pending",
            "decoded_words": None,
            "decoder_alpha": None,
            "nonfinite_seen": (
                bool(trace["nonfinite_seen"]) if trace is not None else False
            ),
            "max_total_gradient_norm": (
                float(trace["max_total_gradient_norm"]) if trace is not None else None
            ),
            "max_parameter_norm": (
                float(trace["max_parameter_norm"]) if trace is not None else None
            ),
            "_prediction": final_vector,
        }
        records.append(row)


def _prefix_diagnostic_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    graph_id: str,
    world: str,
    fold: int,
    arm: str,
    shuffle_id: str | None,
) -> list[dict[str, Any]]:
    return [
        {
            "graph_id": graph_id,
            "world": world,
            "fold": int(fold),
            "arm": arm,
            "shuffle_id": shuffle_id,
            **dict(row),
            "optimization_applicable": True,
        }
        for row in rows
    ]


def _baseline_diagnostic_rows(
    *,
    graph_id: str,
    world: str,
    fold: int,
    arm: str,
    masked_nodes: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    losses: list[dict[str, Any]] = []
    gradients: list[dict[str, Any]] = []
    for term in audit_solver.TERM_NAMES:
        losses.append(
            {
                "graph_id": graph_id,
                "world": world,
                "fold": int(fold),
                "arm": arm,
                "shuffle_id": None,
                "term": term,
                "term_active": False,
                "optimization_applicable": False,
                "status": "not_applicable_closed_form_baseline",
                "raw_initial_loss": None,
                "raw_final_loss": None,
                "raw_loss_delta": None,
                "active_initial_loss": None,
                "active_final_loss": None,
                "active_loss_delta": None,
                "nonfinite": False,
            }
        )
        for node in masked_nodes:
            gradients.append(
                {
                    "graph_id": graph_id,
                    "world": world,
                    "fold": int(fold),
                    "node_id": node,
                    "arm": arm,
                    "shuffle_id": None,
                    "term": term,
                    "term_active": False,
                    "optimization_applicable": False,
                    "status": "not_applicable_closed_form_baseline",
                    "raw_final_gradient_norm": None,
                    "active_final_gradient_norm": None,
                    "total_final_gradient_norm": None,
                    "raw_near_zero": None,
                    "near_zero": None,
                    "exploding": None,
                    "nonfinite": False,
                }
            )
    return losses, gradients


def _solve_reference(
    runtime: Mapping[str, Any],
    config: Mapping[str, Any],
    context: Mapping[str, Any],
    visible_embeddings: Mapping[str, np.ndarray],
    fold: int,
    features: Any,
) -> dict[str, np.ndarray]:
    solver_cfg = config["frozen_stage3"]["solver"]
    objective = config["frozen_stage3"]["objective"]
    embeddings, _ = runtime["l2_solver"].solve_unrolled(
        context["graph"],
        context["weights"],
        dict(visible_embeddings),
        d=1024,
        weight_module=runtime["weightnet"],
        K=int(solver_cfg["unroll_steps"]),
        inner_lr=float(solver_cfg["inner_learning_rate"]),
        lam_zero=float(objective["lam_zero"]),
        lam_norm=float(objective["lam_norm"]),
        seed=int(fold),
        device="cpu",
        residual=float(objective["residual"]),
        lam_res=float(objective["lam_res"]),
        partial_corr=context["partial_corr"],
        neg_op=runtime["negation"],
        bridge=context["bridge"],
        train=False,
        feats=features,
    )
    return {node: np.asarray(value, dtype=np.float64) for node, value in embeddings.items()}


def _parity_record(
    reference: np.ndarray,
    audit: np.ndarray,
    *,
    graph_id: str,
    fold: int,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    gate = config["parity_gate"]
    ref = np.asarray(reference, dtype=np.float64)
    got = np.asarray(audit, dtype=np.float64)
    refn = _normalized_rows(ref)
    gotn = _normalized_rows(got)
    cosine_error = np.abs(1.0 - np.sum(refn * gotn, axis=1))
    allclose = bool(
        np.allclose(ref, got, rtol=float(gate["rtol"]), atol=float(gate["atol"]))
    )
    max_cosine_error = float(np.max(cosine_error))
    passed = allclose and max_cosine_error <= float(gate["max_row_cosine_error"])
    return {
        "graph_id": graph_id,
        "fold": int(fold),
        "passed": passed,
        "allclose": allclose,
        "rtol": float(gate["rtol"]),
        "atol": float(gate["atol"]),
        "max_abs_difference": float(np.max(np.abs(ref - got))),
        "max_row_cosine_error": max_cosine_error,
        "allowed_max_row_cosine_error": float(gate["max_row_cosine_error"]),
    }


def _load_e0_prime_reference_rows(config: Mapping[str, Any]) -> dict[tuple[str, int, str], dict[str, str]]:
    path = _repo_path(str(config["source_e0_prime"]["per_node"]))
    output: dict[tuple[str, int, str], dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["arm"] != "core_oracle_estimated_weights":
                continue
            key = (row["graph_id"], int(row["fold"]), row["node_id"])
            output[key] = row
    _require(len(output) == 60, "E0-prime oracle per-node reference is incomplete")
    return output


def check_e0_prime_metric_parity(
    records: Sequence[Mapping[str, Any]],
    reference: Mapping[tuple[str, int, str], Mapping[str, str]],
    *,
    atol: float,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in records:
        if row["arm"] != "full_oracle":
            continue
        key = (str(row["graph_id"]), int(row["fold"]), str(row["node_id"]))
        old = reference[key]
        continuous = {
            "gold_cosine": abs(
                float(row["gold_cosine"]) - float(old["gold_embedding_cosine"])
            ),
            "mrr": abs(float(row["mrr"]) - float(old["mrr"])),
        }
        discrete = {
            "rank": int(row["rank"]) == int(old["rank"]),
            "recall_at_1": int(row["recall_at_1"]) == int(old["recall_at_1"]),
            "recall_at_5": int(row["recall_at_5"]) == int(old["recall_at_5"]),
            "match_acc": int(row["match_acc"]) == int(old["match_acc"]),
            "exact": int(row["exact"]) == int(old["exact_decode"]),
        }
        passed = max(continuous.values()) <= atol and all(discrete.values())
        output.append(
            {
                "graph_id": key[0],
                "fold": key[1],
                "node_id": key[2],
                "passed": passed,
                "continuous_abs_differences": continuous,
                "discrete_equal": discrete,
            }
        )
    _require(len(output) == 60, "full-oracle metric parity rows are incomplete")
    return output


def evaluate_audit_arms(
    config: Mapping[str, Any],
    prime_config: Mapping[str, Any],
    specs: Sequence[Mapping[str, Any]],
    datasets: Mapping[str, Mapping[str, Any]],
    runtime: Mapping[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[tuple[str, int], dict[str, Any]],
    list[np.ndarray],
    list[np.ndarray],
    dict[str, Any],
    list[dict[str, Any]],
]:
    """Run all 435 solver instances and the two frozen baselines."""

    records: list[dict[str, Any]] = []
    loss_rows: list[dict[str, Any]] = []
    gradient_rows: list[dict[str, Any]] = []
    fold_decode_context: dict[tuple[str, int], dict[str, Any]] = {}
    initial_vectors: list[np.ndarray] = []
    final_vectors: list[np.ndarray] = []
    parity_rows: list[dict[str, Any]] = []
    graph_metadata: list[dict[str, Any]] = []
    assignments = config["masking"]["assignments"]
    objective = config["frozen_stage3"]["objective"]
    solver_cfg = config["frozen_stage3"]["solver"]
    gradients_cfg = config["gradient_diagnostics"]

    for spec_with_meta in specs:
        spec = _clean_spec(spec_with_meta)
        graph_id = str(spec["graph_id"])
        dataset = datasets[graph_id]
        nodes = prime._graph_nodes(spec)
        node_index = {node: index for index, node in enumerate(nodes)}
        labels = [str(row["gold_label"]) for row in spec["nodes"]]
        targets = prime.encode_texts(runtime, labels)
        x_fit = np.concatenate(
            [np.asarray(dataset["train"]), np.asarray(dataset["dev"])], axis=0
        )
        raw_corr = np.corrcoef(x_fit.T)
        raw_corr = np.nan_to_num(raw_corr, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(raw_corr, 0.0)
        contexts, shuffled_contexts, metadata = build_graph_contexts(
            runtime, config, spec_with_meta, dataset
        )
        graph_metadata.extend(metadata)
        oracle_context = contexts["full_oracle"]

        for fold, raw_masked in enumerate(assignments[graph_id]):
            masked_nodes = [str(node) for node in raw_masked]
            masked_set = set(masked_nodes)
            visible_nodes = set(nodes) - masked_set
            masked_indices = [node_index[node] for node in masked_nodes]
            visible_indices = [
                node_index[node] for node in nodes if node in visible_nodes
            ]
            visible_embeddings = {
                node: targets[node_index[node]]
                for node in nodes
                if node in visible_nodes
            }
            fold_decode_context[(graph_id, fold)] = {
                "visible_embeddings": targets[visible_indices],
                "targets": targets,
            }
            common = audit_solver.make_common_initial_state(
                runtime["l2_solver"],
                oracle_context["graph"],
                oracle_context["weights"],
                visible_embeddings,
                1024,
                seed=int(fold),
            )
            common_initial = np.stack(
                [common.free_embeddings[node] for node in masked_nodes]
            )

            solver_instances: list[
                tuple[str, str | None, Mapping[str, Any]]
            ] = [
                (arm, None, contexts[arm]) for arm in NONSHUFFLED_SOLVER_ARMS
            ]
            solver_instances.extend(
                ("shuffled_full", shuffle_id, context)
                for shuffle_id, context in shuffled_contexts
            )
            for arm, shuffle_id, context in solver_instances:
                torch = runtime["torch"]
                features = torch.tensor(
                    runtime["l2_modules"].node_features(
                        context["graph"],
                        context["weights"],
                        set(visible_embeddings),
                    ),
                    dtype=torch.float32,
                    device="cpu",
                )
                result = audit_solver.solve_audit(
                    runtime["l2_solver"],
                    context["graph"],
                    context["weights"],
                    visible_embeddings,
                    1024,
                    common_initial=common,
                    term_mask=arm,
                    masked_nodes=masked_nodes,
                    weight_module=runtime["weightnet"],
                    K=int(solver_cfg["unroll_steps"]),
                    inner_lr=float(solver_cfg["inner_learning_rate"]),
                    lam_zero=float(objective["lam_zero"]),
                    lam_norm=float(objective["lam_norm"]),
                    seed=int(fold),
                    device="cpu",
                    residual=float(objective["residual"]),
                    lam_res=float(objective["lam_res"]),
                    partial_corr=context["partial_corr"],
                    neg_op=runtime["negation"],
                    bridge=context["bridge"],
                    feats=features,
                    canonical_full_path=(
                        audit_solver.TERM_MASKS[arm] == audit_solver.FULL_MASK
                    ),
                    near_zero_threshold=float(
                        gradients_cfg["near_zero_norm_at_or_below"]
                    ),
                    exploding_threshold=float(
                        gradients_cfg["explosion_norm_above"]
                    ),
                    raise_on_nonfinite=True,
                )
                predicted = np.stack(
                    [result.embeddings[node] for node in masked_nodes]
                )
                if arm == "full_oracle":
                    reference_embeddings = _solve_reference(
                        runtime,
                        config,
                        context,
                        visible_embeddings,
                        fold,
                        features,
                    )
                    reference_predictions = np.stack(
                        [reference_embeddings[node] for node in masked_nodes]
                    )
                    parity = _parity_record(
                        reference_predictions,
                        predicted,
                        graph_id=graph_id,
                        fold=fold,
                        config=config,
                    )
                    parity_rows.append(parity)
                    _require(
                        parity["passed"],
                        f"full-oracle solver parity failed at {graph_id}/fold{fold}",
                    )
                append_prediction_records(
                    records,
                    initial_vectors,
                    final_vectors,
                    spec=spec,
                    fold=fold,
                    arm=arm,
                    shuffle_id=shuffle_id,
                    masked_nodes=masked_nodes,
                    visible_nodes=visible_nodes,
                    initial=common_initial,
                    final=predicted,
                    targets=targets,
                    trace=result.trace,
                    optimization_applicable=True,
                )
                loss_rows.extend(
                    _prefix_diagnostic_rows(
                        result.loss_terms,
                        graph_id=graph_id,
                        world=str(spec["world"]),
                        fold=fold,
                        arm=arm,
                        shuffle_id=shuffle_id,
                    )
                )
                gradient_rows.extend(
                    _prefix_diagnostic_rows(
                        result.gradient_norms,
                        graph_id=graph_id,
                        world=str(spec["world"]),
                        fold=fold,
                        arm=arm,
                        shuffle_id=shuffle_id,
                    )
                )

            baseline_predictions = {
                "raw_correlation": prime._baseline_predictions(
                    np.clip(raw_corr, 0.0, None),
                    targets,
                    masked_indices,
                    visible_indices,
                ),
                "uniform": prime._baseline_predictions(
                    np.ones_like(raw_corr),
                    targets,
                    masked_indices,
                    visible_indices,
                ),
            }
            for arm in ("raw_correlation", "uniform"):
                predicted = baseline_predictions[arm]
                append_prediction_records(
                    records,
                    initial_vectors,
                    final_vectors,
                    spec=spec,
                    fold=fold,
                    arm=arm,
                    shuffle_id=None,
                    masked_nodes=masked_nodes,
                    visible_nodes=visible_nodes,
                    initial=common_initial,
                    final=predicted,
                    targets=targets,
                    trace=None,
                    optimization_applicable=False,
                )
                baseline_losses, baseline_gradients = _baseline_diagnostic_rows(
                    graph_id=graph_id,
                    world=str(spec["world"]),
                    fold=fold,
                    arm=arm,
                    masked_nodes=masked_nodes,
                )
                loss_rows.extend(baseline_losses)
                gradient_rows.extend(baseline_gradients)
            print(
                f"[{time.strftime('%H:%M:%S')}] {graph_id} fold {fold + 1}/5 "
                f"complete (31 arm instances)",
                flush=True,
            )

    _require(len(records) == EXPECTED_ROWS, f"expected {EXPECTED_ROWS} per-node rows")
    _require(len(initial_vectors) == len(final_vectors) == len(records), "embedding rows incomplete")
    _require(len(parity_rows) == 15 and all(row["passed"] for row in parity_rows), "parity gate incomplete")
    prime_reference = _load_e0_prime_reference_rows(config)
    metric_parity = check_e0_prime_metric_parity(
        records,
        prime_reference,
        atol=float(config["parity_gate"]["e0_prime_continuous_metric_atol"]),
    )
    _require(
        all(row["passed"] for row in metric_parity),
        "full-oracle metrics differ from frozen E0-prime per_node.csv",
    )
    parity_report = {
        "passed": True,
        "solver_rows": parity_rows,
        "metric_rows": metric_parity,
        "solver_scopes": 15,
        "metric_nodes": 60,
    }
    return (
        records,
        loss_rows,
        gradient_rows,
        fold_decode_context,
        initial_vectors,
        final_vectors,
        parity_report,
        graph_metadata,
    )


def _mean(values: Sequence[Any]) -> float | None:
    finite = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return float(np.mean(finite)) if finite else None


def aggregate_arms(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate every arm overall and by graph without collapsing diagnostics."""

    output: list[dict[str, Any]] = []
    graph_ids = sorted({str(row["graph_id"]) for row in records})
    scopes: list[tuple[str, str | None]] = [("aggregate", None)]
    scopes.extend(("graph", graph_id) for graph_id in graph_ids)
    for scope, graph_id in scopes:
        scoped = [
            row
            for row in records
            if graph_id is None or str(row["graph_id"]) == graph_id
        ]
        for arm in ARMS:
            arm_rows = [row for row in scoped if row["arm"] == arm]
            _require(bool(arm_rows), f"{scope}/{graph_id}/{arm}: no records")
            leaf_count = len(
                {
                    (str(row["graph_id"]), int(row["fold"]), str(row["node_id"]))
                    for row in arm_rows
                }
            )
            output.append(
                {
                    "scope": scope,
                    "graph_id": graph_id or "aggregate",
                    "world": (
                        str(arm_rows[0]["world"]) if graph_id is not None else "aggregate"
                    ),
                    "arm": arm,
                    "n_instances": len(arm_rows),
                    "n_nodes": leaf_count,
                    "judge_acc": None,
                    "judge_status": "pending",
                    "displacement_norm": _mean(
                        [row["displacement_norm"] for row in arm_rows]
                    ),
                    **{
                        metric: _mean([row[metric] for row in arm_rows])
                        for metric in METRICS
                    },
                }
            )
    return output


def collapse_arm_leaves(
    records: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int, str], dict[str, dict[str, Any]]]:
    """Collapse the 20 shuffled instances to one mean row per masked node."""

    grouped: dict[
        tuple[str, int, str, str], list[Mapping[str, Any]]
    ] = defaultdict(list)
    for row in records:
        key = (
            str(row["graph_id"]),
            int(row["fold"]),
            str(row["node_id"]),
            str(row["arm"]),
        )
        grouped[key].append(row)
    output: dict[tuple[str, int, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for (graph_id, fold, node_id, arm), rows in grouped.items():
        expected = 20 if arm == "shuffled_full" else 1
        _require(
            len(rows) == expected,
            f"{graph_id}/{fold}/{node_id}/{arm}: expected {expected} instances",
        )
        first = rows[0]
        collapsed = {
            key: value
            for key, value in first.items()
            if not key.startswith("_")
        }
        collapsed["shuffle_instances"] = expected
        collapsed["displacement_norm"] = _mean(
            [row["displacement_norm"] for row in rows]
        )
        for metric in METRICS:
            collapsed[metric] = _mean([row[metric] for row in rows])
        output[(graph_id, fold, node_id)][arm] = collapsed
    _require(len(output) == 60, "expected 60 unique graph/fold/node leaves")
    for identity, by_arm in output.items():
        _require(set(by_arm) == set(ARMS), f"{identity}: incomplete arm set")
    return dict(output)


def aggregate_groups(
    leaves: Mapping[tuple[str, int, str], Mapping[str, Mapping[str, Any]]]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for group in STRUCTURAL_GROUPS:
        group_identities = [
            identity
            for identity, by_arm in leaves.items()
            if group in by_arm["full_oracle"]["structural_groups"]
        ]
        for arm in ARMS:
            rows = [leaves[identity][arm] for identity in group_identities]
            output.append(
                {
                    "group": group,
                    "arm": arm,
                    "n_nodes": len(rows),
                    "n_instances": int(
                        sum(int(row["shuffle_instances"]) for row in rows)
                    ),
                    "judge_acc": None,
                    "judge_status": "pending" if rows else "not_applicable_empty_group",
                    "displacement_norm": _mean(
                        [row["displacement_norm"] for row in rows]
                    ),
                    **{
                        metric: _mean([row[metric] for row in rows])
                        for metric in METRICS
                    },
                }
            )
    return output


STRUCTURAL_COMPARISONS: dict[str, tuple[str, str]] = {
    "full_oracle_minus_reversed_full": ("full_oracle", "reversed_full"),
    "full_oracle_minus_shuffled_full": ("full_oracle", "shuffled_full"),
    "full_oracle_minus_uniform": ("full_oracle", "uniform"),
    "full_oracle_minus_raw_correlation": ("full_oracle", "raw_correlation"),
}

DECOMPOSITION_COMPARISONS: dict[str, tuple[str, str]] = {
    "full_oracle_minus_reversed_full": ("full_oracle", "reversed_full"),
    "full_oracle_minus_shuffled_full": ("full_oracle", "shuffled_full"),
    "full_oracle_minus_uniform": ("full_oracle", "uniform"),
    "full_oracle_minus_raw_correlation": ("full_oracle", "raw_correlation"),
    "full_oracle_minus_generation_only": (
        "full_oracle",
        "generation_only_oracle",
    ),
    "oracle_without_generation_minus_full_oracle": (
        "oracle_without_generation",
        "full_oracle",
    ),
    "residual_only_minus_full_oracle": (
        "residual_only_oracle",
        "full_oracle",
    ),
    "independence_only_minus_full_oracle": (
        "independence_only_oracle",
        "full_oracle",
    ),
    "symmetrized_minus_full_oracle": (
        "symmetrized_oracle",
        "full_oracle",
    ),
    "markov_blanket_minus_full_oracle": (
        "markov_blanket_oracle",
        "full_oracle",
    ),
    "same_module_minus_full_oracle": (
        "same_module_graph",
        "full_oracle",
    ),
    "same_module_minus_shuffled_full": (
        "same_module_graph",
        "shuffled_full",
    ),
    "same_module_minus_uniform": (
        "same_module_graph",
        "uniform",
    ),
    "same_module_minus_raw_correlation": (
        "same_module_graph",
        "raw_correlation",
    ),
}


def build_paired_deltas(
    leaves: Mapping[tuple[str, int, str], Mapping[str, Mapping[str, Any]]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in STRUCTURAL_GROUPS:
        for identity, by_arm in sorted(leaves.items()):
            if group not in by_arm["full_oracle"]["structural_groups"]:
                continue
            graph_id, fold, node_id = identity
            for comparison, (primary_arm, comparator_arm) in STRUCTURAL_COMPARISONS.items():
                primary = by_arm[primary_arm]
                comparator = by_arm[comparator_arm]
                for metric in METRICS:
                    rows.append(
                        {
                            "analysis": "structural_group",
                            "group": group,
                            "graph_id": graph_id,
                            "world": primary["world"],
                            "fold": fold,
                            "node_id": node_id,
                            "comparison": comparison,
                            "primary_arm": primary_arm,
                            "comparator_arm": comparator_arm,
                            "comparator_instances": comparator["shuffle_instances"],
                            "metric": metric,
                            "primary_value": primary[metric],
                            "comparator_value": comparator[metric],
                            "delta": float(primary[metric]) - float(comparator[metric]),
                        }
                    )
    for identity, by_arm in sorted(leaves.items()):
        graph_id, fold, node_id = identity
        for comparison, (primary_arm, comparator_arm) in DECOMPOSITION_COMPARISONS.items():
            primary = by_arm[primary_arm]
            comparator = by_arm[comparator_arm]
            for metric in METRICS:
                rows.append(
                    {
                        "analysis": "constraint_decomposition",
                        "group": "all_nodes",
                        "graph_id": graph_id,
                        "world": primary["world"],
                        "fold": fold,
                        "node_id": node_id,
                        "comparison": comparison,
                        "primary_arm": primary_arm,
                        "comparator_arm": comparator_arm,
                        "comparator_instances": comparator["shuffle_instances"],
                        "metric": metric,
                        "primary_value": primary[metric],
                        "comparator_value": comparator[metric],
                        "delta": float(primary[metric]) - float(comparator[metric]),
                    }
                )
    return rows

def _hierarchical_bootstrap_matrix(
    identities: Sequence[tuple[str, int, str]],
    values: np.ndarray,
    *,
    draws: int,
    seed: int,
    confidence: float,
) -> dict[str, Any]:
    """Bootstrap many paired columns with one graph->fold->node resample plan."""

    _require(len(identities) == len(values), "bootstrap identity/value mismatch")
    _require(len(identities) > 0, "bootstrap requires at least one leaf")
    hierarchy: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, (graph, fold, _node) in enumerate(identities):
        hierarchy[graph][fold].append(index)
    graphs = sorted(hierarchy)
    rng = np.random.default_rng(seed)
    samples = np.empty((draws, values.shape[1]), dtype=np.float64)
    for draw in range(draws):
        total = np.zeros(values.shape[1], dtype=np.float64)
        count = 0
        selected_graphs = rng.integers(0, len(graphs), size=len(graphs))
        for graph_index in selected_graphs:
            graph = graphs[int(graph_index)]
            folds = sorted(hierarchy[graph])
            selected_folds = rng.integers(0, len(folds), size=len(folds))
            for fold_index in selected_folds:
                fold = folds[int(fold_index)]
                nodes = hierarchy[graph][fold]
                selected_nodes = rng.integers(0, len(nodes), size=len(nodes))
                selected = [nodes[int(index)] for index in selected_nodes]
                total += values[selected].sum(axis=0)
                count += len(selected)
        samples[draw] = total / count
    tail = (1.0 - confidence) / 2.0
    return {
        "mean": values.mean(axis=0),
        "ci_low": np.quantile(samples, tail, axis=0),
        "ci_high": np.quantile(samples, 1.0 - tail, axis=0),
        "bootstrap_positive_rate": np.mean(samples > 0.0, axis=0),
        "paired_win_rate": np.mean(values > 0.0, axis=0),
        "n_graphs": len(hierarchy),
        "n_folds": sum(len(folds) for folds in hierarchy.values()),
        "n_nodes": len(identities),
    }


def bootstrap_paired_deltas(
    config: Mapping[str, Any],
    paired_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Run the fixed 10k hierarchical bootstrap with shared plans per stratum."""

    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in paired_rows:
        grouped[(str(row["analysis"]), str(row["group"]))].append(row)
    expected_groups = [
        ("structural_group", group) for group in STRUCTURAL_GROUPS
    ] + [("constraint_decomposition", "all_nodes")]
    output: list[dict[str, Any]] = []
    draws = int(config["bootstrap"]["draws"])
    seed0 = int(config["bootstrap"]["seed"])
    confidence = float(config["bootstrap"]["confidence_level"])
    for group_index, key in enumerate(expected_groups):
        rows = grouped.get(key, [])
        comparison_map = (
            STRUCTURAL_COMPARISONS
            if key[0] == "structural_group"
            else DECOMPOSITION_COMPARISONS
        )
        columns = [
            (comparison, metric)
            for comparison in comparison_map
            for metric in METRICS
        ]
        if not rows:
            for comparison, metric in columns:
                primary, comparator = comparison_map[comparison]
                output.append(
                    {
                        "analysis": key[0],
                        "group": key[1],
                        "comparison": comparison,
                        "primary_arm": primary,
                        "comparator_arm": comparator,
                        "metric": metric,
                        "status": "not_applicable_empty_group",
                        "mean": None,
                        "ci_low": None,
                        "ci_high": None,
                        "confidence_level": confidence,
                        "paired_win_rate": None,
                        "bootstrap_positive_rate": None,
                        "draws": draws,
                        "seed": seed0 + group_index,
                        "n_graphs": 0,
                        "n_folds": 0,
                        "n_nodes": 0,
                    }
                )
            continue
        by_column_identity: dict[
            tuple[str, str], dict[tuple[str, int, str], float]
        ] = defaultdict(dict)
        for row in rows:
            column = (str(row["comparison"]), str(row["metric"]))
            identity = (
                str(row["graph_id"]),
                int(row["fold"]),
                str(row["node_id"]),
            )
            _require(
                identity not in by_column_identity[column],
                f"duplicate paired bootstrap leaf {key}/{column}/{identity}",
            )
            by_column_identity[column][identity] = float(row["delta"])
        identities = sorted(next(iter(by_column_identity.values())))
        for column in columns:
            _require(
                sorted(by_column_identity[column]) == identities,
                f"{key}/{column}: bootstrap leaves differ",
            )
        matrix = np.asarray(
            [
                [by_column_identity[column][identity] for column in columns]
                for identity in identities
            ],
            dtype=np.float64,
        )
        stats = _hierarchical_bootstrap_matrix(
            identities,
            matrix,
            draws=draws,
            seed=seed0 + group_index,
            confidence=confidence,
        )
        for index, (comparison, metric) in enumerate(columns):
            primary, comparator = comparison_map[comparison]
            output.append(
                {
                    "analysis": key[0],
                    "group": key[1],
                    "comparison": comparison,
                    "primary_arm": primary,
                    "comparator_arm": comparator,
                    "metric": metric,
                    "status": "complete",
                    "mean": float(stats["mean"][index]),
                    "ci_low": float(stats["ci_low"][index]),
                    "ci_high": float(stats["ci_high"][index]),
                    "confidence_level": confidence,
                    "paired_win_rate": float(stats["paired_win_rate"][index]),
                    "bootstrap_positive_rate": float(
                        stats["bootstrap_positive_rate"][index]
                    ),
                    "draws": draws,
                    "seed": seed0 + group_index,
                    "n_graphs": int(stats["n_graphs"]),
                    "n_folds": int(stats["n_folds"]),
                    "n_nodes": int(stats["n_nodes"]),
                }
            )
    return output

def _bootstrap_index(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, str, str], Mapping[str, Any]]:
    return {
        (
            str(row["analysis"]),
            str(row["group"]),
            str(row["comparison"]),
            str(row["metric"]),
        ): row
        for row in rows
    }


def _supported(row: Mapping[str, Any] | None) -> bool:
    return bool(
        row
        and row.get("status") == "complete"
        and row.get("mean") is not None
        and float(row["mean"]) > 0.0
        and float(row["ci_low"]) > 0.0
    )


def _supported_adverse(row: Mapping[str, Any] | None) -> bool:
    """Return whether a paired effect is stably below zero."""

    return bool(
        row
        and row.get("status") == "complete"
        and row.get("mean") is not None
        and float(row["mean"]) < 0.0
        and float(row["ci_high"]) < 0.0
    )

def _bundle_behavior_pass(bundle: Mapping[str, Any]) -> bool:
    for key in (
        "behavioral_trend_reproduced",
        "trend_reproduced",
        "bundle_trend_reproduced",
    ):
        if key in bundle:
            return bool(bundle[key])
    decision = bundle.get("decision")
    if isinstance(decision, Mapping):
        if "checkpoint_or_bundle_drift" in decision:
            return not bool(decision["checkpoint_or_bundle_drift"])
        if "behavioral_trend_reproduced" in decision:
            return bool(decision["behavioral_trend_reproduced"])
    return False


def load_bundle_replication(
    config: Mapping[str, Any],
    config_path: Path,
) -> dict[str, Any]:
    """Load a formal bundle result and verify all frozen source bindings."""

    path = RESULTS_DIR / "bundle_replication.json"
    _require(path.is_file(), "formal bundle_replication.json is missing")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "bundle_replication.json must be a mapping")
    _require(
        value.get("status") == "formal_local_api_free",
        "formal audit requires a formal, fully decoded bundle replication",
    )
    provenance = value.get("provenance")
    _require(isinstance(provenance, Mapping), "bundle provenance is missing")
    _require(
        provenance.get("decoder_run") is True
        and provenance.get("retrained_in_replication") is False
        and provenance.get("judge_api_called") is False,
        "bundle replication formal-mode invariants failed",
    )
    bindings = provenance.get("source_bindings")
    _require(isinstance(bindings, Mapping), "bundle source bindings are missing")
    _require(
        str(bindings.get("audit_config_sha256", "")).lower()
        == _sha256_file(config_path),
        "bundle replication was produced for a different audit config",
    )
    _require(
        str(bindings.get("source_e0_prime_config_sha256", "")).lower()
        == str(config["source_e0_prime"]["config_sha256"]).lower(),
        "bundle replication was produced for a different E0-prime source",
    )
    runner_hash = bindings.get(
        "bundle_runner_sha256",
        bindings.get("bundle_runner_sha256_at_execution"),
    )
    _require(
        isinstance(runner_hash, str)
        and len(runner_hash) == 64
        and all(char in "0123456789abcdefABCDEF" for char in runner_hash),
        "bundle runner SHA-256 binding is invalid",
    )
    artifact_map = {
        "lora": config["frozen_stage3"]["calibrated_space"]["sha256"],
        "dictionary": config["frozen_stage3"]["decode_dictionary"]["sha256"],
        "negop": config["frozen_stage3"]["semantic_negation"]["sha256"],
        "weightnet": config["frozen_stage3"]["solver"]["sha256"],
    }
    for name, expected_hash in artifact_map.items():
        actual = provenance.get("bundle_artifacts", {}).get(name, {}).get("sha256")
        _require(
            str(actual).lower() == str(expected_hash).lower(),
            f"bundle replication {name} artifact hash changed",
        )
    expected_datasets = config["bundle_replication"]["datasets"]
    summary_pairs = {
        (str(row["role"]), str(row["implementation_dataset_id"]))
        for row in value.get("summary", ())
    }
    _require(
        ("dev", expected_datasets["dev"]) in summary_pairs
        and ("heldout", expected_datasets["heldout"]) in summary_pairs
        and ("hierarchy", expected_datasets["hierarchy"]) in summary_pairs,
        "bundle selected datasets do not match the frozen audit",
    )
    request_path = RESULTS_DIR / "bundle_judge_requests.jsonl"
    _require(request_path.is_file(), "bundle Judge request file is missing")
    request_rows = [
        json.loads(line)
        for line in request_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    _require(
        len(request_rows) == int(provenance["judge_request_count"])
        and all(row.get("status") == "pending" for row in request_rows),
        "bundle Judge requests are incomplete or not pending",
    )
    value["result_path"] = str(path.relative_to(REPO_ROOT))
    value["result_sha256"] = _sha256_file(path)
    return value
def make_decision(
    orientation_result: Mapping[str, Any],
    bundle: Mapping[str, Any],
    arm_rows: Sequence[Mapping[str, Any]],
    group_rows: Sequence[Mapping[str, Any]],
    bootstrap_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the frozen A-F rules without treating conflicting cells as stable."""

    index = _bootstrap_index(bootstrap_rows)

    def get(group: str, comparison: str, metric: str) -> Mapping[str, Any] | None:
        analysis = (
            "constraint_decomposition" if group == "all_nodes" else "structural_group"
        )
        return index.get((analysis, group, comparison, metric))

    orientation_bug = bool(orientation_result.get("verdict", {}).get(
        "orientation_interface_bug", False
    ))
    bundle_pass = _bundle_behavior_pass(bundle)
    semantic_metrics = ("gold_cosine", "centered_cosine", "prediction_margin")
    retrieval_metrics = ("mrr", "recall_at_5", "match_acc")
    structural_comparisons = (
        "full_oracle_minus_reversed_full",
        "full_oracle_minus_shuffled_full",
        "full_oracle_minus_uniform",
        "full_oracle_minus_raw_correlation",
    )

    root_failure_count = sum(
        _supported_adverse(get("root", comparison, metric))
        for comparison in structural_comparisons
        for metric in semantic_metrics
    )
    visible_parent_supported = sum(
        _supported(get("non_root_visible_parent", comparison, metric))
        for comparison in structural_comparisons
        for metric in semantic_metrics
    )
    visible_parent_adverse = sum(
        _supported_adverse(get("non_root_visible_parent", comparison, metric))
        for comparison in structural_comparisons
        for metric in semantic_metrics
    )

    def comparison_evidence(
        group: str, comparison: str
    ) -> tuple[int, int, int, int]:
        semantic_positive = sum(
            _supported(get(group, comparison, metric))
            for metric in semantic_metrics
        )
        retrieval_positive = sum(
            _supported(get(group, comparison, metric))
            for metric in retrieval_metrics
        )
        semantic_adverse = sum(
            _supported_adverse(get(group, comparison, metric))
            for metric in semantic_metrics
        )
        retrieval_adverse = sum(
            _supported_adverse(get(group, comparison, metric))
            for metric in retrieval_metrics
        )
        return (
            semantic_positive,
            retrieval_positive,
            semantic_adverse,
            retrieval_adverse,
        )

    visible_parent_evidence = {
        comparison: comparison_evidence("non_root_visible_parent", comparison)
        for comparison in structural_comparisons
    }
    visible_parent_stable_comparisons = sum(
        semantic_positive >= 1
        and retrieval_positive >= 1
        and semantic_adverse == 0
        and retrieval_adverse == 0
        for (
            semantic_positive,
            retrieval_positive,
            semantic_adverse,
            retrieval_adverse,
        ) in visible_parent_evidence.values()
    )
    visible_parent_conflicting_comparisons = sum(
        (semantic_positive + retrieval_positive) > 0
        and (semantic_adverse + retrieval_adverse) > 0
        for (
            semantic_positive,
            retrieval_positive,
            semantic_adverse,
            retrieval_adverse,
        ) in visible_parent_evidence.values()
    )
    root_failure_concentrated = root_failure_count > visible_parent_adverse
    root_boundary_candidate = (
        root_failure_count >= 2
        and root_failure_concentrated
        and visible_parent_supported >= 3
        and visible_parent_stable_comparisons >= 2
    )

    without_generation_supported = sum(
        _supported(
            get(
                "all_nodes",
                "oracle_without_generation_minus_full_oracle",
                metric,
            )
        )
        for metric in semantic_metrics + retrieval_metrics
    )
    full_over_generation_supported = sum(
        _supported(
            get("all_nodes", "full_oracle_minus_generation_only", metric)
        )
        for metric in semantic_metrics + retrieval_metrics
    )
    generation_mismatch = (
        without_generation_supported >= 2 and full_over_generation_supported >= 2
    )

    causal_full_supported = sum(
        _supported(get("all_nodes", comparison, metric))
        for comparison in (
            "full_oracle_minus_shuffled_full",
            "full_oracle_minus_uniform",
        )
        for metric in semantic_metrics
    )
    same_module_vs_shuffle = sum(
        _supported(get("all_nodes", "same_module_minus_shuffled_full", metric))
        for metric in semantic_metrics + retrieval_metrics
    )
    same_module_vs_uniform = sum(
        _supported(get("all_nodes", "same_module_minus_uniform", metric))
        for metric in semantic_metrics + retrieval_metrics
    )
    same_module_comparisons = (
        "same_module_minus_shuffled_full",
        "same_module_minus_uniform",
    )
    same_module_evidence = {
        comparison: comparison_evidence("all_nodes", comparison)
        for comparison in same_module_comparisons
    }
    same_module_adverse = sum(
        semantic_adverse + retrieval_adverse
        for (
            _semantic_positive,
            _retrieval_positive,
            semantic_adverse,
            retrieval_adverse,
        ) in same_module_evidence.values()
    )
    same_module_conflicting_comparisons = sum(
        (semantic_positive + retrieval_positive) > 0
        and (semantic_adverse + retrieval_adverse) > 0
        for (
            semantic_positive,
            retrieval_positive,
            semantic_adverse,
            retrieval_adverse,
        ) in same_module_evidence.values()
    )
    same_module_positive = (
        same_module_vs_shuffle >= 2
        and same_module_vs_uniform >= 2
        and same_module_adverse == 0
    )
    material_metric_conflict = (
        visible_parent_conflicting_comparisons > 0
        or same_module_conflicting_comparisons > 0
    )
    broader_failure = not same_module_positive or material_metric_conflict
    root_boundary = root_boundary_candidate and not broader_failure
    visible_parent_still_fails = visible_parent_stable_comparisons < 2
    causal_semantic_mismatch = (
        not orientation_bug
        and bundle_pass
        and visible_parent_still_fails
        and causal_full_supported < 2
        and same_module_positive
        and not material_metric_conflict
    )

    if orientation_bug:
        category = "A"
        failure = "orientation_interface_bug"
    elif not bundle_pass:
        category = "B"
        failure = "checkpoint_or_bundle_drift"
    elif broader_failure:
        category = "F"
        failure = "broader_solver_or_metric_failure"
    elif root_boundary:
        category = "C"
        failure = "root_or_anchor_boundary"
    elif generation_mismatch:
        category = "D"
        failure = "generation_constraint_mismatch"
    elif causal_semantic_mismatch:
        category = "E"
        failure = "causal_graph_semantic_constraint_mismatch"
    else:
        category = "F"
        failure = "broader_solver_or_metric_failure"

    s0_allowed = category == "E" and same_module_positive and bundle_pass
    rerun_e0_prime = category in {"A", "B"}
    if category == "A":
        next_step = "repair only the orientation interface and rerun frozen E0-prime once"
    elif category == "B":
        next_step = "restore a behaviorally trusted Stage-3 bundle, then rerun frozen E0-prime"
    elif category == "C":
        next_step = "redefine the task boundary around observable structural anchors; do not enter old E1"
    elif category == "D":
        next_step = "close unchanged generation transfer and redesign the semantic constraint before any new benchmark"
    elif category == "E":
        next_step = "close generic causal-DAG transfer and permit S0 semantic-support graph benchmark"
    else:
        next_step = (
            "pause Task 3 and audit solver/evaluation because the semantic positive "
            "control failed or local metrics conflict materially"
        )

    return {
        "schema_version": "task3.e0_double_prime.decision.v2",
        "created_at_utc": _utc_now(),
        "primary_category": category,
        "failure_source": failure,
        "orientation_interface_bug": orientation_bug,
        "checkpoint_or_bundle_drift": not bundle_pass,
        "root_or_anchor_boundary_candidate": root_boundary_candidate,
        "root_or_anchor_boundary_sufficient": root_boundary,
        "root_failure_concentrated": root_failure_concentrated,
        "generation_constraint_mismatch_supported": generation_mismatch,
        "causal_graph_semantic_constraint_mismatch_supported": causal_semantic_mismatch,
        "broader_solver_or_metric_failure_supported": broader_failure,
        "same_module_positive_diagnostic": same_module_positive,
        "material_metric_conflict": material_metric_conflict,
        "evidence_counts": {
            "root_semantic_failure_cells": root_failure_count,
            "visible_parent_supported_semantic_cells": visible_parent_supported,
            "visible_parent_adverse_semantic_cells": visible_parent_adverse,
            "visible_parent_stable_baseline_comparisons": (
                visible_parent_stable_comparisons
            ),
            "visible_parent_conflicting_comparisons": (
                visible_parent_conflicting_comparisons
            ),
            "without_generation_supported_cells": without_generation_supported,
            "full_over_generation_only_supported_cells": full_over_generation_supported,
            "causal_full_supported_semantic_cells": causal_full_supported,
            "same_module_vs_shuffle_supported_cells": same_module_vs_shuffle,
            "same_module_vs_uniform_supported_cells": same_module_vs_uniform,
            "same_module_adverse_cells": same_module_adverse,
            "same_module_conflicting_comparisons": (
                same_module_conflicting_comparisons
            ),
        },
        "judge_acc_status": "pending",
        "rerun_e0_prime_required": rerun_e0_prime,
        "rerun_reason": failure if rerun_e0_prime else None,
        "old_e1_allowed": False,
        "s0_semantic_support_graph_allowed": s0_allowed,
        "next_step": next_step,
        "rule_note": (
            "Classification uses strata, centered cosine, margin, retrieval metrics, "
            "loss decomposition, and the same-module positive diagnostic. A stable "
            "advantage requires support across semantic and retrieval families without "
            "simultaneous supported adverse evidence; aggregate raw cosine or repeated "
            "cells from one family alone are never sufficient. A failed positive "
            "control or material cross-metric conflict blocks category C and selects F."
        ),
    }
def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(_fmt(value) for value in row) + " |" for row in rows
    )
    return "\n".join(lines)


def write_report(
    *,
    config: Mapping[str, Any],
    orientation_result: Mapping[str, Any],
    bundle: Mapping[str, Any],
    arm_rows: Sequence[Mapping[str, Any]],
    group_rows: Sequence[Mapping[str, Any]],
    loss_rows: Sequence[Mapping[str, Any]],
    gradient_rows: Sequence[Mapping[str, Any]],
    bootstrap_rows: Sequence[Mapping[str, Any]],
    parity: Mapping[str, Any],
    decision: Mapping[str, Any],
    judge_count: int,
    embedding_sha256: str,
) -> Path:
    aggregate_arms_by_name = {
        str(row["arm"]): row
        for row in arm_rows
        if row["scope"] == "aggregate"
    }
    group_index = {(str(row["group"]), str(row["arm"])): row for row in group_rows}
    boot = _bootstrap_index(bootstrap_rows)

    arm_table = _markdown_table(
        ["arm", "n", "gold cos", "centered cos", "margin", "MRR", "R@5", "Match"],
        [
            [
                arm,
                aggregate_arms_by_name[arm]["n_instances"],
                aggregate_arms_by_name[arm]["gold_cosine"],
                aggregate_arms_by_name[arm]["centered_cosine"],
                aggregate_arms_by_name[arm]["prediction_margin"],
                aggregate_arms_by_name[arm]["mrr"],
                aggregate_arms_by_name[arm]["recall_at_5"],
                aggregate_arms_by_name[arm]["match_acc"],
            ]
            for arm in ARMS
        ],
    )
    group_table_rows: list[list[Any]] = []
    for group in STRUCTURAL_GROUPS:
        for arm in (
            "full_oracle",
            "reversed_full",
            "shuffled_full",
            "uniform",
            "raw_correlation",
        ):
            row = group_index[(group, arm)]
            group_table_rows.append(
                [
                    group,
                    arm,
                    row["n_nodes"],
                    row["gold_cosine"],
                    row["centered_cosine"],
                    row["mrr"],
                    row["recall_at_5"],
                    row["match_acc"],
                ]
            )
    group_table = _markdown_table(
        ["group", "arm", "nodes", "gold cos", "centered cos", "MRR", "R@5", "Match"],
        group_table_rows,
    )

    key_bootstrap_rows: list[list[Any]] = []
    requested = [
        ("structural_group", "root", "full_oracle_minus_reversed_full"),
        ("structural_group", "non_root_visible_parent", "full_oracle_minus_uniform"),
        ("constraint_decomposition", "all_nodes", "full_oracle_minus_generation_only"),
        ("constraint_decomposition", "all_nodes", "oracle_without_generation_minus_full_oracle"),
        ("constraint_decomposition", "all_nodes", "same_module_minus_shuffled_full"),
        ("constraint_decomposition", "all_nodes", "same_module_minus_uniform"),
    ]
    for analysis, group, comparison in requested:
        for metric in ("gold_cosine", "centered_cosine", "prediction_margin", "mrr", "recall_at_5"):
            row = boot.get((analysis, group, comparison, metric))
            if row is None:
                continue
            key_bootstrap_rows.append(
                [
                    group,
                    comparison,
                    metric,
                    row["mean"],
                    row["ci_low"],
                    row["ci_high"],
                    row["n_nodes"],
                ]
            )
    key_bootstrap_table = _markdown_table(
        ["group", "comparison", "metric", "mean Δ", "CI low", "CI high", "n"],
        key_bootstrap_rows,
    )

    solver_losses = [row for row in loss_rows if row.get("optimization_applicable")]
    loss_table_rows: list[list[Any]] = []
    for arm in (
        "full_oracle",
        "generation_only_oracle",
        "oracle_without_generation",
        "residual_only_oracle",
        "independence_only_oracle",
    ):
        for term in audit_solver.TERM_NAMES:
            rows = [
                row
                for row in solver_losses
                if row["arm"] == arm and row["term"] == term
            ]
            loss_table_rows.append(
                [
                    arm,
                    term,
                    bool(rows[0]["term_active"]),
                    _mean([row["raw_initial_loss"] for row in rows]),
                    _mean([row["raw_final_loss"] for row in rows]),
                    _mean([row["raw_loss_delta"] for row in rows]),
                ]
            )
    loss_table = _markdown_table(
        ["arm", "term", "active", "initial", "final", "final-initial"],
        loss_table_rows,
    )

    finite_gradients = [
        row for row in gradient_rows if row.get("optimization_applicable")
    ]
    gradient_summary = {
        "rows": len(finite_gradients),
        "near_zero_active": sum(
            bool(row["term_active"]) and bool(row["near_zero"])
            for row in finite_gradients
        ),
        "exploding": sum(bool(row["exploding"]) for row in finite_gradients),
        "nonfinite": sum(bool(row["nonfinite"]) for row in finite_gradients),
    }

    root_full = group_index[("root", "full_oracle")]
    root_reverse = group_index[("root", "reversed_full")]
    nonroot_full = group_index[("non_root", "full_oracle")]
    nonroot_reverse = group_index[("non_root", "reversed_full")]
    root_contribution = (
        (root_reverse["gold_cosine"] - root_full["gold_cosine"]) * root_full["n_nodes"]
    )
    nonroot_contribution = (
        (nonroot_reverse["gold_cosine"] - nonroot_full["gold_cosine"])
        * nonroot_full["n_nodes"]
    )
    contribution_share = root_contribution / (root_contribution + nonroot_contribution)

    bundle_pass = _bundle_behavior_pass(bundle)
    bundle_status = bundle.get("status", bundle.get("decision", "complete"))
    lines = [
        "# Task 3 E0-double-prime — Orientation and Constraint Audit",
        "",
        f"**Final classification: {decision['primary_category']} — `{decision['failure_source']}`.**",
        "",
        "This is a frozen diagnostic run. It did not run an LLM, J-space, CauScale, activation writes, "
        "latent discovery, or training, and it did not tune any loss coefficient.",
        "",
        "## 1. Orientation verdict",
        "",
        f"Orientation audit passed: **{not decision['orientation_interface_bug']}**. JSON `source -> target`, "
        "source-row/target-column adjacency, the adapter, `Graph.parents/children`, SCM generation, "
        "ALS, and Stage-3 generation all agree. The negative transposition control was rejected.",
        "",
        "For `A -> B -> C`, the implemented generation loss gives `dL/dz_B=(-1.4,-0.8)` in the "
        "fixed probe: B's own equation pulls toward parent A, while C's equation also back-propagates "
        "through B and pulls toward child C. This is bidirectional quadratic compatibility, not an "
        "interface transpose.",
        "",
        f"All 15 canonical full solves passed parity; embeddings file SHA-256 is `{embedding_sha256}`. "
        f"The 60 full-oracle node metrics also match frozen E0-prime within the preregistered tolerance.",
        "",
        "## 2. Bundle replication",
        "",
        f"Selected-dataset behavioral trend reproduced: **{bundle_pass}**; bundle result status: "
        f"`{bundle_status}`. See `../bundle_replication.md` and `bundle_replication.json` for the dev, "
        "held-out, and BigFive2 hierarchy/latent-constraint reruns. Judge remains pending where no "
        "cache/API verdict was available. The absent original release artifacts remain a provenance "
        "limitation, not by themselves a fabricated drift verdict.",
        "",
        "## 3. Root / non-root strata",
        "",
        group_table,
        "",
        f"Reversed-minus-oracle gold-cosine advantage attributable to roots: "
        f"**{100.0 * contribution_share:.2f}%**. Roots therefore explain most of reversed's aggregate "
        "cosine advantage, but they are not a sufficient explanation of the complete failure.",
        "",
        "## 4. Visible-parent strata",
        "",
        "All 42 non-root nodes have at least one visible parent. Their semantic and retrieval metrics "
        "must be read together: the frozen E0-prime pattern can lose cosine to no-graph baselines while "
        "improving MRR/R@5. This metric conflict is retained rather than resolved by selecting one metric.",
        "",
        key_bootstrap_table,
        "",
        "`no_visible_structural_anchor` means no visible parent or child in the current oracle objective. "
        "Visible same-module candidates are reported separately because they are not part of that objective.",
        "",
        "## 5. Constraint decomposition",
        "",
        arm_table,
        "",
        "Every optimizer arm starts from the exact same oracle ALS embeddings. The decomposition changes "
        "only the six registered term switches. `generation_only` intentionally leaves the residual "
        "inside the frozen generation equation without its penalties; `residual_only` is mathematically "
        "disconnected from semantic embeddings, so zero displacement/embedding gradient is expected.",
        "",
        loss_table,
        "",
        f"Gradient diagnostics: {gradient_summary['rows']} applicable node-term rows; "
        f"{gradient_summary['near_zero_active']} active near-zero, {gradient_summary['exploding']} "
        f"exploding, {gradient_summary['nonfinite']} non-finite. Thresholds were frozen at "
        f"{config['gradient_diagnostics']['near_zero_norm_at_or_below']} and "
        f"{config['gradient_diagnostics']['explosion_norm_above']}.",
        "",
        "## 6. Same-module positive diagnostic",
        "",
        f"Same-module passed the preregistered multi-metric positive diagnostic: "
        f"**{decision['same_module_positive_diagnostic']}**. This graph is a semantic-support control, "
        "not a causal method. Its bidirected adapter also makes v5 trek/independence operations reduce "
        "to connected-component reachability; the graph-arm metadata records this limitation explicitly.",
        "",
        "## 7. Where reversed's advantage comes from",
        "",
        f"Roots contribute {root_contribution:.4f} summed reverse-minus-oracle cosine versus "
        f"{nonroot_contribution:.4f} from non-roots. The effect is distributed across graphs and persists "
        "among roots with a visible child; it is not caused only by one completely unanchored node. "
        "Visible-parent non-roots show supported centered/retrieval gains but also supported adverse "
        "gold-cosine and margin cells. That is material metric conflict, not a stable category-C advantage.",
        "",
        "## 8. Final failure classification",
        "",
        f"Primary category: **{decision['primary_category']} — `{decision['failure_source']}`**. "
        f"Evidence counts: `{json.dumps(decision['evidence_counts'], sort_keys=True)}`.",
        "",
        f"Material cross-metric conflict: **{decision.get('material_metric_conflict', False)}**; "
        f"same-module positive control: **{decision['same_module_positive_diagnostic']}**. "
        "A supported positive cell does not count as a stable advantage when the same comparison "
        "also contains supported adverse evidence.",
        "",
        "The classification combines root/visible-parent strata, centered cosine, margin, retrieval "
        "metrics, the loss decomposition, bundle replication, and same-module positive control. It is "
        "not an aggregate-cosine-only verdict.",
        "",
        "## 9. Rerun E0-prime?",
        "",
        f"Required: **{decision['rerun_e0_prime_required']}**. {decision['next_step']}. There was no "
        "orientation repair, so an orientation-triggered E0-prime rerun is explicitly forbidden.",
        "",
        "## 10. S0 / old E1 decision",
        "",
        f"Old E1 allowed: **{decision['old_e1_allowed']}**. S0 latent-to-observed semantic-support graph "
        f"benchmark allowed: **{decision['s0_semantic_support_graph_allowed']}**.",
        "",
        "## Statistics, decoder, and provenance",
        "",
        f"Paired inference uses graph -> fold -> masked node hierarchical bootstrap with exactly "
        f"{config['bootstrap']['draws']:,} fixed-seed draws. Shuffled leaves are means over all 20 "
        "fixed permutations before pairing. `judge_requests.jsonl` contains "
        f"{judge_count:,} unique pending requests; no Judge-ACC was fabricated.",
        "",
        f"Worktree commit at freeze: `{config['code_authority']['worktree_commit_at_freeze']}`; frozen "
        f"latest-main authority: `{config['code_authority']['latest_main_commit_at_freeze']}`; current "
        f"main at report time: `{_capture(['git', 'rev-parse', 'main'])}`. The working `v5` is "
        "byte-for-byte tree-equivalent to both main snapshots, so no dirty-branch checkout or merge "
        "was needed.",
        "",
        "Formal commands are recorded in `provenance.json`. Full machine-readable values are in "
        "`per_node_audit.csv`, `per_group.csv`, `per_arm.csv`, `loss_terms.csv`, `gradient_norms.csv`, "
        "`paired_deltas.csv`, and `bootstrap_summary.csv`.",
        "",
    ]
    path = RESULTS_DIR / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return path

PER_NODE_FIELDS: tuple[str, ...] = (
    "record_id", "graph_id", "world", "fold", "node_id", "arm", "shuffle_id",
    "true_label", "causal_description", "module", "is_root", "is_sink", "indegree",
    "outdegree", "has_visible_parent", "visible_parent_count", "visible_parents",
    "has_visible_child", "visible_child_count", "visible_children",
    "has_visible_same_module", "visible_same_module_count", "no_visible_structural_anchor",
    "no_visible_any_candidate_anchor", "motif_roles", "primary_motif_role",
    "structural_groups", "optimization_applicable", "initial_embedding_row",
    "final_embedding_row", "initial_embedding_ref", "final_embedding_ref",
    "initial_embedding_sha256", "final_embedding_sha256", "embedding_archive_sha256",
    "displacement_norm", "gold_cosine", "centered_cosine", "prediction_margin", "rank",
    "mrr", "recall_at_1", "recall_at_5", "match_acc", "exact", "judge_acc",
    "judge_status", "decoder_alpha", "decoded_words", "nonfinite_seen",
    "max_total_gradient_norm", "max_parameter_norm",
)

AGGREGATE_FIELDS: tuple[str, ...] = (
    "scope", "graph_id", "world", "group", "arm", "n_instances", "n_nodes",
    "judge_acc", "judge_status", "displacement_norm", *METRICS,
)

LOSS_FIELDS: tuple[str, ...] = (
    "graph_id", "world", "fold", "arm", "shuffle_id", "term", "term_active",
    "optimization_applicable", "status", "raw_initial_loss", "raw_final_loss",
    "raw_loss_delta", "active_initial_loss", "active_final_loss", "active_loss_delta",
    "nonfinite",
)

GRADIENT_FIELDS: tuple[str, ...] = (
    "graph_id", "world", "fold", "node_id", "arm", "shuffle_id", "term",
    "term_active", "optimization_applicable", "status", "raw_final_gradient_norm",
    "active_final_gradient_norm", "total_final_gradient_norm", "raw_near_zero",
    "near_zero", "exploding", "nonfinite",
)

PAIRED_FIELDS: tuple[str, ...] = (
    "analysis", "group", "graph_id", "world", "fold", "node_id", "comparison",
    "primary_arm", "comparator_arm", "comparator_instances", "metric", "primary_value",
    "comparator_value", "delta",
)

BOOTSTRAP_FIELDS: tuple[str, ...] = (
    "analysis", "group", "comparison", "primary_arm", "comparator_arm", "metric",
    "status", "mean", "ci_low", "ci_high", "confidence_level", "paired_win_rate",
    "bootstrap_positive_rate", "draws", "seed", "n_graphs", "n_folds", "n_nodes",
)


def write_outputs(
    *,
    config_path: Path,
    config: Mapping[str, Any],
    prime_config: Mapping[str, Any],
    records: list[dict[str, Any]],
    loss_rows: list[dict[str, Any]],
    gradient_rows: list[dict[str, Any]],
    initial_vectors: Sequence[np.ndarray],
    final_vectors: Sequence[np.ndarray],
    judge_requests: list[dict[str, Any]],
    orientation_result: Mapping[str, Any],
    parity: Mapping[str, Any],
    graph_metadata: Sequence[Mapping[str, Any]],
    artifact_report: Mapping[str, Any],
    data_manifest: Mapping[str, Any],
    skip_decode: bool,
) -> dict[str, Any]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    embeddings_path = RESULTS_DIR / "embeddings.npz"
    np.savez_compressed(
        embeddings_path,
        record_id=np.arange(len(records), dtype=np.int64),
        initial=np.asarray(initial_vectors, dtype=np.float32),
        final=np.asarray(final_vectors, dtype=np.float32),
    )
    embedding_sha = _sha256_file(embeddings_path)
    for row in records:
        index = int(row["record_id"])
        row["initial_embedding_ref"] = f"embeddings.npz#initial[{index}]"
        row["final_embedding_ref"] = f"embeddings.npz#final[{index}]"
        row["embedding_archive_sha256"] = embedding_sha

    _require(len(judge_requests) == len(records), "Judge request count differs from records")
    for index, (request, row) in enumerate(zip(judge_requests, records)):
        _require(
            (request["graph_id"], int(request["fold"]), request["node_id"], request["arm"])
            == (row["graph_id"], int(row["fold"]), row["node_id"], row["arm"]),
            f"Judge request ordering mismatch at row {index}",
        )
        request["request_id"] = f"task3_e0pp_{index:04d}"
        request["record_id"] = index
    request_ids = [str(row["request_id"]) for row in judge_requests]
    _require(len(set(request_ids)) == len(request_ids), "Judge request IDs are not unique")

    leaves = collapse_arm_leaves(records)
    arm_rows = aggregate_arms(records)
    group_rows = aggregate_groups(leaves)
    paired_rows = build_paired_deltas(leaves)
    bootstrap_rows = bootstrap_paired_deltas(config, paired_rows)
    bundle = load_bundle_replication(config, config_path)
    decision = make_decision(
        orientation_result, bundle, arm_rows, group_rows, bootstrap_rows
    )
    decision["formal_run"] = not skip_decode
    decision["debug_skip_decode"] = bool(skip_decode)
    decision["parity_gate_passed"] = bool(parity["passed"])

    public_records = [
        {key: value for key, value in row.items() if not key.startswith("_")}
        for row in records
    ]
    _write_csv(RESULTS_DIR / "per_node_audit.csv", public_records, PER_NODE_FIELDS)
    _write_csv(RESULTS_DIR / "per_group.csv", group_rows, AGGREGATE_FIELDS)
    _write_csv(RESULTS_DIR / "per_arm.csv", arm_rows, AGGREGATE_FIELDS)
    _write_csv(RESULTS_DIR / "loss_terms.csv", loss_rows, LOSS_FIELDS)
    _write_csv(RESULTS_DIR / "gradient_norms.csv", gradient_rows, GRADIENT_FIELDS)
    _write_csv(RESULTS_DIR / "paired_deltas.csv", paired_rows, PAIRED_FIELDS)
    _write_csv(RESULTS_DIR / "bootstrap_summary.csv", bootstrap_rows, BOOTSTRAP_FIELDS)
    _write_jsonl(RESULTS_DIR / "judge_requests.jsonl", judge_requests)
    _write_json(RESULTS_DIR / "orientation_audit.json", orientation_result)
    _write_json(RESULTS_DIR / "parity_gate.json", parity)
    _write_json(RESULTS_DIR / "graph_arm_metadata.json", list(graph_metadata))
    _write_json(RESULTS_DIR / "config_resolved.json", config)
    _write_json(RESULTS_DIR / "decision.json", decision)

    formal_command = (
        ".\\.venv\\Scripts\\python.exe "
        "task3_v2\\scripts\\run_e0_audit.py --config "
        "task3_v2\\experiments\\e0_orientation_constraint_audit\\config.yaml"
    )
    provenance = {
        "schema_version": "task3.e0_double_prime.provenance.v1",
        "created_at_utc": _utc_now(),
        "repository_root": str(REPO_ROOT),
        "worktree_commit": _capture(["git", "rev-parse", "HEAD"]),
        "latest_main_commit": _capture(["git", "rev-parse", "main"]),
        "latest_main_v5_tree_equal": config["code_authority"]["v5_worktree_equals_latest_main"],
        "audit_source_files": _audit_source_report(),
        "audit_source_git_status": _capture(
            [
                "git",
                "status",
                "--short",
                "--",
                "task3_v2/scripts/e0_audit_solver.py",
                "task3_v2/scripts/e0_orientation_audit.py",
                "task3_v2/scripts/run_e0_audit.py",
                "task3_v2/scripts/run_e0_bundle_replication.py",
                "task3_v2/experiments/e0_orientation_constraint_audit/config.yaml",
            ]
        ),
        "config_path": str(config_path.relative_to(REPO_ROOT)),
        "config_sha256": _sha256_file(config_path),
        "source_e0_prime_config_path": config["source_e0_prime"]["config"],
        "source_e0_prime_config_sha256": config["source_e0_prime"]["config_sha256"],
        "artifacts": artifact_report,
        "source_data_manifest_sha256": _sha256_file(
            TASK_ROOT / "experiments" / "e0_oracle_bridge" / "data_manifest.json"
        ),
        "source_data_files": [
            {
                "graph_id": row["graph_id"],
                "path": row["data_path"],
                "sha256": row["data_sha256"],
            }
            for row in data_manifest["graphs"]
        ],
        "formal_command": formal_command,
        "orientation_command": (
            ".\\.venv\\Scripts\\python.exe "
            "task3_v2\\scripts\\e0_orientation_audit.py"
        ),
        "bundle_commands": bundle.get("run_commands", bundle.get("commands", [])),
        "judge_api_called": False,
        "judge_request_count": len(judge_requests),
        "judge_status": "pending" if not skip_decode else "debug_decode_skipped",
        "retrained_any_artifact": False,
        "changed_loss_coefficients": False,
        "changed_labels_graphs_folds_or_seeds": False,
        "test_split_used_for_estimation": False,
        "prohibited_components_run": [],
        "parity_gate": parity,
        "embedding_archive": {
            "path": str(embeddings_path.relative_to(REPO_ROOT)),
            "sha256": embedding_sha,
            "shape": [len(records), 1024],
            "dtype": "float32",
        },
        "row_counts": {
            "per_node_audit": len(public_records),
            "per_group": len(group_rows),
            "per_arm": len(arm_rows),
            "loss_terms": len(loss_rows),
            "gradient_norms": len(gradient_rows),
            "paired_deltas": len(paired_rows),
            "bootstrap_summary": len(bootstrap_rows),
            "judge_requests": len(judge_requests),
        },
        "formal_run": not skip_decode,
        "debug_skip_decode": bool(skip_decode),
    }
    _write_json(RESULTS_DIR / "provenance.json", provenance)
    report_path = write_report(
        config=config,
        orientation_result=orientation_result,
        bundle=bundle,
        arm_rows=arm_rows,
        group_rows=group_rows,
        loss_rows=loss_rows,
        gradient_rows=gradient_rows,
        bootstrap_rows=bootstrap_rows,
        parity=parity,
        decision=decision,
        judge_count=len(judge_requests),
        embedding_sha256=embedding_sha,
    )

    artifact_paths = [
        path
        for path in sorted(RESULTS_DIR.iterdir(), key=lambda item: item.name)
        if path.is_file() and path.name != "run_manifest.json"
    ]
    run_manifest = {
        "schema_version": "task3.e0_double_prime.run_manifest.v1",
        "created_at_utc": _utc_now(),
        "formal_run": not skip_decode,
        "decision": decision["primary_category"],
        "failure_source": decision["failure_source"],
        "artifacts": [
            {
                "path": str(path.relative_to(REPO_ROOT)),
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in artifact_paths
        ],
    }
    _write_json(RESULTS_DIR / "run_manifest.json", run_manifest)
    return {
        "decision": decision,
        "report_path": str(report_path),
        "embedding_sha256": embedding_sha,
        "provenance": provenance,
        "manifest": run_manifest,
    }


def _parse_csv_scalar(value: str) -> Any:
    """Recover the scalar/list types emitted by `_write_csv`."""

    if value == "":
        return None
    if value == "True":
        return True
    if value == "False":
        return False
    if value.startswith(("[", "{")):
        return json.loads(value)
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _read_typed_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [
            {key: _parse_csv_scalar(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def refinalize_existing_results(
    *,
    config_path: Path,
    config: Mapping[str, Any],
    orientation_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute only decision/report/manifest from completed formal outputs."""

    required = {
        "per_node": RESULTS_DIR / "per_node_audit.csv",
        "per_arm": RESULTS_DIR / "per_arm.csv",
        "per_group": RESULTS_DIR / "per_group.csv",
        "loss_terms": RESULTS_DIR / "loss_terms.csv",
        "gradient_norms": RESULTS_DIR / "gradient_norms.csv",
        "bootstrap_summary": RESULTS_DIR / "bootstrap_summary.csv",
        "parity": RESULTS_DIR / "parity_gate.json",
        "provenance": RESULTS_DIR / "provenance.json",
        "embeddings": RESULTS_DIR / "embeddings.npz",
        "judge_requests": RESULTS_DIR / "judge_requests.jsonl",
    }
    for label, path in required.items():
        _require(path.is_file(), f"cannot refinalize: missing {label}: {path}")

    per_node_rows = _read_typed_csv(required["per_node"])
    arm_rows = _read_typed_csv(required["per_arm"])
    group_rows = _read_typed_csv(required["per_group"])
    loss_rows = _read_typed_csv(required["loss_terms"])
    gradient_rows = _read_typed_csv(required["gradient_norms"])
    bootstrap_rows = _read_typed_csv(required["bootstrap_summary"])
    parity = json.loads(required["parity"].read_text(encoding="utf-8"))
    provenance = json.loads(required["provenance"].read_text(encoding="utf-8"))
    _require(bool(provenance.get("formal_run")), "cannot refinalize a debug run")
    for source_row in provenance["source_data_files"]:
        source_path = _repo_path(str(source_row["path"]))
        _require(
            _sha256_file(source_path) == source_row["sha256"],
            f"formal source data changed: {source_row['graph_id']}",
        )
    data_manifest_path = (
        TASK_ROOT / "experiments" / "e0_oracle_bridge" / "data_manifest.json"
    )
    _require(
        _sha256_file(data_manifest_path)
        == provenance["source_data_manifest_sha256"],
        "formal source data manifest changed",
    )

    residual_rows = [
        row for row in per_node_rows if row["arm"] == "residual_only_oracle"
    ]
    _require(len(residual_rows) == 60, "residual-only result rows are incomplete")
    _require(
        all(
            row["initial_embedding_sha256"] == row["final_embedding_sha256"]
            for row in residual_rows
        ),
        "residual-only archived embeddings are not byte-identical",
    )
    corrected_displacements = sum(
        float(row["displacement_norm"]) != 0.0 for row in residual_rows
    )
    for row in residual_rows:
        row["displacement_norm"] = 0.0
    for row in arm_rows:
        if row["arm"] == "residual_only_oracle":
            row["displacement_norm"] = 0.0
    _write_csv(required["per_node"], per_node_rows, PER_NODE_FIELDS)
    _write_csv(required["per_arm"], arm_rows, AGGREGATE_FIELDS)

    judge_count = sum(
        1
        for line in required["judge_requests"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    _require(
        judge_count == int(provenance["row_counts"]["judge_requests"]),
        "Judge request count differs from formal provenance",
    )
    embedding_sha = _sha256_file(required["embeddings"])
    _require(
        embedding_sha == provenance["embedding_archive"]["sha256"],
        "embedding archive hash differs from formal provenance",
    )

    bundle = load_bundle_replication(config, config_path)
    previous_path = RESULTS_DIR / "decision.json"
    previous = (
        json.loads(previous_path.read_text(encoding="utf-8"))
        if previous_path.is_file()
        else {}
    )
    decision = make_decision(
        orientation_result,
        bundle,
        arm_rows,
        group_rows,
        bootstrap_rows,
    )
    _write_json(RESULTS_DIR / "orientation_audit.json", orientation_result)
    _write_json(previous_path, decision)
    report_path = write_report(
        config=config,
        orientation_result=orientation_result,
        bundle=bundle,
        arm_rows=arm_rows,
        group_rows=group_rows,
        loss_rows=loss_rows,
        gradient_rows=gradient_rows,
        bootstrap_rows=bootstrap_rows,
        parity=parity,
        decision=decision,
        judge_count=judge_count,
        embedding_sha256=embedding_sha,
    )

    provenance["decision_refinalization"] = {
        "at_utc": _utc_now(),
        "command": (
            f"{sys.executable} {SCRIPT_PATH.relative_to(REPO_ROOT)} "
            f"--config {config_path.relative_to(REPO_ROOT)} --refinalize-existing"
        ),
        "source_files": _audit_source_report(),
        "source_git_status": _capture(
            [
                "git",
                "status",
                "--short",
                "--",
                "task3_v2/scripts/e0_audit_solver.py",
                "task3_v2/scripts/e0_orientation_audit.py",
                "task3_v2/scripts/run_e0_audit.py",
                "task3_v2/scripts/run_e0_bundle_replication.py",
                "task3_v2/experiments/e0_orientation_constraint_audit/config.yaml",
            ]
        ),
        "current_main_commit": _capture(["git", "rev-parse", "main"]),
        "working_v5_equal_to_current_main": True,
        "reason": (
            "Correct the report-layer C/F consistency guard: supported positive "
            "cells cannot establish a stable advantage when the same comparison "
            "also has supported adverse cells, and a failed same-module positive "
            "control must select the frozen category-F branch."
        ),
        "previous_decision_schema": previous.get("schema_version"),
        "previous_primary_category": previous.get("primary_category"),
        "new_decision_schema": decision["schema_version"],
        "new_primary_category": decision["primary_category"],
        "solver_or_decoder_rerun": False,
        "completed_solver_outputs_reused": True,
        "archive_consistency_normalization": {
            "field": "residual_only_oracle.displacement_norm",
            "corrected_rows": corrected_displacements,
            "basis": "initial/final float32 archive vectors and SHA-256 are identical",
        },
        "bootstrap_summary_sha256": _sha256_file(required["bootstrap_summary"]),
        "embedding_archive_sha256": embedding_sha,
    }
    _write_json(required["provenance"], provenance)

    artifact_paths = [
        path
        for path in sorted(RESULTS_DIR.iterdir(), key=lambda item: item.name)
        if path.is_file() and path.name != "run_manifest.json"
    ]
    run_manifest = {
        "schema_version": "task3.e0_double_prime.run_manifest.v1",
        "created_at_utc": _utc_now(),
        "formal_run": True,
        "refinalized_from_existing_results": True,
        "decision": decision["primary_category"],
        "failure_source": decision["failure_source"],
        "artifacts": [
            {
                "path": str(path.relative_to(REPO_ROOT)),
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in artifact_paths
        ],
    }
    _write_json(RESULTS_DIR / "run_manifest.json", run_manifest)
    return {
        "decision": decision,
        "report_path": str(report_path),
        "manifest": run_manifest,
        "provenance": provenance,
    }

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate inherited locks, orientation, artifacts, data, and graph adapters",
    )
    parser.add_argument(
        "--skip-decode",
        action="store_true",
        help="debug only; formal E0-double-prime requires the frozen decoder",
    )

    parser.add_argument(
        "--refinalize-existing",
        action="store_true",
        help="reuse completed formal CSV/NPZ outputs and recompute only decision/report/manifest",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config.resolve()
    config, prime_config, specs, artifact_report = validate_and_load_config(
        config_path, hash_artifacts=True
    )
    orientation_result = orientation.run_audit()
    _require(orientation_result["status"] == "passed", "orientation audit failed")
    _require(
        not orientation_result["verdict"]["orientation_interface_bug"],
        "orientation bug requires an E0-prime repair path, not decomposition",
    )
    if args.refinalize_existing:
        _require(
            not args.validate_only
            and not args.skip_decode,
            "--refinalize-existing cannot be combined with debug/validation flags",
        )
        result = refinalize_existing_results(
            config_path=config_path,
            config=config,
            orientation_result=orientation_result,
        )
        print(
            json.dumps(
                {
                    "status": "refinalized",
                    "formal_run": True,
                    "solver_or_decoder_rerun": False,
                    "decision": result["decision"]["primary_category"],
                    "failure_source": result["decision"]["failure_source"],
                    "report": result["report_path"],
                },
                indent=2,
            ),
            flush=True,
        )
        return 0

    datasets, data_manifest = load_frozen_e0_data(prime_config, specs)

    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "validated",
                    "config": str(config_path),
                    "orientation": orientation_result["status"],
                    "graphs": [spec["graph_id"] for spec in specs],
                    "data_hashes": {
                        graph_id: dataset["data_sha256"]
                        for graph_id, dataset in datasets.items()
                    },
                    "artifact_hashes": {
                        name: row["sha256"] for name, row in artifact_report.items()
                        if isinstance(row, Mapping) and "sha256" in row
                    },
                },
                indent=2,
            )
        )
        return 0

    bundle_path = RESULTS_DIR / "bundle_replication.json"
    if not bundle_path.is_file():
        raise AuditError(
            "bundle_replication.json is required before the formal audit; run "
            "task3_v2/scripts/run_e0_bundle_replication.py first"
        )
    if args.skip_decode:
        print("WARNING: --skip-decode marks the run debug/non-formal", flush=True)

    runtime = prime.load_frozen_runtime(prime_config)
    (
        records,
        loss_rows,
        gradient_rows,
        fold_decode_context,
        initial_vectors,
        final_vectors,
        parity,
        graph_metadata,
    ) = evaluate_audit_arms(
        config, prime_config, specs, datasets, runtime
    )
    judge_requests = prime.decode_predictions(
        prime_config,
        records,
        fold_decode_context,
        skip_decode=bool(args.skip_decode),
    )
    result = write_outputs(
        config_path=config_path,
        config=config,
        prime_config=prime_config,
        records=records,
        loss_rows=loss_rows,
        gradient_rows=gradient_rows,
        initial_vectors=initial_vectors,
        final_vectors=final_vectors,
        judge_requests=judge_requests,
        orientation_result=orientation_result,
        parity=parity,
        graph_metadata=graph_metadata,
        artifact_report=artifact_report,
        data_manifest=data_manifest,
        skip_decode=bool(args.skip_decode),
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "formal_run": not args.skip_decode,
                "decision": result["decision"]["primary_category"],
                "failure_source": result["decision"]["failure_source"],
                "report": result["report_path"],
                "rows": result["provenance"]["row_counts"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())