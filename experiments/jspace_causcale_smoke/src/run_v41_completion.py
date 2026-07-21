"""Evaluate frozen graph-semantics v4.1 on a custom measured-coordinate graph.

This is a Task-1 protocol: every column is measured, but a fold's semantic
labels are hidden from the solver. Held-out labels are used only for metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


THIS_FILE = Path(__file__).resolve()
EXPERIMENT_ROOT = THIS_FILE.parents[1]
REPO_ROOT = THIS_FILE.parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(THIS_FILE.parent))

import encode  # noqa: E402
import metrics  # noqa: E402
import negop  # noqa: E402
import optimize  # noqa: E402
from graph import Graph  # noqa: E402
from pipeline_L3_v1 import lora  # noqa: E402
from pipeline_v4 import core, l2_modules as LM, release  # noqa: E402

from contracts import load_json, sha256_file, validate_dataset  # noqa: E402


def load_adjacency(path: Path) -> tuple[np.ndarray, list[str] | None]:
    if path.suffix == ".npz":
        payload = np.load(path, allow_pickle=False)
        adjacency = payload["adjacency"]
        names = payload["node_names"].astype(str).tolist() if "node_names" in payload else None
    else:
        adjacency = np.load(path)
        names = None
    adjacency = np.asarray(adjacency, dtype=np.uint8)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(f"adjacency must be square; got {adjacency.shape}")
    if not np.isin(adjacency, [0, 1]).all() or np.any(np.diag(adjacency)):
        raise ValueError("adjacency must be binary with zero diagonal")
    return adjacency, names


def graph_from_adjacency(adjacency: np.ndarray, node_ids: list[str]) -> Graph:
    if adjacency.shape != (len(node_ids), len(node_ids)):
        raise ValueError("adjacency shape does not match dataset nodes")
    edges = [
        (node_ids[source], node_ids[target])
        for source, target in np.argwhere(adjacency == 1)
    ]
    return Graph(latents=[], observed=node_ids, edges=edges)


def is_dag(adjacency: np.ndarray) -> bool:
    indegree = adjacency.sum(axis=0).astype(int)
    stack = [int(i) for i in np.flatnonzero(indegree == 0)]
    visited = 0
    while stack:
        node = stack.pop()
        visited += 1
        for child in np.flatnonzero(adjacency[node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                stack.append(int(child))
    return visited == adjacency.shape[0]


def shuffled_graph(adjacency: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(adjacency.shape[0])
    shuffled = adjacency[np.ix_(permutation, permutation)].copy()
    if not is_dag(shuffled):
        raise AssertionError("node permutation must preserve acyclicity")
    return shuffled


def correlation_graph(X: np.ndarray, nodes: list[dict], n_edges: int) -> np.ndarray:
    """Orient strongest correlations using preregistered layer/level order."""
    d = X.shape[1]
    corr = np.nan_to_num(np.abs(np.corrcoef(X.T)), nan=0.0)

    def order(index: int) -> tuple[float, int]:
        metadata = nodes[index]
        stage = metadata.get("layer", metadata.get("level", index))
        return float(stage), index

    candidates = []
    for i in range(d):
        for j in range(i + 1, d):
            source, target = (i, j) if order(i) < order(j) else (j, i)
            candidates.append((float(corr[i, j]), source, target))
    adjacency = np.zeros((d, d), dtype=np.uint8)
    for _, source, target in sorted(candidates, reverse=True)[:n_edges]:
        adjacency[source, target] = 1
    if not is_dag(adjacency):
        raise AssertionError("strict metadata order must produce a DAG")
    return adjacency


def install_frozen_runtime(device: str) -> tuple[torch.nn.Module, torch.nn.Module, dict]:
    manifest = release.load_manifest(str(REPO_ROOT))
    paths = {
        "l3_checkpoint": REPO_ROOT / release.artifact(manifest, "l3_checkpoint")["path"],
        "l3_dictionary": REPO_ROOT / release.artifact(manifest, "l3_dictionary")["path"],
        "l2_checkpoint": REPO_ROOT / release.artifact(manifest, "l2_checkpoint")["path"],
    }
    release.verify_artifact(manifest, str(REPO_ROOT), "l3_checkpoint", str(paths["l3_checkpoint"]))
    release.verify_artifact(
        manifest,
        str(REPO_ROOT),
        "l3_dictionary",
        str(paths["l3_dictionary"]),
        verify_sha256=False,
    )
    release.verify_artifact(manifest, str(REPO_ROOT), "l2_checkpoint", str(paths["l2_checkpoint"]))
    _, l3_sha = lora.install_as_encode_model(encode, str(paths["l3_checkpoint"]), device)
    l2 = LM.load(str(paths["l2_checkpoint"]), device, expected_l3_sha256=l3_sha)
    neg = negop.load().to(device).eval()
    identities = {
        "release": manifest["release_version"],
        "l3_checkpoint_sha256": l3_sha,
        "l2_checkpoint_sha256": sha256_file(paths["l2_checkpoint"]),
        "dictionary_expected_sha256": release.artifact(manifest, "l3_dictionary")["sha256"],
        "dictionary_sha256_checked_this_run": False,
        "solver_steps": release.SOLVER_STEPS,
    }
    return l2, neg, identities


def solve(
    graph: Graph,
    X: np.ndarray,
    label_embeddings: np.ndarray,
    visible: list[int],
    module: torch.nn.Module | None,
    neg: torch.nn.Module,
    seed: int,
    device: str,
) -> dict[str, np.ndarray]:
    obs = list(graph.observed)
    obs_index = {node: index for index, node in enumerate(obs)}
    visible_embeddings = {obs[index]: label_embeddings[index] for index in visible}
    weights, latent_scores = graph.estimate_weights(X, obs_index)
    partial_corr = optimize.partial_residual_corr(graph, X, obs_index, latent_scores)
    item_corr = optimize.marginal_corr(graph, X, obs_index)
    independent_info = graph.reconcile_independent_pairs(X, obs_index, latent_scores)
    residual_pair_info = optimize.leave_pair_out_residual_pairs(graph, X, obs_index)
    item_info = core.prepare_item_identity(
        graph,
        visible_embeddings,
        item_corr,
        X.shape[0],
        neg_op=neg,
        device=device,
    )
    features = torch.tensor(
        LM.node_features(
            graph,
            weights,
            set(visible_embeddings),
            item_info=item_info,
            independent_info=independent_info,
        ),
        dtype=torch.float32,
        device=device,
    )
    marginal_abs = np.abs(np.corrcoef(X.T))
    np.fill_diagonal(marginal_abs, 0.0)
    bridge = {
        "obs": obs,
        "dep_marg": marginal_abs,
        "lam_upper": 0.3,
        "kappa": 0.5,
        "q": 0.7,
    }
    embeddings, _ = core.solve_unrolled(
        graph,
        weights,
        visible_embeddings,
        d=label_embeddings.shape[1],
        weight_module=module,
        K=release.SOLVER_STEPS,
        inner_lr=2e-2,
        lam_zero=0.3,
        lam_norm=0.1,
        seed=seed,
        device=device,
        residual=1.0,
        lam_res=1.0,
        partial_corr=partial_corr,
        lam_dep=0.0,
        dep_corr=None,
        lam_coll=0.0,
        neg_op=neg,
        bridge=bridge,
        n_samples=X.shape[0],
        independent_info=independent_info,
        item_info=item_info,
        train=False,
        feats=features,
        item_corr=item_corr,
        residual_pair_info=residual_pair_info,
    )
    return embeddings


def fold_metrics(predictions: np.ndarray, masked: list[int], targets: np.ndarray) -> dict[str, float]:
    normalized_targets = metrics.norm_rows(targets)
    return {
        "match": metrics.match_acc(predictions, masked, targets),
        "exact": metrics.exact_acc(predictions, masked, normalized_targets),
        "cosine": metrics.true_cosine(predictions, masked, targets),
    }


def mean_summary(folds: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.mean([fold[key] for fold in folds])) for key in folds[0]}


def evaluate_graph(
    name: str,
    adjacency: np.ndarray,
    X: np.ndarray,
    node_ids: list[str],
    targets: np.ndarray,
    folds: list[list[int]],
    l2: torch.nn.Module,
    neg: torch.nn.Module,
    device: str,
) -> tuple[dict, list[dict]]:
    graph = graph_from_adjacency(adjacency, node_ids)
    if not graph.edges:
        raise ValueError(f"graph arm {name!r} has no edges; v4.1 generation equations are undefined")
    summary = {"v4.1": [], "unit_multipliers": []}
    records: list[dict] = []
    for fold_number, masked in enumerate(folds):
        masked_set = set(masked)
        visible = [index for index in range(len(node_ids)) if index not in masked_set]
        for arm, module in [("v4.1", l2), ("unit_multipliers", None)]:
            started = time.perf_counter()
            embeddings = solve(
                graph,
                X,
                targets,
                visible,
                module,
                neg,
                fold_number,
                device,
            )
            predictions = np.stack([embeddings[node_ids[index]] for index in masked])
            scores = fold_metrics(predictions, masked, targets)
            scores["seconds"] = time.perf_counter() - started
            summary[arm].append(scores)
            for row, index in enumerate(masked):
                p = predictions[row] / (np.linalg.norm(predictions[row]) + 1e-9)
                t = targets[index] / (np.linalg.norm(targets[index]) + 1e-9)
                records.append(
                    {
                        "graph": name,
                        "arm": arm,
                        "fold": fold_number,
                        "node_id": node_ids[index],
                        "true_cosine": float(p @ t),
                    }
                )
        print(f"{name}: fold {fold_number + 1}/{len(folds)} complete", flush=True)
    return {arm: mean_summary(values) for arm, values in summary.items()}, records


def evaluate_no_graph_baselines(
    X: np.ndarray,
    targets: np.ndarray,
    folds: list[list[int]],
) -> dict[str, dict[str, float]]:
    corr = np.nan_to_num(np.corrcoef(X.T), nan=0.0)
    np.fill_diagonal(corr, 0.0)
    out: dict[str, list[dict[str, float]]] = {"visible_mean": [], "positive_correlation": []}
    for masked in folds:
        masked_set = set(masked)
        visible = [index for index in range(X.shape[1]) if index not in masked_set]
        mean_prediction = np.repeat(targets[visible].mean(axis=0, keepdims=True), len(masked), axis=0)
        out["visible_mean"].append(fold_metrics(mean_prediction, masked, targets))
        predictions = []
        for index in masked:
            weights = np.clip(corr[index, visible], 0.0, None)
            if weights.sum() <= 1e-9:
                weights = np.ones(len(visible))
            predictions.append((weights / weights.sum()) @ targets[visible])
        out["positive_correlation"].append(fold_metrics(np.stack(predictions), masked, targets))
    return {arm: mean_summary(values) for arm, values in out.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.folds < 2:
        raise ValueError("at least two folds are required")

    dataset_report = validate_dataset(args.dataset)
    X_all = np.load(args.dataset / "X.npy").astype(np.float64)
    intervention_mask = np.load(args.dataset / "interventions.npy")
    clean_rows = intervention_mask.sum(axis=1) == 0
    if not np.any(clean_rows):
        raise ValueError("v4.1 evaluation requires at least one observational row")
    # Graph discovery may use interventions, but semantic edge weights and all
    # correlation baselines must be estimated from the same observational rows.
    X = X_all[clean_rows]
    nodes = load_json(args.dataset / "nodes.json")
    labels = load_json(args.dataset / "labels.json")
    node_ids = dataset_report["node_ids"]
    adjacency, graph_names = load_adjacency(args.graph)
    if graph_names is not None and graph_names != node_ids:
        raise ValueError("graph node_names do not exactly match dataset column order")
    if not is_dag(adjacency):
        raise ValueError("input graph must be acyclic before v4.1 evaluation")

    torch.set_num_threads(4)
    l2, neg, runtime = install_frozen_runtime(args.device)
    targets = encode.embed([labels[node] for node in node_ids])
    rng = np.random.default_rng(0)
    permutation = rng.permutation(len(node_ids))
    folds = [sorted(int(index) for index in permutation[offset:: args.folds]) for offset in range(args.folds)]

    graph_arms = {
        "input": adjacency,
        "degree_matched_shuffle": shuffled_graph(adjacency, seed=20260721),
        "correlation": correlation_graph(X, nodes, n_edges=int(adjacency.sum())),
    }
    oracle_path = args.dataset / "oracle_graph.npy"
    if oracle_path.is_file():
        graph_arms["oracle"] = np.load(oracle_path).astype(np.uint8)

    result = {
        "status": "fixture_result_not_jspace_evidence"
        if "fixture" in dataset_report["manifest"].get("evidence_status", "")
        else "jspace_task1_result",
        "dataset": dataset_report,
        "input_graph": str(args.graph.resolve()),
        "data_rows_used": {
            "policy": "observational_only",
            "n_rows": int(clean_rows.sum()),
            "excluded_intervention_rows": int((~clean_rows).sum()),
        },
        "runtime": runtime,
        "folds": folds,
        "baselines": evaluate_no_graph_baselines(X, targets, folds),
        "graph_arms": {},
        "records": [],
    }
    for name, arm_adjacency in graph_arms.items():
        arm_summary, records = evaluate_graph(
            name,
            arm_adjacency,
            X,
            node_ids,
            targets,
            folds,
            l2,
            neg,
            args.device,
        )
        result["graph_arms"][name] = {
            "n_edges": int(arm_adjacency.sum()),
            "summary": arm_summary,
        }
        result["records"].extend(records)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    compact = {
        "status": result["status"],
        "baselines": result["baselines"],
        "graph_arms": {name: value["summary"] for name, value in result["graph_arms"].items()},
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
