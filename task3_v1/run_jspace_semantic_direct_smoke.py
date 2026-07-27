"""Direct, API-free semantic-completion diagnostic on frozen J-space artifacts.

This is not a formal Stage-2 result because the current Stage-1 graph did not
pass intervention validation.  It answers a narrower engineering question:
given the frozen innovation matrix and stable CauScale support, can the current
v5 L3+L2 semantic solver recover concept-group-masked labels better than simple
no-graph baselines?
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

# The required encoder is already cached locally. Keep this diagnostic
# deterministic and prevent library metadata checks from attempting network I/O.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
V5 = REPO / "v5"
sys.path.insert(0, str(V5))

import encode  # noqa: E402
import graph as graph_mod  # noqa: E402
import l2_modules  # noqa: E402
import l2_solver  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def norm_rows(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def match_acc(predicted: np.ndarray, truth: np.ndarray) -> float:
    similarity = norm_rows(predicted) @ norm_rows(truth).T
    rows, cols = linear_sum_assignment(-similarity)
    return float(np.mean(cols[np.argsort(rows)] == np.arange(len(truth))))


def exact_acc(
    predicted: np.ndarray,
    masked_concept_indices: list[int],
    all_truth: np.ndarray,
) -> float:
    similarity = norm_rows(predicted) @ norm_rows(all_truth).T
    nearest = np.argmax(similarity, axis=1)
    return float(np.mean(nearest == np.asarray(masked_concept_indices)))


def topk_acc(
    predicted: np.ndarray,
    masked_concept_indices: list[int],
    all_truth: np.ndarray,
    k: int = 5,
) -> float:
    similarity = norm_rows(predicted) @ norm_rows(all_truth).T
    top = np.argpartition(-similarity, kth=min(k, similarity.shape[1]) - 1, axis=1)[
        :, :k
    ]
    return float(
        np.mean(
            [
                concept_index in top[row]
                for row, concept_index in enumerate(masked_concept_indices)
            ]
        )
    )


def load_l3_label_embeddings(
    concepts: list[str],
) -> tuple[np.ndarray, dict[str, str]]:
    dictionary = V5 / "outputs" / "concept_bank_l3.npz"
    checkpoint = V5 / "outputs" / "l3_lora.pt"
    os.environ["GRAPHSEM_DICT"] = str(dictionary)
    bank = np.load(dictionary, allow_pickle=True)
    if abs(float(bank["lora_version"]) - checkpoint.stat().st_mtime) >= 1.0:
        raise RuntimeError("L3 dictionary and LoRA checkpoint versions do not match")
    names = bank["names"]
    first_index = {}
    exact_index = {}
    for index, name in enumerate(names):
        text = str(name).strip()
        first_index.setdefault(text.casefold(), index)
        exact_index.setdefault(text, index)
    indices = []
    for concept in concepts:
        index = exact_index.get(concept, first_index.get(concept.casefold()))
        if index is None:
            raise KeyError(f"Concept is absent from the frozen L3 dictionary: {concept}")
        indices.append(index)
    embeddings = np.asarray(bank["emb"][indices], dtype=np.float64)
    return norm_rows(embeddings), {
        "dictionary": str(dictionary),
        "dictionary_sha256": sha256(dictionary),
        "lora_checkpoint": str(checkpoint),
        "lora_checkpoint_sha256": sha256(checkpoint),
        "label_embedding_source": (
            "exact lookup in the frozen L3-reencoded dictionary; the base E5 "
            "model is not reloaded"
        ),
    }

def frozen_graph_and_weights(
    matrix: np.ndarray,
    columns: list[dict],
    stable_edges: list[dict],
    ridge: float,
) -> tuple[graph_mod.Graph, dict[tuple[str, str], float], list[str]]:
    node_names = [f"j{index:03d}" for index in range(len(columns))]
    edge_pairs = [
        (
            node_names[int(edge["source_index"])],
            node_names[int(edge["target_index"])],
        )
        for edge in stable_edges
    ]
    graph = graph_mod.Graph([], node_names, edge_pairs)

    standardized = (matrix - matrix.mean(axis=0)) / np.maximum(
        matrix.std(axis=0, ddof=1), 1e-8
    )
    parents_by_target: dict[int, list[dict]] = defaultdict(list)
    for edge in stable_edges:
        parents_by_target[int(edge["target_index"])].append(edge)

    weights: dict[tuple[str, str], float] = {}
    for target_index, target_edges in parents_by_target.items():
        source_indices = [int(edge["source_index"]) for edge in target_edges]
        design = standardized[:, source_indices]
        target = standardized[:, target_index]
        gram = design.T @ design + ridge * np.eye(len(source_indices))
        beta = np.linalg.solve(gram, design.T @ target)
        for edge, coefficient in zip(target_edges, beta):
            source_index = int(edge["source_index"])
            probability = float(edge["mean_probability"])
            weights[(node_names[source_index], node_names[target_index])] = (
                probability * float(coefficient)
            )
    return graph, weights, node_names


def component_anchor_fraction(
    graph: graph_mod.Graph,
    masked_nodes: set[str],
    visible_nodes: set[str],
) -> float:
    adjacency = {node: set() for node in graph.nodes}
    for source, target in graph.edges:
        adjacency[source].add(target)
        adjacency[target].add(source)
    anchored = 0
    for node in masked_nodes:
        seen = {node}
        stack = [node]
        found = False
        while stack:
            current = stack.pop()
            if current in visible_nodes:
                found = True
                break
            for neighbor in adjacency[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        anchored += int(found)
    return anchored / max(len(masked_nodes), 1)


def aggregate_by_concept(
    node_predictions: dict[str, np.ndarray],
    concepts: list[str],
    masked_concepts: list[str],
    nodes_by_concept: dict[str, list[str]],
) -> np.ndarray:
    rows = []
    for concept in masked_concepts:
        values = np.stack([node_predictions[node] for node in nodes_by_concept[concept]])
        rows.append(norm_rows(values).mean(axis=0))
    return norm_rows(np.stack(rows))


def bootstrap_gain(
    core_cosine: np.ndarray,
    baseline_cosine: np.ndarray,
    seed: int,
    draws: int = 10000,
) -> dict[str, float]:
    differences = core_cosine - baseline_cosine
    rng = np.random.default_rng(seed)
    samples = rng.choice(differences, size=(draws, len(differences)), replace=True).mean(
        axis=1
    )
    return {
        "mean": float(differences.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "bootstrap_probability_gain_le_zero": float(np.mean(samples <= 0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix",
        type=Path,
        default=HERE / "outputs" / "discovery" / "innovation_matrix_1000x128.npy",
    )
    parser.add_argument(
        "--matrix-meta",
        type=Path,
        default=HERE / "outputs" / "discovery" / "innovation_matrix_1000x128.json",
    )
    parser.add_argument(
        "--bootstrap",
        type=Path,
        default=(
            HERE
            / "outputs"
            / "causcale"
            / "innovation_causcale_bootstrap_20.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "outputs" / "semantic" / "jspace_semantic_direct_smoke.json",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--solver-steps", type=int, default=60)
    args = parser.parse_args()

    started = time.time()
    for path in (args.matrix, args.matrix_meta, args.bootstrap):
        if not path.is_file():
            raise FileNotFoundError(path)

    matrix = np.load(args.matrix).astype(np.float64)
    matrix_meta = json.loads(args.matrix_meta.read_text(encoding="utf-8"))
    bootstrap = json.loads(args.bootstrap.read_text(encoding="utf-8"))
    columns = matrix_meta["columns"]
    stable_edges = bootstrap["stable_edges"]
    if matrix.shape != tuple(matrix_meta["shape"]):
        raise ValueError(f"Matrix shape {matrix.shape} != metadata {matrix_meta['shape']}")

    l2_checkpoint = V5 / "outputs" / "l2_mlp.pt"
    weight_module = l2_modules.load(str(l2_checkpoint), device="cpu")

    graph, weights, node_names = frozen_graph_and_weights(
        matrix, columns, stable_edges, args.ridge
    )
    concepts = []
    for column in columns:
        concept = str(column["concept"]).strip()
        if concept not in concepts:
            concepts.append(concept)
    concept_index = {concept: index for index, concept in enumerate(concepts)}
    nodes_by_concept: dict[str, list[str]] = defaultdict(list)
    labels_by_node = {}
    for node, column in zip(node_names, columns):
        concept = str(column["concept"]).strip()
        nodes_by_concept[concept].append(node)
        labels_by_node[node] = concept
    if any(len(nodes_by_concept[concept]) != len(matrix_meta["layers"]) for concept in concepts):
        raise RuntimeError("Concepts do not have one node per selected layer")

    truth, encoder_record = load_l3_label_embeddings(concepts)
    correlation = np.corrcoef(matrix.T)
    np.fill_diagonal(correlation, 0.0)
    correlation = np.clip(correlation, 0.0, None)

    rng = np.random.default_rng(args.seed)
    permutation = rng.permutation(len(concepts))
    folds = [permutation[index:: args.folds].tolist() for index in range(args.folds)]
    fold_records = []
    item_records = []
    all_cosines = {arm: [] for arm in ("uniform", "rawcorr", "core")}

    for fold_number, masked_concept_indices in enumerate(folds):
        masked_concepts = [concepts[index] for index in masked_concept_indices]
        masked_nodes = {
            node for concept in masked_concepts for node in nodes_by_concept[concept]
        }
        visible_nodes = [node for node in node_names if node not in masked_nodes]
        visible_set = set(visible_nodes)
        visible_embeddings = {
            node: truth[concept_index[labels_by_node[node]]] for node in visible_nodes
        }

        node_predictions: dict[str, dict[str, np.ndarray]] = {
            "uniform": {},
            "rawcorr": {},
        }
        visible_indices = np.asarray(
            [int(node[1:]) for node in visible_nodes], dtype=np.int64
        )
        visible_truth = np.stack([visible_embeddings[node] for node in visible_nodes])
        for node in masked_nodes:
            index = int(node[1:])
            uniform_weights = np.ones(len(visible_nodes), dtype=np.float64)
            raw_weights = correlation[index, visible_indices]
            if raw_weights.sum() < 1e-12:
                raw_weights = uniform_weights
            node_predictions["uniform"][node] = (
                uniform_weights / uniform_weights.sum()
            ) @ visible_truth
            node_predictions["rawcorr"][node] = (
                raw_weights / raw_weights.sum()
            ) @ visible_truth

        features = torch.tensor(
            l2_modules.node_features(graph, weights, set(visible_embeddings)),
            dtype=torch.float32,
            device="cpu",
        )
        solved, _ = l2_solver.solve_unrolled(
            graph,
            weights,
            visible_embeddings,
            truth.shape[1],
            weight_module=weight_module,
            K=args.solver_steps,
            inner_lr=2e-2,
            lam_zero=0.3,
            lam_norm=0.1,
            seed=args.seed + fold_number,
            device="cpu",
            train=False,
            feats=features,
        )
        node_predictions["core"] = {
            node: np.asarray(solved[node], dtype=np.float64) for node in masked_nodes
        }

        fold_metrics = {}
        fold_truth = truth[masked_concept_indices]
        for arm in ("uniform", "rawcorr", "core"):
            predicted = aggregate_by_concept(
                node_predictions[arm],
                concepts,
                masked_concepts,
                nodes_by_concept,
            )
            correct_cosine = np.sum(predicted * fold_truth, axis=1)
            all_cosines[arm].extend(correct_cosine.tolist())
            fold_metrics[arm] = {
                "match": match_acc(predicted, fold_truth),
                "exact": exact_acc(predicted, masked_concept_indices, truth),
                "top5": topk_acc(predicted, masked_concept_indices, truth),
                "mean_correct_cosine": float(correct_cosine.mean()),
            }
            for row, concept in enumerate(masked_concepts):
                item_records.append(
                    {
                        "fold": fold_number,
                        "arm": arm,
                        "anonymous_concept_index": int(masked_concept_indices[row]),
                        "evaluation_label": concept,
                        "correct_cosine": float(correct_cosine[row]),
                        "nearest_concept": concepts[
                            int(np.argmax(predicted[row] @ truth.T))
                        ],
                    }
                )

        fold_records.append(
            {
                "fold": fold_number,
                "masked_concept_indices": masked_concept_indices,
                "masked_concept_count": len(masked_concepts),
                "masked_node_count": len(masked_nodes),
                "graph_anchor_fraction_for_masked_nodes": component_anchor_fraction(
                    graph, masked_nodes, visible_set
                ),
                "metrics": fold_metrics,
            }
        )
        print(
            f"fold {fold_number + 1}/{args.folds}: "
            f"core match={fold_metrics['core']['match']:.3f}, "
            f"rawcorr match={fold_metrics['rawcorr']['match']:.3f}, "
            f"anchor={fold_records[-1]['graph_anchor_fraction_for_masked_nodes']:.3f}",
            flush=True,
        )

    summary = {}
    for arm in ("uniform", "rawcorr", "core"):
        summary[arm] = {
            metric: float(
                np.mean([fold["metrics"][arm][metric] for fold in fold_records])
            )
            for metric in ("match", "exact", "top5", "mean_correct_cosine")
        }
    core_cosine = np.asarray(all_cosines["core"])
    raw_cosine = np.asarray(all_cosines["rawcorr"])
    gain = bootstrap_gain(core_cosine, raw_cosine, args.seed)
    supports_effectiveness = bool(
        summary["core"]["match"] > summary["rawcorr"]["match"]
        and summary["core"]["match"] > summary["uniform"]["match"]
        and gain["ci95_low"] > 0.0
    )

    record = {
        "status": "diagnostic_jspace_semantic_completion",
        "formal_stage2_result": False,
        "reason_not_formal_stage2": (
            "The current Stage-1 graph did not pass held-out cross-concept "
            "intervention validation."
        ),
        "question": (
            "Does the current v5 L3+L2 solver recover concept-group-masked "
            "J-space labels better than uniform and raw-correlation baselines?"
        ),
        "supports_direct_jspace_effectiveness": supports_effectiveness,
        "success_rule": (
            "core mean fold MatchAcc must exceed both baselines and the paired "
            "concept-level correct-cosine gain over raw correlation must have a "
            "bootstrap 95% interval above zero"
        ),
        "matrix": str(args.matrix),
        "matrix_sha256": sha256(args.matrix),
        "matrix_metadata": str(args.matrix_meta),
        "bootstrap_graph": str(args.bootstrap),
        "bootstrap_graph_sha256": sha256(args.bootstrap),
        "shape": list(matrix.shape),
        "concept_count": len(concepts),
        "layer_count": len(matrix_meta["layers"]),
        "stable_edge_count": len(stable_edges),
        "stable_cross_concept_edge_count": int(
            sum(
                str(edge["source_concept"]).strip()
                != str(edge["target_concept"]).strip()
                for edge in stable_edges
            )
        ),
        "masking": (
            "five deterministic folds by concept; every layer-coordinate for a "
            "masked concept is hidden together; solver node IDs are anonymous"
        ),
        "edge_weight": (
            "CauScale mean probability multiplied by discovery-only "
            "parent-conditioned standardized ridge coefficient"
        ),
        "solver": {
            "semantic_space": "E5-large-v2 + adopted L3 LoRA",
            "optimizer": "adopted L2 WeightNet unrolled solver",
            "steps": args.solver_steps,
            "device_for_encoder": "not loaded; frozen dictionary lookup",
            "device_for_solver": "cpu",
            "l2_checkpoint": str(l2_checkpoint),
            "l2_checkpoint_sha256": sha256(l2_checkpoint),
            **encoder_record,
        },
        "summary": summary,
        "core_minus_rawcorr_correct_cosine": gain,
        "mean_graph_anchor_fraction_for_masked_nodes": float(
            np.mean(
                [
                    fold["graph_anchor_fraction_for_masked_nodes"]
                    for fold in fold_records
                ]
            )
        ),
        "folds": fold_records,
        "items": item_records,
        "runtime_seconds": time.time() - started,
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "interpretation_boundary": (
            "This tests compatibility and masked semantic recovery on the "
            "current frozen J-space artifacts. It cannot rescue the failed "
            "Stage-1 causal claim or establish general J-space validity."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "summary": summary, "gain": gain,
                      "supports_effectiveness": supports_effectiveness}, indent=2))


if __name__ == "__main__":
    main()
