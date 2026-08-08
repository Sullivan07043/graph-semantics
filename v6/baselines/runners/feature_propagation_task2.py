"""Run Feature Propagation as a leakage-safe Task 2 baseline.

Each fold embeds only its visible observed labels with frozen base
``intfloat/e5-large-v2``.  Those vectors are clamped anchors for Feature
Propagation; latent vectors are produced before this runner reads or embeds
the gold latent descriptions.  Evaluation is the same dictionary-free,
LLM-free Hungarian latent Match-ACC used by ``v6/run_task2.py``.

Example from the repository root::

    python -m v6.baselines.runners.feature_propagation_task2 --datasets report19
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence

import numpy as np


V6_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = V6_DIR.parent
WORKSPACE_ROOT = REPO_ROOT.parent
if str(V6_DIR) not in sys.path:
    sys.path.insert(0, str(V6_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("HF_CACHE", str(WORKSPACE_ROOT / ".hf_cache"))

from v6.baselines.feature_propagation import (  # noqa: E402
    FEATURE_PROP_VERSION,
    predict_task2_latent_embeddings,
)
from v6.baselines.protocol import (  # noqa: E402
    REPORT_DATASETS,
    atomic_write_json,
    config_hash,
    outer_folds,
    report_loaders,
    select_datasets,
)


METHOD_ID = "feature-propagation"
METHOD_LABEL = "Feature Propagation"
PROTOCOL_VERSION = "feature-propagation-task2-report19-v3"
ENCODER_MODEL = "intfloat/e5-large-v2"
ENCODER_REVISION = "f169b11e22de13617baa190a028a32f3493550b6"
ENCODER_PREFIX = "query: "
LOCAL_SOURCE_FILES = (
    "v6/baselines/runners/feature_propagation_task2.py",
    "v6/baselines/feature_propagation.py",
    "v6/baselines/protocol.py",
    "v6/testbeds.py",
    "v6/pool.py",
    "v6/graph.py",
)
TextEncoder = Callable[[Sequence[str]], np.ndarray]


@dataclass(frozen=True)
class FoldPrediction:
    """Prediction-only state created without accessing latent gold text."""

    latent_nodes: tuple[str, ...]
    embeddings: np.ndarray
    masked_observed_indices: tuple[int, ...]
    visible_observed_indices: tuple[int, ...]
    nearest_visible_labels: tuple[tuple[str, ...], ...]


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="report19")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--embed-batch-size", type=int, default=64)
    parser.add_argument("--max-iter", type=int, default=40)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--fallback", choices=("zeros", "mean"), default="zeros")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--max-dataset-folds",
        type=int,
        default=0,
        help="execution-only smoke-test limit; zero runs every selected fold",
    )
    args = parser.parse_args(argv)
    if args.folds != 5:
        parser.error("the frozen report protocol requires exactly five folds")
    if args.embed_batch_size < 1 or args.max_iter < 1:
        parser.error("batch size and max-iter must be positive")
    if not np.isfinite(args.tol) or args.tol < 0:
        parser.error("tol must be finite and non-negative")
    if args.max_dataset_folds < 0:
        parser.error("max-dataset-folds cannot be negative")
    return args


def _timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _git_dirty() -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return bool(result.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _local_source_sha256() -> dict[str, str]:
    """Hash every repository source file used by this frozen runner."""

    return {
        relative_path: _file_sha256(REPO_ROOT / relative_path)
        for relative_path in LOCAL_SOURCE_FILES
    }


def _encoder_provenance() -> dict[str, Any]:
    """Return the immutable text-encoder identity included in the config hash."""

    return {
        "model_id": ENCODER_MODEL,
        "revision": ENCODER_REVISION,
        "prefix": ENCODER_PREFIX,
        "frozen": True,
    }


def _embedding_function(batch_size: int) -> TextEncoder:
    import torch
    from sentence_transformers import SentenceTransformer

    device = None
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        device = "cuda:1"
    model = SentenceTransformer(
        ENCODER_MODEL,
        revision=ENCODER_REVISION,
        device=device,
        cache_folder=os.environ.get("HF_CACHE", str(WORKSPACE_ROOT / ".hf_cache")),
    )

    def embed(texts: Sequence[str]) -> np.ndarray:
        matrix = np.asarray(
            model.encode(
                [ENCODER_PREFIX + text for text in texts],
                batch_size=batch_size,
                normalize_embeddings=True,
            ),
            dtype=np.float64,
        )
        if matrix.ndim != 2 or matrix.shape[0] != len(texts):
            raise ValueError(
                "text encoder must return [len(texts), dimension]; "
                f"got {matrix.shape}"
            )
        if not np.all(np.isfinite(matrix)):
            raise ValueError("text encoder returned a non-finite value")
        return matrix

    return embed


def _normalise_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("embedding matrix must be two-dimensional")
    return matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)


def latent_match_details(
    predicted_embeddings: np.ndarray,
    gold_embeddings: np.ndarray,
) -> tuple[float | None, np.ndarray, list[int] | None]:
    """Return current Task 2 Hungarian Match-ACC and its audit details."""

    predicted = np.asarray(predicted_embeddings, dtype=float)
    gold = np.asarray(gold_embeddings, dtype=float)
    if predicted.ndim != 2 or gold.ndim != 2 or predicted.shape != gold.shape:
        raise ValueError(
            "predicted and gold embeddings must have the same [latent, dimension] shape"
        )
    if not np.all(np.isfinite(predicted)) or not np.all(np.isfinite(gold)):
        raise ValueError("predicted and gold embeddings must be finite")
    similarities = _normalise_rows(predicted) @ _normalise_rows(gold).T
    if len(predicted) <= 1:
        return None, similarities, None

    from scipy.optimize import linear_sum_assignment

    rows, columns = linear_sum_assignment(-similarities)
    if not np.array_equal(rows, np.arange(len(predicted))):
        raise RuntimeError("unexpected non-square Hungarian assignment")
    score = float(np.mean(columns == np.arange(len(predicted))))
    return score, similarities, [int(value) for value in columns]


def _nearest_visible_labels(
    latent_embeddings: np.ndarray,
    visible_embeddings: np.ndarray,
    visible_labels: Sequence[str],
    *,
    top_k: int = 3,
) -> tuple[tuple[str, ...], ...]:
    """Provide a leakage-safe qualitative readout, never used by Match-ACC."""

    similarities = _normalise_rows(latent_embeddings) @ _normalise_rows(
        visible_embeddings
    ).T
    count = min(top_k, len(visible_labels))
    output = []
    for row in similarities:
        # Stable sort makes equal-similarity zero-vector fallbacks reproducible.
        order = np.argsort(-row, kind="stable")[:count]
        output.append(tuple(str(visible_labels[index]) for index in order))
    return tuple(output)


def predict_fold(
    dataset: Mapping[str, Any],
    masked_observed_indices: Sequence[int],
    embed: TextEncoder,
    *,
    max_iter: int = 40,
    tol: float = 1e-8,
    fallback: str = "zeros",
) -> FoldPrediction:
    """Predict one fold without reading ``dataset['latent_gt']`` or ``X``."""

    graph = dataset["graph"]
    labels = dataset["labels"]
    observed = list(graph.observed)
    masked = tuple(sorted(int(index) for index in masked_observed_indices))
    if len(set(masked)) != len(masked):
        raise ValueError("masked_observed_indices must be unique")
    if any(index < 0 or index >= len(observed) for index in masked):
        raise IndexError("masked observed index is outside graph.observed")

    hidden = set(masked)
    visible = tuple(index for index in range(len(observed)) if index not in hidden)
    if not visible:
        raise ValueError("Feature Propagation needs at least one visible observed label")
    visible_nodes = [observed[index] for index in visible]
    try:
        visible_labels = [str(labels[node]) for node in visible_nodes]
    except KeyError as exc:
        raise ValueError(f"missing observed label: {exc.args[0]}") from exc

    # This is intentionally the only text-encoding call in prediction.  Hidden
    # observed labels and all latent descriptions remain untouched.
    visible_embeddings = np.asarray(embed(visible_labels), dtype=float)
    if visible_embeddings.ndim != 2 or visible_embeddings.shape[0] != len(visible):
        raise ValueError("visible-label encoder output has the wrong shape")
    anchors = {
        node: visible_embeddings[position]
        for position, node in enumerate(visible_nodes)
    }
    latent_nodes = tuple(graph.latents)
    predicted = predict_task2_latent_embeddings(
        graph,
        anchors,
        latent_nodes=latent_nodes,
        max_iter=max_iter,
        tol=tol,
        fallback=fallback,
    )
    matrix = np.stack([predicted[node] for node in latent_nodes])
    return FoldPrediction(
        latent_nodes=latent_nodes,
        embeddings=matrix,
        masked_observed_indices=masked,
        visible_observed_indices=visible,
        nearest_visible_labels=_nearest_visible_labels(
            matrix, visible_embeddings, visible_labels
        ),
    )


def evaluate_fold(
    prediction: FoldPrediction,
    dataset: Mapping[str, Any],
    embed: TextEncoder,
) -> dict[str, Any]:
    """Read latent gold text only after a prediction has been frozen."""

    latent_gt = dataset["latent_gt"]
    missing = [node for node in prediction.latent_nodes if node not in latent_gt]
    if missing:
        raise ValueError(f"latent_gt does not cover graph latents: {missing}")
    gold_texts = [str(latent_gt[node]) for node in prediction.latent_nodes]
    gold_embeddings = np.asarray(embed(gold_texts), dtype=float)
    score, similarities, assignment = latent_match_details(
        prediction.embeddings, gold_embeddings
    )
    return {
        "latent_match_acc": score,
        "latent_similarity_matrix": similarities.tolist(),
        "hungarian_assignment": assignment,
    }


def _fold_path(root: Path, dataset: str, fold: int) -> Path:
    return root / "folds" / dataset / f"fold_{fold:02d}.json"


def _load_fold(path: Path, expected_hash: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid resumable fold record {path}: {exc}") from exc
    if record.get("config_hash") != expected_hash:
        raise RuntimeError(f"fold record {path} belongs to a different configuration")
    if record.get("method_id") != METHOD_ID:
        raise RuntimeError(f"fold record {path} has the wrong baseline identity")
    return record


def _mean(values: Sequence[Any]) -> float | None:
    selected = [
        float(value)
        for value in values
        if value is not None and np.isfinite(value)
    ]
    return float(np.mean(selected)) if selected else None


def _build_summary(
    root: Path,
    config: Mapping[str, Any],
    cfg_hash: str,
) -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    all_scores: list[float] = []
    completed = 0
    for dataset_name in config["datasets"]:
        folds = []
        for fold in range(int(config["folds"])):
            record = _load_fold(_fold_path(root, dataset_name, fold), cfg_hash)
            if record is not None:
                folds.append(record)
        scores = [record.get("latent_match_acc") for record in folds]
        dataset_score = _mean(scores)
        if dataset_score is not None:
            all_scores.append(dataset_score)
        completed += len(folds)
        datasets[dataset_name] = {
            "completed_folds": len(folds),
            "latent_count": (folds[0]["latent_count"] if folds else None),
            "latent_match_acc": dataset_score,
        }

    summary = {
        "protocol": PROTOCOL_VERSION,
        "method": METHOD_LABEL,
        "method_id": METHOD_ID,
        "method_version": FEATURE_PROP_VERSION,
        "config_hash": cfg_hash,
        "completed_dataset_folds": completed,
        "expected_dataset_folds": len(config["datasets"]) * int(config["folds"]),
        "dataset_macro_match_acc": _mean(all_scores),
        "llm_judge": None,
        "datasets": datasets,
        "generated_at": _timestamp(),
    }
    atomic_write_json(root / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    datasets = select_datasets(args.datasets)
    root = (
        args.output_dir
        or V6_DIR
        / "outputs"
        / "feature_propagation_task2"
        / f"report19_seed{args.seed}_v3"
    ).resolve()
    implementation_sha256 = _local_source_sha256()
    encoder = _encoder_provenance()
    config = {
        "protocol": PROTOCOL_VERSION,
        "datasets": datasets,
        "report_dataset_order": list(REPORT_DATASETS),
        "folds": args.folds,
        "seed": args.seed,
        "method": METHOD_LABEL,
        "method_id": METHOD_ID,
        "method_version": FEATURE_PROP_VERSION,
        "encoder": encoder,
        "graph_projection": "binary undirected",
        "known_features": "fold-visible observed-label embeddings only",
        "max_iter": args.max_iter,
        "tol": args.tol,
        "fallback": args.fallback,
        "metric": "dictionary-free latent Hungarian Match-ACC",
        "llm_judge_run": False,
        "implementation_sha256": implementation_sha256,
    }
    cfg_hash = config_hash(config)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("config_hash") != cfg_hash:
            raise RuntimeError(
                f"output directory has a different config: {root}; "
                "choose another --output-dir"
            )
    else:
        atomic_write_json(
            manifest_path,
            {
                "config": config,
                "config_hash": cfg_hash,
                "git_commit": _git_commit(),
                "git_dirty": _git_dirty(),
                "encoder": encoder,
                "implementation_sha256": implementation_sha256,
                "created_at": _timestamp(),
            },
        )

    status = {
        "state": "running",
        "config_hash": cfg_hash,
        "current": None,
        "completed_new_folds": 0,
        "resumed_folds": 0,
        "updated_at": _timestamp(),
    }
    atomic_write_json(root / "status.json", status)
    embed = _embedding_function(args.embed_batch_size)
    loaders = report_loaders()
    visited = 0
    limited = False

    try:
        for dataset_name in datasets:
            dataset = loaders[dataset_name]()
            observed = list(dataset["graph"].observed)
            folds = outer_folds(len(observed), args.folds, args.seed)
            for fold, mask in enumerate(folds):
                if args.max_dataset_folds and visited >= args.max_dataset_folds:
                    limited = True
                    break
                visited += 1
                path = _fold_path(root, dataset_name, fold)
                if _load_fold(path, cfg_hash) is not None:
                    status["resumed_folds"] += 1
                    continue
                status["current"] = {"dataset": dataset_name, "fold": fold}
                status["updated_at"] = _timestamp()
                atomic_write_json(root / "status.json", status)

                # Prediction is completed and returned before evaluate_fold is
                # allowed to look up latent_gt.
                prediction = predict_fold(
                    dataset,
                    mask,
                    embed,
                    max_iter=args.max_iter,
                    tol=args.tol,
                    fallback=args.fallback,
                )
                evaluation = evaluate_fold(prediction, dataset, embed)
                record = {
                    "task": 2,
                    "method": METHOD_LABEL,
                    "method_id": METHOD_ID,
                    "method_version": FEATURE_PROP_VERSION,
                    "dataset": dataset_name,
                    "fold": fold,
                    "masked_observed_indices": list(
                        prediction.masked_observed_indices
                    ),
                    "visible_observed_indices": list(
                        prediction.visible_observed_indices
                    ),
                    "latent_count": len(prediction.latent_nodes),
                    "predictions": [
                        {
                            "latent_index": index,
                            # Recorded after prediction; the node identifier is
                            # never encoded or used as semantic input.
                            "latent_node": node,
                            "embedding": prediction.embeddings[index].tolist(),
                            "embedding_norm": float(
                                np.linalg.norm(prediction.embeddings[index])
                            ),
                            "nearest_visible_labels": list(
                                prediction.nearest_visible_labels[index]
                            ),
                        }
                        for index, node in enumerate(prediction.latent_nodes)
                    ],
                    **evaluation,
                    "config_hash": cfg_hash,
                    "llm_judge": None,
                    "completed_at": _timestamp(),
                }
                atomic_write_json(path, record)
                status["completed_new_folds"] += 1
                print(
                    f"[{dataset_name}] fold {fold + 1}/{args.folds}: "
                    f"latent_match_acc={evaluation['latent_match_acc']}",
                    flush=True,
                )
            if limited:
                break

        summary = _build_summary(root, config, cfg_hash)
        status["state"] = "limited" if limited else "complete"
        status["current"] = None
        status["completed_dataset_folds"] = summary[
            "completed_dataset_folds"
        ]
        status["expected_dataset_folds"] = summary[
            "expected_dataset_folds"
        ]
        status["updated_at"] = _timestamp()
        atomic_write_json(root / "status.json", status)
        print(
            f"[saved {root}] {summary['completed_dataset_folds']}/"
            f"{summary['expected_dataset_folds']} folds; "
            f"dataset_macro_match_acc={summary['dataset_macro_match_acc']}",
            flush=True,
        )
        return 0
    except Exception:
        status["state"] = "failed"
        status["updated_at"] = _timestamp()
        atomic_write_json(root / "status.json", status)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
