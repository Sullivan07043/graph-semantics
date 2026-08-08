"""Evaluate target-excluded GraphMAE checkpoints on Task 1 observed labels.

The checkpoints were trained only to reconstruct masked observed-label
embeddings on other development graphs.  For every target dataset this runner
uses its existing LODO checkpoint, supplies only the fold-visible item labels,
and freezes all five prediction artifacts before embedding any hidden/gold item
text for Match-ACC.  No LLM judge or project LoRA is used.

Example::

    python -m v6.baselines.runners.graphmae_task1 --datasets report19 \
        --device cuda --encoder-device cuda:0
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import os
import platform
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np


V6_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = V6_DIR.parent
WORKSPACE_ROOT = REPO_ROOT.parent
if str(V6_DIR) not in sys.path:
    sys.path.insert(0, str(V6_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pool
from v6.baselines.graphmae_gcn import (  # noqa: E402
    METHOD_VERSION as CHECKPOINT_METHOD_VERSION,
    GraphMAEBaseline,
    file_sha256,
)
from v6.baselines.protocol import (  # noqa: E402
    REPORT_DATASETS,
    outer_folds,
    report_loaders,
    select_datasets,
)
from v6.baselines.runners.graphmae_task2 import (  # noqa: E402
    ENCODER_KEY,
    ENCODER_MODEL,
    ENCODER_PREFIX,
    PROTOCOL_VERSION as CHECKPOINT_PROTOCOL_VERSION,
    FrozenE5Encoder,
    resolve_encoder_revision,
    training_datasets_for,
    validate_checkpoint_metadata,
)


ARM = "graphmae_gcn_lodo_observed_reconstruction"
EVALUATION_METHOD_VERSION = "graphmae-gcn-task1-observed-readout-v1"
PROTOCOL_VERSION = "graphmae-task1-visible-label-folds-lodo-v1"
PINNED_ENCODER_REVISION = "f169b11e22de13617baa190a028a32f3493550b6"
TRAINING_BASELINE_SOURCE_SHA256 = (
    "2699cf71c701931850fd13993dc1736c8e175bb8e4399febbfa5e545971de586"
)
DEFAULT_CHECKPOINT_DIR = V6_DIR / "outputs" / "graphmae_task2_lodo" / "checkpoints"
GRAPHMAE_SOURCE = V6_DIR / "baselines" / "graphmae_gcn.py"

EXPECTED_TRAINING_CONFIG = {
    "hidden_dim": 128,
    "encoder_layers": 2,
    "decoder_layers": 2,
    "dropout": 0.0,
    "mask_rate": 0.5,
    "masks_per_graph": 1,
    "loss_alpha": 2.0,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "epochs": 200,
    "grad_clip": 1.0,
    "seed": 0,
    "deterministic": True,
}


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.stem}.", suffix=".npz"
    )
    os.close(descriptor)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _git_state() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        dirty: bool | None = bool(status.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        dirty = None
    return {"commit": commit, "dirty": dirty}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _prediction_path(root: Path, dataset: str, fold: int) -> Path:
    return root / "predictions" / dataset / f"fold_{fold:02d}.npz"


def _prediction_metadata_path(root: Path, dataset: str, fold: int) -> Path:
    return root / "predictions" / dataset / f"fold_{fold:02d}.json"


def _fold_metric_path(root: Path, dataset: str, fold: int) -> Path:
    return root / "fold_metrics" / dataset / f"fold_{fold:02d}.json"


def _freeze_path(root: Path, dataset: str) -> Path:
    return root / "generation_frozen" / f"{dataset}.json"


def _normalise_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values / (np.linalg.norm(values, axis=1, keepdims=True) + 1e-9)


def task1_match_acc(
    predictions: np.ndarray,
    masked_indices: Sequence[int],
    all_gold_embeddings: np.ndarray,
) -> float:
    """The canonical fold-local Hungarian Match-ACC used by Task 1."""

    from scipy.optimize import linear_sum_assignment

    predicted = np.asarray(predictions, dtype=np.float64)
    masked = np.asarray([int(index) for index in masked_indices], dtype=int)
    gold = np.asarray(all_gold_embeddings, dtype=np.float64)
    if predicted.ndim != 2 or gold.ndim != 2 or predicted.shape[1] != gold.shape[1]:
        raise ValueError("prediction and gold embeddings must be compatible matrices")
    if predicted.shape[0] != len(masked) or not len(masked):
        raise ValueError("one non-empty prediction row is required per masked index")
    if np.min(masked) < 0 or np.max(masked) >= len(gold):
        raise IndexError("masked index is outside the gold embedding matrix")
    similarity = _normalise_rows(predicted) @ _normalise_rows(gold[masked]).T
    _, assignment = linear_sum_assignment(-similarity)
    return float(np.mean(assignment == np.arange(len(masked))))


def _validate_training_config(model: GraphMAEBaseline, target: str) -> None:
    actual = dataclasses.asdict(model.config)
    failures = [
        key for key, expected in EXPECTED_TRAINING_CONFIG.items()
        if actual.get(key) != expected
    ]
    if failures:
        raise RuntimeError(
            f"checkpoint training config mismatch for {target}: {', '.join(failures)}"
        )


def load_reusable_checkpoint(
    target: str,
    checkpoint_dir: Path,
    revision: str,
    device: str,
) -> tuple[GraphMAEBaseline, dict[str, Any]]:
    """Load and audit one target-excluded observed-reconstruction checkpoint."""

    checkpoint = checkpoint_dir / f"{target}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing target-specific checkpoint: {checkpoint}")
    model = GraphMAEBaseline.load_checkpoint(checkpoint, device=device)
    metadata = dict(model.metadata_)
    validate_checkpoint_metadata(metadata, target, revision)
    _validate_training_config(model, target)

    failures: list[str] = []
    if metadata.get("target_excluded_from_training") is not True:
        failures.append("target_excluded_from_training")
    if metadata.get("heldout_datasets_in_training") != []:
        failures.append("heldout_datasets_in_training")
    if metadata.get("latent_gold_used_for_training") is not False:
        failures.append("latent_gold_used_for_training")
    encoder = metadata.get("encoder")
    if not isinstance(encoder, dict):
        failures.append("encoder")
    elif (
        encoder.get("key") != ENCODER_KEY
        or encoder.get("prefix") != ENCODER_PREFIX
        or encoder.get("revision") != revision
    ):
        failures.append("encoder key/prefix/revision")
    sources = metadata.get("source_sha256")
    if not isinstance(sources, dict):
        failures.append("source_sha256")
    elif sources.get("graphmae_baseline.py") != TRAINING_BASELINE_SOURCE_SHA256:
        failures.append("training graphmae_baseline.py source hash")
    elif any(
        not isinstance(name, str)
        or re.fullmatch(r"[0-9a-f]{64}", str(digest)) is None
        for name, digest in sources.items()
    ):
        failures.append("source_sha256 format")
    if failures:
        raise RuntimeError(
            f"checkpoint is not reusable for formal Task 1 evaluation on {target}: "
            + ", ".join(failures)
        )

    audit = {
        "path": str(checkpoint.resolve()),
        "sha256": file_sha256(checkpoint),
        "checkpoint_method_version": CHECKPOINT_METHOD_VERSION,
        "checkpoint_protocol_version": CHECKPOINT_PROTOCOL_VERSION,
        "target_dataset": target,
        "train_datasets": list(training_datasets_for(target)),
        "target_excluded_from_training": True,
        "training_config": {
            key: value for key, value in dataclasses.asdict(model.config).items()
            if key != "device"
        },
        "training_git_commit": metadata.get("git_commit"),
        "training_source_sha256": dict(sources),
    }
    return model, audit


def _prediction_expectation(
    *,
    dataset: str,
    fold: int,
    masked_indices: Sequence[int],
    observed_nodes: Sequence[str],
    checkpoint_audit: Mapping[str, Any],
    config_hash: str,
    revision: str,
) -> dict[str, Any]:
    masked = [int(index) for index in masked_indices]
    masked_set = set(masked)
    return {
        "task": 1,
        "arm": ARM,
        "evaluation_method_version": EVALUATION_METHOD_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "dataset": dataset,
        "fold": int(fold),
        "masked_indices": masked,
        "masked_nodes": [observed_nodes[index] for index in masked],
        "visible_indices": [
            index for index in range(len(observed_nodes)) if index not in masked_set
        ],
        "checkpoint_sha256": checkpoint_audit["sha256"],
        "config_hash": config_hash,
        "encoder_revision": revision,
        "gold_text_available_to_prediction": False,
        "llm_judge_run": False,
    }


def _load_resumable_prediction(
    root: Path,
    expected: Mapping[str, Any],
) -> np.ndarray | None:
    dataset, fold = str(expected["dataset"]), int(expected["fold"])
    path = _prediction_path(root, dataset, fold)
    metadata_path = _prediction_metadata_path(root, dataset, fold)
    if not path.is_file() or not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(
                f"incompatible resumable prediction metadata for {dataset} fold {fold}: {key}"
            )
    digest = file_sha256(path)
    if metadata.get("prediction_sha256") != digest:
        raise RuntimeError(f"prediction hash mismatch for {dataset} fold {fold}")
    with np.load(path, allow_pickle=False) as artifact:
        predictions = np.asarray(artifact["predictions"], dtype=np.float64)
        stored_masked = np.asarray(artifact["masked_indices"], dtype=int)
    np.testing.assert_array_equal(stored_masked, expected["masked_indices"])
    if predictions.ndim != 2 or predictions.shape[0] != len(stored_masked):
        raise RuntimeError(f"invalid prediction shape for {dataset} fold {fold}")
    if not np.isfinite(predictions).all():
        raise RuntimeError(f"non-finite prediction for {dataset} fold {fold}")
    return predictions


def _write_prediction(
    root: Path,
    expected: Mapping[str, Any],
    predictions: np.ndarray,
) -> None:
    dataset, fold = str(expected["dataset"]), int(expected["fold"])
    path = _prediction_path(root, dataset, fold)
    values = np.asarray(predictions, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != len(expected["masked_indices"]):
        raise RuntimeError(f"invalid generated prediction shape for {dataset} fold {fold}")
    if not np.isfinite(values).all():
        raise RuntimeError(f"non-finite generated prediction for {dataset} fold {fold}")
    _atomic_npz(
        path,
        predictions=values,
        masked_indices=np.asarray(expected["masked_indices"], dtype=np.int64),
        masked_nodes=np.asarray(expected["masked_nodes"], dtype=np.str_),
        visible_indices=np.asarray(expected["visible_indices"], dtype=np.int64),
    )
    metadata = dict(expected)
    metadata["prediction_sha256"] = file_sha256(path)
    metadata["generated_at_utc"] = _utc_now()
    _atomic_json(_prediction_metadata_path(root, dataset, fold), metadata)


def _freeze_generation(
    root: Path,
    dataset: str,
    expectations: Sequence[Mapping[str, Any]],
    checkpoint_audit: Mapping[str, Any],
    config_hash: str,
) -> dict[str, Any]:
    predictions = []
    for expected in expectations:
        fold = int(expected["fold"])
        prediction_path = _prediction_path(root, dataset, fold)
        metadata_path = _prediction_metadata_path(root, dataset, fold)
        if not prediction_path.is_file() or not metadata_path.is_file():
            raise RuntimeError(f"cannot freeze incomplete predictions for {dataset}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        predictions.append({
            "fold": fold,
            "prediction_path": _relative(prediction_path, root),
            "prediction_sha256": file_sha256(prediction_path),
            "metadata_path": _relative(metadata_path, root),
            "metadata_sha256": file_sha256(metadata_path),
            "masked_indices": list(expected["masked_indices"]),
        })
        if metadata.get("prediction_sha256") != predictions[-1]["prediction_sha256"]:
            raise RuntimeError(f"prediction metadata changed before freeze: {dataset} fold {fold}")
    frozen = {
        "task": 1,
        "arm": ARM,
        "evaluation_method_version": EVALUATION_METHOD_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "dataset": dataset,
        "fold_count": len(expectations),
        "checkpoint_sha256": checkpoint_audit["sha256"],
        "config_hash": config_hash,
        "gold_embedded_before_freeze": False,
        "predictions": predictions,
    }
    path = _freeze_path(root, dataset)
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        for key, value in frozen.items():
            if existing.get(key) != value:
                raise RuntimeError(f"incompatible generation freeze marker for {dataset}: {key}")
        return existing
    frozen["frozen_at_utc"] = _utc_now()
    _atomic_json(path, frozen)
    return frozen


def run_dataset_two_pass(
    *,
    root: Path,
    dataset: Mapping[str, Any],
    model: GraphMAEBaseline,
    encoder: Any,
    checkpoint_audit: Mapping[str, Any],
    config_hash: str,
    revision: str,
    folds: int,
    fold_seed: int,
    status: dict[str, Any],
) -> dict[str, Any]:
    """Generate every fold first, freeze, then embed gold and score Match-ACC."""

    name = str(dataset["name"])
    graph = dataset["graph"]
    observed = list(graph.observed)
    labels = dataset["labels"]
    missing_labels = set(observed) - set(labels)
    if missing_labels:
        raise RuntimeError(f"{name} lacks observed labels: {sorted(missing_labels)}")
    fold_masks = outer_folds(len(observed), folds=folds, seed=fold_seed)
    expectations: list[dict[str, Any]] = []

    # Pass 1: masked text is neither embedded nor passed to the model.
    for fold, masked_array in enumerate(fold_masks):
        masked = sorted(int(index) for index in masked_array)
        expected = _prediction_expectation(
            dataset=name,
            fold=fold,
            masked_indices=masked,
            observed_nodes=observed,
            checkpoint_audit=checkpoint_audit,
            config_hash=config_hash,
            revision=revision,
        )
        expectations.append(expected)
        status["current"] = {"dataset": name, "fold": fold, "phase": "prediction"}
        _atomic_json(root / "status.json", status)
        prediction = _load_resumable_prediction(root, expected)
        if prediction is not None:
            status["resumed_prediction_folds"] += 1
            continue
        visible = list(expected["visible_indices"])
        visible_matrix = encoder.embed([str(labels[observed[index]]) for index in visible])
        visible_embeddings = {
            observed[index]: visible_matrix[position]
            for position, index in enumerate(visible)
        }
        targets = list(expected["masked_nodes"])
        generated = model.infer_observed_targets(
            graph, visible_embeddings, targets
        )
        prediction = np.stack([generated[node] for node in targets])
        _write_prediction(root, expected, prediction)
        status["completed_prediction_folds"] += 1

    _freeze_generation(
        root, name, expectations, checkpoint_audit, config_hash
    )

    # Pass 2 starts only after all selected folds for this dataset are immutable.
    status["current"] = {"dataset": name, "fold": None, "phase": "gold_embedding"}
    _atomic_json(root / "status.json", status)
    gold = encoder.embed([str(labels[node]) for node in observed])
    fold_scores: list[float] = []
    fold_results: list[dict[str, Any]] = []
    for expected in expectations:
        fold = int(expected["fold"])
        prediction = _load_resumable_prediction(root, expected)
        if prediction is None:
            raise RuntimeError(f"frozen prediction disappeared for {name} fold {fold}")
        score = task1_match_acc(prediction, expected["masked_indices"], gold)
        prediction_path = _prediction_path(root, name, fold)
        metric = {
            "task": 1,
            "arm": ARM,
            "evaluation_method_version": EVALUATION_METHOD_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "dataset": name,
            "fold": fold,
            "masked_count": len(expected["masked_indices"]),
            "masked_indices": list(expected["masked_indices"]),
            "match_acc": score,
            "checkpoint_sha256": checkpoint_audit["sha256"],
            "prediction_sha256": file_sha256(prediction_path),
            "config_hash": config_hash,
            "gold_embedding_phase": "after_dataset_generation_freeze",
            "llm_judge": None,
        }
        metric_path = _fold_metric_path(root, name, fold)
        if metric_path.is_file():
            existing = json.loads(metric_path.read_text(encoding="utf-8"))
            for key, value in metric.items():
                if existing.get(key) != value:
                    raise RuntimeError(
                        f"incompatible resumable fold metric for {name} fold {fold}: {key}"
                    )
            status["resumed_metric_folds"] += 1
            metric = existing
        else:
            metric["evaluated_at_utc"] = _utc_now()
            _atomic_json(metric_path, metric)
            status["completed_metric_folds"] += 1
        fold_scores.append(float(metric["match_acc"]))
        fold_results.append(metric)

    status["completed_datasets"].append(name)
    status["current"] = None
    _atomic_json(root / "status.json", status)
    return {
        "dataset": name,
        "item_count": len(observed),
        "fold_count": len(fold_scores),
        "match_acc": float(np.mean(fold_scores)),
        "fold_match_acc": fold_scores,
        "checkpoint_sha256": checkpoint_audit["sha256"],
        "train_datasets": list(checkpoint_audit["train_datasets"]),
        "target_excluded_from_training": True,
        "generation_freeze": _relative(_freeze_path(root, name), root),
        "fold_metrics": [
            _relative(_fold_metric_path(root, name, int(item["fold"])), root)
            for item in fold_results
        ],
    }


def _config_payload(revision: str, folds: int, fold_seed: int) -> dict[str, Any]:
    return {
        "task": 1,
        "arm": ARM,
        "evaluation_method_version": EVALUATION_METHOD_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "checkpoint_method_version": CHECKPOINT_METHOD_VERSION,
        "checkpoint_protocol_version": CHECKPOINT_PROTOCOL_VERSION,
        "checkpoint_training_config": dict(EXPECTED_TRAINING_CONFIG),
        "folds": folds,
        "fold_seed": fold_seed,
        "encoder": {
            "key": ENCODER_KEY,
            "model": ENCODER_MODEL,
            "revision": revision,
            "prefix": ENCODER_PREFIX,
            "mode": "frozen-base-no-project-lora",
        },
        "metric": "fold-local Hungarian Match-ACC only",
        "llm_judge_run": False,
    }


def build_manifest(
    *,
    root: Path,
    arguments: argparse.Namespace,
    targets: Sequence[str],
    revision: str,
    config_hash: str,
    checkpoints: Mapping[str, Mapping[str, Any]],
    results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    summary_path = root / "summary.json"
    if not summary_path.is_file():
        raise RuntimeError("summary must be written before manifest")
    artifacts: dict[str, Any] = {}
    for target in targets:
        checkpoint = checkpoints[target]
        checkpoint_path = Path(str(checkpoint["path"]))
        if file_sha256(checkpoint_path) != checkpoint["sha256"]:
            raise RuntimeError(f"checkpoint changed during evaluation: {target}")
        fold_artifacts = []
        for fold in range(int(results[target]["fold_count"])):
            prediction = _prediction_path(root, target, fold)
            prediction_metadata = _prediction_metadata_path(root, target, fold)
            metric = _fold_metric_path(root, target, fold)
            fold_artifacts.append({
                "fold": fold,
                "prediction": {
                    "path": _relative(prediction, root),
                    "sha256": file_sha256(prediction),
                },
                "prediction_metadata": {
                    "path": _relative(prediction_metadata, root),
                    "sha256": file_sha256(prediction_metadata),
                },
                "metric": {
                    "path": _relative(metric, root),
                    "sha256": file_sha256(metric),
                },
            })
        freeze = _freeze_path(root, target)
        artifacts[target] = {
            "checkpoint": {
                "path": str(checkpoint_path.resolve()),
                "sha256": checkpoint["sha256"],
                "training_git_commit": checkpoint["training_git_commit"],
                "training_source_sha256": checkpoint["training_source_sha256"],
            },
            "train_datasets": list(checkpoint["train_datasets"]),
            "target_excluded_from_training": True,
            "generation_freeze": {
                "path": _relative(freeze, root),
                "sha256": file_sha256(freeze),
            },
            "folds": fold_artifacts,
        }
    return {
        "task": 1,
        "arm": ARM,
        "evaluation_method_version": EVALUATION_METHOD_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "generated_at_utc": _utc_now(),
        "config_hash": config_hash,
        "targets": list(targets),
        "report_dataset_order": list(REPORT_DATASETS),
        "git": _git_state(),
        "cli": _jsonable(vars(arguments)),
        "encoder": {
            "key": ENCODER_KEY,
            "model": ENCODER_MODEL,
            "revision": revision,
            "prefix": ENCODER_PREFIX,
            "mode": "frozen-base-no-project-lora",
        },
        "checkpoint_reuse": {
            "checkpoint_method_version": CHECKPOINT_METHOD_VERSION,
            "checkpoint_protocol_version": CHECKPOINT_PROTOCOL_VERSION,
            "training_objective": "masked observed-label reconstruction",
            "new_readout": "decoder-remasked hidden observed targets",
            "weights_modified_or_retrained": False,
            "target_specific_lodo_required": True,
        },
        "evaluation": {
            "folds": int(arguments.folds),
            "fold_seed": int(arguments.fold_seed),
            "metric": "Match-ACC",
            "exact_run": False,
            "llm_judge_run": False,
            "gold_isolation": (
                "all fold predictions for a dataset are frozen before its complete "
                "gold-label embedding matrix is created"
            ),
        },
        "evaluator_source_sha256": {
            "graphmae_baseline.py": file_sha256(GRAPHMAE_SOURCE),
            "run_graphmae_task1.py": file_sha256(__file__),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "artifacts": artifacts,
        "summary": {
            "path": _relative(summary_path, root),
            "sha256": file_sha256(summary_path),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="report19")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument(
        "--hf-cache",
        type=Path,
        default=Path(os.environ.get("HF_CACHE", WORKSPACE_ROOT / ".hf_cache")),
    )
    parser.add_argument("--encoder-revision")
    parser.add_argument("--encoder-device")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.folds != 5 or arguments.fold_seed != 0:
        raise SystemExit("the frozen Task 1 protocol requires --folds 5 --fold-seed 0")
    targets = select_datasets(arguments.datasets)
    if len(targets) != len(set(targets)):
        raise SystemExit("dataset selection contains duplicates")
    revision = resolve_encoder_revision(arguments.hf_cache, arguments.encoder_revision)
    if revision != PINNED_ENCODER_REVISION:
        raise SystemExit(
            "formal GraphMAE Task 1 evaluation requires pinned E5 revision "
            + PINNED_ENCODER_REVISION
        )
    root = arguments.output_dir or (
        V6_DIR / "outputs" / "graphmae_task1_lodo" / "report19_seed0_v1"
    )
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    config = _config_payload(revision, arguments.folds, arguments.fold_seed)
    config_hash = _canonical_hash(config)
    status: dict[str, Any] = {
        "task": 1,
        "arm": ARM,
        "protocol_version": PROTOCOL_VERSION,
        "config_hash": config_hash,
        "state": "running",
        "targets": list(targets),
        "completed_datasets": [],
        "completed_prediction_folds": 0,
        "resumed_prediction_folds": 0,
        "completed_metric_folds": 0,
        "resumed_metric_folds": 0,
        "current": None,
        "started_at_utc": _utc_now(),
    }
    _atomic_json(root / "status.json", status)

    encoder = FrozenE5Encoder(
        cache_folder=arguments.hf_cache,
        revision=revision,
        device=arguments.encoder_device,
        local_files_only=not arguments.allow_download,
        batch_size=arguments.batch_size,
    )
    loaders = report_loaders()
    checkpoints: dict[str, dict[str, Any]] = {}
    results: dict[str, dict[str, Any]] = {}
    try:
        for target in targets:
            model, checkpoint_audit = load_reusable_checkpoint(
                target,
                arguments.checkpoint_dir,
                revision,
                arguments.device,
            )
            checkpoints[target] = checkpoint_audit
            result = run_dataset_two_pass(
                root=root,
                dataset=loaders[target](),
                model=model,
                encoder=encoder,
                checkpoint_audit=checkpoint_audit,
                config_hash=config_hash,
                revision=revision,
                folds=arguments.folds,
                fold_seed=arguments.fold_seed,
                status=status,
            )
            results[target] = result
            print(f"[graphmae-task1] {target}: match={result['match_acc']:.6f}", flush=True)

        summary = {
            "task": 1,
            "arm": ARM,
            "evaluation_method_version": EVALUATION_METHOD_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "config_hash": config_hash,
            "encoder": {
                "model": ENCODER_MODEL,
                "revision": revision,
                "mode": "frozen-base-no-project-lora",
            },
            "targets": list(targets),
            "completed": list(results),
            "coverage": {
                "datasets": len(results),
                "folds": sum(item["fold_count"] for item in results.values()),
                "masked_items": sum(item["item_count"] for item in results.values()),
            },
            "dataset_macro_match_acc": float(np.mean([
                item["match_acc"] for item in results.values()
            ])),
            "results": results,
            "llm_judge_run": False,
            "generated_at_utc": _utc_now(),
        }
        _atomic_json(root / "summary.json", summary)
        manifest = build_manifest(
            root=root,
            arguments=arguments,
            targets=targets,
            revision=revision,
            config_hash=config_hash,
            checkpoints=checkpoints,
            results=results,
        )
        _atomic_json(root / "manifest.json", manifest)
        status["state"] = "complete"
        status["current"] = None
        status["completed_at_utc"] = _utc_now()
        status["summary_sha256"] = file_sha256(root / "summary.json")
        status["manifest_sha256"] = file_sha256(root / "manifest.json")
        _atomic_json(root / "status.json", status)
    except Exception as error:
        status["state"] = "failed"
        status["error"] = f"{type(error).__name__}: {error}"
        status["failed_at_utc"] = _utc_now()
        _atomic_json(root / "status.json", status)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
