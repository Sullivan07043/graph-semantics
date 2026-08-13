"""Run the five external baselines on the redesigned robot Task 1.

Protocol: BOSS channel graph, seed-0 five-fold (20%) label masking, frozen base
E5 Match-ACC, no exact match, and no LLM judge.  Development robots use
Robot-LODO for GraphMAE-GCN; the held-out UR5e checkpoint is trained only on
Panda, Sawyer, and IIWA.  Every generation artifact is written before gold
labels are embedded, and interrupted LLM cases resume from an API cache.

From the repository root::

    python -m task3_robotics.baselines.run_task1 --baselines all \
      --datasets all --budget-usd none --case-workers 8
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import dataclasses
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from v6.baselines.api import BaselineAPIClient
from v6.baselines.automated_interpretability import (
    ROBOT_AUTOINTERP_EXPLAIN_PROMPT_VERSION,
    ROBOT_AUTOINTERP_SIMULATE_PROMPT_VERSION,
    run_autointerp,
)
from v6.baselines.clip_dissect_e5 import (
    ClipDissectE5Config,
    load_concept_bank,
    run_clip_dissect_e5,
)
from v6.baselines.delphi import (
    ROBOT_DELPHI_DETECT_PROMPT_VERSION,
    ROBOT_DELPHI_EXPLAIN_PROMPT_VERSION,
    run_delphi,
)
from v6.baselines.feature_propagation import FEATURE_PROP_VERSION, feature_propagation
from v6.baselines.graphmae_gcn import (
    METHOD_VERSION as GRAPHMAE_VERSION,
    GraphExample,
    GraphMAEBaseline,
    GraphMAEConfig,
)
from v6.baselines.protocol import (
    atomic_write_json,
    config_hash,
    file_sha256,
    observed_activation_matrix,
    outer_folds,
    stable_seed,
)

from .clip_dissect_bank import BANK_VERSION as ROBOT_BANK_VERSION
from .clip_dissect_bank import build_robot_wordnet_bank
from .common import (
    DEV_ROBOTS,
    HELDOUT_ROBOTS,
    ROBOT_DATASETS,
    load_robot_dataset,
    render_robot_snapshots,
    required_artifacts,
    select_robot_datasets,
)


HERE = Path(__file__).resolve().parent
TASK3_DIR = HERE.parent
REPO_ROOT = TASK3_DIR.parent
WORKSPACE_ROOT = REPO_ROOT.parent
PROTOCOL_VERSION = "robot-task1-external-baselines-boss-v1"
ENCODER_MODEL = "intfloat/e5-large-v2"
ENCODER_REVISION = "f169b11e22de13617baa190a028a32f3493550b6"
ENCODER_PREFIX = "query: "
METHODS = (
    "feature-propagation",
    "graphmae-gcn",
    "clip-dissect-e5",
    "autointerp",
    "delphi",
)
METHOD_LABELS = {
    "feature-propagation": "Feature Propagation",
    "graphmae-gcn": "GraphMAE-GCN",
    "clip-dissect-e5": "CLIP-Dissect (E5 robot adaptation)",
    "autointerp": "Automated Interpretability",
    "delphi": "Delphi",
}
SOURCE_FILES = (
    "task3_robotics/baselines/common.py",
    "task3_robotics/baselines/clip_dissect_bank.py",
    "task3_robotics/baselines/run_task1.py",
    "v6/baselines/feature_propagation.py",
    "v6/baselines/graphmae_gcn.py",
    "v6/baselines/clip_dissect_e5.py",
    "v6/baselines/automated_interpretability.py",
    "v6/baselines/delphi.py",
    "v6/baselines/_llm_interpretability.py",
    "v6/baselines/api.py",
    "v6/baselines/protocol.py",
)


class FrozenE5Encoder:
    def __init__(
        self,
        cache_folder: Path,
        *,
        device: str | None,
        batch_size: int,
        allow_download: bool,
    ) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = int(batch_size)
        self.model = SentenceTransformer(
            ENCODER_MODEL,
            revision=ENCODER_REVISION,
            device=selected_device,
            cache_folder=str(cache_folder),
            local_files_only=not allow_download,
        )

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            raise ValueError("cannot embed an empty text list")
        values = self.model.encode(
            [ENCODER_PREFIX + str(text) for text in texts],
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        matrix = np.asarray(values, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(texts):
            raise RuntimeError("E5 returned an invalid embedding matrix")
        if not np.isfinite(matrix).all():
            raise RuntimeError("E5 returned a non-finite embedding")
        return matrix


def _parse_methods(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return list(METHODS)
    aliases = {
        "feature": "feature-propagation",
        "feature-prop": "feature-propagation",
        "graphmae": "graphmae-gcn",
        "clip-dissect": "clip-dissect-e5",
        "automated-interpretability": "autointerp",
        "auto": "autointerp",
    }
    selected = [
        aliases.get(item.strip().lower(), item.strip().lower())
        for item in value.split(",")
        if item.strip()
    ]
    unknown = [method for method in selected if method not in METHODS]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown robot baselines: {unknown}")
    return list(dict.fromkeys(selected))


def _parse_budget(value: str | float) -> float | None:
    if isinstance(value, str) and value.strip().lower() in {
        "none",
        "unlimited",
        "no-limit",
        "off",
    }:
        return None
    try:
        budget = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("budget must be non-negative or 'none'") from exc
    if budget < 0:
        raise argparse.ArgumentTypeError("budget cannot be negative")
    return budget


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--baselines", default="all", type=_parse_methods)
    parser.add_argument("--data-dir", type=Path, default=TASK3_DIR / "outputs")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--hf-cache",
        type=Path,
        default=Path(os.environ.get("HF_CACHE", WORKSPACE_ROOT / ".hf_cache")),
    )
    parser.add_argument(
        "--wordnet-zip",
        type=Path,
        default=WORKSPACE_ROOT / ".nltk_data" / "corpora" / "wordnet.zip",
    )
    parser.add_argument("--concept-bank", type=Path)
    parser.add_argument("--text-bank-size", type=int, default=4096)
    parser.add_argument("--embed-batch-size", type=int, default=64)
    parser.add_argument("--encoder-device")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--graphmae-device", default="auto")
    parser.add_argument("--graphmae-epochs", type=int, default=200)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--budget-usd", type=_parse_budget, default=5.0)
    parser.add_argument("--api-cache-dir", type=Path)
    parser.add_argument("--case-workers", type=int, default=8)
    parser.add_argument("--profile-top-k", type=int, default=6)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    return parser


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


def _source_hashes() -> dict[str, str]:
    return {name: file_sha256(REPO_ROOT / name) for name in SOURCE_FILES}


def _normalise_rows(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    return values / (np.linalg.norm(values, axis=1, keepdims=True) + 1e-9)


def task1_match_acc(predicted: np.ndarray, gold: np.ndarray) -> float:
    from scipy.optimize import linear_sum_assignment

    predictions = np.asarray(predicted, dtype=float)
    targets = np.asarray(gold, dtype=float)
    if predictions.ndim != 2 or predictions.shape != targets.shape or not len(predictions):
        raise ValueError("predicted and gold embeddings must share non-empty [item, dim] shape")
    similarities = _normalise_rows(predictions) @ _normalise_rows(targets).T
    rows, columns = linear_sum_assignment(-similarities)
    if not np.array_equal(rows, np.arange(len(predictions))):
        raise RuntimeError("unexpected non-square Hungarian assignment")
    return float(np.mean(columns == np.arange(len(predictions))))


def _fold_specs(dataset: Mapping[str, Any]) -> list[tuple[int, list[int]]]:
    count = len(dataset["graph"].observed)
    return [
        (fold, sorted(int(index) for index in mask))
        for fold, mask in enumerate(outer_folds(count, folds=5, seed=0))
    ]


def _generation_path(root: Path, method: str, dataset: str, fold: int) -> Path:
    return root / "generation" / method / dataset / f"fold_{fold:02d}.json"


def _case_path(
    root: Path, method: str, dataset: str, fold: int, observed_index: int
) -> Path:
    return (
        root
        / "generation_cases"
        / method
        / dataset
        / f"fold_{fold:02d}"
        / f"observed_{observed_index:04d}.json"
    )


def _metric_path(root: Path, method: str, dataset: str, fold: int) -> Path:
    return root / "metrics" / method / dataset / f"fold_{fold:02d}.json"


def _load_record(path: Path, cfg_hash: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("config_hash") != cfg_hash:
        raise RuntimeError(f"artifact belongs to another configuration: {path}")
    return record


def _write_generation(
    path: Path,
    *,
    method: str,
    dataset: str,
    fold: int,
    masked: Sequence[int],
    payload: Mapping[str, Any],
    cfg_hash: str,
) -> None:
    atomic_write_json(
        path,
        {
            "task": 1,
            "method": method,
            "dataset": dataset,
            "fold": int(fold),
            "masked_indices": [int(index) for index in masked],
            **dict(payload),
            "config_hash": cfg_hash,
            "gold_label_text_available": False,
            "generated_at": _timestamp(),
        },
    )


def _write_metric(
    root: Path,
    method: str,
    dataset: Mapping[str, Any],
    fold: int,
    masked: Sequence[int],
    score: float,
    cfg_hash: str,
    *,
    native: Mapping[str, Any] | None = None,
) -> None:
    observed = list(dataset["graph"].observed)
    labels = dataset["labels"]
    atomic_write_json(
        _metric_path(root, method, str(dataset["name"]), fold),
        {
            "task": 1,
            "method": method,
            "method_label": METHOD_LABELS[method],
            "dataset": dataset["name"],
            "split": (
                "heldout"
                if dataset["name"] in HELDOUT_ROBOTS
                else ("dev-lodo" if method == "graphmae-gcn" else "dev")
            ),
            "fold": int(fold),
            "masked_indices": [int(index) for index in masked],
            "gold_labels": [str(labels[observed[index]]) for index in masked],
            "match_acc": float(score),
            "native": dict(native or {}),
            "exact": None,
            "llm_judge": None,
            "config_hash": cfg_hash,
            "evaluated_at": _timestamp(),
        },
    )


def _run_feature_propagation(
    root: Path,
    dataset: Mapping[str, Any],
    encoder: FrozenE5Encoder,
    cfg_hash: str,
) -> None:
    method = "feature-propagation"
    name = str(dataset["name"])
    graph = dataset["graph"]
    observed = list(graph.observed)
    labels = dataset["labels"]
    folds = _fold_specs(dataset)
    for fold, masked in folds:
        path = _generation_path(root, method, name, fold)
        if _load_record(path, cfg_hash) is not None:
            continue
        hidden = set(masked)
        visible = [index for index in range(len(observed)) if index not in hidden]
        visible_matrix = encoder.embed([str(labels[observed[index]]) for index in visible])
        anchors = {
            observed[index]: visible_matrix[position]
            for position, index in enumerate(visible)
        }
        propagated = feature_propagation(
            graph, anchors, max_iter=40, tol=1e-8, fallback="zeros"
        )
        predictions = np.stack([propagated[observed[index]] for index in masked])
        _write_generation(
            path,
            method=method,
            dataset=name,
            fold=fold,
            masked=masked,
            payload={"predicted_embeddings": predictions.tolist()},
            cfg_hash=cfg_hash,
        )
    gold = encoder.embed([str(labels[node]) for node in observed])
    for fold, masked in folds:
        record = _load_record(_generation_path(root, method, name, fold), cfg_hash)
        assert record is not None
        predictions = np.asarray(record["predicted_embeddings"], dtype=float)
        _write_metric(
            root,
            method,
            dataset,
            fold,
            masked,
            task1_match_acc(predictions, gold[masked]),
            cfg_hash,
        )


def _graphmae_train_names(target: str) -> list[str]:
    if target in DEV_ROBOTS:
        return [name for name in DEV_ROBOTS if name != target]
    if target in HELDOUT_ROBOTS:
        return list(DEV_ROBOTS)
    raise ValueError(f"unknown GraphMAE robot target: {target}")


def _graphmae_model(
    root: Path,
    target: str,
    datasets: Mapping[str, Mapping[str, Any]],
    encoder: FrozenE5Encoder,
    cfg_hash: str,
    *,
    epochs: int,
    device: str,
) -> tuple[GraphMAEBaseline, Path]:
    checkpoint = root / "checkpoints" / "graphmae-gcn" / f"{target}.pt"
    train_names = _graphmae_train_names(target)
    if checkpoint.is_file():
        model = GraphMAEBaseline.load_checkpoint(checkpoint, device=device)
        metadata = model.metadata_
        failures = []
        if metadata.get("robot_protocol") != PROTOCOL_VERSION:
            failures.append("robot_protocol")
        if metadata.get("target") != target:
            failures.append("target")
        if metadata.get("train_datasets") != train_names:
            failures.append("train_datasets")
        if metadata.get("config_hash") != cfg_hash:
            failures.append("config_hash")
        if failures:
            raise RuntimeError(
                f"GraphMAE checkpoint provenance mismatch for {target}: {failures}"
            )
        return model, checkpoint

    examples: list[GraphExample] = []
    for name in train_names:
        dataset = datasets[name]
        observed = list(dataset["graph"].observed)
        matrix = encoder.embed([str(dataset["labels"][node]) for node in observed])
        examples.append(
            GraphExample(
                name,
                dataset["graph"],
                {node: matrix[index] for index, node in enumerate(observed)},
            )
        )
    config = GraphMAEConfig(epochs=epochs, device=device)
    model = GraphMAEBaseline(config).fit(examples)
    model.metadata_ = {
        "robot_protocol": PROTOCOL_VERSION,
        "target": target,
        "train_datasets": train_names,
        "target_excluded_from_training": target not in train_names,
        "heldout_datasets_in_training": sorted(set(train_names) & set(HELDOUT_ROBOTS)),
        "latent_gold_used_for_training": False,
        "config_hash": cfg_hash,
        "encoder_model": ENCODER_MODEL,
        "encoder_revision": ENCODER_REVISION,
        "method_version": GRAPHMAE_VERSION,
        "final_training_loss": model.history_[-1],
    }
    model.save_checkpoint(checkpoint)
    return model, checkpoint


def _run_graphmae(
    root: Path,
    dataset: Mapping[str, Any],
    datasets: Mapping[str, Mapping[str, Any]],
    encoder: FrozenE5Encoder,
    cfg_hash: str,
    *,
    epochs: int,
    device: str,
) -> None:
    method = "graphmae-gcn"
    name = str(dataset["name"])
    graph = dataset["graph"]
    observed = list(graph.observed)
    labels = dataset["labels"]
    model, checkpoint = _graphmae_model(
        root, name, datasets, encoder, cfg_hash, epochs=epochs, device=device
    )
    checkpoint_sha256 = file_sha256(checkpoint)
    folds = _fold_specs(dataset)
    for fold, masked in folds:
        path = _generation_path(root, method, name, fold)
        existing = _load_record(path, cfg_hash)
        if existing is not None:
            if existing.get("checkpoint_sha256") != checkpoint_sha256:
                raise RuntimeError(
                    f"GraphMAE prediction checkpoint changed for {name} fold {fold}"
                )
            continue
        hidden = set(masked)
        visible = [index for index in range(len(observed)) if index not in hidden]
        visible_matrix = encoder.embed([str(labels[observed[index]]) for index in visible])
        visible_embeddings = {
            observed[index]: visible_matrix[position]
            for position, index in enumerate(visible)
        }
        targets = [observed[index] for index in masked]
        output = model.infer_observed_targets(graph, visible_embeddings, targets)
        predictions = np.stack([output[node] for node in targets])
        _write_generation(
            path,
            method=method,
            dataset=name,
            fold=fold,
            masked=masked,
            payload={
                "predicted_embeddings": predictions.tolist(),
                "checkpoint_train_datasets": _graphmae_train_names(name),
                "checkpoint_sha256": checkpoint_sha256,
            },
            cfg_hash=cfg_hash,
        )
    gold = encoder.embed([str(labels[node]) for node in observed])
    for fold, masked in folds:
        record = _load_record(_generation_path(root, method, name, fold), cfg_hash)
        assert record is not None
        predictions = np.asarray(record["predicted_embeddings"], dtype=float)
        _write_metric(
            root,
            method,
            dataset,
            fold,
            masked,
            task1_match_acc(predictions, gold[masked]),
            cfg_hash,
            native={
                "checkpoint_train_datasets": _graphmae_train_names(name),
                "checkpoint_sha256": checkpoint_sha256,
            },
        )


def _run_clip_dissect(
    root: Path,
    dataset: Mapping[str, Any],
    encoder: FrozenE5Encoder,
    concept_bank: Any,
    cfg_hash: str,
    *,
    profile_top_k: int,
) -> None:
    method = "clip-dissect-e5"
    name = str(dataset["name"])
    observed = list(dataset["graph"].observed)
    labels = dataset["labels"]
    values = np.asarray(dataset["X"], dtype=float)
    folds = _fold_specs(dataset)
    text_config = ClipDissectE5Config()
    for fold, masked in folds:
        path = _generation_path(root, method, name, fold)
        if _load_record(path, cfg_hash) is not None:
            continue
        hidden = set(masked)
        visible = [index for index in range(len(observed)) if index not in hidden]
        visible_labels = {observed[index]: labels[observed[index]] for index in visible}
        snapshots, _ = render_robot_snapshots(
            values, observed, visible_labels, visible, top_k=profile_top_k
        )
        profile_embeddings = encoder.embed(snapshots)
        activations, activation_metadata = observed_activation_matrix(
            values, len(observed), masked
        )
        results = run_clip_dissect_e5(
            None,
            activations,
            concept_bank,
            profile_embeddings=profile_embeddings,
            latent_ids=[f"masked_channel_{position:03d}" for position in range(len(masked))],
            config=text_config,
        )
        _write_generation(
            path,
            method=method,
            dataset=name,
            fold=fold,
            masked=masked,
            payload={
                "predictions": [result.to_dict() for result in results],
                "activation_metadata": activation_metadata,
            },
            cfg_hash=cfg_hash,
        )
    gold = encoder.embed([str(labels[node]) for node in observed])
    for fold, masked in folds:
        record = _load_record(_generation_path(root, method, name, fold), cfg_hash)
        assert record is not None
        names = [str(item["construct_name"]) for item in record["predictions"]]
        predictions = encoder.embed(names)
        _write_metric(
            root,
            method,
            dataset,
            fold,
            masked,
            task1_match_acc(predictions, gold[masked]),
            cfg_hash,
        )


def _run_llm_method(
    root: Path,
    method: str,
    dataset: Mapping[str, Any],
    encoder: FrozenE5Encoder,
    client: BaselineAPIClient,
    cfg_hash: str,
    *,
    profile_top_k: int,
    case_workers: int,
    status_callback: Callable[[Mapping[str, Any]], None],
) -> None:
    name = str(dataset["name"])
    observed = list(dataset["graph"].observed)
    labels = dataset["labels"]
    values = np.asarray(dataset["X"], dtype=float)
    folds = _fold_specs(dataset)
    for fold, masked in folds:
        hidden = set(masked)
        visible = [index for index in range(len(observed)) if index not in hidden]
        visible_labels = {observed[index]: labels[observed[index]] for index in visible}
        snapshots, response_vectors = render_robot_snapshots(
            values, observed, visible_labels, visible, top_k=profile_top_k
        )
        activations, activation_metadata = observed_activation_matrix(
            values, len(observed), masked
        )
        missing = [
            (position, observed_index)
            for position, observed_index in enumerate(masked)
            if _load_record(
                _case_path(root, method, name, fold, observed_index), cfg_hash
            )
            is None
        ]

        def run_case(case: tuple[int, int]) -> tuple[int, dict[str, Any]]:
            position, observed_index = case
            seed = stable_seed(
                "robot-task1", method, name, fold, observed_index, base_seed=0
            )
            if method == "autointerp":
                prediction = run_autointerp(
                    client,
                    snapshots,
                    activations[:, position],
                    seed=seed,
                    domain="robot",
                    repair_semantic_errors=True,
                )
            else:
                prediction = run_delphi(
                    client,
                    snapshots,
                    activations[:, position],
                    response_vectors,
                    seed=seed,
                    domain="robot",
                    repair_semantic_errors=True,
                )
            record = {
                "task": 1,
                "method": method,
                "dataset": name,
                "fold": fold,
                "observed_index": observed_index,
                "observed_node": observed[observed_index],
                "construct_name": str(prediction["construct_name"]),
                "prediction": prediction,
                "activation_metadata": activation_metadata[position],
                "config_hash": cfg_hash,
                "gold_label_text_available": False,
                "generated_at": _timestamp(),
            }
            atomic_write_json(
                _case_path(root, method, name, fold, observed_index), record
            )
            return observed_index, record

        if missing:
            with ThreadPoolExecutor(
                max_workers=case_workers, thread_name_prefix=f"robot-{method}"
            ) as pool:
                futures = [pool.submit(run_case, case) for case in missing]
                for future in as_completed(futures):
                    observed_index, _ = future.result()
                    status_callback(
                        {
                            "stage": "generation",
                            "method": method,
                            "dataset": name,
                            "fold": fold,
                            "last_observed_index": observed_index,
                            "api_spend_usd": client.spent_usd,
                        }
                    )

    # All gold-free cases for this dataset/method are frozen before gold encoding.
    for fold, masked in folds:
        for observed_index in masked:
            if _load_record(
                _case_path(root, method, name, fold, observed_index), cfg_hash
            ) is None:
                raise RuntimeError(
                    f"incomplete {method} generation for {name} fold {fold}"
                )
    gold = encoder.embed([str(labels[node]) for node in observed])
    for fold, masked in folds:
        records = [
            _load_record(_case_path(root, method, name, fold, index), cfg_hash)
            for index in masked
        ]
        if any(record is None for record in records):
            raise RuntimeError(f"cannot evaluate incomplete {method} fold")
        selected = [dict(record) for record in records if record is not None]
        predicted = encoder.embed([str(record["construct_name"]) for record in selected])
        if method == "autointerp":
            native = {
                "spearman": _mean(
                    record["prediction"].get("spearman") for record in selected
                ),
                "pearson": _mean(
                    record["prediction"].get("pearson") for record in selected
                ),
            }
        else:
            native = {
                "test_auroc": _mean(
                    record["prediction"].get("test_auroc") for record in selected
                ),
                "test_f1": _mean(
                    record["prediction"].get("test_f1") for record in selected
                ),
            }
        _write_metric(
            root,
            method,
            dataset,
            fold,
            masked,
            task1_match_acc(predicted, gold[masked]),
            cfg_hash,
            native=native,
        )


def _mean(values: Sequence[Any] | Any) -> float | None:
    selected = [
        float(value)
        for value in values
        if value is not None and np.isfinite(float(value))
    ]
    return float(np.mean(selected)) if selected else None


def _build_summary(
    root: Path, methods: Sequence[str], datasets: Sequence[str], cfg_hash: str
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "protocol": PROTOCOL_VERSION,
        "config_hash": cfg_hash,
        "metric": "five-fold mean Match-ACC",
        "exact_run": False,
        "llm_judge_run": False,
        "methods": {},
        "generated_at": _timestamp(),
    }
    for method in methods:
        per_dataset: dict[str, Any] = {}
        for dataset in datasets:
            records = [
                _load_record(_metric_path(root, method, dataset, fold), cfg_hash)
                for fold in range(5)
            ]
            completed = [record for record in records if record is not None]
            per_dataset[dataset] = {
                "split": (
                    "heldout"
                    if dataset in HELDOUT_ROBOTS
                    else ("dev-lodo" if method == "graphmae-gcn" else "dev")
                ),
                "completed_folds": len(completed),
                "fold_match_acc": [record["match_acc"] for record in completed],
                "match_acc": _mean(record["match_acc"] for record in completed),
            }
        output["methods"][method] = {
            "label": METHOD_LABELS[method],
            "datasets": per_dataset,
            "dev_macro_match_acc": _mean(
                per_dataset[name]["match_acc"] for name in DEV_ROBOTS
                if name in per_dataset
            ),
            "heldout_match_acc": (
                per_dataset["bodyur5e"]["match_acc"]
                if "bodyur5e" in per_dataset
                else None
            ),
        }
    atomic_write_json(root / "summary.json", output)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.folds != 5 or args.seed != 0:
        raise SystemExit("the frozen robot Task 1 protocol requires --folds 5 --seed 0")
    if args.case_workers < 1 or args.embed_batch_size < 1 or args.profile_top_k < 1:
        raise SystemExit("worker, batch, and profile sizes must be positive")
    if args.graphmae_epochs < 1 or args.text_bank_size < 1:
        raise SystemExit("GraphMAE epochs and concept-bank size must be positive")

    selected_datasets = select_robot_datasets(args.datasets)
    selected_methods = list(args.baselines)
    data_dir = args.data_dir.resolve()
    # GraphMAE always needs all three dev robots, even for a single target run.
    load_names = set(selected_datasets)
    if "graphmae-gcn" in selected_methods:
        load_names.update(DEV_ROBOTS)
    missing = [path for path in required_artifacts(data_dir, sorted(load_names)) if not path.is_file()]
    if missing:
        joined = "\n  - ".join(str(path) for path in missing)
        raise FileNotFoundError(
            "robot Task 1 artifacts are unavailable; missing:\n  - " + joined
        )
    datasets = {
        name: load_robot_dataset(name, data_dir) for name in sorted(load_names)
    }

    root = (
        args.output_dir
        or data_dir / "baselines" / "task1_external_robot4_boss_seed0_v1"
    ).resolve()
    root.mkdir(parents=True, exist_ok=True)
    provenance = {
        str(path.relative_to(data_dir)): file_sha256(path)
        for path in required_artifacts(data_dir, ROBOT_DATASETS)
        if path.is_file()
    }
    config = {
        "protocol": PROTOCOL_VERSION,
        "dataset_suite": list(ROBOT_DATASETS),
        "dev_datasets": list(DEV_ROBOTS),
        "heldout_datasets": list(HELDOUT_ROBOTS),
        "graph": "BOSS channel summary",
        "folds": 5,
        "seed": 0,
        "masking": "interleaved seed-0 permutation; each channel masked exactly once",
        "encoder": {
            "model": ENCODER_MODEL,
            "revision": ENCODER_REVISION,
            "prefix": ENCODER_PREFIX,
            "project_lora": False,
        },
        "metric": "fold-local Hungarian Match-ACC",
        "feature_propagation": {
            "version": FEATURE_PROP_VERSION,
            "graph_projection": "binary undirected",
            "max_iter": 40,
            "tol": 1e-8,
            "fallback": "zeros",
        },
        "graphmae_gcn": {
            "version": GRAPHMAE_VERSION,
            "epochs": args.graphmae_epochs,
            "split": "dev targets LODO; held-out UR5e trains on all three dev robots",
            "config_except_device": dataclasses.asdict(
                GraphMAEConfig(epochs=args.graphmae_epochs, device="auto")
            ),
        },
        "clip_dissect_e5": {
            "bank_version": ROBOT_BANK_VERSION,
            "bank_size": args.text_bank_size,
            "profile_top_k": args.profile_top_k,
        },
        "llm": {
            "model": args.model,
            "domain": "robot",
            "profile_top_k": args.profile_top_k,
            "semantic_repair": {
                "enabled": True,
                "max_attempts": 2,
                "version": "semantic-repair-v1",
            },
            "prompt_versions": {
                "autointerp_explain": ROBOT_AUTOINTERP_EXPLAIN_PROMPT_VERSION,
                "autointerp_simulate": ROBOT_AUTOINTERP_SIMULATE_PROMPT_VERSION,
                "delphi_explain": ROBOT_DELPHI_EXPLAIN_PROMPT_VERSION,
                "delphi_detect": ROBOT_DELPHI_DETECT_PROMPT_VERSION,
            },
        },
        "data_sha256": provenance,
        "exact_run": False,
        "llm_judge_run": False,
    }
    cfg_hash = config_hash(config)
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("config_hash") != cfg_hash:
            raise RuntimeError(f"output directory has another configuration: {root}")
    else:
        atomic_write_json(
            manifest_path,
            {
                "config": config,
                "config_hash": cfg_hash,
                "git_commit": _git_commit(),
                "source_sha256": _source_hashes(),
                "created_at": _timestamp(),
            },
        )

    status: dict[str, Any] = {
        "state": "running",
        "selected_methods": selected_methods,
        "selected_datasets": selected_datasets,
        "current": None,
        "started_at": _timestamp(),
        "updated_at": _timestamp(),
        "config_hash": cfg_hash,
        "llm_judge_run": False,
    }

    def update_status(current: Mapping[str, Any]) -> None:
        status["current"] = dict(current)
        status["updated_at"] = _timestamp()
        atomic_write_json(root / "status.json", status)

    atomic_write_json(root / "status.json", status)
    try:
        encoder = FrozenE5Encoder(
            args.hf_cache.resolve(),
            device=args.encoder_device,
            batch_size=args.embed_batch_size,
            allow_download=args.allow_download,
        )
        concept_bank = None
        if "clip-dissect-e5" in selected_methods:
            bank_path = (
                args.concept_bank.resolve()
                if args.concept_bank is not None
                else root / f"clip_dissect_e5_robot_bank_{args.text_bank_size}.npz"
            )
            build_robot_wordnet_bank(
                args.wordnet_zip.resolve(),
                bank_path,
                encoder.embed,
                size=args.text_bank_size,
            )
            concept_bank = load_concept_bank(
                bank_path, expected_encoder="intfloat/e5-large-v2"
            )

        client = None
        if set(selected_methods) & {"autointerp", "delphi"}:
            api_cache = (
                args.api_cache_dir.resolve()
                if args.api_cache_dir is not None
                else root / "api_cache"
            )
            client = BaselineAPIClient(
                api_cache,
                model=args.model,
                budget_usd=args.budget_usd,
                max_attempts=8,
                retry_base_seconds=2.0,
            )

        for method in selected_methods:
            for name in selected_datasets:
                dataset = datasets[name]
                update_status({"stage": "run", "method": method, "dataset": name})
                if method == "feature-propagation":
                    _run_feature_propagation(root, dataset, encoder, cfg_hash)
                elif method == "graphmae-gcn":
                    _run_graphmae(
                        root,
                        dataset,
                        datasets,
                        encoder,
                        cfg_hash,
                        epochs=args.graphmae_epochs,
                        device=args.graphmae_device,
                    )
                elif method == "clip-dissect-e5":
                    assert concept_bank is not None
                    _run_clip_dissect(
                        root,
                        dataset,
                        encoder,
                        concept_bank,
                        cfg_hash,
                        profile_top_k=args.profile_top_k,
                    )
                else:
                    assert client is not None
                    _run_llm_method(
                        root,
                        method,
                        dataset,
                        encoder,
                        client,
                        cfg_hash,
                        profile_top_k=args.profile_top_k,
                        case_workers=args.case_workers,
                        status_callback=update_status,
                    )
        # Always rebuild the combined five-method view so the documented
        # non-LLM and LLM commands can share one resumable output directory.
        summary = _build_summary(root, METHODS, selected_datasets, cfg_hash)
        status.update(
            {
                "state": "complete",
                "current": None,
                "updated_at": _timestamp(),
                "finished_at": _timestamp(),
                "summary_path": str(root / "summary.json"),
                "api_spend_usd": client.spent_usd if client is not None else 0.0,
            }
        )
        atomic_write_json(root / "status.json", status)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return 0
    except Exception as exc:
        status.update(
            {
                "state": "failed",
                "current": status.get("current"),
                "updated_at": _timestamp(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        atomic_write_json(root / "status.json", status)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
