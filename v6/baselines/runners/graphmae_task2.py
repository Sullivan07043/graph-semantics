"""Train and evaluate leakage-safe GraphMAE-GCN Task 2 baselines.

Every target dataset receives its own checkpoint.  A DEV target is trained on
``pool.DEV - {target}``; a held-out target is trained on all of ``pool.DEV``.
The three held-out datasets are never training examples for any checkpoint.

The text encoder is frozen base E5-large-v2 at an explicit Hugging Face commit.
No project LoRA is loaded.  Training sees observed descriptions only; latent
ground-truth descriptions are embedded only after all fold predictions exist.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np


V6_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = V6_DIR.parent
WORKSPACE_ROOT = REPO_ROOT.parent
if str(V6_DIR) not in sys.path:
    sys.path.insert(0, str(V6_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import metrics
import pool
import testbeds
from v6.baselines.graphmae_gcn import (
    CHECKPOINT_FORMAT,
    METHOD_VERSION,
    GraphExample,
    GraphMAEBaseline,
    GraphMAEConfig,
    file_sha256,
)


PROTOCOL_VERSION = "graphmae-task2-visible-label-folds-lodo-v1"
SPLIT_VERSION = "pool-dev-heldout-2026-07-14"
ENCODER_KEY = "e5-large"
ENCODER_MODEL = "intfloat/e5-large-v2"
ENCODER_PREFIX = "query: "
GRAPHMAE_SOURCE = V6_DIR / "baselines" / "graphmae_gcn.py"
ALL_LOADERS = {**testbeds.LOADERS, **pool.LOADERS}
REPORT_DATASETS = [*pool.DEV, *pool.HELDOUT]

if len(REPORT_DATASETS) != 19 or len(set(REPORT_DATASETS)) != 19:
    raise RuntimeError("the frozen report suite must contain 19 unique datasets")
if set(REPORT_DATASETS) - set(ALL_LOADERS):
    raise RuntimeError("one or more report datasets have no loader")


def training_datasets_for(target: str) -> list[str]:
    """Return the only allowed zero-shot training set for ``target``."""

    if target not in REPORT_DATASETS:
        raise ValueError(f"dataset is outside the frozen report suite: {target}")
    names = [name for name in pool.DEV if name != target]
    if target in names:
        raise AssertionError("target dataset entered its own training split")
    heldout_overlap = set(names) & set(pool.HELDOUT)
    if heldout_overlap:
        raise AssertionError(f"held-out datasets entered training: {heldout_overlap}")
    return names


def fold_indices(number_observed: int, folds: int = 5, seed: int = 0) -> list[np.ndarray]:
    if number_observed < folds:
        raise ValueError("a dataset must have at least one observed variable per fold")
    permutation = np.random.default_rng(seed).permutation(number_observed)
    return [permutation[position::folds] for position in range(folds)]


def resolve_encoder_revision(
    cache_folder: Path, explicit_revision: str | None = None
) -> str:
    revision = explicit_revision or os.environ.get("GRAPHMAE_ENCODER_REVISION")
    if revision is None:
        model_cache = "models--" + ENCODER_MODEL.replace("/", "--")
        reference = cache_folder / model_cache / "refs" / "main"
        if reference.is_file():
            revision = reference.read_text(encoding="utf-8").strip()
    if revision is None or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise RuntimeError(
            "GraphMAE requires a pinned 40-character E5 commit. Set "
            "--encoder-revision or GRAPHMAE_ENCODER_REVISION."
        )
    return revision


class FrozenE5Encoder:
    def __init__(
        self,
        *,
        cache_folder: Path,
        revision: str,
        device: str | None,
        local_files_only: bool,
        batch_size: int,
    ) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.revision = revision
        self.batch_size = batch_size
        self.model = SentenceTransformer(
            ENCODER_MODEL,
            revision=revision,
            device=selected_device,
            cache_folder=str(cache_folder),
            local_files_only=local_files_only,
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            raise ValueError("cannot embed an empty text list")
        values = self.model.encode(
            [ENCODER_PREFIX + text for text in texts],
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(values, dtype=np.float32)


class DatasetCache:
    """Load each dataset and encode its observed labels at most once per process."""

    def __init__(self, encoder: FrozenE5Encoder) -> None:
        self.encoder = encoder
        self.datasets: dict[str, dict[str, Any]] = {}
        self.embeddings: dict[str, dict[str, np.ndarray]] = {}

    def dataset(self, name: str) -> dict[str, Any]:
        if name not in self.datasets:
            self.datasets[name] = ALL_LOADERS[name]()
        return self.datasets[name]

    def observed_embeddings(self, name: str) -> dict[str, np.ndarray]:
        if name not in self.embeddings:
            dataset = self.dataset(name)
            graph = dataset["graph"]
            labels = dataset["labels"]
            missing = set(graph.observed) - set(labels)
            if missing:
                raise RuntimeError(f"{name} lacks observed labels: {sorted(missing)}")
            matrix = self.encoder.embed([labels[node] for node in graph.observed])
            self.embeddings[name] = {
                node: matrix[position] for position, node in enumerate(graph.observed)
            }
        return self.embeddings[name]

    def example(self, name: str) -> GraphExample:
        return GraphExample(
            name,
            self.dataset(name)["graph"],
            self.observed_embeddings(name),
        )


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_state() -> dict[str, Any]:
    """Record the checkout state used by the evaluator, including dirtiness."""

    commit = _git_commit()
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
    """Convert argparse/config values to deterministic JSON-compatible values."""

    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _artifact_path(path: Path, output_dir: Path) -> str:
    """Prefer portable paths relative to the formal output directory."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(output_dir.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _collapse_target_values(values: dict[str, Any]) -> dict[str, Any]:
    """Store one shared value when every checkpoint agrees, else retain all."""

    if not values:
        return {"shared": None, "targets_verified": []}
    targets = list(values)
    first = values[targets[0]]
    canonical = json.dumps(first, sort_keys=True, separators=(",", ":"))
    if all(
        json.dumps(values[target], sort_keys=True, separators=(",", ":"))
        == canonical
        for target in targets[1:]
    ):
        return {"shared": first, "targets_verified": targets}
    return {"by_target": values}


def _checkpoint_manifest_payload(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read training config and provenance exactly as frozen in a checkpoint."""

    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        raise RuntimeError(f"unsupported GraphMAE checkpoint for manifest: {path}")
    if payload.get("method_version") != METHOD_VERSION:
        raise RuntimeError(f"GraphMAE method-version mismatch in {path}")

    training_config = payload.get("config")
    metadata = payload.get("metadata")
    if not isinstance(training_config, dict) or not isinstance(metadata, dict):
        raise RuntimeError(f"checkpoint lacks manifest provenance: {path}")
    source_hashes = metadata.get("source_sha256")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise RuntimeError(f"checkpoint lacks training-time source hashes: {path}")
    for source, digest in source_hashes.items():
        if not isinstance(source, str) or re.fullmatch(r"[0-9a-f]{64}", str(digest)) is None:
            raise RuntimeError(f"checkpoint has an invalid source hash: {path}")
    return dict(training_config), dict(metadata)


def build_manifest(
    arguments: argparse.Namespace,
    targets: list[str],
    revision: str,
    config: GraphMAEConfig,
    results: dict[str, dict[str, Any]],
    *,
    git_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a self-contained eval/run manifest from frozen artifacts.

    Training-time source hashes are deliberately read from each checkpoint.
    The evaluator may have changed since training, so recomputing those hashes
    from the current working tree would create false provenance.
    """

    if list(results) != targets:
        raise RuntimeError(
            "manifest requires one ordered evaluation result for every selected target"
        )
    output_dir = arguments.output_dir.resolve()
    summary_path = output_dir / "summary.json"
    if not summary_path.is_file():
        raise RuntimeError("summary.json must be frozen before manifest generation")

    artifacts: dict[str, Any] = {}
    checkpoint_source_hashes: dict[str, dict[str, str]] = {}
    checkpoint_training_configs: dict[str, dict[str, Any]] = {}
    for target in targets:
        result = results[target]
        result_path = output_dir / "results" / target / "result.json"
        checkpoint_path = Path(result["checkpoint"])
        prediction_path = Path(result["prediction_artifact"])
        for kind, path in (
            ("checkpoint", checkpoint_path),
            ("result", result_path),
            ("prediction", prediction_path),
        ):
            if not path.is_file():
                raise RuntimeError(f"missing {kind} artifact for {target}: {path}")

        checkpoint_digest = file_sha256(checkpoint_path)
        prediction_digest = file_sha256(prediction_path)
        if checkpoint_digest != result.get("checkpoint_sha256"):
            raise RuntimeError(f"checkpoint hash mismatch for {target}")
        if prediction_digest != result.get("prediction_sha256"):
            raise RuntimeError(f"prediction hash mismatch for {target}")

        training_config, metadata = _checkpoint_manifest_payload(checkpoint_path)
        validate_checkpoint_metadata(metadata, target, revision)
        source_hashes = {
            str(source): str(digest)
            for source, digest in metadata["source_sha256"].items()
        }
        checkpoint_source_hashes[target] = source_hashes
        checkpoint_training_configs[target] = _jsonable(training_config)
        artifacts[target] = {
            "split": result["split"],
            "train_datasets": list(result["train_datasets"]),
            "checkpoint": {
                "path": _artifact_path(checkpoint_path, output_dir),
                "sha256": checkpoint_digest,
                "training_git_commit": metadata.get("git_commit"),
            },
            "result": {
                "path": _artifact_path(result_path, output_dir),
                "sha256": file_sha256(result_path),
            },
            "prediction": {
                "path": _artifact_path(prediction_path, output_dir),
                "sha256": prediction_digest,
            },
        }

    return {
        "task": 2,
        "arm": "graphmae_gcn_lodo",
        "method_version": METHOD_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "split_version": SPLIT_VERSION,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "targets": list(targets),
        "git": dict(git_state if git_state is not None else _git_state()),
        "cli": _jsonable(vars(arguments)),
        "model_config": {
            "requested_by_cli": _jsonable(dataclasses.asdict(config)),
            "training_from_checkpoints": _collapse_target_values(
                checkpoint_training_configs
            ),
        },
        "evaluation": {
            "folds": int(arguments.folds),
            "fold_seed": int(arguments.fold_seed),
            "llm_judge_run": False,
        },
        "encoder": {
            "key": ENCODER_KEY,
            "model": ENCODER_MODEL,
            "revision": revision,
            "prefix": ENCODER_PREFIX,
            "mode": "frozen-base-no-project-lora",
        },
        "checkpoint_source_hash_origin": (
            "training-time checkpoint metadata; not recomputed by this evaluator"
        ),
        "checkpoint_source_sha256": _collapse_target_values(
            checkpoint_source_hashes
        ),
        "artifacts": artifacts,
        "summary": {
            "path": _artifact_path(summary_path, output_dir),
            "sha256": file_sha256(summary_path),
        },
    }


def checkpoint_metadata(
    target: str,
    train_names: list[str],
    revision: str,
    config: GraphMAEConfig,
) -> dict[str, Any]:
    return {
        "method": "GraphMAE-GCN causal-graph adaptation",
        "method_version": METHOD_VERSION,
        "task2_readout": "indirect decoder output; latents are never reconstruction targets",
        "protocol_version": PROTOCOL_VERSION,
        "split_version": SPLIT_VERSION,
        "target_dataset": target,
        "train_datasets": list(train_names),
        "target_excluded_from_training": target not in train_names,
        "heldout_datasets_in_training": sorted(set(train_names) & set(pool.HELDOUT)),
        "latent_supervision": False,
        "latent_gold_used_for_training": False,
        "encoder": {
            "key": ENCODER_KEY,
            "model": ENCODER_MODEL,
            "revision": revision,
            "prefix": ENCODER_PREFIX,
            "mode": "frozen-base-no-project-lora",
        },
        "seed": config.seed,
        "git_commit": _git_commit(),
        "source_sha256": {
            "graphmae_baseline.py": file_sha256(GRAPHMAE_SOURCE),
            "run_graphmae_task2.py": file_sha256(__file__),
        },
    }


def validate_checkpoint_metadata(
    metadata: dict[str, Any], target: str, revision: str
) -> None:
    expected_train = training_datasets_for(target)
    failures = []
    if metadata.get("method_version") != METHOD_VERSION:
        failures.append("method_version")
    if metadata.get("protocol_version") != PROTOCOL_VERSION:
        failures.append("protocol_version")
    if metadata.get("target_dataset") != target:
        failures.append("target_dataset")
    if metadata.get("train_datasets") != expected_train:
        failures.append("train_datasets")
    if target in set(metadata.get("train_datasets", [])):
        failures.append("target training overlap")
    if set(metadata.get("train_datasets", [])) & set(pool.HELDOUT):
        failures.append("held-out training overlap")
    if metadata.get("latent_supervision") is not False:
        failures.append("latent_supervision")
    encoder = metadata.get("encoder", {})
    if not isinstance(encoder, dict) or encoder.get("model") != ENCODER_MODEL:
        failures.append("encoder model")
    elif encoder.get("revision") != revision or encoder.get("mode") != "frozen-base-no-project-lora":
        failures.append("encoder revision/mode")
    if failures:
        raise RuntimeError(
            f"checkpoint provenance mismatch for {target}: {', '.join(failures)}"
        )


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


def train_target(
    target: str,
    cache: DatasetCache,
    revision: str,
    output_dir: Path,
    config: GraphMAEConfig,
    *,
    force_retrain: bool,
) -> tuple[GraphMAEBaseline, Path, float]:
    checkpoint = output_dir / "checkpoints" / f"{target}.pt"
    if checkpoint.is_file() and not force_retrain:
        model = GraphMAEBaseline.load_checkpoint(checkpoint, device=config.device)
        validate_checkpoint_metadata(model.metadata_, target, revision)
        return model, checkpoint, 0.0

    train_names = training_datasets_for(target)
    print(
        f"[graphmae] target={target} train={len(train_names)} DEV datasets "
        f"epochs={config.epochs}",
        flush=True,
    )
    started = time.perf_counter()
    model = GraphMAEBaseline(config).fit([cache.example(name) for name in train_names])
    seconds = time.perf_counter() - started
    model.metadata_ = checkpoint_metadata(target, train_names, revision, config)
    model.metadata_["training_seconds"] = seconds
    model.metadata_["final_training_loss"] = model.history_[-1]
    model.save_checkpoint(checkpoint)
    print(
        f"[graphmae] saved {checkpoint} in {seconds:.1f}s "
        f"loss={model.history_[-1]:.6f}",
        flush=True,
    )
    return model, checkpoint, seconds


def load_target(
    target: str,
    revision: str,
    output_dir: Path,
    device: str,
) -> tuple[GraphMAEBaseline, Path]:
    checkpoint = output_dir / "checkpoints" / f"{target}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"missing {checkpoint}; run with --mode train or --mode run first"
        )
    model = GraphMAEBaseline.load_checkpoint(checkpoint, device=device)
    validate_checkpoint_metadata(model.metadata_, target, revision)
    return model, checkpoint


def evaluate_target(
    target: str,
    model: GraphMAEBaseline,
    checkpoint: Path,
    cache: DatasetCache,
    revision: str,
    output_dir: Path,
    *,
    folds: int,
    fold_seed: int,
) -> dict[str, Any]:
    validate_checkpoint_metadata(model.metadata_, target, revision)
    dataset = cache.dataset(target)
    graph = dataset["graph"]
    observed = list(graph.observed)
    all_observed_embeddings = cache.observed_embeddings(target)
    latent_names = [name for name in graph.latents if name in dataset["latent_gt"]]
    if not latent_names:
        raise RuntimeError(f"{target} has no evaluable latent variables")

    fold_rows = fold_indices(len(observed), folds=folds, seed=fold_seed)
    predictions: list[np.ndarray] = []
    fold_records = []
    for fold_number, hidden_rows in enumerate(fold_rows):
        hidden = {observed[int(position)] for position in hidden_rows}
        visible = {
            node: all_observed_embeddings[node]
            for node in observed
            if node not in hidden
        }
        latent_predictions = model.infer_latents(graph, visible)
        matrix = np.stack([latent_predictions[name] for name in latent_names])
        predictions.append(matrix)
        fold_records.append(
            {
                "fold": fold_number,
                "visible_observed_count": len(visible),
                "hidden_observed": [node for node in observed if node in hidden],
            }
        )

    # Gold text enters only here, after every fold prediction has been produced.
    gold_embeddings = cache.encoder.embed(
        [dataset["latent_gt"][name] for name in latent_names]
    )
    fold_match: list[float] = []
    if len(latent_names) > 1:
        for record, prediction in zip(fold_records, predictions):
            score = metrics.latent_match_acc(prediction, gold_embeddings)
            record["latent_match_acc"] = score
            fold_match.append(score)
    else:
        for record in fold_records:
            record["latent_match_acc"] = None

    target_dir = output_dir / "results" / target
    prediction_path = target_dir / "predictions.npz"
    max_hidden = max(len(rows) for rows in fold_rows)
    padded_hidden = np.full((len(fold_rows), max_hidden), -1, dtype=np.int64)
    for fold_number, rows in enumerate(fold_rows):
        padded_hidden[fold_number, : len(rows)] = rows
    _atomic_npz(
        prediction_path,
        predictions=np.stack(predictions).astype(np.float32),
        latent_names=np.asarray(latent_names),
        # Rows are padded with -1 because observed counts need not divide by five.
        fold_hidden_indices=padded_hidden,
    )
    result = {
        "task": 2,
        "arm": "graphmae_gcn_lodo",
        "method_version": METHOD_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "dataset": target,
        "split": "heldout" if target in pool.HELDOUT else "dev-lodo",
        "zero_shot": True,
        "train_datasets": model.metadata_["train_datasets"],
        "latent_supervision": False,
        "encoder": model.metadata_["encoder"],
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint),
        "prediction_artifact": str(prediction_path.resolve()),
        "prediction_sha256": file_sha256(prediction_path),
        "folds": fold_records,
        "summary": {
            "latent_count": len(latent_names),
            "fold_count": folds,
            "latent_match_acc": float(np.mean(fold_match)) if fold_match else None,
            "llm_judge": None,
        },
        "generated_at_utc": dt.datetime.now(dt.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
    }
    result_path = target_dir / "result.json"
    _atomic_json(result_path, result)
    print(
        f"[graphmae] {target}: match="
        + (f"{result['summary']['latent_match_acc']:.3f}" if fold_match else "N/A (single latent)"),
        flush=True,
    )
    return result


def select_datasets(value: str) -> list[str]:
    if value in ("all", "report19"):
        return list(REPORT_DATASETS)
    if value == "dev":
        return list(pool.DEV)
    if value == "heldout":
        return list(pool.HELDOUT)
    names = [name.strip() for name in value.split(",") if name.strip()]
    unknown = set(names) - set(REPORT_DATASETS)
    if unknown:
        raise ValueError(f"unknown report datasets: {sorted(unknown)}")
    if len(names) != len(set(names)):
        raise ValueError("dataset selection contains duplicates")
    return names


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan", "train", "eval", "run"), default="plan")
    parser.add_argument("--datasets", default="all")
    parser.add_argument(
        "--output-dir", type=Path, default=V6_DIR / "outputs" / "graphmae_task2_lodo"
    )
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
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--mask-rate", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=0)
    parser.add_argument("--force-retrain", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    targets = select_datasets(arguments.datasets)
    revision = resolve_encoder_revision(arguments.hf_cache, arguments.encoder_revision)
    plan = {
        target: {
            "split": "heldout" if target in pool.HELDOUT else "dev-lodo",
            "train_datasets": training_datasets_for(target),
            "train_count": len(training_datasets_for(target)),
            "zero_shot": True,
        }
        for target in targets
    }
    if arguments.mode == "plan":
        print(
            json.dumps(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "encoder_model": ENCODER_MODEL,
                    "encoder_revision": revision,
                    "targets": plan,
                },
                indent=2,
            )
        )
        return 0

    config = GraphMAEConfig(
        hidden_dim=arguments.hidden_dim,
        encoder_layers=arguments.encoder_layers,
        decoder_layers=arguments.decoder_layers,
        mask_rate=arguments.mask_rate,
        epochs=arguments.epochs,
        seed=arguments.seed,
        device=arguments.device,
    )
    encoder = FrozenE5Encoder(
        cache_folder=arguments.hf_cache,
        revision=revision,
        device=arguments.encoder_device,
        local_files_only=not arguments.allow_download,
        batch_size=arguments.batch_size,
    )
    cache = DatasetCache(encoder)
    results: dict[str, Any] = {}
    training_seconds: dict[str, float] = {}
    for target in targets:
        if arguments.mode in ("train", "run"):
            model, checkpoint, seconds = train_target(
                target,
                cache,
                revision,
                arguments.output_dir,
                config,
                force_retrain=arguments.force_retrain,
            )
            stored_seconds = model.metadata_.get("training_seconds")
            training_seconds[target] = (
                float(stored_seconds)
                if seconds == 0.0 and stored_seconds is not None
                else seconds
            )
        else:
            model, checkpoint = load_target(
                target, revision, arguments.output_dir, arguments.device
            )
            stored_seconds = model.metadata_.get("training_seconds")
            if stored_seconds is not None:
                training_seconds[target] = float(stored_seconds)
        if arguments.mode in ("eval", "run"):
            results[target] = evaluate_target(
                target,
                model,
                checkpoint,
                cache,
                revision,
                arguments.output_dir,
                folds=arguments.folds,
                fold_seed=arguments.fold_seed,
            )

    multi_scores = [
        result["summary"]["latent_match_acc"]
        for result in results.values()
        if result["summary"]["latent_match_acc"] is not None
    ]
    summary = {
        "task": 2,
        "arm": "graphmae_gcn_lodo",
        "method_version": METHOD_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "encoder": {
            "model": ENCODER_MODEL,
            "revision": revision,
            "mode": "frozen-base-no-project-lora",
        },
        "targets": targets,
        "completed": list(results),
        "training_seconds": training_seconds,
        "dataset_macro_latent_match_acc": (
            float(np.mean(multi_scores)) if multi_scores else None
        ),
        "results": {
            name: value["summary"] for name, value in results.items()
        },
        "generated_at_utc": dt.datetime.now(dt.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
    }
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(arguments.output_dir / "summary.json", summary)
    if arguments.mode in ("eval", "run"):
        manifest = build_manifest(
            arguments,
            targets,
            revision,
            config,
            results,
        )
        _atomic_json(arguments.output_dir / "manifest.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
