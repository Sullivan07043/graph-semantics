"""Run the CLIP-Dissect E5 text adaptation as a leakage-safe Task 1 baseline.

Task 1 hides observed-variable text while retaining the response matrix and
graph.  For each of the shared five folds, respondent profiles are rendered
from fold-visible item text only.  Every masked item's standardized response
column is then interpreted as one numerical feature by the E5 text adaptation.

Predictions are cached atomically per item in gold-free ``generation_cases``.
Only after every selected fold for a dataset is frozen does the runner encode
the full gold-label set, materialize the final case artifacts, and evaluate the
predicted construct names with the same frozen base E5 encoder used by
``run_task1.py``.  Its exact Hungarian Match-ACC and global-label Exact
definitions are reproduced locally.  No LLM judge is imported or run.

Example (from the repository root)::

    python -m v6.baselines.runners.clip_dissect_task1 --datasets report19
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any, Mapping, Sequence

import numpy as np


V6_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = V6_DIR.parent
WORKSPACE_ROOT = REPO_ROOT.parent
if str(V6_DIR) not in sys.path:
    sys.path.insert(0, str(V6_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("HF_CACHE", str(WORKSPACE_ROOT / ".hf_cache"))

from v6.baselines.protocol import (  # noqa: E402
    REPORT_DATASETS,
    atomic_write_json,
    config_hash,
    file_sha256 as _file_sha256,
    observed_activation_matrix,
    outer_folds,
    render_profiles,
    report_loaders,
    select_datasets,
    source_sha256,
)
from v6.baselines.clip_dissect_bank import (  # noqa: E402
    BANK_VERSION,
    build_wordnet_domain_bank,
)
from v6.baselines.clip_dissect_e5 import (  # noqa: E402
    ClipDissectE5Config,
    ConceptBank,
    SCORER_VERSION,
    load_concept_bank,
    run_clip_dissect_e5,
)

METHOD_ID = "text-dissect"
METHOD_LABEL = "CLIP-Dissect (E5 text adaptation)"
SOURCE_FILES = (
    "v6/baselines/runners/clip_dissect_task1.py",
    "v6/baselines/protocol.py",
    "v6/baselines/clip_dissect_bank.py",
    "v6/baselines/clip_dissect_e5.py",
)


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="report19")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile-top-k", type=int, default=6)
    parser.add_argument("--embed-batch-size", type=int, default=64)
    parser.add_argument("--text-bank-size", type=int, default=4096)
    parser.add_argument("--text-concept-batch-size", type=int, default=4096)
    parser.add_argument("--concept-bank", type=Path)
    parser.add_argument("--output-dir", type=Path)
    # Execution-only pilot control; excluded from the scientific config hash so
    # the identical command can resume a smoke run into a complete run.
    parser.add_argument("--max-dataset-folds", type=int, default=0)
    args = parser.parse_args(argv)
    if args.folds != 5:
        parser.error("the frozen Task 1 protocol requires exactly five folds")
    if args.profile_top_k < 1 or args.embed_batch_size < 1:
        parser.error("profile and embedding batch sizes must be positive")
    if args.text_bank_size < 1 or args.text_concept_batch_size < 1:
        parser.error(
            "CLIP-Dissect E5 text-adaptation bank and chunk sizes must be positive"
        )
    if args.max_dataset_folds < 0:
        parser.error("--max-dataset-folds cannot be negative")
    return args


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
            capture_output=True, text=True, timeout=10,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _case_path(root: Path, dataset: str, fold: int, observed_index: int) -> Path:
    return (
        root / "cases" / METHOD_ID / dataset / f"fold_{fold:02d}"
        / f"item_{observed_index:03d}.json"
    )


def _generation_case_path(
    root: Path, dataset: str, fold: int, observed_index: int
) -> Path:
    """Gold-free resumable cache used exclusively by the generation pass."""

    return (
        root / "generation_cases" / METHOD_ID / dataset / f"fold_{fold:02d}"
        / f"item_{observed_index:03d}.json"
    )


def _fold_metric_path(root: Path, dataset: str, fold: int) -> Path:
    return root / "fold_metrics" / METHOD_ID / dataset / f"fold_{fold:02d}.json"


def _load_record(path: Path, expected_hash: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid resumable record {path}: {exc}") from exc
    if record.get("config_hash") != expected_hash:
        raise RuntimeError(f"record {path} belongs to a different configuration")
    if not record.get("construct_name"):
        raise RuntimeError(f"completed record {path} has no construct_name")
    return record


def _embedding_function(batch_size: int):
    import encode

    if encode.ENCODER != "e5-large":
        raise RuntimeError(
            "The CLIP-Dissect E5 text adaptation for Task 1 requires "
            "GRAPHSEM_ENCODER=e5-large; "
            f"current encoder is {encode.ENCODER!r}"
        )

    def embed(texts: Sequence[str]) -> np.ndarray:
        return np.asarray(
            encode.embed(list(texts), batch_size=batch_size), dtype=np.float64
        )

    return embed


def _normalise_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values / (np.linalg.norm(values, axis=1, keepdims=True) + 1e-9)


def _task1_metrics_from_embeddings(
    predicted_embeddings: np.ndarray,
    masked_indices: Sequence[int],
    all_gold_embeddings: np.ndarray,
) -> dict[str, float]:
    """Mirror ``metrics.match_acc`` and ``metrics.exact_acc`` without judge imports."""

    masked = np.asarray([int(index) for index in masked_indices], dtype=int)
    predicted = np.asarray(predicted_embeddings, dtype=np.float64)
    gold = np.asarray(all_gold_embeddings, dtype=np.float64)
    if predicted.ndim != 2 or gold.ndim != 2 or predicted.shape[1] != gold.shape[1]:
        raise ValueError("predicted and gold embeddings must be compatible 2-D arrays")
    if predicted.shape[0] != len(masked):
        raise ValueError("one predicted construct embedding is required per masked item")
    if not len(masked):
        raise ValueError("at least one masked item is required for Task 1 evaluation")
    if np.min(masked) < 0 or np.max(masked) >= len(gold):
        raise IndexError("masked item index is outside the gold embedding matrix")

    from scipy.optimize import linear_sum_assignment

    pred_norm = _normalise_rows(predicted)
    gold_norm = _normalise_rows(gold)
    similarities = pred_norm @ gold_norm[masked].T
    _, assignment = linear_sum_assignment(-similarities)
    match_acc = float(np.mean(assignment == np.arange(len(masked))))
    nearest = np.argmax(pred_norm @ gold_norm.T, axis=1)
    exact = float(np.mean(nearest == masked))
    return {"match_acc": match_acc, "exact": exact}


def _task1_metrics(
    predicted_names: Sequence[str],
    masked_indices: Sequence[int],
    all_gold_labels: Sequence[str],
    embed,
) -> dict[str, float]:
    """Embed generated names and evaluate with the locked base-E5 Task 1 ruler."""

    return _task1_metrics_from_embeddings(
        embed([str(value) for value in predicted_names]),
        masked_indices,
        embed([str(value) for value in all_gold_labels]),
    )


def _interpret_fold(
    dataset: Mapping[str, Any],
    masked_indices: Sequence[int],
    embed,
    concept_bank: ConceptBank,
    text_config: ClipDissectE5Config,
    *,
    profile_top_k: int,
) -> tuple[list[Any], list[dict[str, Any]]]:
    """Interpret masked Task 1 items with the CLIP-Dissect E5 adaptation."""

    graph = dataset["graph"]
    observed = list(graph.observed)
    labels = dataset["labels"]
    X = np.asarray(dataset["X"], dtype=float)
    masked = sorted(int(index) for index in masked_indices)
    masked_set = set(masked)
    visible = [index for index in range(len(observed)) if index not in masked_set]

    # Give the renderer a capability-limited mapping: masked target text is not
    # merely ignored by convention; it is absent from the object passed across
    # the generation boundary.
    visible_labels = {observed[index]: labels[observed[index]] for index in visible}
    profiles, _ = render_profiles(
        X, observed, visible_labels, visible, top_k=profile_top_k
    )
    profile_embeddings = embed(profiles)
    activations, activation_metadata = observed_activation_matrix(
        X, len(observed), masked
    )
    opaque_ids = [f"masked_item_{position:03d}" for position in range(len(masked))]
    results = run_clip_dissect_e5(
        None,
        activations,
        concept_bank,
        profile_embeddings=profile_embeddings,
        latent_ids=opaque_ids,
        config=text_config,
    )
    if len(results) != len(masked):
        raise RuntimeError(
            "CLIP-Dissect E5 adaptation returned "
            f"{len(results)} results for {len(masked)} masked items"
        )
    return results, activation_metadata


def _record_case(
    *,
    root: Path,
    dataset_name: str,
    fold: int,
    observed_index: int,
    observed_node: str,
    prediction: Mapping[str, Any],
    activation_metadata: Mapping[str, Any],
    cfg_hash: str,
) -> dict[str, Any]:
    """Freeze a generated prediction without accepting any gold-label text."""

    record = {
        "task": 1,
        "method": METHOD_LABEL,
        "method_id": METHOD_ID,
        "dataset": dataset_name,
        "fold": fold,
        "observed_index": observed_index,
        "observed_node": observed_node,
        "construct_name": str(prediction["construct_name"]),
        "prediction": dict(prediction),
        "activation": dict(activation_metadata),
        "config_hash": cfg_hash,
        "completed_at": _timestamp(),
        "llm_judge": None,
    }
    atomic_write_json(
        _generation_case_path(root, dataset_name, fold, observed_index), record
    )
    return record


def _attach_gold_labels(
    root: Path,
    dataset: Mapping[str, Any],
    fold_specs: Sequence[tuple[int, Sequence[int]]],
    cfg_hash: str,
) -> None:
    """Attach gold labels only after the dataset generation pass is complete."""

    observed = list(dataset["graph"].observed)
    labels = dataset["labels"]
    for fold, masked_indices in fold_specs:
        for observed_index in sorted(int(index) for index in masked_indices):
            generation_path = _generation_case_path(
                root, dataset["name"], fold, observed_index
            )
            record = _load_record(generation_path, cfg_hash)
            if record is None:
                raise RuntimeError(
                    f"cannot attach gold before prediction is frozen: {generation_path}"
                )
            gold_label = str(labels[observed[observed_index]])
            record["gold_label"] = gold_label
            record["gold_attached_at"] = _timestamp()
            atomic_write_json(
                _case_path(root, dataset["name"], fold, observed_index), record
            )


def _evaluate_completed_fold(
    root: Path,
    dataset: Mapping[str, Any],
    fold: int,
    masked_indices: Sequence[int],
    cfg_hash: str,
    embed,
    all_gold_embeddings: np.ndarray,
) -> dict[str, Any] | None:
    masked = sorted(int(index) for index in masked_indices)
    records = [
        _load_record(_case_path(root, dataset["name"], fold, index), cfg_hash)
        for index in masked
    ]
    if any(record is None for record in records):
        return None
    predictions = [str(record["construct_name"]) for record in records if record]
    scores = _task1_metrics_from_embeddings(
        embed(predictions), masked, all_gold_embeddings
    )
    metric = {
        "task": 1,
        "method": METHOD_LABEL,
        "method_id": METHOD_ID,
        "dataset": dataset["name"],
        "fold": fold,
        "masked_observed_count": len(masked),
        **scores,
        "llm_judge": None,
        "config_hash": cfg_hash,
        "evaluated_at": _timestamp(),
    }
    atomic_write_json(_fold_metric_path(root, dataset["name"], fold), metric)
    return metric


def _run_dataset_two_pass(
    *,
    root: Path,
    dataset: Mapping[str, Any],
    fold_specs: Sequence[tuple[int, Sequence[int]]],
    cfg_hash: str,
    embed,
    text_bank: ConceptBank,
    text_config: ClipDissectE5Config,
    profile_top_k: int,
    status: dict[str, Any],
) -> None:
    """Freeze all fold predictions, then cross the gold/evaluation boundary.

    Generation reads only each fold's visible labels and writes gold-free
    ``generation_cases``.  The final ``cases`` and fold metrics are materialized
    only after every selected fold has a complete generation cache.  Generation
    resumes exclusively from those gold-free files, so even extending a pilot
    run cannot bring an earlier fold's gold text back across the boundary.
    """

    dataset_name = str(dataset["name"])
    observed = list(dataset["graph"].observed)

    # Pass 1: generation and resumable gold-free caching.
    for fold, masked_indices in fold_specs:
        masked = sorted(int(index) for index in masked_indices)
        existing = [
            _load_record(
                _generation_case_path(root, dataset_name, fold, index), cfg_hash
            )
            for index in masked
        ]
        missing_positions = [
            position for position, record in enumerate(existing) if record is None
        ]
        if missing_positions:
            status["current"] = {
                "stage": "generation",
                "method": METHOD_ID,
                "dataset": dataset_name,
                "fold": fold,
            }
            status["updated_at"] = _timestamp()
            atomic_write_json(root / "status.json", status)
            results, activation_metadata = _interpret_fold(
                dataset,
                masked,
                embed,
                text_bank,
                text_config,
                profile_top_k=profile_top_k,
            )
            for position in missing_positions:
                prediction = results[position].to_dict()
                observed_index = masked[position]
                _record_case(
                    root=root,
                    dataset_name=dataset_name,
                    fold=fold,
                    observed_index=observed_index,
                    observed_node=observed[observed_index],
                    prediction=prediction,
                    activation_metadata=activation_metadata[position],
                    cfg_hash=cfg_hash,
                )
                status["completed_new_cases"] += 1
            status["resumed_cases"] += len(masked) - len(missing_positions)
        else:
            status["resumed_cases"] += len(masked)
        status["updated_at"] = _timestamp()
        atomic_write_json(root / "status.json", status)

    # Prove every selected fold is frozen before any all-gold read/embedding.
    frozen_items = 0
    for fold, masked_indices in fold_specs:
        for observed_index in sorted(int(index) for index in masked_indices):
            path = _generation_case_path(root, dataset_name, fold, observed_index)
            if _load_record(path, cfg_hash) is None:
                raise RuntimeError(f"dataset generation pass is incomplete: {path}")
            frozen_items += 1
    atomic_write_json(root / "generation_frozen" / f"{dataset_name}.json", {
        "dataset": dataset_name,
        "folds": [int(fold) for fold, _ in fold_specs],
        "frozen_items": frozen_items,
        "config_hash": cfg_hash,
        "frozen_at": _timestamp(),
    })

    # Pass 2: local evaluator.  This is the first full-gold text access.
    status["current"] = {
        "stage": "evaluation",
        "method": METHOD_ID,
        "dataset": dataset_name,
    }
    status["updated_at"] = _timestamp()
    atomic_write_json(root / "status.json", status)
    labels = dataset["labels"]
    all_gold_embeddings = embed([str(labels[name]) for name in observed])
    _attach_gold_labels(root, dataset, fold_specs, cfg_hash)
    for fold, masked_indices in fold_specs:
        metric = _evaluate_completed_fold(
            root,
            dataset,
            fold,
            masked_indices,
            cfg_hash,
            embed,
            all_gold_embeddings,
        )
        if metric is None:
            raise RuntimeError(
                f"evaluation could not load frozen cases for {dataset_name} fold {fold}"
            )
    status["updated_at"] = _timestamp()
    atomic_write_json(root / "status.json", status)


def _mean(values: Sequence[Any]) -> float | None:
    keep = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(keep)) if keep else None


def _build_summary(
    root: Path, config: Mapping[str, Any], cfg_hash: str
) -> dict[str, Any]:
    records = []
    case_root = root / "cases" / METHOD_ID
    if case_root.is_dir():
        for path in sorted(case_root.glob("**/*.json")):
            record = _load_record(path, cfg_hash)
            if record:
                records.append(record)
    folds = []
    metric_root = root / "fold_metrics" / METHOD_ID
    if metric_root.is_dir():
        for path in sorted(metric_root.glob("**/*.json")):
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("config_hash") == cfg_hash:
                folds.append(value)

    per_dataset: dict[str, Any] = {}
    for dataset in config["datasets"]:
        selected = [record for record in records if record["dataset"] == dataset]
        selected_folds = [metric for metric in folds if metric["dataset"] == dataset]
        per_dataset[dataset] = {
            "completed_items": len(selected),
            "completed_folds": len(selected_folds),
            "match_acc": _mean([metric.get("match_acc") for metric in selected_folds]),
            "exact": _mean([metric.get("exact") for metric in selected_folds]),
            "positive_soft_wpmi": _mean([
                record["prediction"]["native_diagnostics"].get("positive_soft_wpmi")
                for record in selected
            ]),
            "positive_rank_reorder": _mean([
                record["prediction"]["native_diagnostics"].get("positive_rank_reorder")
                for record in selected
            ]),
            # Standard five-fold Task 1 partitions items; an item is masked once,
            # so cross-fold per-item rank stability is not defined.
            "rank_stability": None,
            "llm_judge": None,
        }
    summary = {
        "config": dict(config),
        "config_hash": cfg_hash,
        "llm_judge_run": False,
        "generated_at": _timestamp(),
        "method": {
            "id": METHOD_ID,
            "label": METHOD_LABEL,
            "completed_items": len(records),
            "dataset_macro_match_acc": _mean([
                value["match_acc"] for value in per_dataset.values()
            ]),
            "dataset_macro_exact": _mean([
                value["exact"] for value in per_dataset.values()
            ]),
            "datasets": per_dataset,
        },
    }
    atomic_write_json(root / "summary.json", summary)
    return summary


def _load_or_build_bank(args: argparse.Namespace, embed) -> tuple[ConceptBank, Path]:
    selected_path = (
        args.concept_bank.resolve()
        if args.concept_bank is not None
        else V6_DIR / "outputs" / "interpretability_baselines"
        / f"text_dissect_e5_domain_{args.text_bank_size}.npz"
    )
    if args.concept_bank is None:
        build_wordnet_domain_bank(
            WORKSPACE_ROOT / ".nltk_data" / "corpora" / "wordnet.zip",
            selected_path,
            embed,
            size=args.text_bank_size,
        )
    bank = load_concept_bank(selected_path, expected_encoder="e5-large-v2")
    return bank, selected_path.resolve()


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    datasets = select_datasets(args.datasets)
    text_config = ClipDissectE5Config(
        concept_batch_size=args.text_concept_batch_size
    )
    embed = _embedding_function(args.embed_batch_size)
    text_bank, bank_path = _load_or_build_bank(args, embed)
    bank_sha256 = _file_sha256(bank_path)
    root = (
        args.output_dir
        or V6_DIR / "outputs" / "interpretability_baselines"
        / f"task1_text_dissect_e5_report19_seed{args.seed}_v3"
    ).resolve()
    config = {
        "protocol": "task1-visible-label-folds-v2-text-dissect-e5-v3",
        "task": 1,
        "datasets": datasets,
        "report_dataset_order": list(REPORT_DATASETS),
        "folds": args.folds,
        "seed": args.seed,
        "method": METHOD_ID,
        "profile_top_k": args.profile_top_k,
        "encoder": "intfloat/e5-large-v2 (frozen base)",
        "text_bank": {
            "selection_version": BANK_VERSION,
            "size": args.text_bank_size,
            "sha256": bank_sha256,
            "explicit_path": (
                str(args.concept_bank.resolve()) if args.concept_bank is not None else None
            ),
        },
        "text_dissect_scorer": {
            "version": SCORER_VERSION,
            "parameters": asdict(text_config),
            "tie_policy": "exact-score midranks; lexical identity presentation tie-break",
        },
        "activation": "per-item respondent response column; population z-score",
        "provenance_boundary": (
            "all selected folds frozen in gold-free generation_cases before gold encoding"
        ),
        "llm_judge_run": False,
    }
    cfg_hash = config_hash(config)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("config_hash") != cfg_hash:
            raise RuntimeError(
                f"output directory has a different config: {root}; choose another --output-dir"
            )
        if manifest.get("concept_bank_sha256") != bank_sha256:
            raise RuntimeError(
                f"output directory has a different concept bank: {root}; "
                "choose another --output-dir"
            )
    else:
        atomic_write_json(manifest_path, {
            "config": config,
            "config_hash": cfg_hash,
            "git_commit": _git_commit(),
            "created_at": _timestamp(),
            "concept_bank_path": str(bank_path),
            "concept_bank_sha256": bank_sha256,
            "source_sha256": source_sha256(REPO_ROOT, SOURCE_FILES),
        })

    status: dict[str, Any] = {
        "state": "running",
        "config_hash": cfg_hash,
        "started_at": _timestamp(),
        "updated_at": _timestamp(),
        "current": None,
        "completed_new_cases": 0,
        "resumed_cases": 0,
        "llm_judge_run": False,
    }
    atomic_write_json(root / "status.json", status)

    try:
        loaders = report_loaders()
        dataset_folds_seen = 0

        for dataset_name in datasets:
            dataset = loaders[dataset_name]()
            graph = dataset["graph"]
            observed = list(graph.observed)
            X = np.asarray(dataset["X"], dtype=float)
            labels = dataset["labels"]
            if X.ndim != 2 or X.shape[1] != len(observed):
                raise ValueError(f"{dataset_name}: X columns do not match graph.observed")
            missing_labels = [name for name in observed if name not in labels]
            if missing_labels:
                raise ValueError(f"{dataset_name}: labels missing observed nodes {missing_labels}")

            masks = outer_folds(len(observed), args.folds, args.seed)
            fold_specs: list[tuple[int, Sequence[int]]] = []
            for fold, mask in enumerate(masks):
                if (
                    args.max_dataset_folds
                    and dataset_folds_seen >= args.max_dataset_folds
                ):
                    break
                dataset_folds_seen += 1
                fold_specs.append((fold, sorted(int(index) for index in mask)))

            if fold_specs:
                _run_dataset_two_pass(
                    root=root,
                    dataset=dataset,
                    fold_specs=fold_specs,
                    cfg_hash=cfg_hash,
                    embed=embed,
                    text_bank=text_bank,
                    text_config=text_config,
                    profile_top_k=args.profile_top_k,
                    status=status,
                )

            if (
                args.max_dataset_folds
                and dataset_folds_seen >= args.max_dataset_folds
            ):
                break

        summary = _build_summary(root, config, cfg_hash)
        status.update({
            "state": "pilot_complete" if args.max_dataset_folds else "complete",
            "current": None,
            "updated_at": _timestamp(),
            "finished_at": _timestamp(),
            "summary_path": str(root / "summary.json"),
        })
        atomic_write_json(root / "status.json", status)
        print(json.dumps({
            "status": status["state"],
            "output": str(root),
            "completed_new_cases": status["completed_new_cases"],
            "dataset_macro_match_acc": summary["method"]["dataset_macro_match_acc"],
            "dataset_macro_exact": summary["method"]["dataset_macro_exact"],
        }, ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:
        status.update({
            "state": "failed",
            "updated_at": _timestamp(),
            "error_type": type(exc).__name__,
            "error": str(exc),
        })
        atomic_write_json(root / "status.json", status)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
