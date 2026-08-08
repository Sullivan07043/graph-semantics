"""Run LLM interpretability baselines as leakage-safe Task 1 baselines.

Task 1 masks observed-variable text while retaining the numeric response
matrix.  For every shared five-fold split, respondent response summaries are
rendered only from fold-visible item text.  Each masked item's population-
standardized response column is then treated as the numerical feature to be
explained by Automated Interpretability or Delphi.

Generated names are cached atomically without gold labels.  Gold item text is
attached and encoded only after all five folds for a dataset are frozen.  The
local evaluator reproduces the Hungarian Match-ACC definition used by the
project's other Task 1 runners.  LLM Judge is never imported or run.

Example (from the repository root)::

    python -m v6.baselines.runners.llm_interpretability_task1 \
        --baselines autointerp --datasets report19 --budget-usd none
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
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

from v6.baselines.api import BaselineAPIClient  # noqa: E402
from v6.baselines.protocol import (  # noqa: E402
    REPORT_DATASETS,
    atomic_write_json,
    config_hash,
    observed_activation_matrix,
    outer_folds,
    render_profiles,
    report_loaders,
    select_datasets,
    source_sha256,
    stable_seed,
)
from v6.baselines.automated_interpretability import (  # noqa: E402
    AUTOINTERP_EXPLAIN_PROMPT_VERSION,
    AUTOINTERP_SIMULATE_PROMPT_VERSION,
    run_autointerp,
)
from v6.baselines.delphi import (  # noqa: E402
    DELPHI_DETECT_PROMPT_VERSION,
    DELPHI_EXPLAIN_PROMPT_VERSION,
    run_delphi,
)


METHODS = ("autointerp", "delphi")
SOURCE_FILES = (
    "v6/baselines/runners/llm_interpretability_task1.py",
    "v6/baselines/api.py",
    "v6/baselines/protocol.py",
    "v6/baselines/_llm_interpretability.py",
    "v6/baselines/automated_interpretability.py",
    "v6/baselines/delphi.py",
)
METHOD_LABELS = {
    "autointerp": "Automated Interpretability (Bills et al.)",
    "delphi": "Delphi (contrastive, detection-scored adaptation)",
}


def _parse_methods(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return list(METHODS)
    aliases = {
        "auto": "autointerp",
        "automated-interpretability": "autointerp",
    }
    methods = [
        aliases.get(item.strip().lower(), item.strip().lower())
        for item in value.split(",")
        if item.strip()
    ]
    unknown = [method for method in methods if method not in METHODS]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown Task 1 baselines: {unknown}")
    return list(dict.fromkeys(methods))


def _parse_budget(value: str | float) -> float | None:
    if isinstance(value, str) and value.strip().lower() in {
        "none", "unlimited", "no-limit", "off",
    }:
        return None
    try:
        budget = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "budget must be a non-negative dollar amount or 'none'"
        ) from exc
    if budget < 0:
        raise argparse.ArgumentTypeError("budget cannot be negative")
    return budget


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baselines", default="all", type=_parse_methods)
    parser.add_argument("--datasets", default="report19")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--budget-usd", type=_parse_budget, default=5.0)
    parser.add_argument("--profile-top-k", type=int, default=6)
    parser.add_argument("--embed-batch-size", type=int, default=64)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--api-cache-dir", type=Path)
    parser.add_argument(
        "--defer-match",
        action="store_true",
        help="generate/cache predictions now and compute local E5 Match-ACC on resume",
    )
    parser.add_argument("--status-every", type=int, default=1)
    # Execution-only scheduling control. Seeds, prompts, samples, cache keys,
    # and scientific configuration are independent of worker count.
    parser.add_argument("--case-workers", type=int, default=1)
    args = parser.parse_args(argv)
    if args.folds != 5:
        parser.error("the frozen Task 1 protocol requires exactly five folds")
    if args.profile_top_k < 1 or args.embed_batch_size < 1:
        parser.error("profile and embedding batch sizes must be positive")
    if args.status_every < 1:
        parser.error("--status-every must be positive")
    if args.case_workers < 1:
        parser.error("--case-workers must be positive")
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


def _generation_case_path(
    root: Path, method: str, dataset: str, fold: int, observed_index: int
) -> Path:
    return (
        root / "generation_cases" / method / dataset / f"fold_{fold:02d}"
        / f"item_{observed_index:03d}.json"
    )


def _case_path(
    root: Path, method: str, dataset: str, fold: int, observed_index: int
) -> Path:
    return (
        root / "cases" / method / dataset / f"fold_{fold:02d}"
        / f"item_{observed_index:03d}.json"
    )


def _fold_metric_path(root: Path, method: str, dataset: str, fold: int) -> Path:
    return root / "fold_metrics" / method / dataset / f"fold_{fold:02d}.json"


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
            "Task 1 Match-ACC requires GRAPHSEM_ENCODER=e5-large; "
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


def _task1_match_from_embeddings(
    predicted_embeddings: np.ndarray,
    masked_indices: Sequence[int],
    all_gold_embeddings: np.ndarray,
) -> float:
    """Mirror the project's Hungarian Task 1 Match-ACC definition."""

    from scipy.optimize import linear_sum_assignment

    masked = np.asarray([int(index) for index in masked_indices], dtype=int)
    predicted = np.asarray(predicted_embeddings, dtype=np.float64)
    gold = np.asarray(all_gold_embeddings, dtype=np.float64)
    if predicted.ndim != 2 or gold.ndim != 2 or predicted.shape[1] != gold.shape[1]:
        raise ValueError("predicted and gold embeddings must be compatible 2-D arrays")
    if predicted.shape[0] != len(masked) or not len(masked):
        raise ValueError("one predicted embedding is required per masked item")
    if np.min(masked) < 0 or np.max(masked) >= len(gold):
        raise IndexError("masked item index is outside the gold embedding matrix")
    similarities = _normalise_rows(predicted) @ _normalise_rows(gold[masked]).T
    _, assignment = linear_sum_assignment(-similarities)
    return float(np.mean(assignment == np.arange(len(masked))))


def _record_generation_case(
    *,
    root: Path,
    method: str,
    dataset_name: str,
    fold: int,
    observed_index: int,
    observed_node: str,
    prediction: Mapping[str, Any],
    activation_metadata: Mapping[str, Any],
    cfg_hash: str,
) -> dict[str, Any]:
    """Freeze one prediction without accepting any masked/gold item text."""

    record = {
        "task": 1,
        "method": METHOD_LABELS[method],
        "method_id": method,
        "dataset": dataset_name,
        "fold": int(fold),
        "observed_index": int(observed_index),
        "observed_node": str(observed_node),
        "construct_name": str(prediction["construct_name"]),
        "prediction": dict(prediction),
        "activation": dict(activation_metadata),
        "config_hash": cfg_hash,
        "completed_at": _timestamp(),
        "llm_judge": None,
    }
    atomic_write_json(
        _generation_case_path(root, method, dataset_name, fold, observed_index),
        record,
    )
    return record


def _fold_specs(observed_count: int, folds: int, seed: int) -> list[tuple[int, list[int]]]:
    return [
        (fold, sorted(int(index) for index in mask))
        for fold, mask in enumerate(outer_folds(observed_count, folds, seed))
    ]


def _generate_dataset(
    *,
    root: Path,
    dataset: Mapping[str, Any],
    methods: Sequence[str],
    fold_specs: Sequence[tuple[int, Sequence[int]]],
    client: BaselineAPIClient,
    profile_top_k: int,
    seed: int,
    cfg_hash: str,
    status: dict[str, Any],
    status_every: int,
    case_workers: int = 1,
) -> None:
    dataset_name = str(dataset["name"])
    graph = dataset["graph"]
    observed = list(graph.observed)
    labels = dataset["labels"]
    X = np.asarray(dataset["X"], dtype=float)

    for fold, masked_indices in fold_specs:
        masked = sorted(int(index) for index in masked_indices)
        masked_set = set(masked)
        visible = [index for index in range(len(observed)) if index not in masked_set]
        # The renderer cannot access masked item text: it receives a mapping
        # containing only fold-visible labels.
        visible_labels = {
            observed[index]: str(labels[observed[index]]) for index in visible
        }
        summaries, response_vectors = render_profiles(
            X, observed, visible_labels, visible, top_k=profile_top_k
        )
        activations, activation_metadata = observed_activation_matrix(
            X, len(observed), masked
        )

        missing: list[tuple[str, int, int]] = []
        for position, observed_index in enumerate(masked):
            for method in methods:
                path = _generation_case_path(
                    root, method, dataset_name, fold, observed_index
                )
                if _load_record(path, cfg_hash) is not None:
                    status["resumed_cases"] += 1
                    continue
                missing.append((method, position, observed_index))

        if not missing:
            status["updated_at"] = _timestamp()
            atomic_write_json(root / "status.json", status)
            continue

        status["current"] = {
            "stage": "generation",
            "methods": sorted(set(method for method, _, _ in missing)),
            "dataset": dataset_name,
            "fold": int(fold),
            "pending_cases": len(missing),
            "case_workers": int(case_workers),
        }
        status["updated_at"] = _timestamp()
        atomic_write_json(root / "status.json", status)

        def run_one(case: tuple[str, int, int]) -> tuple[str, int]:
            method, position, observed_index = case
            case_seed = stable_seed(
                "task1", method, dataset_name, fold, observed_index,
                base_seed=seed,
            )
            if method == "autointerp":
                prediction = run_autointerp(
                    client,
                    summaries,
                    activations[:, position],
                    seed=case_seed,
                )
            else:
                prediction = run_delphi(
                    client,
                    summaries,
                    activations[:, position],
                    response_vectors,
                    seed=case_seed,
                )
            _record_generation_case(
                root=root,
                method=method,
                dataset_name=dataset_name,
                fold=fold,
                observed_index=observed_index,
                observed_node=observed[observed_index],
                prediction=prediction,
                activation_metadata=activation_metadata[position],
                cfg_hash=cfg_hash,
            )
            return method, observed_index

        with ThreadPoolExecutor(
            max_workers=case_workers, thread_name_prefix="task1-baseline-case"
        ) as pool:
            futures = [pool.submit(run_one, case) for case in missing]
            for future in as_completed(futures):
                method, observed_index = future.result()
                status["current"].update({
                    "last_completed_method": method,
                    "last_completed_observed_index": int(observed_index),
                })
                status["completed_new_cases"] += 1
                status["api_spend_usd"] = client.spent_usd
                if status["completed_new_cases"] % status_every == 0:
                    status["updated_at"] = _timestamp()
                    atomic_write_json(root / "status.json", status)


def _verify_generation_frozen(
    root: Path,
    dataset_name: str,
    methods: Sequence[str],
    fold_specs: Sequence[tuple[int, Sequence[int]]],
    cfg_hash: str,
) -> None:
    for method in methods:
        frozen_items = 0
        for fold, masked_indices in fold_specs:
            for observed_index in sorted(int(index) for index in masked_indices):
                path = _generation_case_path(
                    root, method, dataset_name, fold, observed_index
                )
                if _load_record(path, cfg_hash) is None:
                    raise RuntimeError(f"dataset generation pass is incomplete: {path}")
                frozen_items += 1
        atomic_write_json(
            root / "generation_frozen" / method / f"{dataset_name}.json",
            {
                "task": 1,
                "method": method,
                "dataset": dataset_name,
                "folds": [int(fold) for fold, _ in fold_specs],
                "frozen_items": frozen_items,
                "config_hash": cfg_hash,
                "frozen_at": _timestamp(),
            },
        )


def _evaluate_dataset(
    *,
    root: Path,
    dataset: Mapping[str, Any],
    methods: Sequence[str],
    fold_specs: Sequence[tuple[int, Sequence[int]]],
    cfg_hash: str,
    embed,
    status: dict[str, Any],
) -> None:
    dataset_name = str(dataset["name"])
    observed = list(dataset["graph"].observed)
    labels = dataset["labels"]
    _verify_generation_frozen(root, dataset_name, methods, fold_specs, cfg_hash)

    # This is the first all-gold read/encoding boundary for the dataset.
    all_gold_labels = [str(labels[name]) for name in observed]
    all_gold_embeddings = embed(all_gold_labels)
    for method in methods:
        for fold, masked_indices in fold_specs:
            masked = sorted(int(index) for index in masked_indices)
            generation_records = [
                _load_record(
                    _generation_case_path(root, method, dataset_name, fold, index),
                    cfg_hash,
                )
                for index in masked
            ]
            if any(record is None for record in generation_records):
                raise RuntimeError(
                    f"cannot evaluate incomplete {method} generation for "
                    f"{dataset_name} fold {fold}"
                )
            records = [dict(record) for record in generation_records if record]
            for record, observed_index in zip(records, masked):
                record["gold_label"] = all_gold_labels[observed_index]
                record["gold_attached_at"] = _timestamp()
                atomic_write_json(
                    _case_path(root, method, dataset_name, fold, observed_index),
                    record,
                )
            predicted = embed([str(record["construct_name"]) for record in records])
            match_acc = _task1_match_from_embeddings(
                predicted, masked, all_gold_embeddings
            )
            atomic_write_json(
                _fold_metric_path(root, method, dataset_name, fold),
                {
                    "task": 1,
                    "method": METHOD_LABELS[method],
                    "method_id": method,
                    "dataset": dataset_name,
                    "fold": int(fold),
                    "masked_observed_count": len(masked),
                    "match_acc": match_acc,
                    "llm_judge": None,
                    "config_hash": cfg_hash,
                    "evaluated_at": _timestamp(),
                },
            )
        status["current"] = {
            "stage": "evaluation",
            "method": method,
            "dataset": dataset_name,
        }
        status["updated_at"] = _timestamp()
        atomic_write_json(root / "status.json", status)


def _mean(values: Sequence[Any]) -> float | None:
    keep = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(keep)) if keep else None


def _build_summary(
    root: Path, config: Mapping[str, Any], cfg_hash: str
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "config": dict(config),
        "config_hash": cfg_hash,
        "llm_judge_run": False,
        "generated_at": _timestamp(),
        "methods": {},
    }
    for method in config["methods"]:
        generation_records: list[dict[str, Any]] = []
        generation_root = root / "generation_cases" / method
        if generation_root.is_dir():
            for path in sorted(generation_root.glob("**/*.json")):
                record = _load_record(path, cfg_hash)
                if record:
                    generation_records.append(record)
        metrics: list[dict[str, Any]] = []
        metric_root = root / "fold_metrics" / method
        if metric_root.is_dir():
            for path in sorted(metric_root.glob("**/*.json")):
                value = json.loads(path.read_text(encoding="utf-8"))
                if value.get("config_hash") == cfg_hash:
                    metrics.append(value)

        per_dataset: dict[str, Any] = {}
        for dataset_name in config["datasets"]:
            selected = [
                record for record in generation_records
                if record["dataset"] == dataset_name
            ]
            selected_metrics = [
                metric for metric in metrics if metric["dataset"] == dataset_name
            ]
            if method == "autointerp":
                native = {
                    "spearman": _mean([
                        record["prediction"].get("spearman") for record in selected
                    ]),
                    "pearson": _mean([
                        record["prediction"].get("pearson") for record in selected
                    ]),
                }
            else:
                native = {
                    "test_auroc": _mean([
                        record["prediction"].get("test_auroc") for record in selected
                    ]),
                    "test_f1": _mean([
                        record["prediction"].get("test_f1") for record in selected
                    ]),
                }
            per_dataset[dataset_name] = {
                "completed_items": len(selected),
                "completed_folds": len(selected_metrics),
                "match_acc": _mean([
                    metric.get("match_acc") for metric in selected_metrics
                ]),
                **native,
            }
        summary["methods"][method] = {
            "label": METHOD_LABELS[method],
            "completed_items": len(generation_records),
            "dataset_macro_match_acc": _mean([
                value["match_acc"] for value in per_dataset.values()
            ]),
            "datasets": per_dataset,
        }
    atomic_write_json(root / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    methods = args.baselines
    datasets = select_datasets(args.datasets)
    if len(methods) == 1:
        run_name = f"task1_full_{methods[0]}"
    else:
        run_name = "task1_full_" + "_".join(methods)
    root = (
        args.output_dir
        or V6_DIR / "outputs" / "interpretability_baselines" / run_name
    ).resolve()
    api_cache = (
        args.api_cache_dir
        or V6_DIR / "outputs" / "interpretability_baselines" / "api_cache"
    ).resolve()
    config = {
        "protocol": "task1-visible-label-folds-v1-llm-interpretability",
        "task": 1,
        "datasets": datasets,
        "report_dataset_order": list(REPORT_DATASETS),
        "folds": args.folds,
        "seed": args.seed,
        "methods": methods,
        "model": args.model,
        "budget_usd": args.budget_usd,
        "temperature": 0,
        "profile_top_k": args.profile_top_k,
        "encoder": "intfloat/e5-large-v2 (frozen base)",
        "activation": "per-masked-item respondent response column; population z-score",
        "profile_text": "fold-visible item text only",
        "prompt_versions": {
            "autointerp_explainer": AUTOINTERP_EXPLAIN_PROMPT_VERSION,
            "autointerp_simulator": AUTOINTERP_SIMULATE_PROMPT_VERSION,
            "delphi_explainer": DELPHI_EXPLAIN_PROMPT_VERSION,
            "delphi_detector": DELPHI_DETECT_PROMPT_VERSION,
        },
        "provenance_boundary": (
            "all five folds frozen in gold-free generation_cases before gold encoding"
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
                "created_at": _timestamp(),
                "api_cache_dir": str(api_cache),
                "source_sha256": source_sha256(REPO_ROOT, SOURCE_FILES),
            },
        )

    client = BaselineAPIClient(
        api_cache,
        model=args.model,
        budget_usd=args.budget_usd,
        max_attempts=8,
        retry_base_seconds=2.0,
    )
    status: dict[str, Any] = {
        "state": "running",
        "config_hash": cfg_hash,
        "started_at": _timestamp(),
        "updated_at": _timestamp(),
        "current": None,
        "completed_new_cases": 0,
        "resumed_cases": 0,
        "api_spend_usd": client.spent_usd,
        "llm_judge_run": False,
    }
    atomic_write_json(root / "status.json", status)

    try:
        loaders = report_loaders()
        loaded: list[tuple[Mapping[str, Any], list[tuple[int, list[int]]]]] = []
        for dataset_name in datasets:
            dataset = loaders[dataset_name]()
            observed = list(dataset["graph"].observed)
            X = np.asarray(dataset["X"], dtype=float)
            labels = dataset["labels"]
            if X.ndim != 2 or X.shape[1] != len(observed):
                raise ValueError(f"{dataset_name}: X columns do not match graph.observed")
            missing_labels = [name for name in observed if name not in labels]
            if missing_labels:
                raise ValueError(
                    f"{dataset_name}: labels missing observed nodes {missing_labels}"
                )
            folds = _fold_specs(len(observed), args.folds, args.seed)
            loaded.append((dataset, folds))
            _generate_dataset(
                root=root,
                dataset=dataset,
                methods=methods,
                fold_specs=folds,
                client=client,
                profile_top_k=args.profile_top_k,
                seed=args.seed,
                cfg_hash=cfg_hash,
                status=status,
                status_every=args.status_every,
                case_workers=args.case_workers,
            )

        if not args.defer_match:
            embed = _embedding_function(args.embed_batch_size)
            for dataset, folds in loaded:
                _evaluate_dataset(
                    root=root,
                    dataset=dataset,
                    methods=methods,
                    fold_specs=folds,
                    cfg_hash=cfg_hash,
                    embed=embed,
                    status=status,
                )

        summary = _build_summary(root, config, cfg_hash)
        status.update(
            {
                "state": "generation_complete" if args.defer_match else "complete",
                "current": None,
                "updated_at": _timestamp(),
                "finished_at": _timestamp(),
                "api_spend_usd": client.spent_usd,
                "summary_path": str(root / "summary.json"),
            }
        )
        atomic_write_json(root / "status.json", status)
        print(
            json.dumps(
                {
                    "status": status["state"],
                    "output": str(root),
                    "completed_new_cases": status["completed_new_cases"],
                    "resumed_cases": status["resumed_cases"],
                    "dataset_macro_match_acc": {
                        method: summary["methods"][method]["dataset_macro_match_acc"]
                        for method in methods
                    },
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 0
    except Exception as exc:
        status.update(
            {
                "state": "failed",
                "updated_at": _timestamp(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "api_spend_usd": client.spent_usd,
            }
        )
        atomic_write_json(root / "status.json", status)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
