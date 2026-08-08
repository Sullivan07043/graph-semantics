"""Run the three Task 2 interpretability baselines without an LLM judge.

The executable deliberately does not import ``judge`` or ``metrics``.  It
generates frozen predictions and native fidelity metrics first, persists every
latent-fold atomically, and computes only the local E5/Hungarian Match-ACC.

Example (from the repository root)::

    python -m v6.baselines.runners.interpretability_task2 --baselines all

All API calls use the independent cached client in ``v6.baselines.api``.  A
failed process can be restarted with the same command and will skip completed
cases and reuse paid completions.
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

# Keep the local, already downloaded E5 model discoverable on Windows.
os.environ.setdefault("HF_CACHE", str(WORKSPACE_ROOT / ".hf_cache"))

from v6.baselines.api import BaselineAPIClient  # noqa: E402
from v6.baselines.protocol import (  # noqa: E402
    REPORT_DATASETS,
    atomic_write_json,
    config_hash,
    file_sha256 as _file_sha256,
    latent_activation,
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
from v6.baselines.clip_dissect_bank import (  # noqa: E402
    BANK_VERSION,
    build_wordnet_domain_bank,
)
from v6.baselines.clip_dissect_e5 import (  # noqa: E402
    SCORER_VERSION,
    ClipDissectE5Config,
    load_concept_bank,
    rank_stability,
    run_clip_dissect_e5,
)

METHODS = ("autointerp", "delphi", "text-dissect")
SOURCE_FILES = (
    "v6/baselines/runners/interpretability_task2.py",
    "v6/baselines/api.py",
    "v6/baselines/protocol.py",
    "v6/baselines/_llm_interpretability.py",
    "v6/baselines/automated_interpretability.py",
    "v6/baselines/delphi.py",
    "v6/baselines/clip_dissect_bank.py",
    "v6/baselines/clip_dissect_e5.py",
)
METHOD_LABELS = {
    "autointerp": "Automated Interpretability (simulation-scored adaptation)",
    "delphi": "Delphi (contrastive, detection-scored adaptation)",
    "text-dissect": "CLIP-Dissect (E5 text adaptation)",
}


def _parse_methods(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return list(METHODS)
    aliases = {
        "auto": "autointerp",
        "autointerp-sim": "autointerp",
        "text": "text-dissect",
        "text-dissect-e5": "text-dissect",
    }
    methods = [aliases.get(item.strip().lower(), item.strip().lower())
               for item in value.split(",") if item.strip()]
    unknown = [method for method in methods if method not in METHODS]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown baselines: {unknown}")
    return list(dict.fromkeys(methods))


def _parse_budget(value: str | float) -> float | None:
    if isinstance(value, str) and value.strip().lower() in {
        "none", "unlimited", "no-limit", "off"
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
    parser.add_argument("--text-bank-size", type=int, default=4096)
    parser.add_argument("--text-concept-batch-size", type=int, default=4096)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--api-cache-dir", type=Path)
    # Execution-only pilot controls; excluded from the scientific config hash.
    parser.add_argument("--max-latent-cases", type=int, default=0)
    parser.add_argument("--max-dataset-folds", type=int, default=0)
    parser.add_argument(
        "--defer-match", action="store_true",
        help="generate/cache predictions now; compute local E5 Match-ACC on a resume pass",
    )
    parser.add_argument("--status-every", type=int, default=1)
    args = parser.parse_args(argv)
    if args.folds != 5:
        parser.error("the frozen report protocol requires exactly five folds")
    if args.profile_top_k < 1 or args.embed_batch_size < 1:
        parser.error("batch/profile sizes must be positive")
    if args.text_bank_size < 1 or args.text_concept_batch_size < 1:
        parser.error(
            "CLIP-Dissect E5 text-adaptation bank and chunk sizes must be positive"
        )
    if args.max_latent_cases < 0 or args.max_dataset_folds < 0:
        parser.error("execution limits cannot be negative")
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


def _case_path(root: Path, method: str, dataset: str, fold: int, latent_index: int) -> Path:
    return root / "cases" / method / dataset / f"fold_{fold:02d}" / f"latent_{latent_index:03d}.json"


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


def _latent_names(dataset: Mapping[str, Any]) -> list[str]:
    graph = dataset["graph"]
    names = [name for name in graph.latents if name in dataset["latent_gt"]]
    if len(names) != len(graph.latents):
        missing = [name for name in graph.latents if name not in dataset["latent_gt"]]
        raise ValueError(f"latent_gt does not cover graph latents: {missing}")
    return names


def _embedding_function(batch_size: int):
    import encode

    def embed(texts: Sequence[str]) -> np.ndarray:
        return np.asarray(encode.embed(list(texts), batch_size=batch_size), dtype=np.float64)

    return embed


def _latent_match(predicted: Sequence[str], gold: Sequence[str], embed) -> float | None:
    if len(predicted) != len(gold):
        raise ValueError("predicted and gold latent lists differ in length")
    if len(predicted) <= 1:
        return None
    from scipy.optimize import linear_sum_assignment

    pred = embed(predicted)
    target = embed(gold)
    pred /= np.linalg.norm(pred, axis=1, keepdims=True) + 1e-9
    target /= np.linalg.norm(target, axis=1, keepdims=True) + 1e-9
    _, assignment = linear_sum_assignment(-(pred @ target.T))
    return float(np.mean(assignment == np.arange(len(predicted))))


def _write_status(root: Path, status: Mapping[str, Any]) -> None:
    atomic_write_json(root / "status.json", dict(status))


def _record_case(
    *,
    root: Path,
    method: str,
    dataset: Mapping[str, Any],
    fold: int,
    latent_index: int,
    latent_name: str,
    prediction: Mapping[str, Any],
    activation_metadata: Mapping[str, Any],
    cfg_hash: str,
) -> dict[str, Any]:
    record = {
        "task": 2,
        "method": METHOD_LABELS[method],
        "method_id": method,
        "dataset": dataset["name"],
        "fold": fold,
        "latent_index": latent_index,
        # Stored only after generation; never passed to a prompt.
        "latent_node": latent_name,
        "construct_name": str(prediction["construct_name"]),
        "prediction": dict(prediction),
        "activation": dict(activation_metadata),
        "config_hash": cfg_hash,
        "completed_at": _timestamp(),
        "llm_judge": None,
    }
    atomic_write_json(
        _case_path(root, method, dataset["name"], fold, latent_index), record
    )
    return record


def _evaluate_completed_fold(
    root: Path,
    method: str,
    dataset: Mapping[str, Any],
    fold: int,
    cfg_hash: str,
    embed,
) -> dict[str, Any] | None:
    latent_names = _latent_names(dataset)
    records = [
        _load_record(_case_path(root, method, dataset["name"], fold, index), cfg_hash)
        for index in range(len(latent_names))
    ]
    if any(record is None for record in records):
        return None
    predictions = [str(record["construct_name"]) for record in records if record]
    gold = [str(dataset["latent_gt"][name]) for name in latent_names]
    metric = {
        "method": METHOD_LABELS[method],
        "dataset": dataset["name"],
        "fold": fold,
        "latent_count": len(latent_names),
        "latent_match_acc": _latent_match(predictions, gold, embed),
        "llm_judge": None,
        "config_hash": cfg_hash,
        "evaluated_at": _timestamp(),
    }
    atomic_write_json(_fold_metric_path(root, method, dataset["name"], fold), metric)
    return metric


def _mean(values: Sequence[Any]) -> float | None:
    keep = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(keep)) if keep else None


def _build_summary(root: Path, config: Mapping[str, Any], cfg_hash: str) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "config": dict(config),
        "config_hash": cfg_hash,
        "llm_judge_run": False,
        "generated_at": _timestamp(),
        "methods": {},
    }
    for method in config["methods"]:
        records = []
        for path in sorted((root / "cases" / method).glob("**/*.json")):
            record = _load_record(path, cfg_hash)
            if record:
                records.append(record)
        folds = []
        metric_root = root / "fold_metrics" / method
        if metric_root.is_dir():
            for path in sorted(metric_root.glob("**/*.json")):
                value = json.loads(path.read_text(encoding="utf-8"))
                if value.get("config_hash") == cfg_hash:
                    folds.append(value)
        per_dataset: dict[str, Any] = {}
        for dataset in config["datasets"]:
            selected = [record for record in records if record["dataset"] == dataset]
            selected_folds = [metric for metric in folds if metric["dataset"] == dataset]
            native: dict[str, Any] = {}
            if method == "autointerp":
                native = {
                    "spearman": _mean([record["prediction"].get("spearman") for record in selected]),
                    "pearson": _mean([record["prediction"].get("pearson") for record in selected]),
                }
            elif method == "delphi":
                native = {
                    "test_auroc": _mean([record["prediction"].get("test_auroc") for record in selected]),
                    "test_f1": _mean([record["prediction"].get("test_f1") for record in selected]),
                }
            elif method == "text-dissect":
                by_latent: dict[int, list[Mapping[str, Any]]] = {}
                for record in selected:
                    by_latent.setdefault(int(record["latent_index"]), []).append(record["prediction"])
                stability = [rank_stability(values) for values in by_latent.values() if len(values) > 1]
                native = {
                    "positive_soft_wpmi": _mean([
                        record["prediction"]["native_diagnostics"].get("positive_soft_wpmi")
                        for record in selected
                    ]),
                    "rank_stability": _mean(stability),
                }
            per_dataset[dataset] = {
                "completed_latent_folds": len(selected),
                "latent_match_acc": _mean([
                    metric.get("latent_match_acc") for metric in selected_folds
                ]),
                **native,
            }
        summary["methods"][method] = {
            "label": METHOD_LABELS[method],
            "completed_latent_folds": len(records),
            "dataset_macro_match_acc": _mean([
                value["latent_match_acc"] for value in per_dataset.values()
            ]),
            "datasets": per_dataset,
        }
    atomic_write_json(root / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    methods = args.baselines
    datasets = select_datasets(args.datasets)
    text_config = ClipDissectE5Config(
        concept_batch_size=args.text_concept_batch_size
    )
    embed = None
    text_bank = None
    text_bank_path: Path | None = None
    text_bank_sha256: str | None = None
    if "text-dissect" in methods:
        embed = _embedding_function(args.embed_batch_size)
        text_bank_path = V6_DIR / "outputs" / "interpretability_baselines" / (
            f"text_dissect_e5_domain_{args.text_bank_size}.npz"
        )
        build_wordnet_domain_bank(
            WORKSPACE_ROOT / ".nltk_data" / "corpora" / "wordnet.zip",
            text_bank_path,
            embed,
            size=args.text_bank_size,
        )
        text_bank = load_concept_bank(
            text_bank_path, expected_encoder="e5-large-v2"
        )
        text_bank_sha256 = _file_sha256(text_bank_path)
    if methods == ["text-dissect"]:
        run_name = f"task2_text_dissect_e5_report19_seed{args.seed}_v3"
    else:
        run_name = f"gpt-4o-mini_report19_seed{args.seed}"
        if "text-dissect" in methods:
            run_name += "_text_dissect_v3"
    root = (args.output_dir or
            V6_DIR / "outputs" / "interpretability_baselines" / run_name).resolve()
    api_cache = (args.api_cache_dir or
                 V6_DIR / "outputs" / "interpretability_baselines" / "api_cache").resolve()
    config = {
        "protocol": (
            "interpretability-baselines-report19-v2-text-dissect-e5-v3"
            if "text-dissect" in methods
            else "interpretability-baselines-report19-v1"
        ),
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
        "text_bank": {
            "selection_version": BANK_VERSION,
            "size": args.text_bank_size,
            "concept_batch_size": args.text_concept_batch_size,
        },
        "prompt_versions": {
            "autointerp_explainer": AUTOINTERP_EXPLAIN_PROMPT_VERSION,
            "autointerp_simulator": AUTOINTERP_SIMULATE_PROMPT_VERSION,
            "delphi_explainer": DELPHI_EXPLAIN_PROMPT_VERSION,
            "delphi_detector": DELPHI_DETECT_PROMPT_VERSION,
        },
        "llm_judge_run": False,
    }
    if "text-dissect" in methods:
        config["text_bank"]["sha256"] = text_bank_sha256
        config["text_dissect_scorer"] = {
            "version": SCORER_VERSION,
            "parameters": asdict(text_config),
            "tie_policy": "exact-score midranks; lexical identity presentation tie-break",
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
        if (
            "text-dissect" in methods
            and manifest.get("concept_bank_sha256") != text_bank_sha256
        ):
            raise RuntimeError(
                f"output directory has a different concept bank: {root}; "
                "choose another --output-dir"
            )
    else:
        manifest = {
            "config": config,
            "config_hash": cfg_hash,
            "git_commit": _git_commit(),
            "created_at": _timestamp(),
            "api_cache_dir": str(api_cache),
            "source_sha256": source_sha256(REPO_ROOT, SOURCE_FILES),
        }
        if "text-dissect" in methods:
            manifest.update({
                "concept_bank_path": str(text_bank_path.resolve()),
                "concept_bank_sha256": text_bank_sha256,
            })
        atomic_write_json(manifest_path, manifest)

    client = None
    if any(method in {"autointerp", "delphi"} for method in methods):
        client = BaselineAPIClient(
            api_cache,
            model=args.model,
            budget_usd=args.budget_usd,
            max_attempts=8,
            retry_base_seconds=2.0,
        )
    loaders = report_loaders()
    status = {
        "state": "running",
        "config_hash": cfg_hash,
        "started_at": _timestamp(),
        "updated_at": _timestamp(),
        "current": None,
        "completed_new_cases": 0,
        "resumed_cases": 0,
        "api_spend_usd": client.spent_usd if client else 0.0,
        "llm_judge_run": False,
    }
    _write_status(root, status)
    unique_llm_cases = 0
    dataset_folds_seen = 0

    try:
        for dataset_name in datasets:
            dataset = loaders[dataset_name]()
            graph, X, labels = dataset["graph"], np.asarray(dataset["X"]), dataset["labels"]
            observed = list(graph.observed)
            if X.ndim != 2 or X.shape[1] != len(observed):
                raise ValueError(f"{dataset_name}: X columns do not match graph.observed")
            latent_names = _latent_names(dataset)
            masks = outer_folds(len(observed), args.folds, args.seed)
            for fold, mask in enumerate(masks):
                if args.max_dataset_folds and dataset_folds_seen >= args.max_dataset_folds:
                    break
                dataset_folds_seen += 1
                masked = set(int(index) for index in mask)
                visible_indices = [index for index in range(len(observed)) if index not in masked]
                visible_names = [observed[index] for index in visible_indices]
                profiles, response_vectors = render_profiles(
                    X, observed, labels, visible_indices, top_k=args.profile_top_k
                )
                activations: list[np.ndarray] = []
                activation_metadata: list[dict[str, Any]] = []
                for latent_name in latent_names:
                    values, metadata = latent_activation(
                        X, graph, latent_name, visible_names
                    )
                    activations.append(values)
                    activation_metadata.append(metadata)

                if "text-dissect" in methods:
                    if embed is None:
                        embed = _embedding_function(args.embed_batch_size)
                    if text_bank is None:
                        selected_path = V6_DIR / "outputs" / "interpretability_baselines" / (
                            f"text_dissect_e5_domain_{args.text_bank_size}.npz"
                        )
                        build_wordnet_domain_bank(
                            WORKSPACE_ROOT / ".nltk_data" / "corpora" / "wordnet.zip",
                            selected_path,
                            embed,
                            size=args.text_bank_size,
                        )
                        text_bank = load_concept_bank(
                            selected_path, expected_encoder="e5-large-v2"
                        )
                    missing_text = [
                        index for index in range(len(latent_names))
                        if _load_record(
                            _case_path(root, "text-dissect", dataset_name, fold, index), cfg_hash
                        ) is None
                    ]
                    if missing_text:
                        status["current"] = {
                            "method": "text-dissect", "dataset": dataset_name, "fold": fold
                        }
                        status["updated_at"] = _timestamp()
                        _write_status(root, status)
                        profile_embeddings = embed(profiles)
                        opaque_ids = [f"latent_{index:03d}" for index in range(len(latent_names))]
                        text_results = run_clip_dissect_e5(
                            None,
                            np.column_stack(activations),
                            text_bank,
                            profile_embeddings=profile_embeddings,
                            latent_ids=opaque_ids,
                            config=text_config,
                        )
                        for index in missing_text:
                            _record_case(
                                root=root, method="text-dissect", dataset=dataset,
                                fold=fold, latent_index=index, latent_name=latent_names[index],
                                prediction=text_results[index].to_dict(),
                                activation_metadata=activation_metadata[index],
                                cfg_hash=cfg_hash,
                            )
                            status["completed_new_cases"] += 1
                    else:
                        status["resumed_cases"] += len(latent_names)

                for latent_index, latent_name in enumerate(latent_names):
                    if args.max_latent_cases and unique_llm_cases >= args.max_latent_cases:
                        break
                    for method in [value for value in methods if value in {"autointerp", "delphi"}]:
                        path = _case_path(root, method, dataset_name, fold, latent_index)
                        if _load_record(path, cfg_hash) is not None:
                            status["resumed_cases"] += 1
                            continue
                        status["current"] = {
                            "method": method, "dataset": dataset_name, "fold": fold,
                            "latent_index": latent_index,
                        }
                        status["updated_at"] = _timestamp()
                        _write_status(root, status)
                        case_seed = stable_seed(
                            method, dataset_name, fold, latent_index, base_seed=args.seed
                        )
                        if method == "autointerp":
                            prediction = run_autointerp(
                                client, profiles, activations[latent_index], seed=case_seed
                            )
                        else:
                            prediction = run_delphi(
                                client, profiles, activations[latent_index], response_vectors,
                                seed=case_seed,
                            )
                        _record_case(
                            root=root, method=method, dataset=dataset, fold=fold,
                            latent_index=latent_index, latent_name=latent_name,
                            prediction=prediction,
                            activation_metadata=activation_metadata[latent_index],
                            cfg_hash=cfg_hash,
                        )
                        status["completed_new_cases"] += 1
                        status["api_spend_usd"] = client.spent_usd
                    if any(value in {"autointerp", "delphi"} for value in methods):
                        unique_llm_cases += 1

                # Gold text enters only this post-prediction, local evaluation block.
                # API-only jobs may defer it to avoid competing with the local
                # CLIP-Dissect E5 adaptation GPU process; rerunning without
                # --defer-match resumes.
                # every cached prediction and fills these metrics locally.
                if not args.defer_match:
                    if embed is None:
                        embed = _embedding_function(args.embed_batch_size)
                    for method in methods:
                        _evaluate_completed_fold(
                            root, method, dataset, fold, cfg_hash, embed
                        )
                status["updated_at"] = _timestamp()
                status["api_spend_usd"] = client.spent_usd if client else 0.0
                _write_status(root, status)

            if args.max_dataset_folds and dataset_folds_seen >= args.max_dataset_folds:
                break
            if args.max_latent_cases and unique_llm_cases >= args.max_latent_cases:
                break

        summary = _build_summary(root, config, cfg_hash)
        status.update({
            "state": (
                "pilot_complete" if (args.max_latent_cases or args.max_dataset_folds)
                else "generation_complete_match_deferred" if args.defer_match
                else "complete"
            ),
            "current": None,
            "updated_at": _timestamp(),
            "finished_at": _timestamp(),
            "api_spend_usd": client.spent_usd if client else 0.0,
            "summary_path": str(root / "summary.json"),
        })
        _write_status(root, status)
        print(json.dumps({
            "status": status["state"], "output": str(root),
            "api_spend_usd": status["api_spend_usd"],
            "completed_new_cases": status["completed_new_cases"],
            "methods": list(summary["methods"]),
        }, ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:
        status.update({
            "state": "failed",
            "updated_at": _timestamp(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "api_spend_usd": client.spent_usd if client else 0.0,
        })
        _write_status(root, status)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
