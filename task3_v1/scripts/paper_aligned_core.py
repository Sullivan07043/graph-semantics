"""Shared utilities for the paper-aligned Qwen J-space experiment.

The coordinate swap in this module is intentionally separate from Task 3's
ridge-dual writer.  It implements the pairwise pseudoinverse construction from
the Anthropic paper:

    V = [v_source, v_target]
    c = pinv(V) h
    h_patched = h + V (swap(c) - c)
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from collections import defaultdict
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch


UPSTREAM_COMMIT = "581d398613e5602a5af361e1c34d3a92ea82ba8e"
EXPERIMENT_LABEL = "a paper-aligned J-space replication on Qwen3.5-4B"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(path: Path) -> dict[str, Any]:
    """Load YAML while keeping the dependency optional for unit-test imports."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment diagnostic
        raise RuntimeError("PyYAML is required to read the experiment config") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return data


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
            )


def resolve_repo_path(root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    candidate = Path(value)
    if not candidate.is_absolute():
        # V1 configs are frozen with their original task3/... paths.
        # Resolve those paths to the versioned directory without editing the
        # frozen configuration content or hashes.
        parts = candidate.parts
        if parts and parts[0] == "task3":
            candidate = Path("task3_v1", *parts[1:])
        candidate = root / candidate
    return candidate.resolve()


def _normalized_surface(value: str) -> str:
    return " ".join(value.strip().casefold().split())


def single_token_representation(tokenizer, value: str) -> dict[str, Any]:
    """Find the standard word-boundary token used for a J-lens concept."""
    tried: list[dict[str, Any]] = []
    surfaces = []
    for candidate in (" " + value.strip(), value.strip()):
        if candidate not in surfaces:
            surfaces.append(candidate)
    expected = _normalized_surface(value)
    for surface in surfaces:
        ids = tokenizer.encode(surface, add_special_tokens=False)
        decoded = tokenizer.decode(ids) if ids else ""
        tried.append({"surface": surface, "token_ids": list(map(int, ids))})
        if len(ids) == 1 and _normalized_surface(decoded) == expected:
            return {
                "valid": True,
                "surface": surface,
                "token_id": int(ids[0]),
                "token_ids": [int(ids[0])],
                "decoded": decoded,
                "tried": tried,
            }
    return {
        "valid": False,
        "surface": None,
        "token_id": None,
        "token_ids": tried[0]["token_ids"] if tried else [],
        "decoded": None,
        "tried": tried,
    }


def answer_token_representation(tokenizer, prompt: str, answer: str) -> dict[str, Any]:
    """Resolve the one-token continuation without rewriting the official prompt."""
    separator = "" if prompt.endswith((" ", "\n", "\t")) else " "
    continuation = separator + answer.strip()
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    combined_ids = tokenizer.encode(
        prompt + continuation, add_special_tokens=False
    )
    suffix: list[int] | None = None
    if combined_ids[: len(prompt_ids)] == prompt_ids:
        suffix = list(map(int, combined_ids[len(prompt_ids) :]))
    standalone = list(
        map(int, tokenizer.encode(continuation, add_special_tokens=False))
    )
    candidates = []
    if suffix:
        candidates.append(("combined_suffix", suffix))
    if standalone not in [ids for _, ids in candidates]:
        candidates.append(("standalone_continuation", standalone))
    expected = _normalized_surface(answer)
    for method, ids in candidates:
        decoded = tokenizer.decode(ids) if ids else ""
        if len(ids) == 1 and _normalized_surface(decoded) == expected:
            return {
                "valid": True,
                "surface": continuation,
                "token_id": int(ids[0]),
                "token_ids": ids,
                "decoded": decoded,
                "method": method,
            }
    fallback = single_token_representation(tokenizer, answer)
    if fallback["valid"]:
        return {
            "valid": True,
            "surface": fallback["surface"],
            "token_id": fallback["token_id"],
            "token_ids": fallback["token_ids"],
            "decoded": fallback["decoded"],
            "method": "word_boundary_fallback",
        }
    chosen = candidates[0] if candidates else ("none", [])
    return {
        "valid": False,
        "surface": continuation,
        "token_id": None,
        "token_ids": chosen[1],
        "decoded": tokenizer.decode(chosen[1]) if chosen[1] else "",
        "method": chosen[0],
    }


def token_subsequence_present(sequence: Sequence[int], query: Sequence[int]) -> bool:
    if not query or len(query) > len(sequence):
        return False
    width = len(query)
    return any(list(sequence[index : index + width]) == list(query) for index in range(len(sequence) - width + 1))


def assign_group_splits(
    rows: Sequence[dict[str, Any]],
    *,
    group_key: str,
    calibration_fraction: float,
    seed: int,
) -> dict[str, str]:
    """Assign whole groups, targeting a fraction of examples rather than groups."""
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be between zero and one")
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row[group_key])] += 1
    groups = sorted(
        counts,
        key=lambda group: hashlib.sha256(
            f"{seed}:{group}".encode("utf-8")
        ).hexdigest(),
    )
    target = round(sum(counts.values()) * calibration_fraction)
    selected: set[str] = set()
    current = 0
    for group in groups:
        add_distance = abs((current + counts[group]) - target)
        keep_distance = abs(current - target)
        if add_distance < keep_distance or not selected:
            selected.add(group)
            current += counts[group]
    if len(selected) == len(groups) and len(groups) > 1:
        selected.remove(groups[-1])
    return {
        group: ("calibration" if group in selected else "heldout")
        for group in groups
    }


def token_rank(logits: torch.Tensor, token_id: int) -> int:
    if logits.ndim != 1:
        raise ValueError("token_rank expects one vocabulary-logit vector")
    target = logits[int(token_id)]
    return int(torch.count_nonzero(logits > target).item()) + 1


def normalized_depth(layer: int, max_fitted_layer: int) -> float:
    if max_fitted_layer <= 0:
        return 0.0
    return 100.0 * float(layer) / float(max_fitted_layer)


def coordinate_swap(
    hidden: torch.Tensor,
    source_vector: torch.Tensor,
    target_vector: torch.Tensor,
    *,
    rcond: float | None = None,
    strength: float = 1.0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Swap two pairwise pseudoinverse coordinates for arbitrary leading shape."""
    if hidden.ndim < 1:
        raise ValueError("hidden must have at least one dimension")
    if source_vector.shape != target_vector.shape:
        raise ValueError("source and target vectors must have the same shape")
    if hidden.shape[-1] != source_vector.numel():
        raise ValueError("vector width must equal hidden's final dimension")
    if not torch.is_floating_point(hidden):
        raise TypeError("hidden must be floating point")

    original_dtype = hidden.dtype
    device = hidden.device
    work_dtype = torch.float64 if original_dtype == torch.float64 else torch.float32
    flat = hidden.reshape(-1, hidden.shape[-1]).to(dtype=work_dtype)
    source = source_vector.reshape(-1).to(device=device, dtype=work_dtype)
    target = target_vector.reshape(-1).to(device=device, dtype=work_dtype)
    matrix = torch.stack((source, target), dim=1)
    if rcond is None:
        inverse = torch.linalg.pinv(matrix)
    else:
        inverse = torch.linalg.pinv(matrix, rcond=float(rcond))

    coordinates = flat @ inverse.T
    swapped = coordinates.flip(-1)
    delta = float(strength) * ((swapped - coordinates) @ matrix.T)
    patched_flat = flat + delta
    patched = patched_flat.reshape(hidden.shape).to(dtype=original_dtype)

    projected = (flat @ inverse.T) @ matrix.T
    patched_projected = (patched_flat @ inverse.T) @ matrix.T
    orthogonal = flat - projected
    patched_orthogonal = patched_flat - patched_projected
    achieved = patched_flat @ inverse.T
    desired = coordinates + float(strength) * (swapped - coordinates)

    condition = torch.linalg.cond(matrix)
    condition_value = float(condition.item())
    diagnostics = {
        "condition_number": (
            condition_value if math.isfinite(condition_value) else None
        ),
        "matrix_rank": int(torch.linalg.matrix_rank(matrix).item()),
        "coordinate_swap_max_abs_error": float(
            (achieved - desired).abs().max().item()
        ),
        "orthogonal_preservation_max_abs_error": float(
            (patched_orthogonal - orthogonal).abs().max().item()
        ),
        "projection_reconstruction_relative_error": float(
            (flat - projected).norm().item()
            / max(float(flat.norm().item()), 1e-12)
        ),
        "delta_norm": float(delta.norm().item()),
        "input_shape": list(hidden.shape),
        "output_shape": list(patched.shape),
        "dtype": str(original_dtype),
        "device": str(device),
    }
    return patched, diagnostics


def select_workspace_band(
    calibration_rows: Sequence[dict[str, Any]],
    candidate_layers: Sequence[int],
    width: int,
) -> dict[str, Any]:
    """Select a contiguous native-layer band using calibration J-lens MRR."""
    layers = sorted(set(map(int, candidate_layers)))
    if width < 1 or width > len(layers):
        raise ValueError("workspace band width is invalid")
    rank_by_layer: dict[int, list[int]] = defaultdict(list)
    for row in calibration_rows:
        rank_by_layer[int(row["layer"])].append(int(row["jlens_rank"]))
    if set(layers) - set(rank_by_layer):
        raise ValueError("Calibration rows are missing candidate layers")

    candidates = []
    for start in range(layers[0], layers[-1] - width + 2):
        band = list(range(start, start + width))
        if not set(band).issubset(layers):
            continue
        reciprocal_ranks = [
            1.0 / rank
            for layer in band
            for rank in rank_by_layer[layer]
        ]
        candidates.append(
            {
                "start_layer": band[0],
                "end_layer": band[-1],
                "layers": band,
                "mean_reciprocal_rank": float(
                    sum(reciprocal_ranks) / len(reciprocal_ranks)
                ),
            }
        )
    if not candidates:
        raise ValueError("No contiguous candidate band exists")
    selected = max(
        candidates,
        key=lambda item: (item["mean_reciprocal_rank"], -item["start_layer"]),
    )
    return {
        "selection_criterion": "calibration_mean_reciprocal_rank",
        "tie_break": "earliest_band",
        "band_width": width,
        "selected": selected,
        "candidates": candidates,
    }


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> dict[str, Any]:
    if total < 0 or successes < 0 or successes > total:
        raise ValueError("Invalid proportion counts")
    if total == 0:
        return {
            "successes": successes,
            "total": total,
            "rate": None,
            "ci95_low": None,
            "ci95_high": None,
        }
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return {
        "successes": successes,
        "total": total,
        "rate": p,
        "ci95_low": max(0.0, center - margin),
        "ci95_high": min(1.0, center + margin),
    }


def _trapezoid(values: Sequence[float], x: Sequence[float]) -> float:
    if len(values) != len(x) or len(values) < 2:
        return 0.0
    area = sum(
        (x[index + 1] - x[index]) * (values[index + 1] + values[index]) / 2.0
        for index in range(len(values) - 1)
    )
    span = x[-1] - x[0]
    return float(area / span) if span else 0.0


def aggregate_readout(
    rows: Sequence[dict[str, Any]],
    band_layers: Sequence[int],
    *,
    rank_key: str,
) -> dict[str, Any]:
    band_set = set(map(int, band_layers))
    by_example: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if int(row["layer"]) in band_set:
            by_example[str(row["example_id"])].append(row)
    minimum_ranks = [
        min(int(row[rank_key]) for row in example_rows)
        for example_rows in by_example.values()
    ]
    trajectories = []
    for example_rows in by_example.values():
        ordered = sorted(example_rows, key=lambda row: float(row["normalized_depth"]))
        ranks = [int(row[rank_key]) for row in ordered]
        depths = [float(row["normalized_depth"]) for row in ordered]
        vocab_size = max(int(row["vocab_size"]) for row in ordered)
        trajectories.append(
            {
                "rank_auc_normalized": _trapezoid(
                    [rank / vocab_size for rank in ranks], depths
                ),
                "reciprocal_rank_auc": _trapezoid(
                    [1.0 / rank for rank in ranks], depths
                ),
            }
        )
    ordered_min = sorted(minimum_ranks)
    median_rank = (
        float(ordered_min[len(ordered_min) // 2])
        if len(ordered_min) % 2
        else (
            float(
                (
                    ordered_min[len(ordered_min) // 2 - 1]
                    + ordered_min[len(ordered_min) // 2]
                )
                / 2.0
            )
            if ordered_min
            else None
        )
    )
    return {
        "n_examples": len(by_example),
        "top1_recovery": wilson_interval(
            sum(rank <= 1 for rank in minimum_ranks), len(minimum_ranks)
        ),
        "top5_recovery": wilson_interval(
            sum(rank <= 5 for rank in minimum_ranks), len(minimum_ranks)
        ),
        "top10_recovery": wilson_interval(
            sum(rank <= 10 for rank in minimum_ranks), len(minimum_ranks)
        ),
        "median_minimum_band_rank": median_rank,
        "mean_rank_auc_normalized": (
            sum(item["rank_auc_normalized"] for item in trajectories)
            / len(trajectories)
            if trajectories
            else None
        ),
        "mean_reciprocal_rank_auc": (
            sum(item["reciprocal_rank_auc"] for item in trajectories)
            / len(trajectories)
            if trajectories
            else None
        ),
    }


def aggregate_swap_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "n_examples": 0,
            "target_answer_top1_swap_success": wilson_interval(0, 0),
            "original_answer_top1_retention": wilson_interval(0, 0),
            "mean_delta_swap_target_log_probability": None,
            "mean_delta_log_probability_margin": None,
        }
    return {
        "n_examples": len(rows),
        "target_answer_top1_swap_success": wilson_interval(
            sum(bool(row["target_answer_top1_success"]) for row in rows), len(rows)
        ),
        "original_answer_top1_retention": wilson_interval(
            sum(bool(row["original_answer_top1_retained"]) for row in rows), len(rows)
        ),
        "mean_delta_swap_target_log_probability": float(
            sum(float(row["delta_swap_target_log_probability"]) for row in rows)
            / len(rows)
        ),
        "mean_delta_log_probability_margin": float(
            sum(float(row["delta_log_probability_margin"]) for row in rows)
            / len(rows)
        ),
    }


def git_snapshot(root: Path) -> dict[str, Any]:
    safe = f"safe.directory={root.as_posix()}"
    try:
        commit = subprocess.run(
            ["git", "-c", safe, "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-c", safe, "status", "--short"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        return {
            "commit": commit,
            "dirty": bool(status),
            "status_short": status,
        }
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"commit": None, "dirty": None, "status_error": str(exc)}


def package_versions(names: Sequence[str]) -> dict[str, str | None]:
    result: dict[str, str | None] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
    }
    for name in names:
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = None
    return result


def write_aggregate_csv(path: Path, aggregate: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []

    def add(scope: str, group: str, metric: str, value: Any) -> None:
        rows.append(
            {
                "scope": scope,
                "group": group,
                "metric": metric,
                "value": value,
            }
        )

    for lens_name, readout in aggregate.get("readout", {}).items():
        for metric, value in readout.items():
            if isinstance(value, dict):
                for child, child_value in value.items():
                    add("readout", lens_name, f"{metric}.{child}", child_value)
            else:
                add("readout", lens_name, metric, value)
    swap = aggregate.get("swap", {})
    for metric, value in swap.get("primary_band", {}).items():
        if isinstance(value, dict):
            for child, child_value in value.items():
                add("swap", "primary_band", f"{metric}.{child}", child_value)
        else:
            add("swap", "primary_band", metric, value)
    for relation, values in swap.get("per_relation_family", {}).items():
        for metric, value in values.items():
            if isinstance(value, dict):
                for child, child_value in value.items():
                    add(
                        "swap_relation",
                        relation,
                        f"{metric}.{child}",
                        child_value,
                    )
            else:
                add("swap_relation", relation, metric, value)
    for layer, values in swap.get("per_layer_diagnostic", {}).items():
        for metric, value in values.items():
            if isinstance(value, dict):
                for child, child_value in value.items():
                    add(
                        "swap_layer",
                        str(layer),
                        f"{metric}.{child}",
                        child_value,
                    )
            else:
                add("swap_layer", str(layer), metric, value)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scope", "group", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)


def exact_command() -> str:
    return " ".join([sys.executable, *sys.argv])


def finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None
