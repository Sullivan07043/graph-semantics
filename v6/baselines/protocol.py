"""Shared, leakage-safe protocol for the Task 1/2 interpretability baselines.

This module deliberately has no dependency on :mod:`judge`.  It fixes the
legacy 19-dataset reporting set, reproduces the existing five-fold item-label
mask, and turns respondent rows into anonymised fold-visible profiles.  Gold
latent text is not accepted by any generation helper in this module.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


# The order is frozen to the report, rather than run_task1.select_datasets("all")
# which also includes newer external-evaluation datasets.
REPORT_DATASETS = (
    "tlvd", "himi", "bigfive", "hs", "rse", "mach", "gcbs",
    "sixteenpf", "hsq", "sd3", "cfcs", "npas", "scs", "tma",
    "darktriad", "wpi", "hexaco", "riasec", "kims",
)


def report_loaders() -> dict[str, Any]:
    """Return loaders for exactly the 19 datasets in the frozen report."""

    import pool
    import testbeds

    loaders = {**testbeds.LOADERS, **pool.LOADERS}
    missing = [name for name in REPORT_DATASETS if name not in loaders]
    if missing:
        raise RuntimeError(f"missing report dataset loaders: {missing}")
    return {name: loaders[name] for name in REPORT_DATASETS}


def select_datasets(which: str | Sequence[str]) -> list[str]:
    """Resolve ``report19`` or an explicit subset without admitting extras."""

    if isinstance(which, str):
        names = list(REPORT_DATASETS) if which in {"report19", "all"} else [
            value.strip() for value in which.split(",") if value.strip()
        ]
    else:
        names = [str(value) for value in which]
    unknown = [name for name in names if name not in REPORT_DATASETS]
    if unknown:
        raise ValueError(
            f"datasets outside the frozen 19-dataset report are not allowed: {unknown}"
        )
    return names


def outer_folds(n_observed: int, folds: int = 5, seed: int = 0) -> list[np.ndarray]:
    """Exactly reproduce the observed-label mask used by run_task1/run_task2."""

    if n_observed < 1 or folds < 2:
        raise ValueError("n_observed must be positive and folds must be at least two")
    permutation = np.random.default_rng(seed).permutation(n_observed)
    return [np.asarray(permutation[index::folds], dtype=int) for index in range(folds)]


def stable_seed(*parts: object, base_seed: int = 0) -> int:
    """Stable cross-process seed; Python's salted ``hash`` is not reproducible."""

    material = json.dumps(
        [base_seed, *[str(part) for part in parts]],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def file_sha256(path: os.PathLike[str] | str, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest for a source or artifact file."""

    digest = hashlib.sha256()
    with open(os.fspath(path), "rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def source_sha256(
    repository: os.PathLike[str] | str,
    relative_paths: Sequence[os.PathLike[str] | str],
) -> dict[str, str]:
    """Hash canonical implementation files using portable repository paths."""

    root = Path(repository)
    result: dict[str, str] = {}
    for value in relative_paths:
        relative = Path(value)
        key = relative.as_posix()
        if key in result:
            raise ValueError(f"duplicate source path: {key}")
        result[key] = file_sha256(root / relative)
    return result


def latent_activation(
    X: np.ndarray,
    graph: Any,
    latent: str,
    visible_observed: Iterable[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    """PC1 activation with its sign anchored by a fold-visible loading.

    PC1 is fit to every numeric observed descendant because the protocol masks
    item *text*, not data.  The largest-absolute loading among visible
    descendants is made positive.  This fixes the arbitrary SVD sign without
    exposing a hidden label or any gold latent description.
    """

    X = np.asarray(X, dtype=float)
    observed = list(graph.observed)
    obs_index = {name: index for index, name in enumerate(observed)}
    descendants = sorted(graph.observed_descendants(latent), key=obs_index.__getitem__)
    if not descendants:
        raise ValueError(f"latent {latent!r} has no observed descendants")
    columns = [obs_index[name] for name in descendants]
    block = np.array(X[:, columns], dtype=float, copy=True)
    block -= block.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(block, full_matrices=False)
    loadings = np.asarray(vt[0], dtype=float)
    activation = block @ loadings

    visible = set(visible_observed)
    anchors = [index for index, name in enumerate(descendants) if name in visible]
    if not anchors:
        raise ValueError(f"latent {latent!r} has no fold-visible observed descendant")
    anchor = min(anchors, key=lambda index: (-abs(loadings[index]), index))
    if loadings[anchor] < 0:
        loadings = -loadings
        activation = -activation
    activation = (activation - activation.mean()) / (activation.std() + 1e-9)
    return activation, {
        "descendant_count": len(descendants),
        "visible_descendant_count": len(anchors),
        "orientation_anchor_observed_index": int(columns[anchor]),
        "orientation_anchor_loading": float(loadings[anchor]),
    }


def render_profiles(
    X: np.ndarray,
    observed: Sequence[str],
    labels: Mapping[str, str],
    visible_indices: Sequence[int],
    *,
    top_k: int = 6,
) -> tuple[list[str], np.ndarray]:
    """Render every respondent using only globally fold-visible item text.

    The Automated Interpretability adaptation receives an already discovered
    numerical feature, not the graph
    support used to obtain it.  Therefore profiles use all visible observed
    items; limiting them to a latent's descendants would leak structure into
    the structure-free baseline.  High and low item sets never overlap.
    """

    X = np.asarray(X, dtype=float)
    visible = np.asarray(sorted(int(index) for index in visible_indices), dtype=int)
    if len(visible) == 0:
        raise ValueError("at least one observed label must remain visible")
    if X.ndim != 2 or X.shape[1] != len(observed):
        raise ValueError("X columns must match graph.observed order")
    visible_names = [observed[index] for index in visible]
    try:
        visible_labels = [str(labels[name]) for name in visible_names]
    except KeyError as exc:
        raise ValueError(f"missing observed label: {exc.args[0]}") from exc

    profiles: list[str] = []
    for row in X[:, visible]:
        # lexsort makes ties deterministic in graph.observed order.
        ascending = np.lexsort((visible, row))
        low_count = min(top_k, len(visible) // 2)
        high_count = min(top_k, len(visible) - low_count)
        low_local = list(ascending[:low_count])
        used = set(low_local)
        high_local = [index for index in reversed(ascending) if index not in used][
            :high_count
        ]
        high_lines = [
            f"- {visible_labels[index]} [standardized response {row[index]:+.2f}]"
            for index in high_local
        ]
        low_lines = [
            f"- {visible_labels[index]} [standardized response {row[index]:+.2f}]"
            for index in low_local
        ]
        profiles.append(
            "Higher-response items:\n"
            + ("\n".join(high_lines) if high_lines else "- none")
            + "\nLower-response items:\n"
            + ("\n".join(low_lines) if low_lines else "- none")
        )
    return profiles, np.asarray(X[:, visible], dtype=float)


def observed_activation_matrix(
    X: np.ndarray,
    observed_count: int,
    masked_indices: Sequence[int],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Return standardized response activations for masked Task 1 items.

    Task 1 masks item *text*, not respondent data, so the target item's numeric
    response column is the feature being interpreted.  This helper deliberately
    accepts no labels: callers cannot accidentally feed masked/gold text into
    activation construction.  Columns retain ``masked_indices`` order.
    """

    values = np.asarray(X, dtype=float)
    if values.ndim != 2 or values.shape[1] != int(observed_count):
        raise ValueError("X columns must match graph.observed order")
    indices = [int(index) for index in masked_indices]
    if not indices:
        raise ValueError("at least one Task 1 item must be masked")
    if len(set(indices)) != len(indices):
        raise ValueError("masked Task 1 item indices must be unique")
    if min(indices) < 0 or max(indices) >= observed_count:
        raise IndexError("masked Task 1 item index is outside graph.observed")

    block = np.asarray(values[:, indices], dtype=float)
    means = block.mean(axis=0)
    standard_deviations = block.std(axis=0)
    constant = np.flatnonzero(standard_deviations <= 1e-12)
    if len(constant):
        bad = [indices[int(position)] for position in constant]
        raise ValueError(f"constant observed response columns cannot be interpreted: {bad}")
    standardized = (block - means[None, :]) / standard_deviations[None, :]
    metadata = [
        {
            "source": "masked_observed_response_column",
            "observed_index": index,
            "mean_before_standardization": float(means[position]),
            "std_before_standardization": float(standard_deviations[position]),
            "mean_after_standardization": float(standardized[:, position].mean()),
            "std_after_standardization": float(standardized[:, position].std()),
        }
        for position, index in enumerate(indices)
    ]
    return standardized, metadata


def atomic_write_json(path: os.PathLike[str] | str, value: Mapping[str, Any]) -> None:
    """Atomically persist a resumable case or summary record."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # On Windows, a concurrent read of ``target`` can briefly prevent an
        # otherwise valid atomic replacement (``WinError 5``).  Status files
        # are intentionally polled while long runs are active, so retry the
        # replace for a short bounded interval instead of aborting a resumable
        # experiment after all expensive work for the case has completed.
        for attempt in range(12):
            try:
                os.replace(temporary, target)
                break
            except PermissionError:
                if attempt == 11:
                    raise
                time.sleep(min(0.01 * (2 ** attempt), 0.25))
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def config_hash(config: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(config), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "REPORT_DATASETS",
    "atomic_write_json",
    "config_hash",
    "file_sha256",
    "latent_activation",
    "observed_activation_matrix",
    "outer_folds",
    "render_profiles",
    "report_loaders",
    "select_datasets",
    "source_sha256",
    "stable_seed",
]
