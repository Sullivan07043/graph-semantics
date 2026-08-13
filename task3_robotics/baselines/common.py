"""Leakage-safe robot Task 1 data and profile adapters.

The numerical and graph construction mirrors ``task3_pipeline_v1/pool_ext.py``.
It lives outside the frozen Task 3 fork so external baselines can reuse the
canonical implementations under ``v6.baselines`` without importing the copied
questionnaire pipeline modules.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from v6 import graph as graph_module


ROBOT_DATASETS = ("liftbody", "bodysawyer", "bodyiiwa", "bodyur5e")
DEV_ROBOTS = ("liftbody", "bodysawyer", "bodyiiwa")
HELDOUT_ROBOTS = ("bodyur5e",)


@dataclass(frozen=True)
class RobotDatasetSpec:
    steps_file: str
    graph_file: str
    rows_per_episode: int
    selected_row: int


SPECS = {
    "liftbody": RobotDatasetSpec(
        "lift_body_steps.npz", "lift_body_summary.json", 149, 70
    ),
    "bodysawyer": RobotDatasetSpec(
        "body_sawyer_steps.npz", "body_sawyer_boss_summary.json", 199, 100
    ),
    "bodyiiwa": RobotDatasetSpec(
        "body_iiwa_steps.npz", "body_iiwa_boss_summary.json", 199, 100
    ),
    "bodyur5e": RobotDatasetSpec(
        "body_ur5e_steps.npz", "body_ur5e_boss_summary.json", 199, 100
    ),
}


def select_robot_datasets(value: str | Sequence[str]) -> list[str]:
    if isinstance(value, str):
        key = value.strip().lower()
        if key in {"all", "robot4"}:
            selected = list(ROBOT_DATASETS)
        elif key == "dev":
            selected = list(DEV_ROBOTS)
        elif key in {"heldout", "test"}:
            selected = list(HELDOUT_ROBOTS)
        else:
            selected = [item.strip().lower() for item in value.split(",") if item.strip()]
    else:
        selected = [str(item).strip().lower() for item in value]
    unknown = [name for name in selected if name not in SPECS]
    if unknown:
        raise ValueError(f"unknown robot datasets: {unknown}")
    if len(selected) != len(set(selected)):
        raise ValueError("robot dataset selection contains duplicates")
    return selected


def required_artifacts(
    data_dir: Path | str, datasets: Sequence[str] = ROBOT_DATASETS
) -> list[Path]:
    root = Path(data_dir).resolve()
    paths: list[Path] = []
    for name in datasets:
        spec = SPECS[name]
        paths.extend((root / spec.steps_file, root / spec.graph_file))
    return paths


def load_robot_dataset(name: str, data_dir: Path | str) -> dict[str, Any]:
    """Load one robot exactly as the current Task 3 Task 1 protocol does."""

    if name not in SPECS:
        raise ValueError(f"unknown robot dataset: {name}")
    root = Path(data_dir).resolve()
    spec = SPECS[name]
    steps_path = root / spec.steps_file
    graph_path = root / spec.graph_file
    if not steps_path.is_file():
        raise FileNotFoundError(f"missing robot step matrix: {steps_path}")
    if not graph_path.is_file():
        raise FileNotFoundError(f"missing robot BOSS graph: {graph_path}")

    with np.load(steps_path, allow_pickle=True) as payload:
        if not {"X", "names", "labels"}.issubset(payload.files):
            raise ValueError(f"robot step matrix has incomplete schema: {steps_path}")
        values = np.asarray(payload["X"], dtype=float)
        columns = [str(value) for value in payload["names"]]
        texts = [str(value) for value in payload["labels"]]
    if values.ndim != 2 or values.shape[1] != len(columns) or len(columns) != len(texts):
        raise ValueError(f"robot step matrix arrays do not align: {steps_path}")
    if len(values) % spec.rows_per_episode:
        raise ValueError(
            f"{name}: {len(values)} transition rows are not divisible by "
            f"{spec.rows_per_episode}"
        )

    # One fixed mid-episode transition per independently generated episode.
    values = values[spec.selected_row :: spec.rows_per_episode]
    keep = [
        index
        for index, column in enumerate(columns)
        if column.endswith("@t") or column.startswith("action")
    ]
    observed = [columns[index].split("@", 1)[0] for index in keep]
    if not observed or len(observed) != len(set(observed)):
        raise ValueError(f"{name}: selected robot channel names are empty or duplicated")
    labels = {node: texts[index] for node, index in zip(observed, keep)}
    matrix = np.asarray(values[:, keep], dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name}: selected robot matrix contains a non-finite value")
    standard_deviations = matrix.std(axis=0)
    if np.any(standard_deviations <= 1e-12):
        bad = [observed[index] for index in np.flatnonzero(standard_deviations <= 1e-12)]
        raise ValueError(f"{name}: constant selected robot channels: {bad}")
    matrix = (matrix - matrix.mean(axis=0)) / (standard_deviations + 1e-9)

    graph_payload = json.loads(graph_path.read_text(encoding="utf-8"))
    raw_edges = graph_payload.get("rlcd_directed")
    if not isinstance(raw_edges, list):
        raise ValueError(f"robot graph lacks rlcd_directed: {graph_path}")
    observed_set = set(observed)
    edges = [
        (str(edge[0]), str(edge[1]))
        for edge in raw_edges
        if isinstance(edge, (list, tuple))
        and len(edge) == 2
        and str(edge[0]) in observed_set
        and str(edge[1]) in observed_set
    ]
    graph = graph_module.Graph([], observed, edges)
    edge_types = graph_payload.get("edge_types", {})
    graph.edge_type = {
        edge: (
            1.0
            if edge_types.get(f"{edge[0]}->{edge[1]}") == "contemp"
            else 0.0
        )
        for edge in edges
    }
    return {
        "name": name,
        "graph": graph,
        "X": matrix,
        "labels": labels,
        "latent_gt": {},
        "artifacts": {
            "steps": str(steps_path),
            "graph": str(graph_path),
        },
    }


def render_robot_snapshots(
    X: np.ndarray,
    observed: Sequence[str],
    labels: Mapping[str, str],
    visible_indices: Sequence[int],
    *,
    top_k: int = 6,
) -> tuple[list[str], np.ndarray]:
    """Render rows using fold-visible channel text and no masked channel names."""

    values = np.asarray(X, dtype=float)
    visible = np.asarray(sorted(int(index) for index in visible_indices), dtype=int)
    if values.ndim != 2 or values.shape[1] != len(observed):
        raise ValueError("X columns must match robot graph.observed order")
    if not len(visible):
        raise ValueError("at least one robot channel label must remain visible")
    if np.min(visible) < 0 or np.max(visible) >= len(observed):
        raise IndexError("visible robot channel index is outside graph.observed")
    visible_names = [observed[index] for index in visible]
    try:
        visible_labels = [str(labels[name]) for name in visible_names]
    except KeyError as exc:
        raise ValueError(f"missing visible robot channel label: {exc.args[0]}") from exc

    snapshots: list[str] = []
    for row in values[:, visible]:
        ascending = np.lexsort((visible, row))
        low_count = min(top_k, len(visible) // 2)
        high_count = min(top_k, len(visible) - low_count)
        low_local = list(ascending[:low_count])
        used = set(low_local)
        high_local = [
            index for index in reversed(ascending) if index not in used
        ][:high_count]
        high_lines = [
            f"- {visible_labels[index]} [standardized channel value {row[index]:+.2f}]"
            for index in high_local
        ]
        low_lines = [
            f"- {visible_labels[index]} [standardized channel value {row[index]:+.2f}]"
            for index in low_local
        ]
        snapshots.append(
            "Higher-valued visible robot channels:\n"
            + ("\n".join(high_lines) if high_lines else "- none")
            + "\nLower-valued visible robot channels:\n"
            + ("\n".join(low_lines) if low_lines else "- none")
        )
    return snapshots, np.asarray(values[:, visible], dtype=float)


__all__ = [
    "DEV_ROBOTS",
    "HELDOUT_ROBOTS",
    "ROBOT_DATASETS",
    "SPECS",
    "load_robot_dataset",
    "render_robot_snapshots",
    "required_artifacts",
    "select_robot_datasets",
]
