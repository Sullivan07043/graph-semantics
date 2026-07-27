"""Remove predictable same-concept cross-layer carryover from the pilot matrix.

The transform is fit only on the discovery split. Grouped cross-fitting audits
generalization, while final frozen ridge coefficients are fit on all discovery
rows and applied consistently to discovery and held-out activations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fit_ridge(
    predictors: np.ndarray,
    target: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, float]:
    """Fit ridge with standardized predictors and return raw-scale weights."""
    x_mean = predictors.mean(axis=0)
    x_std = predictors.std(axis=0)
    x_std = np.where(x_std < 1e-8, 1.0, x_std)
    x_scaled = (predictors - x_mean) / x_std
    y_mean = float(target.mean())
    gram = x_scaled.T @ x_scaled
    coefficient_scaled = np.linalg.solve(
        gram + alpha * np.eye(gram.shape[0], dtype=np.float64),
        x_scaled.T @ (target - y_mean),
    )
    coefficient = coefficient_scaled / x_std
    intercept = y_mean - float(x_mean @ coefficient)
    return coefficient, intercept


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--matrix-npz", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("task3") / "outputs" / "discovery",
    )
    parser.add_argument(
        "--output-stem", default="innovation_matrix_1000x128"
    )
    args = parser.parse_args()
    for path in (args.matrix, args.matrix_npz, args.metadata):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.ridge_alpha < 0:
        raise ValueError("--ridge-alpha must be nonnegative")

    started = time.time()
    matrix = np.load(args.matrix).astype(np.float64)
    source_arrays = np.load(args.matrix_npz)
    group_ids = source_arrays["group_ids"].astype(str)
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    columns = metadata["columns"]
    layers = sorted({int(column["layer"]) for column in columns})
    concepts = []
    for column in columns:
        if column["concept"] not in concepts:
            concepts.append(column["concept"])
    expected_shape = (len(group_ids), len(layers) * len(concepts))
    if matrix.shape != expected_shape:
        raise ValueError(f"Expected {expected_shape}, got {matrix.shape}")
    if len(set(group_ids)) < args.folds:
        raise ValueError("Not enough prompt groups for grouped cross-fitting")

    innovation = matrix.copy()
    splitter = GroupKFold(n_splits=args.folds)
    transforms = []
    oof_r2_values = []
    before_correlations = []
    after_correlations = []

    for concept_index, concept in enumerate(concepts):
        for layer_index in range(1, len(layers)):
            target_index = layer_index * len(concepts) + concept_index
            predictor_indices = np.asarray(
                [
                    earlier_layer_index * len(concepts) + concept_index
                    for earlier_layer_index in range(layer_index)
                ],
                dtype=np.int64,
            )
            predictors = matrix[:, predictor_indices]
            target = matrix[:, target_index]

            oof_prediction = np.full(len(matrix), np.nan, dtype=np.float64)
            for train_indices, test_indices in splitter.split(
                predictors, target, groups=group_ids
            ):
                fold_coefficient, fold_intercept = fit_ridge(
                    predictors[train_indices],
                    target[train_indices],
                    args.ridge_alpha,
                )
                oof_prediction[test_indices] = (
                    predictors[test_indices] @ fold_coefficient + fold_intercept
                )
            if np.isnan(oof_prediction).any():
                raise RuntimeError("Cross-fitting left rows without predictions")

            coefficient, intercept = fit_ridge(
                predictors, target, args.ridge_alpha
            )
            full_prediction = predictors @ coefficient + intercept
            innovation[:, target_index] = target - full_prediction

            centered_total = float(np.sum((target - target.mean()) ** 2))
            oof_error = float(np.sum((target - oof_prediction) ** 2))
            oof_r2 = 1.0 - oof_error / max(centered_total, 1e-12)
            before = [
                float(np.corrcoef(matrix[:, index], target)[0, 1])
                for index in predictor_indices
            ]
            after = [
                float(
                    np.corrcoef(
                        matrix[:, index], innovation[:, target_index]
                    )[0, 1]
                )
                for index in predictor_indices
            ]
            oof_r2_values.append(oof_r2)
            before_correlations.extend(abs(value) for value in before)
            after_correlations.extend(abs(value) for value in after)
            transforms.append(
                {
                    "target_index": int(target_index),
                    "target_layer": int(layers[layer_index]),
                    "concept": concept,
                    "predictor_indices": predictor_indices.tolist(),
                    "predictor_layers": [
                        int(layers[index]) for index in range(layer_index)
                    ],
                    "coefficient": coefficient.tolist(),
                    "intercept": float(intercept),
                    "grouped_oof_r2": float(oof_r2),
                    "abs_correlation_before_max": float(
                        max(abs(value) for value in before)
                    ),
                    "abs_correlation_after_max": float(
                        max(abs(value) for value in after)
                    ),
                }
            )

    innovation = innovation.astype(np.float32)
    innovation_std = innovation.std(axis=0, ddof=1)
    if not np.isfinite(innovation).all():
        raise RuntimeError("Innovation matrix contains non-finite values")
    if np.any(innovation_std < 1e-8):
        raise RuntimeError("Innovation matrix contains zero-variance columns")

    record = {
        "status": "discovery_only_same_concept_innovation_transform",
        "source_matrix": str(args.matrix),
        "source_matrix_sha256": sha256(args.matrix),
        "source_metadata": str(args.metadata),
        "source_metadata_sha256": sha256(args.metadata),
        "shape": list(innovation.shape),
        "columns": columns,
        "layers": layers,
        "concept_count": len(concepts),
        "method": (
            "For each concept and each layer after the first, regress the raw "
            "coordinate on all earlier-layer raw coordinates of the same "
            "concept. Use the frozen full-discovery ridge fit for the final "
            "transform; grouped five-fold predictions audit generalization."
        ),
        "ridge_alpha": args.ridge_alpha,
        "cross_fit_folds": args.folds,
        "cross_fit_group_source": "WikiText top-level article group",
        "transform_entries": transforms,
        "grouped_oof_r2_min": float(np.min(oof_r2_values)),
        "grouped_oof_r2_median": float(np.median(oof_r2_values)),
        "grouped_oof_r2_max": float(np.max(oof_r2_values)),
        "same_concept_abs_correlation_before_median": float(
            np.median(before_correlations)
        ),
        "same_concept_abs_correlation_after_median": float(
            np.median(after_correlations)
        ),
        "column_std_min": float(innovation_std.min()),
        "column_std_median": float(np.median(innovation_std)),
        "column_std_max": float(innovation_std.max()),
        "nan_count": int(np.isnan(innovation).sum()),
        "zero_variance_columns": int(np.sum(innovation_std < 1e-8)),
        "total_seconds": time.time() - started,
        "interpretation": (
            "The resulting variables are cross-layer innovations relative to "
            "same-concept carryover. They remain observed features and are "
            "not assumed to be causally sufficient or identifiable."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_npy = args.output_dir / f"{args.output_stem}.npy"
    output_npz = args.output_dir / f"{args.output_stem}.npz"
    output_json = args.output_dir / f"{args.output_stem}.json"
    np.save(output_npy, innovation)
    np.savez_compressed(
        output_npz,
        X=innovation,
        group_ids=group_ids,
        columns=np.asarray(
            [json.dumps(column, ensure_ascii=False) for column in columns]
        ),
    )
    output_json.write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(record, ensure_ascii=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
