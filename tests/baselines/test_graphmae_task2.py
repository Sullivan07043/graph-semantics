import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "v6"))

from graph import Graph
from v6.baselines.graphmae_gcn import (
    GraphExample,
    GraphMAEBaseline,
    GraphMAEConfig,
    file_sha256,
)
from v6.baselines.runners.graphmae_task2 import (
    ENCODER_MODEL,
    METHOD_VERSION,
    PROTOCOL_VERSION,
    SPLIT_VERSION,
    build_manifest,
    fold_indices,
    REPORT_DATASETS,
    select_datasets,
    training_datasets_for,
    validate_checkpoint_metadata,
)
import pool


def _graph(latent="latent"):
    return Graph(
        [latent],
        ["a", "b", "c"],
        [(latent, "a"), (latent, "b"), (latent, "c")],
    )


def _embeddings():
    return {
        "a": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "b": np.array([0.8, 0.2, 0.0, 0.0], dtype=np.float32),
        "c": np.array([0.7, 0.1, 0.2, 0.0], dtype=np.float32),
    }


def _config():
    return GraphMAEConfig(
        hidden_dim=8,
        encoder_layers=1,
        decoder_layers=1,
        epochs=3,
        device="cpu",
        seed=7,
    )


def test_training_rejects_latent_gold_embedding():
    values = _embeddings()
    values["latent"] = np.ones(4, dtype=np.float32)
    with pytest.raises(ValueError, match="observed nodes only"):
        GraphMAEBaseline(_config()).fit([GraphExample("train", _graph(), values)])


def test_latent_name_is_not_a_model_feature():
    model = GraphMAEBaseline(_config()).fit(
        [GraphExample("train", _graph("secret gold name"), _embeddings())]
    )
    visible = {key: value for key, value in _embeddings().items() if key != "c"}
    first = model.infer_latents(_graph("secret gold name"), visible)["secret gold name"]
    second = model.infer_latents(_graph("anonymous latent"), visible)["anonymous latent"]
    np.testing.assert_allclose(first, second, rtol=0, atol=0)


def test_checkpoint_round_trip_preserves_prediction_and_metadata(tmp_path):
    model = GraphMAEBaseline(_config()).fit(
        [GraphExample("train", _graph(), _embeddings())]
    )
    model.metadata_ = {"train_datasets": ["train"], "latent_supervision": False}
    checkpoint = tmp_path / "graphmae.pt"
    model.save_checkpoint(checkpoint)
    loaded = GraphMAEBaseline.load_checkpoint(checkpoint, device="cpu")
    visible = {key: value for key, value in _embeddings().items() if key != "c"}
    np.testing.assert_allclose(
        model.infer_latents(_graph(), visible)["latent"],
        loaded.infer_latents(_graph(), visible)["latent"],
    )
    assert loaded.metadata_ == model.metadata_


def test_lodo_split_never_trains_on_target_or_any_heldout_dataset():
    for target in [*pool.DEV, *pool.HELDOUT]:
        training = training_datasets_for(target)
        assert target not in training
        assert not (set(training) & set(pool.HELDOUT))
        assert len(training) == (len(pool.DEV) - 1 if target in pool.DEV else len(pool.DEV))


def test_report19_alias_matches_all():
    assert select_datasets("report19") == REPORT_DATASETS
    assert select_datasets("all") == REPORT_DATASETS


def test_checkpoint_provenance_validation_rejects_overlap():
    target = pool.HELDOUT[0]
    metadata = {
        "method_version": METHOD_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "target_dataset": target,
        "train_datasets": training_datasets_for(target),
        "latent_supervision": False,
        "encoder": {
            "model": ENCODER_MODEL,
            "revision": "a" * 40,
            "mode": "frozen-base-no-project-lora",
        },
    }
    validate_checkpoint_metadata(metadata, target, "a" * 40)
    metadata["train_datasets"] = [*metadata["train_datasets"], target]
    with pytest.raises(RuntimeError, match="provenance mismatch"):
        validate_checkpoint_metadata(metadata, target, "a" * 40)


def test_folds_are_deterministic_and_cover_each_observed_once():
    first = fold_indices(23, folds=5, seed=0)
    second = fold_indices(23, folds=5, seed=0)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    np.testing.assert_array_equal(np.sort(np.concatenate(first)), np.arange(23))


def test_manifest_uses_checkpoint_training_provenance_and_hashes(tmp_path):
    target = pool.HELDOUT[0]
    revision = "a" * 40
    training_source_hashes = {
        "graphmae_baseline.py": "1" * 64,
        "run_graphmae_task2.py": "2" * 64,
    }
    model = GraphMAEBaseline(_config()).fit(
        [GraphExample("train", _graph(), _embeddings())]
    )
    model.metadata_ = {
        "method_version": METHOD_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "target_dataset": target,
        "train_datasets": training_datasets_for(target),
        "target_excluded_from_training": True,
        "heldout_datasets_in_training": [],
        "latent_supervision": False,
        "latent_gold_used_for_training": False,
        "encoder": {
            "model": ENCODER_MODEL,
            "revision": revision,
            "mode": "frozen-base-no-project-lora",
        },
        "git_commit": "training-commit",
        "source_sha256": training_source_hashes,
    }

    checkpoint = tmp_path / "checkpoints" / f"{target}.pt"
    model.save_checkpoint(checkpoint)
    prediction = tmp_path / "results" / target / "predictions.npz"
    prediction.parent.mkdir(parents=True)
    np.savez_compressed(prediction, predictions=np.ones((5, 1, 4)))
    result_path = prediction.parent / "result.json"
    result = {
        "dataset": target,
        "split": "heldout",
        "train_datasets": training_datasets_for(target),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "prediction_artifact": str(prediction),
        "prediction_sha256": file_sha256(prediction),
    }
    result_path.write_text(json.dumps(result), encoding="utf-8")
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps({"score": 0.5}), encoding="utf-8")

    arguments = argparse.Namespace(
        mode="eval",
        datasets=target,
        output_dir=tmp_path,
        hf_cache=tmp_path / "hf-cache",
        encoder_revision=revision,
        encoder_device="cpu",
        allow_download=False,
        batch_size=8,
        device="cpu",
        epochs=3,
        hidden_dim=8,
        encoder_layers=1,
        decoder_layers=1,
        mask_rate=0.5,
        seed=7,
        folds=5,
        fold_seed=0,
        force_retrain=False,
    )
    manifest = build_manifest(
        arguments,
        [target],
        revision,
        _config(),
        {target: result},
        git_state={"commit": "evaluation-commit", "dirty": True},
    )

    assert manifest["split_version"] == SPLIT_VERSION
    assert manifest["git"] == {"commit": "evaluation-commit", "dirty": True}
    assert manifest["evaluation"] == {
        "folds": 5,
        "fold_seed": 0,
        "llm_judge_run": False,
    }
    assert manifest["cli"]["epochs"] == 3
    assert manifest["model_config"]["requested_by_cli"]["learning_rate"] == 1e-3
    checkpoint_config = manifest["model_config"]["training_from_checkpoints"][
        "shared"
    ]
    assert checkpoint_config["epochs"] == 3
    assert checkpoint_config["loss_alpha"] == 2.0
    assert manifest["model_config"]["training_from_checkpoints"][
        "targets_verified"
    ] == [target]
    assert manifest["checkpoint_source_sha256"] == {
        "shared": training_source_hashes,
        "targets_verified": [target],
    }
    assert manifest["artifacts"][target]["checkpoint"]["sha256"] == file_sha256(
        checkpoint
    )
    assert manifest["artifacts"][target]["result"]["sha256"] == file_sha256(
        result_path
    )
    assert manifest["artifacts"][target]["prediction"]["sha256"] == file_sha256(
        prediction
    )
    assert manifest["summary"]["sha256"] == file_sha256(summary_path)
