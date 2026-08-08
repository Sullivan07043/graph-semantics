import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
V6 = ROOT / "v6"
sys.path.insert(0, str(V6))

from graph import Graph
from v6.baselines.graphmae_gcn import GraphExample, GraphMAEBaseline, GraphMAEConfig
from v6.baselines.runners import graphmae_task1 as runner


def _graph():
    return Graph(
        ["factor"],
        ["a", "b", "c", "d", "e"],
        [("factor", node) for node in ["a", "b", "c", "d", "e"]],
    )


def _embeddings():
    return {
        node: np.eye(5, dtype=np.float32)[index]
        for index, node in enumerate(["a", "b", "c", "d", "e"])
    }


def test_infer_observed_targets_is_the_decoder_remasked_training_readout():
    config = GraphMAEConfig(
        hidden_dim=8,
        encoder_layers=1,
        decoder_layers=1,
        epochs=2,
        seed=7,
        device="cpu",
    )
    baseline = GraphMAEBaseline(config).fit(
        [GraphExample("train", _graph(), _embeddings())]
    )
    visible = {node: value for node, value in _embeddings().items() if node != "c"}
    prediction = baseline.infer_observed_targets(_graph(), visible, ["c"])

    assert list(prediction) == ["c"]
    assert prediction["c"].shape == (5,)
    np.testing.assert_allclose(np.linalg.norm(prediction["c"]), 1.0, atol=1e-6)

    two_hidden = {
        node: value for node, value in _embeddings().items() if node not in {"c", "d"}
    }
    with pytest.raises(ValueError, match="exactly the hidden observed nodes"):
        baseline.infer_observed_targets(_graph(), two_hidden, ["c"])
    with pytest.raises(ValueError, match="observed graph nodes"):
        baseline.infer_observed_targets(_graph(), visible, ["factor"])


def test_task1_match_acc_is_fold_local_hungarian():
    gold = np.eye(4)
    assert runner.task1_match_acc(gold[[3, 1]], [3, 1], gold) == 1.0
    assert runner.task1_match_acc(gold[[1, 3]], [3, 1], gold) == 0.0


def test_two_pass_runner_freezes_all_predictions_before_gold_embedding(tmp_path):
    graph = _graph()
    nodes = list(graph.observed)
    labels = {node: f"GOLD-{index}" for index, node in enumerate(nodes)}
    dataset = {"name": "toy", "graph": graph, "labels": labels}
    events = []

    class SentinelEncoder:
        def embed(self, texts):
            clean = tuple(str(value) for value in texts)
            kind = "gold_embed" if clean == tuple(labels[node] for node in nodes) else "visible_embed"
            events.append((kind, clean))
            values = np.zeros((len(clean), len(nodes)), dtype=np.float32)
            for row, text in enumerate(clean):
                values[row, int(text.split("-")[-1])] = 1.0
            return values

    class SentinelModel:
        def infer_observed_targets(self, graph_arg, visible, targets):
            events.append(("predict", tuple(targets), tuple(visible)))
            return {
                node: np.eye(len(nodes), dtype=np.float32)[nodes.index(node)]
                for node in targets
            }

    checkpoint = {
        "sha256": "a" * 64,
        "train_datasets": ["train-a", "train-b"],
    }
    status = {
        "completed_datasets": [],
        "completed_prediction_folds": 0,
        "resumed_prediction_folds": 0,
        "completed_metric_folds": 0,
        "resumed_metric_folds": 0,
        "current": None,
    }
    result = runner.run_dataset_two_pass(
        root=tmp_path,
        dataset=dataset,
        model=SentinelModel(),
        encoder=SentinelEncoder(),
        checkpoint_audit=checkpoint,
        config_hash="cfg",
        revision=runner.PINNED_ENCODER_REVISION,
        folds=5,
        fold_seed=0,
        status=status,
    )

    kinds = [event[0] for event in events]
    assert kinds.count("predict") == 5
    assert kinds.count("gold_embed") == 1
    assert kinds.index("gold_embed") > max(
        index for index, kind in enumerate(kinds) if kind == "predict"
    )
    assert result["match_acc"] == 1.0
    assert (tmp_path / "generation_frozen" / "toy.json").is_file()
    freeze = json.loads(
        (tmp_path / "generation_frozen" / "toy.json").read_text(encoding="utf-8")
    )
    assert freeze["gold_embedded_before_freeze"] is False
    for fold in range(5):
        metadata = json.loads(
            (tmp_path / "predictions" / "toy" / f"fold_{fold:02d}.json").read_text(
                encoding="utf-8"
            )
        )
        assert metadata["gold_text_available_to_prediction"] is False
        assert "gold_label" not in metadata

    class NoPredictionModel:
        def infer_observed_targets(self, *args, **kwargs):
            raise AssertionError("a resumable prediction should not be regenerated")

    resumed_status = {
        "completed_datasets": [],
        "completed_prediction_folds": 0,
        "resumed_prediction_folds": 0,
        "completed_metric_folds": 0,
        "resumed_metric_folds": 0,
        "current": None,
    }
    resumed = runner.run_dataset_two_pass(
        root=tmp_path,
        dataset=dataset,
        model=NoPredictionModel(),
        encoder=SentinelEncoder(),
        checkpoint_audit=checkpoint,
        config_hash="cfg",
        revision=runner.PINNED_ENCODER_REVISION,
        folds=5,
        fold_seed=0,
        status=resumed_status,
    )
    assert resumed["match_acc"] == 1.0
    assert resumed_status["resumed_prediction_folds"] == 5
    assert resumed_status["resumed_metric_folds"] == 5


def test_runner_has_no_judge_or_exact_metric_dependency():
    source = (
        V6 / "baselines" / "runners" / "graphmae_task1.py"
    ).read_text(encoding="utf-8")
    assert "import judge" not in source
    assert "import metrics" not in source
    assert '"exact"' not in source
