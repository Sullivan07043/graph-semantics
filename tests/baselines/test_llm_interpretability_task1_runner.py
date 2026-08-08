import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from v6.baselines.runners import llm_interpretability_task1 as runner


class DummyClient:
    spent_usd = 0.0


def test_task1_match_uses_masked_gold_assignment():
    gold = np.eye(4, dtype=float)
    predicted = gold[[3, 1]]
    assert runner._task1_match_from_embeddings(predicted, [3, 1], gold) == 1.0


def test_generation_hides_masked_text_and_freezes_gold_free_record(tmp_path, monkeypatch):
    observed = ["q0", "q1", "q2", "q3"]
    labels = {
        "q0": "Visible zero",
        "q1": "MASKED GOLD TEXT",
        "q2": "Visible two",
        "q3": "Visible three",
    }
    X = np.asarray(
        [
            [-2.0, -1.0, 0.0, 1.0],
            [-1.0, 0.0, 1.0, 2.0],
            [0.0, 1.0, 2.0, -2.0],
            [1.0, 2.0, -2.0, -1.0],
        ]
    )
    dataset = {
        "name": "toy",
        "graph": SimpleNamespace(observed=observed),
        "labels": labels,
        "X": X,
    }
    captured = {}

    def fake_autointerp(client, summaries, activations, seed):
        captured["summaries"] = list(summaries)
        captured["activations"] = np.asarray(activations)
        captured["seed"] = seed
        return {
            "construct_name": "Hidden Construct",
            "explanation": "High values indicate the hidden construct.",
            "spearman": 0.5,
            "pearson": 0.4,
        }

    monkeypatch.setattr(runner, "run_autointerp", fake_autointerp)
    status = {
        "current": None,
        "completed_new_cases": 0,
        "resumed_cases": 0,
        "api_spend_usd": 0.0,
    }
    runner._generate_dataset(
        root=tmp_path,
        dataset=dataset,
        methods=["autointerp"],
        fold_specs=[(0, [1])],
        client=DummyClient(),
        profile_top_k=1,
        seed=0,
        cfg_hash="cfg",
        status=status,
        status_every=1,
    )

    assert all("MASKED GOLD TEXT" not in value for value in captured["summaries"])
    expected = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
    np.testing.assert_allclose(captured["activations"], expected)
    record_path = runner._generation_case_path(
        tmp_path, "autointerp", "toy", 0, 1
    )
    record = json.loads(Path(record_path).read_text(encoding="utf-8"))
    assert record["construct_name"] == "Hidden Construct"
    assert "gold_label" not in record
    assert status["completed_new_cases"] == 1


def test_bipolar_connector_budget_and_method_aliases():
    assert runner._parse_methods("auto,delphi") == ["autointerp", "delphi"]
    assert runner._parse_budget("none") is None
    assert runner._parse_budget("2.5") == 2.5


def test_parallel_case_workers_freeze_every_masked_item(tmp_path, monkeypatch):
    observed = ["q0", "q1", "q2", "q3", "q4"]
    dataset = {
        "name": "parallel-toy",
        "graph": SimpleNamespace(observed=observed),
        "labels": {name: f"Visible label {name}" for name in observed},
        "X": np.asarray(
            [
                [-2.0, -1.0, 0.0, 1.0, 2.0],
                [-1.0, 0.0, 1.0, 2.0, -2.0],
                [0.0, 1.0, 2.0, -2.0, -1.0],
                [1.0, 2.0, -2.0, -1.0, 0.0],
            ]
        ),
    }

    def fake_delphi(client, summaries, activations, response_vectors, seed):
        return {
            "construct_name": f"Construct {seed}",
            "explanation": "High values indicate the generated construct.",
            "test_auroc": 0.5,
            "test_f1": 0.5,
        }

    monkeypatch.setattr(runner, "run_delphi", fake_delphi)
    status = {
        "current": None,
        "completed_new_cases": 0,
        "resumed_cases": 0,
        "api_spend_usd": 0.0,
    }
    runner._generate_dataset(
        root=tmp_path,
        dataset=dataset,
        methods=["delphi"],
        fold_specs=[(0, [1, 3])],
        client=DummyClient(),
        profile_top_k=1,
        seed=0,
        cfg_hash="cfg",
        status=status,
        status_every=1,
        case_workers=2,
    )

    assert status["completed_new_cases"] == 2
    for observed_index in (1, 3):
        assert runner._generation_case_path(
            tmp_path, "delphi", "parallel-toy", 0, observed_index
        ).is_file()
