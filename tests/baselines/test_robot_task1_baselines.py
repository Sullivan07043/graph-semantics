from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from task3_robotics.baselines import run_task1 as runner
from task3_robotics.baselines.common import load_robot_dataset, render_robot_snapshots
from task3_robotics.summarize_graph import dataset_name
from v6.baselines.automated_interpretability import (
    ROBOT_AUTOINTERP_EXPLAIN_PROMPT_VERSION,
    ROBOT_AUTOINTERP_SIMULATE_PROMPT_VERSION,
    run_autointerp,
)
from v6.baselines.clip_dissect_e5 import ConceptBank
from v6.baselines.delphi import (
    ROBOT_DELPHI_DETECT_PROMPT_VERSION,
    ROBOT_DELPHI_EXPLAIN_PROMPT_VERSION,
    run_delphi,
)
from v6.graph import Graph


class FakeClient:
    def __init__(self):
        self.calls = []

    def complete_json(self, **kwargs):
        self.calls.append(kwargs)
        version = kwargs["prompt_version"]
        if version == ROBOT_AUTOINTERP_EXPLAIN_PROMPT_VERSION:
            data = {
                "construct_name": "joint angle",
                "explanation": "Higher values indicate a larger joint angle.",
            }
        elif version == ROBOT_AUTOINTERP_SIMULATE_PROMPT_VERSION:
            ids = kwargs["schema"]["properties"]["predictions"]["items"][
                "properties"
            ]["sample_id"]["enum"]
            data = {
                "predictions": [
                    {"sample_id": sample_id, "activation_bin": index % 11}
                    for index, sample_id in enumerate(ids)
                ]
            }
        elif version == ROBOT_DELPHI_EXPLAIN_PROMPT_VERSION:
            data = {
                "candidates": [
                    {
                        "candidate_id": f"C{index}",
                        "construct_name": f"joint {index} angle",
                        "explanation": "Higher values indicate a larger joint angle.",
                    }
                    for index in range(1, 4)
                ]
            }
        elif version == ROBOT_DELPHI_DETECT_PROMPT_VERSION:
            ids = kwargs["schema"]["properties"]["predictions"]["items"][
                "properties"
            ]["sample_id"]["enum"]
            data = {
                "predictions": [
                    {
                        "sample_id": sample_id,
                        "scores": {"C1": 90, "C2": 50, "C3": 10},
                    }
                    for sample_id in ids
                ]
            }
        else:
            raise AssertionError(f"unexpected prompt version: {version}")
        return {
            "data": data,
            "cache_key": version,
            "prompt_version": version,
            "requested_model": "fake",
            "returned_model": "fake",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            "cost_usd": 0.0,
            "cached": False,
        }


class FakeEncoder:
    dimension = 16

    def embed(self, texts):
        matrix = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            encoded = str(text).encode("utf-8")
            for position, value in enumerate(encoded):
                matrix[row, (position + value) % self.dimension] += 1.0 + value / 255.0
        return matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)


def test_robot_prompt_versions_do_not_describe_humans_or_respondents():
    client = FakeClient()
    profiles = [f"robot-state snapshot {index}" for index in range(120)]
    activations = np.linspace(-3.0, 3.0, len(profiles))
    vectors = np.column_stack((activations, np.sin(activations)))

    run_autointerp(client, profiles, activations, seed=3, domain="robot")
    run_delphi(client, profiles, activations, vectors, seed=4, domain="robot")

    versions = {call["prompt_version"] for call in client.calls}
    assert versions == {
        ROBOT_AUTOINTERP_EXPLAIN_PROMPT_VERSION,
        ROBOT_AUTOINTERP_SIMULATE_PROMPT_VERSION,
        ROBOT_DELPHI_EXPLAIN_PROMPT_VERSION,
        ROBOT_DELPHI_DETECT_PROMPT_VERSION,
    }
    prompt = " ".join(
        call["system_prompt"] + " " + call["user_prompt"] for call in client.calls
    ).lower()
    assert "robot" in prompt
    assert "human dimension" not in prompt
    assert "respondent" not in prompt


def test_robot_loader_matches_mid_episode_channel_protocol(tmp_path: Path):
    rows = 149 * 2
    names = np.asarray(["joint.1@t-1", "action.1@t-1", "joint.1@t"])
    labels = np.asarray(["past joint angle", "x translation command", "joint 1 angle"])
    values = np.column_stack(
        (
            np.linspace(-1.0, 1.0, rows),
            np.linspace(1.0, 3.0, rows),
            np.linspace(2.0, 5.0, rows),
        )
    )
    np.savez(tmp_path / "lift_body_steps.npz", X=values, names=names, labels=labels)
    (tmp_path / "lift_body_summary.json").write_text(
        json.dumps(
            {
                "rlcd_directed": [["action.1", "joint.1"]],
                "edge_types": {"action.1->joint.1": "lag"},
            }
        ),
        encoding="utf-8",
    )

    dataset = load_robot_dataset("liftbody", tmp_path)

    assert dataset["X"].shape == (2, 2)
    assert dataset["graph"].observed == ["action.1", "joint.1"]
    assert dataset["labels"]["joint.1"] == "joint 1 angle"
    assert dataset["graph"].edge_type[("action.1", "joint.1")] == 0.0


def test_robot_snapshot_renderer_never_reads_masked_text():
    X = np.asarray([[-2.0, 0.0, 2.0], [2.0, 0.0, -2.0]])
    observed = ["a", "b", "secret"]
    snapshots, vectors = render_robot_snapshots(
        X,
        observed,
        {"a": "joint 1 angle", "b": "joint 2 angle"},
        [0, 1],
        top_k=1,
    )
    assert vectors.shape == (2, 2)
    assert all("secret" not in snapshot for snapshot in snapshots)
    assert all("robot channels" in snapshot for snapshot in snapshots)


def test_robot_summary_dataset_name_tracks_the_robot_filename():
    assert dataset_name("outputs/lift_body_discovered.json") == "liftbody"
    assert dataset_name("outputs/body_sawyer_discovered.json") == "bodysawyer"
    assert dataset_name("outputs/body_iiwa_discovered.json") == "bodyiiwa"
    assert dataset_name("outputs/body_ur5e_discovered.json") == "bodyur5e"


def _toy_dataset(name: str, offset: int) -> dict:
    observed = [f"channel_{index}" for index in range(5)]
    graph = Graph([], observed, [(observed[index], observed[index + 1]) for index in range(4)])
    rng = np.random.default_rng(offset)
    values = rng.normal(size=(120, len(observed)))
    values[:, 1:] += values[:, :-1] * 0.4
    return {
        "name": name,
        "graph": graph,
        "X": values,
        "labels": {node: f"joint {index + 1} angle" for index, node in enumerate(observed)},
        "latent_gt": {},
    }


def test_non_llm_robot_adapters_complete_five_fold_match(tmp_path: Path):
    datasets = {
        name: _toy_dataset(name, index)
        for index, name in enumerate(runner.ROBOT_DATASETS)
    }
    encoder = FakeEncoder()
    cfg_hash = "test-config"

    runner._run_feature_propagation(
        tmp_path, datasets["bodyur5e"], encoder, cfg_hash
    )
    runner._run_graphmae(
        tmp_path,
        datasets["bodyur5e"],
        datasets,
        encoder,
        cfg_hash,
        epochs=1,
        device="cpu",
    )
    concept_names = tuple(f"robot concept {index}" for index in range(120))
    concept_bank = ConceptBank(encoder.embed(concept_names), concept_names)
    runner._run_clip_dissect(
        tmp_path,
        datasets["bodyur5e"],
        encoder,
        concept_bank,
        cfg_hash,
        profile_top_k=2,
    )

    for method in ("feature-propagation", "graphmae-gcn", "clip-dissect-e5"):
        metrics = [
            json.loads(
                runner._metric_path(tmp_path, method, "bodyur5e", fold).read_text(
                    encoding="utf-8"
                )
            )
            for fold in range(5)
        ]
        assert all(0.0 <= metric["match_acc"] <= 1.0 for metric in metrics)
        assert all(metric["llm_judge"] is None for metric in metrics)
