import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


V6 = Path(__file__).resolve().parents[2] / "v6"

from v6.graph import Graph
from v6.baselines import protocol
from v6.baselines.runners import clip_dissect_task1 as runner
from v6.baselines.clip_dissect_e5 import ConceptBank, TextDissectConfig


def test_observed_activation_matrix_standardizes_masked_columns_in_order():
    X = np.asarray([
        [1.0, 10.0, -2.0],
        [2.0, 20.0, -1.0],
        [3.0, 30.0, 1.0],
        [4.0, 40.0, 2.0],
    ])
    activations, metadata = protocol.observed_activation_matrix(X, 3, [2, 0])

    np.testing.assert_allclose(activations.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(activations.std(axis=0), 1.0, atol=1e-9)
    np.testing.assert_allclose(
        activations[:, 0], (X[:, 2] - X[:, 2].mean()) / X[:, 2].std()
    )
    assert [item["observed_index"] for item in metadata] == [2, 0]
    assert "labels" not in inspect.signature(protocol.observed_activation_matrix).parameters


def test_fold_adapter_batches_targets_and_never_encodes_masked_text():
    graph = Graph(["factor"], ["a", "b", "c", "d"], [
        ("factor", "a"), ("factor", "b"),
        ("factor", "c"), ("factor", "d"),
    ])
    dataset = {
        "name": "toy",
        "graph": graph,
        "X": np.asarray([
            [-2.0, 1.0, 2.0, 4.0],
            [-1.0, 3.0, 1.0, 3.0],
            [1.0, 6.0, -1.0, 1.0],
            [2.0, 8.0, -2.0, -1.0],
        ]),
        "labels": {
            "a": "VISIBLE-A",
            "b": "MASKED-SECRET-B",
            "c": "VISIBLE-C",
            "d": "MASKED-SECRET-D",
        },
    }
    encoded_texts = []

    def fake_embed(texts):
        encoded_texts.extend(str(value) for value in texts)
        return np.tile(np.asarray([[1.0, 0.5]]), (len(texts), 1))

    captured = {}

    def fake_text_dissect(profile_texts, activations, bank, **kwargs):
        captured["profile_texts"] = profile_texts
        captured["activations"] = np.asarray(activations)
        captured["latent_ids"] = list(kwargs["latent_ids"])
        return [SimpleNamespace(to_dict=lambda: {"construct_name": "x"}) for _ in range(2)]

    bank = ConceptBank(np.eye(2), ("x", "y"))
    with patch.object(runner, "run_clip_dissect_e5", side_effect=fake_text_dissect) as call:
        results, metadata = runner._interpret_fold(
            dataset,
            [3, 1],
            fake_embed,
            bank,
            TextDissectConfig(top_k=1, soft_top_k=2),
            profile_top_k=1,
        )

    assert call.call_count == 1
    assert len(results) == 2
    assert captured["profile_texts"] is None
    assert captured["activations"].shape == (4, 2)
    assert captured["latent_ids"] == ["masked_item_000", "masked_item_001"]
    assert [item["observed_index"] for item in metadata] == [1, 3]
    joined = "\n".join(encoded_texts)
    assert "MASKED-SECRET-B" not in joined
    assert "MASKED-SECRET-D" not in joined
    assert "VISIBLE-A" in joined
    assert "VISIBLE-C" in joined


def test_task1_metrics_match_canonical_hungarian_and_global_exact():
    gold = np.asarray([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    perfect = runner._task1_metrics_from_embeddings(gold[[2, 0]], [2, 0], gold)
    swapped = runner._task1_metrics_from_embeddings(gold[[0, 2]], [2, 0], gold)
    assert perfect == {"match_acc": 1.0, "exact": 1.0}
    assert swapped == {"match_acc": 0.0, "exact": 0.0}


def test_case_artifact_marks_gold_as_post_generation_and_disables_judge(tmp_path):
    graph = Graph(["factor"], ["a"], [("factor", "a")])
    dataset = {
        "name": "toy",
        "graph": graph,
        "labels": {"a": "gold item text"},
    }
    prediction = {
        "construct_name": "predicted construct",
        "native_diagnostics": {
            "positive_soft_wpmi": 1.0,
            "positive_rank_reorder": -0.5,
        },
    }
    runner._record_case(
        root=tmp_path,
        dataset_name="toy",
        fold=0,
        observed_index=0,
        observed_node="a",
        prediction=prediction,
        activation_metadata={"source": "masked_observed_response_column"},
        cfg_hash="abc",
    )
    generation_path = runner._generation_case_path(tmp_path, "toy", 0, 0)
    generation_record = json.loads(generation_path.read_text(encoding="utf-8"))
    assert "gold_label" not in generation_record
    assert not runner._case_path(tmp_path, "toy", 0, 0).exists()

    runner._attach_gold_labels(tmp_path, dataset, [(0, [0])], "abc")
    path = runner._case_path(tmp_path, "toy", 0, 0)
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["task"] == 1
    assert record["construct_name"] == "predicted construct"
    assert record["gold_label"] == "gold item text"
    assert record["llm_judge"] is None


def test_two_pass_runner_never_embeds_full_gold_before_all_folds_freeze(tmp_path):
    graph = Graph(["factor"], ["a", "b", "c", "d"], [
        ("factor", "a"), ("factor", "b"),
        ("factor", "c"), ("factor", "d"),
    ])
    gold_labels = ("GOLD-A", "GOLD-B", "GOLD-C", "GOLD-D")
    dataset = {
        "name": "toy",
        "graph": graph,
        "X": np.asarray([
            [-2.0, -1.0, 1.0, 2.0],
            [-1.0, 1.0, 2.0, -2.0],
            [1.0, 2.0, -2.0, -1.0],
            [2.0, -2.0, -1.0, 1.0],
        ]),
        "labels": dict(zip(graph.observed, gold_labels)),
    }
    events = []

    def sentinel_embed(texts):
        clean = tuple(str(value) for value in texts)
        events.append(("gold_embed" if clean == gold_labels else "other_embed", clean))
        vectors = np.zeros((len(clean), 4), dtype=float)
        for index in range(len(clean)):
            vectors[index, index % 4] = 1.0
        return vectors

    def fake_interpret(dataset_arg, masked, *args, **kwargs):
        events.append(("generate", tuple(masked)))
        results = []
        metadata = []
        for position, observed_index in enumerate(masked):
            result = {
                "construct_name": f"prediction-{observed_index}",
                "native_diagnostics": {
                    "positive_soft_wpmi": float(position),
                    "positive_rank_reorder": -float(position),
                },
            }
            results.append(SimpleNamespace(to_dict=lambda value=result: value))
            metadata.append({
                "source": "masked_observed_response_column",
                "observed_index": observed_index,
            })
        return results, metadata

    status = {
        "completed_new_cases": 0,
        "resumed_cases": 0,
        "current": None,
    }
    with patch.object(runner, "_interpret_fold", side_effect=fake_interpret):
        runner._run_dataset_two_pass(
            root=tmp_path,
            dataset=dataset,
            fold_specs=[(0, [0, 1]), (1, [2, 3])],
            cfg_hash="two-pass",
            embed=sentinel_embed,
            text_bank=ConceptBank(np.eye(2), ("x", "y")),
            text_config=TextDissectConfig(top_k=1, soft_top_k=2),
            profile_top_k=1,
            status=status,
        )

    kinds = [event[0] for event in events]
    assert kinds[:3] == ["generate", "generate", "gold_embed"]
    assert kinds.index("gold_embed") > max(
        index for index, kind in enumerate(kinds) if kind == "generate"
    )
    assert (tmp_path / "generation_frozen" / "toy.json").is_file()
    for fold, masked in [(0, [0, 1]), (1, [2, 3])]:
        for observed_index in masked:
            raw = json.loads(runner._generation_case_path(
                tmp_path, "toy", fold, observed_index
            ).read_text(encoding="utf-8"))
            assert "gold_label" not in raw
            final = json.loads(runner._case_path(
                tmp_path, "toy", fold, observed_index
            ).read_text(encoding="utf-8"))
            assert final["gold_label"] == gold_labels[observed_index]


def test_task1_runner_has_no_judge_dependency():
    source = (
        V6 / "baselines" / "runners" / "clip_dissect_task1.py"
    ).read_text(encoding="utf-8")
    assert "import judge" not in source
    assert "from v6 import judge" not in source
    assert "import metrics" not in source


def test_task1_v3_fingerprints_full_scorer_and_concept_bank():
    source = (
        V6 / "baselines" / "runners" / "clip_dissect_task1.py"
    ).read_text(encoding="utf-8")
    assert "task1_text_dissect_e5_report19_seed{args.seed}_v3" in source
    assert "task1-visible-label-folds-v2-text-dissect-e5-v3" in source
    assert '"parameters": asdict(text_config)' in source
    assert '"version": SCORER_VERSION' in source
    assert '"concept_bank_sha256": bank_sha256' in source
