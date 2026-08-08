"""Focused tests for the Feature Propagation Task 2 adaptation."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from v6.baselines.feature_propagation import (
    feature_propagation,
    predict_task2_latent_embeddings,
)
from v6.graph import Graph
from v6.baselines.runners.feature_propagation_task2 import (
    ENCODER_MODEL,
    ENCODER_PREFIX,
    ENCODER_REVISION,
    LOCAL_SOURCE_FILES,
    PROTOCOL_VERSION,
    _embedding_function,
    evaluate_fold,
    latent_match_details,
    main,
    predict_fold,
)


class FeaturePropagationRuleTests(unittest.TestCase):
    def test_historical_rule_clamps_known_nodes_and_ignores_direction(self):
        graph = Graph(
            [],
            ["left", "middle", "right"],
            [("middle", "left"), ("middle", "right")],
        )
        known = {
            "left": np.array([2.0, -1.0]),
            "right": np.array([4.0, 3.0]),
        }

        result = feature_propagation(graph, known)

        np.testing.assert_array_equal(result["left"], known["left"])
        np.testing.assert_array_equal(result["right"], known["right"])
        np.testing.assert_allclose(
            result["middle"],
            (known["left"] + known["right"]) / np.sqrt(2.0),
        )

    def test_task2_adapter_rejects_latent_or_unknown_anchors(self):
        graph = Graph(["latent"], ["item"], [("latent", "item")])
        with self.assertRaisesRegex(ValueError, "must all be observed"):
            predict_task2_latent_embeddings(
                graph, {"latent": np.ones(2)}
            )
        with self.assertRaisesRegex(ValueError, "must all be observed"):
            predict_task2_latent_embeddings(
                graph, {"not-a-node": np.ones(2)}
            )

    def test_unanchored_latent_uses_explicit_zero_fallback(self):
        graph = Graph(
            ["anchored", "unanchored"],
            ["visible", "isolated_item"],
            [("anchored", "visible")],
        )
        result = predict_task2_latent_embeddings(
            graph, {"visible": np.array([1.0, -2.0])}
        )
        np.testing.assert_allclose(result["unanchored"], np.zeros(2))


class Task2FoldProtocolTests(unittest.TestCase):
    def setUp(self):
        self.graph = Graph(
            ["opaque_latent_0", "opaque_latent_1"],
            ["a_visible", "a_hidden", "b_visible", "b_hidden"],
            [
                ("opaque_latent_0", "a_visible"),
                ("opaque_latent_0", "a_hidden"),
                ("opaque_latent_1", "b_visible"),
                ("opaque_latent_1", "b_hidden"),
            ],
        )
        self.dataset_without_gold = {
            "name": "toy",
            "graph": self.graph,
            "labels": {
                "a_visible": "visible alpha",
                "a_hidden": "SECRET HIDDEN ALPHA",
                "b_visible": "visible beta",
                "b_hidden": "SECRET HIDDEN BETA",
            },
            # The prediction helper must not touch numerical responses either.
            "X": object(),
        }
        self.vectors = {
            "visible alpha": [1.0, 0.0],
            "visible beta": [0.0, 1.0],
            "gold alpha": [1.0, 0.0],
            "gold beta": [0.0, 1.0],
        }
        self.calls: list[tuple[str, ...]] = []

    def _embed(self, texts):
        texts = tuple(texts)
        self.calls.append(texts)
        return np.asarray([self.vectors[text] for text in texts], dtype=float)

    def test_prediction_encodes_only_visible_labels_then_gold_at_evaluation(self):
        prediction = predict_fold(
            self.dataset_without_gold,
            [1, 3],
            self._embed,
        )

        self.assertEqual(self.calls, [("visible alpha", "visible beta")])
        self.assertFalse(any("SECRET" in text for call in self.calls for text in call))

        dataset = dict(
            self.dataset_without_gold,
            latent_gt={
                "opaque_latent_0": "gold alpha",
                "opaque_latent_1": "gold beta",
            },
        )
        result = evaluate_fold(prediction, dataset, self._embed)

        self.assertEqual(
            self.calls,
            [
                ("visible alpha", "visible beta"),
                ("gold alpha", "gold beta"),
            ],
        )
        self.assertEqual(result["latent_match_acc"], 1.0)
        self.assertEqual(result["hungarian_assignment"], [0, 1])

    def test_hungarian_metric_matches_current_single_factor_convention(self):
        score, similarities, assignment = latent_match_details(
            np.asarray([[1.0, 0.0]]), np.asarray([[1.0, 0.0]])
        )
        self.assertIsNone(score)
        self.assertIsNone(assignment)
        np.testing.assert_allclose(similarities, [[1.0]])

    def test_runner_cannot_import_or_call_llm_judge(self):
        source = (
            Path(__file__).parents[2]
            / "v6"
            / "baselines"
            / "runners"
            / "feature_propagation_task2.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("import judge", source)
        self.assertNotIn("from v6 import judge", source)


class Task2ProvenanceTests(unittest.TestCase):
    def test_encoder_loader_pins_exact_hugging_face_revision_and_prefix(self):
        encoder = mock.Mock()
        encoder.encode.return_value = np.ones((2, 3), dtype=float)
        with (
            mock.patch(
                "sentence_transformers.SentenceTransformer",
                return_value=encoder,
            ) as constructor,
            mock.patch("torch.cuda.is_available", return_value=False),
        ):
            embed = _embedding_function(batch_size=7)
            result = embed(["alpha", "beta"])

        constructor.assert_called_once_with(
            ENCODER_MODEL,
            revision=ENCODER_REVISION,
            device=None,
            cache_folder=os.environ["HF_CACHE"],
        )
        encoder.encode.assert_called_once_with(
            [ENCODER_PREFIX + "alpha", ENCODER_PREFIX + "beta"],
            batch_size=7,
            normalize_embeddings=True,
        )
        np.testing.assert_array_equal(result, np.ones((2, 3), dtype=float))

    def test_v3_manifest_duplicates_encoder_and_all_local_source_hashes(self):
        graph = Graph(
            ["latent"],
            ["item0", "item1", "item2", "item3", "item4"],
            [("latent", f"item{index}") for index in range(5)],
        )
        dataset = {
            "name": "cfcs",
            "graph": graph,
            "labels": {f"item{index}": f"label {index}" for index in range(5)},
            "latent_gt": {"latent": "gold latent"},
        }

        def fake_embed(texts):
            return np.asarray(
                [[float(index + 1), 1.0] for index, _ in enumerate(texts)],
                dtype=float,
            )

        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "artifact"
            with (
                mock.patch(
                    "v6.baselines.runners.feature_propagation_task2._embedding_function",
                    return_value=fake_embed,
                ),
                mock.patch(
                    "v6.baselines.runners.feature_propagation_task2.report_loaders",
                    return_value={"cfcs": lambda: dataset},
                ),
                mock.patch(
                    "v6.baselines.runners.feature_propagation_task2._git_commit",
                    return_value="test-commit",
                ),
                mock.patch(
                    "v6.baselines.runners.feature_propagation_task2._git_dirty",
                    return_value=True,
                ),
            ):
                result = main(
                    [
                        "--datasets",
                        "cfcs",
                        "--output-dir",
                        str(output),
                        "--max-dataset-folds",
                        "1",
                    ]
                )

            self.assertEqual(result, 0)
            manifest = json.loads(
                (output / "manifest.json").read_text(encoding="utf-8")
            )
            expected_encoder = {
                "model_id": ENCODER_MODEL,
                "revision": ENCODER_REVISION,
                "prefix": ENCODER_PREFIX,
                "frozen": True,
            }
            self.assertEqual(PROTOCOL_VERSION, "feature-propagation-task2-report19-v3")
            self.assertEqual(manifest["encoder"], expected_encoder)
            self.assertEqual(manifest["config"]["encoder"], expected_encoder)
            self.assertEqual(
                set(manifest["implementation_sha256"]), set(LOCAL_SOURCE_FILES)
            )
            self.assertEqual(
                manifest["implementation_sha256"],
                manifest["config"]["implementation_sha256"],
            )
            for digest in manifest["implementation_sha256"].values():
                self.assertRegex(digest, r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
