"""Focused tests for the Feature Propagation Task 1 adaptation."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from v6.graph import Graph
from v6.baselines.runners.feature_propagation_task1 import (
    ENCODER_MODEL,
    ENCODER_PREFIX,
    ENCODER_REVISION,
    LOCAL_SOURCE_FILES,
    PROTOCOL_VERSION,
    _embedding_function,
    evaluate_fold,
    main,
    predict_fold,
    task1_match_details,
)


class Task1FoldProtocolTests(unittest.TestCase):
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
        self.dataset = {
            "name": "toy",
            "graph": self.graph,
            "labels": {
                "a_visible": "visible alpha",
                "a_hidden": "SECRET HIDDEN ALPHA",
                "b_visible": "visible beta",
                "b_hidden": "SECRET HIDDEN BETA",
            },
            # Task 1 retains numeric data, but Feature Propagation must not use it.
            "X": object(),
        }
        self.vectors = {
            "visible alpha": [1.0, 0.0],
            "SECRET HIDDEN ALPHA": [1.0, 0.0],
            "visible beta": [0.0, 1.0],
            "SECRET HIDDEN BETA": [0.0, 1.0],
        }
        self.calls: list[tuple[str, ...]] = []

    def _embed(self, texts):
        texts = tuple(texts)
        self.calls.append(texts)
        return np.asarray([self.vectors[text] for text in texts], dtype=float)

    def test_prediction_sees_only_visible_labels_before_gold_evaluation(self):
        prediction = predict_fold(self.dataset, [1, 3], self._embed)

        self.assertEqual(self.calls, [("visible alpha", "visible beta")])
        self.assertFalse(any("SECRET" in text for text in self.calls[0]))

        result = evaluate_fold(prediction, self.dataset, self._embed)

        self.assertEqual(
            self.calls,
            [
                ("visible alpha", "visible beta"),
                ("SECRET HIDDEN ALPHA", "SECRET HIDDEN BETA"),
            ],
        )
        self.assertEqual(result["observed_match_acc"], 1.0)
        self.assertEqual(result["hungarian_assignment"], [0, 1])

    def test_fold_validation_rejects_empty_duplicate_and_out_of_range_masks(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            predict_fold(self.dataset, [], self._embed)
        with self.assertRaisesRegex(ValueError, "must be unique"):
            predict_fold(self.dataset, [1, 1], self._embed)
        with self.assertRaisesRegex(IndexError, "outside"):
            predict_fold(self.dataset, [4], self._embed)
        with self.assertRaisesRegex(ValueError, "at least one visible"):
            predict_fold(self.dataset, [0, 1, 2, 3], self._embed)

    def test_hungarian_metric_matches_the_historical_task1_convention(self):
        gold = np.asarray([[1.0, 0.0], [0.0, 1.0]])
        perfect, similarities, assignment = task1_match_details(gold, gold)
        swapped, _, swapped_assignment = task1_match_details(gold[::-1], gold)

        self.assertEqual(perfect, 1.0)
        self.assertEqual(assignment, [0, 1])
        self.assertEqual(swapped, 0.0)
        self.assertEqual(swapped_assignment, [1, 0])
        np.testing.assert_allclose(similarities, np.eye(2), atol=1e-8)

    def test_runner_has_no_judge_or_exact_metric_dependency(self):
        source = (
            Path(__file__).parents[2]
            / "v6"
            / "baselines"
            / "runners"
            / "feature_propagation_task1.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("import judge", source)
        self.assertNotIn("exact_acc", source)
        self.assertNotIn('"exact"', source)


class Task1ProvenanceTests(unittest.TestCase):
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

    def test_manifest_fold_summary_and_resume_are_auditable(self):
        graph = Graph(
            ["latent"],
            ["item0", "item1", "item2", "item3", "item4"],
            [("latent", f"item{index}") for index in range(5)],
        )
        dataset = {
            "name": "cfcs",
            "graph": graph,
            "labels": {f"item{index}": f"label {index}" for index in range(5)},
        }
        vectors = {
            f"label {index}": np.asarray([float(index + 1), 1.0])
            for index in range(5)
        }
        calls: list[tuple[str, ...]] = []

        def fake_embed(texts):
            texts = tuple(texts)
            calls.append(texts)
            return np.asarray([vectors[text] for text in texts], dtype=float)

        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "artifact"
            patches = (
                mock.patch(
                    "v6.baselines.runners.feature_propagation_task1._embedding_function",
                    return_value=fake_embed,
                ),
                mock.patch(
                    "v6.baselines.runners.feature_propagation_task1.report_loaders",
                    return_value={"cfcs": lambda: dataset},
                ),
                mock.patch(
                    "v6.baselines.runners.feature_propagation_task1._git_commit",
                    return_value="test-commit",
                ),
                mock.patch(
                    "v6.baselines.runners.feature_propagation_task1._git_dirty",
                    return_value=True,
                ),
            )
            with patches[0], patches[1], patches[2], patches[3]:
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
            calls_after_first_run = list(calls)

            with patches[0], patches[1], patches[2], patches[3]:
                resumed = main(
                    [
                        "--datasets",
                        "cfcs",
                        "--output-dir",
                        str(output),
                        "--max-dataset-folds",
                        "1",
                    ]
                )
            self.assertEqual(resumed, 0)
            self.assertEqual(calls, calls_after_first_run)

            manifest = json.loads(
                (output / "manifest.json").read_text(encoding="utf-8")
            )
            summary = json.loads(
                (output / "summary.json").read_text(encoding="utf-8")
            )
            status = json.loads(
                (output / "status.json").read_text(encoding="utf-8")
            )
            fold = json.loads(
                (output / "folds" / "cfcs" / "fold_00.json").read_text(
                    encoding="utf-8"
                )
            )

            expected_encoder = {
                "model_id": ENCODER_MODEL,
                "revision": ENCODER_REVISION,
                "prefix": ENCODER_PREFIX,
                "frozen": True,
            }
            self.assertEqual(
                PROTOCOL_VERSION, "feature-propagation-task1-report19-v1"
            )
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
            self.assertEqual(summary["completed_dataset_folds"], 1)
            self.assertEqual(summary["expected_dataset_folds"], 5)
            self.assertIsNone(summary["llm_judge"])
            self.assertEqual(status["state"], "limited")
            self.assertEqual(status["resumed_folds"], 1)
            self.assertEqual(fold["task"], 1)
            self.assertIn("observed_match_acc", fold)
            self.assertNotIn("exact", fold)
            self.assertIsNone(fold["llm_judge"])


if __name__ == "__main__":
    unittest.main()
