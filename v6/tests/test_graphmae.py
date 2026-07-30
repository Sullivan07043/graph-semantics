"""Fast CPU smoke tests for the standalone GraphMAE baseline."""

import os
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from graph import Graph
from graphmae import GraphExample, GraphMAEBaseline, GraphMAEConfig, file_sha256


class GraphMAESmokeTest(unittest.TestCase):
    @staticmethod
    def _example():
        graph = Graph(
            latents=["L_warm", "L_cool"],
            observed=["kind", "helpful", "calm", "quiet"],
            edges=[
                ("L_warm", "kind"),
                ("L_warm", "helpful"),
                ("L_cool", "calm"),
                ("L_cool", "quiet"),
                ("L_warm", "L_cool"),
            ],
        )
        embeddings = {
            "kind": np.array([1.0, 0.1, 0.0, 0.0], np.float32),
            "helpful": np.array([0.9, 0.2, 0.0, 0.0], np.float32),
            "calm": np.array([0.0, 0.1, 1.0, 0.0], np.float32),
            "quiet": np.array([0.0, 0.0, 0.9, 0.2], np.float32),
        }
        return GraphExample(graph, embeddings)

    def test_fit_infer_and_checkpoint_roundtrip(self):
        example = self._example()
        config = GraphMAEConfig(
            hidden_dim=16,
            epochs=8,
            masks_per_graph=1,
            dropout=0.0,
            seed=17,
            device="cpu",
        )
        baseline = GraphMAEBaseline(config).fit([example])
        baseline.metadata_ = {"train_datasets": ["toy"], "seed": config.seed}
        self.assertEqual(len(baseline.history_), config.epochs)
        self.assertTrue(np.isfinite(baseline.history_).all())

        visible = {
            name: value
            for name, value in example.observed_embeddings.items()
            if name != "helpful"
        }
        predicted = baseline.infer(
            example.graph,
            visible,
            missing_nodes=["helpful", "L_warm"],
        )
        self.assertEqual(set(predicted), {"helpful", "L_warm"})
        for embedding in predicted.values():
            self.assertEqual(embedding.shape, (4,))
            self.assertTrue(np.isfinite(embedding).all())
            self.assertAlmostEqual(float(np.linalg.norm(embedding)), 1.0, places=5)

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = os.path.join(tmp, "graphmae.pt")
            baseline.save_checkpoint(checkpoint)
            restored = GraphMAEBaseline.load_checkpoint(checkpoint, device="cpu")
            self.assertEqual(restored.metadata_, baseline.metadata_)
            restored_prediction = restored.infer(
                example.graph,
                visible,
                missing_nodes=["helpful", "L_warm"],
            )
        for name in predicted:
            np.testing.assert_allclose(
                predicted[name], restored_prediction[name], rtol=0.0, atol=1e-7
            )

    def test_file_sha256_depends_on_content_not_mtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = os.path.join(tmp, "first.bin")
            second = os.path.join(tmp, "second.bin")
            with open(first, "wb") as handle:
                handle.write(b"same encoder bytes")
            with open(second, "wb") as handle:
                handle.write(b"same encoder bytes")
            os.utime(first, (1, 1))
            os.utime(second, (2, 2))
            self.assertEqual(file_sha256(first), file_sha256(second))
            with open(second, "ab") as handle:
                handle.write(b"changed")
            self.assertNotEqual(file_sha256(first), file_sha256(second))

    def test_training_rejects_latent_gold_embedding(self):
        example = self._example()
        contaminated = dict(example.observed_embeddings)
        contaminated["L_warm"] = np.ones(4, np.float32)
        config = GraphMAEConfig(hidden_dim=8, epochs=1, device="cpu")
        with self.assertRaisesRegex(ValueError, "observed nodes"):
            GraphMAEBaseline(config).fit(
                [GraphExample(example.graph, contaminated)]
            )


if __name__ == "__main__":
    unittest.main()
