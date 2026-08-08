"""Tests for the shared leakage-safe baseline experiment protocol."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from v6.graph import Graph
from v6.baselines.protocol import (
    REPORT_DATASETS,
    atomic_write_json,
    latent_activation,
    outer_folds,
    render_profiles,
    select_datasets,
    source_sha256,
    stable_seed,
)


class InterpretabilityProtocolTests(unittest.TestCase):
    def test_source_hashes_use_portable_relative_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "nested" / "source.py"
            source.parent.mkdir()
            source.write_text("value = 1\n", encoding="utf-8")

            hashes = source_sha256(root, [Path("nested") / "source.py"])
            self.assertEqual(list(hashes), ["nested/source.py"])
            self.assertRegex(hashes["nested/source.py"], r"^[0-9a-f]{64}$")
            with self.assertRaisesRegex(ValueError, "duplicate source path"):
                source_sha256(
                    root,
                    [Path("nested") / "source.py", Path("nested/source.py")],
                )

    def test_atomic_write_json_retries_transient_replace_permission_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "status.json"
            real_replace = __import__("os").replace
            attempts = 0

            def flaky_replace(source, target):
                nonlocal attempts
                attempts += 1
                if attempts < 3:
                    raise PermissionError("transient Windows sharing violation")
                return real_replace(source, target)

            with mock.patch(
                "v6.baselines.protocol.os.replace",
                side_effect=flaky_replace,
            ), mock.patch("v6.baselines.protocol.time.sleep"):
                atomic_write_json(path, {"state": "running"})

            self.assertEqual(attempts, 3)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {
                "state": "running"
            })

    def test_report_set_is_fixed_and_rejects_new_datasets(self):
        self.assertEqual(len(REPORT_DATASETS), 19)
        self.assertEqual(select_datasets("all"), list(REPORT_DATASETS))
        with self.assertRaises(ValueError):
            select_datasets("nhanes")

    def test_outer_folds_match_existing_protocol(self):
        folds = outer_folds(11, folds=5, seed=0)
        expected = np.random.default_rng(0).permutation(11)
        np.testing.assert_array_equal(np.concatenate(folds), np.concatenate(
            [expected[index::5] for index in range(5)]
        ))
        self.assertEqual(sorted(np.concatenate(folds).tolist()), list(range(11)))

    def test_pc1_orientation_uses_visible_descendant(self):
        graph = Graph(["secret-gold-name"], ["a", "b"], [
            ("secret-gold-name", "a"), ("secret-gold-name", "b")
        ])
        X = np.array([[-2.0, 2.0], [-1.0, 1.0], [1.0, -1.0], [2.0, -2.0]])
        activation, metadata = latent_activation(X, graph, "secret-gold-name", ["b"])
        self.assertGreater(np.corrcoef(activation, X[:, 1])[0, 1], 0.99)
        self.assertEqual(metadata["orientation_anchor_observed_index"], 1)

    def test_profiles_only_include_visible_labels_and_do_not_overlap(self):
        observed = ["a", "b", "hidden", "d"]
        labels = {name: f"LABEL-{name}" for name in observed}
        X = np.array([[4.0, 3.0, 100.0, 1.0]])
        profiles, vectors = render_profiles(X, observed, labels, [0, 1, 3], top_k=2)
        self.assertNotIn("LABEL-hidden", profiles[0])
        self.assertEqual(vectors.shape, (1, 3))
        lines = [line for line in profiles[0].splitlines() if line.startswith("- LABEL")]
        self.assertEqual(len(lines), len(set(lines)))

    def test_seed_and_atomic_json_are_deterministic(self):
        self.assertEqual(stable_seed("a", 1), stable_seed("a", 1))
        self.assertNotEqual(stable_seed("a", 1), stable_seed("a", 2))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "record.json"
            atomic_write_json(path, {"ok": True})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"ok": True})
            self.assertEqual(list(path.parent.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
