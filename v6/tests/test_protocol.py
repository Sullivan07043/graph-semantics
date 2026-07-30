"""Protocol-level regression tests shared by every evaluation arm."""
import os
import sys
import unittest
from unittest import mock

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import metrics


class VisibleFoldAlphaTest(unittest.TestCase):
    def test_only_visible_embeddings_reach_alpha_calibration(self):
        embeddings = np.arange(20, dtype=float).reshape(5, 4)
        dictionary = np.ones((3, 4), dtype=float)
        seen = {}

        def fake_auto_alpha(received, concept_bank, target_l0):
            seen["embeddings"] = received.copy()
            seen["dictionary"] = concept_bank
            seen["target_l0"] = target_l0
            return 0.125

        with mock.patch.object(metrics.splice, "auto_alpha", side_effect=fake_auto_alpha):
            alpha = metrics.fold_alpha(embeddings, dictionary, [1, 4], target_l0=6)

        self.assertEqual(alpha, 0.125)
        np.testing.assert_array_equal(seen["embeddings"], embeddings[[1, 4]])
        self.assertIs(seen["dictionary"], dictionary)
        self.assertEqual(seen["target_l0"], 6)

    def test_empty_visible_fold_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "visible label"):
            metrics.fold_alpha(np.ones((2, 3)), np.ones((2, 3)), [])


if __name__ == "__main__":
    unittest.main()
