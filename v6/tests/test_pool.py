"""Regression tests for questionnaire-pool data cleaning."""

import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import pool


class PoolMatrixLoadingTest(unittest.TestCase):
    def test_copy_on_write_array_can_be_cleaned_in_place(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "data.csv")
            pd.DataFrame(
                {
                    "a": [1.0, 0.0, 3.0],
                    "b": [2.0, 2.0, 4.0],
                }
            ).to_csv(path, index=False)

            matrix = pool._load_matrix(
                path, ["a", "b"], ",", vmin=1, cap=10
            )

        self.assertEqual(matrix.shape, (2, 2))
        self.assertTrue(np.isfinite(matrix).all())


if __name__ == "__main__":
    unittest.main()
