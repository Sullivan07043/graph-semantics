from pathlib import Path
import tempfile
import unittest

import numpy as np

from v6.baselines.runners.interpretability_task2 import (
    _case_path,
    _latent_match,
    _parse_budget,
    _parse_methods,
)


class RunInterpretabilityBaselineTests(unittest.TestCase):
    def test_method_aliases_and_case_paths(self):
        self.assertEqual(_parse_methods("all"), [
            "autointerp", "delphi", "text-dissect"
        ])
        self.assertEqual(_parse_methods("auto,text"), ["autointerp", "text-dissect"])
        with tempfile.TemporaryDirectory() as directory:
            path = _case_path(Path(directory), "autointerp", "tlvd", 2, 3)
            self.assertTrue(str(path).endswith("autointerp\\tlvd\\fold_02\\latent_003.json")
                            or str(path).endswith("autointerp/tlvd/fold_02/latent_003.json"))

    def test_budget_can_be_explicitly_unlimited(self):
        self.assertIsNone(_parse_budget("none"))
        self.assertEqual(_parse_budget("2.5"), 2.5)
        with self.assertRaises(Exception):
            _parse_budget("-1")

    def test_match_is_local_hungarian_and_single_factor_is_na(self):
        vectors = {
            "p0": [1.0, 0.0], "p1": [0.0, 1.0],
            "g0": [1.0, 0.0], "g1": [0.0, 1.0],
        }
        embed = lambda texts: np.asarray([vectors[text] for text in texts], float)
        self.assertEqual(_latent_match(["p0", "p1"], ["g0", "g1"], embed), 1.0)
        self.assertIsNone(_latent_match(["p0"], ["g0"], embed))

    def test_runner_has_no_judge_import(self):
        source = (
            Path(__file__).parents[2]
            / "v6"
            / "baselines"
            / "runners"
            / "interpretability_task2.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("import judge", source)
        self.assertNotIn("from v6 import judge", source)

    def test_task2_text_v3_has_distinct_cache_and_full_provenance(self):
        source = (
            Path(__file__).parents[2]
            / "v6"
            / "baselines"
            / "runners"
            / "interpretability_task2.py"
        ).read_text(encoding="utf-8")
        self.assertIn("interpretability-baselines-report19-v2-text-dissect-e5-v3", source)
        self.assertIn("task2_text_dissect_e5_report19_seed{args.seed}_v3", source)
        self.assertIn('"parameters": asdict(text_config)', source)
        self.assertIn('"version": SCORER_VERSION', source)
        self.assertIn('"concept_bank_sha256": text_bank_sha256', source)


if __name__ == "__main__":
    unittest.main()
