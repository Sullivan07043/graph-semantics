"""Regression tests for the preregistered Task 3 E0' decision taxonomy."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from task3_v2.scripts.run_e0_bridge import classify_decision  # noqa: E402


def classify(
    *,
    shuffle_mean: float,
    shuffle_ci_low: float,
    no_graph_mean: float,
    no_graph_ci_low: float,
    reverse_mean: float,
    reverse_ci_low: float,
    consistent_graph_count: int,
) -> str:
    return classify_decision(
        shuffle_mean=shuffle_mean,
        shuffle_ci_low=shuffle_ci_low,
        no_graph_mean=no_graph_mean,
        no_graph_ci_low=no_graph_ci_low,
        reverse_mean=reverse_mean,
        reverse_ci_low=reverse_ci_low,
        consistent_graph_count=consistent_graph_count,
    )


class DecisionRuleTests(unittest.TestCase):
    def test_go_requires_supported_primary_and_reverse_effects(self) -> None:
        self.assertEqual(
            classify(
                shuffle_mean=0.10,
                shuffle_ci_low=0.01,
                no_graph_mean=0.12,
                no_graph_ci_low=0.02,
                reverse_mean=0.08,
                reverse_ci_low=0.01,
                consistent_graph_count=2,
            ),
            "GO",
        )

    def test_directionally_positive_but_uncertain_is_inconclusive(self) -> None:
        self.assertEqual(
            classify(
                shuffle_mean=0.02,
                shuffle_ci_low=-0.01,
                no_graph_mean=0.03,
                no_graph_ci_low=-0.02,
                reverse_mean=-0.01,
                reverse_ci_low=-0.04,
                consistent_graph_count=1,
            ),
            "INCONCLUSIVE",
        )

    def test_observed_adverse_pattern_is_no_go(self) -> None:
        self.assertEqual(
            classify(
                shuffle_mean=0.0131,
                shuffle_ci_low=-0.1106,
                no_graph_mean=-0.1698,
                no_graph_ci_low=-0.3295,
                reverse_mean=-0.1305,
                reverse_ci_low=-0.2747,
                consistent_graph_count=0,
            ),
            "NO-GO",
        )

    def test_conflicting_reverse_evidence_is_inconclusive(self) -> None:
        self.assertEqual(
            classify(
                shuffle_mean=-0.03,
                shuffle_ci_low=-0.08,
                no_graph_mean=-0.02,
                no_graph_ci_low=-0.07,
                reverse_mean=0.05,
                reverse_ci_low=0.01,
                consistent_graph_count=0,
            ),
            "INCONCLUSIVE",
        )

    def test_zero_boundaries_are_not_positive_support(self) -> None:
        self.assertEqual(
            classify(
                shuffle_mean=0.0,
                shuffle_ci_low=0.0,
                no_graph_mean=-0.01,
                no_graph_ci_low=-0.02,
                reverse_mean=0.0,
                reverse_ci_low=0.0,
                consistent_graph_count=3,
            ),
            "NO-GO",
        )


if __name__ == "__main__":
    unittest.main()
