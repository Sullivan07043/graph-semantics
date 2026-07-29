from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from task3_v2.scripts import run_e0_audit as audit  # noqa: E402


class AuditDecisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows: dict[tuple[str, str, str, str], dict[str, Any]] = {}

    def add(
        self,
        group: str,
        comparison: str,
        metric: str,
        direction: str,
    ) -> None:
        analysis = (
            "constraint_decomposition"
            if group == "all_nodes"
            else "structural_group"
        )
        if direction == "positive":
            mean, low, high = 0.10, 0.05, 0.15
        elif direction == "adverse":
            mean, low, high = -0.10, -0.15, -0.05
        else:
            raise ValueError(direction)
        key = (analysis, group, comparison, metric)
        self.assertNotIn(key, self.rows)
        self.rows[key] = {
            "analysis": analysis,
            "group": group,
            "comparison": comparison,
            "metric": metric,
            "status": "complete",
            "mean": mean,
            "ci_low": low,
            "ci_high": high,
        }

    def add_clean_same_module_positive(self) -> None:
        for comparison in (
            "same_module_minus_shuffled_full",
            "same_module_minus_uniform",
        ):
            self.add("all_nodes", comparison, "centered_cosine", "positive")
            self.add("all_nodes", comparison, "mrr", "positive")

    def decide(
        self,
        *,
        orientation_bug: bool = False,
        bundle_pass: bool = True,
    ) -> dict[str, Any]:
        return audit.make_decision(
            {"verdict": {"orientation_interface_bug": orientation_bug}},
            {"behavioral_trend_reproduced": bundle_pass},
            [],
            [],
            list(self.rows.values()),
        )

    def test_orientation_and_bundle_gates_precede_diagnostics(self) -> None:
        self.assertEqual(
            self.decide(orientation_bug=True, bundle_pass=False)["primary_category"],
            "A",
        )
        self.assertEqual(
            self.decide(orientation_bug=False, bundle_pass=False)["primary_category"],
            "B",
        )

    def test_clean_root_boundary_can_select_c(self) -> None:
        self.add_clean_same_module_positive()
        for metric in ("gold_cosine", "centered_cosine", "prediction_margin"):
            self.add(
                "root",
                "full_oracle_minus_reversed_full",
                metric,
                "adverse",
            )
        for metric in ("centered_cosine", "prediction_margin", "mrr"):
            self.add(
                "non_root_visible_parent",
                "full_oracle_minus_shuffled_full",
                metric,
                "positive",
            )
        for metric in ("centered_cosine", "mrr"):
            self.add(
                "non_root_visible_parent",
                "full_oracle_minus_uniform",
                metric,
                "positive",
            )
        decision = self.decide()
        self.assertEqual(decision["primary_category"], "C")
        self.assertTrue(decision["root_or_anchor_boundary_sufficient"])
        self.assertFalse(decision["material_metric_conflict"])

    def test_generation_pattern_can_select_d(self) -> None:
        self.add_clean_same_module_positive()
        for comparison in (
            "oracle_without_generation_minus_full_oracle",
            "full_oracle_minus_generation_only",
        ):
            self.add("all_nodes", comparison, "centered_cosine", "positive")
            self.add("all_nodes", comparison, "mrr", "positive")
        decision = self.decide()
        self.assertEqual(decision["primary_category"], "D")
        self.assertTrue(decision["generation_constraint_mismatch_supported"])

    def test_clean_causal_semantic_pattern_can_select_e(self) -> None:
        self.add_clean_same_module_positive()
        decision = self.decide()
        self.assertEqual(decision["primary_category"], "E")
        self.assertTrue(
            decision["causal_graph_semantic_constraint_mismatch_supported"]
        )
        self.assertTrue(decision["s0_semantic_support_graph_allowed"])

    def test_real_conflict_pattern_must_select_f_not_c(self) -> None:
        for metric in ("gold_cosine", "centered_cosine", "prediction_margin"):
            self.add(
                "root",
                "full_oracle_minus_reversed_full",
                metric,
                "adverse",
            )
            self.add(
                "root",
                "full_oracle_minus_uniform",
                metric,
                "adverse",
            )
        for metric in ("centered_cosine", "prediction_margin", "mrr"):
            self.add(
                "non_root_visible_parent",
                "full_oracle_minus_shuffled_full",
                metric,
                "positive",
            )
        for comparison in (
            "full_oracle_minus_uniform",
            "full_oracle_minus_raw_correlation",
        ):
            self.add(
                "non_root_visible_parent",
                comparison,
                "centered_cosine",
                "positive",
            )
            self.add(
                "non_root_visible_parent",
                comparison,
                "mrr",
                "positive",
            )
            self.add(
                "non_root_visible_parent",
                comparison,
                "gold_cosine",
                "adverse",
            )
            self.add(
                "non_root_visible_parent",
                comparison,
                "prediction_margin",
                "adverse",
            )
        self.add(
            "all_nodes",
            "same_module_minus_shuffled_full",
            "centered_cosine",
            "positive",
        )
        for metric in ("centered_cosine", "mrr", "recall_at_5"):
            self.add(
                "all_nodes",
                "same_module_minus_uniform",
                metric,
                "positive",
            )
        for metric in ("gold_cosine", "prediction_margin"):
            self.add(
                "all_nodes",
                "same_module_minus_uniform",
                metric,
                "adverse",
            )

        decision = self.decide()
        self.assertEqual(decision["primary_category"], "F")
        self.assertTrue(decision["material_metric_conflict"])
        self.assertFalse(decision["same_module_positive_diagnostic"])
        self.assertFalse(decision["root_or_anchor_boundary_sufficient"])
        self.assertTrue(decision["broader_solver_or_metric_failure_supported"])
        self.assertFalse(decision["s0_semantic_support_graph_allowed"])


if __name__ == "__main__":
    unittest.main()
