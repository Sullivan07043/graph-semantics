from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from task3_v2.scripts import e0_orientation_audit as audit  # noqa: E402


class E0OrientationAuditTests(unittest.TestCase):
    def test_json_source_to_target_adjacency(self) -> None:
        spec = audit.build_chain_spec()
        self.assertEqual(audit.edge_pairs(spec), [("A", "B"), ("B", "C")])
        np.testing.assert_array_equal(
            audit.source_by_target_adjacency(spec),
            np.asarray(
                [
                    [0.0, 0.7, 0.0],
                    [0.0, 0.0, 0.4],
                    [0.0, 0.0, 0.0],
                ]
            ),
        )

    def test_adapter_graph_parents_and_children(self) -> None:
        context = audit.build_adapter_context(audit.build_chain_spec())
        graph = context["graph"]
        self.assertEqual(graph.edges, [("A", "B"), ("B", "C")])
        self.assertEqual(
            {node: graph.parents(node) for node in audit.NODE_IDS},
            {"A": [], "B": ["A"], "C": ["B"]},
        )
        self.assertEqual(
            {node: graph.children(node) for node in audit.NODE_IDS},
            {"A": ["B"], "B": ["C"], "C": []},
        )
        self.assertEqual(set(context["weights"]), {("A", "B"), ("B", "C")})
        self.assertEqual(
            {edge: float(value) for edge, value in context["weights"].items()},
            {("A", "B"): 0.7, ("B", "C"): 0.4},
        )

    def test_generation_parses_incoming_and_outgoing_roles(self) -> None:
        result = audit.generation_audit()
        self.assertEqual(result["incoming_edges"], [["A", "B"]])
        self.assertEqual(result["outgoing_edges"], [["B", "C"]])
        self.assertEqual(result["generated_nodes"], ["B", "C"])
        self.assertEqual(
            result["generation_parents"],
            {"B": ["A"], "C": ["B"]},
        )
        np.testing.assert_allclose(
            np.asarray(result["incoming_gradient"]),
            np.asarray([-1.4, 0.0]),
            atol=1e-7,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            np.asarray(result["outgoing_gradient"]),
            np.asarray([0.0, -0.8]),
            atol=1e-7,
            rtol=0.0,
        )

    def test_exact_z_b_generation_gradient(self) -> None:
        result = audit.generation_audit()
        self.assertAlmostEqual(result["full_loss"], 1.49, places=7)
        self.assertEqual(result["expected_full_loss"], 1.49)
        gradient = np.asarray(result["gradient_dL_dz_B"])
        np.testing.assert_allclose(
            gradient,
            np.asarray([-1.4, -0.8]),
            atol=1e-7,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            gradient,
            np.asarray(result["incoming_gradient"])
            + np.asarray(result["outgoing_gradient"]),
            atol=1e-7,
            rtol=0.0,
        )

    def test_als_matches_ridge_aware_analytic_solution(self) -> None:
        result = audit.als_audit()
        np.testing.assert_allclose(
            np.asarray(result["frozen_solver_value"]),
            np.asarray(result["implemented_ridge_closed_form"]),
            atol=1e-12,
            rtol=0.0,
        )
        self.assertLessEqual(result["max_abs_error_vs_implemented"], 1e-12)

    def test_deliberate_transpose_must_fail(self) -> None:
        transposed = audit.build_chain_spec(transpose=True)
        with self.assertRaisesRegex(
            audit.OrientationAuditError,
            "ordered edge pairs mismatch",
        ):
            audit.assert_expected_chain_orientation(transposed)

    def test_structured_audit_verdict(self) -> None:
        result = audit.run_audit()
        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["negative_control"]["rejected"])
        self.assertFalse(result["verdict"]["orientation_interface_bug"])
        self.assertFalse(result["verdict"]["rerun_e0_prime_for_orientation_fix"])
        self.assertTrue(
            result["unit_weight_reverse_symmetry"][
                "exactly_equal_within_tolerance"
            ]
        )


if __name__ == "__main__":
    unittest.main()
