from __future__ import annotations

import pathlib
import sys
import unittest

import numpy as np
import torch


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
V5_ROOT = REPO_ROOT / "v5"
for path in (str(REPO_ROOT), str(V5_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from task3_v2.scripts import e0_audit_solver as audit  # noqa: E402
import graph as graph_module  # noqa: E402
import l2_solver as frozen_solver  # noqa: E402


def fixture():
    graph = graph_module.Graph(
        [],
        ["A", "B", "C", "D"],
        [("A", "B"), ("B", "C")],
    )
    weights = {("A", "B"): 0.7, ("B", "C"): 0.5}
    labeled = {
        "A": np.array([1.0, 0.0, 0.0]),
        "C": np.array([0.0, 1.0, 0.0]),
    }
    partial = (
        ["B", "C"],
        np.array([[0.0, 0.2], [0.2, 0.0]], dtype=np.float64),
    )
    dependence = np.array(
        [
            [0.0, 0.8, 0.6, 0.0],
            [0.8, 0.0, 0.7, 0.0],
            [0.6, 0.7, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    bridge = {
        "obs": ["A", "B", "C", "D"],
        "dep_marg": dependence,
        "lam_upper": 0.3,
        "kappa": 0.5,
        "q": 0.0,
    }
    return graph, weights, labeled, partial, bridge


class AuditSolverTests(unittest.TestCase):
    def test_term_masks_are_exact(self):
        self.assertEqual(
            audit.normalize_term_mask("generation_only_oracle"),
            (1, 0, 0, 0, 0, 0),
        )
        self.assertEqual(
            audit.normalize_term_mask("oracle_without_generation"),
            (0, 1, 1, 1, 1, 1),
        )
        self.assertEqual(
            audit.normalize_term_mask("residual_only_oracle"),
            (0, 1, 1, 0, 0, 0),
        )
        self.assertEqual(
            audit.normalize_term_mask("independence_only_oracle"),
            (0, 0, 0, 1, 1, 0),
        )
        with self.assertRaises(audit.AuditSolverError):
            audit.normalize_term_mask("uniform")

    def test_components_sum_to_frozen_step_loss(self):
        graph, weights, labeled, partial, bridge = fixture()
        free = ["B", "D"]
        context = frozen_solver.build_ctx(
            graph,
            weights,
            dict(weights),
            labeled,
            free,
            3,
            2,
            "cpu",
            1.0,
            1.0,
            partial,
            None,
            bridge,
        )
        embeddings = {
            "B": torch.tensor([0.2, 0.4, -0.1], requires_grad=True),
            "D": torch.tensor([-0.3, 0.1, 0.5], requires_grad=True),
        }
        residuals = {
            node: torch.tensor(value, dtype=torch.float32, requires_grad=True)
            for node, value in context["Rv0"].items()
        }

        def emb(node):
            return context["At"][node] if node in context["At"] else embeddings[node]

        def wt(edge):
            return context["wt_const"][edge]

        components = audit.component_losses(
            context,
            emb,
            wt,
            embeddings,
            residuals,
            0.3,
            0.1,
        )
        decomposed = sum(components.values())
        frozen = frozen_solver.step_loss(
            context,
            emb,
            wt,
            embeddings,
            residuals,
            0.3,
            0.1,
        )
        self.assertTrue(torch.allclose(decomposed, frozen, rtol=1e-6, atol=1e-7))

    def test_three_node_orientation_and_generation_gradient(self):
        graph = graph_module.Graph([], ["A", "B", "C"], [("A", "B"), ("B", "C")])
        self.assertEqual(graph.parents("A"), [])
        self.assertEqual(graph.parents("B"), ["A"])
        self.assertEqual(graph.parents("C"), ["B"])
        self.assertEqual(graph.children("A"), ["B"])
        self.assertEqual(graph.children("B"), ["C"])
        self.assertEqual(graph.children("C"), [])

        weights = {("A", "B"): 2.0, ("B", "C"): 0.5}
        labeled = {
            "A": np.array([1.0]),
            "C": np.array([3.0]),
        }
        context = frozen_solver.build_ctx(
            graph,
            weights,
            dict(weights),
            labeled,
            ["B"],
            1,
            0,
            "cpu",
            0.0,
            0.0,
            None,
            None,
            None,
        )
        z_b = torch.tensor([0.0], requires_grad=True)
        embeddings = {"B": z_b}

        def emb(node):
            return context["At"][node] if node in context["At"] else embeddings[node]

        components = audit.component_losses(
            context,
            emb,
            lambda edge: context["wt_const"][edge],
            embeddings,
            None,
            0.0,
            0.0,
        )
        gradient = torch.autograd.grad(components["generation"], z_b)[0]
        # 2*(B-2*A) - 2*.5*(C-.5*B) = -4 - 3 = -7.
        self.assertAlmostEqual(float(gradient), -7.0, places=6)

    def test_common_residual_map_preserves_frozen_draw_order(self):
        graph, weights, labeled, _partial, _bridge = fixture()
        common = audit.make_common_initial_state(
            frozen_solver, graph, weights, labeled, 3, seed=7
        )
        rng = np.random.default_rng(7)
        for node in ("B", "C"):
            expected = rng.normal(0.0, 1e-3, 3)
            np.testing.assert_array_equal(common.residuals[node], expected)
        # Non-generated A and D are assigned only after all canonical gen nodes.
        expected_a = rng.normal(0.0, 1e-3, 3)
        np.testing.assert_array_equal(common.residuals["A"], expected_a)

    def test_canonical_full_path_matches_frozen_solver(self):
        graph, weights, labeled, partial, bridge = fixture()
        common = audit.make_common_initial_state(
            frozen_solver, graph, weights, labeled, 3, seed=3
        )
        reference, _ = frozen_solver.solve_unrolled(
            graph,
            weights,
            labeled,
            3,
            K=4,
            inner_lr=0.02,
            lam_zero=0.3,
            lam_norm=0.1,
            seed=3,
            device="cpu",
            residual=1.0,
            lam_res=1.0,
            partial_corr=partial,
            bridge=bridge,
        )
        result = audit.solve_audit(
            frozen_solver,
            graph,
            weights,
            labeled,
            3,
            common_initial=common,
            term_mask="full_oracle",
            masked_nodes=["B", "D"],
            K=4,
            inner_lr=0.02,
            lam_zero=0.3,
            lam_norm=0.1,
            seed=3,
            device="cpu",
            residual=1.0,
            lam_res=1.0,
            partial_corr=partial,
            bridge=bridge,
            canonical_full_path=True,
        )
        parity = audit.parity_summary(reference, result.embeddings, ["B", "D"])
        self.assertTrue(parity["passed"], parity)
        one_sided_zero = audit.parity_summary(
            {"x": np.zeros(3)}, {"x": np.ones(3)}, ["x"]
        )
        self.assertFalse(one_sided_zero["passed"], one_sided_zero)
        self.assertEqual(one_sided_zero["max_cosine_error"], 1.0)
        self.assertEqual(len(result.loss_terms), len(audit.TERM_NAMES))
        self.assertEqual(len(result.gradient_norms), 2 * len(audit.TERM_NAMES))

    def test_residual_only_leaves_embeddings_at_common_initial(self):
        graph, weights, labeled, partial, bridge = fixture()
        common = audit.make_common_initial_state(
            frozen_solver, graph, weights, labeled, 3, seed=5
        )
        result = audit.solve_audit(
            frozen_solver,
            graph,
            weights,
            labeled,
            3,
            common_initial=common,
            term_mask="residual_only_oracle",
            masked_nodes=["B", "D"],
            K=3,
            seed=5,
            residual=1.0,
            lam_res=1.0,
            partial_corr=partial,
            bridge=bridge,
        )
        for node in ("B", "D"):
            np.testing.assert_array_equal(
                result.initial_embeddings[node], result.final_embeddings[node]
            )
        self.assertTrue(
            all(row["total_final_gradient_norm"] == 0.0 for row in result.gradient_norms)
        )


if __name__ == "__main__":
    unittest.main()
