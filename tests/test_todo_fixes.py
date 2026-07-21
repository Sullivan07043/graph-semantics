import os
import tempfile
import unittest

import numpy as np
import torch

import graph as graph_module
import optimize
from pipeline_L3_v1 import lora
from pipeline_L3_v1.l3_train import identity_loss, identity_safe_gradients, relational_loss
from pipeline_v4 import core
from pipeline_v4 import l2_modules as LM


class SwapNeg(torch.nn.Module):
    """A deliberately non-vector-negation transform for item-target tests."""

    def forward(self, x):
        return torch.nn.functional.normalize(torch.roll(x, shifts=1, dims=-1), dim=-1)


def _tiny_problem():
    g = graph_module.Graph(["factor"], ["masked", "visible_a", "visible_b"], [
        ("factor", "masked"), ("factor", "visible_a"), ("factor", "visible_b")])
    W = {("factor", "masked"): 0.7, ("factor", "visible_a"): 0.8,
         ("factor", "visible_b"): 0.6}
    names = list(g.observed)
    P = np.array([[0.0, 0.55, -0.45],
                  [0.55, 0.0, 0.02],
                  [-0.45, 0.02, 0.0]], dtype=float)
    visible = {"visible_a": np.array([1.0, 0.0, 0.0], dtype=float),
               "visible_b": np.array([0.0, 1.0, 0.0], dtype=float)}
    return g, W, (names, P), visible


class StructuredObjectiveTests(unittest.TestCase):
    def test_generation_is_mean_and_per_node_multiplier_precedes_mean(self):
        g = graph_module.Graph([], ["parent", "a", "b"],
                               [("parent", "a"), ("parent", "b")])
        W = {("parent", "a"): 0.0, ("parent", "b"): 0.0}
        A = {"parent": np.array([0.0]), "a": np.array([1.0]), "b": np.array([3.0])}
        ctx = core.build_ctx(
            g, W, W, A, [], 1, 0, "cpu", 0.0, 0.0, None,
            0.0, None, 0.5, 0.0, None, None, None)
        At = ctx["At"]
        emb = lambda n: At[n]
        wt = lambda edge: ctx["wt_const"][edge]
        loss = core.step_loss(ctx, emb, wt, {}, None, 0.0, 0.0, 0.0, 0.0)
        self.assertAlmostEqual(float(loss), 5.0, places=6)  # mean([1^2, 3^2])
        nw = {"gen": torch.tensor([2.0, 4.0]), "resnorm": None, "anchor": None,
              "node": torch.ones(3), "norm": torch.zeros(0), "item": torch.zeros(0)}
        weighted = core.step_loss(
            ctx, emb, wt, {}, None, 0.0, 0.0, 0.0, 0.0, nw=nw)
        self.assertAlmostEqual(float(weighted), 19.0, places=6)

    def test_negative_candidate_requires_negop_and_never_uses_vector_negation(self):
        g, _, pc, visible = _tiny_problem()
        # Remove the positive candidate, leaving one reliable negative local candidate.
        pc_negative = (pc[0], pc[1].copy())
        pc_negative[1][0, 1] = pc_negative[1][1, 0] = 0.01
        without = core.prepare_item_identity(
            g, visible, pc_negative, 400, neg_op=None, device="cpu")
        self.assertEqual(without["nodes"], [])
        with_op = core.prepare_item_identity(
            g, visible, pc_negative, 400, neg_op=SwapNeg(), device="cpu")
        self.assertEqual(with_op["nodes"], ["masked"])
        np.testing.assert_allclose(with_op["targets"][0], [0.0, 0.0, 1.0], atol=1e-7)
        self.assertFalse(np.allclose(with_op["targets"][0], -visible["visible_b"]))

    def test_two_pole_views_are_normalized_before_consensus(self):
        observed = ["masked", "positive_a", "positive_b", "negative"]
        g = graph_module.Graph(
            ["factor"], observed, [("factor", node) for node in observed])
        corr = np.array([
            [0.0, 0.90, 0.80, -0.20],
            [0.90, 0.0, 0.0, 0.0],
            [0.80, 0.0, 0.0, 0.0],
            [-0.20, 0.0, 0.0, 0.0],
        ])
        visible = {
            "positive_a": np.array([1.0, 0.0, 0.0]),
            "positive_b": np.array([1.0, 0.0, 0.0]),
            # SwapNeg maps this to the independent negative-pole semantic view.
            "negative": np.array([1.0, 0.0, 0.0]),
        }
        info = core.prepare_item_identity(
            g, visible, (observed, corr), 400, neg_op=SwapNeg(), device="cpu")
        expected = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
        np.testing.assert_allclose(info["targets"][0], expected, atol=1e-7)

    def test_single_parent_star_does_not_duplicate_local_item_as_profile(self):
        g, _, corr, visible = _tiny_problem()
        info = core.prepare_item_identity(
            g, visible, corr, 400, neg_op=SwapNeg(), device="cpu")
        self.assertEqual(info["nodes"], ["masked"])
        self.assertEqual(info["profile_nodes"], [])

    def test_reliable_item_anchor_uses_unit_strength(self):
        g = graph_module.Graph([], ["masked"], [])
        item_info = {
            "nodes": ["masked"],
            "targets": np.array([[1.0, 0.0]], dtype=np.float32),
            "confidence": np.array([0.01], dtype=np.float32),
            "features": np.zeros((1, 7), dtype=np.float32),
            "profile_nodes": [],
        }
        ctx = core.build_ctx(
            g, {}, {}, {}, ["masked"], 2, 0, "cpu", 0.0, 0.0, None,
            0.0, None, 0.5, 0.0, None, None, None, item_info=item_info)
        E = {"masked": torch.tensor([0.0, 1.0])}
        loss = core.step_loss(
            ctx, lambda n: E[n], lambda edge: None, E, None,
            0.0, 0.0, 0.0, 0.0)
        self.assertAlmostEqual(float(loss), 1.0, places=7)

    def test_masked_target_text_cannot_change_solver_input_or_prediction(self):
        g, W, pc, _ = _tiny_problem()

        def fake_embed(text):
            # Only the masked text changes between calls. Visible label embeddings are fixed.
            table = {"visible A": np.array([1.0, 0.0, 0.0]),
                     "visible B": np.array([0.0, 1.0, 0.0]),
                     "masked original": np.array([0.0, 0.0, 1.0]),
                     "completely changed hidden text": np.array([-1.0, -1.0, 0.0])}
            value = table[text].astype(float)
            return value / (np.linalg.norm(value) + 1e-12)

        def run(masked_text):
            labels = {"masked": masked_text, "visible_a": "visible A",
                      "visible_b": "visible B"}
            all_embeddings = {node: fake_embed(text) for node, text in labels.items()}
            # This is the masking boundary: the target embedding is not passed to the solver.
            visible = {n: all_embeddings[n] for n in ("visible_a", "visible_b")}
            item_info = core.prepare_item_identity(
                g, visible, pc, 400, neg_op=None, device="cpu")
            out, _ = core.solve_unrolled(
                g, W, visible, d=3, weight_module=None, K=8, inner_lr=0.02,
                partial_corr=pc, n_samples=400, item_info=item_info,
                train=False, device="cpu")
            return item_info, out["masked"]

        item_a, pred_a = run("masked original")
        item_b, pred_b = run("completely changed hidden text")
        self.assertEqual(item_a["nodes"], item_b["nodes"])
        np.testing.assert_array_equal(item_a["features"], item_b["features"])
        np.testing.assert_array_equal(item_a["targets"], item_b["targets"])
        np.testing.assert_array_equal(item_a["profile_hi"], item_b["profile_hi"])
        np.testing.assert_array_equal(item_a["profile_lo"], item_b["profile_lo"])
        np.testing.assert_array_equal(pred_a, pred_b)

    def test_leave_pair_out_removes_pc1_part_whole_negative_artifact(self):
        rng = np.random.default_rng(12)
        n = 4000
        factor = rng.normal(size=n)
        X = np.stack([factor + rng.normal(scale=0.8, size=n) for _ in range(6)], axis=1)
        observed = [f"q{i}" for i in range(X.shape[1])]
        g = graph_module.Graph(["f"], observed, [("f", o) for o in observed])
        oi = {o: i for i, o in enumerate(observed)}
        _, score = g.estimate_weights(X, oi)
        _, current = optimize.partial_residual_corr(g, X, oi, score)
        leave_out = optimize.leave_pair_out_residual_pairs(g, X, oi)
        values = {(a, b): rho for a, b, rho in leave_out["pairs"]}
        self.assertLess(float(np.mean(current[np.triu_indices(6, 1)])), 0.0)
        self.assertGreater(values[("q0", "q1")], 0.0)
        self.assertGreater(sum(rho > 0 for rho in values.values()),
                           sum(rho < 0 for rho in values.values()))

    def test_weight_module_none_is_supported(self):
        g, W, pc, visible = _tiny_problem()
        out, tensors = core.solve_unrolled(
            g, W, visible, d=3, weight_module=None, K=2, partial_corr=pc,
            n_samples=400, device="cpu")
        self.assertIn("masked", out)
        self.assertIn("masked", tensors)


class IndependenceTests(unittest.TestCase):
    def test_data_conflict_only_removes_zero_constraint(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=400)
        y = x + rng.normal(scale=0.05, size=400)
        X = np.stack([x, y], axis=1)
        g = graph_module.Graph(["left", "right", "no_data"], ["a", "b"],
                               [("left", "a"), ("right", "b")])
        _, scores = g.estimate_weights(X, {"a": 0, "b": 1})
        info = g.reconcile_independent_pairs(X, {"a": 0, "b": 1}, scores)
        self.assertIn(("a", "b"), g.independent_pairs())
        self.assertNotIn(("a", "b"), info["pairs"])
        self.assertGreater(info["conflict_count"], 0)
        # A pair with an unavailable latent representation retains the graph-only behavior.
        self.assertIn(("left", "no_data"), info["pairs"])
        self.assertEqual(g.edges, [("left", "a"), ("right", "b")])

    def test_fixed_fixed_zero_pairs_calibrate_but_do_not_dilute_active_mean(self):
        g = graph_module.Graph([], ["a", "b", "c", "d"], [])
        A = {"a": np.array([1.0, 0.0]), "c": np.array([0.0, 1.0])}
        free = ["b", "d"]
        ctx = core.build_ctx(
            g, {}, {}, A, free, 2, 0, "cpu", 0.0, 0.0, None,
            0.0, None, 0.5, 0.0, None, None, None,
            independent_info={"pairs": g.independent_pairs()})
        self.assertEqual(ctx["zero_pair_counts"]["fixed_fixed"], 1)
        self.assertNotIn(("a", "c"), ctx["zp_pairs"])
        self.assertEqual(ctx["zero_pair_counts"]["active"], 5)


class TrainingMechanismTests(unittest.TestCase):
    def test_identity_loss_is_zero_at_frozen_initialization(self):
        H0 = torch.nn.functional.normalize(torch.tensor(
            [[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]]), dim=1)
        loss = identity_loss(H0.clone(), H0, [(0, 1), (1, 0), (1, 2)])
        self.assertAlmostEqual(float(loss), 0.0, places=7)

    def test_relational_loss_preserves_gram_and_blocks_cross_collapse(self):
        H0 = torch.nn.functional.normalize(torch.tensor(
            [[1.0, 0.0, 0.0], [0.9, 0.3, 0.0],
             [0.0, 1.0, 0.0], [0.0, 0.9, 0.3]]), dim=1)
        same = [[(0, 1)], [(2, 3)]]
        cross = [[(0, 2), (1, 3)]]
        self.assertAlmostEqual(
            float(relational_loss(H0, H0, same, cross)), 0.0, places=7)

        # A global orthogonal transform preserves the adapted-adapted Gram matrix.
        Q = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        rotated = H0 @ Q
        self.assertAlmostEqual(
            float(relational_loss(rotated, H0, same, cross)), 0.0, places=7)

        collapsed = H0.clone()
        collapsed[2] = torch.nn.functional.normalize(H0[2] + 0.7 * H0[0], dim=0)
        collapsed[3] = torch.nn.functional.normalize(H0[3] + 0.7 * H0[1], dim=0)
        self.assertGreater(float(relational_loss(collapsed, H0, same, cross)), 0.0)

        separated = H0.clone()
        separated[2:] = -separated[2:]
        cross_only = relational_loss(separated, H0, [], cross)
        self.assertAlmostEqual(float(cross_only), 0.0, places=7)

    def test_pcgrad_removes_task_conflict_with_safeguard(self):
        p = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
        task = p[0] - p[1]
        safe = (p[0] - 2.0 * p[1]) ** 2
        gradients, _ = identity_safe_gradients(task, safe, [p])
        safe_grad = torch.tensor([-2.0, 2.0])
        self.assertGreaterEqual(float((gradients[0] * safe_grad).sum()), -1e-7)

    def test_chunked_k120_forward_matches_full_but_backward_is_truncated(self):
        g, W, pc, visible = _tiny_problem()
        item_info = core.prepare_item_identity(
            g, visible, pc, 400, neg_op=None, device="cpu")
        feats = torch.tensor(
            LM.node_features(g, W, set(visible), item_info=item_info),
            dtype=torch.float32)
        torch.manual_seed(9)
        full_module = LM.WeightNet(hid=8)
        with torch.no_grad():
            full_module.net[-1].weight.normal_(0.0, 0.03)
            full_module.net[-1].bias.normal_(0.0, 0.03)
        chunk_module = LM.WeightNet(hid=8)
        chunk_module.load_state_dict(full_module.state_dict())

        _, full_tensors = core.solve_unrolled(
            g, W, visible, d=3, weight_module=full_module, K=120, inner_lr=0.01,
            partial_corr=pc, n_samples=400, item_info=item_info, feats=feats,
            train=True, truncation_steps=None, device="cpu")
        _, chunk_tensors = core.solve_unrolled(
            g, W, visible, d=3, weight_module=chunk_module, K=120, inner_lr=0.01,
            partial_corr=pc, n_samples=400, item_info=item_info, feats=feats,
            train=True, truncation_steps=60, device="cpu")
        torch.testing.assert_close(full_tensors["masked"], chunk_tensors["masked"],
                                   rtol=0.0, atol=0.0)

        full_grad = torch.autograd.grad(
            full_tensors["masked"].square().sum(), tuple(full_module.parameters()),
            allow_unused=True)
        chunk_grad = torch.autograd.grad(
            chunk_tensors["masked"].square().sum(), tuple(chunk_module.parameters()),
            allow_unused=True)

        def flatten(grads, params):
            return torch.cat([(g_ if g_ is not None else torch.zeros_like(p)).reshape(-1)
                              for g_, p in zip(grads, params)])

        gf = flatten(full_grad, tuple(full_module.parameters()))
        gc = flatten(chunk_grad, tuple(chunk_module.parameters()))
        self.assertTrue(torch.isfinite(gf).all() and torch.isfinite(gc).all())
        self.assertGreater(float(gc.abs().sum()), 0.0)
        self.assertFalse(torch.allclose(gf, gc, rtol=1e-5, atol=1e-8))

    def test_legacy_checkpoint_fails_clearly(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "legacy.pt")
            torch.save({"kind": "mlp", "state": {}}, path)
            with self.assertRaisesRegex(RuntimeError, "Incompatible L2 checkpoint"):
                LM.load(path)

    def test_previous_item_objective_checkpoint_fails_clearly(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "version3.pt")
            torch.save({
                "format": LM.CHECKPOINT_FORMAT,
                "version": 3,
                "kind": "mlp",
                "feature_dim": LM.FEATURE_DIM,
                "terms": list(LM.TERMS),
                "state": {},
            }, path)
            with self.assertRaisesRegex(RuntimeError, "two-pole/unit-item-anchor"):
                LM.load(path)

    def test_old_l3_checkpoint_version_fails_clearly(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "old_l3.pt")
            torch.save({"format": lora.CHECKPOINT_FORMAT, "version": 2}, path)
            with self.assertRaisesRegex(RuntimeError, "Incompatible L3 checkpoint"):
                lora._validated_checkpoint(path)


if __name__ == "__main__":
    unittest.main()
