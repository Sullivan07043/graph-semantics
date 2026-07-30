"""Lightweight tests for the training-free external baselines."""
import os
import sys
import unittest

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from external_baselines import (
    feature_propagation,
    format_latent_markov_context,
    latent_markov_context,
    loading_centroid,
    loading_centroid_applicability,
)
from graph import Graph


class FeaturePropagationTest(unittest.TestCase):
    def test_known_nodes_are_clamped_and_direction_is_ignored(self):
        graph = Graph([], ["left", "middle", "right"], [
            ("middle", "left"),
            ("middle", "right"),
        ])
        known = {
            "left": np.array([2.0, -1.0]),
            "right": np.array([4.0, 3.0]),
        }

        result = feature_propagation(graph, known)

        np.testing.assert_array_equal(result["left"], known["left"])
        np.testing.assert_array_equal(result["right"], known["right"])
        # middle has degree two; both endpoint degrees are one.
        np.testing.assert_allclose(
            result["middle"],
            (known["left"] + known["right"]) / np.sqrt(2.0),
        )

    def test_unanchored_components_and_isolates_use_fallback(self):
        graph = Graph(
            [],
            ["known", "reached", "unanchored_a", "unanchored_b", "isolated"],
            [
                ("known", "reached"),
                ("unanchored_a", "unanchored_b"),
            ],
        )
        known = {"known": np.array([3.0, -2.0])}

        result = feature_propagation(graph, known)

        np.testing.assert_allclose(result["reached"], known["known"])
        for node in ("unanchored_a", "unanchored_b", "isolated"):
            np.testing.assert_allclose(result[node], np.zeros(2))


class LoadingCentroidTest(unittest.TestCase):
    def test_absolute_loading_weights_and_uniform_fallback(self):
        graph = Graph(["latent"], ["a", "b"], [
            ("latent", "a"),
            ("latent", "b"),
        ])
        visible = {
            "a": np.array([1.0, 0.0]),
            "b": np.array([0.0, 1.0]),
        }

        weighted = loading_centroid(
            graph,
            visible,
            W={("latent", "a"): -2.0, ("latent", "b"): 1.0},
        )
        uniform = loading_centroid(graph, visible, W={})

        np.testing.assert_allclose(weighted["latent"], [2.0 / 3.0, 1.0 / 3.0])
        np.testing.assert_allclose(uniform["latent"], [0.5, 0.5])

    def test_direct_children_precede_descendants_and_empty_uses_fallback(self):
        graph = Graph(
            ["top", "nested", "empty"],
            ["direct", "deep"],
            [
                ("top", "direct"),
                ("top", "nested"),
                ("nested", "deep"),
                ("empty", "deep"),
            ],
        )
        visible = {
            "direct": np.array([2.0, 0.0]),
            "deep": np.array([0.0, 4.0]),
        }

        result = loading_centroid(graph, visible, W={})

        np.testing.assert_allclose(result["top"], visible["direct"])
        np.testing.assert_allclose(result["nested"], visible["deep"])
        # "empty" has a direct observed child in the graph, so it is not empty.
        np.testing.assert_allclose(result["empty"], visible["deep"])

        only_direct = loading_centroid(graph, {"direct": visible["direct"]}, W={})
        np.testing.assert_allclose(only_direct["nested"], visible["direct"])
        np.testing.assert_allclose(only_direct["empty"], visible["direct"])


    def test_general_dag_is_not_mislabeled_as_loading_model(self):
        graph = Graph(
            ["latent"], ["cause", "indicator"],
            [("cause", "latent"), ("latent", "indicator")],
        )
        applicable, reason = loading_centroid_applicability(graph)
        self.assertFalse(applicable)
        self.assertIn("observed-source", reason)
        with self.assertRaisesRegex(ValueError, "measurement DAG"):
            loading_centroid(
                graph,
                {"cause": np.ones(2), "indicator": np.ones(2)},
                W={},
            )


class MarkovBlanketContextTest(unittest.TestCase):
    def test_only_visible_labels_and_anonymous_latent_counts_are_exposed(self):
        graph = Graph(
            ["gold target name", "gold parent name", "gold child name"],
            ["observed_parent", "visible_child", "hidden_collider", "visible_spouse"],
            [
                ("gold parent name", "gold target name"),
                ("observed_parent", "gold target name"),
                ("gold target name", "visible_child"),
                ("gold target name", "hidden_collider"),
                ("visible_spouse", "hidden_collider"),
                ("gold target name", "gold child name"),
            ],
        )
        labels = {
            "observed_parent": "visible parent description",
            "visible_child": "visible child description",
            "hidden_collider": "SECRET HIDDEN DESCRIPTION",
            "visible_spouse": "visible spouse description",
        }
        context = latent_markov_context(
            graph,
            "gold target name",
            labels,
            {"observed_parent", "visible_child", "visible_spouse"},
        )
        rendered = format_latent_markov_context(context)

        self.assertIn("visible parent description", rendered)
        self.assertIn("visible child description", rendered)
        self.assertIn("visible spouse description", rendered)
        self.assertNotIn("SECRET HIDDEN DESCRIPTION", rendered)
        self.assertNotIn("gold target name", rendered)
        self.assertNotIn("gold parent name", rendered)
        self.assertEqual(context["parents"]["anonymous_latent_count"], 1)
        self.assertEqual(context["children"]["anonymous_latent_count"], 1)


if __name__ == "__main__":
    unittest.main()
