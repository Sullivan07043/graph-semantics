"""Model-free regression tests for the frozen Task 3 E0' mechanics."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from task3_v2.scripts import e0_core as core  # noqa: E402


GRAPH_DIR = (
    REPO_ROOT / "task3_v2" / "experiments" / "e0_oracle_bridge" / "graphs"
)

FOLDS = {
    "graph_00": [
        ["node_000", "node_007", "node_012", "node_018"],
        ["node_001", "node_008", "node_013", "node_019"],
        ["node_002", "node_009", "node_014", "node_015"],
        ["node_003", "node_005", "node_010", "node_016"],
        ["node_004", "node_006", "node_011", "node_017"],
    ],
    "graph_01": [
        ["node_000", "node_006", "node_013", "node_019"],
        ["node_001", "node_007", "node_014", "node_017"],
        ["node_002", "node_008", "node_012", "node_018"],
        ["node_003", "node_009", "node_015", "node_016"],
        ["node_004", "node_005", "node_010", "node_011"],
    ],
    "graph_02": [
        ["node_000", "node_008", "node_014", "node_019"],
        ["node_001", "node_009", "node_015", "node_017"],
        ["node_002", "node_010", "node_013", "node_018"],
        ["node_003", "node_006", "node_012", "node_016"],
        ["node_004", "node_005", "node_007", "node_011"],
    ],
}

PERMUTATION_SEEDS = {
    "graph_00": list(range(74101, 74121)),
    "graph_01": list(range(74201, 74221)),
    "graph_02": list(range(74301, 74321)),
}


class GraphValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.specs = core.load_graph_specs(GRAPH_DIR)

    def test_fixed_collection_and_required_motifs(self) -> None:
        self.assertEqual([spec["graph_id"] for spec in self.specs], list(core.EXPECTED_GRAPH_IDS))
        self.assertEqual([len(spec["edges"]) for spec in self.specs], [26, 29, 29])
        for spec in self.specs:
            stats = core.validate_graph_spec(spec)
            self.assertEqual(stats["nodes"], 20)
            self.assertEqual(stats["modules"], 4)
            self.assertLessEqual(stats["max_indegree"], 3)
            self.assertGreater(stats["chain_count"], 0)
            self.assertGreater(stats["fork_count"], 0)
            self.assertGreater(stats["collider_count"], 0)
            self.assertGreater(stats["mediator_count"], 0)
            self.assertTrue(stats["is_dag"])
            self.assertGreaterEqual(stats["coefficient_min"], 0.4)
            self.assertLessEqual(stats["coefficient_max"], 0.9)

    def test_rejects_bad_coefficient_topology_and_indegree(self) -> None:
        bad_coefficient = copy.deepcopy(self.specs[0])
        bad_coefficient["edges"][0]["coefficient"] = 0.39
        with self.assertRaises(core.ValidationError):
            core.validate_graph_spec(bad_coefficient)

        bad_topology = copy.deepcopy(self.specs[0])
        bad_topology["topological_order"] = list(
            reversed(bad_topology["topological_order"])
        )
        with self.assertRaises(core.ValidationError):
            core.validate_graph_spec(bad_topology)

        bad_indegree = copy.deepcopy(self.specs[0])
        bad_indegree["edges"][-1] = {
            "source": "node_008",
            "target": "node_017",
            "coefficient": 0.44,
        }
        with self.assertRaises(core.ValidationError):
            core.validate_graph_spec(bad_indegree)

    def test_folds_partition_each_graph_exactly_once(self) -> None:
        validated = core.validate_fold_assignments(FOLDS, self.specs)
        self.assertEqual(set(validated), set(core.EXPECTED_GRAPH_IDS))
        for folds in validated.values():
            self.assertEqual(len(folds), 5)
            self.assertEqual({len(fold) for fold in folds}, {4})
            self.assertEqual(
                sorted(node for fold in folds for node in fold),
                list(core.EXPECTED_NODE_IDS),
            )

        duplicate = copy.deepcopy(FOLDS["graph_00"])
        duplicate[0][0] = duplicate[1][0]
        with self.assertRaises(core.ValidationError):
            core.validate_folds(duplicate)


class SCMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = core.load_graph_json(GRAPH_DIR / "industrial_cooling_system.json")

    def test_scm_is_deterministic_and_has_fixed_splits(self) -> None:
        first = core.generate_scm(self.spec, data_seed=9101)
        second = core.generate_scm(self.spec, data_seed=9101)
        different = core.generate_scm(self.spec, data_seed=9102)

        np.testing.assert_array_equal(first["raw"], second["raw"])
        np.testing.assert_array_equal(first["standardized"], second["standardized"])
        self.assertFalse(np.array_equal(first["raw"], different["raw"]))
        self.assertEqual(first["raw"].shape, (2000, 20))
        self.assertEqual(first["train"].shape, (1200, 20))
        self.assertEqual(first["dev"].shape, (400, 20))
        self.assertEqual(first["test"].shape, (400, 20))
        self.assertEqual(
            first["split_indices"],
            {"train": (0, 1200), "dev": (1200, 1600), "test": (1600, 2000)},
        )
        np.testing.assert_allclose(first["train"].mean(axis=0), 0.0, atol=2e-15)
        np.testing.assert_allclose(first["train"].std(axis=0), 1.0, atol=2e-15)
        np.testing.assert_allclose(
            first["dev"],
            (first["raw"][1200:1600] - first["train_mean"]) / first["train_std"],
            rtol=0.0,
            atol=0.0,
        )
        expected_weights = {
            (edge["source"], edge["target"]): edge["coefficient"]
            for edge in self.spec["edges"]
        }
        self.assertEqual(first["true_weights"], expected_weights)

    def test_train_only_zscore_does_not_fit_on_later_rows(self) -> None:
        matrix = np.arange(60, dtype=np.float64).reshape(20, 3)
        shifted = matrix.copy()
        shifted[10:] += 100_000.0
        standardized, mean, scale = core.train_only_zscore(shifted, train_size=10)
        np.testing.assert_array_equal(mean, matrix[:10].mean(axis=0))
        np.testing.assert_array_equal(scale, matrix[:10].std(axis=0))
        np.testing.assert_allclose(standardized[:10].mean(axis=0), 0.0, atol=1e-15)


class GraphArmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = core.load_graph_json(GRAPH_DIR / "industrial_cooling_system.json")

    def test_seeded_permutations_are_exact_relabelings(self) -> None:
        permutations = core.generate_permutations(
            20, PERMUTATION_SEEDS["graph_00"]
        )
        self.assertEqual(len(permutations), 20)
        self.assertEqual(len({tuple(p.tolist()) for p in permutations}), 20)
        original = core.adjacency_matrix(self.spec, weighted=True)
        for permutation in permutations:
            matrix = core.permutation_matrix(permutation)
            shuffled = core.permute_adjacency(original, permutation)
            np.testing.assert_array_equal(shuffled, matrix.T @ original @ matrix)
            self.assertTrue(
                core.validate_shuffled_adjacency(
                    original, shuffled, permutation
                )
            )

        relabeled = core.permute_graph(self.spec, permutations[0])
        relabeled_adjacency = core.adjacency_matrix(relabeled, weighted=True)
        np.testing.assert_array_equal(
            relabeled_adjacency,
            core.permute_adjacency(original, permutations[0]),
        )

    def test_reversed_graph_is_exact_transpose_and_dag(self) -> None:
        reversed_spec = core.reverse_graph(self.spec)
        original = core.adjacency_matrix(self.spec, weighted=True)
        reversed_adjacency = core.adjacency_matrix(
            reversed_spec,
            weighted=True,
            validate_design_constraints=False,
        )
        np.testing.assert_array_equal(reversed_adjacency, original.T)
        positions = {
            node: i for i, node in enumerate(reversed_spec["topological_order"])
        }
        self.assertTrue(
            all(
                positions[edge["source"]] < positions[edge["target"]]
                for edge in reversed_spec["edges"]
            )
        )


class MetricTests(unittest.TestCase):
    def test_rank_metrics_and_hungarian_node_hits(self) -> None:
        candidates = np.eye(6, dtype=np.float64)
        predictions = np.stack(
            [
                candidates[0],
                0.8 * candidates[0] + 0.6 * candidates[1],
                candidates[0],
            ]
        )
        metrics = core.rank_metrics(predictions, candidates, [0, 1, 5])
        np.testing.assert_array_equal(metrics["rank"], [1, 2, 6])
        np.testing.assert_array_equal(metrics["recall_at_1"], [1, 0, 0])
        np.testing.assert_array_equal(metrics["recall_at_5"], [1, 1, 0])
        np.testing.assert_array_equal(metrics["exact_decode"], [1, 0, 0])
        self.assertAlmostEqual(metrics["mrr"], (1.0 + 0.5 + 1.0 / 6.0) / 3.0)
        self.assertAlmostEqual(metrics["mean_recall_at_1"], 1.0 / 3.0)
        self.assertAlmostEqual(metrics["mean_recall_at_5"], 2.0 / 3.0)
        self.assertAlmostEqual(
            metrics["mean_gold_embedding_cosine"], (1.0 + 0.6 + 0.0) / 3.0
        )

        truth = np.eye(3, dtype=np.float64)
        swapped = truth[[1, 0, 2]]
        hits = core.hungarian_match_hits(swapped, truth)
        np.testing.assert_array_equal(hits, [0, 0, 1])
        self.assertAlmostEqual(core.hungarian_match(swapped, truth)["match_acc"], 1 / 3)

    def test_metrics_reject_zero_vectors(self) -> None:
        with self.assertRaises(core.ValidationError):
            core.rank_metrics(np.zeros((1, 2)), np.eye(2), [0])


class BootstrapTests(unittest.TestCase):
    @staticmethod
    def records() -> list[dict[str, object]]:
        return [
            {
                "graph_id": f"graph_{graph}",
                "fold": fold,
                "node_id": f"node_{node}",
                "delta": (graph - 1) * 0.2 + fold * 0.01 + node * 0.001,
            }
            for graph in range(3)
            for fold in range(5)
            for node in range(4)
        ]

    def test_hierarchical_bootstrap_is_deterministic_and_order_invariant(self) -> None:
        records = self.records()
        first = core.hierarchical_bootstrap(
            records, draws=1000, seed=88173, return_draws=True
        )
        second = core.hierarchical_bootstrap(
            list(reversed(records)), draws=1000, seed=88173, return_draws=True
        )
        np.testing.assert_array_equal(
            first["bootstrap_draws"], second["bootstrap_draws"]
        )
        self.assertEqual(first["mean"], second["mean"])
        self.assertEqual(first["n_graphs"], 3)
        self.assertEqual(first["n_folds"], 15)
        self.assertEqual(first["n_nodes"], 60)
        self.assertLessEqual(first["ci_low"], first["mean"])
        self.assertGreaterEqual(first["ci_high"], first["mean"])

    def test_constant_deltas_and_duplicate_leaf_validation(self) -> None:
        records = self.records()
        constant = [{**record, "delta": 1.0} for record in records]
        result = core.hierarchical_bootstrap(constant, draws=100, seed=7)
        self.assertEqual(result["mean"], 1.0)
        self.assertEqual(result["ci_low"], 1.0)
        self.assertEqual(result["ci_high"], 1.0)
        self.assertEqual(result["paired_win_rate"], 1.0)
        self.assertEqual(result["bootstrap_positive_rate"], 1.0)

        with self.assertRaises(core.ValidationError):
            core.hierarchical_bootstrap(records + [records[0]], draws=10)


class JsonHelperTests(unittest.TestCase):
    def test_canonical_json_hash_and_file_roundtrip(self) -> None:
        left = {"b": np.asarray([2, 3]), "a": 1}
        right = {"a": 1, "b": [2, 3]}
        self.assertEqual(core.canonical_json_dumps(left), core.canonical_json_dumps(right))
        self.assertEqual(core.sha256_json(left), core.sha256_json(right))

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "nested" / "record.json"
            core.write_json(target, left)
            self.assertEqual(core.load_json(target), right)
            expected = hashlib.sha256(target.read_bytes()).hexdigest()
            self.assertEqual(core.sha256_file(target), expected)
            self.assertTrue(target.read_text(encoding="utf-8").endswith("\n"))
            json.loads(target.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
