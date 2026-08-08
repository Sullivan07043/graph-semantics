import json
from pathlib import Path
import re
import threading
import unittest

import numpy as np

from v6.baselines.automated_interpretability import (
    AUTOINTERP_EXPLAIN_PROMPT_VERSION,
    AUTOINTERP_SIMULATE_PROMPT_VERSION,
    BaselineOutputError,
    run_autointerp,
)
from v6.baselines.delphi import (
    DELPHI_DETECT_PROMPT_VERSION,
    DELPHI_EXPLAIN_PROMPT_VERSION,
    run_delphi,
)


def _interpretation(name="Response Valence", explanation=None):
    return {
        "construct_name": name,
        "explanation": explanation
        or "High values reflect positive responses while low values reflect negative responses.",
    }


class FakeClient:
    def __init__(self, handler):
        self.handler = handler
        self.calls = []

    def complete_json(self, **kwargs):
        self.calls.append(kwargs)
        data = self.handler(kwargs)
        number = len(self.calls)
        return {
            "data": data,
            "raw_content": json.dumps(data),
            "cache_key": f"cache-{number}",
            "prompt_version": kwargs["prompt_version"],
            "requested_model": "gpt-4o-mini",
            "returned_model": "gpt-4o-mini-2024-07-18",
            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
            "pricing_usd_per_million": {"input": 0.15, "output": 0.60},
            "cost_usd": 0.000027,
            "timestamp": "2026-08-04T00:00:00Z",
            "cached": False,
        }


def _profiles_from_prompt(prompt):
    return json.loads(prompt.rsplit("Profiles:\n", 1)[1])


def _profile_number(value):
    return int(re.search(r"(\d+)$", value).group(1))


def _all_object_schemas_are_strict(schema):
    if isinstance(schema, dict):
        if schema.get("type") == "object" and schema.get("additionalProperties") is not False:
            return False
        return all(_all_object_schemas_are_strict(value) for value in schema.values())
    if isinstance(schema, list):
        return all(_all_object_schemas_are_strict(value) for value in schema)
    return True


class AutoInterpTests(unittest.TestCase):
    def test_full_protocol_is_stratified_disjoint_strict_and_provenanced(self):
        respondent_count = 60

        def handler(call):
            if call["prompt_version"] == AUTOINTERP_EXPLAIN_PROMPT_VERSION:
                return _interpretation()
            self.assertEqual(call["prompt_version"], AUTOINTERP_SIMULATE_PROMPT_VERSION)
            rows = _profiles_from_prompt(call["user_prompt"])
            return {
                "predictions": [
                    {
                        "sample_id": row["sample_id"],
                        "activation_bin": int(
                            np.floor(_profile_number(row["profile"]) / 59 * 10 + 0.5)
                        ),
                    }
                    for row in rows
                ]
            }

        client = FakeClient(handler)
        profiles = [f"respondent {index}" for index in range(respondent_count)]
        result = run_autointerp(
            client, profiles, np.linspace(-3.0, 3.0, respondent_count), seed=19
        )

        self.assertEqual(len(client.calls), 2)
        self.assertEqual(len(result["explanation_indices"]), 20)
        self.assertEqual(len(result["simulation_indices"]), 20)
        self.assertTrue(
            set(result["explanation_indices"]).isdisjoint(result["simulation_indices"])
        )
        self.assertGreater(result["spearman"], 0.98)
        self.assertGreater(result["pearson"], 0.98)
        for call in client.calls:
            combined_prompt = call["system_prompt"] + call["user_prompt"]
            self.assertNotIn("dataset", combined_prompt.lower())
            self.assertNotIn("latent", combined_prompt.lower())
            self.assertTrue(_all_object_schemas_are_strict(call["schema"]))
        self.assertNotIn("raw_content", result["api_provenance"]["explainer"])
        self.assertNotIn("api_key", json.dumps(result["api_provenance"]))
        self.assertEqual(
            result["api_provenance"]["simulator"]["requested_model"],
            "gpt-4o-mini",
        )

    def test_small_input_degrades_deterministically_and_remains_disjoint(self):
        def handler(call):
            if call["prompt_version"] == AUTOINTERP_EXPLAIN_PROMPT_VERSION:
                return _interpretation()
            rows = _profiles_from_prompt(call["user_prompt"])
            return {
                "predictions": [
                    {"sample_id": row["sample_id"], "activation_bin": 5}
                    for row in rows
                ]
            }

        profiles = [f"respondent {index}" for index in range(5)]
        first = run_autointerp(FakeClient(handler), profiles, [-2, -1, 0, 1, 2], seed=7)
        second = run_autointerp(FakeClient(handler), profiles, [-2, -1, 0, 1, 2], seed=7)
        self.assertEqual(first["explanation_indices"], second["explanation_indices"])
        self.assertEqual(first["simulation_indices"], second["simulation_indices"])
        self.assertEqual(len(first["explanation_indices"]), 3)
        self.assertEqual(len(first["simulation_indices"]), 2)
        self.assertTrue(
            set(first["explanation_indices"]).isdisjoint(first["simulation_indices"])
        )

    def test_rejects_semantically_invalid_json(self):
        client = FakeClient(
            lambda _: _interpretation(name="This Name Has Far Too Many Words")
        )
        with self.assertRaises(BaselineOutputError):
            run_autointerp(client, ["respondent 0", "respondent 1"], [-1, 1])
        self.assertEqual(len(client.calls), 1)


class DelphiTests(unittest.TestCase):
    @staticmethod
    def candidates():
        return {
            "candidates": [
                {
                    "candidate_id": "C1",
                    **_interpretation(
                        "Positive Orientation",
                        "High values reflect positive answers while low values reflect negative answers.",
                    ),
                },
                {
                    "candidate_id": "C2",
                    **_interpretation(
                        "Emotional Stability vs. Emotional Reactivity",
                        "High values reflect negative answers while low values reflect positive answers.",
                    ),
                },
                {
                    "candidate_id": "C3",
                    **_interpretation(
                        "Response Intensity",
                        "High values reflect intense answers while low values reflect muted answers.",
                    ),
                },
            ]
        }

    def test_full_protocol_hard_negatives_joint_batches_and_selection(self):
        def handler(call):
            if call["prompt_version"] == DELPHI_EXPLAIN_PROMPT_VERSION:
                return self.candidates()
            self.assertEqual(call["prompt_version"], DELPHI_DETECT_PROMPT_VERSION)
            rows = _profiles_from_prompt(call["user_prompt"])
            predictions = []
            for row in rows:
                positive = _profile_number(row["profile"]) >= 70
                predictions.append(
                    {
                        "sample_id": row["sample_id"],
                        "scores": {
                            "C1": 90 if positive else 10,
                            "C2": 10 if positive else 90,
                            "C3": 50,
                        },
                    }
                )
            return {"predictions": predictions}

        client = FakeClient(handler)
        profiles = [f"respondent {index}" for index in range(140)]
        activations = np.linspace(-2.0, 2.0, 140)
        vectors = np.column_stack([activations, activations**2])
        result = run_delphi(client, profiles, activations, vectors, seed=23)

        self.assertEqual(len(client.calls), 3)
        self.assertEqual(len(result["generation_high_indices"]), 15)
        self.assertEqual(len(result["generation_hard_negative_indices"]), 5)
        self.assertTrue(all(index < 70 for index in result["generation_hard_negative_indices"]))
        self.assertEqual(len(result["validation_indices"]), 40)
        self.assertEqual(len(result["test_indices"]), 40)
        groups = [
            set(result["generation_high_indices"]),
            set(result["generation_hard_negative_indices"]),
            set(result["validation_indices"]),
            set(result["test_indices"]),
        ]
        for left in range(len(groups)):
            for right in range(left + 1, len(groups)):
                self.assertTrue(groups[left].isdisjoint(groups[right]))
        self.assertEqual(result["selected_candidate_id"], "C1")
        self.assertEqual(
            result["candidates"][1]["construct_name"],
            "Emotional Stability vs. Emotional Reactivity",
        )
        self.assertAlmostEqual(result["test_auroc"], 1.0)
        self.assertAlmostEqual(result["test_f1"], 1.0)
        self.assertEqual(len(result["validation_probabilities"]), 40)
        self.assertTrue(all(len(row) == 3 for row in result["validation_probabilities"]))

        detector_calls = [
            call for call in client.calls if call["prompt_version"] == DELPHI_DETECT_PROMPT_VERSION
        ]
        self.assertEqual(len(detector_calls), 2)
        self.assertEqual(
            [len(_profiles_from_prompt(call["user_prompt"])) for call in detector_calls],
            [40, 40],
        )
        for call in client.calls:
            combined_prompt = call["system_prompt"] + call["user_prompt"]
            self.assertNotIn("dataset", combined_prompt.lower())
            self.assertNotIn("latent", combined_prompt.lower())
            self.assertTrue(_all_object_schemas_are_strict(call["schema"]))

    def test_small_input_preserves_all_disjoint_stages(self):
        def handler(call):
            if call["prompt_version"] == DELPHI_EXPLAIN_PROMPT_VERSION:
                return self.candidates()
            rows = _profiles_from_prompt(call["user_prompt"])
            return {
                "predictions": [
                    {
                        "sample_id": row["sample_id"],
                        "scores": {"C1": 50, "C2": 50, "C3": 50},
                    }
                    for row in rows
                ]
            }

        client = FakeClient(handler)
        profiles = [f"respondent {index}" for index in range(12)]
        values = np.arange(12, dtype=float)
        result = run_delphi(
            client,
            profiles,
            values,
            np.column_stack([values, -values]),
            seed=5,
        )
        self.assertEqual(len(result["generation_high_indices"]), 4)
        self.assertEqual(len(result["generation_hard_negative_indices"]), 4)
        self.assertEqual(len(result["validation_indices"]), 2)
        self.assertEqual(len(result["test_indices"]), 2)
        used = (
            result["generation_high_indices"]
            + result["generation_hard_negative_indices"]
            + result["validation_indices"]
            + result["test_indices"]
        )
        self.assertEqual(len(used), len(set(used)))

    def test_validation_and_test_detectors_overlap(self):
        detector_barrier = threading.Barrier(2)

        def handler(call):
            if call["prompt_version"] == DELPHI_EXPLAIN_PROMPT_VERSION:
                return self.candidates()
            detector_barrier.wait(timeout=2.0)
            rows = _profiles_from_prompt(call["user_prompt"])
            return {
                "predictions": [
                    {
                        "sample_id": row["sample_id"],
                        "scores": {"C1": 50, "C2": 50, "C3": 50},
                    }
                    for row in rows
                ]
            }

        profiles = [f"respondent {index}" for index in range(140)]
        values = np.linspace(-2.0, 2.0, len(profiles))
        result = run_delphi(
            FakeClient(handler), profiles, values,
            np.column_stack([values, values**2]), seed=11,
        )
        self.assertEqual(len(result["validation_probabilities"]), 40)
        self.assertEqual(len(result["test_probabilities"]), 40)

    def test_rejects_duplicate_candidates(self):
        invalid = self.candidates()
        invalid["candidates"][2]["candidate_id"] = "C2"
        client = FakeClient(lambda _: invalid)
        profiles = [f"respondent {index}" for index in range(10)]
        with self.assertRaises(BaselineOutputError):
            run_delphi(client, profiles, np.arange(10), np.ones((10, 2)))
        self.assertEqual(len(client.calls), 1)


class DependencyBoundaryTests(unittest.TestCase):
    def test_module_does_not_import_judge_or_reference_answers(self):
        module_path = (
            Path(__file__).parents[2]
            / "v6"
            / "baselines"
            / "_llm_interpretability.py"
        )
        source = module_path.read_text(encoding="utf-8")
        self.assertNotIn("import judge", source)
        self.assertNotIn("from .judge", source)
        self.assertNotIn("from v6.judge", source)
        self.assertNotIn("gold_latent", source)


if __name__ == "__main__":
    unittest.main()
