"""Judge-independent LLM interpretability baselines for respondent features.

The two public functions operate on already rendered, fold-visible respondent
profiles.  They deliberately know nothing about task labels or reference
constructs.  All model calls go through :class:`BaselineAPIClient`, which owns
request caching, prompt-version cache keys, cost accounting, and retries.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import math
import re
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .api import BaselineAPIClient, BaselineAPIError


AUTOINTERP_EXPLAIN_PROMPT_VERSION = "autointerp-survey-explainer-v1"
AUTOINTERP_SIMULATE_PROMPT_VERSION = "autointerp-survey-simulator-v1"
DELPHI_EXPLAIN_PROMPT_VERSION = "delphi-survey-explainer-v1"
DELPHI_DETECT_PROMPT_VERSION = "delphi-survey-detector-v1"

_CANDIDATE_IDS = ("C1", "C2", "C3")
_INTERPRETATION_KEYS = {"construct_name", "explanation"}
_PROVENANCE_KEYS = (
    "cache_key",
    "prompt_version",
    "requested_model",
    "returned_model",
    "usage",
    "pricing_usd_per_million",
    "cost_usd",
    "timestamp",
    "cached",
)


class BaselineOutputError(BaselineAPIError):
    """The model returned JSON that violates the baseline protocol."""


_INTERPRETATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "construct_name": {"type": "string"},
        "explanation": {"type": "string"},
    },
    "required": ["construct_name", "explanation"],
    "additionalProperties": False,
}

_AUTOINTERP_EXPLANATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": dict(_INTERPRETATION_SCHEMA["properties"]),
    "required": list(_INTERPRETATION_SCHEMA["required"]),
    "additionalProperties": False,
}

_DELPHI_EXPLANATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "minItems": 3,
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "candidate_id": {"type": "string", "enum": list(_CANDIDATE_IDS)},
                    **_INTERPRETATION_SCHEMA["properties"],
                },
                "required": ["candidate_id", "construct_name", "explanation"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["candidates"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Automated Interpretability


def run_autointerp(
    client: BaselineAPIClient,
    profiles: Sequence[str],
    activations: Sequence[float] | np.ndarray,
    seed: int = 0,
) -> dict[str, Any]:
    """Run explanation plus disjoint held-out activation simulation.

    With at least 40 respondents, 20 activation-stratified profiles are used
    to explain the feature and 20 disjoint profiles are simulated.  Smaller
    inputs are split deterministically as evenly as possible, with the extra
    profile assigned to explanation.  Fewer than two profiles therefore yield
    an explanation but no simulation metric.
    """

    text, values = _validate_profiles_and_activations(profiles, activations)
    rng = _rng(seed)
    count = len(text)
    if count >= 40:
        explain_count, simulate_count = 20, 20
    elif count == 1:
        explain_count, simulate_count = 1, 0
    else:
        explain_count = min(20, (count + 1) // 2)
        simulate_count = min(20, count - explain_count)

    all_indices = np.arange(count, dtype=int)
    explain_indices = _stratified_sample(
        all_indices, values, explain_count, rng
    )
    explain_set = set(explain_indices)
    remaining = np.asarray(
        [index for index in all_indices if int(index) not in explain_set], dtype=int
    )
    simulate_indices = _stratified_sample(
        remaining, values, simulate_count, rng
    )
    bins = _activation_bins(values)

    explanation_examples = [
        {
            "example_id": f"E{position:03d}",
            "profile": text[index],
            "standardized_activation": round(float(values[index]), 6),
            "activation_bin": int(bins[index]),
        }
        for position, index in enumerate(explain_indices, 1)
    ]
    explanation_response = _complete(
        client,
        system_prompt=(
            "You explain a hidden scalar feature from observed response profiles. "
            "Infer only the underlying bipolar human dimension. Return only JSON "
            "matching the supplied schema. Do not mention any study, source, file, "
            "person identity, or hidden label."
        ),
        user_prompt=(
            "The examples below pair a response profile with its standardized "
            "feature activation and a 0-to-10 activation bin. A bin of 10 is the "
            "strong positive pole and 0 is the opposite pole. Produce a neutral "
            "1-to-4-word construct_name and exactly one sentence explaining what "
            "makes the activation high versus low.\n\nExamples:\n"
            + _json(explanation_examples)
        ),
        schema=_AUTOINTERP_EXPLANATION_SCHEMA,
        prompt_version=AUTOINTERP_EXPLAIN_PROMPT_VERSION,
        max_output_tokens=180,
    )
    interpretation = _validate_interpretation(
        _response_data(explanation_response),
        context="Automated Interpretability explanation",
    )

    predicted: list[int] = []
    true_bins: list[int] = []
    true_activations: list[float] = []
    simulator_provenance: Optional[dict[str, Any]] = None
    if simulate_indices:
        simulation_examples = [
            {"sample_id": f"S{position:03d}", "profile": text[index]}
            for position, index in enumerate(simulate_indices, 1)
        ]
        sample_ids = [example["sample_id"] for example in simulation_examples]
        simulation_response = _complete(
            client,
            system_prompt=(
                "You simulate a scalar feature from a supplied interpretation and "
                "previously unseen response profiles. Return only JSON matching "
                "the supplied schema."
            ),
            user_prompt=(
                "Use the interpretation to predict one integer activation_bin from "
                "0 to 10 for every profile. A value of 10 means the positive pole is "
                "strongly present and 0 means the opposite pole. Do not omit or add "
                "sample IDs.\n\nInterpretation:\n"
                + _json(interpretation)
                + "\n\nProfiles:\n"
                + _json(simulation_examples)
            ),
            schema=_simulation_schema(sample_ids),
            prompt_version=AUTOINTERP_SIMULATE_PROMPT_VERSION,
            max_output_tokens=max(160, 26 * len(sample_ids)),
        )
        predicted = _validate_simulation(
            _response_data(simulation_response), sample_ids
        )
        true_bins = [int(bins[index]) for index in simulate_indices]
        true_activations = [float(values[index]) for index in simulate_indices]
        simulator_provenance = _provenance(simulation_response)

    return {
        "method": "Automated Interpretability (simulation-scored adaptation)",
        "seed": int(seed),
        **interpretation,
        "explanation_indices": explain_indices,
        "simulation_indices": simulate_indices,
        "true_activations": true_activations,
        "true_activation_bins": true_bins,
        "predicted_activation_bins": predicted,
        "spearman": _correlation(true_activations, predicted, ranks=True),
        "pearson": _correlation(true_activations, predicted, ranks=False),
        "api_provenance": {
            "explainer": _provenance(explanation_response),
            "simulator": simulator_provenance,
        },
    }


# ---------------------------------------------------------------------------
# Delphi


def run_delphi(
    client: BaselineAPIClient,
    profiles: Sequence[str],
    activations: Sequence[float] | np.ndarray,
    response_vectors: Sequence[Sequence[float]] | np.ndarray,
    seed: int = 0,
) -> dict[str, Any]:
    """Run contrastive explanation, validation selection, and held-out detection.

    The full protocol uses 15 top-activation examples, five response-space hard
    negatives, 40 balanced validation profiles, and 40 disjoint balanced test
    profiles.  Each detector stage is exactly one request that scores all three
    candidates jointly.  If there are too few respondents, generation retains
    at least one example per available class while reserving up to two examples
    per class for disjoint evaluation; remaining evaluation examples are split
    deterministically between validation and test.
    """

    text, values = _validate_profiles_and_activations(profiles, activations)
    vectors = np.asarray(response_vectors, dtype=float)
    if vectors.ndim != 2 or vectors.shape[0] != len(text) or vectors.shape[1] < 1:
        raise ValueError(
            "response_vectors must be a non-empty 2D matrix with one row per profile"
        )
    if not np.isfinite(vectors).all():
        raise ValueError("response_vectors must contain only finite values")
    rng = _rng(seed)

    positive_pool, negative_pool = _balanced_poles(values, rng)
    high_count = _generation_count(len(positive_pool), 15)
    negative_count = _generation_count(len(negative_pool), 5)
    high_indices = _ordered_by_value(
        positive_pool, values, rng, descending=True
    )[:high_count]
    hard_negative_indices = _nearest_hard_negatives(
        negative_pool, high_indices, vectors, negative_count, rng
    )

    used = set(high_indices) | set(hard_negative_indices)
    positive_remaining = [index for index in positive_pool if index not in used]
    negative_remaining = [index for index in negative_pool if index not in used]
    available_per_class = min(len(positive_remaining), len(negative_remaining), 40)
    if available_per_class >= 40:
        validation_per_class, test_per_class = 20, 20
    else:
        validation_per_class = (available_per_class + 1) // 2
        test_per_class = available_per_class - validation_per_class

    positive_order = list(rng.permutation(positive_remaining))
    negative_order = list(rng.permutation(negative_remaining))
    validation_indices, validation_labels = _make_balanced_stage(
        positive_order[:validation_per_class],
        negative_order[:validation_per_class],
        rng,
    )
    test_start = validation_per_class
    test_indices, test_labels = _make_balanced_stage(
        positive_order[test_start : test_start + test_per_class],
        negative_order[test_start : test_start + test_per_class],
        rng,
    )

    generation_context = {
        "strong_examples": [
            {"example_id": f"P{position:03d}", "profile": text[index]}
            for position, index in enumerate(high_indices, 1)
        ],
        "similar_weak_examples": [
            {"example_id": f"N{position:03d}", "profile": text[index]}
            for position, index in enumerate(hard_negative_indices, 1)
        ],
    }
    explanation_response = _complete(
        client,
        system_prompt=(
            "You contrast response profiles where a hidden feature is strong with "
            "similar profiles where it is weak. Return only JSON matching the "
            "supplied schema. Do not mention any study, source, file, person "
            "identity, or hidden label."
        ),
        user_prompt=(
            "Infer three distinct candidate interpretations for the underlying "
            "bipolar human dimension. Each candidate needs its fixed candidate_id, "
            "a neutral 1-to-4-word construct_name, and exactly one sentence "
            "explaining what makes the feature strong versus weak. Return C1, C2, "
            "and C3 exactly once.\n\nContrastive examples:\n"
            + _json(generation_context)
        ),
        schema=_DELPHI_EXPLANATION_SCHEMA,
        prompt_version=DELPHI_EXPLAIN_PROMPT_VERSION,
        max_output_tokens=420,
    )
    candidates = _validate_candidates(_response_data(explanation_response))

    validation_scores: list[list[int]] = []
    validation_provenance: Optional[dict[str, Any]] = None
    test_scores: list[list[int]] = []
    test_provenance: Optional[dict[str, Any]] = None

    # Both stages score all three candidates and neither consumes the other
    # stage's labels or predictions.  Running them concurrently changes only
    # scheduling: prompts, cache keys, samples, and selection remain identical.
    if validation_indices and test_indices:
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="delphi-detector") as pool:
            validation_future = pool.submit(
                _run_detector, client, candidates, text, validation_indices,
                sample_prefix="V",
            )
            test_future = pool.submit(
                _run_detector, client, candidates, text, test_indices,
                sample_prefix="T",
            )
            validation_response, validation_scores = validation_future.result()
            test_response, test_scores = test_future.result()
        validation_provenance = _provenance(validation_response)
        test_provenance = _provenance(test_response)
    elif validation_indices:
        validation_response, validation_scores = _run_detector(
            client, candidates, text, validation_indices, sample_prefix="V"
        )
        validation_provenance = _provenance(validation_response)
    elif test_indices:
        test_response, test_scores = _run_detector(
            client, candidates, text, test_indices, sample_prefix="T"
        )
        test_provenance = _provenance(test_response)

    validation_metrics = (
        _candidate_metrics(validation_labels, validation_scores)
        if validation_scores else
        [{"auroc": None, "f1": None} for _ in _CANDIDATE_IDS]
    )
    selected_index = _select_candidate(validation_metrics)
    selected = candidates[selected_index]
    test_metrics = (
        _candidate_metrics(test_labels, test_scores)
        if test_scores else
        [{"auroc": None, "f1": None} for _ in _CANDIDATE_IDS]
    )

    candidate_results = []
    for index, candidate in enumerate(candidates):
        candidate_results.append(
            {
                **candidate,
                "validation_auroc": validation_metrics[index]["auroc"],
                "validation_f1": validation_metrics[index]["f1"],
                "test_auroc": test_metrics[index]["auroc"],
                "test_f1": test_metrics[index]["f1"],
            }
        )

    return {
        "method": "Delphi (contrastive, detection-scored adaptation)",
        "seed": int(seed),
        "construct_name": selected["construct_name"],
        "explanation": selected["explanation"],
        "selected_candidate_id": selected["candidate_id"],
        "candidates": candidate_results,
        "generation_high_indices": high_indices,
        "generation_hard_negative_indices": hard_negative_indices,
        "validation_indices": validation_indices,
        "validation_labels": validation_labels,
        "validation_probabilities": validation_scores,
        "test_indices": test_indices,
        "test_labels": test_labels,
        "test_probabilities": test_scores,
        "test_auroc": test_metrics[selected_index]["auroc"],
        "test_f1": test_metrics[selected_index]["f1"],
        "api_provenance": {
            "explainer": _provenance(explanation_response),
            "validation_detector": validation_provenance,
            "test_detector": test_provenance,
        },
    }


def _run_detector(
    client: BaselineAPIClient,
    candidates: list[dict[str, str]],
    profiles: list[str],
    indices: list[int],
    *,
    sample_prefix: str,
) -> tuple[Mapping[str, Any], list[list[int]]]:
    examples = [
        {"sample_id": f"{sample_prefix}{position:03d}", "profile": profiles[index]}
        for position, index in enumerate(indices, 1)
    ]
    sample_ids = [example["sample_id"] for example in examples]
    response = _complete(
        client,
        system_prompt=(
            "You detect whether each supplied interpretation is strongly present "
            "in each response profile. Score all interpretations jointly and return "
            "only JSON matching the supplied schema."
        ),
        user_prompt=(
            "For every profile, assign each candidate an integer probability from "
            "0 to 100 that the candidate's positive pole is strongly present. "
            "Return every sample ID once and score C1, C2, and C3 for each.\n\n"
            "Candidate interpretations:\n"
            + _json(candidates)
            + "\n\nProfiles:\n"
            + _json(examples)
        ),
        schema=_detector_schema(sample_ids),
        prompt_version=DELPHI_DETECT_PROMPT_VERSION,
        max_output_tokens=max(260, 48 * len(sample_ids)),
    )
    scores = _validate_detection(_response_data(response), sample_ids)
    return response, scores


def _simulation_schema(sample_ids: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "predictions": {
                "type": "array",
                "minItems": len(sample_ids),
                "maxItems": len(sample_ids),
                "items": {
                    "type": "object",
                    "properties": {
                        "sample_id": {"type": "string", "enum": sample_ids},
                        "activation_bin": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 10,
                        },
                    },
                    "required": ["sample_id", "activation_bin"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["predictions"],
        "additionalProperties": False,
    }


def _detector_schema(sample_ids: list[str]) -> dict[str, Any]:
    score_properties = {
        candidate_id: {"type": "integer", "minimum": 0, "maximum": 100}
        for candidate_id in _CANDIDATE_IDS
    }
    return {
        "type": "object",
        "properties": {
            "predictions": {
                "type": "array",
                "minItems": len(sample_ids),
                "maxItems": len(sample_ids),
                "items": {
                    "type": "object",
                    "properties": {
                        "sample_id": {"type": "string", "enum": sample_ids},
                        "scores": {
                            "type": "object",
                            "properties": score_properties,
                            "required": list(_CANDIDATE_IDS),
                            "additionalProperties": False,
                        },
                    },
                    "required": ["sample_id", "scores"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["predictions"],
        "additionalProperties": False,
    }


def _complete(
    client: BaselineAPIClient,
    *,
    system_prompt: str,
    user_prompt: str,
    schema: Mapping[str, Any],
    prompt_version: str,
    max_output_tokens: int,
) -> Mapping[str, Any]:
    if not hasattr(client, "complete_json"):
        raise TypeError("client must provide complete_json")
    response = client.complete_json(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        schema=schema,
        prompt_version=prompt_version,
        max_output_tokens=max_output_tokens,
    )
    if not isinstance(response, Mapping):
        raise BaselineOutputError("baseline client returned a non-object response")
    return response


def _response_data(response: Mapping[str, Any]) -> Mapping[str, Any]:
    data = response.get("data")
    if not isinstance(data, Mapping):
        raise BaselineOutputError("baseline response data must be a JSON object")
    return data


def _validate_interpretation(
    value: Mapping[str, Any], *, context: str
) -> dict[str, str]:
    _exact_keys(value, _INTERPRETATION_KEYS, context)
    name = value.get("construct_name")
    explanation = value.get("explanation")
    if not isinstance(name, str) or not isinstance(explanation, str):
        raise BaselineOutputError(f"{context} fields must be strings")
    name = " ".join(name.strip().split())
    words = re.findall(r"[\w]+(?:[-'][\w]+)*", name, flags=re.UNICODE)
    # Treat the bipolar connector as punctuation rather than a semantic word.
    # This accepts names such as "Emotional Stability vs. Emotional Reactivity"
    # without weakening the four-content-word limit.
    content_words = [
        word for word in words if word.casefold() not in {"vs", "versus"}
    ]
    if not name or not 1 <= len(content_words) <= 4:
        raise BaselineOutputError(f"{context} construct_name must contain 1-4 words")
    explanation = " ".join(explanation.strip().split())
    if not explanation or explanation[-1:] not in ".!?":
        raise BaselineOutputError(f"{context} explanation must be one sentence")
    if re.search(r"[.!?].+?[.!?]", explanation):
        raise BaselineOutputError(f"{context} explanation must be one sentence")
    return {"construct_name": name, "explanation": explanation}


def _validate_candidates(data: Mapping[str, Any]) -> list[dict[str, str]]:
    _exact_keys(data, {"candidates"}, "Delphi explanation")
    raw = data.get("candidates")
    if not isinstance(raw, list) or len(raw) != 3:
        raise BaselineOutputError("Delphi explanation must contain three candidates")
    by_id: dict[str, dict[str, str]] = {}
    for item in raw:
        if not isinstance(item, Mapping):
            raise BaselineOutputError("each Delphi candidate must be an object")
        _exact_keys(
            item,
            {"candidate_id", "construct_name", "explanation"},
            "Delphi candidate",
        )
        candidate_id = item.get("candidate_id")
        if candidate_id not in _CANDIDATE_IDS or candidate_id in by_id:
            raise BaselineOutputError("Delphi candidate IDs must be C1, C2, C3 once each")
        interpretation = _validate_interpretation(
            {
                "construct_name": item.get("construct_name"),
                "explanation": item.get("explanation"),
            },
            context=f"Delphi candidate {candidate_id}",
        )
        by_id[str(candidate_id)] = {
            "candidate_id": str(candidate_id),
            **interpretation,
        }
    if set(by_id) != set(_CANDIDATE_IDS):
        raise BaselineOutputError("Delphi candidate IDs must be C1, C2, C3 once each")
    return [by_id[candidate_id] for candidate_id in _CANDIDATE_IDS]


def _validate_simulation(data: Mapping[str, Any], ids: list[str]) -> list[int]:
    _exact_keys(data, {"predictions"}, "Automated Interpretability simulation")
    raw = data.get("predictions")
    if not isinstance(raw, list) or len(raw) != len(ids):
        raise BaselineOutputError(
            "Automated Interpretability simulation returned the wrong row count"
        )
    by_id: dict[str, int] = {}
    for item in raw:
        if not isinstance(item, Mapping):
            raise BaselineOutputError(
                "Automated Interpretability prediction must be an object"
            )
        _exact_keys(
            item,
            {"sample_id", "activation_bin"},
            "Automated Interpretability prediction",
        )
        sample_id, value = item.get("sample_id"), item.get("activation_bin")
        if sample_id not in ids or sample_id in by_id:
            raise BaselineOutputError(
                "Automated Interpretability simulation has missing or duplicate IDs"
            )
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 10:
            raise BaselineOutputError("activation_bin must be an integer from 0 to 10")
        by_id[str(sample_id)] = value
    if set(by_id) != set(ids):
        raise BaselineOutputError(
            "Automated Interpretability simulation has missing or duplicate IDs"
        )
    return [by_id[sample_id] for sample_id in ids]


def _validate_detection(data: Mapping[str, Any], ids: list[str]) -> list[list[int]]:
    _exact_keys(data, {"predictions"}, "Delphi detection")
    raw = data.get("predictions")
    if not isinstance(raw, list) or len(raw) != len(ids):
        raise BaselineOutputError("Delphi detection returned the wrong row count")
    by_id: dict[str, list[int]] = {}
    for item in raw:
        if not isinstance(item, Mapping):
            raise BaselineOutputError("Delphi prediction must be an object")
        _exact_keys(item, {"sample_id", "scores"}, "Delphi prediction")
        sample_id, scores = item.get("sample_id"), item.get("scores")
        if sample_id not in ids or sample_id in by_id:
            raise BaselineOutputError("Delphi detection has missing or duplicate IDs")
        if not isinstance(scores, Mapping):
            raise BaselineOutputError("Delphi scores must be an object")
        _exact_keys(scores, set(_CANDIDATE_IDS), "Delphi scores")
        row: list[int] = []
        for candidate_id in _CANDIDATE_IDS:
            score = scores.get(candidate_id)
            if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= 100:
                raise BaselineOutputError(
                    "Delphi probabilities must be integers from 0 to 100"
                )
            row.append(score)
        by_id[str(sample_id)] = row
    if set(by_id) != set(ids):
        raise BaselineOutputError("Delphi detection has missing or duplicate IDs")
    return [by_id[sample_id] for sample_id in ids]


def _validate_profiles_and_activations(
    profiles: Sequence[str], activations: Sequence[float] | np.ndarray
) -> tuple[list[str], np.ndarray]:
    if isinstance(profiles, (str, bytes)):
        raise TypeError("profiles must be a sequence of rendered strings")
    text = list(profiles)
    if not text:
        raise ValueError("at least one profile is required")
    if any(not isinstance(item, str) or not item.strip() for item in text):
        raise ValueError("every profile must be a non-empty rendered string")
    try:
        values = np.asarray(activations, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("activations must be numeric") from exc
    if values.ndim != 1 or len(values) != len(text):
        raise ValueError("activations must be 1D with one value per profile")
    if not np.isfinite(values).all():
        raise ValueError("activations must contain only finite values")
    return text, values


def _rng(seed: int) -> np.random.Generator:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    return np.random.default_rng(int(seed))


def _ordered_by_value(
    indices: Sequence[int],
    values: np.ndarray,
    rng: np.random.Generator,
    *,
    descending: bool,
) -> list[int]:
    ties = {int(index): float(rng.random()) for index in indices}
    direction = -1.0 if descending else 1.0
    return sorted(
        (int(index) for index in indices),
        key=lambda index: (direction * float(values[index]), ties[index], index),
    )


def _stratified_sample(
    indices: np.ndarray,
    values: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> list[int]:
    if count <= 0 or len(indices) == 0:
        return []
    count = min(count, len(indices))
    ordered = _ordered_by_value(indices, values, rng, descending=False)
    strata = np.array_split(np.asarray(ordered, dtype=int), count)
    chosen = [int(group[int(rng.integers(0, len(group)))]) for group in strata]
    return sorted(chosen, key=lambda index: (float(values[index]), index))


def _activation_bins(values: np.ndarray) -> np.ndarray:
    minimum, maximum = float(values.min()), float(values.max())
    if math.isclose(minimum, maximum, rel_tol=0.0, abs_tol=1e-12):
        return np.full(len(values), 5, dtype=int)
    scaled = np.clip((values - minimum) / (maximum - minimum), 0.0, 1.0)
    return np.floor(scaled * 10.0 + 0.5).astype(int)


def _balanced_poles(
    values: np.ndarray, rng: np.random.Generator
) -> tuple[list[int], list[int]]:
    ordered = _ordered_by_value(
        np.arange(len(values), dtype=int), values, rng, descending=False
    )
    half = len(ordered) // 2
    if half == 0:
        return ordered, []
    return ordered[-half:], ordered[:half]


def _generation_count(available: int, target: int) -> int:
    if available <= 0:
        return 0
    reserve = 2 if available >= 3 else (1 if available == 2 else 0)
    return min(target, max(1, available - reserve))


def _nearest_hard_negatives(
    negative_pool: Sequence[int],
    high_indices: Sequence[int],
    vectors: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> list[int]:
    if count <= 0 or not negative_pool:
        return []
    if not high_indices:
        return list(negative_pool[:count])
    negative = np.asarray(negative_pool, dtype=int)
    high = np.asarray(high_indices, dtype=int)
    differences = vectors[negative, None, :] - vectors[high, :][None, :, :]
    distances = np.sqrt(np.sum(differences * differences, axis=2)).min(axis=1)
    ties = rng.random(len(negative))
    order = np.lexsort((negative, ties, distances))
    return [int(negative[position]) for position in order[:count]]


def _make_balanced_stage(
    positive: Sequence[int],
    negative: Sequence[int],
    rng: np.random.Generator,
) -> tuple[list[int], list[int]]:
    pairs = [(int(index), 1) for index in positive] + [
        (int(index), 0) for index in negative
    ]
    if not pairs:
        return [], []
    order = rng.permutation(len(pairs))
    shuffled = [pairs[int(position)] for position in order]
    return [pair[0] for pair in shuffled], [pair[1] for pair in shuffled]


def _candidate_metrics(
    labels: list[int], score_rows: list[list[int]]
) -> list[dict[str, Optional[float]]]:
    if not labels:
        return [{"auroc": None, "f1": None} for _ in _CANDIDATE_IDS]
    matrix = np.asarray(score_rows, dtype=float)
    return [
        {
            "auroc": _auroc(labels, matrix[:, column].tolist()),
            "f1": _f1(labels, matrix[:, column].tolist()),
        }
        for column in range(len(_CANDIDATE_IDS))
    ]


def _select_candidate(metrics: list[dict[str, Optional[float]]]) -> int:
    def key(index: int) -> tuple[float, float, int]:
        auroc = metrics[index]["auroc"]
        f1 = metrics[index]["f1"]
        return (
            -(auroc if auroc is not None else -math.inf),
            -(f1 if f1 is not None else -math.inf),
            index,
        )

    return min(range(len(_CANDIDATE_IDS)), key=key)


def _auroc(labels: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    labels_array = np.asarray(labels, dtype=int)
    scores_array = np.asarray(scores, dtype=float)
    positives = int(labels_array.sum())
    negatives = len(labels_array) - positives
    if positives == 0 or negatives == 0:
        return None
    ranks = _average_ranks(scores_array)
    rank_sum = float(ranks[labels_array == 1].sum())
    value = (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
    return float(value)


def _f1(labels: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    if not labels:
        return None
    truth = np.asarray(labels, dtype=int)
    predicted = np.asarray(scores, dtype=float) >= 50.0
    tp = int(np.sum((truth == 1) & predicted))
    fp = int(np.sum((truth == 0) & predicted))
    fn = int(np.sum((truth == 1) & ~predicted))
    denominator = 2 * tp + fp + fn
    return float(2 * tp / denominator) if denominator else 0.0


def _correlation(
    first: Sequence[float], second: Sequence[float], *, ranks: bool
) -> Optional[float]:
    if len(first) < 2 or len(first) != len(second):
        return None
    left, right = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if ranks:
        left, right = _average_ranks(left), _average_ranks(right)
    if np.ptp(left) <= 1e-12 or np.ptp(right) <= 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    return ranks


def _provenance(response: Mapping[str, Any]) -> dict[str, Any]:
    """Keep reproducibility metadata, excluding prompts, content, and credentials."""

    return {key: response[key] for key in _PROVENANCE_KEYS if key in response}


def _exact_keys(value: Mapping[str, Any], keys: set[str], context: str) -> None:
    if set(value) != keys:
        raise BaselineOutputError(f"{context} returned unexpected or missing fields")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


__all__ = [
    "AUTOINTERP_EXPLAIN_PROMPT_VERSION",
    "AUTOINTERP_SIMULATE_PROMPT_VERSION",
    "DELPHI_EXPLAIN_PROMPT_VERSION",
    "DELPHI_DETECT_PROMPT_VERSION",
    "BaselineOutputError",
    "run_autointerp",
    "run_delphi",
]
