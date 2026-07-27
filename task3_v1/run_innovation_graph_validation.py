"""Corrected Stage-1 validation on cross-layer innovation features.

The discovery-only innovation transform and CauScale scores are frozen inputs.
The primary endpoint is cross-concept total downstream effect, scored with the
max-product reachability of the frozen graph. Raw same-concept propagation is
retained only as a positive-control audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import defaultdict, deque
from pathlib import Path

import jlens
import numpy as np
import torch
import transformers
from jlens.hooks import ActivationRecorder

MODEL_NAME = "Qwen/Qwen3.5-4B"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_REVISION = "qwen-n1000"
LENS_FILE = (
    "qwen3.5-4b/jlens/Salesforce-wikitext/"
    "Qwen3.5-4B_jacobian_lens_n1000.pt"
)
MANDATORY_CONCEPTS = [" water", " fire", " code", " happy"]
DOSES = np.asarray([-2.0, -1.0, 1.0, 2.0], dtype=np.float32)
RIDGE = 1e-4


def clean_wikitext(line: str) -> str:
    line = (
        line.replace("@-@", "-")
        .replace("@.@", ".")
        .replace("@,@", ",")
        .replace("<unk>", "unknown")
    )
    return re.sub(r"\s+", " ", line).strip()


def heldout_prompts(
    corpus: Path,
    tokenizer,
    forbidden_concepts: list[str],
    count: int = 20,
    max_tokens: int = 128,
) -> tuple[list[str], list[str], str]:
    pattern = re.compile(
        r"\b(?:"
        + "|".join(re.escape(concept.strip()) for concept in forbidden_concepts)
        + r")\b",
        flags=re.IGNORECASE,
    )
    forbidden_token_ids = set()
    for concept in forbidden_concepts:
        ids = tokenizer.encode(concept, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Forbidden concept is not one token: {concept!r}")
        forbidden_token_ids.add(ids[0])
    raw_text = corpus.read_text(encoding="utf-8")
    grouped: dict[str, deque[str]] = defaultdict(deque)
    article = "unknown"
    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        heading = re.fullmatch(r"= ([^=].*?) =", line)
        if heading:
            article = clean_wikitext(heading.group(1))
            continue
        if not line or line.startswith("="):
            continue
        cleaned = clean_wikitext(line)
        if len(cleaned.split()) < 20 or pattern.search(cleaned):
            continue
        ids = tokenizer.encode(cleaned, add_special_tokens=False)[:max_tokens]
        if forbidden_token_ids.intersection(ids):
            continue
        prompt = tokenizer.decode(ids).strip()
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if (
            prompt
            and not pattern.search(prompt)
            and not forbidden_token_ids.intersection(prompt_ids)
        ):
            grouped[article].append(prompt)

    prompts, groups = [], []
    active = deque(sorted(grouped))
    while active and len(prompts) < count:
        group = active.popleft()
        prompts.append(grouped[group].popleft())
        groups.append(group)
        if grouped[group]:
            active.append(group)
    if len(prompts) != count:
        raise ValueError(f"Only {len(prompts)} held-out prompts passed filtering")
    corpus_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
    return prompts, groups, corpus_hash


def capture(model, prompt: str, layers: list[int]) -> dict[int, torch.Tensor]:
    ids = model.encode(prompt)
    with ActivationRecorder(model.layers, at=layers) as recorder:
        model.forward(ids)
        return {layer: recorder.activations[layer].detach() for layer in layers}


def patched_capture_batch(
    model,
    prompt: str,
    source_layer: int,
    record_layers: list[int],
    deltas: torch.Tensor,
) -> dict[int, torch.Tensor]:
    ids = model.encode(prompt).repeat(deltas.shape[0], 1)

    def patch_hook(module, inputs, output):
        tensor = output if torch.is_tensor(output) else output[0]
        changed = tensor.clone()
        changed[:, -1, :] += deltas.to(device=tensor.device, dtype=tensor.dtype)
        if torch.is_tensor(output):
            return changed
        return (changed, *output[1:])

    handle = model.layers[source_layer].register_forward_hook(patch_hook)
    try:
        with ActivationRecorder(model.layers, at=record_layers) as recorder:
            model.forward(ids)
            return {
                layer: recorder.activations[layer].detach()
                for layer in record_layers
            }
    finally:
        handle.remove()


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    return ranks


def binary_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    labels = labels.astype(bool)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return {
            "auroc": None,
            "auprc": None,
            "positive_pairs": positives,
            "negative_pairs": negatives,
        }
    ranks = average_ranks(scores)
    auroc = (
        ranks[labels].sum() - positives * (positives + 1) / 2
    ) / (positives * negatives)

    order = np.argsort(scores)[::-1]
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    true_positive = 0
    false_positive = 0
    average_precision = 0.0
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        group = sorted_labels[start:end]
        previous_true_positive = true_positive
        true_positive += int(group.sum())
        false_positive += len(group) - int(group.sum())
        precision = true_positive / (true_positive + false_positive)
        average_precision += (
            (true_positive - previous_true_positive) / positives * precision
        )
        start = end

    result = {
        "auroc": float(auroc),
        "auprc": float(average_precision),
        "positive_pairs": positives,
        "negative_pairs": negatives,
    }
    descending = np.argsort(scores)[::-1]
    for k in (10, 32, 64):
        use_k = min(k, len(labels))
        result[f"precision_at_{k}"] = float(labels[descending[:use_k]].mean())
    return result


def max_product_path_scores(
    edge_probability: np.ndarray,
    columns: list[dict],
    exclude_same_concept: bool = True,
) -> np.ndarray:
    """Return max-product reachability on the layer-ordered DAG."""
    layers = np.asarray([int(column["layer"]) for column in columns])
    allowed = layers[:, None] < layers[None, :]
    scores = np.where(allowed, edge_probability, 0.0).astype(np.float64)
    if exclude_same_concept:
        concepts = np.asarray([column["concept"] for column in columns])
        scores[concepts[:, None] == concepts[None, :]] = 0.0
    for intermediate in np.argsort(layers):
        through = scores[:, intermediate, None] * scores[intermediate, None, :]
        scores = np.maximum(scores, through)
    scores[~allowed] = 0.0
    np.fill_diagonal(scores, 0.0)
    return scores.astype(np.float32)


def apply_delta_transform(
    raw_delta: np.ndarray,
    transform_entries: list[dict],
) -> np.ndarray:
    """Apply a frozen affine transform to a change vector (intercepts cancel)."""
    transformed = raw_delta.copy()
    for entry in transform_entries:
        target = int(entry["target_index"])
        predictors = np.asarray(entry["predictor_indices"], dtype=np.int64)
        coefficients = np.asarray(entry["coefficient"], dtype=np.float32)
        transformed[target] -= float(raw_delta[predictors] @ coefficients)
    return transformed


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heldout-corpus", type=Path, required=True)
    parser.add_argument("--discovery-matrix", type=Path, required=True)
    parser.add_argument("--discovery-metadata", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--bootstrap-json", type=Path, required=True)
    parser.add_argument("--bootstrap-npz", type=Path, required=True)
    parser.add_argument("--innovation-transform", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("task3") / "outputs" / "validation",
    )
    parser.add_argument(
        "--output-stem", default="innovation_heldout_graph_validation"
    )
    args = parser.parse_args()
    for path in [
        args.heldout_corpus,
        args.discovery_matrix,
        args.discovery_metadata,
        args.calibration,
        args.bootstrap_json,
        args.bootstrap_npz,
        args.innovation_transform,
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    discovery = np.load(args.discovery_matrix).astype(np.float32)
    discovery_meta = json.loads(
        args.discovery_metadata.read_text(encoding="utf-8")
    )
    columns = discovery_meta["columns"]
    concepts = [column["concept"] for column in columns[:32]]
    layers = sorted({int(column["layer"]) for column in columns})
    layer_to_index = {layer: index for index, layer in enumerate(layers)}
    calibration = np.load(args.calibration)
    raw_coordinate_std = calibration["coordinates"].std(axis=0, ddof=1)
    innovation_std = discovery.std(axis=0, ddof=1)
    if np.any(innovation_std < 1e-8):
        raise RuntimeError("Innovation discovery matrix has zero-variance columns")
    bootstrap = json.loads(args.bootstrap_json.read_text(encoding="utf-8"))
    bootstrap_arrays = np.load(args.bootstrap_npz)
    mean_probability = bootstrap_arrays["mean_probability"]
    selection_frequency = bootstrap_arrays["selection_frequency"]
    innovation_transform = json.loads(
        args.innovation_transform.read_text(encoding="utf-8")
    )
    transform_entries = innovation_transform["transform_entries"]
    path_probability = max_product_path_scores(
        mean_probability, columns, exclude_same_concept=True
    )

    mandatory_sources = [
        layer_index * len(concepts) + concepts.index(concept)
        for layer_index in range(3)
        for concept in MANDATORY_CONCEPTS
    ]
    source_indices = list(mandatory_sources)
    for edge in bootstrap["stable_edges"]:
        source = int(edge["source_index"])
        target = int(edge["target_index"])
        if columns[source]["concept"] == columns[target]["concept"]:
            continue
        if source not in source_indices and columns[source]["layer"] < layers[-1]:
            source_indices.append(source)
        if len(source_indices) == 16:
            break
    if len(source_indices) < 16:
        candidate_sources = [
            index
            for index, column in enumerate(columns)
            if int(column["layer"]) < layers[-1]
        ]
        candidate_sources.sort(
            key=lambda source: float(path_probability[source].max()), reverse=True
        )
        for source in candidate_sources:
            if source not in source_indices:
                source_indices.append(source)
            if len(source_indices) == 16:
                break
    if len(source_indices) != 16:
        raise RuntimeError(f"Could only select {len(source_indices)} sources")
    source_concepts = sorted({columns[index]["concept"] for index in source_indices})
    source_selection_rule = (
        "water/fire/code/happy at layers 8/16/24; then highest-ranked unique "
        "source nodes from stable cross-concept edges, falling back to frozen "
        "cross-concept max-product path score until 16"
    )

    print("Loading cached model and Jacobian lens...", flush=True)
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16
    ).cuda()
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    model = jlens.from_hf(hf_model, tokenizer)
    lens = jlens.JacobianLens.from_pretrained(
        LENS_REPO, filename=LENS_FILE, revision=LENS_REVISION
    )
    token_ids = []
    for concept in concepts:
        ids = tokenizer.encode(concept, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Not one token: {concept!r} -> {ids}")
        token_ids.append(ids[0])
    prompts, prompt_groups, corpus_hash = heldout_prompts(
        args.heldout_corpus, tokenizer, concepts
    )
    print(
        f"Selected {len(prompts)} held-out prompts from "
        f"{len(set(prompt_groups))} test-only article groups.",
        flush=True,
    )

    vectors_by_layer = {}
    dual_by_layer = {}
    for layer in layers:
        jacobian = lens.jacobians[layer].to(device="cuda", dtype=torch.float32)
        raw_vectors = jacobian.T @ model._lm_head.weight[token_ids].float().T
        vectors = torch.nn.functional.normalize(raw_vectors, dim=0)
        gram = vectors.T @ vectors
        dual = vectors @ torch.linalg.inv(
            gram + RIDGE * torch.eye(len(concepts), device="cuda")
        )
        vectors_by_layer[layer] = vectors
        dual_by_layer[layer] = dual

    effects = np.full(
        (len(source_indices), len(prompts), len(DOSES), len(columns)),
        np.nan,
        dtype=np.float32,
    )
    raw_effects = np.full_like(effects, np.nan)
    write_valid = np.zeros(effects.shape[:3], dtype=bool)
    local_rows = []
    sources_by_layer: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for source_position, source_index in enumerate(source_indices):
        sources_by_layer[int(columns[source_index]["layer"])].append(
            (source_position, source_index)
        )

    print("Running paired held-out interventions...", flush=True)
    for prompt_index, prompt in enumerate(prompts):
        baseline_acts = capture(model, prompt, layers)
        baseline_coordinates = {
            layer: (
                baseline_acts[layer][0, -1].float() @ vectors_by_layer[layer]
            )
            for layer in layers
        }
        for source_layer, layer_sources in sources_by_layer.items():
            record_layers = [layer for layer in layers if layer >= source_layer]
            for chunk_start in range(0, len(layer_sources), 4):
                chunk = layer_sources[chunk_start : chunk_start + 4]
                deltas, row_metadata = [], []
                for source_position, source_index in chunk:
                    concept_index = source_index % len(concepts)
                    source_std = float(
                        raw_coordinate_std[layer_to_index[source_layer], concept_index]
                    )
                    for dose_index, dose in enumerate(DOSES):
                        requested = float(dose * source_std)
                        deltas.append(
                            dual_by_layer[source_layer][:, concept_index] * requested
                        )
                        row_metadata.append(
                            (
                                source_position,
                                source_index,
                                concept_index,
                                dose_index,
                                float(dose),
                                requested,
                            )
                        )
                changed_acts = patched_capture_batch(
                    model,
                    prompt,
                    source_layer,
                    record_layers,
                    torch.stack(deltas),
                )
                changed_coordinates = {
                    layer: (
                        changed_acts[layer][:, -1].float()
                        @ vectors_by_layer[layer]
                    )
                    for layer in record_layers
                }
                for row_index, (
                    source_position,
                    source_index,
                    concept_index,
                    dose_index,
                    dose,
                    requested,
                ) in enumerate(row_metadata):
                    local_change = (
                        changed_coordinates[source_layer][row_index]
                        - baseline_coordinates[source_layer]
                    ).cpu().numpy()
                    local_std = np.maximum(
                        raw_coordinate_std[layer_to_index[source_layer]], 1e-8
                    )
                    standardized_local = local_change / local_std
                    off_target = np.delete(standardized_local, concept_index)
                    target_error = float(
                        abs(local_change[concept_index] - requested)
                        / local_std[concept_index]
                    )
                    mean_abs_offtarget = float(np.mean(np.abs(off_target)))
                    valid_write = bool(
                        target_error <= 0.1 and mean_abs_offtarget <= 0.1
                    )
                    write_valid[
                        source_position, prompt_index, dose_index
                    ] = valid_write
                    local_rows.append(
                        {
                            "source_index": source_index,
                            "source_layer": source_layer,
                            "source_concept": concepts[concept_index],
                            "prompt_index": prompt_index,
                            "dose_sd": dose,
                            "target_error_sd": target_error,
                            "mean_abs_offtarget_sd": mean_abs_offtarget,
                            "max_abs_offtarget_sd": float(
                                np.max(np.abs(off_target))
                            ),
                            "valid_write": valid_write,
                        }
                    )
                    raw_delta = np.zeros(len(columns), dtype=np.float32)
                    for observed_layer in record_layers:
                        observed_start = (
                            layer_to_index[observed_layer] * len(concepts)
                        )
                        raw_delta[
                            observed_start : observed_start + len(concepts)
                        ] = (
                            changed_coordinates[observed_layer][row_index]
                            - baseline_coordinates[observed_layer]
                        ).cpu().numpy()
                    innovation_delta = apply_delta_transform(
                        raw_delta, transform_entries
                    )
                    for target_layer in layers:
                        if target_layer <= source_layer:
                            continue
                        start = layer_to_index[target_layer] * len(concepts)
                        stop = start + len(concepts)
                        target_raw_std = np.maximum(
                            raw_coordinate_std[layer_to_index[target_layer]], 1e-8
                        )
                        raw_effects[
                            source_position,
                            prompt_index,
                            dose_index,
                            start:stop,
                        ] = raw_delta[start:stop] / target_raw_std
                        effects[
                            source_position,
                            prompt_index,
                            dose_index,
                            start:stop,
                        ] = innovation_delta[start:stop] / np.maximum(
                            innovation_std[start:stop], 1e-8
                        )
        print(f"  held-out prompt {prompt_index + 1}/{len(prompts)}", flush=True)

    pair_sources, pair_targets = [], []
    slope_rows, raw_slope_rows = [], []
    rms_rows, raw_rms_rows, consistency_rows = [], [], []
    valid_prompt_count_rows = []
    for source_position, source_index in enumerate(source_indices):
        source_layer = int(columns[source_index]["layer"])
        valid_prompts = write_valid[source_position].all(axis=1)
        if not valid_prompts.any():
            raise RuntimeError(f"No fully valid prompts for source {source_index}")
        for target_index, target_column in enumerate(columns):
            if int(target_column["layer"]) <= source_layer:
                continue
            pair_effects = effects[
                source_position, valid_prompts, :, target_index
            ]
            pair_raw_effects = raw_effects[
                source_position, valid_prompts, :, target_index
            ]
            valid_slopes = pair_effects @ DOSES / float(np.sum(DOSES**2))
            valid_raw_slopes = (
                pair_raw_effects @ DOSES / float(np.sum(DOSES**2))
            )
            pair_prompt_slopes = np.full(len(prompts), np.nan, dtype=np.float32)
            pair_raw_prompt_slopes = np.full(
                len(prompts), np.nan, dtype=np.float32
            )
            pair_prompt_slopes[valid_prompts] = valid_slopes
            pair_raw_prompt_slopes[valid_prompts] = valid_raw_slopes
            overall_slope = float(np.mean(valid_slopes))
            expected_sign = np.sign(overall_slope) * DOSES[None, :]
            consistency = float(
                np.mean(np.sign(pair_effects) == np.sign(expected_sign))
            )
            pair_sources.append(source_index)
            pair_targets.append(target_index)
            slope_rows.append(pair_prompt_slopes)
            raw_slope_rows.append(pair_raw_prompt_slopes)
            rms_rows.append(float(np.sqrt(np.mean(pair_effects**2))))
            raw_rms_rows.append(float(np.sqrt(np.mean(pair_raw_effects**2))))
            consistency_rows.append(consistency)
            valid_prompt_count_rows.append(int(valid_prompts.sum()))
    pair_sources = np.asarray(pair_sources, dtype=np.int64)
    pair_targets = np.asarray(pair_targets, dtype=np.int64)
    prompt_slopes = np.asarray(slope_rows, dtype=np.float32)
    raw_prompt_slopes = np.asarray(raw_slope_rows, dtype=np.float32)
    rms_effect = np.asarray(rms_rows, dtype=np.float32)
    raw_rms_effect = np.asarray(raw_rms_rows, dtype=np.float32)
    sign_consistency = np.asarray(consistency_rows, dtype=np.float32)
    valid_prompt_count = np.asarray(valid_prompt_count_rows, dtype=np.int64)
    same_concept = np.asarray(
        [
            columns[source]["concept"] == columns[target]["concept"]
            for source, target in zip(pair_sources, pair_targets)
        ],
        dtype=bool,
    )
    cross_concept = ~same_concept

    permutation_rng = np.random.RandomState(20260723)
    sign_flips = permutation_rng.choice(
        [-1.0, 1.0], size=(4096, len(prompts))
    ).astype(np.float32)

    def paired_sign_flip_p(slopes_by_pair: np.ndarray) -> np.ndarray:
        values = []
        for slopes in slopes_by_pair:
            valid = np.isfinite(slopes)
            observed = abs(float(np.mean(slopes[valid])))
            permuted = np.abs(
                sign_flips[:, valid] @ slopes[valid] / int(valid.sum())
            )
            values.append(
                (1 + int(np.sum(permuted >= observed)))
                / (len(sign_flips) + 1)
            )
        return np.asarray(values, dtype=np.float64)

    p_values = paired_sign_flip_p(prompt_slopes)
    q_values = np.ones_like(p_values)
    q_values[cross_concept] = bh_adjust(p_values[cross_concept])
    q_values[same_concept] = bh_adjust(p_values[same_concept])
    effect_positive = (q_values < 0.05) & (rms_effect >= 0.1)

    raw_p_values = paired_sign_flip_p(raw_prompt_slopes)
    raw_q_values = np.ones_like(raw_p_values)
    raw_q_values[cross_concept] = bh_adjust(raw_p_values[cross_concept])
    raw_q_values[same_concept] = bh_adjust(raw_p_values[same_concept])
    raw_effect_positive = (raw_q_values < 0.05) & (raw_rms_effect >= 0.1)

    correlation = np.corrcoef(discovery, rowvar=False)
    causcale_score = mean_probability[pair_sources, pair_targets]
    path_score = path_probability[pair_sources, pair_targets]
    bootstrap_score = selection_frequency[pair_sources, pair_targets]
    correlation_score = np.abs(correlation[pair_sources, pair_targets])
    source_layers = np.asarray(
        [columns[index]["layer"] for index in pair_sources]
    )
    target_layers = np.asarray(
        [columns[index]["layer"] for index in pair_targets]
    )
    architecture_score = 1.0 / (target_layers - source_layers)
    same_concept_score = same_concept.astype(np.float32)
    random_score = np.random.RandomState(20260723).rand(len(pair_sources))
    predictors = {
        "causcale_direct_probability": causcale_score,
        "causcale_max_product_path": path_score,
        "bootstrap_selection_frequency": bootstrap_score,
        "absolute_correlation": correlation_score,
        "same_concept_heuristic": same_concept_score,
        "architecture_only": architecture_score,
        "seeded_random": random_score,
    }
    prediction_metrics_all = {
        name: binary_metrics(effect_positive, score)
        for name, score in predictors.items()
    }
    prediction_metrics_cross_concept = {
        name: binary_metrics(effect_positive[cross_concept], score[cross_concept])
        for name, score in predictors.items()
        if name != "same_concept_heuristic"
    }

    stable_predicted = (causcale_score >= 0.5) & (bootstrap_score >= 0.8)
    stable_predicted_cross_concept = stable_predicted & cross_concept
    primary_metrics = prediction_metrics_cross_concept[
        "causcale_max_product_path"
    ]
    baseline_names = ["absolute_correlation", "architecture_only", "seeded_random"]
    enough_cross_concept_positives = primary_metrics["positive_pairs"] >= 10
    baseline_auprcs = [
        prediction_metrics_cross_concept[name]["auprc"]
        for name in baseline_names
    ]
    path_auprc = primary_metrics["auprc"]
    stage1_go = bool(
        enough_cross_concept_positives
        and path_auprc is not None
        and all(
            baseline is not None and path_auprc > baseline
            for baseline in baseline_auprcs
        )
    )
    if stage1_go:
        stage1_decision = "GO"
    elif not enough_cross_concept_positives:
        stage1_decision = "NO-GO_INSUFFICIENT_CROSS_CONCEPT_POSITIVES"
    else:
        stage1_decision = "NO-GO_DID_NOT_BEAT_ALL_BASELINES"
    pair_rows = []
    for index, (source, target) in enumerate(zip(pair_sources, pair_targets)):
        pair_rows.append(
            {
                "source_index": int(source),
                "target_index": int(target),
                "source_layer": int(columns[source]["layer"]),
                "target_layer": int(columns[target]["layer"]),
                "source_concept": columns[source]["concept"],
                "target_concept": columns[target]["concept"],
                "same_concept": bool(same_concept[index]),
                "valid_prompt_count": int(valid_prompt_count[index]),
                "innovation_mean_prompt_slope": float(
                    np.nanmean(prompt_slopes[index])
                ),
                "innovation_rms_standardized_effect": float(rms_effect[index]),
                "raw_mean_prompt_slope": float(np.nanmean(raw_prompt_slopes[index])),
                "raw_rms_standardized_effect": float(raw_rms_effect[index]),
                "sign_consistency": float(sign_consistency[index]),
                "permutation_p": float(p_values[index]),
                "groupwise_bh_q": float(q_values[index]),
                "innovation_effect_positive": bool(effect_positive[index]),
                "raw_permutation_p": float(raw_p_values[index]),
                "raw_groupwise_bh_q": float(raw_q_values[index]),
                "raw_effect_positive": bool(raw_effect_positive[index]),
                "causcale_direct_probability": float(causcale_score[index]),
                "causcale_max_product_path": float(path_score[index]),
                "bootstrap_selection_frequency": float(bootstrap_score[index]),
                "absolute_innovation_correlation": float(
                    correlation_score[index]
                ),
                "stable_predicted_direct_edge": bool(stable_predicted[index]),
            }
        )

    local_target_errors = [row["target_error_sd"] for row in local_rows]
    local_offtargets = [row["mean_abs_offtarget_sd"] for row in local_rows]
    stable_effects = rms_effect[stable_predicted_cross_concept]
    unrelated_effects = rms_effect[(~stable_predicted) & cross_concept]
    same_concept_raw_effects = raw_rms_effect[same_concept]
    record = {
        "status": "corrected_stage1_cross_concept_innovation_validation",
        "stage1_decision": stage1_decision,
        "stage1_go": stage1_go,
        "stage1_gate": {
            "minimum_cross_concept_positive_pairs": 10,
            "observed_cross_concept_positive_pairs": int(
                effect_positive[cross_concept].sum()
            ),
            "requires_path_auprc_above": baseline_names,
            "enough_cross_concept_positives": enough_cross_concept_positives,
        },
        "innovation_transform": str(args.innovation_transform),
        "innovation_transform_method": innovation_transform["method"],
        "primary_graph_score": (
            "max over layer-ordered cross-concept paths of the product of "
            "CauScale mean edge probabilities; same-concept edges excluded"
        ),
        "source_selection_rule": source_selection_rule,
        "source_indices": source_indices,
        "source_nodes": [columns[index] for index in source_indices],
        "heldout_corpus": str(args.heldout_corpus),
        "heldout_corpus_sha256": corpus_hash,
        "heldout_forbidden_target_concepts": concepts,
        "heldout_token_id_leakage_count": 0,
        "heldout_prompt_count": len(prompts),
        "heldout_article_groups": prompt_groups,
        "heldout_prompts": prompts,
        "doses_sd": DOSES.tolist(),
        "evaluated_pair_count": len(pair_rows),
        "cross_concept_pair_count": int(cross_concept.sum()),
        "same_concept_pair_count": int(same_concept.sum()),
        "innovation_effect_positive_pair_count": int(effect_positive.sum()),
        "innovation_cross_concept_positive_pair_count": int(
            effect_positive[cross_concept].sum()
        ),
        "raw_same_concept_positive_control_count": int(
            raw_effect_positive[same_concept].sum()
        ),
        "effect_definition": (
            "Within cross-concept and same-concept families separately: BH "
            "q<0.05 from 4096 paired prompt sign-flips and RMS standardized "
            "effect >=0.1. Source-prompt groups with any invalid dose are "
            "excluded. The primary endpoint uses innovation effects for "
            "cross-concept pairs; raw same-concept effects are controls only."
        ),
        "write_audit": {
            "n": len(local_rows),
            "invalid_n": int((~write_valid).sum()),
            "invalid_rows": [
                row for row in local_rows if not row["valid_write"]
            ],
            "valid_prompt_count_min": int(valid_prompt_count.min()),
            "median_target_error_sd": float(np.median(local_target_errors)),
            "p95_target_error_sd": float(
                np.percentile(local_target_errors, 95)
            ),
            "median_mean_abs_offtarget_sd": float(
                np.median(local_offtargets)
            ),
            "p95_mean_abs_offtarget_sd": float(
                np.percentile(local_offtargets, 95)
            ),
            "pass_rate": float(
                np.mean(
                    [
                        row["target_error_sd"] <= 0.1
                        and row["mean_abs_offtarget_sd"] <= 0.1
                        for row in local_rows
                    ]
                )
            ),
        },
        "prediction_metrics_all_pairs": prediction_metrics_all,
        "prediction_metrics_cross_concept_primary": (
            prediction_metrics_cross_concept
        ),
        "stable_predicted_cross_concept_edge_count_evaluated": int(
            stable_predicted_cross_concept.sum()
        ),
        "stable_predicted_cross_concept_positive_rate": float(
            effect_positive[stable_predicted_cross_concept].mean()
            if stable_predicted_cross_concept.any()
            else 0.0
        ),
        "stable_predicted_cross_concept_median_rms_effect": float(
            np.median(stable_effects) if len(stable_effects) else 0.0
        ),
        "unrelated_cross_concept_nonedge_median_rms_effect": float(
            np.median(unrelated_effects) if len(unrelated_effects) else 0.0
        ),
        "raw_same_concept_positive_control_median_rms_effect": float(
            np.median(same_concept_raw_effects)
            if len(same_concept_raw_effects)
            else 0.0
        ),
        "raw_same_concept_positive_control_rate": float(
            raw_effect_positive[same_concept].mean()
            if same_concept.any()
            else 0.0
        ),
        "pair_rows": pair_rows,
        "total_seconds": time.time() - started,
        "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
        "interpretation": (
            "Single-source interventions test total downstream effects or "
            "reachability, not direct edges. Residualization removes only the "
            "discovery-predictable same-concept carryover and does not ensure "
            "causal sufficiency."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{args.output_stem}.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        args.output_dir / f"{args.output_stem}.npz",
        effects=effects,
        raw_effects=raw_effects,
        write_valid=write_valid,
        valid_prompt_count=valid_prompt_count,
        source_indices=np.asarray(source_indices),
        pair_sources=pair_sources,
        pair_targets=pair_targets,
        cross_concept=cross_concept,
        prompt_slopes=prompt_slopes,
        raw_prompt_slopes=raw_prompt_slopes,
        rms_effect=rms_effect,
        raw_rms_effect=raw_rms_effect,
        sign_consistency=sign_consistency,
        p_values=p_values,
        q_values=q_values,
        raw_p_values=raw_p_values,
        raw_q_values=raw_q_values,
        effect_positive=effect_positive,
        raw_effect_positive=raw_effect_positive,
        max_product_path_probability=path_probability,
    )
    print(json.dumps(record, ensure_ascii=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
