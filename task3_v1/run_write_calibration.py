"""Batch calibration of Task 3's static J-lens coordinates and write operators.

This is a local measurement/write audit. It does not establish causal edges.
"""
from __future__ import annotations

import argparse
import json
import platform
import re
import time
from collections import defaultdict
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

CONCEPTS = [
    " water",
    " fire",
    " music",
    " danger",
    " Italy",
    " code",
    " animal",
    " happy",
    " money",
    " doctor",
    " city",
    " truth",
    " false",
    " love",
    " anger",
    " fear",
    " food",
    " sleep",
    " work",
    " school",
    " family",
    " war",
    " peace",
    " science",
    " art",
    " language",
    " number",
    " time",
    " future",
    " past",
    " safe",
    " risk",
]
SOURCE_CONCEPTS = [" water", " fire", " code", " happy"]
WRONG_SOURCE = {" water": " fire", " fire": " water", " code": " happy", " happy": " code"}
ARMS = ["dual_target", "naive_target", "wrong_dual", "random_norm_matched"]
RIDGE = 1e-4

CALIBRATION_STEMS = [
    "A sailor studied dark clouds gathering above the distant harbor.",
    "An engineer reviewed a failing test before changing the program.",
    "A traveler compared two maps before choosing the next destination.",
    "A child listened carefully as the orchestra began its rehearsal.",
    "Thick smoke rose beyond the forest and the hikers moved away.",
    "A chef prepared several drinks after a long afternoon in the kitchen.",
    "A biologist recorded how a creature moved through its habitat.",
    "A student smiled after receiving unexpectedly good news.",
    "A mechanic inspected the engine after hearing a strange sound.",
    "A physician compared the scan with the earlier examination.",
]
CALIBRATION_ENDINGS = [
    "An observer later summarized the event in a notebook.",
    "Several witnesses compared their memories beside the station.",
    "By evening, the group had agreed on the next decision.",
    "The account concluded with a careful description of the museum.",
    "A visitor remembered the episode during the following winter.",
    "The team waited quietly for a final signal.",
    "Someone photographed the scene near the horizon.",
    "The discussion continued after everyone reached the village.",
    "A short summary was presented at the morning meeting.",
    "The details were reconsidered during the return journey.",
]
CALIBRATION_PROMPTS = [
    f"{stem} {ending}" for stem in CALIBRATION_STEMS for ending in CALIBRATION_ENDINGS
]
AUDIT_PROMPTS = [
    "After hours in the sun, the exhausted hiker opened the container.",
    "The analyst found a subtle mistake after checking every line twice.",
    "A bright glow appeared behind the ridge as the campers packed quickly.",
    "The singer stepped onto the stage and waited for the first note.",
    "At the border, the traveler unfolded a map and checked the route.",
    "The nurse entered the quiet room carrying a sealed tray.",
    "When the alarm sounded, everyone followed the marked exit.",
    "The teacher placed the final puzzle on the desk before class.",
    "A sudden result made the entire team smile with relief.",
    "The gardener examined the dry soil and reached for the container.",
]


def validate_inputs(tokenizer) -> list[int]:
    token_ids = []
    for concept in CONCEPTS:
        ids = tokenizer.encode(concept, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Not one token: {concept!r} -> {ids}")
        token_ids.append(ids[0])

    if len(CALIBRATION_PROMPTS) != 100 or len(set(CALIBRATION_PROMPTS)) != 100:
        raise ValueError("Calibration prompts must contain 100 unique entries")
    source_pattern = re.compile(
        r"\b(?:" + "|".join(re.escape(c.strip()) for c in SOURCE_CONCEPTS) + r")\b",
        flags=re.IGNORECASE,
    )
    leaked = [
        prompt
        for prompt in CALIBRATION_PROMPTS + AUDIT_PROMPTS
        if source_pattern.search(prompt)
    ]
    if leaked:
        raise ValueError(f"Source-word leakage in prompts: {leaked}")
    return token_ids


def capture(model, prompt: str, at: list[int]):
    ids = model.encode(prompt)
    with ActivationRecorder(model.layers, at=at) as recorder:
        model.forward(ids)
        acts = {layer: recorder.activations[layer].detach() for layer in at}
    return acts


def patched_capture_batch(
    model,
    prompt: str,
    layer: int,
    final_layer: int,
    deltas: torch.Tensor,
):
    ids = model.encode(prompt).repeat(deltas.shape[0], 1)

    def patch_hook(module, inputs, output):
        tensor = output if torch.is_tensor(output) else output[0]
        changed = tensor.clone()
        changed[:, -1, :] += deltas.to(device=tensor.device, dtype=tensor.dtype)
        if torch.is_tensor(output):
            return changed
        return (changed, *output[1:])

    handle = model.layers[layer].register_forward_hook(patch_hook)
    try:
        with ActivationRecorder(model.layers, at=[layer, final_layer]) as recorder:
            model.forward(ids)
            return {
                at: recorder.activations[at].detach() for at in [layer, final_layer]
            }
    finally:
        handle.remove()


def selected_unembed(model, residual: torch.Tensor, token_ids: list[int]) -> torch.Tensor:
    """Apply the model's final norm and only the requested unembedding rows."""
    target_dtype = model._lm_head.weight.dtype
    target_device = model._lm_head.weight.device
    normalized = model._final_norm(residual.to(target_device, dtype=target_dtype))
    weight = model._lm_head.weight[token_ids]
    logits = normalized @ weight.T
    if model._lm_head.bias is not None:
        logits = logits + model._lm_head.bias[token_ids]
    if model._logit_softcap is not None:
        logits = model._logit_softcap * torch.tanh(logits / model._logit_softcap)
    return logits.float()


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def aggregate_rows(rows: list[dict]) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["arm"]].append(row)

    result = {}
    for arm, arm_rows in grouped.items():
        target_errors = [row["target_error_sd"] for row in arm_rows]
        off_targets = [row["mean_abs_offtarget_sd"] for row in arm_rows]
        result[arm] = {
            "n": len(arm_rows),
            "median_target_error_sd": float(np.median(target_errors)),
            "p95_target_error_sd": percentile(target_errors, 95),
            "median_mean_abs_offtarget_sd": float(np.median(off_targets)),
            "p95_mean_abs_offtarget_sd": percentile(off_targets, 95),
            "direction_correct_rate": float(
                np.mean([row["direction_correct"] for row in arm_rows])
            ),
            "coordinate_pass_rate": float(
                np.mean(
                    [
                        row["target_error_sd"] <= 0.1
                        and row["mean_abs_offtarget_sd"] <= 0.1
                        for row in arm_rows
                    ]
                )
            ),
            "mean_signed_target_output_delta_logit": float(
                np.mean(
                    [row["signed_target_output_delta_logit"] for row in arm_rows]
                )
            ),
            "mean_output_selectivity": float(
                np.mean([row["output_selectivity"] for row in arm_rows])
            ),
        }
    return result


def aggregate_dual_breakdown(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        if row["arm"] == "dual_target":
            groups[(row["layer"], row["source_concept"])].append(row)
    result = []
    for (layer, source), group in sorted(groups.items()):
        result.append(
            {
                "layer": layer,
                "source_concept": source,
                "n": len(group),
                "median_target_error_sd": float(
                    np.median([row["target_error_sd"] for row in group])
                ),
                "median_mean_abs_offtarget_sd": float(
                    np.median([row["mean_abs_offtarget_sd"] for row in group])
                ),
                "direction_correct_rate": float(
                    np.mean([row["direction_correct"] for row in group])
                ),
                "coordinate_pass_rate": float(
                    np.mean(
                        [
                            row["target_error_sd"] <= 0.1
                            and row["mean_abs_offtarget_sd"] <= 0.1
                            for row in group
                        ]
                    )
                ),
                "mean_signed_target_output_delta_logit": float(
                    np.mean(
                        [row["signed_target_output_delta_logit"] for row in group]
                    )
                ),
            }
        )
    return result


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("task3") / "outputs" / "calibration",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    print("Loading cached model and Jacobian lens...", flush=True)
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16
    ).cuda()
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    model = jlens.from_hf(hf_model, tokenizer)
    lens = jlens.JacobianLens.from_pretrained(
        LENS_REPO, filename=LENS_FILE, revision=LENS_REVISION
    )
    layers = [
        model.n_layers // 4,
        model.n_layers // 2,
        model.n_layers // 4 * 3,
        model.n_layers - 2,
    ]
    final_layer = model.n_layers - 1
    token_ids = validate_inputs(tokenizer)
    source_indices = [CONCEPTS.index(concept) for concept in SOURCE_CONCEPTS]

    print("Building 32-dimensional static coordinates and ridge duals...", flush=True)
    vectors_by_layer = {}
    dual_by_layer = {}
    jacobians = {}
    condition_numbers = {}
    for layer in layers:
        jacobian = lens.jacobians[layer].to(device="cuda", dtype=torch.float32)
        raw_vectors = jacobian.T @ model._lm_head.weight[token_ids].float().T
        vectors = torch.nn.functional.normalize(raw_vectors, dim=0)
        gram = vectors.T @ vectors
        dual = vectors @ torch.linalg.inv(
            gram + RIDGE * torch.eye(len(CONCEPTS), device="cuda")
        )
        jacobians[layer] = jacobian
        vectors_by_layer[layer] = vectors
        dual_by_layer[layer] = dual
        condition_numbers[layer] = float(torch.linalg.cond(gram).item())

    print("Calibrating on 100 source-word-free prompts...", flush=True)
    coordinate_rows = {layer: [] for layer in layers}
    direct_jlens_rows = {layer: [] for layer in layers}
    calibration_seconds = []
    for prompt_index, prompt in enumerate(CALIBRATION_PROMPTS):
        tick = time.time()
        acts = capture(model, prompt, layers)
        for layer in layers:
            residual = acts[layer][0, -1].float()
            coordinate_rows[layer].append(
                (residual @ vectors_by_layer[layer]).cpu().numpy()
            )
            transported = residual @ jacobians[layer].T
            direct_jlens_rows[layer].append(
                selected_unembed(model, transported[None], token_ids)[0]
                .cpu()
                .numpy()
            )
        torch.cuda.synchronize()
        calibration_seconds.append(time.time() - tick)
        if (prompt_index + 1) % 20 == 0:
            print(f"  calibrated {prompt_index + 1}/100", flush=True)

    coordinate_arrays = {
        layer: np.asarray(coordinate_rows[layer], dtype=np.float32)
        for layer in layers
    }
    direct_jlens_arrays = {
        layer: np.asarray(direct_jlens_rows[layer], dtype=np.float32)
        for layer in layers
    }
    coordinate_std = {
        layer: coordinate_arrays[layer].std(axis=0, ddof=1) for layer in layers
    }
    direct_jlens_std = {
        layer: direct_jlens_arrays[layer].std(axis=0, ddof=1) for layer in layers
    }
    if any(np.any(std < 1e-8) for std in coordinate_std.values()):
        raise RuntimeError("At least one normalized coordinate has zero variance")

    correlations = []
    for layer in layers:
        for concept_index, concept in enumerate(CONCEPTS):
            correlations.append(
                {
                    "layer": layer,
                    "concept": concept,
                    "pearson": float(
                        np.corrcoef(
                            coordinate_arrays[layer][:, concept_index],
                            direct_jlens_arrays[layer][:, concept_index],
                        )[0, 1]
                    ),
                }
            )

    print("Running 1,280 batched interventions and controls...", flush=True)
    rows = []
    for prompt_index, prompt in enumerate(AUDIT_PROMPTS):
        baseline_acts = capture(model, prompt, layers + [final_layer])
        baseline_output = (
            selected_unembed(
                model, baseline_acts[final_layer][0, -1].float()[None], token_ids
            )[0]
            .cpu()
            .numpy()
        )
        for layer in layers:
            baseline_h = baseline_acts[layer][0, -1].float()
            baseline_coordinates = baseline_h @ vectors_by_layer[layer]
            deltas = []
            metadata = []
            for source_index in source_indices:
                source = CONCEPTS[source_index]
                wrong_index = CONCEPTS.index(WRONG_SOURCE[source])
                generator = torch.Generator(device="cuda")
                generator.manual_seed(20260723 + layer * 100 + source_index)
                random_direction = torch.randn(
                    model.d_model,
                    generator=generator,
                    device="cuda",
                    dtype=torch.float32,
                )
                target_vector = vectors_by_layer[layer][:, source_index]
                random_direction -= (
                    random_direction @ target_vector
                ) * target_vector
                random_direction = torch.nn.functional.normalize(
                    random_direction, dim=0
                )
                for dose in (-1.0, 1.0):
                    requested = dose * float(coordinate_std[layer][source_index])
                    target_delta = dual_by_layer[layer][:, source_index] * requested
                    wrong_direction = dual_by_layer[layer][:, wrong_index]
                    signed_control_norm = target_delta.norm() * dose
                    arm_deltas = {
                        "dual_target": target_delta,
                        "naive_target": target_vector * requested,
                        "wrong_dual": (
                            wrong_direction
                            / torch.clamp(wrong_direction.norm(), min=1e-12)
                            * signed_control_norm
                        ),
                        "random_norm_matched": (
                            random_direction * signed_control_norm
                        ),
                    }
                    for arm in ARMS:
                        deltas.append(arm_deltas[arm])
                        metadata.append(
                            {
                                "prompt_index": prompt_index,
                                "prompt": prompt,
                                "layer": layer,
                                "source_concept": source,
                                "source_index": source_index,
                                "dose_sd": dose,
                                "requested_coordinate_change": requested,
                                "arm": arm,
                            }
                        )
            delta_batch = torch.stack(deltas)
            changed_acts = patched_capture_batch(
                model, prompt, layer, final_layer, delta_batch
            )
            changed_coordinates = (
                changed_acts[layer][:, -1].float() @ vectors_by_layer[layer]
            )
            coordinate_changes = (
                changed_coordinates - baseline_coordinates[None]
            ).cpu().numpy()
            changed_output = (
                selected_unembed(
                    model, changed_acts[final_layer][:, -1].float(), token_ids
                )
                .cpu()
                .numpy()
            )
            output_changes = changed_output - baseline_output[None]

            for row_index, meta in enumerate(metadata):
                source_index = meta.pop("source_index")
                requested = meta["requested_coordinate_change"]
                std = np.maximum(coordinate_std[layer], 1e-8)
                standardized = coordinate_changes[row_index] / std
                off_target = np.delete(standardized, source_index)
                target_change = float(coordinate_changes[row_index, source_index])
                signed_output = meta["dose_sd"] * output_changes[row_index]
                signed_distractors = np.delete(signed_output, source_index)
                rows.append(
                    {
                        **meta,
                        "achieved_coordinate_change": target_change,
                        "achieved_target_change_sd": float(
                            target_change / std[source_index]
                        ),
                        "target_error_sd": float(
                            abs(target_change - requested) / std[source_index]
                        ),
                        "mean_abs_offtarget_sd": float(
                            np.mean(np.abs(off_target))
                        ),
                        "max_abs_offtarget_sd": float(np.max(np.abs(off_target))),
                        "direction_correct": bool(target_change * requested > 0),
                        "delta_residual_norm": float(
                            delta_batch[row_index].norm().item()
                        ),
                        "signed_target_output_delta_logit": float(
                            signed_output[source_index]
                        ),
                        "signed_mean_distractor_output_delta_logit": float(
                            np.mean(signed_distractors)
                        ),
                        "output_selectivity": float(
                            signed_output[source_index]
                            - np.mean(signed_distractors)
                        ),
                    }
                )
        print(f"  audited prompt {prompt_index + 1}/10", flush=True)

    aggregate = aggregate_rows(rows)
    dual_breakdown = aggregate_dual_breakdown(rows)
    dual_metrics = aggregate["dual_target"]
    control_coordinate_pass = max(
        aggregate[arm]["coordinate_pass_rate"]
        for arm in ["wrong_dual", "random_norm_matched"]
    )
    gate = {
        "thresholds": {
            "dual_coordinate_pass_rate_min": 0.90,
            "dual_direction_correct_rate_min": 0.95,
            "dual_median_mean_abs_offtarget_sd_max": 0.10,
            "control_coordinate_pass_rate_max": 0.10,
        },
        "dual_coordinate_pass_rate_ok": (
            dual_metrics["coordinate_pass_rate"] >= 0.90
        ),
        "dual_direction_correct_rate_ok": (
            dual_metrics["direction_correct_rate"] >= 0.95
        ),
        "dual_offtarget_ok": (
            dual_metrics["median_mean_abs_offtarget_sd"] <= 0.10
        ),
        "controls_separated": control_coordinate_pass <= 0.10,
    }
    gate["pass"] = all(value for key, value in gate.items() if key != "thresholds")

    corr_values = [row["pearson"] for row in correlations]
    all_coordinate_std = np.concatenate([coordinate_std[layer] for layer in layers])
    all_direct_std = np.concatenate([direct_jlens_std[layer] for layer in layers])
    record = {
        "status": "local_write_calibration_not_causal_evidence",
        "model": MODEL_NAME,
        "lens_revision": LENS_REVISION,
        "layers": layers,
        "concepts": [
            {"concept": concept, "token_id": token_id}
            for concept, token_id in zip(CONCEPTS, token_ids)
        ],
        "source_concepts": SOURCE_CONCEPTS,
        "arms": ARMS,
        "calibration_prompt_count": len(CALIBRATION_PROMPTS),
        "audit_prompt_count": len(AUDIT_PROMPTS),
        "intervention_row_count": len(rows),
        "ridge": RIDGE,
        "gram_condition_number_by_layer": condition_numbers,
        "calibration": {
            "coordinate_std_min": float(all_coordinate_std.min()),
            "coordinate_std_median": float(np.median(all_coordinate_std)),
            "coordinate_std_max": float(all_coordinate_std.max()),
            "direct_jlens_std_min": float(all_direct_std.min()),
            "direct_jlens_std_median": float(np.median(all_direct_std)),
            "direct_jlens_std_max": float(all_direct_std.max()),
            "coordinate_direct_jlens_pearson_min": float(np.min(corr_values)),
            "coordinate_direct_jlens_pearson_median": float(
                np.median(corr_values)
            ),
            "coordinate_direct_jlens_pearson_max": float(np.max(corr_values)),
            "mean_prompt_seconds": float(np.mean(calibration_seconds)),
        },
        "aggregate_by_arm": aggregate,
        "dual_breakdown": dual_breakdown,
        "gate": gate,
        "total_seconds": time.time() - started,
        "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "write_calibration_32x100.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "write_calibration_rows.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    np.savez_compressed(
        args.output_dir / "write_calibration_32x100.npz",
        prompts=np.asarray(CALIBRATION_PROMPTS),
        layers=np.asarray(layers),
        concepts=np.asarray(CONCEPTS),
        coordinates=np.stack([coordinate_arrays[layer] for layer in layers], axis=1),
        direct_jlens_logits=np.stack(
            [direct_jlens_arrays[layer] for layer in layers], axis=1
        ),
    )
    print(json.dumps(record, ensure_ascii=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
