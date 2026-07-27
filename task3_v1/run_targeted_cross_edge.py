"""Targeted held-out test of the sole stable cross-concept pilot edge.

The edge risk@24 -> danger@30 was selected from the frozen 20-bootstrap graph.
This script tests it separately from the 16-source primary evaluation.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jlens
import numpy as np
import torch
import transformers

from run_heldout_graph_validation import (
    RIDGE,
    bh_adjust,
    capture,
    heldout_prompts,
    patched_capture_batch,
)

MODEL_NAME = "Qwen/Qwen3.5-4B"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_REVISION = "qwen-n1000"
LENS_FILE = (
    "qwen3.5-4b/jlens/Salesforce-wikitext/"
    "Qwen3.5-4B_jacobian_lens_n1000.pt"
)
DOSES = np.asarray([-2.0, -1.0, 1.0, 2.0], dtype=np.float32)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heldout-corpus", type=Path, required=True)
    parser.add_argument("--discovery-metadata", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--bootstrap-npz", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("task3") / "outputs" / "validation",
    )
    args = parser.parse_args()
    for path in [
        args.heldout_corpus,
        args.discovery_metadata,
        args.calibration,
        args.bootstrap_npz,
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    metadata = json.loads(args.discovery_metadata.read_text(encoding="utf-8"))
    columns = metadata["columns"]
    concepts = [column["concept"] for column in columns[:32]]
    layers = sorted({int(column["layer"]) for column in columns})
    layer_to_index = {layer: index for index, layer in enumerate(layers)}
    coordinate_std = np.load(args.calibration)["coordinates"].std(axis=0, ddof=1)
    bootstrap = np.load(args.bootstrap_npz)

    source_layer = 24
    target_layer = 30
    source_concept = " risk"
    target_concept = " danger"
    source_concept_index = concepts.index(source_concept)
    target_concept_index = concepts.index(target_concept)
    source_index = (
        layer_to_index[source_layer] * len(concepts) + source_concept_index
    )
    target_index = (
        layer_to_index[target_layer] * len(concepts) + target_concept_index
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
        args.heldout_corpus,
        tokenizer,
        [source_concept, target_concept],
        count=20,
        max_tokens=128,
    )

    vectors = {}
    for layer in [source_layer, target_layer]:
        jacobian = lens.jacobians[layer].to(device="cuda", dtype=torch.float32)
        raw_vectors = jacobian.T @ model._lm_head.weight[token_ids].float().T
        vectors[layer] = torch.nn.functional.normalize(raw_vectors, dim=0)
    gram = vectors[source_layer].T @ vectors[source_layer]
    dual = vectors[source_layer] @ torch.linalg.inv(
        gram + RIDGE * torch.eye(len(concepts), device="cuda")
    )

    downstream = np.empty((len(prompts), len(DOSES), len(concepts)), np.float32)
    local_rows = []
    source_std = float(
        coordinate_std[layer_to_index[source_layer], source_concept_index]
    )
    target_std = np.maximum(
        coordinate_std[layer_to_index[target_layer]], 1e-8
    )
    print("Testing frozen risk@24 -> danger@30 candidate...", flush=True)
    for prompt_index, prompt in enumerate(prompts):
        baseline = capture(model, prompt, [source_layer, target_layer])
        baseline_source = (
            baseline[source_layer][0, -1].float() @ vectors[source_layer]
        )
        baseline_target = (
            baseline[target_layer][0, -1].float() @ vectors[target_layer]
        )
        deltas = torch.stack(
            [
                dual[:, source_concept_index] * float(dose * source_std)
                for dose in DOSES
            ]
        )
        changed = patched_capture_batch(
            model,
            prompt,
            source_layer,
            [source_layer, target_layer],
            deltas,
        )
        source_changes = (
            changed[source_layer][:, -1].float() @ vectors[source_layer]
            - baseline_source[None]
        ).cpu().numpy()
        target_changes = (
            changed[target_layer][:, -1].float() @ vectors[target_layer]
            - baseline_target[None]
        ).cpu().numpy()
        downstream[prompt_index] = target_changes / target_std[None]
        source_scale = np.maximum(
            coordinate_std[layer_to_index[source_layer]], 1e-8
        )
        for dose_index, dose in enumerate(DOSES):
            standardized = source_changes[dose_index] / source_scale
            local_rows.append(
                {
                    "prompt_index": prompt_index,
                    "dose_sd": float(dose),
                    "target_error_sd": float(
                        abs(
                            source_changes[dose_index, source_concept_index]
                            - dose * source_std
                        )
                        / source_std
                    ),
                    "mean_abs_offtarget_sd": float(
                        np.mean(
                            np.abs(
                                np.delete(standardized, source_concept_index)
                            )
                        )
                    ),
                }
            )
        print(f"  targeted prompt {prompt_index + 1}/20", flush=True)

    prompt_slopes = np.tensordot(
        downstream, DOSES, axes=([1], [0])
    ) / float(np.sum(DOSES**2))
    rms = np.sqrt(np.mean(downstream**2, axis=(0, 1)))
    observed = np.abs(prompt_slopes.mean(axis=0))
    rng = np.random.RandomState(20260723)
    flips = rng.choice([-1.0, 1.0], size=(4096, len(prompts))).astype(np.float32)
    permuted = np.abs(flips @ prompt_slopes / len(prompts))
    p_values = (1 + np.sum(permuted >= observed[None], axis=0)) / (
        len(flips) + 1
    )
    q_values = bh_adjust(p_values)

    rows = []
    for concept_index, concept in enumerate(concepts):
        slope = float(prompt_slopes[:, concept_index].mean())
        expected_sign = np.sign(slope) * DOSES[None]
        rows.append(
            {
                "target_concept": concept,
                "mean_prompt_slope": slope,
                "rms_standardized_effect": float(rms[concept_index]),
                "sign_consistency": float(
                    np.mean(
                        np.sign(downstream[:, :, concept_index])
                        == np.sign(expected_sign)
                    )
                ),
                "permutation_p": float(p_values[concept_index]),
                "bh_q_across_32_targets": float(q_values[concept_index]),
                "effect_positive": bool(
                    q_values[concept_index] < 0.05
                    and rms[concept_index] >= 0.1
                ),
            }
        )
    danger_result = rows[target_concept_index]
    risk_result = rows[source_concept_index]
    local_errors = [row["target_error_sd"] for row in local_rows]
    local_offtargets = [row["mean_abs_offtarget_sd"] for row in local_rows]
    record = {
        "status": "targeted_frozen_cross_edge_test",
        "source": {
            "index": source_index,
            "layer": source_layer,
            "concept": source_concept,
        },
        "target": {
            "index": target_index,
            "layer": target_layer,
            "concept": target_concept,
        },
        "bootstrap_mean_probability": float(
            bootstrap["mean_probability"][source_index, target_index]
        ),
        "bootstrap_selection_frequency": float(
            bootstrap["selection_frequency"][source_index, target_index]
        ),
        "heldout_corpus_sha256": corpus_hash,
        "heldout_prompts": prompts,
        "heldout_article_groups": prompt_groups,
        "doses_sd": DOSES.tolist(),
        "risk_to_danger": danger_result,
        "risk_to_risk_positive_control": risk_result,
        "all_layer30_targets": rows,
        "write_audit": {
            "n": len(local_rows),
            "median_target_error_sd": float(np.median(local_errors)),
            "p95_target_error_sd": float(np.percentile(local_errors, 95)),
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
        "total_seconds": time.time() - started,
        "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
        "interpretation": (
            "This is a preselected cross-concept total-effect test. A null "
            "result would reject this candidate edge at the current dose and "
            "coordinate definition, not prove the absence of every mechanism."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "targeted_risk_to_danger.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        args.output_dir / "targeted_risk_to_danger.npz",
        downstream=downstream,
        prompt_slopes=prompt_slopes,
        rms=rms,
        p_values=p_values,
        q_values=q_values,
    )
    print(json.dumps(record, ensure_ascii=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
