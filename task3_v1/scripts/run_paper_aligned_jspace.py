"""Run the held-out paper-aligned J-space read-and-swap experiment on Qwen.

This is a substrate sanity check, separate from the main Task 3 Stage 1
artifacts and from its ridge-dual writer.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as functional

from paper_aligned_core import (
    EXPERIMENT_LABEL,
    UPSTREAM_COMMIT,
    aggregate_readout,
    aggregate_swap_rows,
    coordinate_swap,
    exact_command,
    git_snapshot,
    load_config,
    normalized_depth,
    package_versions,
    read_jsonl,
    repo_root,
    resolve_repo_path,
    select_workspace_band,
    sha256_file,
    token_rank,
    wilson_interval,
    write_aggregate_csv,
    write_json,
    write_jsonl,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_components(config: dict[str, Any], root: Path):
    import jlens
    import transformers

    model_config = config["model"]
    lens_config = config["lens"]
    device = torch.device(model_config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Configured CUDA device is unavailable")
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[model_config["dtype"]]
    local_model = resolve_repo_path(root, model_config.get("local_path"))
    model_source = (
        str(local_model)
        if local_model is not None and local_model.is_dir()
        else model_config["id"]
    )
    local_revision_verified = False
    if local_model is not None and local_model.is_dir():
        revision_marker = (
            local_model
            / ".cache"
            / "huggingface"
            / "trees"
            / f"{model_config['revision']}.json"
        )
        if not revision_marker.is_file():
            raise RuntimeError(
                "Local model cache does not contain the configured revision marker: "
                f"{revision_marker}"
            )
        local_revision_verified = True
    model_kwargs: dict[str, Any] = {"dtype": dtype}
    if model_source == model_config["id"]:
        model_kwargs["revision"] = model_config["revision"]
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_source, **model_kwargs
    )
    hf_model.to(device)
    tokenizer_source = model_source if local_model and local_model.is_dir() else model_config["tokenizer_id"]
    tokenizer_kwargs = {}
    if tokenizer_source == model_config["tokenizer_id"]:
        tokenizer_kwargs["revision"] = model_config["tokenizer_revision"]
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_source, **tokenizer_kwargs
    )
    model = jlens.from_hf(hf_model, tokenizer)

    local_lens = resolve_repo_path(root, lens_config.get("local_path"))
    if local_lens is not None and local_lens.is_file():
        if sha256_file(local_lens) != lens_config["sha256"]:
            raise RuntimeError("Local J-lens checkpoint hash does not match config")
        lens = jlens.JacobianLens.from_pretrained(str(local_lens))
        lens_source = str(local_lens)
    else:
        lens = jlens.JacobianLens.from_pretrained(
            lens_config["repository"],
            filename=lens_config["file"],
            revision=lens_config["revision"],
        )
        lens_source = f"{lens_config['repository']}@{lens_config['revision']}"
    return model, tokenizer, lens, {
        "model_source": model_source,
        "tokenizer_source": tokenizer_source,
        "lens_source": lens_source,
        "local_model_revision_verified": local_revision_verified,
        "device": str(device),
        "dtype": str(dtype),
    }


def cache_jacobians(lens, layers: Sequence[int], device: torch.device):
    return {
        int(layer): lens.jacobians[int(layer)].to(
            device=device, dtype=torch.float32
        )
        for layer in layers
    }


def concept_vector(
    model,
    jacobian: torch.Tensor,
    token_id: int,
) -> torch.Tensor:
    unembedding = model._lm_head.weight[int(token_id)].float()
    return functional.normalize(jacobian.T @ unembedding, dim=0)


@torch.no_grad()
def clean_readout(
    model,
    jacobians: dict[int, torch.Tensor],
    example: dict[str, Any],
    *,
    max_fitted_layer: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from jlens.hooks import ActivationRecorder

    layers = sorted(jacobians)
    final_layer = model.n_layers - 1
    input_ids = model.encode(example["prompt"])
    with ActivationRecorder(
        model.layers, at=sorted(set(layers + [final_layer]))
    ) as recorder:
        model.forward(input_ids)
        activations = {
            layer: recorder.activations[layer].detach()
            for layer in sorted(set(layers + [final_layer]))
        }

    target_id = int(example["source_concept_token_id"])
    rows = []
    for layer in layers:
        residual = activations[layer][0, -1].float()
        transported = residual @ jacobians[layer].T
        jlens_logits = model.unembed(transported[None])[0].float()
        logit_lens_logits = model.unembed(residual[None])[0].float()
        vector = concept_vector(model, jacobians[layer], target_id)
        jlens_rank = token_rank(jlens_logits, target_id)
        logit_rank = token_rank(logit_lens_logits, target_id)
        rows.append(
            {
                "example_id": example["example_id"],
                "split": example["split"],
                "eligibility_status": example["eligibility_status"],
                "relation_or_task_family": example["relation_or_task_family"],
                "layer": layer,
                "normalized_depth": normalized_depth(layer, max_fitted_layer),
                "measurement_token_position": -1,
                "source_intermediate_concept": example[
                    "source_intermediate_concept"
                ],
                "source_concept_token_id": target_id,
                "jlens_pre_softmax_score": float(
                    jlens_logits[target_id].item()
                ),
                "jlens_cosine_similarity": float(
                    functional.cosine_similarity(residual, vector, dim=0).item()
                ),
                "jlens_rank": jlens_rank,
                "jlens_top1": jlens_rank <= 1,
                "jlens_top5": jlens_rank <= 5,
                "jlens_top10": jlens_rank <= 10,
                "logit_lens_pre_softmax_score": float(
                    logit_lens_logits[target_id].item()
                ),
                "logit_lens_rank": logit_rank,
                "logit_lens_top1": logit_rank <= 1,
                "logit_lens_top5": logit_rank <= 5,
                "logit_lens_top10": logit_rank <= 10,
                "vocab_size": int(jlens_logits.numel()),
            }
        )

    final_residual = activations[final_layer][0, -1].float()
    final_logits = model.unembed(final_residual[None])[0].float().detach()
    clean_answer_id = int(example["clean_answer_token_id"])
    swapped_answer_id = int(example["swapped_answer_token_id"])
    clean_log_probabilities = torch.log_softmax(final_logits, dim=-1)
    return rows, {
        "input_token_count": int(input_ids.shape[-1]),
        "logits": final_logits,
        "top1_token_id": int(final_logits.argmax().item()),
        "clean_answer_log_probability": float(
            clean_log_probabilities[clean_answer_id].item()
        ),
        "swapped_answer_log_probability": float(
            clean_log_probabilities[swapped_answer_id].item()
        ),
    }


@torch.no_grad()
def patched_logits(
    model,
    jacobians: dict[int, torch.Tensor],
    prompt: str,
    source_token_id: int,
    target_token_id: int,
    patch_layers: Sequence[int],
    *,
    rcond: float,
    strength: float,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    from jlens.hooks import ActivationRecorder

    diagnostics: list[dict[str, Any]] = []
    handles = []
    for layer in patch_layers:
        source = concept_vector(model, jacobians[layer], source_token_id)
        target = concept_vector(model, jacobians[layer], target_token_id)

        def patch_hook(module, inputs, output, *, layer=layer, source=source, target=target):
            tensor = output if torch.is_tensor(output) else output[0]
            changed, record = coordinate_swap(
                tensor,
                source,
                target,
                rcond=rcond,
                strength=strength,
            )
            record["layer"] = int(layer)
            record["normalized_depth"] = normalized_depth(
                int(layer), max(jacobians)
            )
            diagnostics.append(record)
            if torch.is_tensor(output):
                return changed
            return (changed, *output[1:])

        handles.append(model.layers[layer].register_forward_hook(patch_hook))

    final_layer = model.n_layers - 1
    try:
        input_ids = model.encode(prompt)
        with ActivationRecorder(model.layers, at=[final_layer]) as recorder:
            model.forward(input_ids)
            residual = recorder.activations[final_layer][0, -1].float()
        return model.unembed(residual[None])[0].float().detach(), diagnostics
    finally:
        for handle in handles:
            handle.remove()


def swap_result(
    example: dict[str, Any],
    clean: dict[str, Any],
    swapped_logits: torch.Tensor,
    *,
    intervention_scope: str,
    patch_layers: Sequence[int],
    diagnostics: list[dict[str, Any]],
    tokenizer,
) -> dict[str, Any]:
    clean_logits = clean["logits"]
    clean_answer_id = int(example["clean_answer_token_id"])
    target_answer_id = int(example["swapped_answer_token_id"])
    clean_log_probs = torch.log_softmax(clean_logits, dim=-1)
    swapped_log_probs = torch.log_softmax(swapped_logits, dim=-1)
    clean_margin = (
        clean_log_probs[target_answer_id] - clean_log_probs[clean_answer_id]
    )
    swapped_margin = (
        swapped_log_probs[target_answer_id] - swapped_log_probs[clean_answer_id]
    )
    post_top1 = int(swapped_logits.argmax().item())
    return {
        "example_id": example["example_id"],
        "split": example["split"],
        "relation_or_task_family": example["relation_or_task_family"],
        "intervention_scope": intervention_scope,
        "patch_layers": list(map(int, patch_layers)),
        "source_intermediate_concept": example["source_intermediate_concept"],
        "swap_target_intermediate_concept": example[
            "swap_target_intermediate_concept"
        ],
        "source_concept_token_id": int(example["source_concept_token_id"]),
        "target_concept_token_id": int(example["target_concept_token_id"]),
        "clean_answer_token_id": clean_answer_id,
        "swapped_answer_token_id": target_answer_id,
        "clean_top1_token_id": clean["top1_token_id"],
        "clean_top1_token": tokenizer.decode([clean["top1_token_id"]]),
        "post_swap_top1_token_id": post_top1,
        "post_swap_top1_token": tokenizer.decode([post_top1]),
        "target_answer_top1_success": post_top1 == target_answer_id,
        "original_answer_top1_retained": post_top1 == clean_answer_id,
        "clean_swap_target_log_probability": float(
            clean_log_probs[target_answer_id].item()
        ),
        "post_swap_target_log_probability": float(
            swapped_log_probs[target_answer_id].item()
        ),
        "delta_swap_target_log_probability": float(
            (
                swapped_log_probs[target_answer_id]
                - clean_log_probs[target_answer_id]
            ).item()
        ),
        "clean_log_probability_margin": float(clean_margin.item()),
        "post_swap_log_probability_margin": float(swapped_margin.item()),
        "delta_log_probability_margin": float(
            (swapped_margin - clean_margin).item()
        ),
        "coordinate_diagnostics": diagnostics,
    }


def per_layer_readout(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["layer"])].append(row)
    result = {}
    for layer, layer_rows in sorted(grouped.items()):
        ranks = [int(row["jlens_rank"]) for row in layer_rows]
        logit_ranks = [int(row["logit_lens_rank"]) for row in layer_rows]
        result[str(layer)] = {
            "normalized_depth": layer_rows[0]["normalized_depth"],
            "n": len(layer_rows),
            "jlens_top1": wilson_interval(sum(rank <= 1 for rank in ranks), len(ranks)),
            "jlens_top5": wilson_interval(sum(rank <= 5 for rank in ranks), len(ranks)),
            "jlens_top10": wilson_interval(sum(rank <= 10 for rank in ranks), len(ranks)),
            "logit_lens_top1": wilson_interval(
                sum(rank <= 1 for rank in logit_ranks), len(logit_ranks)
            ),
            "logit_lens_top5": wilson_interval(
                sum(rank <= 5 for rank in logit_ranks), len(logit_ranks)
            ),
            "logit_lens_top10": wilson_interval(
                sum(rank <= 10 for rank in logit_ranks), len(logit_ranks)
            ),
        }
    return result


def interpretation(
    config: dict[str, Any],
    *,
    eligible_count: int,
    readout: dict[str, Any],
    swap: dict[str, Any],
) -> dict[str, str]:
    thresholds = config["interpretation"]
    if eligible_count < int(thresholds["minimum_primary_examples"]):
        return {
            "outcome": "D",
            "summary": "Too few official prompts pass clean Qwen validation.",
        }
    readable = readout["top5_recovery"]["rate"]
    if readable is None or readable < float(thresholds["readable_band_top5_rate"]):
        return {
            "outcome": "C",
            "summary": "Correct intermediates are not reliably readable in the selected Qwen J-space band.",
        }
    success = swap["target_answer_top1_swap_success"]["rate"]
    if success is not None and success >= float(thresholds["successful_swap_rate"]):
        return {
            "outcome": "A",
            "summary": "The paper-aligned Qwen read-and-swap substrate check succeeds under the declared thresholds.",
        }
    return {
        "outcome": "B",
        "summary": "Correct intermediates are readable, but swaps do not reliably redirect Qwen answers.",
    }


def summary_markdown(aggregate: dict[str, Any]) -> str:
    counts = aggregate["counts"]
    readout = aggregate["readout"]["jlens"]
    baseline = aggregate["readout"]["logit_lens"]
    swap = aggregate["swap"]["primary_band"]
    target = swap["target_answer_top1_swap_success"]
    retention = swap["original_answer_top1_retention"]
    return (
        f"# Paper-aligned Qwen J-space summary\n\n"
        f"This is {EXPERIMENT_LABEL}, not an exact reproduction of the Claude paper.\n\n"
        f"- Official prompts: {counts['official_examples_available']}\n"
        f"- Tokenizer-compatible: {counts['tokenizer_checks_passed']} "
        f"({counts['tokenizer_checks_passed_percentage']:.1f}%)\n"
        f"- Primary eligible: {counts['eligible_primary']}\n"
        f"- Top-5 secondary diagnostic: {counts['eligible_top5_diagnostic']}\n"
        f"- Excluded: {counts['excluded']}\n"
        f"- Workspace band: {aggregate['workspace_band']['start_layer']}"
        f"–{aggregate['workspace_band']['end_layer']}\n"
        f"- J-lens band intermediate recovery (top-1/top-5/top-10): "
        f"{readout['top1_recovery']['rate']:.3f} / "
        f"{readout['top5_recovery']['rate']:.3f} / "
        f"{readout['top10_recovery']['rate']:.3f}\n"
        f"- Logit-lens band top-5 recovery: "
        f"{baseline['top5_recovery']['rate']:.3f}\n"
        f"- Target-answer top-1 swap success: {target['successes']}/"
        f"{target['total']} ({target['rate']:.3f}; Wilson 95% CI "
        f"{target['ci95_low']:.3f}–{target['ci95_high']:.3f})\n"
        f"- Original-answer retention after swap: {retention['rate']:.3f}\n"
        f"- Mean target log-probability change: "
        f"{swap['mean_delta_swap_target_log_probability']:.3f} nats\n"
        f"- Mean target-vs-clean margin change: "
        f"{swap['mean_delta_log_probability_margin']:.3f} nats\n"
        f"- Interpretation: {aggregate['interpretation']['outcome']} — "
        f"{aggregate['interpretation']['summary']}\n"
    )


def write_synthetic_smoke(
    config: dict[str, Any], config_path: Path, output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fake_readout = [
        {
            "example_id": "synthetic",
            "split": "heldout",
            "eligibility_status": "synthetic_smoke",
            "relation_or_task_family": "synthetic",
            "layer": layer,
            "normalized_depth": float(layer * 50),
            "measurement_token_position": -1,
            "source_intermediate_concept": "source",
            "source_concept_token_id": 1,
            "jlens_pre_softmax_score": 1.0,
            "jlens_cosine_similarity": 0.5,
            "jlens_rank": rank,
            "jlens_top1": rank <= 1,
            "jlens_top5": rank <= 5,
            "jlens_top10": rank <= 10,
            "logit_lens_pre_softmax_score": 0.0,
            "logit_lens_rank": rank + 2,
            "logit_lens_top1": False,
            "logit_lens_top5": rank + 2 <= 5,
            "logit_lens_top10": rank + 2 <= 10,
            "vocab_size": 100,
        }
        for layer, rank in [(0, 4), (1, 2), (2, 1)]
    ]
    fake_swap = {
        "example_id": "synthetic",
        "split": "heldout",
        "relation_or_task_family": "synthetic",
        "intervention_scope": "band",
        "patch_layers": [0, 1, 2],
        "target_answer_top1_success": True,
        "original_answer_top1_retained": False,
        "delta_swap_target_log_probability": 1.0,
        "delta_log_probability_margin": 2.0,
        "coordinate_diagnostics": [],
    }
    selection = {
        "selection_criterion": "synthetic_smoke",
        "selected": {"start_layer": 0, "end_layer": 2, "layers": [0, 1, 2]},
    }
    aggregate = {
        "status": "synthetic_cpu_smoke_not_model_evidence",
        "counts": {
            "official_examples_available": 0,
            "eligible_primary": 0,
            "excluded": 0,
        },
        "workspace_band": selection["selected"],
        "readout": {
            "jlens": aggregate_readout(fake_readout, [0, 1, 2], rank_key="jlens_rank"),
            "logit_lens": aggregate_readout(
                fake_readout, [0, 1, 2], rank_key="logit_lens_rank"
            ),
        },
        "swap": {
            "primary_band": aggregate_swap_rows([fake_swap]),
            "per_relation_family": {
                "synthetic": aggregate_swap_rows([fake_swap])
            },
            "per_layer_diagnostic": {},
        },
        "interpretation": {
            "outcome": "not_applicable",
            "summary": "Synthetic smoke test only.",
        },
    }
    write_json(
        output_dir / "run_manifest.json",
        {
            "status": "synthetic_cpu_smoke_not_model_evidence",
            "experiment_label": EXPERIMENT_LABEL,
            "config_sha256": sha256_file(config_path),
            "upstream_anthropic_prompt_commit": UPSTREAM_COMMIT,
            "exact_command": exact_command(),
        },
    )
    write_json(
        output_dir / "prompt_filter_report.json",
        {"status": "synthetic_smoke", "official_examples_available": 0},
    )
    write_jsonl(output_dir / "eligible_examples.jsonl", [])
    write_jsonl(output_dir / "excluded_examples.jsonl", [])
    write_json(output_dir / "layer_calibration.json", selection)
    write_jsonl(output_dir / "layerwise_readout.jsonl", fake_readout)
    write_jsonl(output_dir / "per_example_swap_results.jsonl", [fake_swap])
    write_json(output_dir / "aggregate_results.json", aggregate)
    write_aggregate_csv(output_dir / "aggregate_results.csv", aggregate)
    (output_dir / "concise_summary.md").write_text(
        "# Synthetic CPU smoke\n\nNo model-backed claim is made.\n",
        encoding="utf-8",
    )


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, Any]:
    root = repo_root()
    config = load_config(args.config)
    output_dir = args.output_dir or resolve_repo_path(
        root, config["outputs"]["results_dir"]
    )
    if output_dir is None:
        raise ValueError("results_dir is required")
    if args.synthetic_smoke:
        write_synthetic_smoke(config, args.config, output_dir)
        return {"status": "synthetic_cpu_smoke_not_model_evidence"}

    started_wall = time.time()
    started_at = utc_now()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    processed_dir = resolve_repo_path(root, config["prompts"]["processed_dir"])
    assert processed_dir is not None
    processed_path = processed_dir / "processed_examples.jsonl"
    if not processed_path.is_file():
        raise FileNotFoundError(
            f"{processed_path} is missing; run prepare_paper_aligned_prompts.py first"
        )
    all_examples = read_jsonl(processed_path)
    if any(row["eligibility_status"] == "pending_clean_validation" for row in all_examples):
        raise RuntimeError("Processed prompts were metadata-only; clean validation is required")
    primary = [
        row for row in all_examples if row["eligibility_status"] == "eligible_primary"
    ]
    secondary = [
        row
        for row in all_examples
        if row["eligibility_status"] == "eligible_top5_diagnostic"
    ]
    excluded = [
        row
        for row in all_examples
        if row["eligibility_status"]
        not in {"eligible_primary", "eligible_top5_diagnostic"}
    ]
    calibration_examples = [row for row in primary if row["split"] == "calibration"]
    heldout_examples = [row for row in primary if row["split"] == "heldout"]
    if args.max_calibration is not None:
        calibration_examples = calibration_examples[: args.max_calibration]
    if args.max_heldout is not None:
        heldout_examples = heldout_examples[: args.max_heldout]
    if not calibration_examples or not heldout_examples:
        raise RuntimeError("Both calibration and held-out primary examples are required")

    model, tokenizer, lens, sources = load_components(config, root)
    layers = sorted(int(layer) for layer in lens.source_layers)
    jacobians = cache_jacobians(lens, layers, model.input_device)
    max_fitted_layer = max(layers)
    readout_rows = []
    clean_by_example: dict[str, dict[str, Any]] = {}
    logging.info("Reading %d calibration examples over %d layers", len(calibration_examples), len(layers))
    for index, example in enumerate(calibration_examples, start=1):
        rows, clean = clean_readout(
            model, jacobians, example, max_fitted_layer=max_fitted_layer
        )
        readout_rows.extend(rows)
        clean_by_example[example["example_id"]] = clean
        logging.info("Calibration readout %d/%d", index, len(calibration_examples))

    calibration_rows = [
        row
        for row in readout_rows
        if row["split"] == "calibration"
        and row["eligibility_status"] == "eligible_primary"
    ]
    calibration = select_workspace_band(
        calibration_rows,
        layers,
        int(config["layer_protocol"]["workspace_band_width"]),
    )
    band_layers = calibration["selected"]["layers"]
    calibration["normalized_depths"] = [
        normalized_depth(layer, max_fitted_layer) for layer in band_layers
    ]

    logging.info("Frozen workspace band: %s", band_layers)
    for index, example in enumerate(heldout_examples, start=1):
        rows, clean = clean_readout(
            model, jacobians, example, max_fitted_layer=max_fitted_layer
        )
        readout_rows.extend(rows)
        clean_by_example[example["example_id"]] = clean
        logging.info("Held-out readout %d/%d", index, len(heldout_examples))

    intervention = config["intervention"]
    rcond = float(intervention["pseudoinverse_rcond"])
    strength = float(intervention["strength"])
    swap_rows = []
    for index, example in enumerate(heldout_examples, start=1):
        clean = clean_by_example[example["example_id"]]
        logits, diagnostics = patched_logits(
            model,
            jacobians,
            example["prompt"],
            int(example["source_concept_token_id"]),
            int(example["target_concept_token_id"]),
            band_layers,
            rcond=rcond,
            strength=strength,
        )
        swap_rows.append(
            swap_result(
                example,
                clean,
                logits,
                intervention_scope="band",
                patch_layers=band_layers,
                diagnostics=diagnostics,
                tokenizer=tokenizer,
            )
        )
        if intervention.get("run_single_layer_diagnostic", False):
            for layer in band_layers:
                logits, diagnostics = patched_logits(
                    model,
                    jacobians,
                    example["prompt"],
                    int(example["source_concept_token_id"]),
                    int(example["target_concept_token_id"]),
                    [layer],
                    rcond=rcond,
                    strength=strength,
                )
                swap_rows.append(
                    swap_result(
                        example,
                        clean,
                        logits,
                        intervention_scope="single_layer",
                        patch_layers=[layer],
                        diagnostics=diagnostics,
                        tokenizer=tokenizer,
                    )
                )
        if intervention.get("run_noop_control", False):
            logits, diagnostics = patched_logits(
                model,
                jacobians,
                example["prompt"],
                int(example["source_concept_token_id"]),
                int(example["source_concept_token_id"]),
                band_layers,
                rcond=rcond,
                strength=strength,
            )
            no_op = swap_result(
                example,
                clean,
                logits,
                intervention_scope="source_to_source_noop",
                patch_layers=band_layers,
                diagnostics=diagnostics,
                tokenizer=tokenizer,
            )
            no_op["maximum_absolute_output_logit_change"] = float(
                (logits - clean["logits"]).abs().max().item()
            )
            swap_rows.append(no_op)
        logging.info("Held-out swap %d/%d", index, len(heldout_examples))

    heldout_readout = [
        row
        for row in readout_rows
        if row["split"] == "heldout"
        and row["eligibility_status"] == "eligible_primary"
    ]
    primary_swap_rows = [
        row for row in swap_rows if row["intervention_scope"] == "band"
    ]
    relation_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in primary_swap_rows:
        relation_groups[row["relation_or_task_family"]].append(row)
    layer_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in swap_rows:
        if row["intervention_scope"] == "single_layer":
            layer_groups[int(row["patch_layers"][0])].append(row)

    jlens_aggregate = aggregate_readout(
        heldout_readout, band_layers, rank_key="jlens_rank"
    )
    logit_aggregate = aggregate_readout(
        heldout_readout, band_layers, rank_key="logit_lens_rank"
    )
    swap_aggregate = aggregate_swap_rows(primary_swap_rows)
    filter_report = json.loads(
        (processed_dir / "prompt_filter_report.json").read_text(encoding="utf-8")
    )
    aggregate = {
        "status": "paper_aligned_qwen_jspace_result_not_exact_claude_replication",
        "experiment_label": EXPERIMENT_LABEL,
        "counts": {
            "official_examples_available": len(all_examples),
            "tokenizer_checks_passed": filter_report["tokenizer_checks_passed"],
            "tokenizer_checks_passed_percentage": filter_report[
                "tokenizer_checks_passed_percentage"
            ],
            "eligible_primary": len(primary),
            "eligible_top5_diagnostic": len(secondary),
            "excluded": len(excluded),
            "calibration_primary_used": len(calibration_examples),
            "heldout_primary_used": len(heldout_examples),
            "clean_top1_accuracy_primary": wilson_interval(
                sum(
                    clean_by_example[row["example_id"]]["top1_token_id"]
                    == int(row["clean_answer_token_id"])
                    for row in calibration_examples + heldout_examples
                ),
                len(calibration_examples) + len(heldout_examples),
            ),
        },
        "workspace_band": {
            **calibration["selected"],
            "normalized_depths": calibration["normalized_depths"],
        },
        "readout": {
            "jlens": jlens_aggregate,
            "logit_lens": logit_aggregate,
            "per_layer": per_layer_readout(heldout_readout),
        },
        "swap": {
            "primary_band": swap_aggregate,
            "per_relation_family": {
                key: aggregate_swap_rows(value)
                for key, value in sorted(relation_groups.items())
            },
            "per_layer_diagnostic": {
                str(key): aggregate_swap_rows(value)
                for key, value in sorted(layer_groups.items())
            },
        },
    }
    aggregate["interpretation"] = interpretation(
        config,
        eligible_count=len(primary),
        readout=jlens_aggregate,
        swap=swap_aggregate,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        processed_dir / "prompt_filter_report.json",
        output_dir / "prompt_filter_report.json",
    )
    write_jsonl(output_dir / "eligible_examples.jsonl", primary + secondary)
    write_jsonl(output_dir / "excluded_examples.jsonl", excluded)
    write_json(output_dir / "layer_calibration.json", calibration)
    write_jsonl(output_dir / "layerwise_readout.jsonl", readout_rows)
    write_jsonl(output_dir / "per_example_swap_results.jsonl", swap_rows)
    write_json(output_dir / "aggregate_results.json", aggregate)
    write_aggregate_csv(output_dir / "aggregate_results.csv", aggregate)
    (output_dir / "concise_summary.md").write_text(
        summary_markdown(aggregate), encoding="utf-8"
    )

    manifest = {
        "status": aggregate["status"],
        "experiment_label": EXPERIMENT_LABEL,
        "git": git_snapshot(root),
        "model_id": config["model"]["id"],
        "model_revision": config["model"]["revision"],
        "model_source": sources["model_source"],
        "local_model_revision_verified": sources[
            "local_model_revision_verified"
        ],
        "lens_repository": config["lens"]["repository"],
        "lens_file": config["lens"]["file"],
        "lens_revision": config["lens"]["revision"],
        "lens_sha256": config["lens"]["sha256"],
        "tokenizer_id": config["model"]["tokenizer_id"],
        "tokenizer_revision": config["model"]["tokenizer_revision"],
        "upstream_anthropic_prompt_commit": UPSTREAM_COMMIT,
        "configuration_file": str(args.config.resolve()),
        "configuration_file_sha256": sha256_file(args.config),
        "random_seed": int(config["seed"]),
        "selected_layers": [
            {
                "native_layer": layer,
                "normalized_depth": normalized_depth(layer, max_fitted_layer),
            }
            for layer in layers
        ],
        "selected_workspace_band": aggregate["workspace_band"],
        "dtype": sources["dtype"],
        "device": sources["device"],
        "package_versions": package_versions(
            ["transformers", "huggingface-hub", "PyYAML", "numpy"]
        ),
        "exact_command": exact_command(),
        "start_time": started_at,
        "end_time": utc_now(),
        "elapsed_seconds": time.time() - started_wall,
        "peak_cuda_memory_gib": (
            torch.cuda.max_memory_allocated() / 2**30
            if torch.cuda.is_available()
            else None
        ),
        "eligible_count": len(primary),
        "excluded_count": len(excluded),
        "sample_limited": (
            args.max_calibration is not None or args.max_heldout is not None
        ),
    }
    write_json(output_dir / "run_manifest.json", manifest)
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=repo_root() / "task3" / "configs" / "paper_aligned_jspace.yaml",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-calibration", type=int)
    parser.add_argument("--max-heldout", type=int)
    parser.add_argument("--synthetic-smoke", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    root = repo_root()
    logs_dir = resolve_repo_path(root, config["outputs"]["logs_dir"])
    assert logs_dir is not None
    logs_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                logs_dir / "run_paper_aligned_jspace.log", encoding="utf-8"
            ),
        ],
    )
    result = run(args)
    logging.info("%s", json.dumps(result, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
