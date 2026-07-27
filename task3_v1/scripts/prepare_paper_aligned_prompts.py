"""Prepare the official Anthropic probe-swap prompts for frozen Qwen3.5-4B.

Filtering uses tokenizer compatibility and clean model behavior only.  No
intervention, CauScale, semantic-recovery, or held-out swap result is consulted.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from paper_aligned_core import (
    UPSTREAM_COMMIT,
    answer_token_representation,
    assign_group_splits,
    load_config,
    repo_root,
    resolve_repo_path,
    sha256_file,
    single_token_representation,
    token_subsequence_present,
    write_json,
    write_jsonl,
)


def load_tokenizer(config: dict[str, Any], root: Path):
    import transformers

    model_config = config["model"]
    local_path = resolve_repo_path(root, model_config.get("local_path"))
    if local_path and local_path.is_dir():
        revision_marker = (
            local_path
            / ".cache"
            / "huggingface"
            / "trees"
            / f"{model_config['tokenizer_revision']}.json"
        )
        if not revision_marker.is_file():
            raise RuntimeError(
                "Local tokenizer cache does not match the configured revision: "
                f"{revision_marker}"
            )
    source = str(local_path) if local_path and local_path.is_dir() else model_config["tokenizer_id"]
    kwargs = {}
    if source == model_config["tokenizer_id"]:
        kwargs["revision"] = model_config["tokenizer_revision"]
    tokenizer = transformers.AutoTokenizer.from_pretrained(source, **kwargs)
    if (
        getattr(tokenizer, "bos_token_id", None) is not None
        and hasattr(tokenizer, "add_bos_token")
    ):
        tokenizer.add_bos_token = True
    return tokenizer, source


def load_clean_model(config: dict[str, Any], root: Path, tokenizer):
    import jlens
    import transformers

    model_config = config["model"]
    device = torch.device(model_config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Configured CUDA device is not available")
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[model_config["dtype"]]
    local_path = resolve_repo_path(root, model_config.get("local_path"))
    source = str(local_path) if local_path and local_path.is_dir() else model_config["id"]
    kwargs: dict[str, Any] = {"dtype": dtype}
    if source == model_config["id"]:
        kwargs["revision"] = model_config["revision"]
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(source, **kwargs)
    hf_model.to(device)
    return jlens.from_hf(hf_model, tokenizer), source


@torch.no_grad()
def clean_top_tokens(model, prompt: str, k: int = 5) -> tuple[list[int], list[float]]:
    from jlens.hooks import ActivationRecorder

    final_layer = model.n_layers - 1
    input_ids = model.encode(prompt)
    with ActivationRecorder(model.layers, at=[final_layer]) as recorder:
        model.forward(input_ids)
        residual = recorder.activations[final_layer][0, -1].float()
    logits = model.unembed(residual[None])[0].float()
    values, ids = logits.topk(k)
    return [int(value) for value in ids.cpu()], [float(value) for value in values.cpu()]


def exclusion_reasons(
    *,
    source_representation: dict[str, Any],
    target_representation: dict[str, Any],
    clean_answer_representation: dict[str, Any],
    swapped_answer_representation: dict[str, Any],
    prompt_token_length: int,
    max_context_tokens: int,
    source_leaked: bool,
    target_leaked: bool,
) -> list[str]:
    reasons = []
    if not source_representation["valid"]:
        reasons.append("source_intermediate_not_single_token")
    if not target_representation["valid"]:
        reasons.append("swap_target_intermediate_not_single_token")
    if not clean_answer_representation["valid"]:
        reasons.append("clean_answer_not_single_token")
    if not swapped_answer_representation["valid"]:
        reasons.append("swapped_answer_not_single_token")
    if prompt_token_length > max_context_tokens:
        reasons.append("prompt_exceeds_context_limit")
    if source_leaked:
        reasons.append("source_intermediate_present_in_prompt")
    if target_leaked:
        reasons.append("swap_target_intermediate_present_in_prompt")
    return reasons


def prepare(
    config: dict[str, Any],
    *,
    metadata_only: bool,
    limit: int | None,
    output_dir: Path | None,
) -> dict[str, Any]:
    root = repo_root()
    prompt_config = config["prompts"]
    official_path = resolve_repo_path(root, prompt_config["official_file"])
    if official_path is None or not official_path.is_file():
        raise FileNotFoundError(official_path)
    if sha256_file(official_path) != "a0edd27ca23f7b4d0fbe90448c2ddcc7457a3d812121bf024ed12a032ff86796":
        raise RuntimeError("The official probe-swap copy does not match its pinned SHA-256")
    raw_items = json.loads(official_path.read_text(encoding="utf-8"))["items"]
    if limit is not None:
        raw_items = raw_items[:limit]

    tokenizer, tokenizer_source = load_tokenizer(config, root)
    model = None if metadata_only else load_clean_model(config, root, tokenizer)[0]
    max_context = int(config["model"]["max_context_tokens"])
    rows = []
    for source_index, item in enumerate(raw_items):
        prompt = item["prompt"]
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
        source_rep = single_token_representation(tokenizer, item["intermediate"])
        target_rep = single_token_representation(tokenizer, item["swap_to"])
        clean_answer_rep = answer_token_representation(tokenizer, prompt, item["answer"])
        swapped_answer_rep = answer_token_representation(
            tokenizer, prompt, item["swap_answer"]
        )
        source_leaked = (
            source_rep["valid"]
            and token_subsequence_present(prompt_ids, source_rep["token_ids"])
        )
        target_leaked = (
            target_rep["valid"]
            and token_subsequence_present(prompt_ids, target_rep["token_ids"])
        )
        reasons = exclusion_reasons(
            source_representation=source_rep,
            target_representation=target_rep,
            clean_answer_representation=clean_answer_rep,
            swapped_answer_representation=swapped_answer_rep,
            prompt_token_length=len(prompt_ids),
            max_context_tokens=max_context,
            source_leaked=source_leaked,
            target_leaked=target_leaked,
        )

        top_ids: list[int] = []
        top_logits: list[float] = []
        if not metadata_only and not reasons:
            top_ids, top_logits = clean_top_tokens(model, prompt)
            if top_ids[0] == clean_answer_rep["token_id"]:
                eligibility = "eligible_primary"
            elif clean_answer_rep["token_id"] in top_ids:
                eligibility = "eligible_top5_diagnostic"
                reasons.append("clean_answer_top5_but_not_top1")
            else:
                eligibility = "excluded"
                reasons.append("clean_answer_not_in_top5")
        elif metadata_only and not reasons:
            eligibility = "pending_clean_validation"
            reasons.append("metadata_only_clean_validation_not_run")
        else:
            eligibility = "excluded"

        rows.append(
            {
                "example_id": item["name"],
                "official_source_file": "data/experiments/probe-swap.json",
                "official_source_index": source_index,
                "prompt": prompt,
                "relation_or_task_family": item["category"],
                "source_intermediate_concept": item["intermediate"],
                "swap_target_intermediate_concept": item["swap_to"],
                "clean_expected_answer": item["answer"],
                "swapped_expected_answer": item["swap_answer"],
                "source_concept_token_id": source_rep["token_id"],
                "target_concept_token_id": target_rep["token_id"],
                "clean_answer_token_id": clean_answer_rep["token_id"],
                "swapped_answer_token_id": swapped_answer_rep["token_id"],
                "source_concept_token_surface": source_rep["surface"],
                "target_concept_token_surface": target_rep["surface"],
                "clean_answer_token_surface": clean_answer_rep["surface"],
                "swapped_answer_token_surface": swapped_answer_rep["surface"],
                "measurement_token_position": -1,
                "measurement_position_rule": "final_prompt_token_immediately_before_next_answer",
                "prompt_token_length": len(prompt_ids),
                "parent_template_category_group": item["category"],
                "eligibility_status": eligibility,
                "exclusion_reason": ";".join(reasons) if reasons else None,
                "upstream_commit_sha": UPSTREAM_COMMIT,
                "clean_model_top5_token_ids": top_ids,
                "clean_model_top5_tokens": [
                    tokenizer.decode([token_id]) for token_id in top_ids
                ],
                "clean_model_top5_logits": top_logits,
                "tokenizer_checks_passed": not any(
                    reason
                    for reason in reasons
                    if reason
                    not in {
                        "clean_answer_top5_but_not_top1",
                        "clean_answer_not_in_top5",
                        "metadata_only_clean_validation_not_run",
                    }
                ),
            }
        )

    split_by_group = assign_group_splits(
        rows,
        group_key="parent_template_category_group",
        calibration_fraction=float(prompt_config["calibration_fraction"]),
        seed=int(config["seed"]),
    )
    for row in rows:
        row["split"] = split_by_group[row["parent_template_category_group"]]

    destination = output_dir or resolve_repo_path(root, prompt_config["processed_dir"])
    if destination is None:
        raise ValueError("processed_dir is required")
    destination.mkdir(parents=True, exist_ok=True)
    primary = [row for row in rows if row["eligibility_status"] == "eligible_primary"]
    secondary = [
        row
        for row in rows
        if row["eligibility_status"] == "eligible_top5_diagnostic"
    ]
    excluded = [
        row
        for row in rows
        if row["eligibility_status"]
        not in {"eligible_primary", "eligible_top5_diagnostic"}
    ]
    reason_counts = Counter(
        reason
        for row in rows
        for reason in (row["exclusion_reason"] or "").split(";")
        if reason
    )
    report = {
        "status": (
            "metadata_only_not_primary_eligible_data"
            if metadata_only
            else "clean_behavior_and_tokenizer_filtered"
        ),
        "official_examples_available": len(rows),
        "tokenizer_checks_passed": sum(
            bool(row["tokenizer_checks_passed"]) for row in rows
        ),
        "tokenizer_checks_passed_percentage": (
            100.0
            * sum(bool(row["tokenizer_checks_passed"]) for row in rows)
            / len(rows)
            if rows
            else 0.0
        ),
        "eligible_primary": len(primary),
        "eligible_top5_diagnostic": len(secondary),
        "excluded": len(excluded),
        "exclusion_reasons": dict(sorted(reason_counts.items())),
        "calibration_groups": sorted(
            group for group, split in split_by_group.items() if split == "calibration"
        ),
        "heldout_groups": sorted(
            group for group, split in split_by_group.items() if split == "heldout"
        ),
        "split_rule": "whole category groups; deterministic SHA-256 ordering",
        "filtering_rule": "tokenizer compatibility and clean model behavior only",
        "tokenizer_source": tokenizer_source,
        "model_id": config["model"]["id"],
        "model_revision": config["model"]["revision"],
        "upstream_commit_sha": UPSTREAM_COMMIT,
    }
    write_jsonl(destination / "processed_examples.jsonl", rows)
    write_jsonl(destination / "eligible_primary.jsonl", primary)
    write_jsonl(destination / "eligible_top5_diagnostic.jsonl", secondary)
    write_jsonl(destination / "excluded_examples.jsonl", excluded)
    write_json(destination / "prompt_filter_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=repo_root() / "task3" / "configs" / "paper_aligned_jspace.yaml",
    )
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    started = time.time()
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
                logs_dir / "prepare_paper_aligned_prompts.log",
                encoding="utf-8",
            ),
        ],
    )
    report = prepare(
        config,
        metadata_only=args.metadata_only,
        limit=args.limit,
        output_dir=args.output_dir,
    )
    report["elapsed_seconds"] = time.time() - started
    logging.info("%s", json.dumps(report, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
