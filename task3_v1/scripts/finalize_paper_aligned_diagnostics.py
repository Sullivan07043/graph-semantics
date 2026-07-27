#!/usr/bin/env python3
"""Validate, freeze, and package the completed paper-aligned prompt experiment.

This script is deliberately one-way.  The first successful invocation writes a
FROZEN manifest last.  Later invocations only verify the recorded hashes.
"""

from __future__ import annotations

import csv
import difflib
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
PROMPT_ROOT = ROOT / "task3_v1" / "data" / "prompts"
OFFICIAL_ROOT = PROMPT_ROOT / "official_anthropic"
PROCESSED_ROOT = PROMPT_ROOT / "paper_aligned_qwen"
FROZEN_ROOT = PROMPT_ROOT / "frozen" / "official_probe_swap"
OUTPUT_ROOT = ROOT / "task3_v1" / "outputs" / "paper_aligned"
LOG_ROOT = ROOT / "task3_v1" / "logs" / "paper_aligned"
MANIFEST_PATH = OUTPUT_ROOT / "frozen_dataset_manifest.json"

UPSTREAM_COMMIT = "581d398613e5602a5af361e1c34d3a92ea82ba8e"
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
LENS_REPOSITORY = "neuronpedia/jacobian-lens"
LENS_FILE = (
    "qwen3.5-4b/jlens/Salesforce-wikitext/"
    "Qwen3.5-4B_jacobian_lens_n1000.pt"
)
LENS_REVISION = "b62c39069a0740aebcc70462231b68612cae367f"
LENS_SHA256 = "1f9a8f8fd593f0ffec1a9640993257ca4560f8ae3e5602315643d5cc6818534e"
SEED = 20260725
NEAR_DUPLICATE_THRESHOLD = 0.92
FROZEN_BAND = [23, 24, 25, 26, 27, 28]
REPRESENTATIVE_LAYERS = [23, 25, 26, 28]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def dump_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    path.write_text(text, encoding="utf-8", newline="\n")


def normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def contains_term(prompt: str, term: str) -> bool:
    return f" {normalize_text(term)} " in f" {normalize_text(prompt)} "


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> dict[str, Any]:
    if total == 0:
        return {
            "successes": successes,
            "total": total,
            "rate": None,
            "ci95_low": None,
            "ci95_high": None,
        }
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    radius = (
        z
        * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
        / denom
    )
    return {
        "successes": successes,
        "total": total,
        "rate": p,
        "ci95_low": max(0.0, center - radius),
        "ci95_high": min(1.0, center + radius),
    }


def git_snapshot() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={ROOT.as_posix()}", *args],
            cwd=ROOT,
            text=True,
            encoding="utf-8",
        ).strip()

    try:
        commit = run("rev-parse", "HEAD")
        status = run("status", "--short")
        return {
            "commit": commit,
            "dirty": bool(status),
            "status_short": status.splitlines(),
        }
    except (OSError, subprocess.CalledProcessError) as error:
        return {"commit": None, "dirty": None, "error": str(error)}


def verify_existing_freeze(manifest: dict[str, Any]) -> None:
    failures: list[str] = []
    for record in manifest.get("frozen_files", []):
        path = ROOT / record["path"]
        if not path.is_file():
            failures.append(f"missing: {record['path']}")
        elif sha256(path) != record["sha256"]:
            failures.append(f"hash mismatch: {record['path']}")
    if failures:
        raise SystemExit("FROZEN verification failed:\n" + "\n".join(failures))
    if not manifest.get("git", {}).get("commit"):
        repaired_git = git_snapshot()
        if not repaired_git.get("commit"):
            raise SystemExit(
                "Frozen file hashes passed, but Git metadata could not be repaired."
            )
        manifest["git"] = repaired_git
        manifest["freeze_metadata_correction"] = {
            "corrected_at_utc": datetime.now(timezone.utc).isoformat(),
            "reason": (
                "Initial sandbox Git safe-directory check prevented commit/dirty "
                "capture; frozen data and experiment files were not modified."
            ),
        }
        dump_json(MANIFEST_PATH, manifest)
    print(
        f"FROZEN verification passed for {len(manifest.get('frozen_files', []))} files."
    )


def validate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    required = {
        "example_id",
        "prompt",
        "relation_or_task_family",
        "source_intermediate_concept",
        "swap_target_intermediate_concept",
        "clean_expected_answer",
        "swapped_expected_answer",
        "source_concept_token_id",
        "target_concept_token_id",
        "clean_answer_token_id",
        "swapped_answer_token_id",
        "measurement_token_position",
        "prompt_token_length",
        "parent_template_category_group",
        "eligibility_status",
        "clean_model_top5_token_ids",
        "tokenizer_checks_passed",
        "split",
    }
    missing_fields = {
        row.get("example_id", f"row-{index}"): sorted(required - row.keys())
        for index, row in enumerate(rows)
        if required - row.keys()
    }
    ids = [row.get("example_id") for row in rows]
    duplicate_ids = sorted(key for key, count in Counter(ids).items() if count > 1)
    invalid_splits = sorted(
        {
            str(row.get("split"))
            for row in rows
            if row.get("split") not in {"calibration", "heldout"}
        }
    )

    primary_errors: list[dict[str, str]] = []
    secondary_errors: list[dict[str, str]] = []
    token_errors: list[dict[str, str]] = []
    leakage: list[dict[str, Any]] = []
    for row in rows:
        status = row["eligibility_status"]
        top5 = row["clean_model_top5_token_ids"]
        answer_id = row["clean_answer_token_id"]
        if status == "eligible_primary" and (not top5 or top5[0] != answer_id):
            primary_errors.append(
                {"example_id": row["example_id"], "error": "clean answer is not top-1"}
            )
        if status == "eligible_top5_diagnostic" and (
            answer_id not in top5 or (top5 and top5[0] == answer_id)
        ):
            secondary_errors.append(
                {
                    "example_id": row["example_id"],
                    "error": "clean answer is not strictly top-5-but-not-top-1",
                }
            )
        token_fields = [
            "source_concept_token_id",
            "target_concept_token_id",
            "clean_answer_token_id",
            "swapped_answer_token_id",
        ]
        if row["tokenizer_checks_passed"] and any(
            not isinstance(row[field], int) for field in token_fields
        ):
            token_errors.append(
                {
                    "example_id": row["example_id"],
                    "error": "tokenizer pass has a non-integer token id",
                }
            )
        leaked = {
            field: contains_term(row["prompt"], row[field])
            for field in [
                "source_intermediate_concept",
                "swap_target_intermediate_concept",
                "clean_expected_answer",
                "swapped_expected_answer",
            ]
        }
        if any(leaked.values()):
            leakage.append(
                {
                    "example_id": row["example_id"],
                    "eligibility_status": status,
                    "split": row["split"],
                    "fields": leaked,
                }
            )

    by_prompt: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        by_prompt[normalize_text(row["prompt"])].append(row["example_id"])
    exact_duplicates = [
        values for values in by_prompt.values() if len(values) > 1
    ]
    id_to_row = {row["example_id"]: row for row in rows}
    exact_cross_split = [
        group
        for group in exact_duplicates
        if len({id_to_row[item]["split"] for item in group}) > 1
    ]

    near_duplicates: list[dict[str, Any]] = []
    cross_split_near_duplicates: list[dict[str, Any]] = []
    for left_index, left in enumerate(rows):
        left_text = normalize_text(left["prompt"])
        for right in rows[left_index + 1 :]:
            right_text = normalize_text(right["prompt"])
            if left_text == right_text:
                continue
            ratio = difflib.SequenceMatcher(None, left_text, right_text).ratio()
            if ratio >= NEAR_DUPLICATE_THRESHOLD:
                record = {
                    "left": left["example_id"],
                    "right": right["example_id"],
                    "similarity": ratio,
                    "left_split": left["split"],
                    "right_split": right["split"],
                }
                near_duplicates.append(record)
                if left["split"] != right["split"]:
                    cross_split_near_duplicates.append(record)

    split_entities: dict[str, set[str]] = {
        "calibration": set(),
        "heldout": set(),
    }
    split_groups: dict[str, set[str]] = {
        "calibration": set(),
        "heldout": set(),
    }
    for row in rows:
        split = row["split"]
        split_entities[split].update(
            {
                row["source_intermediate_concept"].casefold(),
                row["swap_target_intermediate_concept"].casefold(),
            }
        )
        split_groups[split].add(row["parent_template_category_group"])

    entity_overlap = sorted(
        split_entities["calibration"] & split_entities["heldout"]
    )
    group_overlap = sorted(split_groups["calibration"] & split_groups["heldout"])
    lengths = [row["prompt_token_length"] for row in rows]
    status_counts = Counter(row["eligibility_status"] for row in rows)
    split_counts = Counter(row["split"] for row in rows)
    category_counts = Counter(row["relation_or_task_family"] for row in rows)

    hard_errors = (
        bool(missing_fields)
        or bool(duplicate_ids)
        or bool(invalid_splits)
        or bool(primary_errors)
        or bool(secondary_errors)
        or bool(token_errors)
        or len(rows) != 90
    )
    warnings = []
    if exact_duplicates:
        warnings.append("exact_prompt_duplicates_present")
    if near_duplicates:
        warnings.append("near_prompt_duplicates_present")
    if leakage:
        warnings.append("literal_concept_or_answer_leakage_present")
    if entity_overlap:
        warnings.append("entity_overlap_across_splits")
    if cross_split_near_duplicates:
        warnings.append("near_duplicate_overlap_across_splits")

    return {
        "validator_status": "FAIL" if hard_errors else "PASS_WITH_WARNINGS",
        "hard_errors_present": hard_errors,
        "warnings": warnings,
        "format_and_schema": {
            "utf8_jsonl_parse": "PASS",
            "row_count": len(rows),
            "expected_row_count": 90,
            "missing_required_fields": missing_fields,
            "duplicate_example_ids": duplicate_ids,
            "invalid_splits": invalid_splits,
        },
        "tokenizer_compatibility": {
            "model_tokenizer": MODEL_ID,
            "revision": MODEL_REVISION,
            "recorded_full_clean_run_audited": True,
            "checks_passed": sum(bool(row["tokenizer_checks_passed"]) for row in rows),
            "checks_failed": sum(not row["tokenizer_checks_passed"] for row in rows),
            "token_record_errors": token_errors,
            "prompt_token_length": {
                "minimum": min(lengths),
                "maximum": max(lengths),
                "mean": sum(lengths) / len(lengths),
            },
        },
        "clean_correctness": {
            "verification_mode": "audit_of_saved_full_clean_model_run",
            "primary_rule": "clean answer token is top-1",
            "secondary_rule": "clean answer token is in top-5 but not top-1",
            "primary_errors": primary_errors,
            "secondary_errors": secondary_errors,
            "status_counts": dict(sorted(status_counts.items())),
        },
        "literal_leakage": {
            "count": len(leakage),
            "records": leakage,
            "primary_eligible_count": sum(
                item["eligibility_status"] == "eligible_primary" for item in leakage
            ),
        },
        "duplicates": {
            "normalization": "casefold; non-alphanumeric collapsed to one space",
            "exact_duplicate_groups": exact_duplicates,
            "exact_cross_split_groups": exact_cross_split,
            "near_duplicate_threshold": NEAR_DUPLICATE_THRESHOLD,
            "near_duplicate_pairs": near_duplicates,
            "cross_split_near_duplicate_pairs": cross_split_near_duplicates,
        },
        "split_leakage": {
            "split_rule": (
                "deterministic category/template-group assignment, seed 20260725; "
                "frozen after held-out evaluation"
            ),
            "split_counts_all_rows": dict(sorted(split_counts.items())),
            "template_or_parent_group_overlap": group_overlap,
            "entity_overlap_count": len(entity_overlap),
            "entity_overlap": entity_overlap,
            "assessment": (
                "LIMITATION: template groups are disjoint, but entity-level overlap "
                "is present. The split is not changed post hoc because held-out "
                "results have already been observed."
            ),
        },
        "coverage": {
            "official_anthropic": {
                "available": True,
                "count": len(rows),
                "category_counts": dict(sorted(category_counts.items())),
            },
            "controlled_current32": {
                "available": False,
                "count": 0,
                "status": "MISSING_NOT_GENERATED",
                "action": (
                    "Reported only. No supplemental prompts were generated because "
                    "the requested controlled-current32 input does not exist."
                ),
            },
        },
    }


def concept_category(relation: str) -> str:
    if relation.startswith(("city-", "language-", "river-")):
        return "geography"
    if relation.startswith(("element-", "planet-")) or relation == "multihop":
        return "science_or_general_multihop"
    if relation.startswith(("organ-", "func-")):
        return "biology_and_body"
    if relation.startswith(("animal-", "bird-", "food-", "fruit-", "beverage-")):
        return "living_world_and_food"
    return "culture_history_and_other"


def summarize_swap_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    target_successes = sum(bool(row["target_answer_top1_success"]) for row in rows)
    retained = sum(bool(row["original_answer_top1_retained"]) for row in rows)
    return {
        "n": n,
        "target_answer_top1": wilson(target_successes, n),
        "original_answer_top1_retained": wilson(retained, n),
        "mean_delta_target_log_probability": (
            sum(row["delta_swap_target_log_probability"] for row in rows) / n
            if n
            else None
        ),
        "mean_delta_log_probability_margin": (
            sum(row["delta_log_probability_margin"] for row in rows) / n
            if n
            else None
        ),
    }


def write_csv(path: Path, aggregate: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for grouping in ["relation_family", "concept_category", "template_group"]:
        for name, metric in aggregate["grouped_band_swap"][grouping].items():
            rows.append(
                {
                    "grouping": grouping,
                    "group": name,
                    "n": metric["n"],
                    "target_top1_successes": metric["target_answer_top1"]["successes"],
                    "target_top1_rate": metric["target_answer_top1"]["rate"],
                    "target_top1_ci95_low": metric["target_answer_top1"]["ci95_low"],
                    "target_top1_ci95_high": metric["target_answer_top1"]["ci95_high"],
                    "original_retained_rate": metric[
                        "original_answer_top1_retained"
                    ]["rate"],
                    "mean_delta_target_log_probability": metric[
                        "mean_delta_target_log_probability"
                    ],
                    "mean_delta_log_probability_margin": metric[
                        "mean_delta_log_probability_margin"
                    ],
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if MANIFEST_PATH.exists():
        manifest = load_json(MANIFEST_PATH)
        if manifest.get("status") == "FROZEN":
            verify_existing_freeze(manifest)
            return
        raise SystemExit("Refusing to overwrite a non-FROZEN manifest.")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    FROZEN_ROOT.mkdir(parents=True, exist_ok=True)

    source_manifest = load_json(OFFICIAL_ROOT / "SOURCE_MANIFEST.json")
    run_manifest = load_json(OUTPUT_ROOT / "run_manifest.json")
    calibration = load_json(OUTPUT_ROOT / "layer_calibration.json")
    original_aggregate = load_json(OUTPUT_ROOT / "aggregate_results.json")
    rows = load_jsonl(PROCESSED_ROOT / "processed_examples.jsonl")
    validation = validate_rows(rows)

    upstream_hashes = {
        "LICENSE": sha256(OFFICIAL_ROOT / "LICENSE"),
        "README.md": sha256(OFFICIAL_ROOT / "README.md"),
        "probe-swap.json": sha256(OFFICIAL_ROOT / "probe-swap.json"),
    }
    expected_probe_hash = (
        "a0edd27ca23f7b4d0fbe90448c2ddcc7457a3d812121bf024ed12a032ff86796"
    )
    validation["provenance"] = {
        "upstream_repository": source_manifest.get(
            "upstream_repository", "anthropics/jacobian-lens"
        ),
        "upstream_commit": UPSTREAM_COMMIT,
        "official_file_hashes": upstream_hashes,
        "probe_swap_expected_sha256": expected_probe_hash,
        "probe_swap_hash_matches": upstream_hashes["probe-swap.json"]
        == expected_probe_hash,
        "license": "Apache-2.0",
    }
    if validation["hard_errors_present"] or not validation["provenance"][
        "probe_swap_hash_matches"
    ]:
        dump_json(OUTPUT_ROOT / "prompt_validation_report.json", validation)
        raise SystemExit("Prompt validation failed; dataset was not frozen.")

    primary = [row for row in rows if row["eligibility_status"] == "eligible_primary"]
    secondary = [
        row
        for row in rows
        if row["eligibility_status"] == "eligible_top5_diagnostic"
    ]
    excluded = [row for row in rows if row["eligibility_status"] == "excluded"]
    dump_jsonl(OUTPUT_ROOT / "eligible_examples.jsonl", primary)
    dump_jsonl(OUTPUT_ROOT / "secondary_examples.jsonl", secondary)
    dump_jsonl(OUTPUT_ROOT / "excluded_examples.jsonl", excluded)
    dump_json(OUTPUT_ROOT / "prompt_validation_report.json", validation)

    prompt_rows = [
        {
            "example_id": row["example_id"],
            "prompt": row["prompt"],
            "split": row["split"],
            "source_file": row["official_source_file"],
            "source_index": row["official_source_index"],
            "prompt_token_length": row["prompt_token_length"],
            "measurement_token_position": row["measurement_token_position"],
            "measurement_position_rule": row["measurement_position_rule"],
        }
        for row in rows
    ]
    annotation_rows = [
        {key: value for key, value in row.items() if key != "prompt"} for row in rows
    ]
    dump_jsonl(FROZEN_ROOT / "prompts.jsonl", prompt_rows)
    dump_jsonl(FROZEN_ROOT / "annotations.jsonl", annotation_rows)

    selected = calibration["selected"]
    if selected["layers"] != FROZEN_BAND:
        raise SystemExit(
            f"Expected selected band {FROZEN_BAND}, found {selected['layers']}."
        )
    frozen_layer_selection = {
        "status": "FROZEN",
        "selection_data": "calibration split only",
        "selection_criterion": calibration["selection_criterion"],
        "tie_break": calibration["tie_break"],
        "band_width": calibration["band_width"],
        "selected_band": {
            **selected,
            "normalized_depths": calibration["normalized_depths"],
        },
        "representative_layer_rule": (
            "deterministic approximately-even coverage of the frozen band, "
            "including both endpoints; independent of held-out outcomes"
        ),
        "representative_layers": [
            {"native_layer": layer, "normalized_depth": layer / 30 * 100}
            for layer in REPRESENTATIVE_LAYERS
        ],
        "post_heldout_change_prohibited": True,
    }
    dump_json(OUTPUT_ROOT / "frozen_layer_selection.json", frozen_layer_selection)

    swap_path = OUTPUT_ROOT / "per_example_swap_results.jsonl"
    all_swap_rows = load_jsonl(swap_path)
    archival_swap_path = OUTPUT_ROOT / "per_example_swap_results_all_band_layers.jsonl"
    dump_jsonl(archival_swap_path, all_swap_rows)
    final_swap_rows = [
        row
        for row in all_swap_rows
        if row["intervention_scope"] != "single_layer"
        or row["patch_layers"][0] in REPRESENTATIVE_LAYERS
    ]
    dump_jsonl(swap_path, final_swap_rows)

    primary_by_id = {row["example_id"]: row for row in primary}
    band_rows = [
        row for row in final_swap_rows if row["intervention_scope"] == "band"
    ]
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {
        "relation_family": defaultdict(list),
        "concept_category": defaultdict(list),
        "template_group": defaultdict(list),
    }
    for result in band_rows:
        source = primary_by_id[result["example_id"]]
        relation = source["relation_or_task_family"]
        grouped["relation_family"][relation].append(result)
        grouped["concept_category"][concept_category(relation)].append(result)
        grouped["template_group"][source["parent_template_category_group"]].append(
            result
        )

    aggregate = {
        **original_aggregate,
        "status": "FROZEN_DIAGNOSTIC_HANDOFF",
        "representative_layers": REPRESENTATIVE_LAYERS,
        "final_per_example_rows": len(final_swap_rows),
        "grouping_note": (
            "The upstream file has category but no separate template identifier; "
            "parent_template_category_group is therefore conservatively identical "
            "to the official category."
        ),
        "concept_category_mapping": (
            "Deterministic mapping from relation/task family implemented in "
            "finalize_paper_aligned_diagnostics.py; no held-out outcome is used."
        ),
        "grouped_band_swap": {
            grouping: {
                key: summarize_swap_group(value)
                for key, value in sorted(groups.items())
            }
            for grouping, groups in grouped.items()
        },
        "scope": {
            "causcale_run": False,
            "discovery_features_rebuilt": False,
            "stage1_modified": False,
            "concept_128_expansion_started": False,
        },
    }
    dump_json(OUTPUT_ROOT / "aggregate_swap_results.json", aggregate)
    write_csv(OUTPUT_ROOT / "aggregate_swap_results.csv", aggregate)

    handoff = f"""# Paper-aligned prompt diagnostic handoff

## Decision

**Outcome B — interpretable diagnostic signal, but weak causal answer flipping.**

On the 33 frozen primary held-out examples, the six-layer band ({FROZEN_BAND[0]}–{FROZEN_BAND[-1]})
produced target-answer top-1 on 2/33 examples (6.06%, Wilson 95% CI
1.68%–19.61%). The original clean answer remained top-1 on 29/33 (87.88%).
Mean target log-probability changed by +2.668 nats and mean target-vs-clean
margin changed by +3.105 nats. The result supports a measurable directional
effect, not a robust causal swap.

## Frozen inputs

- Official Anthropic `probe-swap.json`: 90 examples, upstream commit
  `{UPSTREAM_COMMIT}`, SHA-256 `{expected_probe_hash}`.
- Primary eligible: {len(primary)} (12 calibration, 33 held-out).
- Secondary top-5-not-top-1 diagnostic: {len(secondary)}.
- Excluded: {len(excluded)}.
- `controlled-current32`: **missing (0 examples)**. No replacement or
  supplemental prompts were generated.

## Validation limitations

- Four exact duplicate-prompt groups are present in the upstream set.
- One secondary example (`bird-time-owl`) literally contains its swap answer
  `day`; no primary example has literal concept/answer leakage.
- Template/category groups are disjoint across splits.
- There are 22 source/target entities shared across calibration and held-out.
  This is a material leakage limitation. The split was not changed because the
  held-out results had already been observed.

## Frozen calibration and intervention

- Layer selection used calibration mean reciprocal rank only.
- Frozen band: layers {FROZEN_BAND}.
- Frozen representative layers: {REPRESENTATIVE_LAYERS}; chosen by deterministic
  approximately-even coverage of the band, including endpoints.
- J-lens held-out readout in the band: top-1 6/33, top-5 25/33, top-10 27/33;
  median minimum rank 4.
- Logit-lens held-out top-5: 11/33; median minimum rank 29.
- Numerical checks: coordinate swap max error below 6.7e-6, orthogonal
  preservation error below 3.9e-6, and source-to-source no-op output-logit
  change exactly 0 in the completed run.

## Guardrails and next decision

No CauScale run was started, discovery features were not rebuilt, Stage 1 was
not changed, and no 128-concept expansion was started.

Before any downstream graph/discovery experiment, decide whether the documented
entity overlap and missing controlled-current32 coverage are acceptable. If not,
the next work item is a **new preregistered dataset version and new untouched
held-out split**, not a mutation of this frozen result.
"""
    (OUTPUT_ROOT / "diagnostic_handoff.md").write_text(
        handoff, encoding="utf-8", newline="\n"
    )

    now = datetime.now(timezone.utc).isoformat()
    logs = {
        "prompt_preparation.log": (
            f"[{now}] Reused completed full clean preparation run.\n"
            f"official=90 tokenizer_pass=81 primary=45 secondary=19 excluded=26\n"
            "controlled_current32=0 status=MISSING_NOT_GENERATED\n"
        ),
        "prompt_validation.log": (
            f"[{now}] validator={validation['validator_status']}\n"
            f"exact_duplicate_groups={len(validation['duplicates']['exact_duplicate_groups'])}\n"
            f"near_duplicate_pairs={len(validation['duplicates']['near_duplicate_pairs'])}\n"
            f"literal_leakage={validation['literal_leakage']['count']}\n"
            f"cross_split_entity_overlap={validation['split_leakage']['entity_overlap_count']}\n"
        ),
        "layer_calibration.log": (
            f"[{now}] criterion={calibration['selection_criterion']}\n"
            f"frozen_band={FROZEN_BAND}\n"
            f"representative_layers={REPRESENTATIVE_LAYERS}\n"
            "heldout_used_for_selection=false\n"
        ),
        "swap_experiment.log": (
            f"[{now}] Repackaged completed swap run without rerunning the model.\n"
            f"band_rows={len(band_rows)} representative_layers={REPRESENTATIVE_LAYERS}\n"
            f"final_per_example_rows={len(final_swap_rows)}\n"
            "causcale_run=false discovery_features_rebuilt=false\n"
        ),
    }
    for filename, text in logs.items():
        (LOG_ROOT / filename).write_text(text, encoding="utf-8", newline="\n")

    config_path = ROOT / "task3_v1" / "configs" / "paper_aligned_jspace.yaml"
    frozen_paths = [
        OFFICIAL_ROOT / "LICENSE",
        OFFICIAL_ROOT / "README.md",
        OFFICIAL_ROOT / "probe-swap.json",
        OFFICIAL_ROOT / "SOURCE_MANIFEST.json",
        OFFICIAL_ROOT / "SHA256SUMS.txt",
        PROCESSED_ROOT / "processed_examples.jsonl",
        PROCESSED_ROOT / "prompt_filter_report.json",
        FROZEN_ROOT / "prompts.jsonl",
        FROZEN_ROOT / "annotations.jsonl",
        OUTPUT_ROOT / "prompt_validation_report.json",
        OUTPUT_ROOT / "eligible_examples.jsonl",
        OUTPUT_ROOT / "secondary_examples.jsonl",
        OUTPUT_ROOT / "excluded_examples.jsonl",
        OUTPUT_ROOT / "layer_calibration.json",
        OUTPUT_ROOT / "frozen_layer_selection.json",
        OUTPUT_ROOT / "per_example_swap_results.jsonl",
        OUTPUT_ROOT / "aggregate_swap_results.json",
        OUTPUT_ROOT / "aggregate_swap_results.csv",
        OUTPUT_ROOT / "diagnostic_handoff.md",
        config_path,
    ]
    manifest = {
        "status": "FROZEN",
        "frozen_at_utc": now,
        "freeze_policy": (
            "Prompt text, annotations, eligibility, split, selected band, and "
            "representative layers must not be modified. A new dataset version "
            "is required for any change."
        ),
        "dataset": {
            "official_anthropic": {
                "status": "FROZEN",
                "count": 90,
                "upstream_commit": UPSTREAM_COMMIT,
                "probe_swap_sha256": expected_probe_hash,
            },
            "controlled_current32": {
                "status": "MISSING_NOT_FROZEN",
                "count": 0,
                "supplement_generated": False,
            },
            "primary_eligible": len(primary),
            "secondary": len(secondary),
            "excluded": len(excluded),
        },
        "model_and_lens": {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "tokenizer_id": MODEL_ID,
            "tokenizer_revision": MODEL_REVISION,
            "local_model_revision_verified": run_manifest.get(
                "local_model_revision_verified"
            ),
            "lens_repository": LENS_REPOSITORY,
            "lens_file": LENS_FILE,
            "lens_revision": LENS_REVISION,
            "lens_sha256": LENS_SHA256,
        },
        "seed": SEED,
        "split_rule": validation["split_leakage"]["split_rule"],
        "known_split_limitation": validation["split_leakage"]["assessment"],
        "frozen_band": FROZEN_BAND,
        "representative_layers": REPRESENTATIVE_LAYERS,
        "configuration": {
            "path": rel(config_path),
            "sha256": sha256(config_path),
        },
        "git": git_snapshot(),
        "frozen_files": [
            {
                "path": rel(path),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in frozen_paths
        ],
        "logs": [rel(LOG_ROOT / filename) for filename in logs],
    }
    dump_json(MANIFEST_PATH, manifest)
    print(
        f"Wrote FROZEN manifest with {len(manifest['frozen_files'])} hashed files: "
        f"{MANIFEST_PATH}"
    )


if __name__ == "__main__":
    main()
