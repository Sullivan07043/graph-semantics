#!/usr/bin/env python3
"""Static validator for the controlled-current32 Stage-1 prompt candidate."""

from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = (
    ROOT / "task3_v1" / "data" / "prompts" / "controlled_current32" / "v1"
)
DEFAULT_OFFICIAL = (
    ROOT / "task3_v1" / "data" / "prompts" / "official_anthropic" / "probe-swap.json"
)
CONCEPT_SOURCE = ROOT / "task3_v1" / "build_discovery_matrix.py"
SEED = 20260725
CONDITIONS = [
    "explicit_single",
    "implicit_single",
    "contrast_single",
    "implicit_pair",
    "counterfactual_pair",
]
PAIR_CONDITIONS = {"implicit_pair", "counterfactual_pair"}
NEAR_DUPLICATE_THRESHOLD = 0.92
STATUS = "CANDIDATE_STATIC_VALIDATION_PASSED_NOT_BEHAVIORALLY_FROZEN"
REQUIRED_FIELDS = {
    "id",
    "prompt",
    "primary_concept",
    "secondary_concept",
    "condition",
    "fold",
    "template_family",
    "domain",
    "entities",
    "expected_answer",
    "primary_literal_present",
    "source",
    "generation_seed",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_concepts(source: Path = CONCEPT_SOURCE) -> list[str]:
    tree = ast.parse(source.read_text(encoding="utf-8-sig"), filename=str(source))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(
                isinstance(target, ast.Name) and target.id == "CONCEPTS"
                for target in node.targets
            ):
                value = ast.literal_eval(node.value)
                if (
                    not isinstance(value, list)
                    or len(value) != 32
                    or not all(isinstance(item, str) for item in value)
                ):
                    raise ValueError("CONCEPTS must be a literal list of 32 strings")
                return value
    raise ValueError(f"CONCEPTS was not found in {source}")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def literal_present(text: str, value: str) -> bool:
    target = normalize(value)
    return bool(target) and f" {target} " in f" {normalize(text)} "


def prompt_similarity(left: str, right: str) -> float:
    return difflib.SequenceMatcher(None, normalize(left), normalize(right)).ratio()


def near_duplicate_pairs(
    left_rows: list[dict[str, Any]],
    right_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    same_collection = right_rows is None
    right_rows = left_rows if right_rows is None else right_rows
    for left_index, left in enumerate(left_rows):
        start = left_index + 1 if same_collection else 0
        left_normalized = normalize(left["prompt"])
        for right in right_rows[start:]:
            right_normalized = normalize(right["prompt"])
            if left_normalized == right_normalized:
                continue
            shorter = min(len(left_normalized), len(right_normalized))
            longer = max(len(left_normalized), len(right_normalized))
            if not longer or shorter / longer < NEAR_DUPLICATE_THRESHOLD:
                continue
            ratio = difflib.SequenceMatcher(
                None, left_normalized, right_normalized
            ).ratio()
            if ratio >= NEAR_DUPLICATE_THRESHOLD:
                records.append(
                    {
                        "left": left["id"],
                        "right": right["id"],
                        "similarity": ratio,
                        "left_fold": left.get("fold"),
                        "right_fold": right.get("fold"),
                    }
                )
    return records


def range_record(counter: Counter[str | int]) -> dict[str, Any]:
    values = list(counter.values())
    return {
        "counts": {str(key): value for key, value in sorted(counter.items())},
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
        "range": max(values) - min(values) if values else None,
    }


def validate(
    rows: list[dict[str, Any]],
    official_rows: list[dict[str, Any]],
    concept_source: Path = CONCEPT_SOURCE,
) -> dict[str, Any]:
    concepts = load_concepts(concept_source)
    concept_set = set(concepts)
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    def fail(check: str, detail: Any) -> None:
        errors.append({"check": check, "detail": detail})

    if len(rows) != 1000:
        fail("row_count_exactly_1000", {"actual": len(rows)})

    missing_fields = {
        row.get("id", f"row_{index}"): sorted(REQUIRED_FIELDS - row.keys())
        for index, row in enumerate(rows)
        if REQUIRED_FIELDS - row.keys()
    }
    if missing_fields:
        fail("required_fields", missing_fields)

    ids = [row.get("id") for row in rows]
    expected_ids = [f"cc32_{index:04d}" for index in range(1, 1001)]
    if ids != expected_ids:
        fail(
            "id_sequence",
            {
                "expected_first_last": [expected_ids[0], expected_ids[-1]],
                "actual_first_last": [ids[0], ids[-1]] if ids else [],
                "unique": len(set(ids)),
            },
        )

    prompt_norms = [normalize(row.get("prompt", "")) for row in rows]
    exact_groups: dict[str, list[str]] = defaultdict(list)
    for row, prompt_norm in zip(rows, prompt_norms):
        exact_groups[prompt_norm].append(row.get("id"))
    exact_duplicates = [
        group for group in exact_groups.values() if len(group) > 1
    ]
    if exact_duplicates:
        fail("normalized_exact_duplicates", exact_duplicates)

    invalid_prompt_endings = [
        row.get("id")
        for row in rows
        if not isinstance(row.get("prompt"), str)
        or not row["prompt"].endswith("Answer:")
        or row["prompt"].count("Answer:") != 1
    ]
    if invalid_prompt_endings:
        fail("uniform_answer_marker", invalid_prompt_endings)

    invalid_source_seed = [
        row.get("id")
        for row in rows
        if row.get("source") != "generated_controlled_current32"
        or row.get("generation_seed") != SEED
    ]
    if invalid_source_seed:
        fail("source_and_seed", invalid_source_seed)

    invalid_primary = [
        row.get("id") for row in rows if row.get("primary_concept") not in concept_set
    ]
    invalid_secondary = [
        row.get("id")
        for row in rows
        if row.get("secondary_concept") is not None
        and row.get("secondary_concept") not in concept_set
    ]
    if invalid_primary:
        fail("primary_concept_inventory", invalid_primary)
    if invalid_secondary:
        fail("secondary_concept_inventory", invalid_secondary)

    primary_counts = Counter(row.get("primary_concept") for row in rows)
    missing_primary = [concept for concept in concepts if concept not in primary_counts]
    if missing_primary:
        fail("all_concepts_primary", missing_primary)
    primary_count_values = [primary_counts[concept] for concept in concepts]
    if (
        primary_count_values
        and (
            min(primary_count_values) != 31
            or max(primary_count_values) != 32
            or max(primary_count_values) - min(primary_count_values) > 1
        )
    ):
        fail("primary_count_balance_31_or_32", range_record(primary_counts))

    fold_counts = Counter(row.get("fold") for row in rows)
    if fold_counts != Counter({fold: 200 for fold in range(5)}):
        fail("five_folds_of_200", range_record(fold_counts))

    concept_fold_counts = Counter(
        (row.get("primary_concept"), row.get("fold")) for row in rows
    )
    bad_concept_folds = [
        {
            "concept": concept,
            "fold": fold,
            "count": concept_fold_counts[(concept, fold)],
        }
        for concept in concepts
        for fold in range(5)
        if concept_fold_counts[(concept, fold)] not in {6, 7}
    ]
    if bad_concept_folds:
        fail("concept_fold_counts_6_or_7", bad_concept_folds)

    condition_counts = Counter(row.get("condition") for row in rows)
    if condition_counts != Counter({condition: 200 for condition in CONDITIONS}):
        fail("condition_counts_exactly_200", range_record(condition_counts))

    concept_condition_counts = Counter(
        (row.get("primary_concept"), row.get("condition")) for row in rows
    )
    bad_concept_conditions = [
        {
            "concept": concept,
            "condition": condition,
            "count": concept_condition_counts[(concept, condition)],
        }
        for concept in concepts
        for condition in CONDITIONS
        if concept_condition_counts[(concept, condition)] not in {6, 7}
    ]
    if bad_concept_conditions:
        fail("concept_condition_counts_6_or_7", bad_concept_conditions)

    invalid_pair_schema = [
        row.get("id")
        for row in rows
        if (
            row.get("condition") in PAIR_CONDITIONS
            and (
                row.get("secondary_concept") is None
                or row.get("secondary_concept") == row.get("primary_concept")
            )
        )
        or (
            row.get("condition") not in PAIR_CONDITIONS
            and row.get("secondary_concept") is not None
        )
    ]
    if invalid_pair_schema:
        fail("single_pair_schema", invalid_pair_schema)
    pair_count = sum(row.get("secondary_concept") is not None for row in rows)
    if not 400 <= pair_count <= 600:
        fail("approximately_half_pair_rows", {"actual": pair_count})

    secondary_counts = Counter(
        row["secondary_concept"]
        for row in rows
        if row.get("secondary_concept") is not None
    )
    directed_pair_counts = Counter(
        (row["primary_concept"], row["secondary_concept"])
        for row in rows
        if row.get("secondary_concept") is not None
    )
    unordered_pair_counts = Counter(
        tuple(sorted((row["primary_concept"], row["secondary_concept"])))
        for row in rows
        if row.get("secondary_concept") is not None
    )
    if (
        set(secondary_counts) != concept_set
        or max(secondary_counts.values()) - min(secondary_counts.values()) > 2
    ):
        fail("secondary_concept_balance", range_record(secondary_counts))
    if directed_pair_counts and max(directed_pair_counts.values()) > 1:
        fail(
            "directed_pair_non_repetition",
            {
                "maximum": max(directed_pair_counts.values()),
                "repeated": [
                    {"pair": list(pair), "count": count}
                    for pair, count in directed_pair_counts.items()
                    if count > 1
                ],
            },
        )

    literal_errors: list[dict[str, Any]] = []
    answer_leakage: list[dict[str, Any]] = []
    for row in rows:
        prompt_body = row["prompt"][: -len("Answer:")].rstrip()
        primary_present = literal_present(prompt_body, row["primary_concept"])
        secondary_present = (
            literal_present(prompt_body, row["secondary_concept"])
            if row.get("secondary_concept")
            else False
        )
        expected_primary_presence = row["condition"] == "explicit_single"
        if primary_present != expected_primary_presence:
            literal_errors.append(
                {
                    "id": row["id"],
                    "condition": row["condition"],
                    "primary_present": primary_present,
                    "expected": expected_primary_presence,
                }
            )
        if row["primary_literal_present"] != primary_present:
            literal_errors.append(
                {
                    "id": row["id"],
                    "metadata_primary_literal_present": row[
                        "primary_literal_present"
                    ],
                    "computed": primary_present,
                }
            )
        if row["condition"] in {
            "implicit_single",
            "contrast_single",
            "implicit_pair",
            "counterfactual_pair",
        } and (primary_present or secondary_present):
            literal_errors.append(
                {
                    "id": row["id"],
                    "condition": row["condition"],
                    "primary_present": primary_present,
                    "secondary_present": secondary_present,
                }
            )
        if literal_present(prompt_body, row["expected_answer"]):
            answer_leakage.append(
                {"id": row["id"], "expected_answer": row["expected_answer"]}
            )
    if literal_errors:
        fail("concept_literal_policy", literal_errors)
    if answer_leakage:
        fail("literal_answer_leakage", answer_leakage)

    entity_folds: dict[str, set[int]] = defaultdict(set)
    template_folds: dict[str, set[int]] = defaultdict(set)
    empty_entities = []
    for row in rows:
        if not isinstance(row.get("entities"), list) or not row["entities"]:
            empty_entities.append(row.get("id"))
        for entity in row.get("entities", []):
            entity_folds[normalize(entity)].add(row["fold"])
        template_folds[row["template_family"]].add(row["fold"])
    entity_overlap = {
        entity: sorted(folds) for entity, folds in entity_folds.items() if len(folds) > 1
    }
    template_overlap = {
        family: sorted(folds)
        for family, folds in template_folds.items()
        if len(folds) > 1
    }
    if empty_entities:
        fail("entities_nonempty", empty_entities)
    if entity_overlap:
        fail("cross_fold_entity_overlap", entity_overlap)
    if template_overlap:
        fail("cross_fold_template_overlap", template_overlap)

    internal_near = near_duplicate_pairs(rows)
    if internal_near:
        fail("internal_near_duplicates", internal_near)

    official_adapter = [
        {
            "id": f"official90_{index:03d}",
            "prompt": item["prompt"],
            "fold": None,
        }
        for index, item in enumerate(official_rows)
    ]
    official_norms = {
        normalize(item["prompt"]): item["id"] for item in official_adapter
    }
    official_exact = [
        {"candidate": row["id"], "official": official_norms[normalize(row["prompt"])]}
        for row in rows
        if normalize(row["prompt"]) in official_norms
    ]
    if official_exact:
        fail("official90_exact_duplicates", official_exact)
    official_near = near_duplicate_pairs(rows, official_adapter)
    if official_near:
        fail("official90_near_duplicates", official_near)

    domain_counts = Counter(row.get("domain") for row in rows)
    template_counts = Counter(row.get("template_family") for row in rows)
    if len(domain_counts) < 8:
        warnings.append(
            {
                "check": "domain_diversity",
                "detail": f"Only {len(domain_counts)} domains are present.",
            }
        )

    return {
        "status": STATUS if not errors else "STATIC_VALIDATION_FAILED",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "hard_checks_passed": not errors,
        "behavioral_validation_performed": False,
        "behavioral_freeze_performed": False,
        "methodology": {
            "normalization": (
                "Unicode casefold, replace every run of non-ASCII-alphanumeric "
                "characters with one space, then trim."
            ),
            "exact_duplicate_check": "equality after normalization",
            "near_duplicate_check": (
                "Python difflib.SequenceMatcher ratio over normalized full prompt; "
                f"hard-fail threshold >= {NEAR_DUPLICATE_THRESHOLD:.2f}; pairs "
                "whose normalized length ratio is below the threshold are skipped "
                "because they cannot reach the threshold."
            ),
            "literal_check": (
                "case-insensitive whole normalized word/phrase containment in the "
                "prompt body before the terminal Answer: marker"
            ),
        },
        "summary": {
            "row_count": len(rows),
            "concept_count": len(concepts),
            "primary_concepts": range_record(primary_counts),
            "folds": range_record(fold_counts),
            "conditions": range_record(condition_counts),
            "pair_rows": pair_count,
            "single_rows": len(rows) - pair_count,
            "secondary_concepts": range_record(secondary_counts),
            "directed_pair_unique_count": len(directed_pair_counts),
            "directed_pair_maximum_repetition": (
                max(directed_pair_counts.values()) if directed_pair_counts else 0
            ),
            "unordered_pair_unique_count": len(unordered_pair_counts),
            "unordered_pair_maximum_repetition": (
                max(unordered_pair_counts.values()) if unordered_pair_counts else 0
            ),
            "domain_counts": dict(sorted(domain_counts.items())),
            "template_family_count": len(template_counts),
        },
        "duplicate_checks": {
            "internal_normalized_exact_duplicate_groups": exact_duplicates,
            "internal_near_duplicate_pairs": internal_near,
            "official90_exact_duplicate_pairs": official_exact,
            "official90_near_duplicate_pairs": official_near,
        },
        "leakage_checks": {
            "cross_fold_entity_overlap": entity_overlap,
            "cross_fold_template_overlap": template_overlap,
            "literal_answer_leakage": answer_leakage,
            "concept_literal_policy_errors": literal_errors,
        },
        "concept_inventory": concepts,
        "concept_source": {
            "path": concept_source.resolve().relative_to(ROOT.resolve()).as_posix(),
            "sha256": sha256(concept_source),
            "symbol": "CONCEPTS",
        },
        "errors": errors,
        "warnings": warnings,
        "manual_warning": (
            "Qwen behavioral validation has not been run. This dataset is a "
            "statically validated candidate and must not be treated as frozen."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", type=Path, default=DEFAULT_DATA_DIR
    )
    parser.add_argument(
        "--official-prompts", type=Path, default=DEFAULT_OFFICIAL
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    prompts_path = args.data_dir / "prompts.jsonl"
    report_path = args.report or args.data_dir / "validation_report.json"
    rows = load_jsonl(prompts_path)
    official = json.loads(args.official_prompts.read_text(encoding="utf-8"))["items"]
    report = validate(rows, official)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "rows": report["summary"]["row_count"],
                "errors": len(report["errors"]),
                "warnings": len(report["warnings"]),
                "report": str(report_path),
            },
            ensure_ascii=False,
        )
    )
    raise SystemExit(0 if report["hard_checks_passed"] else 1)


if __name__ == "__main__":
    main()
