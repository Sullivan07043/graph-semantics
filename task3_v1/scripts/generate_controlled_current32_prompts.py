#!/usr/bin/env python3
"""Generate the deterministic controlled-current32 v1 Stage-1 candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from validate_controlled_current32_prompts import (
    CONDITIONS,
    CONCEPT_SOURCE,
    DEFAULT_OFFICIAL,
    PAIR_CONDITIONS,
    ROOT,
    SEED,
    STATUS,
    load_concepts,
    sha256,
    validate,
)


DEFAULT_OUTPUT = (
    ROOT / "task3_v1" / "data" / "prompts" / "controlled_current32" / "v1"
)
DEFAULT_LOG_DIR = ROOT / "task3_v1" / "logs" / "controlled_current32" / "v1"
MAX_GENERATION_ATTEMPTS = 20


PROFILES: dict[str, dict[str, str]] = {
    "water": {
        "clue": "a clear drink drawn from a tap and poured into a glass",
        "question": "After water freezes completely, what common solid is formed?",
        "answer": "ice",
        "domain": "physical_world",
    },
    "fire": {
        "clue": "open flames giving off heat and smoke",
        "question": "What airborne material commonly rises above a fire?",
        "answer": "smoke",
        "domain": "physical_world",
    },
    "music": {
        "clue": "organized sounds arranged with melody and a steady beat",
        "question": "What structural pulse helps listeners follow music?",
        "answer": "rhythm",
        "domain": "culture",
    },
    "danger": {
        "clue": "an immediate threat that calls for protective action",
        "question": "What prudent response should a sign of danger trigger?",
        "answer": "caution",
        "domain": "safety",
    },
    "Italy": {
        "clue": "the boot-shaped European nation whose capital lies on the Tiber",
        "question": "Which capital city is the seat of Italy's national government?",
        "answer": "Rome",
        "domain": "geography",
    },
    "code": {
        "clue": "instructions written for a computer to execute",
        "question": "What runnable artifact can source code become after processing?",
        "answer": "program",
        "domain": "technology",
    },
    "animal": {
        "clue": "a living creature that moves, feeds, and responds to its surroundings",
        "question": "What biological term can classify an animal as a living individual?",
        "answer": "organism",
        "domain": "biology",
    },
    "happy": {
        "clue": "feeling pleased, cheerful, and satisfied with an outcome",
        "question": "Which facial expression commonly accompanies feeling happy?",
        "answer": "smile",
        "domain": "emotion",
    },
    "money": {
        "clue": "coins and banknotes accepted in exchange for goods",
        "question": "What general economic medium does money represent?",
        "answer": "currency",
        "domain": "economics",
    },
    "doctor": {
        "clue": "a licensed clinician who diagnoses illness and treats patients",
        "question": "Which professional field does a doctor practice?",
        "answer": "medicine",
        "domain": "health",
    },
    "city": {
        "clue": "a densely populated urban settlement with extensive public services",
        "question": "Which adjective commonly distinguishes city life from rural life?",
        "answer": "urban",
        "domain": "society",
    },
    "truth": {
        "clue": "a statement that matches the available facts",
        "question": "Which adjective describes a claim that expresses the truth?",
        "answer": "accurate",
        "domain": "reasoning",
    },
    "false": {
        "clue": "a claim that conflicts with the evidence",
        "question": "Which adjective describes a statement known to be false?",
        "answer": "incorrect",
        "domain": "reasoning",
    },
    "love": {
        "clue": "deep care, attachment, and concern for another person",
        "question": "What close emotional bond is most associated with love?",
        "answer": "affection",
        "domain": "emotion",
    },
    "anger": {
        "clue": "intense irritation after a perceived insult or injustice",
        "question": "What stronger word can describe an extreme burst of anger?",
        "answer": "rage",
        "domain": "emotion",
    },
    "fear": {
        "clue": "an uneasy response to a perceived threat",
        "question": "Which related feeling often accompanies persistent fear?",
        "answer": "anxiety",
        "domain": "emotion",
    },
    "food": {
        "clue": "edible material prepared to nourish a person",
        "question": "What benefit does food provide beyond satisfying appetite?",
        "answer": "nutrition",
        "domain": "daily_life",
    },
    "sleep": {
        "clue": "the nightly period of reduced awareness and bodily recovery",
        "question": "What restorative activity is another name for sleep?",
        "answer": "rest",
        "domain": "health",
    },
    "work": {
        "clue": "purposeful effort carried out to complete a useful task",
        "question": "What formal noun can describe sustained work?",
        "answer": "labor",
        "domain": "daily_life",
    },
    "school": {
        "clue": "an institution where pupils attend lessons with teachers",
        "question": "What central activity is a school designed to support?",
        "answer": "learning",
        "domain": "education",
    },
    "family": {
        "clue": "a household or kinship group connected across generations",
        "question": "What general term describes members of a family?",
        "answer": "relatives",
        "domain": "society",
    },
    "war": {
        "clue": "organized armed fighting between political groups",
        "question": "What broad category of organized hostilities does war represent?",
        "answer": "conflict",
        "domain": "history",
    },
    "peace": {
        "clue": "a condition without fighting, marked by stable cooperation",
        "question": "What social quality is closely associated with peace?",
        "answer": "harmony",
        "domain": "society",
    },
    "science": {
        "clue": "systematic study using observation, measurement, and experiments",
        "question": "What organized activity advances science by testing questions?",
        "answer": "research",
        "domain": "knowledge",
    },
    "art": {
        "clue": "creative expression through images, performance, or crafted forms",
        "question": "What human capacity is strongly expressed through art?",
        "answer": "creativity",
        "domain": "culture",
    },
    "language": {
        "clue": "a shared system of words and grammar used to convey meaning",
        "question": "What broad human function is enabled by language?",
        "answer": "communication",
        "domain": "linguistics",
    },
    "number": {
        "clue": "a mathematical value used for counting or measurement",
        "question": "What general property can a number represent?",
        "answer": "quantity",
        "domain": "mathematics",
    },
    "time": {
        "clue": "measured duration between events and changes",
        "question": "What measurable interval does time describe?",
        "answer": "duration",
        "domain": "reasoning",
    },
    "future": {
        "clue": "the period of events that have not yet occurred",
        "question": "Which everyday word points from the future to the next day?",
        "answer": "tomorrow",
        "domain": "temporality",
    },
    "past": {
        "clue": "the period containing events that have already occurred",
        "question": "Which everyday word points from the past to the previous day?",
        "answer": "yesterday",
        "domain": "temporality",
    },
    "safe": {
        "clue": "protected from likely harm under the stated conditions",
        "question": "Which adjective is a close synonym for safe?",
        "answer": "secure",
        "domain": "safety",
    },
    "risk": {
        "clue": "the possibility of loss or harm under uncertain conditions",
        "question": "What underlying feature makes a risk difficult to predict?",
        "answer": "uncertainty",
        "domain": "decision_making",
    },
}


FOLD_NAMES: list[list[str]] = [
    [
        "Avery Bennett", "Avery Foster", "Avery Griffin", "Avery Hayes",
        "Avery Jordan", "Blair Bennett", "Blair Foster", "Blair Griffin",
        "Blair Hayes", "Blair Jordan", "Casey Bennett", "Casey Foster",
        "Casey Griffin", "Casey Hayes", "Casey Jordan", "Drew Bennett",
        "Drew Foster", "Drew Griffin", "Drew Hayes", "Drew Jordan",
        "Emery Bennett", "Emery Foster", "Emery Griffin", "Emery Hayes",
        "Emery Jordan", "Finley Bennett", "Finley Foster", "Finley Griffin",
        "Finley Hayes", "Finley Jordan", "Gray Bennett", "Gray Foster",
        "Gray Griffin", "Gray Hayes", "Gray Jordan", "Harper Bennett",
        "Harper Foster", "Harper Griffin", "Harper Hayes", "Harper Jordan",
    ],
    [
        "Imani Keller", "Imani Lawson", "Imani Mercer", "Imani Nolan",
        "Imani Ortiz", "Jules Keller", "Jules Lawson", "Jules Mercer",
        "Jules Nolan", "Jules Ortiz", "Kai Keller", "Kai Lawson",
        "Kai Mercer", "Kai Nolan", "Kai Ortiz", "Lane Keller",
        "Lane Lawson", "Lane Mercer", "Lane Nolan", "Lane Ortiz",
        "Morgan Keller", "Morgan Lawson", "Morgan Mercer", "Morgan Nolan",
        "Morgan Ortiz", "Nico Keller", "Nico Lawson", "Nico Mercer",
        "Nico Nolan", "Nico Ortiz", "Oakley Keller", "Oakley Lawson",
        "Oakley Mercer", "Oakley Nolan", "Oakley Ortiz", "Parker Keller",
        "Parker Lawson", "Parker Mercer", "Parker Nolan", "Parker Ortiz",
    ],
    [
        "Quinn Patel", "Quinn Reed", "Quinn Silva", "Quinn Turner",
        "Quinn Vaughn", "Riley Patel", "Riley Reed", "Riley Silva",
        "Riley Turner", "Riley Vaughn", "Sage Patel", "Sage Reed",
        "Sage Silva", "Sage Turner", "Sage Vaughn", "Tatum Patel",
        "Tatum Reed", "Tatum Silva", "Tatum Turner", "Tatum Vaughn",
        "Uma Patel", "Uma Reed", "Uma Silva", "Uma Turner",
        "Uma Vaughn", "Val Patel", "Val Reed", "Val Silva",
        "Val Turner", "Val Vaughn", "Winter Patel", "Winter Reed",
        "Winter Silva", "Winter Turner", "Winter Vaughn", "Xen Patel",
        "Xen Reed", "Xen Silva", "Xen Turner", "Xen Vaughn",
    ],
    [
        "Yara Walker", "Yara Xu", "Yara Young", "Yara Zane",
        "Yara Abbott", "Zuri Walker", "Zuri Xu", "Zuri Young",
        "Zuri Zane", "Zuri Abbott", "Alden Walker", "Alden Xu",
        "Alden Young", "Alden Zane", "Alden Abbott", "Bria Walker",
        "Bria Xu", "Bria Young", "Bria Zane", "Bria Abbott",
        "Cleo Walker", "Cleo Xu", "Cleo Young", "Cleo Zane",
        "Cleo Abbott", "Dara Walker", "Dara Xu", "Dara Young",
        "Dara Zane", "Dara Abbott", "Enzo Walker", "Enzo Xu",
        "Enzo Young", "Enzo Zane", "Enzo Abbott", "Freya Walker",
        "Freya Xu", "Freya Young", "Freya Zane", "Freya Abbott",
    ],
    [
        "Galen Brooks", "Galen Chen", "Galen Diaz", "Galen Ellis",
        "Galen Frost", "Hana Brooks", "Hana Chen", "Hana Diaz",
        "Hana Ellis", "Hana Frost", "Idris Brooks", "Idris Chen",
        "Idris Diaz", "Idris Ellis", "Idris Frost", "Jo Brooks",
        "Jo Chen", "Jo Diaz", "Jo Ellis", "Jo Frost",
        "Kira Brooks", "Kira Chen", "Kira Diaz", "Kira Ellis",
        "Kira Frost", "Leif Brooks", "Leif Chen", "Leif Diaz",
        "Leif Ellis", "Leif Frost", "Mina Brooks", "Mina Chen",
        "Mina Diaz", "Mina Ellis", "Mina Frost", "Noor Brooks",
        "Noor Chen", "Noor Diaz", "Noor Ellis", "Noor Frost",
    ],
]

FOLD_CONTEXTS = [
    [
        "during a museum reference review",
        "while checking a field guide",
        "during a community briefing",
        "while editing a public information card",
        "during an archive fact check",
        "while preparing a library display",
        "during a radio research segment",
        "while reviewing a classroom handout",
    ],
    [
        "during a clinic simulation",
        "while planning a neighborhood event",
        "during a design meeting",
        "while reviewing an operations note",
        "during a training exercise",
        "while preparing a visitor guide",
        "during a laboratory orientation",
        "while drafting a service memo",
    ],
    [
        "during a comparative reading session",
        "while evaluating two case summaries",
        "during a policy comparison",
        "while reviewing contrasting reports",
        "during an editorial assessment",
        "while comparing two explanations",
        "during a structured debate",
        "while checking alternative interpretations",
    ],
    [
        "during a hypothetical planning exercise",
        "while testing an altered scenario",
        "during a contingency discussion",
        "while examining a changed assumption",
        "during a counterfactual review",
        "while exploring an imagined constraint",
        "during a what-if analysis",
        "while revising a hypothetical case",
    ],
    [
        "during a short classification task",
        "while tagging a concise case note",
        "during a rapid categorization exercise",
        "while labeling a briefing excerpt",
        "during a compact review task",
        "while sorting a scenario card",
        "during a one-label assessment",
        "while coding a short observation",
    ],
]


IMPLICIT_TEMPLATES = [
    "{entity}, {context}, reads about {clue}. Which single concept best names the central subject?",
    "{context_cap}, {entity} encounters a report centered on {clue}. What concept provides the clearest label?",
    "{entity} studies a concise case involving {clue} {context}. Which concept captures its main theme?",
    "A note reviewed by {entity} {context} describes {clue}. What is the most direct conceptual label?",
    "{context_cap}, a scenario given to {entity} focuses on {clue}. Which concept is primarily evoked?",
    "{entity} must classify an observation about {clue} {context}. What single concept should be assigned?",
    "The central detail in {entity}'s case {context} is {clue}. Which concept organizes that detail?",
    "{entity} summarizes {clue} {context}. What concept should head the summary?",
]

CONTRAST_TEMPLATES = [
    "{entity}, {context}, contrasts {clue} with an unrelated routine detail. Which concept names the first side?",
    "{context_cap}, {entity} compares {clue} against a neutral background case. What concept distinguishes the former?",
    "{entity} reviews one case about {clue} and another with no matching theme {context}. Which concept belongs to the first?",
    "A comparison prepared by {entity} {context} places {clue} beside an ordinary control case. What labels the focal case?",
    "{context_cap}, the emphasized side of {entity}'s comparison concerns {clue}, not the control. Which concept is emphasized?",
    "{entity} separates a description of {clue} from a generic baseline {context}. What concept defines the meaningful contrast?",
    "In {entity}'s paired notes {context}, only the first concerns {clue}. Which concept identifies that difference?",
    "{entity} highlights {clue} while dismissing a neutral comparison {context}. What is the highlighted concept?",
]

PAIR_TEMPLATES = [
    "{entity}, {context}, treats {primary} as the main issue; a side note mentions {secondary}. Which concept best labels the main issue?",
    "{context_cap}, {entity} prioritizes a case about {primary} over background material about {secondary}. What is the primary concept?",
    "{entity} reads two signals {context}: the focal one concerns {primary}, while the supporting one concerns {secondary}. Which concept is focal?",
    "A brief reviewed by {entity} {context} centers on {primary} and only secondarily notes {secondary}. What label belongs to the center?",
    "{context_cap}, the main thread in {entity}'s case is {primary}; {secondary} supplies context. Which concept should receive the primary tag?",
    "{entity} must label the dominant theme {context}. The dominant evidence concerns {primary}, with {secondary} as a secondary cue. What is the label?",
    "In a two-topic note, {entity} {context} gives priority to {primary} and less weight to {secondary}. Which concept has priority?",
    "{entity} summarizes a case {context} where {primary} drives the question and {secondary} remains peripheral. What concept drives it?",
]

COUNTERFACTUAL_TEMPLATES = [
    "{entity}, {context}, imagines that the element involving {primary} disappears while {secondary} remains unchanged. Which concept was removed?",
    "{context_cap}, {entity} assumes the situation involving {primary} never occurs, but the one involving {secondary} still does. What concept is absent?",
    "{entity} alters a case {context}: evidence for {primary} is taken away and evidence for {secondary} is retained. Which concept lost its evidence?",
    "In {entity}'s hypothetical revision {context}, {primary} is eliminated while {secondary} stays in place. What concept was eliminated?",
    "{context_cap}, the changed assumption in {entity}'s case suppresses {primary} without affecting {secondary}. Which concept is suppressed?",
    "{entity} tests a counterfactual {context} where {primary} no longer applies and {secondary} still applies. What concept no longer applies?",
    "A what-if note for {entity} {context} removes the circumstance described as {primary} but preserves {secondary}. Which concept does the removal target?",
    "{entity} revises one premise {context}: {primary} is missing, whereas {secondary} continues normally. Which concept is missing?",
]

EXPLICIT_LEADS = [
    "{entity}, {context}, asks a direct factual question.",
    "{context_cap}, {entity} prepares a reference answer.",
    "{entity} checks a clearly named topic {context}.",
    "A factual card reviewed by {entity} {context} names its subject directly.",
    "{context_cap}, {entity} resolves a straightforward query.",
    "{entity} works from an explicit subject label {context}.",
    "The direct-reference task assigned to {entity} occurs {context}.",
    "{entity} verifies a named concept {context}.",
]


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
        newline="\n",
    )


def axis_counts(concept_count: int, offset: int) -> list[list[int]]:
    """Return per-concept counts over five bins: 6 each plus balanced extras."""
    counts = [[6] * 5 for _ in range(concept_count)]
    first_targets = [(index + offset) % 5 for index in range(concept_count)]
    totals = [6 * concept_count] * 5
    for index, target in enumerate(first_targets):
        counts[index][target] += 1
        totals[target] += 1
    for index in range(8):
        candidates = [
            target
            for target in range(5)
            if target != first_targets[index] and totals[target] < 200
        ]
        target = min(
            candidates,
            key=lambda item: (
                totals[item],
                (item - (index * 2 + offset + 1)) % 5,
                item,
            ),
        )
        counts[index][target] += 1
        totals[target] += 1
    if totals != [200] * 5:
        raise RuntimeError(f"Axis balancing failed: {totals}")
    return counts


def build_slots(concepts: list[str], attempt: int) -> list[dict[str, Any]]:
    fold_axis = axis_counts(len(concepts), offset=0)
    condition_axis = axis_counts(len(concepts), offset=2)
    slots: list[dict[str, Any]] = []
    for concept_index, concept in enumerate(concepts):
        folds = [
            fold
            for fold, count in enumerate(fold_axis[concept_index])
            for _ in range(count)
        ]
        conditions = [
            condition
            for condition, count in zip(
                CONDITIONS, condition_axis[concept_index]
            )
            for _ in range(count)
        ]
        random.Random(SEED + 101 * concept_index + attempt).shuffle(folds)
        random.Random(SEED + 211 * concept_index + attempt).shuffle(conditions)
        for occurrence, (fold, condition) in enumerate(zip(folds, conditions)):
            slots.append(
                {
                    "concept_index": concept_index,
                    "primary_concept": concept,
                    "fold": fold,
                    "condition": condition,
                    "occurrence": occurrence,
                }
            )

    pair_slots = [slot for slot in slots if slot["condition"] in PAIR_CONDITIONS]
    target_capacities = {
        index: 13 if index < 16 else 12 for index in range(32)
    }
    remaining = dict(target_capacities)
    slots_by_source: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for slot in pair_slots:
        slots_by_source[slot["concept_index"]].append(slot)
    source_order = sorted(
        slots_by_source,
        key=lambda source: (-len(slots_by_source[source]), source),
    )
    for source in source_order:
        source_slots = sorted(
            slots_by_source[source],
            key=lambda slot: (slot["fold"], slot["condition"], slot["occurrence"]),
        )
        candidates = [target for target in range(32) if target != source]
        targets = sorted(
            candidates,
            key=lambda target: (
                -remaining[target],
                (target - source - 1) % 32,
                target,
            ),
        )[: len(source_slots)]
        if (
            len(targets) != len(source_slots)
            or any(remaining[target] <= 0 for target in targets)
        ):
            raise RuntimeError("Unable to construct a balanced nonrepeating pairing")
        for slot, target in zip(source_slots, targets):
            slot["secondary_concept"] = concepts[target]
            remaining[target] -= 1
    target_usage = Counter(
        {
            target: target_capacities[target] - remaining[target]
            for target in range(32)
        }
    )
    if any(remaining.values()):
        raise RuntimeError(f"Secondary capacities were not exhausted: {remaining}")
    if sorted(target_usage.values()) != [12] * 16 + [13] * 16:
        raise RuntimeError(f"Secondary balancing failed: {target_usage}")

    random.Random(SEED + 997 * attempt).shuffle(slots)
    return slots


def render_prompt(
    slot: dict[str, Any],
    entity: str,
    fold_position: int,
    variant: int,
) -> tuple[str, str, str]:
    concept = slot["primary_concept"]
    label = concept.strip()
    profile = PROFILES[label]
    fold = slot["fold"]
    condition = slot["condition"]
    context = FOLD_CONTEXTS[fold][(fold_position + variant) % 8]
    values = {
        "entity": entity,
        "context": context,
        "context_cap": context[0].upper() + context[1:],
        "clue": profile["clue"],
    }
    if condition == "explicit_single":
        lead = EXPLICIT_LEADS[variant].format(**values)
        prompt = f"{lead} {profile['question']} Answer:"
        return prompt, profile["answer"], "factual_question"
    if condition == "implicit_single":
        prompt = IMPLICIT_TEMPLATES[variant].format(**values) + " Answer:"
        return prompt, label, "scenario_completion"
    if condition == "contrast_single":
        prompt = CONTRAST_TEMPLATES[variant].format(**values) + " Answer:"
        return prompt, label, "comparison"

    secondary = slot["secondary_concept"].strip()
    pair_values = {
        **values,
        "primary": profile["clue"],
        "secondary": PROFILES[secondary]["clue"],
    }
    if condition == "implicit_pair":
        prompt = PAIR_TEMPLATES[variant].format(**pair_values) + " Answer:"
        return prompt, label, "short_classification"
    prompt = COUNTERFACTUAL_TEMPLATES[variant].format(**pair_values) + " Answer:"
    return prompt, label, "counterfactual_reasoning"


def generate_rows(concepts: list[str], attempt: int) -> list[dict[str, Any]]:
    if list(PROFILES) != [concept.strip() for concept in concepts]:
        raise RuntimeError(
            "Semantic profiles do not exactly match the source concept order"
        )
    slots = build_slots(concepts, attempt)
    fold_positions = Counter()
    local_variants = Counter()
    rows: list[dict[str, Any]] = []
    for index, slot in enumerate(slots, start=1):
        fold = slot["fold"]
        fold_position = fold_positions[fold]
        entity = FOLD_NAMES[fold][fold_position % len(FOLD_NAMES[fold])]
        variant_key = (
            slot["primary_concept"],
            fold,
            slot["condition"],
        )
        variant = (
            local_variants[variant_key]
            + fold * 3
            + slot["concept_index"]
            + attempt
        ) % 8
        local_variants[variant_key] += 1
        prompt, expected_answer, task_type = render_prompt(
            slot, entity, fold_position, variant
        )
        rows.append(
            {
                "id": f"cc32_{index:04d}",
                "prompt": prompt,
                "primary_concept": slot["primary_concept"],
                "secondary_concept": slot.get("secondary_concept"),
                "condition": slot["condition"],
                "fold": fold,
                "template_family": (
                    f"fold{fold}_{task_type}_v{variant:02d}"
                ),
                "task_type": task_type,
                "domain": PROFILES[slot["primary_concept"].strip()]["domain"],
                "entities": [entity],
                "expected_answer": expected_answer,
                "primary_literal_present": slot["condition"] == "explicit_single",
                "source": "generated_controlled_current32",
                "generation_seed": SEED,
            }
        )
        fold_positions[fold] += 1
    return rows


def git_snapshot() -> dict[str, Any]:
    safe = ROOT.as_posix()

    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={safe}", *args],
            cwd=ROOT,
            encoding="utf-8",
            text=True,
        ).strip()

    try:
        status = run("status", "--short")
        return {
            "commit": run("rev-parse", "HEAD"),
            "dirty": bool(status),
            "status_short": status.splitlines(),
        }
    except (OSError, subprocess.CalledProcessError) as error:
        return {"commit": None, "dirty": None, "error": str(error)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--official-prompts", type=Path, default=DEFAULT_OFFICIAL)
    args = parser.parse_args()

    concepts = load_concepts()
    official = json.loads(args.official_prompts.read_text(encoding="utf-8"))["items"]
    attempt_records = []
    rows: list[dict[str, Any]] | None = None
    report: dict[str, Any] | None = None
    selected_attempt = None
    for attempt in range(MAX_GENERATION_ATTEMPTS):
        candidate = generate_rows(concepts, attempt)
        candidate_report = validate(candidate, official)
        attempt_records.append(
            {
                "attempt": attempt,
                "hard_checks_passed": candidate_report["hard_checks_passed"],
                "error_checks": [
                    item["check"] for item in candidate_report["errors"]
                ],
            }
        )
        if candidate_report["hard_checks_passed"]:
            rows = candidate
            report = candidate_report
            selected_attempt = attempt
            break
    if rows is None or report is None or selected_attempt is None:
        args.log_dir.mkdir(parents=True, exist_ok=True)
        dump_json(
            args.log_dir / "generation_failures.json",
            {"seed": SEED, "attempts": attempt_records},
        )
        raise SystemExit(
            f"No valid 1,000-row candidate after {MAX_GENERATION_ATTEMPTS} attempts"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    prompts_path = args.output_dir / "prompts.jsonl"
    inventory_path = args.output_dir / "concept_inventory.json"
    folds_path = args.output_dir / "fold_assignments.json"
    validation_path = args.output_dir / "validation_report.json"
    readme_path = args.output_dir / "README.md"
    manifest_path = args.output_dir / "generation_manifest.json"

    dump_jsonl(prompts_path, rows)
    concept_inventory = {
        "status": STATUS,
        "count": len(concepts),
        "source": {
            "path": CONCEPT_SOURCE.resolve().relative_to(ROOT.resolve()).as_posix(),
            "symbol": "CONCEPTS",
            "sha256": sha256(CONCEPT_SOURCE),
        },
        "concept_tokens": [
            {
                "index": index,
                "token": concept,
                "leading_space_preserved": concept.startswith(" "),
            }
            for index, concept in enumerate(concepts)
        ],
    }
    dump_json(inventory_path, concept_inventory)

    fold_assignments = {
        "status": STATUS,
        "rule": (
            "Deterministic balanced assignment: every fold has 200 rows; every "
            "concept appears 6 or 7 times per fold. Named entities and template "
            "families are fold-exclusive."
        ),
        "seed": SEED,
        "folds": {
            str(fold): {
                "count": sum(row["fold"] == fold for row in rows),
                "ids": [row["id"] for row in rows if row["fold"] == fold],
                "entities": sorted(
                    {
                        entity
                        for row in rows
                        if row["fold"] == fold
                        for entity in row["entities"]
                    }
                ),
                "template_families": sorted(
                    {
                        row["template_family"]
                        for row in rows
                        if row["fold"] == fold
                    }
                ),
            }
            for fold in range(5)
        },
    }
    dump_json(folds_path, fold_assignments)
    dump_json(validation_path, report)

    readme = f"""# controlled-current32 v1

This directory contains a deterministic 1,000-prompt candidate for Stage 1
discovery. It is **not** an intervention set and it is **not behaviorally
frozen**.

Status: `{STATUS}`

The only concept inventory is the ordered `CONCEPTS` literal in
`task3_v1/build_discovery_matrix.py`. Leading spaces, capitalization, order, and
the total of 32 tokens are preserved in `concept_inventory.json` and every
prompt record.

All prompts end exactly with `Answer:`. Answers and concept labels live only in
metadata, except that `explicit_single` deliberately names its primary concept.
The other four conditions do not literally name their target concepts.

Files:

- `prompts.jsonl`: 1,000 Stage-1 discovery candidate records.
- `concept_inventory.json`: exact concept list plus source provenance.
- `fold_assignments.json`: five folds of 200 with fold-exclusive entities and
  template families.
- `validation_report.json`: static schema, balance, duplicate, and leakage
  checks.
- `generation_manifest.json`: generation settings, complete concept list,
  provenance, hashes, and repository snapshot.

Generation and validation logs are stored separately under
`task3_v1/logs/controlled_current32/v1/`.

No artificial concept co-occurrence in this dataset is asserted to be causal
ground truth. Future intervention prompts must be generated separately after
candidate edges have been selected.
"""
    readme_path.write_text(readme, encoding="utf-8", newline="\n")

    generated_files = [
        prompts_path,
        inventory_path,
        folds_path,
        validation_path,
        readme_path,
    ]
    manifest = {
        "status": STATUS,
        "dataset_id": "controlled_current32_v1",
        "purpose": "Stage 1 discovery only",
        "intervention_prompt_set": False,
        "behavioral_validation": {
            "performed": False,
            "reason": (
                "The existing behavioral flow is tailored to official "
                "single-token answer prompts and is not directly reusable for "
                "this mixed-task candidate without changing its protocol."
            ),
        },
        "frozen": False,
        "frozen_copy_created": False,
        "generation_seed": SEED,
        "selected_generation_attempt": selected_attempt,
        "row_count": len(rows),
        "fold_count": 5,
        "fold_size": 200,
        "conditions": CONDITIONS,
        "concept_inventory": {
            "source_file": CONCEPT_SOURCE.resolve()
            .relative_to(ROOT.resolve())
            .as_posix(),
            "source_symbol": "CONCEPTS",
            "source_file_sha256": sha256(CONCEPT_SOURCE),
            "count": len(concepts),
            "ordered_tokens": concepts,
            "preservation_rule": (
                "Names, leading spaces, capitalization, order, and count are "
                "copied exactly from the source literal."
            ),
        },
        "official90_comparison_source": {
            "path": args.official_prompts.resolve()
            .relative_to(ROOT.resolve())
            .as_posix(),
            "sha256": sha256(args.official_prompts),
        },
        "validation": {
            "hard_checks_passed": report["hard_checks_passed"],
            "near_duplicate_method": report["methodology"][
                "near_duplicate_check"
            ],
            "warnings": report["warnings"],
            "manual_warning": report["manual_warning"],
        },
        "files": [
            {
                "path": path.resolve().relative_to(ROOT.resolve()).as_posix(),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in generated_files
        ],
        "logs": [
            (
                args.log_dir / "generation.log"
            ).resolve().relative_to(ROOT.resolve()).as_posix(),
            (
                args.log_dir / "validation.log"
            ).resolve().relative_to(ROOT.resolve()).as_posix(),
        ],
        "git": git_snapshot(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "causal_ground_truth_claimed": False,
        "guardrails": {
            "official90_modified": False,
            "existing_results_overwritten": False,
            "causcale_run": False,
            "new_jspace_features_extracted": False,
            "concept_128_scale_started": False,
        },
    }
    dump_json(manifest_path, manifest)

    generation_log = {
        "status": STATUS,
        "seed": SEED,
        "selected_attempt": selected_attempt,
        "attempts": attempt_records,
        "rows": len(rows),
        "primary_count_range": [
            min(Counter(row["primary_concept"] for row in rows).values()),
            max(Counter(row["primary_concept"] for row in rows).values()),
        ],
        "fold_counts": dict(sorted(Counter(row["fold"] for row in rows).items())),
        "condition_counts": dict(
            sorted(Counter(row["condition"] for row in rows).items())
        ),
    }
    dump_json(args.log_dir / "generation.log", generation_log)
    validation_log = {
        "status": report["status"],
        "hard_checks_passed": report["hard_checks_passed"],
        "error_count": len(report["errors"]),
        "warning_count": len(report["warnings"]),
        "duplicate_checks": {
            key: len(value)
            for key, value in report["duplicate_checks"].items()
        },
        "leakage_checks": {
            key: len(value) for key, value in report["leakage_checks"].items()
        },
    }
    dump_json(args.log_dir / "validation.log", validation_log)

    print(
        json.dumps(
            {
                "status": STATUS,
                "rows": len(rows),
                "selected_attempt": selected_attempt,
                "output_dir": str(args.output_dir),
                "validation_errors": len(report["errors"]),
                "validation_warnings": len(report["warnings"]),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
