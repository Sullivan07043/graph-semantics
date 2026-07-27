from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "task3_v1" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import generate_controlled_current32_prompts as generator  # noqa: E402
import validate_controlled_current32_prompts as validator  # noqa: E402


DATA_DIR = (
    ROOT / "task3_v1" / "data" / "prompts" / "controlled_current32" / "v1"
)


def load_rows() -> list[dict]:
    return [
        json.loads(line)
        for line in (DATA_DIR / "prompts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]


def test_source_inventory_is_exact_and_ordered() -> None:
    concepts = validator.load_concepts()
    inventory = json.loads(
        (DATA_DIR / "concept_inventory.json").read_text(encoding="utf-8")
    )
    assert len(concepts) == 32
    assert [item["token"] for item in inventory["concept_tokens"]] == concepts
    assert all(concept.startswith(" ") for concept in concepts)


def test_saved_candidate_has_required_balancing() -> None:
    rows = load_rows()
    assert len(rows) == 1000
    assert Counter(row["fold"] for row in rows) == Counter(
        {fold: 200 for fold in range(5)}
    )
    assert Counter(row["condition"] for row in rows) == Counter(
        {condition: 200 for condition in validator.CONDITIONS}
    )
    primary = Counter(row["primary_concept"] for row in rows)
    assert min(primary.values()) == 31
    assert max(primary.values()) == 32
    per_fold = Counter((row["primary_concept"], row["fold"]) for row in rows)
    assert set(per_fold.values()) <= {6, 7}


def test_generation_is_deterministic_for_seed() -> None:
    concepts = validator.load_concepts()
    first = generator.generate_rows(concepts, attempt=0)
    second = generator.generate_rows(concepts, attempt=0)
    assert first == second
    assert first == load_rows()


def test_report_is_static_candidate_not_frozen() -> None:
    report = json.loads(
        (DATA_DIR / "validation_report.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (DATA_DIR / "generation_manifest.json").read_text(encoding="utf-8")
    )
    assert report["hard_checks_passed"] is True
    assert report["errors"] == []
    assert report["warnings"] == []
    assert report["status"] == validator.STATUS
    assert manifest["status"] == validator.STATUS
    assert manifest["behavioral_validation"]["performed"] is False
    assert manifest["frozen"] is False


def test_manifest_hashes_match_candidate_files() -> None:
    manifest = json.loads(
        (DATA_DIR / "generation_manifest.json").read_text(encoding="utf-8")
    )
    for record in manifest["files"]:
        path = ROOT / record["path"]
        if not path.is_file() and record["path"].startswith("task3/"):
            path = ROOT / "task3_v1" / Path(record["path"]).relative_to("task3")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == record["sha256"]

