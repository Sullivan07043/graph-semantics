"""Build a fixed robot-domain concept bank for the CLIP-Dissect adaptation.

Selection uses only generic robotics anchors, WordNet, and a static robot
telemetry vocabulary.  It never reads a dataset, a fold, or a masked label.
The explicit joint/axis atoms are necessary because WordNet names do not encode
the index distinctions that define much of the robot Task 1 label space.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import tempfile
from typing import Callable, Sequence
import zipfile

import numpy as np


BANK_VERSION = "clip-dissect-e5-wordnet-robot-v1"
ROBOT_ANCHORS = (
    "robot kinematics and telemetry",
    "robot arm joint angle",
    "robot arm joint angular velocity",
    "end effector Cartesian position",
    "end effector orientation quaternion",
    "robot gripper finger opening",
    "commanded robot translation",
    "commanded robot rotation",
    "robot gripper command",
)
WORDNET_LEXNAMES = (
    "noun.act",
    "noun.artifact",
    "noun.attribute",
    "noun.body",
    "noun.event",
    "noun.object",
    "noun.phenomenon",
    "noun.process",
    "noun.quantity",
    "noun.relation",
    "noun.shape",
    "noun.state",
)
_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z' -]*[A-Za-z]$|^[A-Za-z]$")


def _robot_terms() -> tuple[str, ...]:
    terms = [
        "joint angle",
        "joint angular velocity",
        "end effector position",
        "orientation quaternion",
        "gripper opening",
        "gripper opening speed",
        "translation command",
        "rotation command",
        "gripper command",
    ]
    for index in range(1, 8):
        terms.extend(
            (
                f"joint {index}",
                f"joint {index} angle",
                f"joint {index} angular velocity",
            )
        )
    for axis in ("x", "y", "z"):
        terms.extend(
            (
                f"{axis} coordinate",
                f"end effector {axis} position",
                f"{axis} quaternion component",
                f"{axis} translation command",
                f"{axis} rotation command",
            )
        )
    terms.append("w quaternion component")
    for index in (1, 2):
        terms.extend((f"finger {index} opening", f"finger {index} speed"))
    return tuple(dict.fromkeys(term.lower() for term in terms))


ROBOT_TERMS = _robot_terms()


def _normalise_rows(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    return matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)


def _eligible(name: str) -> bool:
    clean = str(name).strip()
    return (
        2 <= len(clean) <= 60
        and 1 <= len(clean.split()) <= 4
        and _NAME_RE.fullmatch(clean) is not None
    )


def _wordnet_terms(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"WordNet archive not found: {path}")
    with zipfile.ZipFile(path) as archive:
        lexnames = {
            int(line.split()[0]): line.split()[1]
            for line in archive.read("wordnet/lexnames").decode("ascii").splitlines()
            if line.split()
        }
        lines = archive.read("wordnet/data.noun").decode("ascii").splitlines()
    terms: set[str] = set()
    for line in lines:
        fields = line.split("|", 1)[0].split()
        if (
            len(fields) < 5
            or not fields[0].isdigit()
            or not fields[1].isdigit()
            or lexnames.get(int(fields[1])) not in WORDNET_LEXNAMES
        ):
            continue
        word_count = int(fields[3], 16)
        for index in range(word_count):
            name = fields[4 + 2 * index].replace("_", " ").strip().lower()
            if _eligible(name) and name not in ROBOT_TERMS:
                terms.add(name)
    if not terms:
        raise ValueError("robot-relevant WordNet categories contain no eligible terms")
    return sorted(terms)


def build_robot_wordnet_bank(
    wordnet_zip: os.PathLike[str] | str,
    output_path: os.PathLike[str] | str,
    encoder: Callable[[Sequence[str]], np.ndarray],
    *,
    size: int = 4096,
    encode_chunk_size: int = 2048,
) -> Path:
    """Build and atomically save the frozen base-E5 robot concept bank."""

    source = Path(wordnet_zip).resolve()
    output = Path(output_path).resolve()
    if size < len(ROBOT_TERMS):
        raise ValueError(f"bank size must retain all {len(ROBOT_TERMS)} robot atoms")
    if encode_chunk_size < 1:
        raise ValueError("encode_chunk_size must be positive")
    source_stat = source.stat()
    if output.is_file():
        try:
            with np.load(output, allow_pickle=True) as existing:
                valid = (
                    str(existing["selection_version"].item()) == BANK_VERSION
                    and int(existing["requested_size"].item()) == size
                    and int(existing["source_size"].item()) == source_stat.st_size
                    and int(existing["source_mtime_ns"].item()) == source_stat.st_mtime_ns
                    and tuple(str(value) for value in existing["anchors"]) == ROBOT_ANCHORS
                    and tuple(str(value) for value in existing["robot_terms"]) == ROBOT_TERMS
                    and tuple(str(value) for value in existing["wordnet_lexnames"])
                    == WORDNET_LEXNAMES
                )
            if valid:
                return output
        except (KeyError, OSError, ValueError):
            pass

    anchors = _normalise_rows(np.asarray(encoder(list(ROBOT_ANCHORS))))
    wordnet = _wordnet_terms(source)
    target_wordnet = size - len(ROBOT_TERMS)
    candidate_scores: list[np.ndarray] = []
    candidate_names: list[list[str]] = []
    for start in range(0, len(wordnet), encode_chunk_size):
        names = wordnet[start : start + encode_chunk_size]
        embeddings = _normalise_rows(np.asarray(encoder(names)))
        if embeddings.shape[1] != anchors.shape[1]:
            raise ValueError("WordNet and anchor embedding dimensions differ")
        scores = np.max(embeddings @ anchors.T, axis=1)
        keep = min(target_wordnet, len(names))
        positions = np.argpartition(scores, -keep)[-keep:]
        candidate_scores.append(scores[positions])
        candidate_names.append([names[int(position)] for position in positions])
    scores = np.concatenate(candidate_scores)
    names = [name for group in candidate_names for name in group]
    order = sorted(range(len(names)), key=lambda index: (-float(scores[index]), names[index]))
    selected_names = [names[index] for index in order[:target_wordnet]]
    selected_scores = np.asarray([scores[index] for index in order[:target_wordnet]], dtype=np.float32)

    final_names = [*ROBOT_TERMS, *selected_names]
    final_embeddings = _normalise_rows(np.asarray(encoder(final_names)))
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=output.parent, prefix=f".{output.name}.", suffix=".npz"
    )
    os.close(descriptor)
    try:
        np.savez_compressed(
            temporary,
            emb=np.asarray(final_embeddings, dtype=np.float32),
            names=np.asarray(final_names, object),
            domain_score=np.concatenate(
                (np.full(len(ROBOT_TERMS), np.inf, dtype=np.float32), selected_scores)
            ),
            encoder=np.asarray("intfloat/e5-large-v2"),
            selection_version=np.asarray(BANK_VERSION),
            requested_size=np.asarray(size, dtype=np.int64),
            source_path=np.asarray(str(source)),
            source_size=np.asarray(source_stat.st_size, dtype=np.int64),
            source_mtime_ns=np.asarray(source_stat.st_mtime_ns, dtype=np.int64),
            anchors=np.asarray(ROBOT_ANCHORS, object),
            robot_terms=np.asarray(ROBOT_TERMS, object),
            wordnet_lexnames=np.asarray(WORDNET_LEXNAMES, object),
        )
        os.replace(temporary, output)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return output


__all__ = [
    "BANK_VERSION",
    "ROBOT_ANCHORS",
    "ROBOT_TERMS",
    "WORDNET_LEXNAMES",
    "build_robot_wordnet_bank",
]
