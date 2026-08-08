"""Build the frozen concept bank for the CLIP-Dissect E5 text adaptation.

The project's full E5 bank contains more than half a million concepts.  The
original CLIP-Dissect protocol assumes a domain-appropriate bank, and scoring
the full bank against every respondent profile would be needlessly
intractable.  We therefore freeze a data-independent psychology/cognition
subset once, using only the generic domain anchors below.  No dataset text,
fold labels, latent names, or gold descriptions participate in selection.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import tempfile
from typing import Callable, Sequence
import zipfile

import numpy as np


BANK_VERSION = "text-dissect-e5-wordnet-psych-v2"
DOMAIN_ANCHORS = (
    "psychological construct",
    "personality trait",
    "cognitive ability",
    "executive function",
    "behavioral tendency",
    "social behavior",
    "emotion and motivation",
    "attitude belief and value",
    "interest and preference",
    "mental health symptom",
)
_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z' -]*[A-Za-z]$|^[A-Za-z]$")
WORDNET_LEXNAMES = (
    "noun.attribute",
    "noun.cognition",
    "noun.feeling",
    "noun.motive",
    "noun.state",
)


def _normalise_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / (np.linalg.norm(values, axis=1, keepdims=True) + 1e-9)


def _eligible(name: str) -> bool:
    clean = str(name).strip()
    return (
        2 <= len(clean) <= 60
        and 1 <= len(clean.split()) <= 4
        and _NAME_RE.fullmatch(clean) is not None
    )


def build_domain_bank(
    source_path: os.PathLike[str] | str,
    output_path: os.PathLike[str] | str,
    encoder: Callable[[Sequence[str]], np.ndarray],
    *,
    size: int = 4096,
    chunk_size: int = 8192,
) -> Path:
    """Select and atomically save a fixed domain subset of an E5 bank."""

    source = Path(source_path).resolve()
    output = Path(output_path).resolve()
    if size < 1 or chunk_size < 1:
        raise ValueError("size and chunk_size must be positive")
    if not source.is_file():
        raise FileNotFoundError(f"source concept bank not found: {source}")

    source_stat = source.stat()
    if output.is_file():
        try:
            with np.load(output, allow_pickle=True) as existing:
                valid = (
                    str(existing["selection_version"].item()) == BANK_VERSION
                    and int(existing["requested_size"].item()) == size
                    and int(existing["source_size"].item()) == source_stat.st_size
                    and int(existing["source_mtime_ns"].item()) == source_stat.st_mtime_ns
                    and tuple(str(x) for x in existing["anchors"]) == DOMAIN_ANCHORS
                )
            if valid:
                return output
        except (KeyError, OSError, ValueError):
            pass

    with np.load(source, allow_pickle=True) as payload:
        if not {"emb", "names"}.issubset(payload.files):
            raise ValueError("source concept bank must contain emb and names arrays")
        embeddings = payload["emb"]
        names = payload["names"]
        if embeddings.ndim != 2 or len(names) != len(embeddings):
            raise ValueError("source concept bank rows and names do not align")
        anchors = _normalise_rows(np.asarray(encoder(list(DOMAIN_ANCHORS))))
        if anchors.shape[1] != embeddings.shape[1]:
            raise ValueError("domain-anchor and concept embedding dimensions differ")

        candidate_scores: list[np.ndarray] = []
        candidate_indices: list[np.ndarray] = []
        for start in range(0, len(names), chunk_size):
            end = min(start + chunk_size, len(names))
            local_names = [str(value).strip() for value in names[start:end]]
            eligible_local = np.asarray(
                [index for index, name in enumerate(local_names) if _eligible(name)],
                dtype=int,
            )
            if not len(eligible_local):
                continue
            vectors = _normalise_rows(np.asarray(embeddings[start:end][eligible_local]))
            scores = np.max(vectors @ anchors.T, axis=1)
            keep = min(size, len(scores))
            positions = np.argpartition(scores, -keep)[-keep:]
            candidate_scores.append(scores[positions])
            candidate_indices.append(start + eligible_local[positions])

        if not candidate_indices:
            raise ValueError("source concept bank has no eligible 1-4 word concepts")
        scores = np.concatenate(candidate_scores)
        indices = np.concatenate(candidate_indices)
        order = np.lexsort((indices, -scores))
        selected = order[: min(size, len(order))]
        chosen_indices = indices[selected]
        chosen_scores = scores[selected]
        chosen_embeddings = np.asarray(embeddings[chosen_indices], dtype=np.float32)
        chosen_names = np.asarray([str(names[index]).strip() for index in chosen_indices], object)

    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=output.parent, prefix=f".{output.name}.", suffix=".npz"
    )
    os.close(descriptor)
    try:
        np.savez_compressed(
            temporary,
            emb=chosen_embeddings,
            names=chosen_names,
            domain_score=np.asarray(chosen_scores, dtype=np.float32),
            source_index=np.asarray(chosen_indices, dtype=np.int64),
            encoder=np.asarray("intfloat/e5-large-v2"),
            selection_version=np.asarray(BANK_VERSION),
            requested_size=np.asarray(size, dtype=np.int64),
            source_path=np.asarray(str(source)),
            source_size=np.asarray(source_stat.st_size, dtype=np.int64),
            source_mtime_ns=np.asarray(source_stat.st_mtime_ns, dtype=np.int64),
            anchors=np.asarray(DOMAIN_ANCHORS, object),
        )
        os.replace(temporary, output)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return output


def _wordnet_terms(path: Path) -> list[str]:
    """Read selected construct-like noun categories from a local WordNet zip."""

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
        if (len(fields) < 5 or not fields[0].isdigit()
                or not fields[1].isdigit()
                or lexnames.get(int(fields[1])) not in WORDNET_LEXNAMES):
            continue
        word_count = int(fields[3], 16)
        for index in range(word_count):
            name = fields[4 + 2 * index].replace("_", " ").strip().lower()
            if _eligible(name):
                terms.add(name)
    if not terms:
        raise ValueError("selected WordNet lexicographer categories contain no terms")
    return sorted(terms)


def build_wordnet_domain_bank(
    wordnet_zip: os.PathLike[str] | str,
    output_path: os.PathLike[str] | str,
    encoder: Callable[[Sequence[str]], np.ndarray],
    *,
    size: int = 4096,
) -> Path:
    """Build the frozen E5 psychology bank without dataset or gold text."""

    source = Path(wordnet_zip).resolve()
    output = Path(output_path).resolve()
    if size < 1:
        raise ValueError("size must be positive")
    source_stat = source.stat()
    if output.is_file():
        try:
            with np.load(output, allow_pickle=True) as existing:
                valid = (
                    str(existing["selection_version"].item()) == BANK_VERSION
                    and int(existing["requested_size"].item()) == size
                    and int(existing["source_size"].item()) == source_stat.st_size
                    and int(existing["source_mtime_ns"].item()) == source_stat.st_mtime_ns
                    and tuple(str(x) for x in existing["anchors"]) == DOMAIN_ANCHORS
                    and tuple(str(x) for x in existing["wordnet_lexnames"])
                    == WORDNET_LEXNAMES
                )
            if valid:
                return output
        except (KeyError, OSError, ValueError):
            pass

    names = _wordnet_terms(source)
    vectors = _normalise_rows(np.asarray(encoder(names)))
    anchors = _normalise_rows(np.asarray(encoder(list(DOMAIN_ANCHORS))))
    if vectors.shape[1] != anchors.shape[1]:
        raise ValueError("WordNet term and domain-anchor embedding dimensions differ")
    scores = np.max(vectors @ anchors.T, axis=1)
    indices = np.arange(len(names), dtype=np.int64)
    order = np.lexsort((indices, -scores))[: min(size, len(names))]
    chosen_embeddings = np.asarray(vectors[order], dtype=np.float32)
    chosen_names = np.asarray([names[index] for index in order], object)

    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=output.parent, prefix=f".{output.name}.", suffix=".npz"
    )
    os.close(descriptor)
    try:
        np.savez_compressed(
            temporary,
            emb=chosen_embeddings,
            names=chosen_names,
            domain_score=np.asarray(scores[order], dtype=np.float32),
            source_index=np.asarray(order, dtype=np.int64),
            encoder=np.asarray("intfloat/e5-large-v2"),
            selection_version=np.asarray(BANK_VERSION),
            requested_size=np.asarray(size, dtype=np.int64),
            source_path=np.asarray(str(source)),
            source_size=np.asarray(source_stat.st_size, dtype=np.int64),
            source_mtime_ns=np.asarray(source_stat.st_mtime_ns, dtype=np.int64),
            anchors=np.asarray(DOMAIN_ANCHORS, object),
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
    "DOMAIN_ANCHORS",
    "WORDNET_LEXNAMES",
    "build_domain_bank",
    "build_wordnet_domain_bank",
]
