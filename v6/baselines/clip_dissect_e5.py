"""Deterministic CLIP-Dissect E5 text adaptation for Task 2.

This is a text-domain adaptation of CLIP-Dissect (Oikarinen & Weng, ICLR
2023); this E5 text variant is a project adaptation, not an upstream release.
Probe images become fold-visible respondent/profile texts, CLIP becomes the
project's frozen E5 encoder, and every scalar latent is scored as two target
units: its positive and negative activation poles.

The implementation keeps the source method's SoftWPMI defaults (temperature
10, lambda 1, top 100 probes, membership 0.998 -> 0.97) and rank-reordering
defaults (top 5%, p=3).  CLIP-Dissect reports those as alternative similarity
functions.  Here their ordinal ranks are averaged (Borda-style), because the
report specifies both and their raw scales are not commensurate.  The original
rank-reorder code uses unseeded random permutations for normalization; this
module uses a local, fixed-seed generator so reruns are bitwise deterministic
for identical NumPy/BLAS inputs.

No API, LLM, semantic judge, or gold latent description is accepted or used.
The caller is responsible for supplying fold-visible profile text only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


TextEncoder = Callable[[Sequence[str]], np.ndarray]
SCORER_VERSION = "text-dissect-e5-tie-neutral-v3"


@dataclass(frozen=True)
class ConceptBank:
    """A fixed concept vocabulary and its embeddings in the profile-text space."""

    embeddings: np.ndarray
    names: tuple[str, ...]
    source: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        embeddings = np.asarray(self.embeddings)
        names = tuple(str(name).strip() for name in self.names)
        if embeddings.ndim != 2:
            raise ValueError(
                f"concept embeddings must be a 2-D [concept, dimension] array; "
                f"got shape {embeddings.shape}"
            )
        if embeddings.shape[0] != len(names):
            raise ValueError(
                "concept-bank row/name mismatch: "
                f"{embeddings.shape[0]} embedding rows versus {len(names)} names"
            )
        if embeddings.shape[0] == 0 or embeddings.shape[1] == 0:
            raise ValueError("concept bank must contain at least one non-empty embedding")
        if not np.issubdtype(embeddings.dtype, np.number):
            raise TypeError(f"concept embeddings must be numeric; got {embeddings.dtype}")
        empty = next((i for i, name in enumerate(names) if not name), None)
        if empty is not None:
            raise ValueError(f"concept name at row {empty} is empty")
        object.__setattr__(self, "embeddings", embeddings)
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class TextDissectConfig:
    """Fixed scoring settings for the CLIP-Dissect E5 text adaptation."""

    top_k: int = 6
    soft_top_k: int = 100
    temperature: float = 10.0
    background_lambda: float = 1.0
    membership_start: float = 0.998
    membership_end: float = 0.97
    min_probability: float = 1e-7
    rank_top_fraction: float = 0.05
    rank_p: float = 3.0
    rank_scale_p: float = 0.5
    rank_baseline_permutations: int = 5
    rank_seed: int = 0
    soft_wpmi_weight: float = 1.0
    rank_reorder_weight: float = 1.0
    max_construct_words: int = 4
    concept_batch_size: int = 4096

    def __post_init__(self) -> None:
        if self.top_k < 1 or self.soft_top_k < 1:
            raise ValueError("top_k and soft_top_k must be positive")
        if self.temperature <= 0 or self.min_probability <= 0:
            raise ValueError("temperature and min_probability must be positive")
        if not (0 < self.membership_end <= self.membership_start < 1):
            raise ValueError("membership probabilities must satisfy 0 < end <= start < 1")
        if not (0 < self.rank_top_fraction <= 1):
            raise ValueError("rank_top_fraction must be in (0, 1]")
        if self.rank_p <= 0 or self.rank_scale_p < 0:
            raise ValueError("rank_p must be positive and rank_scale_p non-negative")
        if self.rank_baseline_permutations < 1:
            raise ValueError("rank_baseline_permutations must be positive")
        if self.soft_wpmi_weight < 0 or self.rank_reorder_weight < 0:
            raise ValueError("ranking weights cannot be negative")
        if self.soft_wpmi_weight + self.rank_reorder_weight <= 0:
            raise ValueError("at least one ranking weight must be positive")
        if self.max_construct_words < 1 or self.concept_batch_size < 1:
            raise ValueError("max_construct_words and concept_batch_size must be positive")


@dataclass(frozen=True)
class RankedConcept:
    name: str
    concept_index: int
    combined_score: float
    soft_wpmi: float
    rank_reorder: float
    soft_wpmi_rank: float
    rank_reorder_rank: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "concept_index": self.concept_index,
            "combined_score": self.combined_score,
            "soft_wpmi": self.soft_wpmi,
            "rank_reorder": self.rank_reorder,
            "soft_wpmi_rank": self.soft_wpmi_rank,
            "rank_reorder_rank": self.rank_reorder_rank,
        }


@dataclass(frozen=True)
class PoleInterpretation:
    pole: str
    construct_name: str
    top_concepts: tuple[RankedConcept, ...]
    soft_profile_indices: tuple[int, ...]
    rank_profile_indices: tuple[int, ...]
    rank_agreement_at_k: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "pole": self.pole,
            "construct_name": self.construct_name,
            "top_concepts": [concept.to_dict() for concept in self.top_concepts],
            "soft_profile_indices": list(self.soft_profile_indices),
            "rank_profile_indices": list(self.rank_profile_indices),
            "rank_agreement_at_k": self.rank_agreement_at_k,
        }


@dataclass(frozen=True)
class TextDissectResult:
    """One latent-fold prediction; ``construct_name`` is always the positive pole."""

    latent_id: str
    positive_pole: PoleInterpretation
    negative_pole: PoleInterpretation
    n_profiles: int
    n_concepts: int

    @property
    def construct_name(self) -> str:
        return self.positive_pole.construct_name

    @property
    def top_concepts(self) -> tuple[RankedConcept, ...]:
        return self.positive_pole.top_concepts

    def to_dict(self) -> dict[str, Any]:
        """Return the runner/cache-friendly JSON representation."""
        positive = self.positive_pole.to_dict()
        negative = self.negative_pole.to_dict()
        return {
            "latent_id": self.latent_id,
            "construct_name": self.construct_name,
            "top_concepts": positive["top_concepts"],
            "negative_construct_name": self.negative_pole.construct_name,
            "negative_top_concepts": negative["top_concepts"],
            "poles": {"positive": positive, "negative": negative},
            "native_diagnostics": {
                "positive_soft_wpmi": self.positive_pole.top_concepts[0].soft_wpmi,
                "positive_rank_reorder": self.positive_pole.top_concepts[0].rank_reorder,
                "negative_soft_wpmi": self.negative_pole.top_concepts[0].soft_wpmi,
                "negative_rank_reorder": self.negative_pole.top_concepts[0].rank_reorder,
                "positive_rank_agreement_at_k": self.positive_pole.rank_agreement_at_k,
                "negative_rank_agreement_at_k": self.negative_pole.rank_agreement_at_k,
                "n_profiles": self.n_profiles,
                "n_concepts": self.n_concepts,
            },
        }


def _metadata_value(value: Any) -> Any:
    array = np.asarray(value)
    if array.ndim == 0:
        return array.item()
    if array.size == 1:
        return array.reshape(()).item()
    return array.tolist()


def load_concept_bank(
    path: str | Path,
    *,
    expected_encoder: str | None = None,
    allow_adapted_encoder: bool = False,
) -> ConceptBank:
    """Load the project's ``emb``/``names`` NPZ schema with explicit errors.

    ``expected_encoder`` is optional because older banks do not carry metadata.
    When supplied, missing metadata is an error rather than silently mixing E5 and
    GTE/MiniLM spaces.  LoRA-tagged banks are rejected by default: the report calls
    for the frozen base E5 space, while the main Core pipeline currently also has
    an L3/LoRA dictionary whose ``encoder`` field still names its E5 base model.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"concept bank not found at {path}. Pass an existing frozen E5 bank, "
            "or build one with v6/tools/build_dictionary.py."
        )
    try:
        with np.load(path, allow_pickle=True) as payload:
            missing = {"emb", "names"}.difference(payload.files)
            if missing:
                raise ValueError(
                    f"concept bank {path} is missing required arrays: {sorted(missing)}; "
                    "expected keys 'emb' and 'names'"
                )
            embeddings = payload["emb"]
            names = tuple(str(name) for name in payload["names"])
            metadata = {
                key: _metadata_value(payload[key])
                for key in payload.files
                if key not in {"emb", "names"}
            }
    except (OSError, ValueError) as exc:
        if isinstance(exc, ValueError) and "missing required arrays" in str(exc):
            raise
        raise ValueError(f"could not read concept bank {path}: {exc}") from exc

    if expected_encoder is not None:
        actual = metadata.get("encoder")
        if actual is None:
            raise ValueError(
                f"concept bank {path} has no encoder metadata; cannot verify "
                f"required encoder {expected_encoder!r}"
            )
        if expected_encoder.casefold() not in str(actual).casefold():
            raise ValueError(
                f"concept bank encoder mismatch: required {expected_encoder!r}, "
                f"bank records {actual!r}"
            )
    adapted = metadata.get("lora_checkpoint_sha256")
    if adapted and not allow_adapted_encoder:
        raise ValueError(
            f"concept bank {path} is tagged with a LoRA checkpoint ({adapted}); "
            "The CLIP-Dissect E5 text adaptation requires a frozen base-E5 bank. Pass "
            "allow_adapted_encoder=True only for an explicitly labelled diagnostic."
        )
    return ConceptBank(embeddings, names, str(path.resolve()), metadata)


def load_project_concept_bank(
    path: str | Path | None = None,
    *,
    expected_encoder: str | None = None,
    allow_adapted_encoder: bool = False,
) -> ConceptBank:
    """Load ``path`` or the path selected by ``v6/encode.py``.

    Importing ``encode`` does not instantiate E5.  The large model is loaded only
    if text encoding is later requested without an injected encoder.
    """
    if path is None:
        try:
            import encode as project_encode
        except ImportError:  # pragma: no cover - package-style import outside current repo layout
            from . import encode as project_encode
        path = project_encode.DICT_PATH
    return load_concept_bank(
        path,
        expected_encoder=expected_encoder,
        allow_adapted_encoder=allow_adapted_encoder,
    )


def _default_encoder(texts: Sequence[str]) -> np.ndarray:
    try:
        import encode as project_encode
    except ImportError:  # pragma: no cover - package-style import outside current repo layout
        from . import encode as project_encode
    return project_encode.embed(list(texts))


def _encode_in_batches(
    texts: Sequence[str],
    encoder: TextEncoder,
    batch_size: int,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    dimension: int | None = None
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])
        encoded = np.asarray(encoder(batch))
        if encoded.ndim != 2 or encoded.shape[0] != len(batch):
            raise ValueError(
                "text encoder must return [len(texts), dimension]; "
                f"got {encoded.shape} for a batch of {len(batch)}"
            )
        if not np.issubdtype(encoded.dtype, np.number) or not np.all(np.isfinite(encoded)):
            raise ValueError("text encoder returned non-numeric or non-finite embeddings")
        if dimension is None:
            dimension = encoded.shape[1]
        elif encoded.shape[1] != dimension:
            raise ValueError("text encoder changed embedding dimension between batches")
        rows.append(encoded)
    return np.concatenate(rows, axis=0)


def build_concept_bank(
    concepts: Sequence[str],
    *,
    encoder: TextEncoder | None = None,
    batch_size: int = 1024,
    source: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ConceptBank:
    """Build a fixed bank with E5 or an injected, API-free text encoder."""
    names = tuple(str(concept).strip() for concept in concepts)
    if not names:
        raise ValueError("cannot build an empty concept bank")
    if any(not name for name in names):
        raise ValueError("concept names cannot be empty")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    embeddings = _encode_in_batches(names, encoder or _default_encoder, batch_size)
    return ConceptBank(embeddings, names, source, metadata or {})


def _coerce_bank(bank: ConceptBank | tuple[np.ndarray, Sequence[str]]) -> ConceptBank:
    if isinstance(bank, ConceptBank):
        return bank
    if isinstance(bank, tuple) and len(bank) == 2:
        embeddings, names = bank
        return ConceptBank(np.asarray(embeddings), tuple(names), source="in-memory tuple")
    raise TypeError(
        "concept_bank must be ConceptBank or the (embeddings, names) tuple returned "
        "by encode.load_dictionary()"
    )


def _activation_matrix(
    activations: np.ndarray | Sequence[float] | Mapping[str, Sequence[float]],
    n_profiles: int,
    latent_ids: Sequence[str] | None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    if isinstance(activations, Mapping):
        if latent_ids is not None:
            raise ValueError("latent_ids must be omitted when activations is a mapping")
        ids = tuple(str(key) for key in activations)
        if not ids:
            raise ValueError("activations mapping cannot be empty")
        matrix = np.column_stack([np.asarray(activations[key], dtype=float) for key in activations])
    else:
        matrix = np.asarray(activations, dtype=float)
        if matrix.ndim == 1:
            matrix = matrix[:, None]
        if matrix.ndim != 2:
            raise ValueError(
                f"activations must be [profile] or [profile, latent]; got {matrix.shape}"
            )
        ids = (
            tuple(str(value) for value in latent_ids)
            if latent_ids is not None
            else tuple(f"latent_{i}" for i in range(matrix.shape[1]))
        )
    if matrix.shape[0] != n_profiles:
        raise ValueError(
            f"activation/profile mismatch: {matrix.shape[0]} rows for {n_profiles} profiles"
        )
    if matrix.shape[1] != len(ids):
        raise ValueError(
            f"activation/latent-id mismatch: {matrix.shape[1]} columns for {len(ids)} IDs"
        )
    if len(set(ids)) != len(ids):
        raise ValueError("latent_ids must be unique opaque identifiers")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("activations contain NaN or infinite values")
    spread = np.ptp(matrix, axis=0)
    if np.any(spread <= 1e-12):
        bad = [ids[i] for i in np.flatnonzero(spread <= 1e-12)]
        raise ValueError(f"constant activations cannot be interpreted: {bad}")
    return matrix, ids


def _normalise_rows(matrix: np.ndarray, what: str) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{what} must be a finite 2-D array")
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms <= 1e-12):
        raise ValueError(f"{what} contains a zero-norm row")
    return matrix / norms[:, None]


def _top_indices(values: np.ndarray, k: int) -> np.ndarray:
    # Stable sorting fixes all activation-tie behavior by original profile index.
    return np.argsort(-values, kind="stable")[:k]


def _rank_quality(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return tie-neutral Borda quality and one-based descending midranks.

    Stable ordinal ranks leak concept-bank row order whenever scores tie.  This
    is especially damaging for discrete questionnaire activations: an all-tied
    rank-reorder slice gives every concept the same native score, yet ordinal
    ranks would still create a complete 1..N preference ordering.  Exact ties
    therefore receive their conventional average (mid-)rank and consequently
    the same Borda contribution.  The same helper is deliberately used for
    SoftWPMI so duplicate/equal concept scores cannot reintroduce that leak.
    """

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("rank scores must be a non-empty finite one-dimensional array")
    order = np.argsort(-values, kind="stable")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        # Positions are one-based; ``end`` is exclusive and numerically equal
        # to the final one-based position in this tie group.
        ranks[order[start:end]] = ((start + 1) + end) / 2.0
        start = end
    if len(values) == 1:
        quality = np.ones(1, dtype=float)
    else:
        quality = 1.0 - (ranks - 1.0) / (len(values) - 1.0)
    return quality, ranks


def _canonical_score_order(scores: np.ndarray, names: Sequence[str]) -> np.ndarray:
    """Order by score, then lexical concept identity—not concept-bank row."""

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or len(values) != len(names):
        raise ValueError("score/name rows must align")
    folded = np.asarray([str(name).casefold() for name in names])
    original = np.asarray([str(name) for name in names])
    return np.lexsort((original, folded, -values))


def _top_tie_names(
    scores: np.ndarray, names: Sequence[str], k: int
) -> set[str]:
    """Return the top-k identity set, including the complete kth-score tie."""

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or len(values) != len(names) or k < 1:
        raise ValueError("top-tie score/name rows must align and k must be positive")
    if k >= len(values):
        selected = np.arange(len(values))
    else:
        threshold = np.partition(values, len(values) - k)[len(values) - k]
        selected = np.flatnonzero(values >= threshold)
    return {str(names[index]).casefold() for index in selected}


def _logsumexp_mean(values: np.ndarray, axis: int) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    result = maximum + np.log(np.mean(np.exp(values - maximum), axis=axis, keepdims=True))
    return np.squeeze(result, axis=axis)


def _rank_baseline(values_descending: np.ndarray, config: TextDissectConfig) -> float:
    ascending = values_descending[::-1]
    # The same fixed position permutations are used for every pole, so native
    # scores do not depend on whether latents are passed singly or as a batch.
    rng = np.random.default_rng(config.rank_seed)
    shuffled = np.column_stack(
        [rng.permutation(ascending) for _ in range(config.rank_baseline_permutations)]
    )
    baseline = float(np.mean(np.abs(ascending[:, None] - shuffled) ** config.rank_p))
    # A tied top slice has no rank information.  Returning epsilon makes all
    # equal-order concepts score zero or worse without producing NaN/inf.
    return max(baseline, config.min_probability)


def _eligible_concepts(names: Sequence[str], max_words: int) -> np.ndarray:
    eligible = np.array([1 <= len(name.split()) <= max_words for name in names], dtype=bool)
    if not np.any(eligible):
        raise ValueError(f"concept bank has no names with 1-{max_words} whitespace-delimited words")
    return np.flatnonzero(eligible)


def _make_pole_result(
    pole: str,
    pole_id: int,
    names: Sequence[str],
    candidate_indices: np.ndarray,
    soft_scores: np.ndarray,
    rank_scores: np.ndarray,
    soft_indices: Sequence[np.ndarray],
    rank_indices: Sequence[np.ndarray],
    config: TextDissectConfig,
) -> PoleInterpretation:
    soft = soft_scores[pole_id, candidate_indices]
    reorder = rank_scores[pole_id, candidate_indices]
    soft_quality, soft_rank = _rank_quality(soft)
    reorder_quality, reorder_rank = _rank_quality(reorder)
    weight_sum = config.soft_wpmi_weight + config.rank_reorder_weight
    combined = (
        config.soft_wpmi_weight * soft_quality
        + config.rank_reorder_weight * reorder_quality
    ) / weight_sum
    candidate_names = [names[int(index)] for index in candidate_indices]
    # Lexical identity is only a deterministic presentation tie-break.  Unlike
    # bank row/index it is invariant to a permutation of the same concept bank.
    positions = _canonical_score_order(combined, candidate_names)
    ranked: list[RankedConcept] = []
    seen_names: set[str] = set()
    for position in positions:
        source_index = int(candidate_indices[position])
        name = names[source_index]
        if name.casefold() in seen_names:
            continue
        seen_names.add(name.casefold())
        ranked.append(
            RankedConcept(
                name=name,
                concept_index=source_index,
                combined_score=float(combined[position]),
                soft_wpmi=float(soft[position]),
                rank_reorder=float(reorder[position]),
                soft_wpmi_rank=float(soft_rank[position]),
                rank_reorder_rank=float(reorder_rank[position]),
            )
        )
        if len(ranked) >= config.top_k:
            break
    if not ranked:  # Defensive: eligibility guarantees at least one candidate.
        raise RuntimeError("concept ranking unexpectedly produced no construct name")

    compare_k = min(config.top_k, len(candidate_indices))
    # Include the entire kth-score tie rather than selecting arbitrary bank
    # rows at the threshold.  Compare semantic identities, not local positions.
    top_soft = _top_tie_names(soft, candidate_names, compare_k)
    top_rank = _top_tie_names(reorder, candidate_names, compare_k)
    agreement = len(top_soft & top_rank) / max(len(top_soft | top_rank), 1)
    return PoleInterpretation(
        pole=pole,
        construct_name=ranked[0].name,
        top_concepts=tuple(ranked),
        soft_profile_indices=tuple(int(i) for i in soft_indices[pole_id]),
        rank_profile_indices=tuple(int(i) for i in rank_indices[pole_id]),
        rank_agreement_at_k=float(agreement),
    )


def text_dissect(
    profile_texts: Sequence[str] | None,
    activations: np.ndarray | Sequence[float] | Mapping[str, Sequence[float]],
    concept_bank: ConceptBank | tuple[np.ndarray, Sequence[str]],
    *,
    encoder: TextEncoder | None = None,
    profile_embeddings: np.ndarray | None = None,
    latent_ids: Sequence[str] | None = None,
    config: TextDissectConfig | None = None,
) -> list[TextDissectResult]:
    """Interpret one or more latent activation columns without gold text.

    Args:
        profile_texts: Fold-visible respondent/profile texts.  May be ``None``
            only when ``profile_embeddings`` is supplied.
        activations: ``[n_profiles]``, ``[n_profiles, n_latents]``, or an
            insertion-ordered mapping from opaque latent ID to activation vector.
        concept_bank: :class:`ConceptBank`, or the ``(C, words)`` tuple returned
            by the existing ``encode.load_dictionary`` helper.
        encoder: Injectable local text encoder.  Defaults lazily to ``encode.embed``.
        profile_embeddings: Optional already-encoded profiles in the same space.
        latent_ids: Opaque IDs for array activations; never semantic/gold text.
        config: Fixed scoring configuration.

    Returns:
        One :class:`TextDissectResult` per latent column.  ``construct_name`` is
        the top positive-pole concept; the negative pole and all native scores
        remain available for audit and fold-level stability computation.
    """
    config = config or TextDissectConfig()
    bank = _coerce_bank(concept_bank)

    if profile_embeddings is None:
        if profile_texts is None:
            raise ValueError("profile_texts are required when profile_embeddings are not supplied")
        clean_texts = tuple(str(text).strip() for text in profile_texts)
        if not clean_texts or any(not text for text in clean_texts):
            raise ValueError("profile_texts must contain at least two non-empty fold-visible texts")
        profile_embeddings = _encode_in_batches(
            clean_texts, encoder or _default_encoder, batch_size=1024
        )
    else:
        profile_embeddings = np.asarray(profile_embeddings)
        if profile_texts is not None and len(profile_texts) != profile_embeddings.shape[0]:
            raise ValueError("profile_texts and profile_embeddings have different row counts")

    if profile_embeddings.ndim != 2 or profile_embeddings.shape[0] < 2:
        raise ValueError("at least two profile embeddings are required")
    profiles = _normalise_rows(profile_embeddings, "profile embeddings")
    if profiles.shape[1] != bank.embeddings.shape[1]:
        raise ValueError(
            "profile/concept embedding dimension mismatch: "
            f"{profiles.shape[1]} versus {bank.embeddings.shape[1]}; use the same frozen encoder"
        )
    activation_matrix, ids = _activation_matrix(activations, len(profiles), latent_ids)

    # Interleave positive and negative columns so each result maps to 2*j / 2*j+1.
    pole_activations = np.column_stack(
        [pole for column in activation_matrix.T for pole in (column, -column)]
    )
    n_poles = pole_activations.shape[1]
    soft_count = min(config.soft_top_k, len(profiles))
    rank_count = min(len(profiles), max(2, int(len(profiles) * config.rank_top_fraction)))
    soft_indices = [_top_indices(pole_activations[:, i], soft_count) for i in range(n_poles)]
    rank_indices = [_top_indices(pole_activations[:, i], rank_count) for i in range(n_poles)]
    memberships = config.membership_start - (
        np.arange(soft_count, dtype=float) / soft_count
    ) * (config.membership_start - config.membership_end)
    rank_targets = [pole_activations[idx, i] for i, idx in enumerate(rank_indices)]
    rank_baselines = [
        _rank_baseline(rank_targets[i], config) for i in range(n_poles)
    ]

    concept_embeddings = np.asarray(bank.embeddings)
    concept_norms = np.linalg.norm(concept_embeddings, axis=1)
    if not np.all(np.isfinite(concept_norms)) or np.any(concept_norms <= 1e-12):
        bad = int(np.flatnonzero(~np.isfinite(concept_norms) | (concept_norms <= 1e-12))[0])
        raise ValueError(f"concept bank contains a non-finite or zero-norm embedding at row {bad}")

    # Streaming log-softmax avoids materializing [profiles, 542k concepts].
    row_max = np.full(len(profiles), -np.inf, dtype=np.float64)
    for start in range(0, len(bank.names), config.concept_batch_size):
        end = min(start + config.concept_batch_size, len(bank.names))
        # Always copy: in-place normalisation must never mutate an injected
        # float64 concept bank (np.asarray would otherwise return a view).
        concepts = np.array(concept_embeddings[start:end], dtype=np.float64, copy=True)
        concepts /= concept_norms[start:end, None]
        logits = config.temperature * (profiles @ concepts.T)
        row_max = np.maximum(row_max, np.max(logits, axis=1))
    row_sum = np.zeros(len(profiles), dtype=np.float64)
    for start in range(0, len(bank.names), config.concept_batch_size):
        end = min(start + config.concept_batch_size, len(bank.names))
        concepts = np.array(concept_embeddings[start:end], dtype=np.float64, copy=True)
        concepts /= concept_norms[start:end, None]
        logits = config.temperature * (profiles @ concepts.T)
        row_sum += np.sum(np.exp(logits - row_max[:, None]), axis=1)
    log_normalizer = row_max + np.log(row_sum)

    soft_scores = np.empty((n_poles, len(bank.names)), dtype=np.float64)
    rank_scores = np.empty_like(soft_scores)
    for start in range(0, len(bank.names), config.concept_batch_size):
        end = min(start + config.concept_batch_size, len(bank.names))
        concepts = np.array(concept_embeddings[start:end], dtype=np.float64, copy=True)
        concepts /= concept_norms[start:end, None]
        similarities = profiles @ concepts.T
        probabilities = np.exp(config.temperature * similarities - log_normalizer[:, None])

        conditional = np.empty((n_poles, end - start), dtype=np.float64)
        for pole_id in range(n_poles):
            selected = probabilities[soft_indices[pole_id]]
            terms = 1.0 + memberships[:, None] * (selected - 1.0)
            conditional[pole_id] = np.sum(
                np.log(terms + config.min_probability), axis=0
            )
        background = _logsumexp_mean(conditional, axis=0)
        soft_scores[:, start:end] = conditional - config.background_lambda * background[None]

        for pole_id in range(n_poles):
            selected = similarities[rank_indices[pole_id]]
            order = np.argsort(selected, axis=0, kind="stable")
            ranks = np.argsort(order, axis=0, kind="stable")
            ascending_target = rank_targets[pole_id][::-1]
            reordered = np.take_along_axis(
                np.broadcast_to(ascending_target[:, None], selected.shape), ranks, axis=0
            )
            error = np.mean(
                np.abs(rank_targets[pole_id][:, None] - reordered) ** config.rank_p,
                axis=0,
            ) / rank_baselines[pole_id]
            if config.rank_scale_p:
                scale = np.maximum(
                    np.mean(selected, axis=0), config.min_probability
                ) ** config.rank_scale_p
                error = error / scale
            rank_scores[pole_id, start:end] = -error

    candidate_indices = _eligible_concepts(bank.names, config.max_construct_words)
    results: list[TextDissectResult] = []
    for latent_index, latent_id in enumerate(ids):
        positive = _make_pole_result(
            "positive", 2 * latent_index, bank.names, candidate_indices,
            soft_scores, rank_scores, soft_indices, rank_indices, config,
        )
        negative = _make_pole_result(
            "negative", 2 * latent_index + 1, bank.names, candidate_indices,
            soft_scores, rank_scores, soft_indices, rank_indices, config,
        )
        results.append(
            TextDissectResult(
                latent_id=latent_id,
                positive_pole=positive,
                negative_pole=negative,
                n_profiles=len(profiles),
                n_concepts=len(bank.names),
            )
        )
    return results


# Backwards-compatible and canonical runner-facing aliases. The legacy name is
# retained in artifact keys; the canonical name matches the report.
run_text_dissect = text_dissect
run_clip_dissect_e5 = text_dissect
ClipDissectE5Config = TextDissectConfig
ClipDissectE5Result = TextDissectResult


def rank_stability(
    results: Sequence[TextDissectResult | Mapping[str, Any]],
    *,
    pole: str = "positive",
    k: int = 6,
) -> float:
    """Mean pairwise top-k Jaccard stability across frozen fold results."""
    if pole not in {"positive", "negative"}:
        raise ValueError("pole must be 'positive' or 'negative'")
    if k < 1 or not results:
        raise ValueError("rank_stability requires results and a positive k")
    rankings: list[set[str]] = []
    for result in results:
        if isinstance(result, TextDissectResult):
            selected = result.positive_pole if pole == "positive" else result.negative_pole
            names = [entry.name for entry in selected.top_concepts[:k]]
        else:
            try:
                entries = result["poles"][pole]["top_concepts"]
                names = [str(entry["name"]) for entry in entries[:k]]
            except (KeyError, TypeError) as exc:
                raise ValueError("mapping result does not match TextDissectResult.to_dict()") from exc
        rankings.append(set(names))
    if len(rankings) == 1:
        return 1.0
    values = []
    for i, left in enumerate(rankings):
        for right in rankings[i + 1 :]:
            values.append(len(left & right) / max(len(left | right), 1))
    return float(np.mean(values))


__all__ = [
    "SCORER_VERSION",
    "ClipDissectE5Config",
    "ClipDissectE5Result",
    "ConceptBank",
    "PoleInterpretation",
    "RankedConcept",
    "TextDissectConfig",
    "TextDissectResult",
    "build_concept_bank",
    "load_concept_bank",
    "load_project_concept_bank",
    "rank_stability",
    "run_clip_dissect_e5",
    "run_text_dissect",
    "text_dissect",
]
