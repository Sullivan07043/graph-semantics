from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "task3_v1" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from paper_aligned_core import (  # noqa: E402
    aggregate_swap_rows,
    assign_group_splits,
    coordinate_swap,
    select_workspace_band,
    token_rank,
    wilson_interval,
)


def test_pairwise_coordinates_are_exchanged() -> None:
    generator = torch.Generator().manual_seed(7)
    hidden = torch.randn(2, 5, 8, generator=generator)
    source = torch.randn(8, generator=generator)
    target = torch.randn(8, generator=generator)
    matrix = torch.stack((source, target), dim=1)
    inverse = torch.linalg.pinv(matrix)
    before = hidden.reshape(-1, 8) @ inverse.T

    patched, diagnostics = coordinate_swap(
        hidden, source, target, rcond=1.0e-7
    )
    after = patched.reshape(-1, 8) @ inverse.T

    torch.testing.assert_close(after, before.flip(-1), atol=2.0e-5, rtol=2.0e-5)
    assert diagnostics["coordinate_swap_max_abs_error"] < 2.0e-5
    assert diagnostics["matrix_rank"] == 2


def test_pairwise_orthogonal_component_is_preserved() -> None:
    generator = torch.Generator().manual_seed(11)
    hidden = torch.randn(4, 9, generator=generator, dtype=torch.float64)
    source = torch.randn(9, generator=generator, dtype=torch.float64)
    target = torch.randn(9, generator=generator, dtype=torch.float64)
    matrix = torch.stack((source, target), dim=1)
    projector = matrix @ torch.linalg.pinv(matrix)

    patched, diagnostics = coordinate_swap(hidden, source, target)
    orthogonal_before = hidden @ (torch.eye(9, dtype=torch.float64) - projector)
    orthogonal_after = patched @ (torch.eye(9, dtype=torch.float64) - projector)

    torch.testing.assert_close(
        orthogonal_after, orthogonal_before, atol=1.0e-10, rtol=1.0e-10
    )
    assert diagnostics["orthogonal_preservation_max_abs_error"] < 1.0e-10


def test_identity_swap_preserves_shape_dtype_and_device() -> None:
    for dtype in (torch.float32, torch.float64):
        hidden = torch.randn(2, 3, 6, dtype=dtype)
        vector = torch.randn(6, dtype=dtype)
        patched, diagnostics = coordinate_swap(
            hidden, vector, vector, rcond=1.0e-6
        )

        torch.testing.assert_close(patched, hidden)
        assert patched.shape == hidden.shape
        assert patched.dtype == dtype
        assert patched.device == hidden.device
        assert diagnostics["matrix_rank"] == 1


def test_zero_strength_is_noop() -> None:
    hidden = torch.randn(3, 7)
    source = torch.randn(7)
    target = torch.randn(7)
    patched, _ = coordinate_swap(hidden, source, target, strength=0.0)
    torch.testing.assert_close(patched, hidden)


def test_group_split_has_no_leakage_and_is_deterministic() -> None:
    rows = [
        {"category": "a"},
        {"category": "a"},
        {"category": "b"},
        {"category": "c"},
        {"category": "c"},
    ]
    first = assign_group_splits(
        rows, group_key="category", calibration_fraction=0.4, seed=42
    )
    second = assign_group_splits(
        rows, group_key="category", calibration_fraction=0.4, seed=42
    )
    assert first == second
    assert set(first.values()) == {"calibration", "heldout"}
    assert len(first) == 3


def test_band_selection_uses_highest_calibration_mrr() -> None:
    rows = []
    for example in ("a", "b"):
        for layer, rank in enumerate((50, 10, 1, 1, 20)):
            rows.append(
                {"example_id": example, "layer": layer, "jlens_rank": rank}
            )
    selected = select_workspace_band(rows, list(range(5)), width=2)
    assert selected["selected"]["layers"] == [2, 3]


def test_rank_and_wilson_helpers() -> None:
    logits = torch.tensor([0.0, 4.0, 2.0, 3.0])
    assert token_rank(logits, 1) == 1
    assert token_rank(logits, 2) == 3
    interval = wilson_interval(5, 10)
    assert interval["rate"] == 0.5
    assert 0.2 < interval["ci95_low"] < 0.3
    assert 0.7 < interval["ci95_high"] < 0.8


def test_swap_aggregate_keeps_top1_and_log_probability_metrics_separate() -> None:
    rows = [
        {
            "target_answer_top1_success": True,
            "original_answer_top1_retained": False,
            "delta_swap_target_log_probability": 1.5,
            "delta_log_probability_margin": 2.0,
        },
        {
            "target_answer_top1_success": False,
            "original_answer_top1_retained": True,
            "delta_swap_target_log_probability": -0.5,
            "delta_log_probability_margin": -1.0,
        },
    ]
    aggregate = aggregate_swap_rows(rows)
    assert aggregate["target_answer_top1_swap_success"]["rate"] == 0.5
    assert aggregate["original_answer_top1_retention"]["rate"] == 0.5
    assert aggregate["mean_delta_swap_target_log_probability"] == 0.5
    assert aggregate["mean_delta_log_probability_margin"] == 0.5


def test_official_probe_swap_copy_and_manifest_are_auditable() -> None:
    official = ROOT / "task3_v1" / "data" / "prompts" / "official_anthropic"
    prompt_path = official / "probe-swap.json"
    digest = hashlib.sha256(prompt_path.read_bytes()).hexdigest()
    assert digest == "a0edd27ca23f7b4d0fbe90448c2ddcc7457a3d812121bf024ed12a032ff86796"
    assert len(json.loads(prompt_path.read_text(encoding="utf-8"))["items"]) == 90
    manifest = json.loads(
        (official / "SOURCE_MANIFEST.json").read_text(encoding="utf-8")
    )
    assert manifest["commit_sha"] == "581d398613e5602a5af361e1c34d3a92ea82ba8e"
    assert manifest["license"] == "Apache-2.0"
