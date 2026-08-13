from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "task3_robotics" / "task3_pipeline_v1" / "gen_operator.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("robot_core_gen_operator_test", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scalar_negative_mode_flips_physical_edge_without_negop(monkeypatch, tmp_path):
    module = _load_module()
    monkeypatch.setenv("GENOP_NEGATIVE_MODE", "scalar")
    checkpoint = tmp_path / "operator.pt"
    operator = module.load_or_init(d=3, device="cpu", path=checkpoint)
    parent = torch.tensor([[1.0, -2.0, 0.5]])
    condition = torch.tensor([[-1.0, 0.25, 0.0]])

    with torch.no_grad():
        output = operator(parent, condition)

    assert operator.negative_mode == "scalar"
    assert torch.allclose(output, -0.25 * parent)

    module.save(operator, checkpoint)
    restored = module.load_or_init(d=3, device="cpu", path=checkpoint)
    assert restored.negative_mode == "scalar"


def test_checkpoint_rejects_negative_mode_mismatch(monkeypatch, tmp_path):
    module = _load_module()
    checkpoint = tmp_path / "operator.pt"
    monkeypatch.setenv("GENOP_NEGATIVE_MODE", "scalar")
    module.save(module.load_or_init(d=2, device="cpu", path=checkpoint), checkpoint)

    monkeypatch.setenv("GENOP_NEGATIVE_MODE", "semantic")
    with pytest.raises(ValueError, match="negative mode"):
        module.load_or_init(d=2, device="cpu", path=checkpoint)
