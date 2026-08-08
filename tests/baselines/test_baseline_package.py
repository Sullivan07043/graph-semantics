"""Structural checks for the canonical five-baseline package."""

from importlib import import_module, util

from v6.baselines import BASELINE_SPECS, BASELINES_BY_SLUG


EXPECTED_SLUGS = {
    "feature-propagation",
    "graphmae-gcn",
    "clip-dissect-e5",
    "automated-interpretability",
    "delphi",
}


def test_catalog_contains_exactly_the_five_report_baselines():
    assert {spec.slug for spec in BASELINE_SPECS} == EXPECTED_SLUGS
    assert set(BASELINES_BY_SLUG) == EXPECTED_SLUGS
    assert len(BASELINE_SPECS) == 5


def test_catalog_implementations_and_runners_are_importable():
    for spec in BASELINE_SPECS:
        assert import_module(spec.implementation) is not None
        assert util.find_spec(spec.task1_runner) is not None
        assert util.find_spec(spec.task2_runner) is not None


def test_only_llm_baselines_require_openai():
    requiring_api = {
        spec.slug for spec in BASELINE_SPECS if spec.requires_openai
    }
    assert requiring_api == {"automated-interpretability", "delphi"}


def test_clip_dissect_exposes_canonical_runner_names():
    from v6.baselines import clip_dissect_e5

    assert clip_dissect_e5.run_clip_dissect_e5 is clip_dissect_e5.text_dissect
    assert clip_dissect_e5.ClipDissectE5Config is clip_dissect_e5.TextDissectConfig


def test_llm_public_interfaces_are_method_specific():
    from v6.baselines import automated_interpretability, delphi

    assert callable(automated_interpretability.run_autointerp)
    assert not hasattr(automated_interpretability, "run_delphi")
    assert callable(delphi.run_delphi)
    assert not hasattr(delphi, "run_autointerp")
