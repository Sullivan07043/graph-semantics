"""Canonical implementations and metadata for the external baselines.

The package intentionally avoids importing implementation modules at import
time. In particular, importing :mod:`v6.baselines` must not eagerly import
PyTorch, the E5 encoder, or any API client. CLI entry points live only in
``v6.baselines.runners`` so baseline orchestration stays out of the ``v6`` root.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BaselineSpec:
    """Stable code-location and execution metadata for one report baseline."""

    slug: str
    display_name: str
    implementation: str
    task1_runner: str
    task2_runner: str
    method_key: str | None
    requires_openai: bool


BASELINE_SPECS = (
    BaselineSpec(
        slug="feature-propagation",
        display_name="Feature Propagation",
        implementation="v6.baselines.feature_propagation",
        task1_runner="v6.baselines.runners.feature_propagation_task1",
        task2_runner="v6.baselines.runners.feature_propagation_task2",
        method_key=None,
        requires_openai=False,
    ),
    BaselineSpec(
        slug="graphmae-gcn",
        display_name="GraphMAE-GCN",
        implementation="v6.baselines.graphmae_gcn",
        task1_runner="v6.baselines.runners.graphmae_task1",
        task2_runner="v6.baselines.runners.graphmae_task2",
        method_key=None,
        requires_openai=False,
    ),
    BaselineSpec(
        slug="clip-dissect-e5",
        display_name="CLIP-Dissect (E5 text adaptation)",
        implementation="v6.baselines.clip_dissect_e5",
        task1_runner="v6.baselines.runners.clip_dissect_task1",
        task2_runner="v6.baselines.runners.interpretability_task2",
        method_key="text-dissect",
        requires_openai=False,
    ),
    BaselineSpec(
        slug="automated-interpretability",
        display_name="Automated Interpretability",
        implementation="v6.baselines.automated_interpretability",
        task1_runner="v6.baselines.runners.llm_interpretability_task1",
        task2_runner="v6.baselines.runners.interpretability_task2",
        method_key="autointerp",
        requires_openai=True,
    ),
    BaselineSpec(
        slug="delphi",
        display_name="Delphi",
        implementation="v6.baselines.delphi",
        task1_runner="v6.baselines.runners.llm_interpretability_task1",
        task2_runner="v6.baselines.runners.interpretability_task2",
        method_key="delphi",
        requires_openai=True,
    ),
)

BASELINES_BY_SLUG = {spec.slug: spec for spec in BASELINE_SPECS}

if len(BASELINES_BY_SLUG) != len(BASELINE_SPECS):
    raise RuntimeError("baseline slugs must be unique")


__all__ = ["BASELINE_SPECS", "BASELINES_BY_SLUG", "BaselineSpec"]
