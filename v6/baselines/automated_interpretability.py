"""Public Automated Interpretability interface.

The shared LLM validation/sampling implementation is private so Automated
Interpretability and Delphi can expose independent, method-specific imports.
"""

from ._llm_interpretability import (
    AUTOINTERP_EXPLAIN_PROMPT_VERSION,
    AUTOINTERP_SIMULATE_PROMPT_VERSION,
    ROBOT_AUTOINTERP_EXPLAIN_PROMPT_VERSION,
    ROBOT_AUTOINTERP_SIMULATE_PROMPT_VERSION,
    BaselineOutputError,
    run_autointerp,
)

__all__ = [
    "AUTOINTERP_EXPLAIN_PROMPT_VERSION",
    "AUTOINTERP_SIMULATE_PROMPT_VERSION",
    "ROBOT_AUTOINTERP_EXPLAIN_PROMPT_VERSION",
    "ROBOT_AUTOINTERP_SIMULATE_PROMPT_VERSION",
    "BaselineOutputError",
    "run_autointerp",
]
