"""Public Delphi interface for the project's detection-scored adaptation."""

from ._llm_interpretability import (
    DELPHI_DETECT_PROMPT_VERSION,
    DELPHI_EXPLAIN_PROMPT_VERSION,
    ROBOT_DELPHI_DETECT_PROMPT_VERSION,
    ROBOT_DELPHI_EXPLAIN_PROMPT_VERSION,
    BaselineOutputError,
    run_delphi,
)

__all__ = [
    "DELPHI_DETECT_PROMPT_VERSION",
    "DELPHI_EXPLAIN_PROMPT_VERSION",
    "ROBOT_DELPHI_DETECT_PROMPT_VERSION",
    "ROBOT_DELPHI_EXPLAIN_PROMPT_VERSION",
    "BaselineOutputError",
    "run_delphi",
]
