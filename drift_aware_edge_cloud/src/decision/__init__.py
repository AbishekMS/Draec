"""Drift-Aware Adaptive Edge-Cloud-Hybrid Orchestration for Non-Stationary IoT.

Module   : src/decision/__init__.py
Phase    : Phase 5 & 6
Status   : IMPLEMENTED

Public API for the DRAEC Decision Engine and Hardened Execution Layer.
"""

from __future__ import annotations

from src.decision.base import (
    BaseController,
    BaseDecisionEngine,
    DecisionAction,
    DecisionInputs,
    DecisionResult,
    ExecutionResult,
    ExecutionStatus,
)
from src.decision.engine import (
    AdaptiveController,
    DecisionEngine,
    DecisionInstrumentation,
    StaticBaselineController,
    validate_input,
    validate_output,
)

__all__ = [
    "DecisionAction",
    "ExecutionStatus",
    "DecisionInputs",
    "DecisionResult",
    "ExecutionResult",
    "BaseController",
    "BaseDecisionEngine",
    "AdaptiveController",
    "StaticBaselineController",
    "DecisionInstrumentation",
    "DecisionEngine",
    "validate_input",
    "validate_output",
]

