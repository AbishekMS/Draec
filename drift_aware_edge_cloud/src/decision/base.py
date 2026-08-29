"""Drift-Aware Adaptive Edge-Cloud-Hybrid Orchestration for Non-Stationary IoT.

Module   : src/decision/base.py
Phase    : Phase 5 & 6
Status   : IMPLEMENTED

Base interface and data contracts for the DRAEC Decision Engine and Execution Layer.
Defines:
- DecisionAction: action space {EDGE, CLOUD, HYBRID}
- ExecutionStatus: execution status {SUCCESS, FALLBACK, FAILED}
- DecisionInputs: causal runtime inputs (R_t, C_t, D_t, Q_t, index/time)
- DecisionResult: structured output of the routing decision
- ExecutionResult: structured outcome of hardened model execution
- BaseController: abstract controller interface
- BaseDecisionEngine: abstract decision engine interface
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class DecisionAction(str, Enum):
    """The three-way action space a_t in {EDGE, CLOUD, HYBRID} for DRAEC inference.

    EDGE:
        Execute Edge Hoeffding Tree and return its prediction.
    CLOUD:
        Execute Cloud XGBoost model and return its prediction.
    HYBRID:
        Execute Edge-first inference. If current Edge confidence is insufficient,
        invoke Cloud fallback; Cloud provides the final result.
    """

    EDGE = "EDGE"
    CLOUD = "CLOUD"
    HYBRID = "HYBRID"

    @classmethod
    def from_str(cls, val: str | DecisionAction) -> DecisionAction:
        """Parse action case-insensitively from string or enum."""
        if isinstance(val, cls):
            return val
        s = str(val).strip().upper()
        for member in cls:
            if member.value == s:
                return member
        raise ValueError(f"Unknown action {val!r}; expected one of {[m.value for m in cls]}")


@dataclass(frozen=True)
class DecisionInputs:
    """Causal runtime inputs provided to the decision engine at observation t.

    Attributes
    ----------
    reliability : float
        Current DRAEC reliability score R_t in [0, 1].
    confidence : float | None
        Current model prediction confidence C_t in [0, 1], if available.
    drift_severity : float | None
        Current smoothed drift severity D_t in [0, 1], if available.
    quality : float | None
        Current data quality score Q_t in [0, 1], if available.
    observation_index : int | None
        Streaming sequence index t, if available.
    timestamp : Any | None
        Observation timestamp, if available.
    """

    reliability: float
    confidence: float | None = None
    drift_severity: float | None = None
    quality: float | None = None
    observation_index: int | None = None
    timestamp: Any | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.reliability, (int, float)):
            raise TypeError(f"reliability must be numeric, got {type(self.reliability).__name__}")
        if not (0.0 <= float(self.reliability) <= 1.0):
            raise ValueError(f"reliability R_t must be in [0, 1], got {self.reliability}")
        for name, val in (
            ("confidence", self.confidence),
            ("drift_severity", self.drift_severity),
            ("quality", self.quality),
        ):
            if val is not None:
                if not isinstance(val, (int, float)):
                    raise TypeError(f"{name} must be numeric, got {type(val).__name__}")
                if not (0.0 <= float(val) <= 1.0):
                    raise ValueError(f"{name} must be in [0, 1], got {val}")


@dataclass(frozen=True)
class DecisionResult:
    """Structured decision output produced by an adaptive or baseline controller.

    Attributes
    ----------
    selected_action : DecisionAction
        The selected action a_t in {EDGE, CLOUD, HYBRID}.
    reliability : float
        The reliability score R_t evaluated for this decision.
    previous_action : DecisionAction | None
        The action active prior to this decision step.
    decision_reason : str
        Human-readable explanation of why this action was selected.
    decision_inputs : DecisionInputs | None
        Snapshot of causal inputs available at decision time.
    switch_count : int
        Cumulative count of action switches up to and including this step.
    hybrid_fallback : bool
        Whether this decision was or led to a Hybrid Cloud fallback.
    observation_index : int | None
        Streaming observation index t, if available.
    timestamp : Any | None
        Observation timestamp, if available.
    """

    selected_action: DecisionAction
    reliability: float
    previous_action: DecisionAction | None
    decision_reason: str
    decision_inputs: DecisionInputs | None = None
    switch_count: int = 0
    hybrid_fallback: bool = False
    observation_index: int | None = None
    timestamp: Any | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.selected_action, DecisionAction):
            raise TypeError(
                f"selected_action must be DecisionAction, got {type(self.selected_action).__name__}"
            )
        if not (0.0 <= float(self.reliability) <= 1.0):
            raise ValueError(f"reliability must be in [0, 1], got {self.reliability}")


class ExecutionStatus(str, Enum):
    """The execution status of an inference step in DRAEC.

    SUCCESS:
        Inference executed successfully on the selected primary path (Edge or Cloud)
        or Hybrid without fallback.
    FALLBACK:
        Hybrid path executed Edge first, found confidence insufficient, and successfully
        fell back to Cloud; Cloud provides the final prediction.
    FAILED:
        Inference failed due to validation, model, runtime, or fallback errors.
    """

    SUCCESS = "SUCCESS"
    FALLBACK = "FALLBACK"
    FAILED = "FAILED"

    @classmethod
    def from_str(cls, val: str | ExecutionStatus) -> ExecutionStatus:
        """Parse status case-insensitively from string or enum."""
        if isinstance(val, cls):
            return val
        s = str(val).strip().upper()
        for member in cls:
            if member.value == s:
                return member
        raise ValueError(f"Unknown status {val!r}; expected one of {[m.value for m in cls]}")


@dataclass(frozen=True)
class ExecutionResult:
    """Structured outcome of model execution under the chosen action.

    Attributes
    ----------
    decision : DecisionResult
        The underlying decision that guided this execution.
    action : DecisionAction
        The action executed ({EDGE, CLOUD, HYBRID}).
    prediction : int | None
        The final predicted class label in {0, 1} on success, or None on failure.
    probabilities : dict[int, float] | None
        The final predicted class probability distribution {0: p0, 1: p1} on success,
        or None on failure.
    model_used : str
        Description of the model providing the final prediction
        ('edge', 'cloud', 'hybrid_edge', 'hybrid_cloud', 'none').
    inference_latency_s : float
        Measured wall-clock execution latency in seconds.
    cloud_fallback : bool
        True if Hybrid action evaluated Edge and fell back to Cloud.
    success : bool
        True if execution succeeded, False if any validation or execution failure occurred.
    status : ExecutionStatus
        Explicit execution status ({SUCCESS, FALLBACK, FAILED}).
    edge_latency_s : float | None
        Measured local software execution latency for Edge model, if executed.
    cloud_latency_s : float | None
        Measured local software execution latency for Cloud model, if executed.
    hybrid_latency_s : float | None
        Measured complete wall-clock duration for Hybrid path, if action was HYBRID.
    error : str | None
        Diagnostic failure error message if execution failed, else None.
    observation_index : int | None
        Observation index from decision or streaming sequence, if available.
    timestamp : Any | None
        Observation timestamp, if available.
    """

    decision: DecisionResult
    action: DecisionAction
    prediction: int | None
    probabilities: dict[int, float] | None
    model_used: str
    inference_latency_s: float
    cloud_fallback: bool = False
    success: bool = True
    status: ExecutionStatus = ExecutionStatus.SUCCESS
    edge_latency_s: float | None = None
    cloud_latency_s: float | None = None
    hybrid_latency_s: float | None = None
    network_latency_s: float | None = None
    packet_lost: bool = False
    error: str | None = None
    observation_index: int | None = None
    timestamp: Any | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.action, DecisionAction):
            raise TypeError(f"action must be DecisionAction, got {type(self.action).__name__}")

        # Auto-align status if passed as string
        if isinstance(self.status, str) and not isinstance(self.status, ExecutionStatus):
            object.__setattr__(self, "status", ExecutionStatus.from_str(self.status))
        elif not isinstance(self.status, ExecutionStatus):
            raise TypeError(f"status must be ExecutionStatus, got {type(self.status).__name__}")

        # Auto-align status if cloud_fallback is True and status is SUCCESS
        if self.cloud_fallback and self.status == ExecutionStatus.SUCCESS:
            object.__setattr__(self, "status", ExecutionStatus.FALLBACK)

        # Auto-populate observation_index and timestamp from decision if None
        if self.observation_index is None and self.decision is not None:
            object.__setattr__(self, "observation_index", self.decision.observation_index)
        if self.timestamp is None and self.decision is not None:
            object.__setattr__(self, "timestamp", self.decision.timestamp)

        if self.success:
            if self.prediction not in (0, 1):
                raise ValueError(f"prediction must be 0 or 1 on successful execution, got {self.prediction}")
            if not isinstance(self.probabilities, Mapping):
                raise TypeError("probabilities must be a mapping {class: prob} on successful execution")
        else:
            if self.status != ExecutionStatus.FAILED:
                object.__setattr__(self, "status", ExecutionStatus.FAILED)



class BaseController(abc.ABC):
    """Abstract base class for routing controllers (adaptive and static baseline)."""

    @abc.abstractmethod
    def decide(self, inputs: DecisionInputs | float) -> DecisionResult:
        """Compute the next routing decision a_t in {EDGE, CLOUD, HYBRID}."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset internal temporal state (current action, switch count, counters)."""

    @abc.abstractmethod
    def get_info(self) -> dict[str, Any]:
        """Return introspection metadata, configuration, and current state."""


class BaseDecisionEngine(abc.ABC):
    """Abstract base class for the unified DRAEC Decision Engine."""

    @abc.abstractmethod
    def decide(self, inputs: DecisionInputs | float) -> DecisionResult:
        """Route without executing model inference."""

    @abc.abstractmethod
    def execute(self, x: Any, inputs: DecisionInputs | float) -> ExecutionResult:
        """Route and execute minimal inference for observation x."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset engine, controller, and instrumentation state."""

    @abc.abstractmethod
    def get_info(self) -> dict[str, Any]:
        """Return diagnostic metadata across controller, models, and instrumentation."""
