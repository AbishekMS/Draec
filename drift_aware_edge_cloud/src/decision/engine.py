"""Drift-Aware Adaptive Edge-Cloud-Hybrid Orchestration for Non-Stationary IoT.

Module   : src/decision/engine.py
Phase    : Phase 5 & 6
Status   : IMPLEMENTED

Decision Engine and Hardened Execution Layer for DRAEC inference.
Implements:
- AdaptiveController: state-machine controller with hysteresis driven by R_t
- StaticBaselineController: static baseline routing independent of R_t / D_t
- DecisionInstrumentation: lightweight memory-bounded telemetry tracker
- DecisionEngine: unified coordinator executing Edge, Cloud, or two-level Hybrid
- validate_input: validation for inference observations
- validate_output: validation for model predictions and probabilities
"""

from __future__ import annotations

import collections
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.decision.base import (
    BaseController,
    BaseDecisionEngine,
    DecisionAction,
    DecisionInputs,
    DecisionResult,
    ExecutionResult,
    ExecutionStatus,
)
from src.models.base import BaseModel


def validate_input(
    x: Any,
    expected_dim: int | None = None,
    feature_names: Sequence[str] | None = None,
    forbidden_keys: Sequence[str] | None = None,
) -> None:
    """Validate observation input for inference models.

    Validates:
    - Input is not None or empty
    - Type is supported (Mapping, pd.Series, list, tuple, np.ndarray)
    - Values are numeric and finite (no NaN or Inf)
    - Dimension matches expected dimension if provided
    - No forbidden leakage keys (e.g. Target, ground truth) are present
    """
    if x is None:
        raise ValueError("Input observation x cannot be None")

    forbidden = set(forbidden_keys or ("Target", "target", "Traffic", "ground_truth"))

    if isinstance(x, Mapping):
        if not x:
            raise ValueError("Input mapping cannot be empty")
        found_forbidden = [k for k in x if str(k) in forbidden]
        if found_forbidden:
            raise ValueError(f"Input contains forbidden leakage key(s): {found_forbidden}")
        for k, v in x.items():
            if v is None or not isinstance(v, (int, float, np.number)):
                raise TypeError(f"Input feature {k!r} must be numeric, got {type(v).__name__}")
            if not np.isfinite(float(v)):
                raise ValueError(f"Input feature {k!r} must be finite, got {v}")
        if expected_dim is not None and len(x) != expected_dim:
            raise ValueError(f"Input dimension mismatch: expected {expected_dim} features, got {len(x)}")

    elif isinstance(x, pd.Series):
        if x.empty:
            raise ValueError("Input Series cannot be empty")
        found_forbidden = [k for k in x.index if str(k) in forbidden]
        if found_forbidden:
            raise ValueError(f"Input contains forbidden leakage column(s): {found_forbidden}")
        if not np.issubdtype(x.dtype, np.number):
            raise TypeError("Input Series must contain numeric values")
        if not np.isfinite(x.to_numpy(dtype=float)).all():
            raise ValueError("Input Series contains non-finite (NaN or Inf) values")
        if expected_dim is not None and len(x) != expected_dim:
            raise ValueError(f"Input dimension mismatch: expected {expected_dim} features, got {len(x)}")

    elif isinstance(x, (list, tuple, np.ndarray)):
        arr = np.asarray(x)
        if arr.size == 0:
            raise ValueError("Input sequence cannot be empty")
        if arr.ndim > 1:
            raise ValueError(f"Input observation must be 1D, got ndim={arr.ndim}")
        if not np.issubdtype(arr.dtype, np.number):
            raise TypeError("Input sequence must contain numeric values")
        if not np.isfinite(arr.astype(float)).all():
            raise ValueError("Input sequence contains non-finite (NaN or Inf) values")
        if expected_dim is not None and len(arr) != expected_dim:
            raise ValueError(f"Input dimension mismatch: expected {expected_dim} features, got {len(arr)}")

    else:
        raise TypeError(f"Unsupported observation input type: {type(x).__name__}")


def validate_output(pred: Any, probas: Any) -> None:
    """Validate inference model outputs.

    Validates:
    - Predicted class label is binary {0, 1}
    - Probabilities is a mapping containing classes 0 and 1
    - Probabilities are finite floats in [0.0, 1.0]
    - Probabilities sum to 1.0 within tolerance (1e-4)
    """
    if pred not in (0, 1):
        raise ValueError(f"Predicted class label must be 0 or 1, got {pred!r}")
    if not isinstance(probas, Mapping):
        raise TypeError(f"Predicted probabilities must be a mapping {{class: prob}}, got {type(probas).__name__}")
    if 0 not in probas or 1 not in probas:
        raise ValueError(f"Probabilities mapping must contain classes 0 and 1, got keys {list(probas.keys())}")

    p0 = probas[0]
    p1 = probas[1]
    for cls_idx, p in ((0, p0), (1, p1)):
        if not isinstance(p, (int, float, np.number)):
            raise TypeError(f"Probability for class {cls_idx} must be numeric, got {type(p).__name__}")
        p_flt = float(p)
        if not np.isfinite(p_flt):
            raise ValueError(f"Probability for class {cls_idx} must be finite, got {p_flt}")
        if not (-1e-9 <= p_flt <= 1.0 + 1e-9):
            raise ValueError(f"Probability for class {cls_idx} must be in [0, 1], got {p_flt}")

    total = float(p0) + float(p1)
    if abs(total - 1.0) > 1e-4:
        raise ValueError(f"Probabilities must sum to 1.0 +/- 1e-4, got sum={total}")



class AdaptiveController(BaseController):
    """Reliability-aware adaptive controller with deterministic state-machine hysteresis.

    State Machine Dynamics:
    -----------------------
    Thresholds:
        critical_cloud_threshold (0.30) < cloud_threshold (0.50) < edge_return_threshold (0.70)

    When currently EDGE:
        R_t >= cloud_threshold (0.50):
            Remain EDGE.
        0.30 <= R_t < cloud_threshold (0.50):
            Transition EDGE -> HYBRID (if hybrid enabled, else CLOUD).
        R_t < critical_cloud_threshold (0.30):
            Transition EDGE -> CLOUD.

    When currently HYBRID:
        R_t >= edge_return_threshold (0.70):
            Recover HYBRID -> EDGE.
        R_t < critical_cloud_threshold (0.30):
            Degrade HYBRID -> CLOUD.
        0.30 <= R_t < edge_return_threshold (0.70):
            Maintain HYBRID (deadband prevents chatter).

    When currently CLOUD:
        R_t >= edge_return_threshold (0.70):
            Recover CLOUD -> EDGE.
        R_t < edge_return_threshold (0.70):
            Maintain CLOUD.

    Between thresholds [0.50, 0.70), the controller maintains the previous action
    to prevent high-frequency oscillation (chatter) between Edge and Cloud.
    """

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        *,
        cloud_threshold: float = 0.50,
        edge_return_threshold: float = 0.70,
        critical_cloud_threshold: float = 0.30,
        initial_action: DecisionAction | str = DecisionAction.EDGE,
        hybrid_enabled: bool = True,
        **kwargs: Any,
    ) -> None:
        cfg_params: dict[str, Any] = {}
        if config is not None:
            # Check top-level decision or nested under drift/decision
            dec_sec = config.get("decision")
            if isinstance(dec_sec, Mapping):
                adap_sec = dec_sec.get("adaptive")
                if isinstance(adap_sec, Mapping):
                    cfg_params.update(adap_sec)
                hyb_sec = dec_sec.get("hybrid")
                if isinstance(hyb_sec, Mapping) and "enabled" in hyb_sec:
                    cfg_params["hybrid_enabled"] = hyb_sec["enabled"]
            elif isinstance(config.get("drift"), Mapping):
                dec_sec = config["drift"].get("decision")
                if isinstance(dec_sec, Mapping):
                    adap_sec = dec_sec.get("adaptive")
                    if isinstance(adap_sec, Mapping):
                        cfg_params.update(adap_sec)
                    hyb_sec = dec_sec.get("hybrid")
                    if isinstance(hyb_sec, Mapping) and "enabled" in hyb_sec:
                        cfg_params["hybrid_enabled"] = hyb_sec["enabled"]

        self._cloud_threshold = float(
            kwargs.get("cloud_threshold", cfg_params.get("cloud_threshold", cloud_threshold))
        )
        self._edge_return_threshold = float(
            kwargs.get(
                "edge_return_threshold",
                cfg_params.get("edge_return_threshold", edge_return_threshold),
            )
        )
        self._critical_cloud_threshold = float(
            kwargs.get(
                "critical_cloud_threshold",
                cfg_params.get("critical_cloud_threshold", critical_cloud_threshold),
            )
        )
        raw_init = kwargs.get("initial_action", cfg_params.get("initial_action", initial_action))
        self._initial_action = DecisionAction.from_str(raw_init)
        self._hybrid_enabled = bool(
            kwargs.get("hybrid_enabled", cfg_params.get("hybrid_enabled", hybrid_enabled))
        )

        if not (0.0 <= self._critical_cloud_threshold < self._cloud_threshold < self._edge_return_threshold <= 1.0):
            raise ValueError(
                f"Threshold ordering must satisfy 0.0 <= critical ({self._critical_cloud_threshold}) < "
                f"cloud ({self._cloud_threshold}) < return ({self._edge_return_threshold}) <= 1.0"
            )

        self._current_action: DecisionAction = self._initial_action
        self._previous_action: DecisionAction | None = None
        self._switch_count: int = 0
        self._decision_count: int = 0
        self._last_result: DecisionResult | None = None

    @property
    def cloud_threshold(self) -> float:
        return self._cloud_threshold

    @property
    def edge_return_threshold(self) -> float:
        return self._edge_return_threshold

    @property
    def critical_cloud_threshold(self) -> float:
        return self._critical_cloud_threshold

    @property
    def hybrid_enabled(self) -> bool:
        return self._hybrid_enabled

    @property
    def current_action(self) -> DecisionAction:
        return self._current_action

    @property
    def previous_action(self) -> DecisionAction | None:
        return self._previous_action

    @property
    def switch_count(self) -> int:
        return self._switch_count

    @property
    def decision_count(self) -> int:
        return self._decision_count

    def decide(self, inputs: DecisionInputs | float) -> DecisionResult:
        """Compute next deterministic action a_t given current reliability R_t."""
        if isinstance(inputs, (int, float)):
            inp = DecisionInputs(reliability=float(inputs))
        elif isinstance(inputs, DecisionInputs):
            inp = inputs
        else:
            raise TypeError(f"inputs must be DecisionInputs or float, got {type(inputs).__name__}")

        r_val = float(inp.reliability)
        prev = self._current_action
        selected: DecisionAction
        reason: str

        if prev == DecisionAction.EDGE:
            if r_val >= self._cloud_threshold:
                selected = DecisionAction.EDGE
                reason = (
                    f"R_t ({r_val:.4f}) >= cloud_threshold ({self._cloud_threshold:.2f}); "
                    "maintain EDGE"
                )
            elif self._hybrid_enabled and r_val >= self._critical_cloud_threshold:
                selected = DecisionAction.HYBRID
                reason = (
                    f"R_t ({r_val:.4f}) in hybrid zone [{self._critical_cloud_threshold:.2f}, "
                    f"{self._cloud_threshold:.2f}); transition EDGE -> HYBRID"
                )
            else:
                selected = DecisionAction.CLOUD
                reason = (
                    f"R_t ({r_val:.4f}) < critical_cloud_threshold ({self._critical_cloud_threshold:.2f}); "
                    "transition EDGE -> CLOUD"
                )

        elif prev == DecisionAction.HYBRID:
            if r_val >= self._edge_return_threshold:
                selected = DecisionAction.EDGE
                reason = (
                    f"R_t ({r_val:.4f}) >= edge_return_threshold ({self._edge_return_threshold:.2f}); "
                    "recover HYBRID -> EDGE"
                )
            elif r_val < self._critical_cloud_threshold or not self._hybrid_enabled:
                selected = DecisionAction.CLOUD
                reason = (
                    f"R_t ({r_val:.4f}) < critical_cloud_threshold ({self._critical_cloud_threshold:.2f}); "
                    "degrade HYBRID -> CLOUD"
                )
            else:
                selected = DecisionAction.HYBRID
                reason = (
                    f"R_t ({r_val:.4f}) in hysteresis/hybrid zone; maintain HYBRID"
                )

        elif prev == DecisionAction.CLOUD:
            if r_val >= self._edge_return_threshold:
                selected = DecisionAction.EDGE
                reason = (
                    f"R_t ({r_val:.4f}) >= edge_return_threshold ({self._edge_return_threshold:.2f}); "
                    "recover CLOUD -> EDGE"
                )
            else:
                selected = DecisionAction.CLOUD
                reason = (
                    f"R_t ({r_val:.4f}) < edge_return_threshold ({self._edge_return_threshold:.2f}); "
                    "maintain CLOUD"
                )
        else:
            selected = self._initial_action
            reason = f"Fallback to initial_action {self._initial_action.value}"

        if selected != prev:
            self._switch_count += 1

        self._previous_action = prev
        self._current_action = selected
        self._decision_count += 1

        result = DecisionResult(
            selected_action=selected,
            reliability=r_val,
            previous_action=prev,
            decision_reason=reason,
            decision_inputs=inp,
            switch_count=self._switch_count,
            hybrid_fallback=False,
            observation_index=inp.observation_index,
            timestamp=inp.timestamp,
        )
        self._last_result = result
        return result

    def reset(self) -> None:
        """Reset controller to configured pre-stream initial state."""
        self._current_action = self._initial_action
        self._previous_action = None
        self._switch_count = 0
        self._decision_count = 0
        self._last_result = None

    def get_info(self) -> dict[str, Any]:
        return {
            "controller_type": "AdaptiveController",
            "initial_action": self._initial_action.value,
            "current_action": self._current_action.value,
            "previous_action": self._previous_action.value if self._previous_action else None,
            "cloud_threshold": self._cloud_threshold,
            "edge_return_threshold": self._edge_return_threshold,
            "critical_cloud_threshold": self._critical_cloud_threshold,
            "hybrid_enabled": self._hybrid_enabled,
            "switch_count": self._switch_count,
            "decision_count": self._decision_count,
        }


class StaticBaselineController(BaseController):
    """Static baseline controller that does NOT consume reliability R_t or drift D_t.

    Required for future scientific comparisons. Implements fixed or deterministic
    routing policies without any adaptive reliability or drift feedback.

    Supported Policies:
    -------------------
    - 'edge_only'     : Always routes to EDGE.
    - 'cloud_only'    : Always routes to CLOUD.
    - 'static_hybrid' : Always routes to HYBRID.
    - 'round_robin'   : Alternates deterministically between EDGE and CLOUD.
    - 'fixed_ratio'   : Routes to EDGE with configured probability/ratio, else CLOUD.
    """

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        *,
        policy: str = "edge_only",
        ratio: float = 0.5,
        **kwargs: Any,
    ) -> None:
        cfg_params: dict[str, Any] = {}
        if config is not None:
            dec_sec = config.get("decision")
            if isinstance(dec_sec, Mapping):
                base_sec = dec_sec.get("baseline")
                if isinstance(base_sec, Mapping):
                    cfg_params.update(base_sec)
            elif isinstance(config.get("drift"), Mapping):
                dec_sec = config["drift"].get("decision")
                if isinstance(dec_sec, Mapping):
                    base_sec = dec_sec.get("baseline")
                    if isinstance(base_sec, Mapping):
                        cfg_params.update(base_sec)

        self._policy = str(kwargs.get("policy", cfg_params.get("policy", policy))).lower()
        self._ratio = float(kwargs.get("ratio", cfg_params.get("ratio", ratio)))

        valid_policies = {"edge_only", "cloud_only", "static_hybrid", "round_robin", "fixed_ratio"}
        if self._policy not in valid_policies:
            raise ValueError(f"Unknown baseline policy {self._policy!r}; expected one of {valid_policies}")
        if not (0.0 <= self._ratio <= 1.0):
            raise ValueError(f"ratio must be in [0, 1], got {self._ratio}")

        self._current_action: DecisionAction = DecisionAction.EDGE
        self._previous_action: DecisionAction | None = None
        self._step_count: int = 0
        self._switch_count: int = 0

    @property
    def policy(self) -> str:
        return self._policy

    @property
    def ratio(self) -> float:
        return self._ratio

    def decide(self, inputs: DecisionInputs | float | None = None) -> DecisionResult:
        """Route deterministically without consuming R_t, D_t, or reliability factors."""
        # Intentionally ignore R_t and D_t values inside inputs
        r_reported = 1.0
        obs_idx = None
        ts = None
        if isinstance(inputs, DecisionInputs):
            r_reported = float(inputs.reliability)
            obs_idx = inputs.observation_index
            ts = inputs.timestamp
        elif isinstance(inputs, (int, float)):
            r_reported = float(inputs)

        prev = self._previous_action
        selected: DecisionAction
        reason = f"static baseline policy: {self._policy}"

        if self._policy == "edge_only":
            selected = DecisionAction.EDGE
        elif self._policy == "cloud_only":
            selected = DecisionAction.CLOUD
        elif self._policy == "static_hybrid":
            selected = DecisionAction.HYBRID
        elif self._policy == "round_robin":
            selected = DecisionAction.EDGE if (self._step_count % 2 == 0) else DecisionAction.CLOUD
        elif self._policy == "fixed_ratio":
            # Deterministic fractional counter
            selected = DecisionAction.EDGE if ((self._step_count * self._ratio) % 1.0 < self._ratio) else DecisionAction.CLOUD
        else:
            selected = DecisionAction.EDGE

        if prev is not None and selected != prev:
            self._switch_count += 1

        self._previous_action = self._current_action
        self._current_action = selected
        self._step_count += 1

        return DecisionResult(
            selected_action=selected,
            reliability=r_reported,
            previous_action=prev,
            decision_reason=reason,
            decision_inputs=None,
            switch_count=self._switch_count,
            hybrid_fallback=False,
            observation_index=obs_idx,
            timestamp=ts,
        )

    def reset(self) -> None:
        self._current_action = DecisionAction.EDGE
        self._previous_action = None
        self._step_count = 0
        self._switch_count = 0

    def get_info(self) -> dict[str, Any]:
        return {
            "controller_type": "StaticBaselineController",
            "policy": self._policy,
            "ratio": self._ratio,
            "current_action": self._current_action.value,
            "step_count": self._step_count,
            "switch_count": self._switch_count,
        }


class DecisionInstrumentation:
    """Lightweight, memory-bounded instrumentation tracker for decision and execution telemetry."""

    def __init__(self, max_records: int = 10000, enabled: bool = True) -> None:
        self._enabled = bool(enabled)
        self._max_records = int(max_records)
        self._edge_count: int = 0
        self._cloud_count: int = 0
        self._hybrid_count: int = 0
        self._hybrid_fallback_count: int = 0
        self._switch_count: int = 0
        self._total_decisions: int = 0
        self._total_executions: int = 0
        self._successful_executions: int = 0
        self._failed_executions: int = 0
        self._edge_failure_count: int = 0
        self._cloud_failure_count: int = 0
        self._total_latency_s: float = 0.0
        self._min_latency_s: float | None = None
        self._max_latency_s: float | None = None
        self._records: collections.deque = collections.deque(maxlen=self._max_records)

    @property
    def edge_count(self) -> int:
        return self._edge_count

    @property
    def cloud_count(self) -> int:
        return self._cloud_count

    @property
    def hybrid_count(self) -> int:
        return self._hybrid_count

    @property
    def hybrid_fallback_count(self) -> int:
        return self._hybrid_fallback_count

    @property
    def switch_count(self) -> int:
        return self._switch_count

    @property
    def total_decisions(self) -> int:
        return self._total_decisions

    @property
    def total_executions(self) -> int:
        return self._total_executions

    @property
    def successful_executions(self) -> int:
        return self._successful_executions

    @property
    def failed_executions(self) -> int:
        return self._failed_executions

    @property
    def edge_failure_count(self) -> int:
        return self._edge_failure_count

    @property
    def cloud_failure_count(self) -> int:
        return self._cloud_failure_count

    @property
    def total_latency_s(self) -> float:
        return self._total_latency_s

    @property
    def min_latency_s(self) -> float | None:
        return self._min_latency_s

    @property
    def max_latency_s(self) -> float | None:
        return self._max_latency_s

    def record(
        self,
        decision: DecisionResult,
        execution: ExecutionResult | None = None,
    ) -> None:
        """Record decision and execution metadata without altering operational behavior."""
        if not self._enabled:
            return

        self._total_decisions += 1
        act = decision.selected_action

        if act == DecisionAction.EDGE:
            self._edge_count += 1
        elif act == DecisionAction.CLOUD:
            self._cloud_count += 1
        elif act == DecisionAction.HYBRID:
            self._hybrid_count += 1

        if decision.previous_action is not None and act != decision.previous_action:
            self._switch_count += 1

        lat_s = 0.0
        fb = False
        pred = None
        model_used = None
        status_val = "UNKNOWN"
        succ = True
        err_msg = None

        if execution is not None:
            self._total_executions += 1
            lat_s = float(execution.inference_latency_s)
            self._total_latency_s += lat_s
            if self._min_latency_s is None or lat_s < self._min_latency_s:
                self._min_latency_s = lat_s
            if self._max_latency_s is None or lat_s > self._max_latency_s:
                self._max_latency_s = lat_s

            fb = bool(execution.cloud_fallback)
            if fb:
                self._hybrid_fallback_count += 1
            pred = execution.prediction
            model_used = execution.model_used
            succ = bool(execution.success)
            status_val = (
                execution.status.value
                if isinstance(execution.status, ExecutionStatus)
                else str(execution.status)
            )
            err_msg = execution.error

            if succ:
                self._successful_executions += 1
            else:
                self._failed_executions += 1
                if act == DecisionAction.EDGE:
                    self._edge_failure_count += 1
                elif act == DecisionAction.CLOUD:
                    self._cloud_failure_count += 1
                elif act == DecisionAction.HYBRID:
                    if fb or "cloud" in str(model_used).lower() or "cloud" in str(err_msg).lower():
                        self._cloud_failure_count += 1
                    else:
                        self._edge_failure_count += 1

        entry = {
            "index": decision.observation_index,
            "action": act.value,
            "reliability": decision.reliability,
            "latency_s": lat_s,
            "fallback": fb,
            "prediction": pred,
            "model_used": model_used,
            "status": status_val,
            "success": succ,
            "error": err_msg,
            "reason": decision.decision_reason,
        }
        self._records.append(entry)

    def get_summary(self) -> dict[str, Any]:
        """Return cumulative summary statistics."""
        mean_lat = (
            self._total_latency_s / self._total_decisions
            if self._total_decisions > 0
            else 0.0
        )
        return {
            "total_decisions": self._total_decisions,
            "total_executions": self._total_executions,
            "successful_executions": self._successful_executions,
            "failed_executions": self._failed_executions,
            "edge_count": self._edge_count,
            "cloud_count": self._cloud_count,
            "hybrid_count": self._hybrid_count,
            "hybrid_fallback_count": self._hybrid_fallback_count,
            "edge_failure_count": self._edge_failure_count,
            "cloud_failure_count": self._cloud_failure_count,
            "switch_count": self._switch_count,
            "total_latency_s": self._total_latency_s,
            "mean_latency_s": mean_lat,
            "latency_stats": {
                "count": self._total_executions,
                "mean_s": mean_lat,
                "min_s": self._min_latency_s if self._min_latency_s is not None else 0.0,
                "max_s": self._max_latency_s if self._max_latency_s is not None else 0.0,
                "total_s": self._total_latency_s,
            },
            "records_stored": len(self._records),
        }

    def reset(self) -> None:
        """Reset all instrumentation counters and buffers."""
        self._edge_count = 0
        self._cloud_count = 0
        self._hybrid_count = 0
        self._hybrid_fallback_count = 0
        self._switch_count = 0
        self._total_decisions = 0
        self._total_executions = 0
        self._successful_executions = 0
        self._failed_executions = 0
        self._edge_failure_count = 0
        self._cloud_failure_count = 0
        self._total_latency_s = 0.0
        self._min_latency_s = None
        self._max_latency_s = None
        self._records.clear()


class DecisionEngine(BaseDecisionEngine):
    """Unified DRAEC Decision Engine managing action selection and hardened execution.

    Coordinates:
    - Routing controller: AdaptiveController or StaticBaselineController
    - Edge model: River Hoeffding Tree
    - Cloud model: XGBoost Classifier (local software execution)
    - Two-level Hybrid policy:
        Level 1: Action selection a_t in {EDGE, CLOUD, HYBRID} driven by R_t
        Level 2: When HYBRID, execute Edge first -> check Edge confidence -> fallback to Cloud if insufficient
    - Input and output validation
    - Explicit execution failure handling
    - Fine-grained software latency profiling (T_edge, T_cloud, T_hybrid)
    - Lightweight, memory-bounded instrumentation
    """

    def __init__(
        self,
        controller: BaseController,
        edge_model: BaseModel,
        cloud_model: BaseModel,
        config: Mapping[str, Any] | None = None,
        *,
        fallback_confidence_threshold: float = 0.60,
        fallback_on_uncertainty: bool = True,
        instrumentation_enabled: bool = True,
        max_instrumentation_records: int = 10000,
        input_validation: bool = True,
        output_validation: bool = True,
        timing_enabled: bool = True,
        failure_policy: str = "fail",
        **kwargs: Any,
    ) -> None:
        self._controller = controller
        self._edge_model = edge_model
        self._cloud_model = cloud_model

        cfg_params: dict[str, Any] = {}
        inst_params: dict[str, Any] = {}
        exec_params: dict[str, Any] = {}

        if config is not None:
            # Check decision section
            dec_sec = config.get("decision")
            if not isinstance(dec_sec, Mapping) and isinstance(config.get("drift"), Mapping):
                dec_sec = config["drift"].get("decision")
            if isinstance(dec_sec, Mapping):
                hyb_sec = dec_sec.get("hybrid")
                if isinstance(hyb_sec, Mapping):
                    cfg_params.update(hyb_sec)
                inst_sec = dec_sec.get("instrumentation")
                if isinstance(inst_sec, Mapping):
                    inst_params.update(inst_sec)

            # Check execution section (top-level or nested)
            exec_sec = config.get("execution")
            if not isinstance(exec_sec, Mapping) and isinstance(dec_sec, Mapping):
                exec_sec = dec_sec.get("execution")
            if isinstance(exec_sec, Mapping):
                val_sec = exec_sec.get("validation")
                if isinstance(val_sec, Mapping):
                    exec_params["input_validation"] = val_sec.get("enabled", True)
                    exec_params["output_validation"] = val_sec.get("enabled", True)
                tim_sec = exec_sec.get("timing")
                if isinstance(tim_sec, Mapping):
                    exec_params["timing_enabled"] = tim_sec.get("enabled", True)
                tel_sec = exec_sec.get("telemetry")
                if isinstance(tel_sec, Mapping):
                    inst_params.update(tel_sec)
                fail_sec = exec_sec.get("failure")
                if isinstance(fail_sec, Mapping):
                    exec_params["failure_policy"] = fail_sec.get("edge_policy", "fail")

        self._fallback_confidence_threshold = float(
            kwargs.get(
                "fallback_confidence_threshold",
                cfg_params.get("fallback_confidence_threshold", fallback_confidence_threshold),
            )
        )
        self._fallback_on_uncertainty = bool(
            kwargs.get(
                "fallback_on_uncertainty",
                cfg_params.get("fallback_on_uncertainty", fallback_on_uncertainty),
            )
        )

        self._input_validation = bool(
            kwargs.get("input_validation", exec_params.get("input_validation", input_validation))
        )
        self._output_validation = bool(
            kwargs.get("output_validation", exec_params.get("output_validation", output_validation))
        )
        self._timing_enabled = bool(
            kwargs.get("timing_enabled", exec_params.get("timing_enabled", timing_enabled))
        )
        self._failure_policy = str(
            kwargs.get("failure_policy", exec_params.get("failure_policy", failure_policy))
        ).lower()

        inst_en = bool(kwargs.get("instrumentation_enabled", inst_params.get("enabled", instrumentation_enabled)))
        max_rec = int(kwargs.get("max_instrumentation_records", inst_params.get("max_records", max_instrumentation_records)))
        self._instrumentation = DecisionInstrumentation(max_records=max_rec, enabled=inst_en)

    @property
    def controller(self) -> BaseController:
        return self._controller

    @property
    def edge_model(self) -> BaseModel:
        return self._edge_model

    @property
    def cloud_model(self) -> BaseModel:
        return self._cloud_model

    @property
    def fallback_confidence_threshold(self) -> float:
        return self._fallback_confidence_threshold

    @property
    def fallback_on_uncertainty(self) -> bool:
        return self._fallback_on_uncertainty

    @property
    def input_validation(self) -> bool:
        return self._input_validation

    @property
    def output_validation(self) -> bool:
        return self._output_validation

    @property
    def timing_enabled(self) -> bool:
        return self._timing_enabled

    @property
    def failure_policy(self) -> str:
        return self._failure_policy

    @property
    def instrumentation(self) -> DecisionInstrumentation:
        return self._instrumentation

    def _get_expected_dim(self) -> int | None:
        """Infer expected feature dimension from models if available."""
        if hasattr(self._edge_model, "n_features") and self._edge_model.n_features is not None:
            return int(self._edge_model.n_features)
        if hasattr(self._cloud_model, "n_features") and self._cloud_model.n_features is not None:
            return int(self._cloud_model.n_features)
        return None

    def decide(self, inputs: DecisionInputs | float) -> DecisionResult:
        """Route observation to action a_t without executing models."""
        return self._controller.decide(inputs)

    def execute(self, x: Any, inputs: DecisionInputs | float) -> ExecutionResult:
        """Evaluate routing decision and execute hardened inference for observation x.

        Two-Level Execution Architecture:
        ---------------------------------
        LEVEL 1 (Action Selection):
            The controller selects action a_t in {EDGE, CLOUD, HYBRID} based on R_t.

        LEVEL 2 (Execution):
            - If a_t == EDGE:
                Execute Edge model only with validation, timing, and error handling.
            - If a_t == CLOUD:
                Execute Cloud model only with validation, timing, and error handling.
            - If a_t == HYBRID:
                Execute Edge model first.
                Compute Edge confidence C_edge = 2 * (max(P(0), P(1)) - 0.5).
                If C_edge < fallback_confidence_threshold:
                    Edge is insufficient -> invoke Cloud fallback.
                    Cloud output becomes final prediction (cloud_fallback = True).
                Else:
                    Edge confidence is sufficient -> Edge output is final (cloud_fallback = False).
        """
        decision = self.decide(inputs)
        action = decision.selected_action

        # 1. Input Validation
        if self._input_validation:
            try:
                validate_input(x, expected_dim=self._get_expected_dim())
            except Exception as err:
                fail_res = ExecutionResult(
                    decision=decision,
                    action=action,
                    prediction=None,
                    probabilities=None,
                    model_used="none",
                    inference_latency_s=0.0,
                    edge_latency_s=None,
                    cloud_latency_s=None,
                    hybrid_latency_s=None,
                    cloud_fallback=False,
                    success=False,
                    status=ExecutionStatus.FAILED,
                    error=f"Input validation failed: {err}",
                )
                self._instrumentation.record(decision, fail_res)
                return fail_res

        # 2. Action Routing & Hardened Execution
        if action == DecisionAction.EDGE:
            return self._execute_edge(x, decision)
        elif action == DecisionAction.CLOUD:
            return self._execute_cloud(x, decision)
        elif action == DecisionAction.HYBRID:
            return self._execute_hybrid(x, decision)
        else:
            raise ValueError(f"Unknown action {action}")

    def execute_edge(self, x: Any, decision: DecisionResult | None = None) -> ExecutionResult:
        """Directly execute Edge model with full validation and timing."""
        if decision is None:
            decision = DecisionResult(
                selected_action=DecisionAction.EDGE,
                reliability=1.0,
                previous_action=None,
                decision_reason="direct Edge execution",
            )
        if self._input_validation:
            try:
                validate_input(x, expected_dim=self._get_expected_dim())
            except Exception as err:
                fail_res = ExecutionResult(
                    decision=decision,
                    action=DecisionAction.EDGE,
                    prediction=None,
                    probabilities=None,
                    model_used="none",
                    inference_latency_s=0.0,
                    edge_latency_s=None,
                    cloud_latency_s=None,
                    hybrid_latency_s=None,
                    cloud_fallback=False,
                    success=False,
                    status=ExecutionStatus.FAILED,
                    error=f"Input validation failed: {err}",
                )
                self._instrumentation.record(decision, fail_res)
                return fail_res
        return self._execute_edge(x, decision)

    def execute_cloud(self, x: Any, decision: DecisionResult | None = None) -> ExecutionResult:
        """Directly execute Cloud model with full validation and timing."""
        if decision is None:
            decision = DecisionResult(
                selected_action=DecisionAction.CLOUD,
                reliability=1.0,
                previous_action=None,
                decision_reason="direct Cloud execution",
            )
        if self._input_validation:
            try:
                validate_input(x, expected_dim=self._get_expected_dim())
            except Exception as err:
                fail_res = ExecutionResult(
                    decision=decision,
                    action=DecisionAction.CLOUD,
                    prediction=None,
                    probabilities=None,
                    model_used="none",
                    inference_latency_s=0.0,
                    edge_latency_s=None,
                    cloud_latency_s=None,
                    hybrid_latency_s=None,
                    cloud_fallback=False,
                    success=False,
                    status=ExecutionStatus.FAILED,
                    error=f"Input validation failed: {err}",
                )
                self._instrumentation.record(decision, fail_res)
                return fail_res
        return self._execute_cloud(x, decision)

    def execute_hybrid(self, x: Any, decision: DecisionResult | None = None) -> ExecutionResult:
        """Directly execute Hybrid Edge-first path with full validation and timing."""
        if decision is None:
            decision = DecisionResult(
                selected_action=DecisionAction.HYBRID,
                reliability=1.0,
                previous_action=None,
                decision_reason="direct Hybrid execution",
            )
        if self._input_validation:
            try:
                validate_input(x, expected_dim=self._get_expected_dim())
            except Exception as err:
                fail_res = ExecutionResult(
                    decision=decision,
                    action=DecisionAction.HYBRID,
                    prediction=None,
                    probabilities=None,
                    model_used="none",
                    inference_latency_s=0.0,
                    edge_latency_s=None,
                    cloud_latency_s=None,
                    hybrid_latency_s=None,
                    cloud_fallback=False,
                    success=False,
                    status=ExecutionStatus.FAILED,
                    error=f"Input validation failed: {err}",
                )
                self._instrumentation.record(decision, fail_res)
                return fail_res
        return self._execute_hybrid(x, decision)

    def _execute_edge(self, x: Any, decision: DecisionResult) -> ExecutionResult:
        """Harden Edge execution with validation, timing, and error handling."""
        action = DecisionAction.EDGE
        if self._edge_model is None or not getattr(self._edge_model, "is_trained", False):
            fail_res = ExecutionResult(
                decision=decision,
                action=action,
                prediction=None,
                probabilities=None,
                model_used="none",
                inference_latency_s=0.0,
                edge_latency_s=None,
                cloud_latency_s=None,
                hybrid_latency_s=None,
                cloud_fallback=False,
                success=False,
                status=ExecutionStatus.FAILED,
                error="Edge model is missing or untrained",
            )
            self._instrumentation.record(decision, fail_res)
            return fail_res

        t0 = time.perf_counter()
        try:
            pred = int(self._edge_model.predict_one(x))
            probs = self._edge_model.predict_proba_one(x)
            t_edge = time.perf_counter() - t0

            if self._output_validation:
                validate_output(pred, probs)

            exec_res = ExecutionResult(
                decision=decision,
                action=action,
                prediction=pred,
                probabilities=probs,
                model_used="edge",
                inference_latency_s=t_edge,
                edge_latency_s=t_edge,
                cloud_latency_s=None,
                hybrid_latency_s=None,
                cloud_fallback=False,
                success=True,
                status=ExecutionStatus.SUCCESS,
            )
        except Exception as err:
            t_edge = time.perf_counter() - t0
            exec_res = ExecutionResult(
                decision=decision,
                action=action,
                prediction=None,
                probabilities=None,
                model_used="none",
                inference_latency_s=t_edge,
                edge_latency_s=t_edge,
                cloud_latency_s=None,
                hybrid_latency_s=None,
                cloud_fallback=False,
                success=False,
                status=ExecutionStatus.FAILED,
                error=f"Edge execution failed: {err}",
            )

        self._instrumentation.record(decision, exec_res)
        return exec_res

    def _execute_cloud(self, x: Any, decision: DecisionResult) -> ExecutionResult:
        """Harden Cloud execution with validation, timing, and error handling.

        Note: Cloud model executes locally in the current simulation environment.
        T_cloud represents LOCAL SOFTWARE CLOUD-MODEL EXECUTION LATENCY.
        """
        action = DecisionAction.CLOUD
        if self._cloud_model is None or not getattr(self._cloud_model, "is_trained", False):
            fail_res = ExecutionResult(
                decision=decision,
                action=action,
                prediction=None,
                probabilities=None,
                model_used="none",
                inference_latency_s=0.0,
                edge_latency_s=None,
                cloud_latency_s=None,
                hybrid_latency_s=None,
                cloud_fallback=False,
                success=False,
                status=ExecutionStatus.FAILED,
                error="Cloud model is missing or untrained",
            )
            self._instrumentation.record(decision, fail_res)
            return fail_res

        t0 = time.perf_counter()
        try:
            pred = int(self._cloud_model.predict_one(x))
            probs = self._cloud_model.predict_proba_one(x)
            t_cloud = time.perf_counter() - t0

            if self._output_validation:
                validate_output(pred, probs)

            exec_res = ExecutionResult(
                decision=decision,
                action=action,
                prediction=pred,
                probabilities=probs,
                model_used="cloud",
                inference_latency_s=t_cloud,
                edge_latency_s=None,
                cloud_latency_s=t_cloud,
                hybrid_latency_s=None,
                cloud_fallback=False,
                success=True,
                status=ExecutionStatus.SUCCESS,
            )
        except Exception as err:
            t_cloud = time.perf_counter() - t0
            exec_res = ExecutionResult(
                decision=decision,
                action=action,
                prediction=None,
                probabilities=None,
                model_used="none",
                inference_latency_s=t_cloud,
                edge_latency_s=None,
                cloud_latency_s=t_cloud,
                hybrid_latency_s=None,
                cloud_fallback=False,
                success=False,
                status=ExecutionStatus.FAILED,
                error=f"Cloud execution failed: {err}",
            )

        self._instrumentation.record(decision, exec_res)
        return exec_res

    def _execute_hybrid(self, x: Any, decision: DecisionResult) -> ExecutionResult:
        """Harden Hybrid Edge-first execution with fallback to Cloud if confidence insufficient.

        Wall-clock timing:
        - T_hybrid measures complete wall-clock duration of the Hybrid execution path.
        - T_edge measures independent local software Edge execution duration.
        - T_cloud measures independent local software Cloud execution duration when fallback occurs.
        """
        action = DecisionAction.HYBRID
        t_hyb_start = time.perf_counter()

        # Step 1: Check Edge model
        if self._edge_model is None or not getattr(self._edge_model, "is_trained", False):
            t_hybrid = time.perf_counter() - t_hyb_start
            fail_res = ExecutionResult(
                decision=decision,
                action=action,
                prediction=None,
                probabilities=None,
                model_used="none",
                inference_latency_s=t_hybrid,
                edge_latency_s=None,
                cloud_latency_s=None,
                hybrid_latency_s=t_hybrid,
                cloud_fallback=False,
                success=False,
                status=ExecutionStatus.FAILED,
                error="Hybrid Edge execution failed: Edge model is missing or untrained",
            )
            self._instrumentation.record(decision, fail_res)
            return fail_res

        # Step 2: Execute Edge model
        t0_edge = time.perf_counter()
        edge_ok = False
        edge_pred = None
        edge_probs = None
        edge_err = None
        try:
            edge_pred = int(self._edge_model.predict_one(x))
            edge_probs = self._edge_model.predict_proba_one(x)
            if self._output_validation:
                validate_output(edge_pred, edge_probs)
            edge_ok = True
        except Exception as err:
            edge_err = err
        t_edge = time.perf_counter() - t0_edge

        if not edge_ok:
            # Case 4: Edge itself fails during Hybrid -> explicit execution failure
            t_hybrid = time.perf_counter() - t_hyb_start
            fail_res = ExecutionResult(
                decision=decision,
                action=action,
                prediction=None,
                probabilities=None,
                model_used="none",
                inference_latency_s=t_hybrid,
                edge_latency_s=t_edge,
                cloud_latency_s=None,
                hybrid_latency_s=t_hybrid,
                cloud_fallback=False,
                success=False,
                status=ExecutionStatus.FAILED,
                error=f"Hybrid Edge execution failed: {edge_err}",
            )
            self._instrumentation.record(decision, fail_res)
            return fail_res

        # Step 3: Evaluate Edge confidence C_edge = 2 * (max(P(0), P(1)) - 0.5)
        p0 = float(edge_probs.get(0, 0.5))
        p1 = float(edge_probs.get(1, 0.5))
        max_p = max(p0, p1)
        c_edge = 2.0 * (max_p - 0.5)

        # Step 4: If Edge confidence is sufficient, return Edge final result (Case 1)
        if not (self._fallback_on_uncertainty and (c_edge < self._fallback_confidence_threshold)):
            t_hybrid = time.perf_counter() - t_hyb_start
            exec_res = ExecutionResult(
                decision=decision,
                action=action,
                prediction=edge_pred,
                probabilities=edge_probs,
                model_used="hybrid_edge",
                inference_latency_s=t_hybrid,
                edge_latency_s=t_edge,
                cloud_latency_s=None,
                hybrid_latency_s=t_hybrid,
                cloud_fallback=False,
                success=True,
                status=ExecutionStatus.SUCCESS,
            )
            self._instrumentation.record(decision, exec_res)
            return exec_res

        # Step 5: Insufficient Edge confidence -> Invoke Cloud fallback
        if self._cloud_model is None or not getattr(self._cloud_model, "is_trained", False):
            # Case 3: Cloud model missing or untrained during fallback
            t_hybrid = time.perf_counter() - t_hyb_start
            fail_res = ExecutionResult(
                decision=decision,
                action=action,
                prediction=None,
                probabilities=None,
                model_used="none",
                inference_latency_s=t_hybrid,
                edge_latency_s=t_edge,
                cloud_latency_s=None,
                hybrid_latency_s=t_hybrid,
                cloud_fallback=True,
                success=False,
                status=ExecutionStatus.FAILED,
                error="Hybrid Cloud fallback failed: Cloud model is missing or untrained",
            )
            self._instrumentation.record(decision, fail_res)
            return fail_res

        t0_cloud = time.perf_counter()
        cloud_ok = False
        cloud_pred = None
        cloud_probs = None
        cloud_err = None
        try:
            cloud_pred = int(self._cloud_model.predict_one(x))
            cloud_probs = self._cloud_model.predict_proba_one(x)
            if self._output_validation:
                validate_output(cloud_pred, cloud_probs)
            cloud_ok = True
        except Exception as err:
            cloud_err = err
        t_cloud = time.perf_counter() - t0_cloud

        t_hybrid = time.perf_counter() - t_hyb_start

        if not cloud_ok:
            # Case 3: Cloud fails during fallback -> explicit execution failure
            fail_res = ExecutionResult(
                decision=decision,
                action=action,
                prediction=None,
                probabilities=None,
                model_used="none",
                inference_latency_s=t_hybrid,
                edge_latency_s=t_edge,
                cloud_latency_s=t_cloud,
                hybrid_latency_s=t_hybrid,
                cloud_fallback=True,
                success=False,
                status=ExecutionStatus.FAILED,
                error=f"Hybrid Cloud fallback failed: {cloud_err}",
            )
            self._instrumentation.record(decision, fail_res)
            return fail_res

        # Case 2: Cloud succeeds -> Cloud final result
        if not decision.hybrid_fallback:
            decision = DecisionResult(
                selected_action=decision.selected_action,
                reliability=decision.reliability,
                previous_action=decision.previous_action,
                decision_reason=f"{decision.decision_reason} -> Cloud fallback triggered (C_edge={c_edge:.4f} < {self._fallback_confidence_threshold})",
                decision_inputs=decision.decision_inputs,
                switch_count=decision.switch_count,
                hybrid_fallback=True,
                observation_index=decision.observation_index,
                timestamp=decision.timestamp,
            )

        exec_res = ExecutionResult(
            decision=decision,
            action=action,
            prediction=cloud_pred,
            probabilities=cloud_probs,
            model_used="hybrid_cloud",
            inference_latency_s=t_hybrid,
            edge_latency_s=t_edge,
            cloud_latency_s=t_cloud,
            hybrid_latency_s=t_hybrid,
            cloud_fallback=True,
            success=True,
            status=ExecutionStatus.FALLBACK,
        )
        self._instrumentation.record(decision, exec_res)
        return exec_res

    def reset(self) -> None:
        """Reset controller and instrumentation state."""
        self._controller.reset()
        self._instrumentation.reset()

    def get_info(self) -> dict[str, Any]:
        """Return introspection metadata across controller, models, and instrumentation."""
        return {
            "engine": "DecisionEngine",
            "controller": self._controller.get_info(),
            "edge_model": self._edge_model.get_info(),
            "cloud_model": self._cloud_model.get_info(),
            "fallback_confidence_threshold": self._fallback_confidence_threshold,
            "fallback_on_uncertainty": self._fallback_on_uncertainty,
            "input_validation": self._input_validation,
            "output_validation": self._output_validation,
            "timing_enabled": self._timing_enabled,
            "failure_policy": self._failure_policy,
            "instrumentation": self._instrumentation.get_summary(),
        }

