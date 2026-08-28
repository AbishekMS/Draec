"""Drift-Aware Adaptive Edge-Cloud-Hybrid Orchestration for Non-Stationary IoT.

Module   : src/decision/engine.py
Phase    : Phase 5
Status   : IMPLEMENTED

Decision Engine and Controllers for DRAEC inference routing.
Implements:
- AdaptiveController: state-machine controller with hysteresis driven by R_t
- StaticBaselineController: static baseline routing independent of R_t / D_t
- DecisionInstrumentation: lightweight memory-bounded telemetry tracker
- DecisionEngine: unified coordinator executing Edge, Cloud, or two-level Hybrid
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
)
from src.models.base import BaseModel


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
        self._total_latency_s: float = 0.0
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
    def total_latency_s(self) -> float:
        return self._total_latency_s

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

        if execution is not None:
            lat_s = float(execution.inference_latency_s)
            self._total_latency_s += lat_s
            fb = bool(execution.cloud_fallback)
            if fb:
                self._hybrid_fallback_count += 1
            pred = execution.prediction
            model_used = execution.model_used

        entry = {
            "index": decision.observation_index,
            "action": act.value,
            "reliability": decision.reliability,
            "latency_s": lat_s,
            "fallback": fb,
            "prediction": pred,
            "model_used": model_used,
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
            "edge_count": self._edge_count,
            "cloud_count": self._cloud_count,
            "hybrid_count": self._hybrid_count,
            "hybrid_fallback_count": self._hybrid_fallback_count,
            "switch_count": self._switch_count,
            "total_latency_s": self._total_latency_s,
            "mean_latency_s": mean_lat,
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
        self._total_latency_s = 0.0
        self._records.clear()


class DecisionEngine(BaseDecisionEngine):
    """Unified DRAEC Decision Engine managing action selection and minimal execution.

    Coordinates:
    - Routing controller: AdaptiveController or StaticBaselineController
    - Edge model: River Hoeffding Tree
    - Cloud model: XGBoost Classifier
    - Two-level Hybrid policy:
        Level 1: Action selection a_t in {EDGE, CLOUD, HYBRID}
        Level 2: When HYBRID, execute Edge first -> check Edge confidence -> fallback to Cloud if insufficient
    - Lightweight instrumentation
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
        **kwargs: Any,
    ) -> None:
        self._controller = controller
        self._edge_model = edge_model
        self._cloud_model = cloud_model

        cfg_params: dict[str, Any] = {}
        inst_params: dict[str, Any] = {}
        if config is not None:
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
    def instrumentation(self) -> DecisionInstrumentation:
        return self._instrumentation

    def decide(self, inputs: DecisionInputs | float) -> DecisionResult:
        """Route observation to action a_t without executing models."""
        return self._controller.decide(inputs)

    def execute(self, x: Any, inputs: DecisionInputs | float) -> ExecutionResult:
        """Evaluate routing decision and execute minimal inference for observation x.

        Two-Level Execution Architecture:
        ---------------------------------
        LEVEL 1 (Action Selection):
            The controller selects action a_t in {EDGE, CLOUD, HYBRID} based on R_t.

        LEVEL 2 (Execution):
            - If a_t == EDGE:
                Execute Edge model only; returns Edge prediction.
            - If a_t == CLOUD:
                Execute Cloud model only; returns Cloud prediction.
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

        pred: int
        probs: dict[int, float]
        model_used: str
        fallback: bool = False

        t_start = time.perf_counter()

        if action == DecisionAction.EDGE:
            pred = int(self._edge_model.predict_one(x))
            probs = self._edge_model.predict_proba_one(x)
            model_used = "edge"
            fallback = False

        elif action == DecisionAction.CLOUD:
            pred = int(self._cloud_model.predict_one(x))
            probs = self._cloud_model.predict_proba_one(x)
            model_used = "cloud"
            fallback = False

        elif action == DecisionAction.HYBRID:
            # LEVEL 2: Execute Edge first
            edge_pred = int(self._edge_model.predict_one(x))
            edge_probs = self._edge_model.predict_proba_one(x)

            # Evaluate current causal Edge confidence: C_edge = 2 * (max(P(0), P(1)) - 0.5)
            p0 = float(edge_probs.get(0, 0.5))
            p1 = float(edge_probs.get(1, 0.5))
            max_p = max(p0, p1)
            c_edge = 2.0 * (max_p - 0.5)

            # Insufficient Edge confidence condition
            if self._fallback_on_uncertainty and (c_edge < self._fallback_confidence_threshold):
                # Fall back to Cloud; Cloud provides the final result
                cloud_pred = int(self._cloud_model.predict_one(x))
                cloud_probs = self._cloud_model.predict_proba_one(x)
                pred = cloud_pred
                probs = cloud_probs
                model_used = "hybrid_cloud"
                fallback = True
            else:
                pred = edge_pred
                probs = edge_probs
                model_used = "hybrid_edge"
                fallback = False
        else:
            raise ValueError(f"Unknown action {action}")

        elapsed_s = time.perf_counter() - t_start

        # If hybrid fallback occurred, reflect it in the decision result copy
        if fallback and not decision.hybrid_fallback:
            decision = DecisionResult(
                selected_action=decision.selected_action,
                reliability=decision.reliability,
                previous_action=decision.previous_action,
                decision_reason=f"{decision.decision_reason} -> Cloud fallback triggered",
                decision_inputs=decision.decision_inputs,
                switch_count=decision.switch_count,
                hybrid_fallback=True,
                observation_index=decision.observation_index,
                timestamp=decision.timestamp,
            )

        execution_res = ExecutionResult(
            decision=decision,
            action=action,
            prediction=pred,
            probabilities=probs,
            model_used=model_used,
            inference_latency_s=elapsed_s,
            cloud_fallback=fallback,
        )

        self._instrumentation.record(decision, execution_res)
        return execution_res

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
            "instrumentation": self._instrumentation.get_summary(),
        }
