"""Drift-Aware Adaptive Edge-Cloud-Hybrid Orchestration for Non-Stationary IoT.

Module   : src/deployment/runtimes.py
Phase    : Phase 8
Status   : IMPLEMENTED

Runtime execution abstractions for Edge and Cloud models.
Encapsulates model execution, hardware/service availability, failure injection,
and local execution timing.
"""

from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

from src.models.base import BaseModel


class EdgeRuntime:
    """Runtime environment for executing an Edge model.

    Monitors device availability, injects deterministic failure schedules, and measures
    fine-grained local software inference duration (T_edge).
    """

    def __init__(
        self,
        model: Any,
        available: bool = True,
        failure_schedule: Sequence[int] | None = None,
    ) -> None:
        self.model = model
        self.available = bool(available)
        self.failure_schedule = set(failure_schedule) if failure_schedule is not None else set()
        self._total_executions = 0
        self._successful_executions = 0
        self._failed_executions = 0

    def execute(
        self,
        x: Any,
        observation_index: int | None = None,
    ) -> tuple[int | None, dict[int, float] | None, float, bool, str | None]:
        """Execute Edge model inference with local timing and availability checks.

        Returns:
            (prediction, probabilities, latency_s, success, error)
        """
        self._total_executions += 1

        # 1. Check device availability
        if not self.available:
            self._failed_executions += 1
            return None, None, 0.0, False, "EdgeRuntime failure: device offline / unavailable"

        # 2. Check deterministic failure schedule
        if observation_index is not None and observation_index in self.failure_schedule:
            self._failed_executions += 1
            return None, None, 0.0, False, "EdgeRuntime failure: scheduled device error"

        # 3. Execute inference with fine-grained local timing
        t_start = time.perf_counter()
        try:
            if hasattr(self.model, "predict_one") and hasattr(self.model, "predict_proba_one"):
                pred = int(self.model.predict_one(x))
                raw_probs = self.model.predict_proba_one(x)
                probs = {int(k): float(v) for k, v in raw_probs.items()}
            elif hasattr(self.model, "predict_proba"):
                arr = x if hasattr(x, "ndim") and x.ndim == 2 else [x]
                p_arr = self.model.predict_proba(arr)[0]
                probs = {0: float(p_arr[0]), 1: float(p_arr[1])}
                pred = 1 if probs[1] > probs[0] else 0
            else:
                pred = int(self.model.predict(x)[0])
                probs = {pred: 1.0, 1 - pred: 0.0}

            t_elapsed = time.perf_counter() - t_start
            self._successful_executions += 1
            return pred, probs, t_elapsed, True, None

        except Exception as exc:
            t_elapsed = time.perf_counter() - t_start
            self._failed_executions += 1
            return None, None, t_elapsed, False, f"Edge model execution exception: {exc}"

    def set_availability(self, available: bool) -> None:
        self.available = bool(available)

    def schedule_failure(self, observation_index: int) -> None:
        self.failure_schedule.add(int(observation_index))

    def clear_schedule(self) -> None:
        self.failure_schedule.clear()

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_executions": self._total_executions,
            "successful_executions": self._successful_executions,
            "failed_executions": self._failed_executions,
            "available": self.available,
        }

    def reset(self) -> None:
        self._total_executions = 0
        self._successful_executions = 0
        self._failed_executions = 0
        self.failure_schedule.clear()


class CloudRuntime:
    """Runtime environment for executing a Cloud model.

    Monitors service availability, injects deterministic failure schedules, and measures
    fine-grained local software execution duration (T_cloud).
    """

    def __init__(
        self,
        model: Any,
        available: bool = True,
        failure_schedule: Sequence[int] | None = None,
    ) -> None:
        self.model = model
        self.available = bool(available)
        self.failure_schedule = set(failure_schedule) if failure_schedule is not None else set()
        self._total_executions = 0
        self._successful_executions = 0
        self._failed_executions = 0

    def execute(
        self,
        x: Any,
        observation_index: int | None = None,
    ) -> tuple[int | None, dict[int, float] | None, float, bool, str | None]:
        """Execute Cloud model inference with local timing and availability checks.

        Returns:
            (prediction, probabilities, latency_s, success, error)
        """
        self._total_executions += 1

        # 1. Check service availability
        if not self.available:
            self._failed_executions += 1
            return None, None, 0.0, False, "CloudRuntime failure: service offline / unavailable"

        # 2. Check deterministic failure schedule
        if observation_index is not None and observation_index in self.failure_schedule:
            self._failed_executions += 1
            return None, None, 0.0, False, "CloudRuntime failure: scheduled service error"

        # 3. Execute inference with fine-grained local timing
        t_start = time.perf_counter()
        try:
            if hasattr(self.model, "predict_one") and hasattr(self.model, "predict_proba_one"):
                pred = int(self.model.predict_one(x))
                raw_probs = self.model.predict_proba_one(x)
                probs = {int(k): float(v) for k, v in raw_probs.items()}
            elif hasattr(self.model, "predict_proba"):
                arr = x if hasattr(x, "ndim") and x.ndim == 2 else [x]
                p_arr = self.model.predict_proba(arr)[0]
                probs = {0: float(p_arr[0]), 1: float(p_arr[1])}
                pred = 1 if probs[1] > probs[0] else 0
            else:
                pred = int(self.model.predict(x)[0])
                probs = {pred: 1.0, 1 - pred: 0.0}

            t_elapsed = time.perf_counter() - t_start
            self._successful_executions += 1
            return pred, probs, t_elapsed, True, None

        except Exception as exc:
            t_elapsed = time.perf_counter() - t_start
            self._failed_executions += 1
            return None, None, t_elapsed, False, f"Cloud model execution exception: {exc}"

    def set_availability(self, available: bool) -> None:
        self.available = bool(available)

    def schedule_failure(self, observation_index: int) -> None:
        self.failure_schedule.add(int(observation_index))

    def clear_schedule(self) -> None:
        self.failure_schedule.clear()

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_executions": self._total_executions,
            "successful_executions": self._successful_executions,
            "failed_executions": self._failed_executions,
            "available": self.available,
        }

    def reset(self) -> None:
        self._total_executions = 0
        self._successful_executions = 0
        self._failed_executions = 0
        self.failure_schedule.clear()
