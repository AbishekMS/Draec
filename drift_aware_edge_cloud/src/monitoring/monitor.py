"""Drift-Aware Adaptive Edge-Cloud-Hybrid Orchestration for Non-Stationary IoT.

Module   : src/monitoring/monitor.py
Phase    : Phase 7
Status   : IMPLEMENTED

DRAEC Observability and Telemetry Engine.
Coordinates model state tracking, causal streaming monitoring, bounded historical telemetry,
and system-wide health snapshots to support downstream Phase 10 evaluations.
"""

from __future__ import annotations

import collections
import time
from typing import Any, Mapping, Sequence

import pandas as pd

from src.decision.base import DecisionAction, DecisionResult, ExecutionResult, ExecutionStatus
from src.monitoring.base import (
    ModelHealthStatus,
    ModelMetadata,
    MonitoringRecord,
    MonitoringSnapshot,
    StreamStatistics,
)
from src.monitoring.registry import ModelRegistry
from src.reliability.base import ReliabilityScore


class DRAECMonitor:
    """Central observability, model state, and telemetry monitor for DRAEC.

    Observes and records system state across Phases 1-6.
    Does NOT modify scientific formulations, alter routing decisions, or trigger adaptation.
    """

    def __init__(
        self,
        registry: ModelRegistry | None = None,
        config: Mapping[str, Any] | None = None,
        max_records: int = 10000,
    ) -> None:
        self.config = dict(config or {})
        mon_cfg = self.config.get("monitoring", {})
        self.max_records = int(mon_cfg.get("max_records", max_records))
        self.enabled = bool(mon_cfg.get("enabled", True))

        alert_cfg = mon_cfg.get("alerts", {})
        self.reliability_degraded_threshold = float(alert_cfg.get("reliability_degraded_threshold", 0.50))
        self.drift_active_threshold = float(alert_cfg.get("drift_active_threshold", 0.10))

        self.registry = registry if registry is not None else ModelRegistry()

        # Bounded recent history buffer (deque with maxlen)
        self._history: collections.deque[MonitoringRecord] = collections.deque(maxlen=self.max_records)

        # Global cumulative streaming statistics (never reset on deque overflow)
        self._total_observations: int = 0
        self._total_decisions: int = 0
        self._routing_counts: dict[str, int] = {"EDGE": 0, "CLOUD": 0, "HYBRID": 0}
        self._switch_count: int = 0
        self._last_action: str | None = None
        self._last_policy: str | None = None

        # Hybrid execution tracking
        self._hybrid_executions: int = 0
        self._hybrid_fallbacks: int = 0
        self._hybrid_successes: int = 0
        self._hybrid_failures: int = 0

        # Model execution tracking
        self._total_executions: int = 0
        self._successful_executions: int = 0
        self._failed_executions: int = 0
        self._edge_executions: int = 0
        self._edge_failures: int = 0
        self._cloud_executions: int = 0
        self._cloud_failures: int = 0

        # Streaming numeric statistics
        self._stat_reliability = StreamStatistics()
        self._stat_drift_severity = StreamStatistics()
        self._stat_edge_latency = StreamStatistics()
        self._stat_cloud_latency = StreamStatistics()
        self._stat_hybrid_latency = StreamStatistics()

        # Current observed state
        self._current_reliability: float | None = None
        self._current_drift_severity: float | None = None
        self._current_drift_detected: bool = False
        self._current_is_persistent: bool = False
        self._current_raw_severity: float | None = None

    def observe_step(
        self,
        observation_index: int,
        execution_result: ExecutionResult | None = None,
        decision_result: DecisionResult | None = None,
        reliability_score: ReliabilityScore | None = None,
        drift_status: Mapping[str, Any] | None = None,
        controller_policy: str | None = None,
        timestamp: float | None = None,
        extra_alerts: Sequence[str] | None = None,
    ) -> MonitoringRecord:
        """Causally record the observed system state at observation index t.

        Consumes outputs from Phases 3, 4, 5, and 6. Enforces strict causal isolation.
        """
        obs_idx = int(observation_index)
        t_stamp = float(timestamp) if timestamp is not None else time.time()

        # 1. Resolve Reliability (Phase 4)
        r_val: float | None = None
        c_val: float | None = None
        e_val: float | None = None
        d_val: float | None = None
        q_val: float | None = None

        if reliability_score is not None:
            r_val = float(reliability_score.reliability)
            if hasattr(reliability_score, "inputs"):
                c_val = getattr(reliability_score.inputs, "confidence", None)
                e_val = getattr(reliability_score.inputs, "error", None)
                d_val = getattr(reliability_score.inputs, "drift", None)
                q_val = getattr(reliability_score.inputs, "quality", None)
        elif decision_result is not None and decision_result.reliability is not None:
            r_val = float(decision_result.reliability)
        elif execution_result is not None and execution_result.decision is not None:
            r_val = float(execution_result.decision.reliability)

        # 2. Resolve Decision (Phase 5)
        dec = decision_result
        if dec is None and execution_result is not None:
            dec = execution_result.decision

        action_str: str | None = None
        prev_action_str: str | None = None
        reason_str: str | None = None

        if dec is not None:
            action_str = dec.selected_action.value if hasattr(dec.selected_action, "value") else str(dec.selected_action)
            if dec.previous_action is not None:
                prev_action_str = (
                    dec.previous_action.value
                    if hasattr(dec.previous_action, "value")
                    else str(dec.previous_action)
                )
            reason_str = dec.decision_reason
        elif execution_result is not None and execution_result.action is not None:
            action_str = (
                execution_result.action.value
                if hasattr(execution_result.action, "value")
                else str(execution_result.action)
            )

        # 3. Resolve Execution (Phase 6)
        pred_val: int | None = None
        model_used: str | None = None
        exec_status: str | None = None
        fallback_occurred: bool = False
        t_edge: float | None = None
        t_cloud: float | None = None
        t_hybrid: float | None = None
        error_msg: str | None = None

        if execution_result is not None:
            pred_val = execution_result.prediction
            model_used = execution_result.model_used
            if hasattr(execution_result, "status") and hasattr(execution_result.status, "value"):
                exec_status = execution_result.status.value
            elif hasattr(execution_result, "status") and execution_result.status is not None:
                exec_status = str(execution_result.status)
            else:
                exec_status = "SUCCESS" if execution_result.success else "FAILED"

            fallback_occurred = bool(execution_result.cloud_fallback)
            t_edge = getattr(execution_result, "edge_latency_s", None)
            t_cloud = getattr(execution_result, "cloud_latency_s", None)
            t_hybrid = getattr(execution_result, "hybrid_latency_s", None)
            error_msg = getattr(execution_result, "error", None)

            # If inference_latency_s is provided but specific latencies are None
            if t_edge is None and t_cloud is None and t_hybrid is None:
                lat = getattr(execution_result, "inference_latency_s", None)
                if lat is not None:
                    if action_str == "EDGE":
                        t_edge = lat
                    elif action_str == "CLOUD":
                        t_cloud = lat
                    elif action_str == "HYBRID":
                        t_hybrid = lat

        # 4. Resolve Drift (Phase 3)
        drift_det: bool = False
        drift_pers: bool = False
        raw_sev: float | None = None
        smooth_sev: float | None = None

        if drift_status is not None:
            drift_det = bool(drift_status.get("drift_detected", False))
            drift_pers = bool(drift_status.get("is_persistent", False))
            raw_sev = drift_status.get("raw_severity")
            smooth_sev = drift_status.get("smoothed_severity")
            if d_val is None and smooth_sev is not None:
                d_val = smooth_sev
        elif d_val is not None:
            smooth_sev = d_val

        # 5. Resolve Model Version
        model_version: str | None = None
        if model_used and self.registry.has_model(model_used):
            model_version = self.registry.get_metadata(model_used).model_version
        elif action_str == "EDGE" and self.registry.has_model("edge"):
            model_version = self.registry.get_metadata("edge").model_version
        elif action_str == "CLOUD" and self.registry.has_model("cloud"):
            model_version = self.registry.get_metadata("cloud").model_version

        # 6. Policy tracking
        policy_str = controller_policy or self._last_policy
        if controller_policy:
            self._last_policy = controller_policy

        # 7. Evaluate Informational Alerts (strictly non-actionable)
        alerts_list: list[str] = list(extra_alerts or [])
        if r_val is not None and r_val < self.reliability_degraded_threshold:
            alerts_list.append("reliability_degraded")
        if drift_det or drift_pers or (smooth_sev is not None and smooth_sev > self.drift_active_threshold):
            alerts_list.append("drift_active")
        if exec_status == "FAILED" or (execution_result is not None and not execution_result.success):
            alerts_list.append("execution_failure_detected")
        if fallback_occurred:
            alerts_list.append("cloud_fallback_active")

        # 8. Update Global Streaming Counters (Independent of Bounded History)
        self._total_observations += 1
        if action_str in self._routing_counts:
            self._total_decisions += 1
            self._routing_counts[action_str] += 1
            if self._last_action is not None and action_str != self._last_action:
                self._switch_count += 1
            self._last_action = action_str

        # Update streaming statistics
        if r_val is not None:
            self._stat_reliability.update(r_val)
            self._current_reliability = r_val
        if smooth_sev is not None:
            self._stat_drift_severity.update(smooth_sev)
            self._current_drift_severity = smooth_sev
        self._current_drift_detected = drift_det
        self._current_is_persistent = drift_pers
        self._current_raw_severity = raw_sev

        if t_edge is not None:
            self._stat_edge_latency.update(t_edge)
        if t_cloud is not None:
            self._stat_cloud_latency.update(t_cloud)
        if t_hybrid is not None:
            self._stat_hybrid_latency.update(t_hybrid)

        # Update execution tracking
        if execution_result is not None:
            self._total_executions += 1
            if execution_result.success:
                self._successful_executions += 1
            else:
                self._failed_executions += 1

            if action_str == "EDGE":
                self._edge_executions += 1
                if not execution_result.success:
                    self._edge_failures += 1
                if self.registry.has_model("edge"):
                    self.registry.record_execution(
                        "edge",
                        success=execution_result.success,
                        latency_s=t_edge,
                        status=exec_status,
                        error=error_msg,
                    )

            elif action_str == "CLOUD":
                self._cloud_executions += 1
                if not execution_result.success:
                    self._cloud_failures += 1
                if self.registry.has_model("cloud"):
                    self.registry.record_execution(
                        "cloud",
                        success=execution_result.success,
                        latency_s=t_cloud,
                        status=exec_status,
                        error=error_msg,
                    )

            elif action_str == "HYBRID":
                self._hybrid_executions += 1
                if fallback_occurred:
                    self._hybrid_fallbacks += 1
                if execution_result.success:
                    self._hybrid_successes += 1
                else:
                    self._hybrid_failures += 1

                # Update registry for models actually invoked in hybrid
                if self.registry.has_model("edge"):
                    self.registry.record_execution("edge", success=True, latency_s=t_edge, status="SUCCESS")
                if fallback_occurred and self.registry.has_model("cloud"):
                    self.registry.record_execution(
                        "cloud",
                        success=execution_result.success,
                        latency_s=t_cloud,
                        status=exec_status,
                        error=error_msg,
                    )

        # 9. Construct MonitoringRecord and store in bounded buffer
        record = MonitoringRecord(
            observation_index=obs_idx,
            timestamp=t_stamp,
            reliability=r_val,
            confidence=c_val,
            error_ema=e_val,
            drift_severity=smooth_sev,
            quality=q_val,
            selected_action=action_str,
            previous_action=prev_action_str,
            decision_reason=reason_str,
            prediction=pred_val,
            model_used=model_used,
            execution_status=exec_status,
            cloud_fallback=fallback_occurred,
            edge_latency_s=t_edge,
            cloud_latency_s=t_cloud,
            hybrid_latency_s=t_hybrid,
            model_version=model_version,
            drift_detected=drift_det,
            is_persistent=drift_pers,
            raw_severity=raw_sev,
            smoothed_severity=smooth_sev,
            controller_policy=policy_str,
            alerts=tuple(alerts_list),
        )

        self._history.append(record)
        return record

    def get_snapshot(self) -> MonitoringSnapshot:
        """Generate an aggregated snapshot of current system-wide observability state."""
        # Calculate routing percentages
        tot_d = self._total_decisions
        if tot_d > 0:
            dist = {
                "EDGE": (self._routing_counts["EDGE"] / tot_d) * 100.0,
                "CLOUD": (self._routing_counts["CLOUD"] / tot_d) * 100.0,
                "HYBRID": (self._routing_counts["HYBRID"] / tot_d) * 100.0,
            }
        else:
            dist = {"EDGE": 0.0, "CLOUD": 0.0, "HYBRID": 0.0}

        # Calculate hybrid fallback rate
        if self._hybrid_executions > 0:
            fb_rate = self._hybrid_fallbacks / self._hybrid_executions
        else:
            fb_rate = 0.0

        # Calculate execution success rate
        tot_ex = self._total_executions
        if tot_ex > 0:
            succ_rate = self._successful_executions / tot_ex
        else:
            succ_rate = 1.0

        active_alerts: list[str] = []
        if self._current_reliability is not None and self._current_reliability < self.reliability_degraded_threshold:
            active_alerts.append("reliability_degraded")
        if self._current_drift_detected or self._current_is_persistent:
            active_alerts.append("drift_active")
        if self._failed_executions > 0:
            active_alerts.append("execution_failures_recorded")

        return MonitoringSnapshot(
            timestamp=time.time(),
            total_observations=self._total_observations,
            current_reliability=self._current_reliability,
            current_drift_severity=self._current_drift_severity,
            current_action=self._last_action,
            current_policy=self._last_policy,
            routing_counts={
                "EDGE": self._routing_counts["EDGE"],
                "CLOUD": self._routing_counts["CLOUD"],
                "HYBRID": self._routing_counts["HYBRID"],
                "total": tot_d,
                "switches": self._switch_count,
            },
            routing_distribution=dist,
            hybrid_stats={
                "executions": self._hybrid_executions,
                "fallbacks": self._hybrid_fallbacks,
                "fallback_rate": fb_rate,
                "successes": self._hybrid_successes,
                "failures": self._hybrid_failures,
            },
            execution_stats={
                "total": tot_ex,
                "successful": self._successful_executions,
                "failed": self._failed_executions,
                "success_rate": succ_rate,
                "edge_executions": self._edge_executions,
                "edge_failures": self._edge_failures,
                "cloud_executions": self._cloud_executions,
                "cloud_failures": self._cloud_failures,
            },
            latency_stats={
                "edge": self._stat_edge_latency.to_dict(),
                "cloud": self._stat_cloud_latency.to_dict(),
                "hybrid": self._stat_hybrid_latency.to_dict(),
            },
            drift_stats={
                "current_detected": self._current_drift_detected,
                "current_persistent": self._current_is_persistent,
                "raw_severity": self._current_raw_severity,
                "smoothed_severity": self._current_drift_severity,
                "severity_stats": self._stat_drift_severity.to_dict(),
            },
            reliability_stats=self._stat_reliability.to_dict(),
            model_health=self.registry.get_health_summary(),
            active_alerts=active_alerts,
        )

    def get_records(self, limit: int | None = None) -> list[MonitoringRecord]:
        """Return historical monitoring records up to the specified limit (bounded by max_records)."""
        if limit is None or limit >= len(self._history):
            return list(self._history)
        return list(self._history)[-int(limit):]

    def get_records_dataframe(self) -> pd.DataFrame:
        """Export bounded history as a pandas DataFrame with a fixed, stable schema for Phase 10."""
        schema_columns = [
            "observation_index",
            "timestamp",
            "reliability",
            "confidence",
            "error_ema",
            "drift_severity",
            "quality",
            "selected_action",
            "previous_action",
            "decision_reason",
            "prediction",
            "model_used",
            "execution_status",
            "cloud_fallback",
            "edge_latency_s",
            "cloud_latency_s",
            "hybrid_latency_s",
            "model_version",
            "drift_detected",
            "is_persistent",
            "raw_severity",
            "smoothed_severity",
            "controller_policy",
        ]

        if not self._history:
            return pd.DataFrame(columns=schema_columns)

        data = [rec.to_dict() for rec in self._history]
        df = pd.DataFrame(data)
        # Ensure all columns in schema are present
        for col in schema_columns:
            if col not in df.columns:
                df[col] = None
        return df[schema_columns]

    def reset(self) -> None:
        """Reset historical records and global telemetry counters cleanly."""
        self._history.clear()
        self._total_observations = 0
        self._total_decisions = 0
        self._routing_counts = {"EDGE": 0, "CLOUD": 0, "HYBRID": 0}
        self._switch_count = 0
        self._last_action = None
        self._last_policy = None
        self._hybrid_executions = 0
        self._hybrid_fallbacks = 0
        self._hybrid_successes = 0
        self._hybrid_failures = 0
        self._total_executions = 0
        self._successful_executions = 0
        self._failed_executions = 0
        self._edge_executions = 0
        self._edge_failures = 0
        self._cloud_executions = 0
        self._cloud_failures = 0
        self._stat_reliability = StreamStatistics()
        self._stat_drift_severity = StreamStatistics()
        self._stat_edge_latency = StreamStatistics()
        self._stat_cloud_latency = StreamStatistics()
        self._stat_hybrid_latency = StreamStatistics()
        self._current_reliability = None
        self._current_drift_severity = None
        self._current_drift_detected = False
        self._current_is_persistent = False
        self._current_raw_severity = None
        self.registry.reset_metrics()


# Public alias
SystemMonitor = DRAECMonitor
