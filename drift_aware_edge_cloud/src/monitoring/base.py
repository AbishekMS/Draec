"""Drift-Aware Adaptive Edge-Cloud-Hybrid Orchestration for Non-Stationary IoT.

Module   : src/monitoring/base.py
Phase    : Phase 7
Status   : IMPLEMENTED

Base contracts, datastructures, and telemetry records for Phase 7
Model Management & Monitoring.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence


class ModelHealthStatus(str, Enum):
    """Observational health status of registered models.

    These states are observational metadata only and never trigger automated
    retraining, parameter replacement, or adaptation.
    """

    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"
    INACTIVE = "INACTIVE"

    @classmethod
    def from_str(cls, val: str | ModelHealthStatus) -> ModelHealthStatus:
        if isinstance(val, cls):
            return val
        norm = str(val).strip().upper()
        for member in cls:
            if member.value == norm:
                return member
        raise ValueError(f"Unknown ModelHealthStatus '{val}'. Valid: {[m.value for m in cls]}")


@dataclass
class ModelMetadata:
    """Metadata tracking the registration, feature contract, and lifecycle state of a model."""

    model_id: str
    model_name: str
    model_type: str
    execution_location: str
    model_version: str = "1.0.0"
    status: ModelHealthStatus = ModelHealthStatus.HEALTHY
    active: bool = True
    created_at: float = field(default_factory=time.time)
    n_features: int | None = None
    feature_names: tuple[str, ...] | None = None
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    last_latency_s: float | None = None
    last_execution_status: str | None = None
    last_error: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, ModelHealthStatus):
            self.status = ModelHealthStatus.from_str(self.status)
        loc = self.execution_location.strip().lower()
        if loc not in ("edge", "cloud"):
            raise ValueError(f"execution_location must be 'edge' or 'cloud', got '{self.execution_location}'")
        self.execution_location = loc

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "model_type": self.model_type,
            "execution_location": self.execution_location,
            "model_version": self.model_version,
            "status": self.status.value,
            "active": self.active,
            "created_at": self.created_at,
            "n_features": self.n_features,
            "feature_names": list(self.feature_names) if self.feature_names is not None else None,
            "total_executions": self.total_executions,
            "successful_executions": self.successful_executions,
            "failed_executions": self.failed_executions,
            "last_latency_s": self.last_latency_s,
            "last_execution_status": self.last_execution_status,
            "last_error": self.last_error,
        }


@dataclass
class StreamStatistics:
    """Incremental running statistics for a numeric stream without unbounded history storage."""

    count: int = 0
    min_val: float = float("inf")
    max_val: float = float("-inf")
    sum_val: float = 0.0
    current: float | None = None

    def update(self, val: float | None) -> None:
        if val is None or not math.isfinite(val):
            return
        self.count += 1
        self.sum_val += float(val)
        if val < self.min_val:
            self.min_val = float(val)
        if val > self.max_val:
            self.max_val = float(val)
        self.current = float(val)

    @property
    def mean(self) -> float | None:
        if self.count == 0:
            return None
        return self.sum_val / self.count

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "mean": self.mean,
            "min": self.min_val if self.count > 0 else None,
            "max": self.max_val if self.count > 0 else None,
            "current": self.current,
        }


@dataclass(frozen=True)
class MonitoringRecord:
    """Structured telemetry record representing one observation step in the DRAEC pipeline.

    Preserves full data readiness for downstream Phase 10 evaluations while enforcing
    strict causal isolation (no Target, no ground truth, no future rows).
    """

    observation_index: int
    timestamp: float
    reliability: float | None = None
    confidence: float | None = None
    error_ema: float | None = None
    drift_severity: float | None = None
    quality: float | None = None
    selected_action: str | None = None
    previous_action: str | None = None
    decision_reason: str | None = None
    prediction: int | None = None
    model_used: str | None = None
    execution_status: str | None = None
    cloud_fallback: bool = False
    edge_latency_s: float | None = None
    cloud_latency_s: float | None = None
    hybrid_latency_s: float | None = None
    network_latency_s: float | None = None
    packet_lost: bool = False
    model_version: str | None = None
    drift_detected: bool = False
    is_persistent: bool = False
    raw_severity: float | None = None
    smoothed_severity: float | None = None
    controller_policy: str | None = None
    alerts: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_index": self.observation_index,
            "timestamp": self.timestamp,
            "reliability": self.reliability,
            "confidence": self.confidence,
            "error_ema": self.error_ema,
            "drift_severity": self.drift_severity,
            "quality": self.quality,
            "selected_action": self.selected_action,
            "previous_action": self.previous_action,
            "decision_reason": self.decision_reason,
            "prediction": self.prediction,
            "model_used": self.model_used,
            "execution_status": self.execution_status,
            "cloud_fallback": self.cloud_fallback,
            "edge_latency_s": self.edge_latency_s,
            "cloud_latency_s": self.cloud_latency_s,
            "hybrid_latency_s": self.hybrid_latency_s,
            "network_latency_s": self.network_latency_s,
            "packet_lost": self.packet_lost,
            "model_version": self.model_version,
            "drift_detected": self.drift_detected,
            "is_persistent": self.is_persistent,
            "raw_severity": self.raw_severity,
            "smoothed_severity": self.smoothed_severity,
            "controller_policy": self.controller_policy,
            "alerts": list(self.alerts),
        }


@dataclass
class MonitoringSnapshot:
    """Comprehensive snapshot of current system-wide observability and health."""

    timestamp: float
    total_observations: int
    current_reliability: float | None
    current_drift_severity: float | None
    current_action: str | None
    current_policy: str | None
    routing_counts: dict[str, int]
    routing_distribution: dict[str, float]
    hybrid_stats: dict[str, Any]
    execution_stats: dict[str, Any]
    latency_stats: dict[str, dict[str, Any]]
    drift_stats: dict[str, Any]
    reliability_stats: dict[str, Any]
    model_health: dict[str, str]
    active_alerts: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "total_observations": self.total_observations,
            "current_reliability": self.current_reliability,
            "current_drift_severity": self.current_drift_severity,
            "current_action": self.current_action,
            "current_policy": self.current_policy,
            "routing_counts": dict(self.routing_counts),
            "routing_distribution": dict(self.routing_distribution),
            "hybrid_stats": dict(self.hybrid_stats),
            "execution_stats": dict(self.execution_stats),
            "latency_stats": {k: dict(v) for k, v in self.latency_stats.items()},
            "drift_stats": dict(self.drift_stats),
            "reliability_stats": dict(self.reliability_stats),
            "model_health": dict(self.model_health),
            "active_alerts": list(self.active_alerts),
        }
