"""Drift-Aware Adaptive Edge-Cloud-Hybrid Orchestration for Non-Stationary IoT.

Module   : src/deployment/base.py
Phase    : Phase 8
Status   : IMPLEMENTED

Base contracts, datastructures, and results for Phase 8
Edge-Cloud Deployment & Network Execution Layer.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from src.decision.base import DecisionAction, DecisionResult, ExecutionResult, ExecutionStatus


class TransmissionStatus(str, Enum):
    """Status of a network transmission attempt between Edge and Cloud."""

    DELIVERED = "DELIVERED"
    PACKET_LOSS = "PACKET_LOSS"
    DISCONNECTED = "DISCONNECTED"
    TIMEOUT = "TIMEOUT"

    @classmethod
    def from_str(cls, val: str | TransmissionStatus) -> TransmissionStatus:
        if isinstance(val, cls):
            return val
        s = str(val).strip().upper()
        for member in cls:
            if member.value == s:
                return member
        raise ValueError(f"Unknown TransmissionStatus '{val}'. Valid: {[m.value for m in cls]}")


class RuntimeState(str, Enum):
    """Availability state of an Edge device or Cloud service."""

    ONLINE = "ONLINE"
    OFFLINE = "OFFLINE"
    DEGRADED = "DEGRADED"

    @classmethod
    def from_str(cls, val: str | RuntimeState) -> RuntimeState:
        if isinstance(val, cls):
            return val
        s = str(val).strip().upper()
        for member in cls:
            if member.value == s:
                return member
        raise ValueError(f"Unknown RuntimeState '{val}'. Valid: {[m.value for m in cls]}")


@dataclass(frozen=True)
class NetworkPacket:
    """Represents a simulated data packet transmitted across Edge-Cloud boundary."""

    payload: Any
    source: str
    destination: str
    size_bytes: int = 0
    sequence_number: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class TransmissionResult:
    """Outcome of a simulated network transmission."""

    status: TransmissionStatus
    success: bool
    latency_s: float
    packet_lost: bool
    error: str | None = None
    bytes_transferred: int = 0


@dataclass(frozen=True)
class DeploymentExecutionResult:
    """Comprehensive execution outcome from the deployment environment.

    Extends Phase 6 execution contracts with simulated network metrics and explicit
    failure provenance. Never fabricates predictions on failure.
    """

    action: DecisionAction
    prediction: int | None
    probabilities: dict[int, float] | None
    model_used: str
    success: bool
    status: ExecutionStatus
    edge_latency_s: float | None = None
    cloud_latency_s: float | None = None
    network_latency_s: float | None = None
    hybrid_latency_s: float | None = None
    total_latency_s: float = 0.0
    cloud_fallback: bool = False
    packet_lost: bool = False
    error: str | None = None
    decision: DecisionResult | None = None
    observation_index: int | None = None
    timestamp: float = field(default_factory=time.time)

    def to_execution_result(self) -> ExecutionResult:
        """Convert to a standard Phase 6 ExecutionResult for seamless downstream compatibility."""
        dec = self.decision
        if dec is None:
            # Construct a minimal synthetic DecisionResult if none was provided
            dec = DecisionResult(
                selected_action=self.action,
                reliability=1.0,
                previous_action=None,
                decision_reason="Deployment execution",
                switch_count=0,
                observation_index=self.observation_index,
                timestamp=self.timestamp,
            )

        inf_lat = self.total_latency_s if self.total_latency_s > 0.0 else (self.edge_latency_s or self.cloud_latency_s or 0.0)

        return ExecutionResult(
            decision=dec,
            action=self.action,
            prediction=self.prediction,
            probabilities=self.probabilities,
            model_used=self.model_used,
            inference_latency_s=inf_lat,
            cloud_fallback=self.cloud_fallback,
            success=self.success,
            status=self.status,
            edge_latency_s=self.edge_latency_s,
            cloud_latency_s=self.cloud_latency_s,
            hybrid_latency_s=self.hybrid_latency_s,
            network_latency_s=self.network_latency_s,
            error=self.error,
            observation_index=self.observation_index,
            timestamp=self.timestamp,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "prediction": self.prediction,
            "probabilities": self.probabilities,
            "model_used": self.model_used,
            "success": self.success,
            "status": self.status.value,
            "edge_latency_s": self.edge_latency_s,
            "cloud_latency_s": self.cloud_latency_s,
            "network_latency_s": self.network_latency_s,
            "hybrid_latency_s": self.hybrid_latency_s,
            "total_latency_s": self.total_latency_s,
            "cloud_fallback": self.cloud_fallback,
            "packet_lost": self.packet_lost,
            "error": self.error,
            "observation_index": self.observation_index,
            "timestamp": self.timestamp,
        }
