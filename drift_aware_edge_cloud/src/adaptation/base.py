"""Drift-Aware Adaptive Edge-Cloud-Hybrid Orchestration for Non-Stationary IoT.

Module   : src/adaptation/base.py
Phase    : Phase 9
Status   : IMPLEMENTED

Base contracts, enums, and data structures for Phase 9 Model Adaptation & Retraining.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class AdaptationState(str, Enum):
    """Lifecycle state of the adaptation and retraining mechanism."""

    IDLE = "IDLE"
    ELIGIBLE = "ELIGIBLE"
    TRAINING = "TRAINING"
    VALIDATING = "VALIDATING"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    COOLDOWN = "COOLDOWN"
    FAILED = "FAILED"
    ROLLBACK = "ROLLBACK"

    @classmethod
    def from_str(cls, val: str | AdaptationState) -> AdaptationState:
        if isinstance(val, cls):
            return val
        s = str(val).strip().upper()
        for member in cls:
            if member.value == s:
                return member
        raise ValueError(f"Unknown AdaptationState '{val}'. Valid: {[m.value for m in cls]}")


@dataclass(frozen=True)
class FeedbackRecord:
    """Represents a single observation prediction awaiting or paired with delayed ground truth.

    Strictly tracks origin stream source to enforce adaptation/evaluation separation.
    """

    observation_index: int
    features: Any
    prediction: int | None
    probabilities: dict[int, float] | None
    model_version: str
    label: int | None = None
    arrival_index: int | None = None
    is_labeled: bool = False
    source: str = "adaptation"  # 'adaptation', 'train1', etc. NEVER 'test1'
    timestamp: float = field(default_factory=time.time)

    def with_label(self, label: int, arrival_index: int) -> FeedbackRecord:
        """Return a new FeedbackRecord with label and causal arrival index bound."""
        if arrival_index < self.observation_index:
            raise ValueError(
                f"Acausal feedback arrival: arrival_index {arrival_index} < observation_index {self.observation_index}"
            )
        return FeedbackRecord(
            observation_index=self.observation_index,
            features=self.features,
            prediction=self.prediction,
            probabilities=self.probabilities,
            model_version=self.model_version,
            label=int(label),
            arrival_index=int(arrival_index),
            is_labeled=True,
            source=self.source,
            timestamp=self.timestamp,
        )


@dataclass(frozen=True)
class ValidationResult:
    """Result of candidate model validation against clean validation data."""

    candidate_valid: bool
    metric_name: str
    candidate_metric: float
    active_metric: float
    metric_delta: float
    status: str
    reason: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_valid": self.candidate_valid,
            "metric_name": self.metric_name,
            "candidate_metric": self.candidate_metric,
            "active_metric": self.active_metric,
            "metric_delta": self.metric_delta,
            "status": self.status,
            "reason": self.reason,
            "details": self.details,
        }


@dataclass(frozen=True)
class ModelVersionRecord:
    """Historical audit record for a candidate or active model version."""

    version: str
    parent_version: str | None
    created_at: float = field(default_factory=time.time)
    status: str = "CANDIDATE"  # 'CANDIDATE', 'ACTIVE', 'REJECTED', 'SUPERSEDED'
    samples_used: int = 0
    baseline_samples_used: int = 0
    feedback_samples_used: int = 0
    validation_metric: float | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "parent_version": self.parent_version,
            "created_at": self.created_at,
            "status": self.status,
            "samples_used": self.samples_used,
            "baseline_samples_used": self.baseline_samples_used,
            "feedback_samples_used": self.feedback_samples_used,
            "validation_metric": self.validation_metric,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class AdaptationResult:
    """Outcome of an adaptation cycle evaluated by AdaptationManager."""

    state: AdaptationState
    triggered: bool
    candidate_version: str | None
    active_version: str
    cloud_version: str
    edge_version: str
    validation_result: ValidationResult | None = None
    deployment_success: bool = False
    rolled_back: bool = False
    error: str | None = None
    samples_used: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "triggered": self.triggered,
            "candidate_version": self.candidate_version,
            "active_version": self.active_version,
            "cloud_version": self.cloud_version,
            "edge_version": self.edge_version,
            "validation_result": self.validation_result.to_dict() if self.validation_result else None,
            "deployment_success": self.deployment_success,
            "rolled_back": self.rolled_back,
            "error": self.error,
            "samples_used": self.samples_used,
        }
