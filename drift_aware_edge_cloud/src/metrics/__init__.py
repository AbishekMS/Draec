"""Drift-Aware Adaptive Edge-Cloud-Hybrid Orchestration for Non-Stationary IoT.

Module   : src/metrics/__init__.py
Phase    : Phase 10
Status   : IMPLEMENTED

Public API for Phase 10 evaluation and benchmarking metrics.
"""

from __future__ import annotations

from src.metrics.decision import compute_routing_metrics
from src.metrics.drift import compute_drift_metrics
from src.metrics.evaluation import (
    Phase10Evaluator,
    compute_confidence_interval,
    compute_feature_scalar,
    find_representative_window,
)
from src.metrics.prediction import compute_classification_metrics, compute_pre_post_metrics
from src.metrics.system import (
    compute_execution_reliability,
    compute_latency_summary,
    compute_network_metrics,
    get_metric_completeness_matrix,
    get_unmeasured_system_status,
)

__all__ = [
    "Phase10Evaluator",
    "find_representative_window",
    "compute_feature_scalar",
    "compute_confidence_interval",
    "compute_classification_metrics",
    "compute_pre_post_metrics",
    "compute_drift_metrics",
    "compute_routing_metrics",
    "compute_latency_summary",
    "compute_execution_reliability",
    "compute_network_metrics",
    "get_unmeasured_system_status",
    "get_metric_completeness_matrix",
]
