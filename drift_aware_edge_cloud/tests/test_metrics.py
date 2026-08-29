"""Drift-Aware Adaptive Edge-Cloud-Hybrid Orchestration for Non-Stationary IoT.

Module   : tests/test_metrics.py
Phase    : Phase 10
Purpose  : Test suite for Phase 10 metrics suite and statistical evaluation.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.metrics.decision import compute_routing_metrics
from src.metrics.drift import compute_drift_metrics
from src.metrics.evaluation import compute_confidence_interval
from src.metrics.prediction import compute_classification_metrics, compute_pre_post_metrics
from src.metrics.system import (
    compute_execution_reliability,
    compute_latency_summary,
    compute_network_metrics,
    get_metric_completeness_matrix,
    get_unmeasured_system_status,
)


def test_classification_metrics_basic():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]
    res = compute_classification_metrics(y_true, y_pred)
    assert res["accuracy"] == 1.0
    assert res["f1"] == 1.0
    assert res["mcc"] == 1.0
    assert res["sample_count"] == 4
    assert res["confusion_matrix"] == {"tp": 2, "fp": 0, "fn": 0, "tn": 2}


def test_classification_metrics_imperfect():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 0, 1]
    res = compute_classification_metrics(y_true, y_pred)
    assert res["accuracy"] == 0.5
    assert 0.0 <= res["f1"] <= 1.0
    assert -1.0 <= res["mcc"] <= 1.0


def test_classification_metrics_empty():
    res = compute_classification_metrics([], [])
    assert res["sample_count"] == 0
    assert res["accuracy"] == 0.0
    assert res["f1"] == 0.0


def test_pre_post_metrics():
    y_true = [0, 1, 0, 1, 0, 1]
    y_pred = [0, 1, 0, 0, 0, 0]  # Perfect pre (0..2), poor post (3..5)
    res = compute_pre_post_metrics(y_true, y_pred, drift_onset_index=3)
    assert res["pre_drift_samples"] == 3
    assert res["post_drift_samples"] == 3
    assert res["pre_drift"]["accuracy"] == 1.0
    assert res["post_drift"]["accuracy"] < 1.0
    assert "delta_f1" in res["delta"]
    assert "pct_change_f1" in res["percentage_change"]


def test_drift_metrics_with_onset():
    detections = [450, 520, 560]
    res = compute_drift_metrics(detections, drift_onset_index=500, total_steps=1000)
    assert res["drift_onset"] == 500
    assert res["first_detection_point"] == 520
    assert res["detection_delay"] == 20
    assert res["total_alarms"] == 3
    assert res["false_alarms_pre_drift"] == 1
    assert res["post_drift_alarms"] == 2
    assert res["detection_status"] == "DETECTED"


def test_drift_metrics_no_onset():
    detections = [100, 200]
    res = compute_drift_metrics(detections, drift_onset_index=None, total_steps=1000)
    assert res["drift_scenario"] == "no_drift"
    assert res["detection_status"] == "NO_DRIFT"
    assert res["false_alarms_pre_drift"] == 2


def test_routing_metrics():
    actions = ["EDGE", "EDGE", "HYBRID", "CLOUD", "CLOUD"]
    res = compute_routing_metrics(actions, switch_count=2, hybrid_fallbacks=1)
    assert res["total_decisions"] == 5
    assert res["edge_count"] == 2
    assert res["hybrid_count"] == 1
    assert res["cloud_count"] == 2
    assert res["edge_percentage"] == 40.0
    assert res["hybrid_percentage"] == 20.0
    assert res["cloud_percentage"] == 40.0
    assert res["offloading_ratio"] == 40.0
    assert res["switch_count"] == 2
    assert res["hybrid_fallback_rate"] == 100.0
    assert res["hybrid_status"] == "OBSERVED"


def test_routing_metrics_zero_hybrid():
    actions = ["EDGE", "EDGE", "EDGE"]
    res = compute_routing_metrics(actions)
    assert res["hybrid_count"] == 0
    assert res["hybrid_fallback_rate"] == 0.0
    assert res["hybrid_status"] == "NOT OBSERVED"


def test_latency_summary():
    lats = [0.010, 0.020, 0.030]
    res = compute_latency_summary(lats)
    assert res["count"] == 3
    assert abs(res["mean_ms"] - 20.0) < 1e-4
    assert abs(res["median_ms"] - 20.0) < 1e-4
    assert abs(res["max_ms"] - 30.0) < 1e-4


def test_latency_summary_empty():
    res = compute_latency_summary([])
    assert res["count"] == 0
    assert res["mean_ms"] == 0.0


def test_execution_reliability():
    res = compute_execution_reliability(
        total_executions=100,
        successful_executions=96,
        edge_failures=1,
        cloud_failures=1,
        packet_loss_failures=2,
    )
    assert res["total_executions"] == 100
    assert res["successful_executions"] == 96
    assert res["failed_executions"] == 4
    assert res["success_rate"] == 0.96
    assert res["failure_rate"] == 0.04
    assert res["packet_loss_failures"] == 2


def test_network_metrics():
    res = compute_network_metrics(
        total_transmissions=50,
        delivered_transmissions=45,
        packet_loss_count=5,
        latencies_s=[0.025] * 45,
        total_bytes_transmitted=10240,
    )
    assert res["total_transmissions"] == 50
    assert res["delivered_transmissions"] == 45
    assert res["delivery_rate"] == 0.90
    assert res["packet_loss_rate"] == 0.10
    assert "10,240 bytes" in res["bandwidth_usage"]
    assert "SIMULATED NETWORK ONLY" in res["network_simulation_note"]


def test_unmeasured_system_status():
    status = get_unmeasured_system_status()
    assert status["cpu_utilization"] == "NOT MEASURED"
    assert status["ram_utilization"] == "NOT MEASURED"
    assert status["energy_consumption"] == "NOT MEASURED"
    assert "NOT MEASURED" in status["physical_hardware_deployment"]
    assert status["bandwidth"] == "NOT MEASURED"
    assert status["formal_constraint_satisfaction"] == "NOT IMPLEMENTED / NOT MEASURED"


def test_metric_completeness_matrix():
    matrix = get_metric_completeness_matrix({
        "drift_detected": True,
        "hybrid_fallback_observed": False,
    })
    assert matrix["Accuracy"] == "MEASURED"
    assert matrix["Drift delay"] == "MEASURED"
    assert matrix["Hybrid fallback"] == "NOT OBSERVED"
    assert matrix["CPU"] == "NOT MEASURED"
    assert matrix["RAM"] == "NOT MEASURED"


def test_confidence_interval_estimation():
    vals = [10.0, 10.0, 10.0]
    mean, std, ci_l, ci_u = compute_confidence_interval(vals)
    assert mean == 10.0
    assert std == 0.0
    assert ci_l == 10.0
    assert ci_u == 10.0

    noisy = [1.0, 2.0, 3.0, 4.0, 5.0]
    mean_n, std_n, ci_l_n, ci_u_n = compute_confidence_interval(noisy)
    assert mean_n == 3.0
    assert ci_l_n < mean_n < ci_u_n
