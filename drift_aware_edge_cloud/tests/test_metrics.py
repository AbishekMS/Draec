"""Drift-Aware Adaptive Edge-Cloud-Hybrid Orchestration for Non-Stationary IoT.

Module   : tests/test_metrics.py
Phase    : Phase 10
Purpose  : Test suite for Phase 10 metrics suite and statistical evaluation.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.drift import ADWINDetector, DriftPersistence, DriftPipeline, DriftSeverity
from src.metrics.decision import compute_routing_metrics
from src.metrics.drift import compute_drift_metrics
from src.metrics.evaluation import (
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
from src.reliability.estimator import ReliabilityEstimator


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
    assert res["failed_transmissions"] == 5
    assert res["delivery_rate"] == 0.90
    assert res["failure_rate"] == 0.10
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


def test_find_representative_window_logic():
    # 100 samples, window size 20, min minority 3 in both halves
    y = np.zeros(100, dtype=int)
    y[[11, 13, 15]] = 1
    y[[21, 23, 25]] = 1

    start = find_representative_window(y, window_size=20, min_minority_count=3)
    assert start == 6
    # Verify both halves satisfy at start
    assert np.sum(y[start : start + 10] == 1) >= 3
    assert np.sum(y[start + 10 : start + 20] == 1) >= 3
    # At start - 1, condition was not satisfied
    assert np.sum(y[start - 1 : start - 1 + 10] == 1) < 3 or np.sum(y[start - 1 + 10 : start - 1 + 20] == 1) < 3

    # Boundary at end (start = len(y) - window_size = 30)
    y2 = np.zeros(50, dtype=int)
    y2[[37, 38, 39, 47, 48, 49]] = 1
    start2 = find_representative_window(y2, window_size=20, min_minority_count=3)
    assert start2 == 30

    # Failure case
    y_empty = np.zeros(50, dtype=int)
    with pytest.raises(ValueError, match="No contiguous window"):
        find_representative_window(y_empty, window_size=20, min_minority_count=1)


def test_regression_prediction_probability_extraction():
    """TEST A: A dictionary {0: 0.8, 1: 0.2} must extract probabilities, NOT dict keys."""
    raw_probs = {0: 0.8, 1: 0.2}
    # Unpacking dict keys would give p0=0, p1=1
    p0 = float(raw_probs.get(0, 0.0))
    p1 = float(raw_probs.get(1, 0.0))
    assert p0 == 0.8
    assert p1 == 0.2
    assert p0 != 0
    assert p1 != 1

    # In ADWINDetector:
    detector = ADWINDetector(monitored_signal="prediction_probability")
    detector.update_from_prediction(raw_probs)
    assert detector.last_signal_value == 0.2  # class 1 prob, NOT 1.0


def test_regression_drift_pipeline_integration_and_severity():
    """TEST B & C: Verify DriftPipeline integration and proper DriftSeverity calculation."""
    detector = ADWINDetector(delta=0.002, clock=32, monitored_signal="prediction_probability")
    persistence = DriftPersistence(consecutive_threshold=3)
    severity_scorer = DriftSeverity(
        formula="relative_shift",
        baseline_mean=0.02,
        max_shift=0.98,
        smoothing_factor=0.8,
    )
    pipeline = DriftPipeline(
        detector=detector,
        persistence=persistence,
        severity=severity_scorer,
    )

    # Observation with normal class-1 prob = 0.02
    status = pipeline.update_from_prediction({0: 0.98, 1: 0.02})
    assert status.monitored_value == 0.02
    assert status.raw_severity == 0.0  # |0.02 - 0.02| / 0.98 == 0.0
    assert status.smoothed_severity == 0.0
    # Must NOT be derived from abs(drift_val - 0.5) * 2.0 (which would be |1.0 - 0.5|*2 = 1.0)
    assert status.smoothed_severity != 1.0


def test_regression_causal_delayed_feedback_reliability():
    """TEST D: Verify that eligible delayed feedback changes E_t with causal timing."""
    rel_est = ReliabilityEstimator(alpha_E=0.8)
    assert rel_est.current_error == 0.0

    # Steps 0 to 14: no feedback eligible yet
    for _ in range(15):
        score = rel_est.update(probs={0: 1.0, 1: 0.0}, quality=[True] * 37)
        assert score.inputs.error == 0.0
        assert rel_est.current_error == 0.0

    # At step 15: feedback from step 0 arrives: model predicted 0, true label was 1 (error = 1.0)
    score15 = rel_est.update(
        probs={0: 1.0, 1: 0.0},
        quality=[True] * 37,
        y_true=1,
        y_pred=0,
    )
    # E_15 = 0.8 * 0.0 + 0.2 * 1.0 = 0.2
    assert score15.inputs.error == pytest.approx(0.2, abs=1e-5)
    assert rel_est.current_error == pytest.approx(0.2, abs=1e-5)

    # At step 16: feedback from step 1 arrives: model predicted 0, true label was 0 (error = 0.0)
    score16 = rel_est.update(
        probs={0: 1.0, 1: 0.0},
        quality=[True] * 37,
        y_true=0,
        y_pred=0,
    )
    # E_16 = 0.8 * 0.2 + 0.2 * 0.0 = 0.16
    assert score16.inputs.error == pytest.approx(0.16, abs=1e-5)
    assert rel_est.current_error == pytest.approx(0.16, abs=1e-5)


def test_feature_scalar_calculation_and_clipping():
    """Verify S(x) = mean(min(|x|, 5.0)) and values > 5.0 or < -5.0 are clipped."""
    # 4 features: 1.0, -3.0, 10.0, -20.0
    # Abs: 1.0, 3.0, 10.0, 20.0
    # Clipped to 5.0: 1.0, 3.0, 5.0, 5.0 -> mean = 14.0 / 4 = 3.5
    x = np.array([1.0, -3.0, 10.0, -20.0])
    s = compute_feature_scalar(x, clip=5.0)
    assert isinstance(s, float)
    assert s == pytest.approx(3.5, abs=1e-6)

    # 2D matrix: 2 rows of 4 features
    X = np.array([
        [1.0, -3.0, 10.0, -20.0],
        [0.0, 0.0, 0.0, 0.0],
    ])
    s_arr = compute_feature_scalar(X, clip=5.0)
    assert isinstance(s_arr, np.ndarray)
    assert len(s_arr) == 2
    assert s_arr[0] == pytest.approx(3.5, abs=1e-6)
    assert s_arr[1] == pytest.approx(0.0, abs=1e-6)


def test_feature_scalar_baseline_calculation():
    """Verify baseline is calculated strictly from X_train."""
    X_train_mock = np.ones((100, 37)) * 0.5  # all 0.5
    s_train = compute_feature_scalar(X_train_mock, clip=5.0)
    base_mean = float(np.mean(s_train))
    assert base_mean == pytest.approx(0.5, abs=1e-6)


def test_drift_pipeline_update_scalar_integration():
    """Verify update_scalar reaches ADWIN, persistence, and severity according to contract."""
    det = ADWINDetector(delta=0.002, clock=32)
    persist = DriftPersistence(consecutive_threshold=3)
    sev = DriftSeverity(formula="relative_shift", baseline_mean=0.5, max_shift=1.0, smoothing_factor=0.8)
    pipeline = DriftPipeline(detector=det, persistence=persist, severity=sev)

    # Normal scalar
    status = pipeline.update_scalar(0.5)
    assert status.monitored_value == 0.5
    assert status.raw_severity == 0.0
    assert status.drift_detected is False
    assert status.is_persistent is False

    # Shifted scalar
    status2 = pipeline.update_scalar(1.0)
    assert status2.monitored_value == 1.0
    # |1.0 - 0.5| / 1.0 = 0.5
    assert status2.raw_severity == pytest.approx(0.5, abs=1e-6)


def test_feature_scalar_causality():
    """Verify feature-space signal does not require y_true, y_pred, or drift ground truth."""
    x_t = np.random.RandomState(42).randn(37)
    # Causal call using ONLY x_t:
    s_t = compute_feature_scalar(x_t, clip=5.0)
    assert isinstance(s_t, float)
    assert not np.isnan(s_t)
    assert not np.isinf(s_t)
