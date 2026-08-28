"""Unit and integration tests for Phase 4 DRAEC Reliability Estimation."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.preprocessing import QualityReport
from src.drift import DriftPipeline, DriftStatus
from src.models.edge_model import EdgeHoeffdingTree
from src.models.cloud_model import CloudXGBoost
from src.reliability import (
    BaseReliabilityEstimator,
    ReliabilityEstimator,
    ReliabilityFactors,
    ReliabilityInputs,
    ReliabilityScore,
    compute_confidence,
    compute_harmonic_reliability,
    compute_instantaneous_error,
    compute_quality,
)


def test_01_reliability_estimator_creation():
    est = ReliabilityEstimator()
    assert isinstance(est, BaseReliabilityEstimator)
    assert est.current_error == 0.0
    assert est.alpha_E == 0.8
    assert est.epsilon == 1e-8
    assert est.default_n_features == 37


def test_02_default_configuration():
    est = ReliabilityEstimator()
    info = est.get_info()
    assert info["alpha_E"] == 0.8
    assert info["epsilon"] == 1e-8
    assert info["weights"] == {
        "confidence": 0.25,
        "error": 0.25,
        "drift": 0.25,
        "quality": 0.25,
    }
    assert info["current_error"] == 0.0


def test_03_yaml_configuration(cfg):
    est = ReliabilityEstimator(config=cfg)
    assert est.alpha_E == 0.8
    assert est.epsilon == 1e-8
    assert est.default_n_features == 37
    assert est.weights["confidence"] == 0.25


def test_04_confidence_calculation():
    # Symmetric 50/50 -> 0.0
    assert compute_confidence({0: 0.5, 1: 0.5}) == 0.0
    # Complete certainty -> 1.0
    assert compute_confidence({0: 1.0, 1: 0.0}) == 1.0
    assert compute_confidence({0: 0.0, 1: 1.0}) == 1.0
    # Intermediate: max_p = 0.8 -> 2 * (0.8 - 0.5) = 0.6
    assert abs(compute_confidence({0: 0.2, 1: 0.8}) - 0.6) < 1e-6
    assert abs(compute_confidence([0.7, 0.3]) - 0.4) < 1e-6


def test_05_confidence_bounds():
    for p in np.linspace(0.0, 1.0, 50):
        c = compute_confidence({0: p, 1: 1.0 - p})
        assert 0.0 <= c <= 1.0


def test_06_confidence_maximum_minimum_behavior():
    # Maximum ambiguity
    c_min = compute_confidence({0: 0.5, 1: 0.5})
    assert c_min == 0.0
    # Maximum certainty
    c_max = compute_confidence({0: 1.0, 1: 0.0})
    assert c_max == 1.0


def test_07_binary_zero_one_prediction_error():
    assert compute_instantaneous_error(y_pred=0, y_true=0) == 0.0
    assert compute_instantaneous_error(y_pred=1, y_true=1) == 0.0
    assert compute_instantaneous_error(y_pred=0, y_true=1) == 1.0
    assert compute_instantaneous_error(y_pred=1, y_true=0) == 1.0


def test_08_error_ema_update():
    est = ReliabilityEstimator(alpha_E=0.8, initial_error=0.0)
    # E_1 = 0.8 * 0.0 + 0.2 * 1.0 = 0.2
    assert abs(est.update_error(1.0) - 0.2) < 1e-6
    # E_2 = 0.8 * 0.2 + 0.2 * 1.0 = 0.36
    assert abs(est.update_error(1.0) - 0.36) < 1e-6
    # E_3 with 0.0 error: 0.8 * 0.36 + 0.2 * 0.0 = 0.288
    assert abs(est.update_error(0.0) - 0.288) < 1e-6


def test_09_error_repeated_feedback_convergence():
    est = ReliabilityEstimator(alpha_E=0.8, initial_error=0.0)
    # 50 continuous errors should converge to 1.0
    for _ in range(50):
        est.update_error(1.0)
    assert abs(est.current_error - 1.0) < 1e-4

    # 50 continuous successes should converge to 0.0
    for _ in range(50):
        est.update_error(0.0)
    assert abs(est.current_error - 0.0) < 1e-4


def test_10_delayed_feedback_behavior():
    est = ReliabilityEstimator(alpha_E=0.8, initial_error=0.0)
    # Observation 1 inference: no feedback available yet
    s1 = est.update(confidence=0.9, drift_severity=0.0, quality=1.0)
    assert s1.inputs.error == 0.0  # Uses previous state
    assert est.n_error_updates == 0

    # Observation 2 inference: still no feedback
    s2 = est.update(confidence=0.9, drift_severity=0.0, quality=1.0)
    assert s2.inputs.error == 0.0
    assert est.n_error_updates == 0

    # Delayed feedback for observation 1 arrives now (y_true=1, y_pred=0 -> error 1)
    new_e = est.update_feedback(y_true=1, y_pred=0)
    assert abs(new_e - 0.2) < 1e-6
    assert est.n_error_updates == 1

    # Observation 3 inference: consumes updated E_t = 0.2
    s3 = est.update(confidence=0.9, drift_severity=0.0, quality=1.0)
    assert abs(s3.inputs.error - 0.2) < 1e-6


def test_11_error_retains_previous_value_without_feedback():
    est = ReliabilityEstimator(initial_error=0.35)
    score1 = est.update(confidence=0.8)
    assert score1.inputs.error == 0.35
    score2 = est.update(confidence=0.7)
    assert score2.inputs.error == 0.35
    assert est.current_error == 0.35


def test_12_drift_consumes_phase3_smoothed_severity():
    est = ReliabilityEstimator()
    status = DriftStatus(
        drift_detected=True,
        is_persistent=False,
        raw_severity=0.95,
        smoothed_severity=0.45,
        estimation=0.5,
        monitored_value=0.5,
    )
    score = est.update(confidence=0.8, drift_status=status, quality=1.0)
    assert score.inputs.drift == 0.45
    assert abs(score.factors.r_D - 0.55) < 1e-6


def test_13_quality_general_n_features_formulation():
    # 10 features, 8 valid
    q = compute_quality([1, 1, 1, 1, 1, 1, 1, 1, 0, 0], n_features=10)
    assert abs(q - 0.8) < 1e-6

    # 4 features, all valid
    q4 = compute_quality([True, True, True, True], n_features=4)
    assert q4 == 1.0

    # Direct scalar
    assert compute_quality(0.75) == 0.75


def test_14_quality_wustl_37_features_instantiation():
    # 37 features, 35 valid -> 35 / 37
    bool_vec = [True] * 35 + [False] * 2
    q = compute_quality(bool_vec, n_features=37)
    assert abs(q - (35.0 / 37.0)) < 1e-6

    # QualityReport with 3 range violation channels
    qr = QualityReport(
        n_rows=10,
        valid=np.ones(10, dtype=bool),
        validation_failed=np.zeros(10, dtype=bool),
        range_violation=np.zeros(10, dtype=bool),
        outlier=np.zeros(10, dtype=bool),
        filled=np.zeros(10, dtype=bool),
        unfilled=np.zeros(10, dtype=bool),
        n_range_violations_by_column={"c1": 2, "c2": 3, "c3": 1},
    )
    q_rep = compute_quality(qr, n_features=37)
    assert abs(q_rep - (34.0 / 37.0)) < 1e-6


def test_15_quality_bounds():
    assert compute_quality(1.5) == 1.0
    assert compute_quality(-0.2) == 0.0
    assert compute_quality(None) == 1.0


def test_16_weighted_harmonic_reliability_calculation():
    # All factors optimal (1.0)
    r_opt = compute_harmonic_reliability(r_C=1.0, r_E=1.0, r_D=1.0, r_Q=1.0, epsilon=1e-8)
    assert abs(r_opt - 1.0) < 1e-5

    # Equal factors 0.5 -> R = 0.5
    r_half = compute_harmonic_reliability(r_C=0.5, r_E=0.5, r_D=0.5, r_Q=0.5, epsilon=1e-8)
    assert abs(r_half - 0.5) < 1e-5


def test_17_reliability_bounds():
    rng = np.random.RandomState(42)
    for _ in range(100):
        rc, re, rd, rq = rng.uniform(0.0, 1.0, 4)
        r = compute_harmonic_reliability(rc, re, rd, rq)
        assert 0.0 <= r <= 1.0


def test_18_reliability_monotonicity():
    base_c, base_e, base_d, base_q = 0.6, 0.2, 0.3, 0.9
    r_base = compute_harmonic_reliability(
        r_C=base_c, r_E=1.0 - base_e, r_D=1.0 - base_d, r_Q=base_q
    )

    # Increasing C_t increases or keeps R_t
    r_high_c = compute_harmonic_reliability(
        r_C=0.9, r_E=1.0 - base_e, r_D=1.0 - base_d, r_Q=base_q
    )
    assert r_high_c > r_base

    # Increasing E_t decreases R_t
    r_high_e = compute_harmonic_reliability(
        r_C=base_c, r_E=1.0 - 0.8, r_D=1.0 - base_d, r_Q=base_q
    )
    assert r_high_e < r_base

    # Increasing D_t decreases R_t
    r_high_d = compute_harmonic_reliability(
        r_C=base_c, r_E=1.0 - base_e, r_D=1.0 - 0.9, r_Q=base_q
    )
    assert r_high_d < r_base

    # Increasing Q_t increases R_t
    r_low_q = compute_harmonic_reliability(
        r_C=base_c, r_E=1.0 - base_e, r_D=1.0 - base_d, r_Q=0.4
    )
    assert r_low_q < r_base


def test_19_epsilon_stability_and_weakest_link():
    # Weakest-link property: if one factor is 0.0, R_t must collapse near 0
    r_zero_c = compute_harmonic_reliability(r_C=0.0, r_E=1.0, r_D=1.0, r_Q=1.0, epsilon=1e-8)
    assert r_zero_c < 1e-6

    r_zero_e = compute_harmonic_reliability(r_C=1.0, r_E=0.0, r_D=1.0, r_Q=1.0, epsilon=1e-8)
    assert r_zero_e < 1e-6

    r_zero_d = compute_harmonic_reliability(r_C=1.0, r_E=1.0, r_D=0.0, r_Q=1.0, epsilon=1e-8)
    assert r_zero_d < 1e-6

    r_zero_q = compute_harmonic_reliability(r_C=1.0, r_E=1.0, r_D=1.0, r_Q=0.0, epsilon=1e-8)
    assert r_zero_q < 1e-6


def test_20_determinism():
    est1 = ReliabilityEstimator(alpha_E=0.8)
    est2 = ReliabilityEstimator(alpha_E=0.8)

    inputs = [
        ({"probs": {0: 0.8, 1: 0.2}, "drift_severity": 0.1, "quality": 1.0}),
        ({"probs": {0: 0.4, 1: 0.6}, "drift_severity": 0.3, "quality": 0.9}),
        ({"probs": {0: 0.9, 1: 0.1}, "drift_severity": 0.5, "quality": 0.8, "error": 1.0}),
    ]
    for inp in inputs:
        s1 = est1.update(**inp)
        s2 = est2.update(**inp)
        assert s1.reliability == s2.reliability


def test_21_reset_behavior():
    est = ReliabilityEstimator(initial_error=0.1)
    est.update_error(1.0)
    assert est.current_error > 0.1
    assert est.n_error_updates == 1

    est.reset()
    assert est.current_error == 0.1
    assert est.n_error_updates == 0
    assert est.last_score is None


def test_22_no_future_label_leakage():
    # Verify that ReliabilityEstimator update() operates purely on observable inputs
    # without requiring Target or future information
    est = ReliabilityEstimator()
    score = est.update(probs={0: 0.85, 1: 0.15}, drift_severity=0.1, quality=1.0)
    assert score.reliability > 0.0
    # No Target was queried or passed


def test_23_no_ground_truth_json_consumption():
    # Verify forbidden consumers rule
    est = ReliabilityEstimator()
    # Confirm it does not inspect or require ground_truth sidecar
    info = est.get_info()
    assert "scenario" not in info
    assert "drift_magnitude" not in info


def test_24_edge_model_compatibility():
    # Train small EdgeHoeffdingTree and calculate reliability from its prediction
    model = EdgeHoeffdingTree(grace_period=10)
    X = np.array([[1.0, 2.0], [2.0, 1.0], [1.5, 1.8], [0.5, 0.2]])
    y = np.array([0, 1, 0, 1])
    model.fit(X, y)

    x_obs = [1.2, 1.9]
    probs = model.predict_proba_one(x_obs)

    est = ReliabilityEstimator()
    score = est.update(probs=probs, drift_severity=0.05, quality=1.0)
    assert 0.0 <= score.reliability <= 1.0
    assert 0.0 <= score.inputs.confidence <= 1.0


def test_25_cloud_model_compatibility():
    # Train small CloudXGBoost and calculate reliability from its prediction
    model = CloudXGBoost(n_estimators=5, max_depth=2, random_state=42)
    X = np.array([[1.0, 2.0], [2.0, 1.0], [1.5, 1.8], [0.5, 0.2]])
    y = np.array([0, 1, 0, 1])
    model.fit(X, y)

    x_obs = [1.2, 1.9]
    probs = model.predict_proba_one(x_obs)

    est = ReliabilityEstimator()
    score = est.update(probs=probs, drift_severity=0.05, quality=1.0)
    assert 0.0 <= score.reliability <= 1.0
    assert 0.0 <= score.inputs.confidence <= 1.0


def test_26_edge_cases_all_zeros_and_ones():
    # All combinations of extreme 0 and 1 inputs
    for c in (0.0, 1.0):
        for e in (0.0, 1.0):
            for d in (0.0, 1.0):
                for q in (0.0, 1.0):
                    inp = ReliabilityInputs(confidence=c, error=e, drift=d, quality=q)
                    est = ReliabilityEstimator()
                    score = est.calculate(inp)
                    assert not np.isnan(score.reliability)
                    assert not np.isinf(score.reliability)
                    assert 0.0 <= score.reliability <= 1.0
