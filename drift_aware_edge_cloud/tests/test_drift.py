"""Phase 3 Unit and Integration Tests -- ADWIN, Persistence, and Severity.

Covers:
ADWIN:
 1. creation
 2. configuration
 3. no false alarm on stable stationary stream
 4. detection on controlled distribution change
 5. sequential update behavior and window tracking
 6. deterministic behavior
 7. no Target / ground-truth dependency

Persistence:
 8. transient alarm does not trigger persistence
 9. repeated/consecutive evidence reaches persistence threshold
10. windowed_count criterion
11. reset behavior
12. configurable threshold validation

Severity:
13. stable/no-change condition produces zero/low severity
14. larger observable change produces monotonically greater severity in [0, 1]
15. deterministic calculation across formulas
16. exact relative_shift mathematical definition (raw vs smoothed)
17. causal baseline mean computation from baseline_train only

Integration:
18. ADWIN -> Persistence -> Severity pipeline
19. small streaming smoke test
20. Phase 1 / Phase 2 compatibility
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.drift import (
    ADWINDetector,
    DriftPersistence,
    DriftPipeline,
    DriftSeverity,
    DriftStatus,
    compute_baseline_signal_mean,
)
from src.models import CloudModel, EdgeModel
from src.utils import config as cfgmod


# -----------------------------------------------------------------------------
# ADWIN Tests (1 - 7)
# -----------------------------------------------------------------------------


def test_1_adwin_creation():
    detector = ADWINDetector()
    assert detector.delta == 0.002
    assert detector.clock == 32
    assert detector.monitored_signal == "prediction_probability"
    assert detector.n_samples_seen == 0
    assert detector.n_drifts_detected == 0
    assert not detector.drift_detected
    info = detector.get_info()
    assert info["detector_type"] == "ADWINDetector"
    assert info["n_samples_seen"] == 0


def test_2_adwin_configuration(cfg):
    detector = ADWINDetector(cfg)
    assert detector.delta == 0.002
    assert detector.clock == 32
    assert detector.monitored_signal == "prediction_probability"


def test_3_adwin_stationary_stream_stability():
    """ADWIN must not trigger false alarms on a stationary Gaussian/uniform stream."""
    rng = np.random.default_rng(42)
    detector = ADWINDetector(delta=0.002, clock=16, grace_period=10)
    # Stream of 300 stationary low values around 0.03 (typical baseline attack prob)
    vals = rng.normal(loc=0.03, scale=0.005, size=300).clip(0.0, 1.0)
    alarms = [detector.update(v) for v in vals]
    assert sum(alarms) == 0, f"False alarm triggered on stationary stream: {sum(alarms)} alarms"
    assert detector.n_samples_seen == 300
    assert detector.n_drifts_detected == 0


def test_4_adwin_detection_on_controlled_change():
    """ADWIN must detect a clear shift in distribution."""
    rng = np.random.default_rng(42)
    detector = ADWINDetector(delta=0.002, clock=8, grace_period=10)

    # Pre-drift segment: mean ~ 0.05
    pre_drift = rng.normal(loc=0.05, scale=0.01, size=200).clip(0.0, 1.0)
    for v in pre_drift:
        detector.update(v)
    assert detector.n_drifts_detected == 0

    # Post-drift segment: abrupt shift to mean ~ 0.75
    post_drift = rng.normal(loc=0.75, scale=0.02, size=150).clip(0.0, 1.0)
    detected = False
    detection_step = None
    for i, v in enumerate(post_drift):
        if detector.update(v):
            detected = True
            detection_step = i
            break

    assert detected, "ADWIN failed to detect controlled abrupt shift"
    assert detection_step is not None and detection_step < 120, f"Detection delayed: step {detection_step}"


def test_5_adwin_sequential_update_behavior():
    detector = ADWINDetector(clock=4)
    assert detector.width == 0
    for i in range(20):
        detector.update(0.1)
    assert detector.n_samples_seen == 20
    assert detector.width > 0
    assert abs(detector.estimation - 0.1) < 1e-4

    # Reset behavior
    detector.reset()
    assert detector.n_samples_seen == 0
    assert detector.width == 0
    assert detector.n_drifts_detected == 0


def test_6_adwin_deterministic_behavior():
    rng1 = np.random.default_rng(123)
    stream1 = rng1.uniform(0.0, 0.1, size=100)
    det1 = ADWINDetector(delta=0.005)
    r1 = [det1.update(v) for v in stream1]

    rng2 = np.random.default_rng(123)
    stream2 = rng2.uniform(0.0, 0.1, size=100)
    det2 = ADWINDetector(delta=0.005)
    r2 = [det2.update(v) for v in stream2]

    assert r1 == r2
    assert det1.estimation == det2.estimation
    assert det1.width == det2.width


def test_7_adwin_no_ground_truth_dependency():
    """Detector operates solely on observable model output; rejects Target requirements."""
    detector = ADWINDetector(monitored_signal="prediction_probability")
    # Model predicted probability output
    probs = {0: 0.95, 1: 0.05}
    det = detector.update_from_prediction(probs)
    assert isinstance(det, bool)
    assert detector.last_signal_value == 0.05

    # Uncertainty signal mode
    det_unc = ADWINDetector(monitored_signal="uncertainty")
    det_unc.update_from_prediction({0: 0.6, 1: 0.4})
    # 2 * (1 - 0.6) = 0.8
    assert abs(det_unc.last_signal_value - 0.8) < 1e-5

    # Prediction error mode requires explicit error without accessing Target
    det_err = ADWINDetector(monitored_signal="prediction_error")
    with pytest.raises(ValueError, match="requires an explicit scalar 'error'"):
        det_err.update_from_prediction(probs)


# -----------------------------------------------------------------------------
# Persistence Tests (8 - 12)
# -----------------------------------------------------------------------------


def test_8_persistence_transient_alarm_does_not_trigger():
    persist = DriftPersistence(criterion="consecutive", consecutive_threshold=3)
    assert not persist.is_persistent

    # Isolated single alarm followed by False
    persist.update(True)
    assert persist.current_streak == 1
    assert not persist.is_persistent

    persist.update(False)
    assert persist.current_streak == 0
    assert not persist.is_persistent


def test_9_persistence_consecutive_threshold_trigger():
    persist = DriftPersistence(criterion="consecutive", consecutive_threshold=3)
    persist.update(True)
    assert not persist.is_persistent
    persist.update(True)
    assert not persist.is_persistent
    # 3rd consecutive alarm
    persist.update(True)
    assert persist.is_persistent
    assert persist.current_streak == 3

    # Reset on False
    persist.update(False)
    assert not persist.is_persistent
    assert persist.current_streak == 0


def test_10_persistence_windowed_count_criterion():
    persist = DriftPersistence(criterion="windowed_count", window_size=5, count_threshold=3)
    # [True, False, True, False, True] -> 3 in 5 -> persistent
    assert not persist.update(True)
    assert not persist.update(False)
    assert not persist.update(True)
    assert not persist.update(False)
    assert persist.update(True)
    assert persist.is_persistent

    # Next two are False: window becomes [True, False, True, False, False] -> 2 in 5 -> not persistent
    persist.update(False)
    assert not persist.is_persistent


def test_11_persistence_reset():
    persist = DriftPersistence(consecutive_threshold=2)
    persist.update(True)
    persist.update(True)
    assert persist.is_persistent
    persist.reset()
    assert not persist.is_persistent
    assert persist.current_streak == 0
    assert persist.total_alarms == 0
    assert persist.total_updates == 0


def test_12_persistence_config_validation():
    with pytest.raises(ValueError, match="Unknown persistence criterion"):
        DriftPersistence(criterion="invalid_rule")
    with pytest.raises(ValueError, match="consecutive_threshold must be >= 1"):
        DriftPersistence(consecutive_threshold=0)


# -----------------------------------------------------------------------------
# Severity Tests (13 - 17)
# -----------------------------------------------------------------------------


def test_13_severity_stable_condition_low_severity():
    sev = DriftSeverity(formula="relative_shift", baseline_mean=0.03, max_shift=0.97, smoothing_factor=0.0)
    # At baseline_mean, raw severity is exactly 0.0
    d = sev.compute_raw_severity(0.03)
    assert d == 0.0
    sev.update(0.03)
    assert sev.raw_severity == 0.0
    assert sev.severity == 0.0


def test_14_severity_monotonicity_in_unit_interval():
    sev = DriftSeverity(formula="relative_shift", baseline_mean=0.0, max_shift=1.0, smoothing_factor=0.0)
    shifts = [0.0, 0.2, 0.5, 0.8, 1.0, 1.5]
    scores = [sev.compute_raw_severity(s) for s in shifts]
    assert scores == sorted(scores), "Severity must be monotonic with respect to shift"
    assert scores[0] == 0.0
    assert scores[4] == 1.0
    assert scores[5] == 1.0  # Clipped to 1.0
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_15_severity_exact_relative_shift_formula():
    """Verifies D = min(1.0, abs(current_shift - baseline_mean) / max_shift)."""
    base_mean = 0.05
    max_shift = 0.95
    sev = DriftSeverity(formula="relative_shift", baseline_mean=base_mean, max_shift=max_shift, smoothing_factor=0.0)

    val = 0.525
    expected_d = min(1.0, abs(val - base_mean) / max_shift)  # 0.475 / 0.95 = 0.50
    assert abs(sev.compute_raw_severity(val) - expected_d) < 1e-6


def test_16_severity_distinguishes_raw_and_smoothed():
    """Separation of raw_severity and smoothed_severity must be strict."""
    sev = DriftSeverity(
        formula="relative_shift",
        baseline_mean=0.0,
        max_shift=1.0,
        smoothing_factor=0.5,
    )
    # First update: smoothed initialized to raw
    sev.update(1.0)
    assert sev.raw_severity == 1.0
    assert sev.smoothed_severity == 1.0

    # Second update with 0.0: raw is 0.0, smoothed is 0.5 * 1.0 + 0.5 * 0.0 = 0.5
    sev.update(0.0)
    assert sev.raw_severity == 0.0
    assert abs(sev.smoothed_severity - 0.5) < 1e-6
    assert sev.severity == sev.smoothed_severity


def test_17_causal_baseline_mean_computation():
    """compute_baseline_signal_mean must run inference on baseline_train only."""
    rng = np.random.default_rng(42)
    X_train = pd.DataFrame(rng.standard_normal((100, 37)), columns=[f"f{i}" for i in range(37)])
    y_train = np.zeros(100, dtype=int)
    y_train[rng.choice(100, size=5, replace=False)] = 1

    model = EdgeModel().fit(X_train, y_train)
    b_mean = compute_baseline_signal_mean(model, X_train, signal_type="prediction_probability")
    assert 0.0 <= b_mean <= 1.0

    # Model probability predictions on training data match
    probs = model.predict_proba(X_train)
    assert abs(b_mean - np.mean(probs[:, 1])) < 1e-6


# -----------------------------------------------------------------------------
# Integration Tests (18 - 20)
# -----------------------------------------------------------------------------


def test_18_drift_pipeline_integration(cfg):
    pipeline = DriftPipeline(cfg)
    assert isinstance(pipeline.detector, ADWINDetector)
    assert isinstance(pipeline.persistence, DriftPersistence)
    assert isinstance(pipeline.severity, DriftSeverity)

    status = pipeline.update_scalar(0.03)
    assert isinstance(status, DriftStatus)
    assert not status.drift_detected
    assert not status.is_persistent
    assert 0.0 <= status.raw_severity <= 1.0


def test_19_streaming_smoke_test():
    """Simulates online streaming: stationary baseline followed by persistent drift."""
    rng = np.random.default_rng(999)
    pipeline = DriftPipeline(
        detector=ADWINDetector(delta=0.002, clock=4, grace_period=10),
        persistence=DriftPersistence(criterion="windowed_count", window_size=50, count_threshold=1),
        severity=DriftSeverity(baseline_mean=0.05, max_shift=0.95, smoothing_factor=0.7),
    )

    # 100 stationary observations
    for _ in range(100):
        val = float(rng.normal(0.05, 0.01))
        status = pipeline.update_scalar(val)
    assert not status.is_persistent
    assert status.raw_severity < 0.1

    # 100 persistent high-anomaly observations
    persistent_observed = False
    for _ in range(100):
        val = float(rng.normal(0.85, 0.02))
        status = pipeline.update_scalar(val)
        if status.is_persistent:
            persistent_observed = True
            assert status.raw_severity > 0.5

    assert persistent_observed, "Pipeline should flag persistent drift on prolonged distribution shift"


def test_20_phase1_phase2_model_compatibility():
    """Verifies that Edge and Cloud models can feed prediction outputs directly to Phase 3."""
    rng = np.random.default_rng(42)
    feature_names = [f"f{i}" for i in range(37)]
    X_train = pd.DataFrame(rng.standard_normal((100, 37)), columns=feature_names)
    y_train = np.zeros(100, dtype=int)
    y_train[:5] = 1

    edge = EdgeModel().fit(X_train, y_train)
    cloud = CloudModel().fit(X_train, y_train)

    pipeline_edge = DriftPipeline()
    pipeline_cloud = DriftPipeline()

    x_single = X_train.iloc[0]
    p_edge = edge.predict_proba_one(x_single)
    p_cloud = cloud.predict_proba_one(x_single)

    status_edge = pipeline_edge.update_from_prediction(p_edge)
    status_cloud = pipeline_cloud.update_from_prediction(p_cloud)

    assert isinstance(status_edge, DriftStatus)
    assert isinstance(status_cloud, DriftStatus)
    assert 0.0 <= status_edge.raw_severity <= 1.0
    assert 0.0 <= status_cloud.raw_severity <= 1.0
