"""Drift-Aware Adaptive Edge-Cloud-Hybrid Orchestration for Non-Stationary IoT.

Package  : src/drift
Phase    : Phase 3
Status   : IMPLEMENTED

Phase 3 Drift Detection Layer:
- ADWIN statistical change detection (ADWINDetector)
- Drift persistence tracking (DriftPersistence)
- Normalized continuous drift severity (DriftSeverity)
- Integrated streaming coordinator (DriftPipeline)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from src.drift.adwin_detector import ADWINDetector
from src.drift.persistence import DriftPersistence
from src.drift.severity import DriftSeverity, compute_baseline_signal_mean


@dataclass(frozen=True)
class DriftStatus:
    """Immutable snapshot of the drift detection pipeline at observation t."""

    drift_detected: bool
    is_persistent: bool
    raw_severity: float
    smoothed_severity: float
    estimation: float
    monitored_value: float


class DriftPipeline:
    """Coordinating pipeline for Phase 3 components in sequential streaming.

    Causal execution chain:
    Observable inference signal -> ADWIN change detection -> Persistence -> Severity
    """

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        *,
        detector: ADWINDetector | None = None,
        persistence: DriftPersistence | None = None,
        severity: DriftSeverity | None = None,
    ) -> None:
        self.detector = detector if detector is not None else ADWINDetector(config)
        self.persistence = persistence if persistence is not None else DriftPersistence(config)
        self.severity = severity if severity is not None else DriftSeverity(config)

    def update_from_prediction(
        self,
        probs: Mapping[int, float] | Sequence[float] | np.ndarray,
        y_pred: int | None = None,
        *,
        error: float | None = None,
    ) -> DriftStatus:
        """Process model prediction output through the complete Phase 3 drift pipeline.

        Parameters
        ----------
        probs : Mapping[int, float] or Sequence[float] or np.ndarray
            Prediction probabilities (e.g. from predict_proba_one).
        y_pred : int, optional
            Predicted class label (e.g. from predict_one).
        error : float, optional
            Explicit scalar error if using 'prediction_error' signal.

        Returns
        -------
        DriftStatus
            Structured result containing detection, persistence, and severity.
        """
        drift_detected = self.detector.update_from_prediction(probs, y_pred, error=error)
        is_persistent = self.persistence.update(drift_detected)

        # Severity quantifies the magnitude of observed change
        val = self.detector.last_signal_value if self.detector.last_signal_value is not None else self.detector.estimation
        self.severity.update(val)

        return DriftStatus(
            drift_detected=drift_detected,
            is_persistent=is_persistent,
            raw_severity=self.severity.raw_severity,
            smoothed_severity=self.severity.smoothed_severity,
            estimation=self.detector.estimation,
            monitored_value=val,
        )

    def update_scalar(self, value: float) -> DriftStatus:
        """Process a direct scalar observation through the drift pipeline."""
        drift_detected = self.detector.update(value)
        is_persistent = self.persistence.update(drift_detected)
        self.severity.update(value)

        return DriftStatus(
            drift_detected=drift_detected,
            is_persistent=is_persistent,
            raw_severity=self.severity.raw_severity,
            smoothed_severity=self.severity.smoothed_severity,
            estimation=self.detector.estimation,
            monitored_value=float(value),
        )

    def reset(self) -> None:
        """Reset all three drift pipeline stages."""
        self.detector.reset()
        self.persistence.reset()
        self.severity.reset()


__all__ = [
    "ADWINDetector",
    "DriftPersistence",
    "DriftSeverity",
    "DriftPipeline",
    "DriftStatus",
    "compute_baseline_signal_mean",
]
