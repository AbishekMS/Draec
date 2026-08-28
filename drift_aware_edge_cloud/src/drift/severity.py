"""Drift-Aware Adaptive Edge-Cloud-Hybrid Orchestration for Non-Stationary IoT.

Module   : src/drift/severity.py
Phase    : Phase 3
Status   : IMPLEMENTED

Normalised drift severity D in [0, 1].
Quantifies the magnitude of observed change relative to the pre-drift baseline expectation.
Strictly distinguishes raw_severity from smoothed_severity.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.models.base import BaseModel


def compute_baseline_signal_mean(
    model: BaseModel,
    X_baseline_train: pd.DataFrame | np.ndarray,
    signal_type: str = "prediction_probability",
) -> float:
    """Causally compute the empirical baseline reference mean from baseline_train.

    LEAKAGE GUARD:
    Must be computed ONLY on causal baseline_train features using a model fitted
    on baseline_train. Never computed on validation or test partitions.

    Parameters
    ----------
    model : BaseModel
        Fitted Edge or Cloud prediction model.
    X_baseline_train : pd.DataFrame or np.ndarray
        Causal baseline training features (e.g. from load_causal_train_data).
    signal_type : str
        Signal type: 'prediction_probability', 'uncertainty', or 'prediction'.

    Returns
    -------
    float
        Empirical mean of the monitored signal across baseline_train.
    """
    probs = model.predict_proba(X_baseline_train)
    if signal_type in ("prediction_probability", "predicted_probability"):
        # Mean predicted probability of class 1 (attack/anomaly)
        p1 = probs[:, 1] if probs.ndim > 1 and probs.shape[1] > 1 else probs.ravel()
        return float(np.mean(p1))
    elif signal_type == "uncertainty":
        # Mean normalized classification uncertainty 2 * (1 - max(p0, p1))
        max_p = np.max(probs, axis=1)
        return float(np.mean(2.0 * (1.0 - max_p)))
    elif signal_type in ("prediction", "predicted_class"):
        preds = model.predict(X_baseline_train)
        return float(np.mean(preds))
    else:
        raise ValueError(f"Unknown signal_type for baseline mean computation: {signal_type!r}")


class DriftSeverity:
    """Calculates continuous normalized drift severity D in [0, 1].

    Quantifies the magnitude of observed change relative to the causal baseline
    expectation.

    Mathematical Definition of raw_severity:
    ----------------------------------------
    For formula='relative_shift':
        raw_severity = min(1.0, abs(current_value - baseline_mean) / max_shift)

    For formula='exponential':
        raw_severity = 1.0 - exp(-sensitivity * abs(current_value - baseline_mean))

    Smoothing Operation:
    --------------------
    Smoothing is computed and tracked separately from raw_severity:
        smoothed_severity_t = smoothing_factor * smoothed_severity_{t-1} + (1 - smoothing_factor) * raw_severity_t
    where smoothing_factor in [0.0, 1.0). When smoothing_factor = 0.0, smoothed_severity == raw_severity.
    """

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        *,
        formula: str = "relative_shift",
        baseline_mean: float | None = None,
        max_shift: float | None = None,
        sensitivity: float = 3.0,
        smoothing_factor: float = 0.8,
        **kwargs: Any,
    ) -> None:
        cfg_params: dict[str, Any] = {}
        if config is not None:
            dd_sec = config.get("drift_detection")
            if not isinstance(dd_sec, Mapping):
                drift_sec = config.get("drift")
                if isinstance(drift_sec, Mapping):
                    dd_sec = drift_sec.get("detection") or drift_sec.get("drift_detection")
            if isinstance(dd_sec, Mapping):
                sev_sec = dd_sec.get("severity")
                if isinstance(sev_sec, Mapping):
                    cfg_params.update(sev_sec)

        self._formula = str(kwargs.get("formula", cfg_params.get("formula", formula)))
        b_mean = kwargs.get("baseline_mean", cfg_params.get("baseline_mean", baseline_mean))
        self._baseline_mean = float(b_mean) if b_mean is not None else None

        m_shift = kwargs.get("max_shift", cfg_params.get("max_shift", max_shift))
        self._max_shift = float(m_shift) if m_shift is not None else None

        self._sensitivity = float(kwargs.get("sensitivity", cfg_params.get("sensitivity", sensitivity)))
        self._smoothing_factor = float(
            kwargs.get("smoothing_factor", cfg_params.get("smoothing_factor", smoothing_factor))
        )

        if self._formula not in ("relative_shift", "exponential"):
            raise ValueError(f"Unknown drift severity formula: {self._formula!r}")
        if not (0.0 <= self._smoothing_factor < 1.0):
            raise ValueError(f"smoothing_factor must be in [0.0, 1.0), got {self._smoothing_factor}")

        self._raw_severity: float = 0.0
        self._smoothed_severity: float = 0.0
        self._max_raw_severity_seen: float = 0.0
        self._total_updates: int = 0

    @property
    def formula(self) -> str:
        return self._formula

    @property
    def baseline_mean(self) -> float | None:
        return self._baseline_mean

    @property
    def max_shift(self) -> float | None:
        return self._max_shift

    @property
    def smoothing_factor(self) -> float:
        return self._smoothing_factor

    @property
    def raw_severity(self) -> float:
        """The un-smoothed, exact mathematical drift severity D in [0, 1]."""
        return self._raw_severity

    @property
    def smoothed_severity(self) -> float:
        """The exponentially smoothed drift severity in [0, 1]."""
        return self._smoothed_severity

    @property
    def severity(self) -> float:
        """Primary severity metric (returns smoothed_severity if smoothing_factor > 0 else raw_severity)."""
        return self._smoothed_severity if self._smoothing_factor > 0.0 else self._raw_severity

    @property
    def max_raw_severity_seen(self) -> float:
        return self._max_raw_severity_seen

    @property
    def total_updates(self) -> int:
        return self._total_updates

    def set_baseline_mean(self, baseline_mean: float, max_shift: float | None = None) -> None:
        """Set the causal baseline mean (e.g. measured from baseline_train)."""
        self._baseline_mean = float(baseline_mean)
        if max_shift is not None:
            self._max_shift = float(max_shift)
        elif self._max_shift is None:
            self._max_shift = max(1.0 - self._baseline_mean, 1e-6)

    def compute_raw_severity(self, current_value: float) -> float:
        """Compute exact mathematical raw severity D in [0, 1] without smoothing.

        Formula for 'relative_shift':
            D = min(1.0, abs(current_value - baseline_mean) / max_shift)

        Formula for 'exponential':
            D = 1.0 - exp(-sensitivity * abs(current_value - baseline_mean))

        Parameters
        ----------
        current_value : float
            Current observed signal value or ADWIN window estimation.

        Returns
        -------
        float
            Normalized severity in [0.0, 1.0].
        """
        val = float(current_value)
        base_mean = self._baseline_mean if self._baseline_mean is not None else 0.0
        shift = abs(val - base_mean)

        if self._formula == "relative_shift":
            m_shift = self._max_shift if self._max_shift is not None else max(1.0 - base_mean, 1e-6)
            if m_shift <= 0.0:
                m_shift = 1e-6
            raw_d = min(1.0, shift / m_shift)
        elif self._formula == "exponential":
            raw_d = 1.0 - float(np.exp(-self._sensitivity * shift))
        else:
            raise ValueError(f"Unknown formula: {self._formula!r}")

        return float(np.clip(raw_d, 0.0, 1.0))

    def update(self, current_value: float) -> float:
        """Update severity state with a new observed value.

        Calculates raw_severity, updates smoothed_severity, and updates historical tracking.

        Parameters
        ----------
        current_value : float
            Current observed signal value.

        Returns
        -------
        float
            Current severity (smoothed if smoothing_factor > 0 else raw).
        """
        raw_d = self.compute_raw_severity(current_value)
        self._raw_severity = raw_d
        self._total_updates += 1

        if raw_d > self._max_raw_severity_seen:
            self._max_raw_severity_seen = raw_d

        if self._total_updates == 1 or self._smoothing_factor == 0.0:
            self._smoothed_severity = raw_d
        else:
            alpha = self._smoothing_factor
            self._smoothed_severity = float(np.clip(alpha * self._smoothed_severity + (1.0 - alpha) * raw_d, 0.0, 1.0))

        return self.severity

    def reset(self) -> None:
        """Reset severity tracking state to zero."""
        self._raw_severity = 0.0
        self._smoothed_severity = 0.0
        self._max_raw_severity_seen = 0.0
        self._total_updates = 0

    def get_info(self) -> dict[str, Any]:
        """Return introspection, diagnostic, and state metadata."""
        return {
            "formula": self._formula,
            "baseline_mean": self._baseline_mean,
            "max_shift": self._max_shift,
            "smoothing_factor": self._smoothing_factor,
            "sensitivity": self._sensitivity,
            "raw_severity": self._raw_severity,
            "smoothed_severity": self._smoothed_severity,
            "severity": self.severity,
            "max_raw_severity_seen": self._max_raw_severity_seen,
            "total_updates": self._total_updates,
        }
