"""Drift-Aware Adaptive Edge-Cloud-Hybrid Orchestration for Non-Stationary IoT.

Module   : src/drift/adwin_detector.py
Phase    : Phase 3
Status   : IMPLEMENTED

Online adaptive windowing drift detector wrapping River's ADWIN.
Monitors an observable inference-time signal from prediction models in real-time streaming,
strictly isolated from ground truth and future labels.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from river.drift import ADWIN


class ADWINDetector:
    """Online drift detector based on River's ADWIN (Adaptive Windowing) algorithm.

    ADWIN mathematically bounds the false positive rate on stationary streams using
    the Hoeffding bound with confidence parameter delta. It dynamically shrinks its
    window when statistically significant changes in mean are detected.

    Default monitored signal is 'prediction_probability' (class-1 predicted probability),
    which is fully observable at inference time without access to true labels.
    """

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        *,
        delta: float = 0.002,
        clock: int = 32,
        max_buckets: int = 5,
        min_window_length: int = 5,
        grace_period: int = 10,
        monitored_signal: str = "prediction_probability",
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
                adwin_sec = dd_sec.get("adwin")
                if isinstance(adwin_sec, Mapping):
                    cfg_params.update(adwin_sec)

        self._delta = float(kwargs.get("delta", cfg_params.get("delta", delta)))
        self._clock = int(kwargs.get("clock", cfg_params.get("clock", clock)))
        self._max_buckets = int(kwargs.get("max_buckets", cfg_params.get("max_buckets", max_buckets)))
        self._min_window_length = int(kwargs.get("min_window_length", cfg_params.get("min_window_length", min_window_length)))
        self._grace_period = int(kwargs.get("grace_period", cfg_params.get("grace_period", grace_period)))
        self._monitored_signal = str(kwargs.get("monitored_signal", cfg_params.get("monitored_signal", monitored_signal)))

        self._adwin_kwargs: dict[str, Any] = {
            "delta": self._delta,
            "clock": self._clock,
            "max_buckets": self._max_buckets,
            "min_window_length": self._min_window_length,
            "grace_period": self._grace_period,
        }
        for k, v in kwargs.items():
            if k not in self._adwin_kwargs and k != "monitored_signal":
                self._adwin_kwargs[k] = v

        self._adwin = ADWIN(**self._adwin_kwargs)
        self._n_samples_seen: int = 0
        self._drift_detected: bool = False
        self._total_drifts_detected: int = 0
        self._last_signal_value: float | None = None
        self._last_estimation: float = 0.0

    @property
    def raw_model(self) -> ADWIN:
        """Direct access to underlying River ADWIN instance."""
        return self._adwin

    @property
    def delta(self) -> float:
        return self._delta

    @property
    def clock(self) -> int:
        return self._clock

    @property
    def monitored_signal(self) -> str:
        return self._monitored_signal

    @property
    def drift_detected(self) -> bool:
        """True if drift was detected on the most recent observation update."""
        return self._drift_detected

    @property
    def estimation(self) -> float:
        """Current estimated mean within ADWIN's adaptive window."""
        return float(getattr(self._adwin, "estimation", 0.0))

    @property
    def variance(self) -> float:
        """Current estimated variance within ADWIN's adaptive window."""
        return float(getattr(self._adwin, "variance", 0.0))

    @property
    def width(self) -> int:
        """Current number of items in ADWIN's adaptive window."""
        return int(getattr(self._adwin, "width", 0))

    @property
    def total(self) -> float:
        """Sum of items in ADWIN's adaptive window."""
        return float(getattr(self._adwin, "total", 0.0))

    @property
    def n_samples_seen(self) -> int:
        """Total count of observations passed to update()."""
        return self._n_samples_seen

    @property
    def n_drifts_detected(self) -> int:
        """Total number of drift events flagged across stream lifetime."""
        return self._total_drifts_detected

    @property
    def last_signal_value(self) -> float | None:
        """The most recent scalar value provided to the detector."""
        return self._last_signal_value

    def update(self, value: float) -> bool:
        """Update ADWIN with a single real-valued observation in streaming order.

        Parameters
        ----------
        value : float
            Real-valued signal value, typically in [0, 1].

        Returns
        -------
        bool
            True if ADWIN statistically detected change on this observation.
        """
        val_float = float(value)
        if np.isnan(val_float) or np.isinf(val_float):
            raise ValueError(f"ADWIN update requires a finite real value, got: {val_float}")

        self._adwin.update(val_float)
        self._n_samples_seen += 1
        self._last_signal_value = val_float
        self._drift_detected = bool(self._adwin.drift_detected)
        if self._drift_detected:
            self._total_drifts_detected += 1
        self._last_estimation = self.estimation

        return self._drift_detected

    def update_from_prediction(
        self,
        probs: Mapping[int, float] | Sequence[float] | np.ndarray,
        y_pred: int | None = None,
        *,
        error: float | None = None,
    ) -> bool:
        """Extract monitored signal from model prediction output and update ADWIN.

        Parameters
        ----------
        probs : Mapping[int, float] or Sequence[float] or np.ndarray
            Predicted probability distribution (e.g. from predict_proba_one).
        y_pred : int, optional
            Predicted class label (e.g. from predict_one).
        error : float, optional
            Explicit scalar prediction error (used ONLY if monitored_signal is 'prediction_error').

        Returns
        -------
        bool
            True if drift was detected.
        """
        if self._monitored_signal in ("prediction_probability", "predicted_probability"):
            if isinstance(probs, Mapping):
                val = float(probs.get(1, 0.0))
            elif isinstance(probs, (Sequence, np.ndarray)):
                arr = np.asarray(probs).ravel()
                val = float(arr[1]) if len(arr) > 1 else float(arr[0])
            else:
                raise TypeError(f"Unsupported probs type for ADWIN: {type(probs).__name__}")

        elif self._monitored_signal == "uncertainty":
            if isinstance(probs, Mapping):
                p0 = float(probs.get(0, 0.5))
                p1 = float(probs.get(1, 0.5))
                val = float(2.0 * (1.0 - max(p0, p1)))
            elif isinstance(probs, (Sequence, np.ndarray)):
                arr = np.asarray(probs).ravel()
                max_p = float(np.max(arr)) if len(arr) > 0 else 0.5
                val = float(2.0 * (1.0 - max_p))
            else:
                raise TypeError(f"Unsupported probs type for ADWIN: {type(probs).__name__}")

        elif self._monitored_signal in ("prediction", "predicted_class"):
            if y_pred is not None:
                val = float(y_pred)
            elif isinstance(probs, Mapping):
                val = 1.0 if probs.get(1, 0.0) >= probs.get(0, 0.0) else 0.0
            elif isinstance(probs, (Sequence, np.ndarray)):
                arr = np.asarray(probs).ravel()
                val = 1.0 if len(arr) > 1 and arr[1] >= arr[0] else float(arr[0])
            else:
                raise ValueError("Cannot derive predicted class without y_pred or probs")

        elif self._monitored_signal == "prediction_error":
            if error is None:
                raise ValueError(
                    "Signal 'prediction_error' requires an explicit scalar 'error' argument. "
                    "Future Target labels must NOT be queried online."
                )
            val = float(error)

        else:
            raise ValueError(f"Unknown monitored_signal: {self._monitored_signal!r}")

        return self.update(val)

    def reset(self) -> None:
        """Reset the detector state to initial empty window."""
        self._adwin = ADWIN(**self._adwin_kwargs)
        self._n_samples_seen = 0
        self._drift_detected = False
        self._total_drifts_detected = 0
        self._last_signal_value = None
        self._last_estimation = 0.0

    def get_info(self) -> dict[str, Any]:
        """Return introspection, diagnostic, and state metadata."""
        return {
            "detector_type": "ADWINDetector",
            "delta": self._delta,
            "clock": self._clock,
            "monitored_signal": self._monitored_signal,
            "n_samples_seen": self._n_samples_seen,
            "n_drifts_detected": self._total_drifts_detected,
            "drift_detected": self._drift_detected,
            "estimation": self.estimation,
            "variance": self.variance,
            "width": self.width,
            "total": self.total,
            "last_signal_value": self._last_signal_value,
        }
