"""Drift-Aware Adaptive Edge-Cloud-Hybrid Orchestration for Non-Stationary IoT.

Module   : src/reliability/estimator.py
Phase    : Phase 4
Status   : IMPLEMENTED

DRAEC Prediction Reliability Estimator R_t in [0, 1].
Implements the multi-factor weighted harmonic aggregation of:
- Prediction confidence C_t
- Recent prediction error E_t (with delayed feedback support)
- Smoothed drift severity D_t
- Causal data/sensor quality Q_t
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from src.data.preprocessing import QualityReport
from src.drift import DriftStatus
from src.reliability.base import (
    BaseReliabilityEstimator,
    ReliabilityFactors,
    ReliabilityInputs,
    ReliabilityScore,
)


def compute_confidence(
    probs: Mapping[int, float] | Sequence[float] | np.ndarray,
) -> float:
    """Calculate prediction confidence C_t in [0, 1] from binary class probabilities.

    Equation:
        C_t = 2 * (max(P_t(0), P_t(1)) - 0.5)

    Interpretation:
        C_t = 0.0: Maximum ambiguity (P(0) = P(1) = 0.5)
        C_t = 1.0: Maximum confidence (P(k) = 1.0)
    """
    if isinstance(probs, Mapping):
        p0 = float(probs.get(0, 0.5))
        p1 = float(probs.get(1, 0.5))
        max_p = max(p0, p1)
    elif isinstance(probs, (Sequence, np.ndarray)):
        arr = np.asarray(probs, dtype=float).ravel()
        if len(arr) == 0:
            return 0.0
        elif len(arr) == 1:
            max_p = max(arr[0], 1.0 - arr[0])
        else:
            max_p = float(np.max(arr[:2]))
    else:
        raise TypeError(f"Unsupported probability type: {type(probs).__name__}")

    # Theoretical bounds on max_p for binary classification are [0.5, 1.0]
    max_p = float(np.clip(max_p, 0.5, 1.0))
    c_t = 2.0 * (max_p - 0.5)
    return float(np.clip(c_t, 0.0, 1.0))


def compute_instantaneous_error(y_pred: int, y_true: int) -> float:
    """Calculate binary 0-1 prediction loss: e_t = I(y_pred != y_true)."""
    return 0.0 if int(y_pred) == int(y_true) else 1.0


def compute_quality(
    quality_input: QualityReport | Sequence[bool | int | float] | np.ndarray | float | int | None,
    n_features: int | None = None,
) -> float:
    """Calculate causal data/sensor quality Q_t in [0, 1].

    Equation:
        Q_t = (1 / N_F) * sum_{j=1}^{N_F} q_{j,t}

    Parameters
    ----------
    quality_input : QualityReport, Sequence, array, or float
        Quality indicator. Can be a pre-calculated scalar Q_t, a boolean vector
        of per-feature valid flags, or a Phase 1 QualityReport instance.
    n_features : int, optional
        Total feature count N_F. If None, defaults to length of quality vector,
        or 37 for WUSTL-IIoT-2021.
    """
    if quality_input is None:
        return 1.0

    if isinstance(quality_input, (int, float)):
        return float(np.clip(float(quality_input), 0.0, 1.0))

    if isinstance(quality_input, QualityReport):
        nf = int(n_features) if n_features is not None else 37
        # In QualityReport, range violations and missing values identify corrupted channels
        n_range_viols = len(quality_input.n_range_violations_by_column)
        n_valid = max(0, nf - n_range_viols)
        return float(np.clip(n_valid / max(nf, 1), 0.0, 1.0))

    if isinstance(quality_input, (Sequence, np.ndarray)):
        arr = np.asarray(quality_input, dtype=float).ravel()
        nf = int(n_features) if n_features is not None else len(arr)
        if nf <= 0:
            return 1.0
        q_sum = float(np.sum(arr))
        return float(np.clip(q_sum / nf, 0.0, 1.0))

    raise TypeError(f"Unsupported quality input type: {type(quality_input).__name__}")


def compute_harmonic_reliability(
    r_C: float,
    r_E: float,
    r_D: float,
    r_Q: float,
    weights: Mapping[str, float] | None = None,
    epsilon: float = 1e-8,
) -> float:
    """Calculate DRAEC overall prediction reliability R_t using weighted harmonic mean.

    Equation:
        R_t = (w_C + w_E + w_D + w_Q) / [
            w_C / (r_C + epsilon) +
            w_E / (r_E + epsilon) +
            w_D / (r_D + epsilon) +
            w_Q / (r_Q + epsilon)
        ]
    where:
        r_C = C_t
        r_E = 1 - E_t
        r_D = 1 - D_t
        r_Q = Q_t
    """
    w = {
        "confidence": 0.25,
        "error": 0.25,
        "drift": 0.25,
        "quality": 0.25,
    }
    if weights is not None:
        for k in w:
            if k in weights:
                w[k] = float(weights[k])
            elif f"w_{k[0]}" in weights:  # e.g. w_C, w_E, w_D, w_Q
                w[k] = float(weights[f"w_{k[0]}"])

    total_w = sum(w.values())
    if total_w <= 0.0:
        raise ValueError("Sum of reliability weights must be strictly positive")

    # Normalize weights so they sum to 1.0
    w_C = w["confidence"] / total_w
    w_E = w["error"] / total_w
    w_D = w["drift"] / total_w
    w_Q = w["quality"] / total_w

    eps = max(float(epsilon), 1e-12)

    c_val = float(np.clip(r_C, 0.0, 1.0))
    e_val = float(np.clip(r_E, 0.0, 1.0))
    d_val = float(np.clip(r_D, 0.0, 1.0))
    q_val = float(np.clip(r_Q, 0.0, 1.0))

    denom = (
        w_C / (c_val + eps)
        + w_E / (e_val + eps)
        + w_D / (d_val + eps)
        + w_Q / (q_val + eps)
    )

    if denom <= 0.0:
        return 0.0

    r_t = 1.0 / denom
    return float(np.clip(r_t, 0.0, 1.0))


class ReliabilityEstimator(BaseReliabilityEstimator):
    """DRAEC Prediction Reliability Estimator.

    Monitors:
    1. Confidence C_t = 2 * (max(P(0), P(1)) - 0.5)
    2. Recent prediction error E_t via EMA with delayed feedback
    3. Smoothed drift severity D_t from Phase 3
    4. Causal sensor/data quality Q_t from Phase 1

    Aggregates via weighted harmonic mean providing weakest-link degradation response.
    """

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        *,
        alpha_E: float = 0.8,
        epsilon: float = 1e-8,
        initial_error: float = 0.0,
        weights: Mapping[str, float] | None = None,
        default_n_features: int = 37,
        **kwargs: Any,
    ) -> None:
        cfg_params: dict[str, Any] = {}
        if config is not None:
            # Check drift.reliability or top-level reliability
            drift_sec = config.get("drift")
            if isinstance(drift_sec, Mapping):
                rel_sec = drift_sec.get("reliability")
                if isinstance(rel_sec, Mapping):
                    cfg_params.update(rel_sec)
            if not cfg_params:
                rel_sec = config.get("reliability")
                if isinstance(rel_sec, Mapping):
                    cfg_params.update(rel_sec)

        self._alpha_E = float(kwargs.get("alpha_E", cfg_params.get("alpha_E", alpha_E)))
        self._epsilon = float(kwargs.get("epsilon", cfg_params.get("epsilon", epsilon)))
        self._initial_error = float(
            kwargs.get("initial_error", cfg_params.get("initial_error", initial_error))
        )
        self._default_n_features = int(
            kwargs.get("default_n_features", cfg_params.get("default_n_features", default_n_features))
        )

        w_cfg = kwargs.get("weights", cfg_params.get("weights", weights))
        self._weights: dict[str, float] = {
            "confidence": 0.25,
            "error": 0.25,
            "drift": 0.25,
            "quality": 0.25,
        }
        if w_cfg is not None and isinstance(w_cfg, Mapping):
            for k in self._weights:
                if k in w_cfg:
                    self._weights[k] = float(w_cfg[k])
                elif f"w_{k[0]}" in w_cfg:
                    self._weights[k] = float(w_cfg[f"w_{k[0]}"])

        tot = sum(self._weights.values())
        if tot <= 0.0:
            raise ValueError("Sum of reliability weights must be > 0")
        for k in self._weights:
            self._weights[k] /= tot

        if not (0.0 <= self._alpha_E < 1.0):
            raise ValueError(f"alpha_E must be in [0, 1), got {self._alpha_E}")
        if self._epsilon <= 0.0:
            raise ValueError(f"epsilon must be positive, got {self._epsilon}")
        if not (0.0 <= self._initial_error <= 1.0):
            raise ValueError(f"initial_error must be in [0, 1], got {self._initial_error}")
        if self._default_n_features < 1:
            raise ValueError(f"default_n_features must be >= 1, got {self._default_n_features}")

        self._current_error: float = self._initial_error
        self._last_score: ReliabilityScore | None = None
        self._n_evaluations: int = 0
        self._n_error_updates: int = 0

    @property
    def alpha_E(self) -> float:
        return self._alpha_E

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @property
    def current_error(self) -> float:
        return self._current_error

    @property
    def weights(self) -> dict[str, float]:
        return dict(self._weights)

    @property
    def default_n_features(self) -> int:
        return self._default_n_features

    @property
    def n_evaluations(self) -> int:
        return self._n_evaluations

    @property
    def n_error_updates(self) -> int:
        return self._n_error_updates

    @property
    def last_score(self) -> ReliabilityScore | None:
        return self._last_score

    def calculate(
        self,
        inputs: ReliabilityInputs | Mapping[str, float],
    ) -> ReliabilityScore:
        """Pure calculation of reliability R_t without mutating internal estimator state."""
        if isinstance(inputs, ReliabilityInputs):
            inp = inputs
        elif isinstance(inputs, Mapping):
            inp = ReliabilityInputs(
                confidence=float(inputs["confidence"]),
                error=float(inputs["error"]),
                drift=float(inputs["drift"]),
                quality=float(inputs["quality"]),
            )
        else:
            raise TypeError(f"Expected ReliabilityInputs or Mapping, got {type(inputs).__name__}")

        factors = ReliabilityFactors(
            r_C=inp.confidence,
            r_E=1.0 - inp.error,
            r_D=1.0 - inp.drift,
            r_Q=inp.quality,
        )

        r_t = compute_harmonic_reliability(
            r_C=factors.r_C,
            r_E=factors.r_E,
            r_D=factors.r_D,
            r_Q=factors.r_Q,
            weights=self._weights,
            epsilon=self._epsilon,
        )

        return ReliabilityScore(
            reliability=r_t,
            inputs=inp,
            factors=factors,
            weights=dict(self._weights),
            epsilon=self._epsilon,
        )

    def update_error(self, error: float) -> float:
        """Update recent prediction error E_t with new ground-truth error feedback e_t.

        Equation:
            E_t = alpha_E * E_{t-1} + (1 - alpha_E) * e_t
        """
        e_val = float(np.clip(float(error), 0.0, 1.0))
        self._current_error = float(
            np.clip(self._alpha_E * self._current_error + (1.0 - self._alpha_E) * e_val, 0.0, 1.0)
        )
        self._n_error_updates += 1
        return self._current_error

    def update_feedback(self, y_true: int, y_pred: int) -> float:
        """Incorporate ground-truth feedback (y_true, y_pred) and update E_t."""
        e_t = compute_instantaneous_error(y_pred=y_pred, y_true=y_true)
        return self.update_error(e_t)

    def update(
        self,
        *,
        probs: Mapping[int, float] | Sequence[float] | np.ndarray | None = None,
        confidence: float | None = None,
        drift_severity: float | None = None,
        drift_status: DriftStatus | None = None,
        quality: QualityReport | Sequence[bool | int | float] | np.ndarray | float | int | None = None,
        error: float | None = None,
        y_true: int | None = None,
        y_pred: int | None = None,
        n_features: int | None = None,
        **kwargs: Any,
    ) -> ReliabilityScore:
        """Update estimator and evaluate current reliability R_t.

        Supports immediate inference-time evaluation and delayed error feedback.
        If no new ground-truth error feedback is passed, retains previous E_t.
        """
        # 1. Resolve confidence C_t
        if confidence is not None:
            c_t = float(np.clip(float(confidence), 0.0, 1.0))
        elif probs is not None:
            c_t = compute_confidence(probs)
        else:
            c_t = 0.5  # Neutral default

        # 2. Resolve drift severity D_t
        if drift_severity is not None:
            d_t = float(np.clip(float(drift_severity), 0.0, 1.0))
        elif drift_status is not None:
            d_t = float(np.clip(drift_status.smoothed_severity, 0.0, 1.0))
        else:
            d_t = 0.0  # Zero drift default

        # 3. Resolve data quality Q_t
        nf = n_features if n_features is not None else self._default_n_features
        q_t = compute_quality(quality, n_features=nf)

        # 4. Resolve recent prediction error E_t
        # If delayed feedback is provided now, update E_t first
        if y_true is not None and y_pred is not None:
            self.update_feedback(y_true=y_true, y_pred=y_pred)
        elif error is not None:
            self.update_error(error)

        e_t = self._current_error

        # 5. Assemble inputs and compute R_t
        inp = ReliabilityInputs(
            confidence=c_t,
            error=e_t,
            drift=d_t,
            quality=q_t,
        )
        score = self.calculate(inp)

        self._last_score = score
        self._n_evaluations += 1
        return score

    def reset(self) -> None:
        """Reset internal error and diagnostic state."""
        self._current_error = self._initial_error
        self._last_score = None
        self._n_evaluations = 0
        self._n_error_updates = 0

    def get_info(self) -> dict[str, Any]:
        """Return introspection metadata and current operational state."""
        return {
            "alpha_E": self._alpha_E,
            "epsilon": self._epsilon,
            "initial_error": self._initial_error,
            "current_error": self._current_error,
            "weights": dict(self._weights),
            "default_n_features": self._default_n_features,
            "n_evaluations": self._n_evaluations,
            "n_error_updates": self._n_error_updates,
            "last_reliability": self._last_score.reliability if self._last_score else None,
        }
