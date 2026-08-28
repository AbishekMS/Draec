"""Drift-Aware Adaptive Edge-Cloud-Hybrid Orchestration for Non-Stationary IoT.

Module   : src/reliability/base.py
Phase    : Phase 4
Status   : IMPLEMENTED

Base interface and data contracts for DRAEC prediction reliability estimation.
Defines ReliabilityInputs, ReliabilityFactors, ReliabilityScore, and BaseReliabilityEstimator.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ReliabilityInputs:
    """Raw inputs to the DRAEC reliability estimator at observation t.

    Attributes
    ----------
    confidence : float
        Prediction confidence C_t in [0, 1].
    error : float
        Recent prediction-error level E_t in [0, 1].
    drift : float
        Drift severity D_t in [0, 1].
    quality : float
        Data/sensor quality Q_t in [0, 1].
    """

    confidence: float
    error: float
    drift: float
    quality: float

    def __post_init__(self) -> None:
        for name, val in (
            ("confidence", self.confidence),
            ("error", self.error),
            ("drift", self.drift),
            ("quality", self.quality),
        ):
            if not isinstance(val, (int, float)):
                raise TypeError(f"{name} must be numeric, got {type(val).__name__}")
            if val < 0.0 or val > 1.0:
                raise ValueError(f"{name} must be within [0, 1], got {val}")


@dataclass(frozen=True)
class ReliabilityFactors:
    """Reliability-oriented factors r_k in [0, 1] where 1 is optimal reliability.

    r_C = C_t
    r_E = 1 - E_t
    r_D = 1 - D_t
    r_Q = Q_t
    """

    r_C: float
    r_E: float
    r_D: float
    r_Q: float

    def __post_init__(self) -> None:
        for name, val in (
            ("r_C", self.r_C),
            ("r_E", self.r_E),
            ("r_D", self.r_D),
            ("r_Q", self.r_Q),
        ):
            if not isinstance(val, (int, float)):
                raise TypeError(f"{name} must be numeric, got {type(val).__name__}")
            if val < 0.0 or val > 1.0:
                raise ValueError(f"{name} must be within [0, 1], got {val}")


@dataclass(frozen=True)
class ReliabilityScore:
    """Comprehensive snapshot of reliability estimation at observation t."""

    reliability: float
    inputs: ReliabilityInputs
    factors: ReliabilityFactors
    weights: dict[str, float]
    epsilon: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.reliability <= 1.0):
            raise ValueError(f"reliability R_t must be in [0, 1], got {self.reliability}")


class BaseReliabilityEstimator(abc.ABC):
    """Abstract base class defining the uniform interface for reliability estimators."""

    @abc.abstractmethod
    def calculate(
        self,
        inputs: ReliabilityInputs | Mapping[str, float],
    ) -> ReliabilityScore:
        """Calculate reliability R_t from inputs without modifying internal state."""

    @abc.abstractmethod
    def update(
        self,
        *,
        probs: Mapping[int, float] | Any | None = None,
        confidence: float | None = None,
        drift_severity: float | None = None,
        quality: Any | None = None,
        error: float | None = None,
        y_true: int | None = None,
        y_pred: int | None = None,
        n_features: int | None = None,
        **kwargs: Any,
    ) -> ReliabilityScore:
        """Update estimator state and calculate current reliability R_t."""

    @abc.abstractmethod
    def update_error(self, error: float) -> float:
        """Incorporate delayed error feedback e_t and update recent error state E_t."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset internal temporal state to initial pre-stream condition."""

    @abc.abstractmethod
    def get_info(self) -> dict[str, Any]:
        """Return diagnostic, configuration, and state introspection metadata."""
