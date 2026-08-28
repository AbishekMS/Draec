"""Drift-Aware Adaptive Edge-Cloud-Hybrid Orchestration for Non-Stationary IoT.

Package  : src/reliability
Phase    : Phase 4
Status   : IMPLEMENTED

Phase 4 Prediction Reliability Estimation Layer:
- Multi-factor reliability inputs and contracts (base.py)
- Harmonic mean reliability calculation R_t in [0, 1] (estimator.py)
- Prediction confidence C_t
- Instantaneous error e_t and recent error E_t with delayed feedback support
- Smoothed drift severity D_t integration
- Dataset-independent data/sensor quality Q_t
"""

from __future__ import annotations

from src.reliability.base import (
    BaseReliabilityEstimator,
    ReliabilityFactors,
    ReliabilityInputs,
    ReliabilityScore,
)
from src.reliability.estimator import (
    ReliabilityEstimator,
    compute_confidence,
    compute_harmonic_reliability,
    compute_instantaneous_error,
    compute_quality,
)

__all__ = [
    "BaseReliabilityEstimator",
    "ReliabilityFactors",
    "ReliabilityInputs",
    "ReliabilityScore",
    "ReliabilityEstimator",
    "compute_confidence",
    "compute_harmonic_reliability",
    "compute_instantaneous_error",
    "compute_quality",
]
