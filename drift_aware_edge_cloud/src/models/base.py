"""Drift-Aware Adaptive Edge-Cloud-Hybrid Orchestration for Non-Stationary IoT.

Module   : src/models/base.py
Phase    : Phase 2
Status   : IMPLEMENTED

Common model interface and abstract base class for Edge and Cloud models.
"""

from __future__ import annotations

import abc
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


class ModelError(RuntimeError):
    """Base exception for model initialization, training, or inference errors."""


class NotTrainedError(ModelError):
    """Raised when prediction is requested before initial supervised training."""


class InputDimensionError(ModelError):
    """Raised when input feature shape or columns do not match model expectation."""


class BaseModel(abc.ABC):
    """Abstract base class defining the uniform interface for Edge and Cloud models.

    Both Edge (River Hoeffding Tree) and Cloud (XGBoost) implement this interface,
    ensuring downstream orchestration components (WDS, reliability, LRI) can consume
    either model without framework-specific coupling.
    """

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._is_trained = False
        self._n_features: int | None = None
        self._feature_names: tuple[str, ...] | None = None
        self._classes: tuple[int, ...] = (0, 1)
        self._last_inference_time_s: float | None = None
        self._total_samples_inferred: int = 0
        self._total_inference_time_s: float = 0.0

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def n_features(self) -> int | None:
        return self._n_features

    @property
    def feature_names(self) -> tuple[str, ...] | None:
        return self._feature_names

    @property
    def classes_(self) -> tuple[int, ...]:
        return self._classes

    @property
    def last_inference_time_s(self) -> float | None:
        return self._last_inference_time_s

    @property
    def total_samples_inferred(self) -> int:
        return self._total_samples_inferred

    @property
    def total_inference_time_s(self) -> float:
        return self._total_inference_time_s

    @property
    def mean_inference_time_per_sample_s(self) -> float | None:
        if self._total_samples_inferred == 0:
            return None
        return self._total_inference_time_s / self._total_samples_inferred

    def _record_inference_time(self, elapsed_s: float, n_samples: int = 1) -> None:
        self._last_inference_time_s = elapsed_s
        self._total_inference_time_s += elapsed_s
        self._total_samples_inferred += n_samples

    def _check_is_trained(self) -> None:
        if not self._is_trained:
            raise NotTrainedError(
                f"Model '{self._model_name}' has not been trained yet. Call fit() first."
            )

    def _check_feature_alignment(self, X: pd.DataFrame | np.ndarray) -> None:
        if self._n_features is None:
            return
        if isinstance(X, pd.DataFrame):
            n_cols = X.shape[1]
            if n_cols != self._n_features:
                raise InputDimensionError(
                    f"Feature count mismatch for '{self._model_name}': expected {self._n_features} "
                    f"features, got {n_cols}."
                )
            if self._feature_names is not None:
                x_cols = tuple(X.columns)
                if x_cols != self._feature_names:
                    missing = set(self._feature_names) - set(x_cols)
                    extra = set(x_cols) - set(self._feature_names)
                    if missing or extra:
                        raise InputDimensionError(
                            f"Feature column mismatch for '{self._model_name}': "
                            f"missing={sorted(missing)}, extra={sorted(extra)}"
                        )
        elif isinstance(X, np.ndarray):
            n_cols = X.shape[1] if X.ndim > 1 else (X.shape[0] if X.ndim == 1 else 0)
            if n_cols != self._n_features:
                raise InputDimensionError(
                    f"Feature count mismatch for '{self._model_name}': expected {self._n_features} "
                    f"features, got {n_cols}."
                )

    @abc.abstractmethod
    def fit(self, X: pd.DataFrame | np.ndarray, y: Sequence[int] | np.ndarray) -> BaseModel:
        """Fit model on causal training observations."""

    @abc.abstractmethod
    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Batch prediction returning 1D numpy array of class labels."""

    @abc.abstractmethod
    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Batch probability prediction returning 2D numpy array of shape (N, 2)."""

    @abc.abstractmethod
    def predict_one(self, x: Mapping[str, float] | Sequence[float] | pd.Series) -> int:
        """Predict class label for a single observation."""

    @abc.abstractmethod
    def predict_proba_one(self, x: Mapping[str, float] | Sequence[float] | pd.Series) -> dict[int, float]:
        """Predict class probabilities for a single observation as {0: p0, 1: p1}."""

    @abc.abstractmethod
    def get_info(self) -> dict[str, Any]:
        """Return diagnostic, resource, and configuration metadata."""
