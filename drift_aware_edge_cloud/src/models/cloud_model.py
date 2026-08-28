"""Drift-Aware Adaptive Edge-Cloud-Hybrid Orchestration for Non-Stationary IoT.

Module   : src/models/cloud_model.py
Phase    : Phase 2
Status   : IMPLEMENTED

High-capacity Cloud prediction model wrapping XGBoost Classifier.
Supports batch supervised training on causal baseline data, high-throughput
batch inference, probability estimation, and inference latency tracking.
"""

from __future__ import annotations

import pickle
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.models.base import BaseModel, InputDimensionError
from src.utils import seed as seedmod


class CloudXGBoost(BaseModel):
    """Cloud supervised prediction model using XGBoost Classifier.

    Designed for high capacity, accurate batch inference on cloud nodes.
    Trained on causal baseline observations using the verified Phase 1 feature representation.
    """

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        *,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        tree_method: str = "hist",
        eval_metric: str = "logloss",
        random_state: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name="CloudXGBoostClassifier")

        cfg_params: dict[str, Any] = {}
        seed_val = random_state
        if config is not None:
            r_phases = config.get("reserved_for_later_phases") or {}
            p2_models = r_phases.get("phase_2_models")
            if isinstance(p2_models, Mapping):
                cfg_params.update(p2_models.get("cloud") or {})
            models_sec = config.get("models")
            if isinstance(models_sec, Mapping):
                cfg_params.update(models_sec.get("cloud") or {})

            if seed_val is None:
                try:
                    seed_val = seedmod.master_seed(config)
                except Exception:
                    seed_val = 42

        if seed_val is None:
            seed_val = 42

        self._params: dict[str, Any] = {
            "n_estimators": kwargs.get("n_estimators", cfg_params.get("n_estimators", n_estimators)),
            "max_depth": kwargs.get("max_depth", cfg_params.get("max_depth", max_depth)),
            "learning_rate": kwargs.get("learning_rate", cfg_params.get("learning_rate", learning_rate)),
            "subsample": kwargs.get("subsample", cfg_params.get("subsample", subsample)),
            "tree_method": kwargs.get("tree_method", cfg_params.get("tree_method", tree_method)),
            "eval_metric": kwargs.get("eval_metric", cfg_params.get("eval_metric", eval_metric)),
            "random_state": seed_val,
        }
        for k, v in kwargs.items():
            if k not in self._params:
                self._params[k] = v

        self._model = XGBClassifier(**self._params)
        self._n_samples_trained: int = 0

    @property
    def raw_model(self) -> XGBClassifier:
        """Direct access to the underlying XGBClassifier estimator."""
        return self._model

    @property
    def n_samples_trained(self) -> int:
        return self._n_samples_trained

    def _prepare_input_frame(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        """Ensure input is a DataFrame with expected columns for XGBoost."""
        if isinstance(X, pd.DataFrame):
            self._check_feature_alignment(X)
            if self._feature_names is not None:
                return X[list(self._feature_names)]
            return X
        if isinstance(X, np.ndarray):
            self._check_feature_alignment(X)
            if self._feature_names is not None:
                return pd.DataFrame(X, columns=list(self._feature_names))
            return pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        raise TypeError(f"Unsupported feature matrix type: {type(X).__name__}")

    def fit(self, X: pd.DataFrame | np.ndarray, y: Sequence[int] | np.ndarray) -> CloudXGBoost:
        """Batch supervised training on causal training observations."""
        if isinstance(X, pd.DataFrame):
            self._n_features = X.shape[1]
            self._feature_names = tuple(X.columns)
            frame = X
        elif isinstance(X, np.ndarray):
            self._n_features = X.shape[1]
            self._feature_names = tuple(f"f{i}" for i in range(X.shape[1]))
            frame = pd.DataFrame(X, columns=list(self._feature_names))
        else:
            raise TypeError(f"Unsupported feature matrix type: {type(X).__name__}")

        y_arr = np.asarray(y, dtype=int)
        if len(frame) != len(y_arr):
            raise ValueError(f"Length mismatch: X has {len(frame)} rows, y has {len(y_arr)}")

        self._model = XGBClassifier(**self._params)
        self._model.fit(frame, y_arr)
        self._n_samples_trained = len(y_arr)
        self._is_trained = True
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Batch prediction returning 1D numpy array of predicted class labels."""
        self._check_is_trained()
        frame = self._prepare_input_frame(X)

        t0 = time.perf_counter()
        preds = self._model.predict(frame)
        elapsed = time.perf_counter() - t0
        self._record_inference_time(elapsed, len(frame))

        return np.asarray(preds, dtype=int)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Batch probability prediction returning 2D numpy array of shape (N, 2)."""
        self._check_is_trained()
        frame = self._prepare_input_frame(X)

        t0 = time.perf_counter()
        proba = self._model.predict_proba(frame)
        elapsed = time.perf_counter() - t0
        self._record_inference_time(elapsed, len(frame))

        if proba.shape[1] == 1:
            # Degenerate single-class edge case in XGBoost
            single_class = int(self._model.classes_[0])
            out = np.zeros((len(frame), 2), dtype=float)
            out[:, single_class] = 1.0
            return out

        return np.asarray(proba, dtype=float)

    def predict_one(self, x: Mapping[str, float] | Sequence[float] | pd.Series) -> int:
        """Predict class label for a single observation."""
        self._check_is_trained()
        if isinstance(x, dict):
            row = pd.DataFrame([x])
        elif isinstance(x, pd.Series):
            row = pd.DataFrame([x.to_dict()])
        elif isinstance(x, (list, tuple, np.ndarray)):
            names = self._feature_names or tuple(f"f{i}" for i in range(len(x)))
            row = pd.DataFrame([dict(zip(names, x))])
        else:
            raise TypeError(f"Unsupported sample type: {type(x).__name__}")

        preds = self.predict(row)
        return int(preds[0])

    def predict_proba_one(self, x: Mapping[str, float] | Sequence[float] | pd.Series) -> dict[int, float]:
        """Predict class probabilities for a single observation as {0: p0, 1: p1}."""
        self._check_is_trained()
        if isinstance(x, dict):
            row = pd.DataFrame([x])
        elif isinstance(x, pd.Series):
            row = pd.DataFrame([x.to_dict()])
        elif isinstance(x, (list, tuple, np.ndarray)):
            names = self._feature_names or tuple(f"f{i}" for i in range(len(x)))
            row = pd.DataFrame([dict(zip(names, x))])
        else:
            raise TypeError(f"Unsupported sample type: {type(x).__name__}")

        probas = self.predict_proba(row)[0]
        return {0: float(probas[0]), 1: float(probas[1])}

    def get_info(self) -> dict[str, Any]:
        """Return diagnostic, resource, and configuration metadata."""
        size_bytes: int | None = None
        try:
            size_bytes = len(pickle.dumps(self._model))
        except Exception:
            pass

        return {
            "model_name": self._model_name,
            "is_trained": self._is_trained,
            "n_features": self._n_features,
            "feature_names": list(self._feature_names) if self._feature_names else None,
            "n_samples_trained": self._n_samples_trained,
            "classes": list(self._classes),
            "hyperparameters": dict(self._params),
            "n_estimators": self._params.get("n_estimators"),
            "max_depth": self._params.get("max_depth"),
            "model_size_bytes": size_bytes,
            "last_inference_time_s": self._last_inference_time_s,
            "mean_inference_time_per_sample_s": self.mean_inference_time_per_sample_s,
        }


# Convenience alias
CloudModel = CloudXGBoost
