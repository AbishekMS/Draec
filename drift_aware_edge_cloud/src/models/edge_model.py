"""Drift-Aware Adaptive Edge-Cloud-Hybrid Orchestration for Non-Stationary IoT.

Module   : src/models/edge_model.py
Phase    : Phase 2
Status   : IMPLEMENTED

Lightweight Edge prediction model wrapping River's HoeffdingTreeClassifier.
Supports online incremental learning, single-observation and batch inference,
probability estimation, and inference latency tracking.
"""

from __future__ import annotations

import pickle
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from river.tree import HoeffdingTreeClassifier

from src.models.base import BaseModel, InputDimensionError


class EdgeHoeffdingTree(BaseModel):
    """Edge supervised prediction model using River's Hoeffding Tree Classifier.

    Designed for lightweight, low-latency online inference on edge nodes.
    Supports incremental learning on streaming observations without full retraining.
    """

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        *,
        grace_period: int = 200,
        max_depth: int | None = None,
        split_criterion: str = "info_gain",
        delta: float = 1e-07,
        tau: float = 0.05,
        leaf_prediction: str = "nba",
        nb_threshold: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name="RiverHoeffdingTreeClassifier")

        # Resolve hyperparameters from config if provided, letting kwargs override
        cfg_params: dict[str, Any] = {}
        if config is not None:
            r_phases = config.get("reserved_for_later_phases") or {}
            p2_models = r_phases.get("phase_2_models")
            if isinstance(p2_models, Mapping):
                cfg_params.update(p2_models.get("edge") or {})
            models_sec = config.get("models")
            if isinstance(models_sec, Mapping):
                cfg_params.update(models_sec.get("edge") or {})

        self._params: dict[str, Any] = {
            "grace_period": int(kwargs.get("grace_period", cfg_params.get("grace_period", grace_period))),
            "max_depth": (
                int(kwargs.get("max_depth", cfg_params.get("max_depth", max_depth)))
                if kwargs.get("max_depth", cfg_params.get("max_depth", max_depth)) is not None
                else None
            ),
            "split_criterion": kwargs.get("split_criterion", cfg_params.get("split_criterion", split_criterion)),
            "delta": float(kwargs.get("delta", cfg_params.get("delta", delta))),
            "tau": float(kwargs.get("tau", cfg_params.get("tau", tau))),
            "leaf_prediction": kwargs.get("leaf_prediction", cfg_params.get("leaf_prediction", leaf_prediction)),
            "nb_threshold": int(kwargs.get("nb_threshold", cfg_params.get("nb_threshold", nb_threshold))),
        }
        for k, v in kwargs.items():
            if k not in self._params:
                self._params[k] = v

        self._model = HoeffdingTreeClassifier(**self._params)
        self._n_samples_trained: int = 0

    @property
    def raw_model(self) -> HoeffdingTreeClassifier:
        """Direct access to the underlying River estimator."""
        return self._model

    @property
    def n_samples_trained(self) -> int:
        return self._n_samples_trained

    def _convert_sample_to_dict(self, x: Mapping[str, float] | Sequence[float] | pd.Series) -> dict[str, float]:
        """Convert any supported input representation into River's dict format."""
        if isinstance(x, dict):
            return {str(k): float(v) for k, v in x.items()}
        if isinstance(x, pd.Series):
            return {str(k): float(v) for k, v in x.to_dict().items()}
        if isinstance(x, (list, tuple, np.ndarray)):
            if self._feature_names is not None:
                if len(x) != len(self._feature_names):
                    raise InputDimensionError(
                        f"Expected {len(self._feature_names)} features, got {len(x)}"
                    )
                return {col: float(val) for col, val in zip(self._feature_names, x)}
            return {f"f{i}": float(v) for i, v in enumerate(x)}
        raise TypeError(f"Unsupported sample type: {type(x).__name__}")

    def learn_one(self, x: Mapping[str, float] | Sequence[float] | pd.Series, y: int) -> EdgeHoeffdingTree:
        """Incrementally train on a single labeled observation."""
        x_dict = self._convert_sample_to_dict(x)
        if not self._is_trained:
            self._n_features = len(x_dict)
            self._feature_names = tuple(x_dict.keys())
            self._is_trained = True

        self._model.learn_one(x_dict, int(y))
        self._n_samples_trained += 1
        return self

    def learn_many(self, X: pd.DataFrame | np.ndarray, y: Sequence[int] | np.ndarray) -> EdgeHoeffdingTree:
        """Incrementally train on a sequence of labeled observations."""
        self._check_feature_alignment(X)
        if isinstance(X, pd.DataFrame):
            records = X.to_dict(orient="records")
        elif isinstance(X, np.ndarray):
            if self._feature_names is not None:
                records = [dict(zip(self._feature_names, row)) for row in X]
            else:
                self._n_features = X.shape[1]
                self._feature_names = tuple(f"f{i}" for i in range(X.shape[1]))
                records = [dict(zip(self._feature_names, row)) for row in X]
        else:
            raise TypeError(f"Unsupported feature matrix type: {type(X).__name__}")

        y_arr = np.asarray(y, dtype=int)
        if len(records) != len(y_arr):
            raise ValueError(f"Length mismatch: X has {len(records)} rows, y has {len(y_arr)}")

        if not self._is_trained and records:
            self._n_features = len(records[0])
            self._feature_names = tuple(records[0].keys())
            self._is_trained = True

        for xi, yi in zip(records, y_arr):
            self._model.learn_one(xi, int(yi))
            self._n_samples_trained += 1
        return self

    def fit(self, X: pd.DataFrame | np.ndarray, y: Sequence[int] | np.ndarray) -> EdgeHoeffdingTree:
        """Initial supervised training on causal training observations."""
        if isinstance(X, pd.DataFrame):
            self._n_features = X.shape[1]
            self._feature_names = tuple(X.columns)
        elif isinstance(X, np.ndarray):
            self._n_features = X.shape[1]
            self._feature_names = tuple(f"f{i}" for i in range(X.shape[1]))
        else:
            raise TypeError(f"Unsupported feature matrix type: {type(X).__name__}")

        self._model = HoeffdingTreeClassifier(**self._params)
        self._n_samples_trained = 0
        self._is_trained = False
        return self.learn_many(X, y)

    def predict_one(self, x: Mapping[str, float] | Sequence[float] | pd.Series) -> int:
        """Predict class label for a single observation with inference timing."""
        self._check_is_trained()
        x_dict = self._convert_sample_to_dict(x)
        t0 = time.perf_counter()
        pred = self._model.predict_one(x_dict)
        elapsed = time.perf_counter() - t0
        self._record_inference_time(elapsed, 1)

        if pred is None:
            return 0
        return int(pred)

    def predict_proba_one(self, x: Mapping[str, float] | Sequence[float] | pd.Series) -> dict[int, float]:
        """Predict class probabilities for a single observation as {0: p0, 1: p1}."""
        self._check_is_trained()
        x_dict = self._convert_sample_to_dict(x)
        t0 = time.perf_counter()
        proba = self._model.predict_proba_one(x_dict) or {}
        elapsed = time.perf_counter() - t0
        self._record_inference_time(elapsed, 1)

        p0 = float(proba.get(0, 0.0))
        p1 = float(proba.get(1, 0.0))
        total = p0 + p1
        if total > 0:
            return {0: p0 / total, 1: p1 / total}
        return {0: 1.0, 1: 0.0}

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Batch prediction returning 1D numpy array of predicted class labels."""
        self._check_is_trained()
        self._check_feature_alignment(X)

        if isinstance(X, pd.DataFrame):
            records = X.to_dict(orient="records")
        elif isinstance(X, np.ndarray):
            names = self._feature_names or tuple(f"f{i}" for i in range(X.shape[1]))
            records = [dict(zip(names, row)) for row in X]
        else:
            raise TypeError(f"Unsupported feature matrix type: {type(X).__name__}")

        t0 = time.perf_counter()
        preds = [self._model.predict_one(xi) for xi in records]
        elapsed = time.perf_counter() - t0
        self._record_inference_time(elapsed, len(records))

        out = np.zeros(len(preds), dtype=int)
        for i, p in enumerate(preds):
            if p is not None:
                out[i] = int(p)
        return out

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Batch probability prediction returning 2D numpy array of shape (N, 2)."""
        self._check_is_trained()
        self._check_feature_alignment(X)

        if isinstance(X, pd.DataFrame):
            records = X.to_dict(orient="records")
        elif isinstance(X, np.ndarray):
            names = self._feature_names or tuple(f"f{i}" for i in range(X.shape[1]))
            records = [dict(zip(names, row)) for row in X]
        else:
            raise TypeError(f"Unsupported feature matrix type: {type(X).__name__}")

        t0 = time.perf_counter()
        probas = [self._model.predict_proba_one(xi) or {} for xi in records]
        elapsed = time.perf_counter() - t0
        self._record_inference_time(elapsed, len(records))

        out = np.zeros((len(probas), 2), dtype=float)
        for i, pr in enumerate(probas):
            p0 = float(pr.get(0, 0.0))
            p1 = float(pr.get(1, 0.0))
            tot = p0 + p1
            if tot > 0:
                out[i, 0] = p0 / tot
                out[i, 1] = p1 / tot
            else:
                out[i, 0] = 1.0
                out[i, 1] = 0.0
        return out

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
            "tree_height": getattr(self._model, "height", None),
            "n_nodes": getattr(self._model, "n_nodes", None),
            "n_leaves": getattr(self._model, "n_leaves", None),
            "model_size_bytes": size_bytes,
            "last_inference_time_s": self._last_inference_time_s,
            "mean_inference_time_per_sample_s": self.mean_inference_time_per_sample_s,
        }


# Convenience alias
EdgeModel = EdgeHoeffdingTree
