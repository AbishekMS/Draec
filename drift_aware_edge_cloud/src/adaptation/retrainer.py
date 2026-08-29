"""Drift-Aware Adaptive Edge-Cloud-Hybrid Orchestration for Non-Stationary IoT.

Module   : src/adaptation/retrainer.py
Phase    : Phase 9
Status   : IMPLEMENTED

Cloud model retrainer implementing anti-catastrophic forgetting.
Combines representative baseline data with causally eligible feedback to fit candidate Cloud models.
"""

from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.adaptation.base import FeedbackRecord
from src.models.cloud_model import CloudXGBoost


class CloudRetrainer:
    """Cloud model retraining component with anti-catastrophic forgetting protection.

    Creates and fits a candidate CloudXGBoost model using a hybrid dataset:
        D_candidate = D_baseline_representative UNION D_causally_eligible_feedback

    Guarantees:
    1. Preserves Cloud model type (CloudXGBoost / XGBoost).
    2. Preserves baseline knowledge via representative baseline sampling.
    3. Memory bounded via max_baseline_samples and max_feedback_samples caps.
    4. Deterministic training using seeded PRNG.
    5. Returns candidate without automatically activating or deploying it.
    """

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        baseline_data: tuple[pd.DataFrame | np.ndarray, Sequence[int] | np.ndarray] | None = None,
        *,
        min_feedback_samples: int = 50,
        max_feedback_samples: int = 1000,
        max_baseline_samples: int = 500,
        random_seed: int = 42,
    ) -> None:
        self.config = dict(config or {})

        adapt_cfg = self.config.get("adaptation", {})
        trig_cfg = adapt_cfg.get("trigger", {})
        train_cfg = adapt_cfg.get("training", {})

        self.min_feedback_samples = int(trig_cfg.get("min_feedback_samples", min_feedback_samples))
        self.max_feedback_samples = int(train_cfg.get("max_feedback_samples", max_feedback_samples))
        self.max_baseline_samples = int(train_cfg.get("baseline_sample_size", max_baseline_samples))
        self.random_seed = int(train_cfg.get("random_seed", random_seed))

        self._baseline_X: np.ndarray | None = None
        self._baseline_y: np.ndarray | None = None
        self._feature_names: tuple[str, ...] | None = None

        if baseline_data is not None:
            self.set_baseline_data(baseline_data[0], baseline_data[1])

        self._total_retrainings = 0
        self._last_retrain_duration_s = 0.0

    def set_baseline_data(
        self,
        X: pd.DataFrame | np.ndarray,
        y: Sequence[int] | np.ndarray,
    ) -> None:
        """Cache a bounded representative sample of baseline data (train1)."""
        rng = np.random.default_rng(self.random_seed)

        if isinstance(X, pd.DataFrame):
            self._feature_names = tuple(X.columns)
            X_arr = X.to_numpy(dtype=float)
        else:
            X_arr = np.asarray(X, dtype=float)

        y_arr = np.asarray(y, dtype=int)

        n_samples = len(X_arr)
        if n_samples <= self.max_baseline_samples:
            self._baseline_X = X_arr.copy()
            self._baseline_y = y_arr.copy()
        else:
            # Deterministic stratified or random sample up to max_baseline_samples
            indices = rng.choice(n_samples, size=self.max_baseline_samples, replace=False)
            indices.sort()
            self._baseline_X = X_arr[indices].copy()
            self._baseline_y = y_arr[indices].copy()

    @property
    def has_baseline_data(self) -> bool:
        return self._baseline_X is not None and self._baseline_y is not None

    def retrain(
        self,
        eligible_feedback: Sequence[FeedbackRecord],
        parent_version: str = "v1",
        candidate_version: str = "v2",
    ) -> tuple[CloudXGBoost, dict[str, Any]]:
        """Train a candidate CloudXGBoost model using baseline + feedback data.

        Returns:
            (candidate_model, training_metadata)
        """
        n_feedback = len(eligible_feedback)
        if n_feedback < self.min_feedback_samples:
            raise ValueError(
                f"Insufficient eligible feedback for retraining: got {n_feedback}, minimum required is {self.min_feedback_samples}."
            )

        t_start = time.perf_counter()

        # 1. Extract feature arrays and labels from feedback records
        capped_feedback = eligible_feedback[-self.max_feedback_samples :]
        X_feed_list = []
        y_feed_list = []
        for rec in capped_feedback:
            if not rec.is_labeled or rec.label is None:
                continue
            feats = rec.features
            if isinstance(feats, (pd.Series, pd.DataFrame)):
                arr = feats.to_numpy(dtype=float).flatten()
            elif isinstance(feats, Mapping):
                # Map according to feature_names if available
                if self._feature_names:
                    arr = np.array([float(feats.get(k, 0.0)) for k in self._feature_names], dtype=float)
                else:
                    arr = np.array(list(feats.values()), dtype=float)
            else:
                arr = np.asarray(feats, dtype=float).flatten()
            X_feed_list.append(arr)
            y_feed_list.append(int(rec.label))

        X_feed = np.vstack(X_feed_list)
        y_feed = np.array(y_feed_list, dtype=int)

        # 2. Merge with representative baseline sample (Anti-Catastrophic Forgetting)
        n_base = 0
        if self._baseline_X is not None and self._baseline_y is not None:
            n_base = len(self._baseline_X)
            X_train = np.vstack([self._baseline_X, X_feed])
            y_train = np.concatenate([self._baseline_y, y_feed])
        else:
            X_train = X_feed
            y_train = y_feed

        # 3. Format DataFrame if feature names are present
        if self._feature_names and len(self._feature_names) == X_train.shape[1]:
            train_df = pd.DataFrame(X_train, columns=list(self._feature_names))
        else:
            train_df = pd.DataFrame(X_train)

        # 4. Instantiate fresh candidate CloudXGBoost model
        cloud_cfg = self.config.get("adaptation", {}).get("training", {}).get("cloud_model", {})
        candidate = CloudXGBoost(
            config=self.config,
            random_state=self.random_seed,
            **cloud_cfg,
        )

        # 5. Fit candidate
        candidate.fit(train_df, y_train)
        duration_s = time.perf_counter() - t_start

        self._total_retrainings += 1
        self._last_retrain_duration_s = duration_s

        meta = {
            "candidate_version": candidate_version,
            "parent_version": parent_version,
            "total_samples_trained": len(y_train),
            "baseline_samples_used": n_base,
            "feedback_samples_used": len(y_feed),
            "training_duration_s": duration_s,
            "feature_count": X_train.shape[1],
            "random_seed": self.random_seed,
        }

        return candidate, meta

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_retrainings": self._total_retrainings,
            "last_retrain_duration_s": self._last_retrain_duration_s,
            "has_baseline_data": self.has_baseline_data,
            "baseline_samples_cached": len(self._baseline_X) if self._baseline_X is not None else 0,
            "min_feedback_samples": self.min_feedback_samples,
            "max_feedback_samples": self.max_feedback_samples,
        }
