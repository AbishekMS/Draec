"""Drift-Aware Adaptive Edge-Cloud-Hybrid Orchestration for Non-Stationary IoT.

Module   : src/adaptation/validator.py
Phase    : Phase 9
Status   : IMPLEMENTED

Candidate model validation component.
Evaluates candidate vs active model on clean validation data (train2) to prevent regressions.
Strictly quarantined from final evaluation stream (test1).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

from src.adaptation.base import ValidationResult
from src.models.base import BaseModel


class CandidateValidator:
    """Validator assessing candidate model quality and non-regression against clean validation data.

    Guarantees:
    1. Zero access to test1: strictly evaluates on clean validation partition (train2).
    2. Compares candidate quality against active model quality.
    3. Enforces minimum metric and bounds regression relative to active model.
    4. Explicit acceptance or rejection outcome with detailed diagnostic metrics.
    """

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        val_data: tuple[pd.DataFrame | np.ndarray, Sequence[int] | np.ndarray] | None = None,
        *,
        metric: str = "macro_f1",
        minimum_metric: float = 0.70,
        max_regression_margin: float = 0.05,
    ) -> None:
        self.config = dict(config or {})

        val_cfg = self.config.get("adaptation", {}).get("validation", {})
        self.metric_name = str(val_cfg.get("metric", metric)).lower()
        self.minimum_metric = float(val_cfg.get("minimum_metric", minimum_metric))
        self.max_regression_margin = float(val_cfg.get("max_regression_margin", max_regression_margin))

        self._val_X: pd.DataFrame | np.ndarray | None = None
        self._val_y: np.ndarray | None = None

        if val_data is not None:
            self.set_validation_data(val_data[0], val_data[1])

        self._total_validations = 0
        self._passed_validations = 0
        self._failed_validations = 0

    def set_validation_data(
        self,
        X: pd.DataFrame | np.ndarray,
        y: Sequence[int] | np.ndarray,
        source: str = "train2",
    ) -> None:
        """Cache validation data, strictly enforcing that test1 cannot be used."""
        src_lower = str(source).strip().lower()
        if "test1" in src_lower:
            raise ValueError(
                f"Data contamination guard: source '{source}' cannot be used for candidate validation. "
                f"test1 is reserved strictly for Phase 10 final evaluation."
            )

        if isinstance(X, pd.DataFrame):
            self._val_X = X.copy()
        else:
            self._val_X = np.asarray(X, dtype=float).copy()
        self._val_y = np.asarray(y, dtype=int).copy()

    @property
    def has_validation_data(self) -> bool:
        return self._val_X is not None and self._val_y is not None

    def _compute_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if self.metric_name in ("macro_f1", "f1_macro", "f1"):
            return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        elif self.metric_name in ("accuracy", "acc"):
            return float(accuracy_score(y_true, y_pred))
        else:
            return float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    def validate(
        self,
        candidate_model: BaseModel,
        active_model: BaseModel,
        adaptation_val_data: tuple[pd.DataFrame | np.ndarray, Sequence[int] | np.ndarray] | None = None,
    ) -> ValidationResult:
        """Validate candidate model against active model on clean validation data.

        Returns:
            ValidationResult detailing candidate_valid, scores, delta, and reason.
        """
        if self._val_X is None or self._val_y is None:
            raise RuntimeError(
                "Cannot validate candidate model: no validation data configured. Call set_validation_data() first."
            )

        self._total_validations += 1

        # 1. Evaluate candidate on clean validation data (train2)
        y_pred_cand = candidate_model.predict(self._val_X)
        cand_score = self._compute_metric(self._val_y, y_pred_cand)

        # 2. Evaluate active model on clean validation data (train2)
        y_pred_active = active_model.predict(self._val_X)
        active_score = self._compute_metric(self._val_y, y_pred_active)

        delta = cand_score - active_score
        details: dict[str, Any] = {
            "validation_samples": len(self._val_y),
            "candidate_metric": cand_score,
            "active_metric": active_score,
            "metric_delta": delta,
            "minimum_metric_required": self.minimum_metric,
            "max_regression_margin": self.max_regression_margin,
        }

        # 3. Optional adaptation-regime validation check if provided
        cand_adapt_score = None
        active_adapt_score = None
        if adaptation_val_data is not None:
            X_ad, y_ad = adaptation_val_data
            y_ad_arr = np.asarray(y_ad, dtype=int)
            cand_ad_pred = candidate_model.predict(X_ad)
            active_ad_pred = active_model.predict(X_ad)
            cand_adapt_score = self._compute_metric(y_ad_arr, cand_ad_pred)
            active_adapt_score = self._compute_metric(y_ad_arr, active_ad_pred)
            details["candidate_adaptation_metric"] = cand_adapt_score
            details["active_adaptation_metric"] = active_adapt_score
            details["adaptation_metric_delta"] = cand_adapt_score - active_adapt_score

        # 4. Check acceptance conditions
        # Condition A: candidate meets minimum quality threshold
        if cand_score < self.minimum_metric:
            self._failed_validations += 1
            return ValidationResult(
                candidate_valid=False,
                metric_name=self.metric_name,
                candidate_metric=cand_score,
                active_metric=active_score,
                metric_delta=delta,
                status="REJECTED",
                reason=f"Candidate {self.metric_name} ({cand_score:.4f}) below minimum threshold ({self.minimum_metric:.4f}).",
                details=details,
            )

        # Condition B: candidate does not regress beyond allowed margin relative to active model
        allowed_floor = active_score - self.max_regression_margin
        if cand_score < allowed_floor:
            self._failed_validations += 1
            return ValidationResult(
                candidate_valid=False,
                metric_name=self.metric_name,
                candidate_metric=cand_score,
                active_metric=active_score,
                metric_delta=delta,
                status="REJECTED",
                reason=(
                    f"Candidate {self.metric_name} ({cand_score:.4f}) regresses by {abs(delta):.4f} "
                    f"beyond allowed margin {self.max_regression_margin:.4f} (floor: {allowed_floor:.4f})."
                ),
                details=details,
            )

        # Candidate meets all validation criteria -> ACCEPT
        self._passed_validations += 1
        return ValidationResult(
            candidate_valid=True,
            metric_name=self.metric_name,
            candidate_metric=cand_score,
            active_metric=active_score,
            metric_delta=delta,
            status="ACCEPTED",
            reason=f"Candidate accepted: {self.metric_name}={cand_score:.4f} (active={active_score:.4f}, delta={delta:+.4f}).",
            details=details,
        )

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_validations": self._total_validations,
            "passed_validations": self._passed_validations,
            "failed_validations": self._failed_validations,
            "metric_name": self.metric_name,
            "minimum_metric": self.minimum_metric,
            "max_regression_margin": self.max_regression_margin,
            "has_validation_data": self.has_validation_data,
        }
