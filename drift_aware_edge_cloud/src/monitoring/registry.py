"""Drift-Aware Adaptive Edge-Cloud-Hybrid Orchestration for Non-Stationary IoT.

Module   : src/monitoring/registry.py
Phase    : Phase 7
Status   : IMPLEMENTED

Model state registry and metadata management component. Tracks model instances,
versions, feature contracts, and execution outcomes without mutating model parameters.
"""

from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

from src.models.base import BaseModel
from src.monitoring.base import ModelHealthStatus, ModelMetadata


class ModelRegistry:
    """Lightweight registry tracking model metadata, lifecycle state, and health.

    Maintains execution and health metadata for Edge and Cloud models.
    Does NOT perform training, parameter updating, model replacement, or deployment.
    """

    def __init__(self) -> None:
        self._models: dict[str, Any] = {}
        self._metadata: dict[str, ModelMetadata] = {}

    def register_model(
        self,
        model: Any,
        model_id: str,
        execution_location: str,
        version: str = "1.0.0",
        feature_names: Sequence[str] | None = None,
        n_features: int | None = None,
    ) -> ModelMetadata:
        """Register a model instance and initialize its observational metadata."""
        if not model_id or not isinstance(model_id, str):
            raise ValueError(f"model_id must be a non-empty string, got {model_id}")

        loc = execution_location.strip().lower()
        if loc not in ("edge", "cloud"):
            raise ValueError(f"execution_location must be 'edge' or 'cloud', got '{execution_location}'")

        model_name = getattr(model, "model_name", getattr(model, "name", type(model).__name__))
        model_type = type(model).__name__

        # Resolve feature contract
        resolved_feat_names = feature_names or getattr(model, "feature_names", None)
        if resolved_feat_names is not None:
            resolved_feat_names = tuple(resolved_feat_names)

        resolved_n_features = n_features or getattr(model, "n_features", None)
        if resolved_n_features is None and resolved_feat_names is not None:
            resolved_n_features = len(resolved_feat_names)

        meta = ModelMetadata(
            model_id=model_id,
            model_name=model_name,
            model_type=model_type,
            execution_location=loc,
            model_version=version,
            status=ModelHealthStatus.HEALTHY,
            active=True,
            created_at=time.time(),
            n_features=resolved_n_features,
            feature_names=resolved_feat_names,
        )

        self._models[model_id] = model
        self._metadata[model_id] = meta
        return meta

    def get_model(self, model_id: str) -> Any:
        """Retrieve the registered model instance by identifier."""
        if model_id not in self._models:
            raise KeyError(f"Model '{model_id}' not found in registry. Registered: {list(self._models.keys())}")
        return self._models[model_id]

    def get_metadata(self, model_id: str) -> ModelMetadata:
        """Retrieve metadata for a registered model by identifier."""
        if model_id not in self._metadata:
            raise KeyError(f"Model '{model_id}' not found in registry. Registered: {list(self._metadata.keys())}")
        return self._metadata[model_id]

    def has_model(self, model_id: str) -> bool:
        """Check if a model identifier exists in the registry."""
        return model_id in self._models

    def list_models(self) -> list[ModelMetadata]:
        """Return a list of metadata for all registered models."""
        return list(self._metadata.values())

    def update_status(self, model_id: str, status: ModelHealthStatus | str) -> ModelMetadata:
        """Update the observational health status of a model (non-actionable metadata)."""
        meta = self.get_metadata(model_id)
        if not isinstance(status, ModelHealthStatus):
            status = ModelHealthStatus.from_str(status)
        meta.status = status
        return meta

    def set_active(self, model_id: str, active: bool) -> ModelMetadata:
        """Set the active/inactive flag for a registered model."""
        meta = self.get_metadata(model_id)
        meta.active = bool(active)
        return meta

    def record_execution(
        self,
        model_id: str,
        success: bool,
        latency_s: float | None = None,
        status: str | None = None,
        error: str | None = None,
    ) -> None:
        """Record an execution event outcome for a model without modifying model weights."""
        if model_id not in self._metadata:
            return
        meta = self._metadata[model_id]
        meta.total_executions += 1
        if success:
            meta.successful_executions += 1
        else:
            meta.failed_executions += 1
            meta.last_error = str(error) if error else "Execution failure"

        if latency_s is not None:
            meta.last_latency_s = float(latency_s)
        if status is not None:
            meta.last_execution_status = str(status)

    def get_health_summary(self) -> dict[str, str]:
        """Return a mapping of model_id -> status string."""
        return {m_id: meta.status.value for m_id, meta in self._metadata.items()}

    def reset_metrics(self) -> None:
        """Reset execution counters and last status across all registered models."""
        for meta in self._metadata.values():
            meta.total_executions = 0
            meta.successful_executions = 0
            meta.failed_executions = 0
            meta.last_latency_s = None
            meta.last_execution_status = None
            meta.last_error = None
            meta.status = ModelHealthStatus.HEALTHY
