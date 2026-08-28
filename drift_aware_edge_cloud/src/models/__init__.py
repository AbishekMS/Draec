"""Drift-Aware Adaptive Edge-Cloud-Hybrid Orchestration for Non-Stationary IoT.

Module   : src/models/__init__.py
Phase    : Phase 2
Status   : IMPLEMENTED

Supervised prediction models for Edge and Cloud tiers.
"""

from __future__ import annotations

from src.models.base import BaseModel, InputDimensionError, ModelError, NotTrainedError
from src.models.cloud_model import CloudModel, CloudXGBoost
from src.models.edge_model import EdgeHoeffdingTree, EdgeModel
from src.models.trainer import (
    evaluate_model,
    evaluate_predictions,
    extract_partition_labels,
    get_key_by_role,
    load_causal_eval_data,
    load_causal_train_data,
    train_cloud_model,
    train_edge_model,
)

__all__ = [
    "BaseModel",
    "ModelError",
    "NotTrainedError",
    "InputDimensionError",
    "EdgeHoeffdingTree",
    "EdgeModel",
    "CloudXGBoost",
    "CloudModel",
    "load_causal_train_data",
    "load_causal_eval_data",
    "extract_partition_labels",
    "get_key_by_role",
    "train_edge_model",
    "train_cloud_model",
    "evaluate_model",
    "evaluate_predictions",
]
