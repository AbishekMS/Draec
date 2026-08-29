"""Drift-Aware Adaptive Edge-Cloud-Hybrid Orchestration for Non-Stationary IoT.

Module   : src/adaptation/__init__.py
Phase    : Phase 9
Status   : IMPLEMENTED

DRAEC Model Adaptation & Retraining Layer public API.
"""

from src.adaptation.base import (
    AdaptationResult,
    AdaptationState,
    FeedbackRecord,
    ModelVersionRecord,
    ValidationResult,
)
from src.adaptation.deployment import AtomicModelDeployer
from src.adaptation.feedback import FeedbackQueue
from src.adaptation.manager import AdaptationManager
from src.adaptation.retrainer import CloudRetrainer
from src.adaptation.validator import CandidateValidator

__all__ = [
    "AdaptationState",
    "FeedbackRecord",
    "ValidationResult",
    "ModelVersionRecord",
    "AdaptationResult",
    "FeedbackQueue",
    "CloudRetrainer",
    "CandidateValidator",
    "AtomicModelDeployer",
    "AdaptationManager",
]
