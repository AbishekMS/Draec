"""Drift-Aware Adaptive Edge-Cloud-Hybrid Orchestration for Non-Stationary IoT.

Module   : src/deployment/__init__.py
Phase    : Phase 8
Status   : IMPLEMENTED

DRAEC Edge-Cloud Deployment & Network Execution Layer public API.
"""

from src.deployment.base import (
    DeploymentExecutionResult,
    NetworkPacket,
    RuntimeState,
    TransmissionResult,
    TransmissionStatus,
)
from src.deployment.environment import DeploymentEnvironment
from src.deployment.network import NetworkSimulator
from src.deployment.runtimes import CloudRuntime, EdgeRuntime

__all__ = [
    "TransmissionStatus",
    "RuntimeState",
    "NetworkPacket",
    "TransmissionResult",
    "DeploymentExecutionResult",
    "NetworkSimulator",
    "EdgeRuntime",
    "CloudRuntime",
    "DeploymentEnvironment",
]
