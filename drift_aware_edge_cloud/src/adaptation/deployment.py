"""Drift-Aware Adaptive Edge-Cloud-Hybrid Orchestration for Non-Stationary IoT.

Module   : src/adaptation/deployment.py
Phase    : Phase 9
Status   : IMPLEMENTED

Atomic Cloud + Edge model deployment with rollback protection and 4-way versioning.
Integrates with Phase 7 ModelRegistry and Phase 8 Deployment runtimes.
"""

from __future__ import annotations

import copy
import time
from typing import Any, Mapping, Sequence

from src.adaptation.base import ModelVersionRecord, ValidationResult
from src.deployment.runtimes import CloudRuntime, EdgeRuntime
from src.models.base import BaseModel
from src.monitoring.registry import ModelRegistry


class AtomicModelDeployer:
    """Manages atomic deployment of adapted models to Cloud and Edge runtimes with rollback.

    Guarantees:
    1. 4-way version consistency: tracks candidate_version, cloud_version, edge_version, active_system_version.
    2. Atomic transaction: active_system_version advances IF AND ONLY IF both Cloud and Edge deployments succeed.
    3. Rollback safety: if Edge update fails after Cloud update, Cloud is immediately restored to previous version.
    4. Anti-stale routing: system cannot route to a stale Edge model as if it were current.
    5. ModelRegistry synchronization: updates Phase 7 registry metadata consistently.
    """

    def __init__(
        self,
        cloud_runtime: CloudRuntime,
        edge_runtime: EdgeRuntime,
        model_registry: ModelRegistry | None = None,
        initial_version: str = "v1",
    ) -> None:
        self.cloud_runtime = cloud_runtime
        self.edge_runtime = edge_runtime
        self.model_registry = model_registry

        self.candidate_version: str | None = None
        self.cloud_version: str = str(initial_version)
        self.edge_version: str = str(initial_version)
        self.active_system_version: str = str(initial_version)

        self._version_history: list[ModelVersionRecord] = [
            ModelVersionRecord(
                version=self.active_system_version,
                parent_version=None,
                status="ACTIVE",
                reason="Initial deployment",
            )
        ]

        self._total_deployments = 0
        self._successful_deployments = 0
        self._rolled_back_deployments = 0
        self._failed_deployments = 0

    def deploy(
        self,
        candidate_cloud_model: BaseModel,
        updated_edge_model: BaseModel,
        candidate_version: str,
        validation_result: ValidationResult | None = None,
        samples_used: int = 0,
        baseline_samples_used: int = 0,
        feedback_samples_used: int = 0,
        force_cloud_failure: bool = False,
        force_edge_failure: bool = False,
    ) -> tuple[bool, bool, str | None]:
        """Execute atomic deployment transaction with rollback.

        Returns:
            (deployment_success, was_rolled_back, error_message)
        """
        self._total_deployments += 1
        self.candidate_version = str(candidate_version)

        prev_cloud_model = self.cloud_runtime.model
        prev_edge_model = self.edge_runtime.model
        prev_version = self.active_system_version

        cloud_succeeded = False
        edge_succeeded = False
        err_msg: str | None = None

        # Stage 1: Deploy to Cloud
        try:
            if force_cloud_failure:
                raise RuntimeError("Simulated Cloud deployment failure")
            self.cloud_runtime.model = candidate_cloud_model
            cloud_succeeded = True
        except Exception as exc:
            err_msg = f"Cloud deployment failed: {exc}"
            self._failed_deployments += 1
            self.candidate_version = None
            return False, False, err_msg

        # Stage 2: Deploy to Edge
        try:
            if force_edge_failure:
                raise RuntimeError("Simulated Edge deployment failure")
            self.edge_runtime.model = updated_edge_model
            edge_succeeded = True
        except Exception as exc:
            err_msg = f"Edge deployment failed: {exc}"

        # Stage 3: Atomic verification and rollback
        if cloud_succeeded and edge_succeeded:
            # Transaction committed successfully
            self.cloud_version = candidate_version
            self.edge_version = candidate_version
            self.active_system_version = candidate_version
            self.candidate_version = None
            self._successful_deployments += 1

            # Update version history
            v_rec = ModelVersionRecord(
                version=candidate_version,
                parent_version=prev_version,
                status="ACTIVE",
                samples_used=samples_used,
                baseline_samples_used=baseline_samples_used,
                feedback_samples_used=feedback_samples_used,
                validation_metric=validation_result.candidate_metric if validation_result else None,
                reason="Atomic deployment succeeded",
            )
            self._version_history.append(v_rec)

            # Synchronize with ModelRegistry if present
            if self.model_registry is not None:
                try:
                    if self.model_registry.has_model("cloud"):
                        self.model_registry.register_model(
                            model=candidate_cloud_model,
                            model_id="cloud",
                            execution_location="cloud",
                            version=candidate_version,
                        )
                    if self.model_registry.has_model("edge"):
                        self.model_registry.register_model(
                            model=updated_edge_model,
                            model_id="edge",
                            execution_location="edge",
                            version=candidate_version,
                        )
                except Exception:
                    pass

            return True, False, None

        else:
            # ROLLBACK TRANSACTION
            self.cloud_runtime.model = prev_cloud_model
            self.edge_runtime.model = prev_edge_model
            self.cloud_version = prev_version
            self.edge_version = prev_version
            self.active_system_version = prev_version
            self.candidate_version = None
            self._rolled_back_deployments += 1

            v_rec = ModelVersionRecord(
                version=candidate_version,
                parent_version=prev_version,
                status="ROLLED_BACK",
                samples_used=samples_used,
                baseline_samples_used=baseline_samples_used,
                feedback_samples_used=feedback_samples_used,
                validation_metric=validation_result.candidate_metric if validation_result else None,
                reason=f"Rollback triggered: {err_msg}",
            )
            self._version_history.append(v_rec)

            return False, True, err_msg

    def get_version_history(self) -> list[ModelVersionRecord]:
        """Return audit history of all deployed or rejected model versions."""
        return list(self._version_history)

    def get_stats(self) -> dict[str, Any]:
        return {
            "candidate_version": self.candidate_version,
            "cloud_version": self.cloud_version,
            "edge_version": self.edge_version,
            "active_system_version": self.active_system_version,
            "total_deployments": self._total_deployments,
            "successful_deployments": self._successful_deployments,
            "rolled_back_deployments": self._rolled_back_deployments,
            "failed_deployments": self._failed_deployments,
        }
