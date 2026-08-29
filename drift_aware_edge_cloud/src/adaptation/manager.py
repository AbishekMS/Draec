"""Drift-Aware Adaptive Edge-Cloud-Hybrid Orchestration for Non-Stationary IoT.

Module   : src/adaptation/manager.py
Phase    : Phase 9
Status   : IMPLEMENTED

Central adaptation coordinator.
Orchestrates feedback queue, trigger eligibility, Cloud retraining, candidate validation,
atomic deployment with rollback, and cooldown timing.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.adaptation.base import AdaptationResult, AdaptationState
from src.adaptation.deployment import AtomicModelDeployer
from src.adaptation.feedback import FeedbackQueue
from src.adaptation.retrainer import CloudRetrainer
from src.adaptation.validator import CandidateValidator
from src.models.edge_model import EdgeHoeffdingTree


class AdaptationManager:
    """Central orchestrator for the DRAEC Model Adaptation & Retraining lifecycle.

    Guarantees:
    1. Persistent drift requirement: single instantaneous ADWIN alarms do not trigger adaptation.
    2. Causal delayed feedback: only labeled observations arriving on or before t are eligible.
    3. Strict evaluation separation: observations from test1 are strictly quarantined.
    4. Anti-catastrophic forgetting: retraining combines representative baseline with feedback.
    5. Comparative validation: candidate must meet quality thresholds and not regress vs active.
    6. Atomic deployment with rollback: active system version advances only if both Cloud and Edge updates succeed.
    7. Configurable cooldown: prevents repeated retraining loops.
    """

    def __init__(
        self,
        feedback_queue: FeedbackQueue,
        retrainer: CloudRetrainer,
        validator: CandidateValidator,
        deployer: AtomicModelDeployer,
        config: Mapping[str, Any] | None = None,
        *,
        enabled: bool = True,
        require_persistent_drift: bool = True,
        min_severity: float = 0.30,
        min_feedback_samples: int = 50,
        cooldown_steps: int = 100,
    ) -> None:
        self.feedback_queue = feedback_queue
        self.retrainer = retrainer
        self.validator = validator
        self.deployer = deployer
        self.config = dict(config or {})

        adapt_cfg = self.config.get("adaptation", {})
        trig_cfg = adapt_cfg.get("trigger", {})
        cool_cfg = adapt_cfg.get("cooldown", {})

        self.enabled = bool(adapt_cfg.get("enabled", enabled))
        self.require_persistent_drift = bool(trig_cfg.get("require_persistent_drift", require_persistent_drift))
        self.min_severity = float(trig_cfg.get("min_severity", min_severity))
        self.min_feedback_samples = int(trig_cfg.get("min_feedback_samples", min_feedback_samples))
        self.cooldown_steps = int(cool_cfg.get("steps", cooldown_steps))

        self.current_state = AdaptationState.IDLE
        self.last_adaptation_index: int | None = None
        self.adaptation_count = 0
        self._total_triggers = 0
        self._successful_adaptations = 0
        self._rejected_adaptations = 0
        self._failed_adaptations = 0

    def provide_feedback(
        self,
        observation_index: int,
        label: int,
        arrival_index: int,
    ) -> Any:
        """Provide delayed true label for a previously recorded observation."""
        return self.feedback_queue.provide_feedback(
            observation_index=observation_index,
            label=label,
            arrival_index=arrival_index,
        )

    def is_in_cooldown(self, current_index: int) -> bool:
        """Check if adaptation cooldown is currently active."""
        if self.last_adaptation_index is None:
            return False
        return (current_index - self.last_adaptation_index) < self.cooldown_steps

    def check_eligibility(
        self,
        current_index: int,
        is_persistent_drift: bool,
        drift_severity: float,
    ) -> tuple[bool, str]:
        """Evaluate whether conditions for model adaptation are satisfied."""
        if not self.enabled:
            return False, "Adaptation is disabled in configuration"

        if self.is_in_cooldown(current_index):
            rem = self.cooldown_steps - (current_index - self.last_adaptation_index)  # type: ignore[operator]
            return False, f"Adaptation in cooldown ({rem} steps remaining)"

        if self.require_persistent_drift and not is_persistent_drift:
            return False, "Drift is not confirmed persistent (transient alarms ignored)"

        if drift_severity < self.min_severity:
            return False, f"Drift severity {drift_severity:.4f} below threshold {self.min_severity:.4f}"

        eligible_count = self.feedback_queue.count_eligible(current_index=current_index)
        if eligible_count < self.min_feedback_samples:
            return False, f"Eligible feedback samples {eligible_count} below minimum {self.min_feedback_samples}"

        return True, f"Eligible for adaptation ({eligible_count} samples, severity={drift_severity:.4f})"

    def _prepare_updated_edge_model(
        self,
        eligible_feedback: Sequence[Any],
    ) -> EdgeHoeffdingTree:
        """Construct an adapted Edge model incorporating eligible feedback observations."""
        active_edge = self.deployer.edge_runtime.model
        # Create fresh Edge model or clone active with incremental updates
        new_edge = EdgeHoeffdingTree(config=self.config)
        # Train on eligible feedback observations
        for rec in eligible_feedback:
            if rec.is_labeled and rec.label is not None:
                new_edge.learn_one(rec.features, rec.label)
        return new_edge

    def step(
        self,
        observation_index: int,
        x: Any,
        prediction: int | None,
        probabilities: dict[int, float] | None,
        model_version: str,
        is_persistent_drift: bool = False,
        drift_severity: float = 0.0,
        reliability_score: float | None = None,
        source: str = "adaptation",
        force_edge_deployment_failure: bool = False,
        force_cloud_deployment_failure: bool = False,
    ) -> AdaptationResult:
        """Evaluate a single streaming step for feedback logging and adaptation trigger."""
        obs_idx = int(observation_index)

        # 1. Record prediction in feedback queue (enforcing test1 quarantine)
        self.feedback_queue.record_prediction(
            observation_index=obs_idx,
            features=x,
            prediction=prediction,
            probabilities=probabilities,
            model_version=str(model_version),
            source=source,
        )

        # 2. Check adaptation eligibility
        eligible, reason = self.check_eligibility(
            current_index=obs_idx,
            is_persistent_drift=is_persistent_drift,
            drift_severity=drift_severity,
        )

        if not eligible:
            if self.is_in_cooldown(obs_idx):
                self.current_state = AdaptationState.COOLDOWN
            else:
                self.current_state = AdaptationState.IDLE

            return AdaptationResult(
                state=self.current_state,
                triggered=False,
                candidate_version=None,
                active_version=self.deployer.active_system_version,
                cloud_version=self.deployer.cloud_version,
                edge_version=self.deployer.edge_version,
                validation_result=None,
                deployment_success=False,
                rolled_back=False,
                error=None,
                samples_used=0,
            )

        # 3. Adaptation Triggered!
        self._total_triggers += 1
        self.current_state = AdaptationState.ELIGIBLE
        cand_v = f"v{self.adaptation_count + 2}"
        parent_v = self.deployer.active_system_version

        # Step 3a: Retrieve causally eligible feedback
        eligible_feedback = self.feedback_queue.get_eligible_feedback(
            current_index=obs_idx,
            max_samples=self.retrainer.max_feedback_samples,
        )

        # Step 3b: Cloud Retraining with Anti-Forgetting
        self.current_state = AdaptationState.TRAINING
        try:
            candidate_cloud, train_meta = self.retrainer.retrain(
                eligible_feedback=eligible_feedback,
                parent_version=parent_v,
                candidate_version=cand_v,
            )
        except Exception as exc:
            self.current_state = AdaptationState.FAILED
            self._failed_adaptations += 1
            self.last_adaptation_index = obs_idx
            return AdaptationResult(
                state=AdaptationState.FAILED,
                triggered=True,
                candidate_version=cand_v,
                active_version=self.deployer.active_system_version,
                cloud_version=self.deployer.cloud_version,
                edge_version=self.deployer.edge_version,
                validation_result=None,
                deployment_success=False,
                rolled_back=False,
                error=f"Cloud retraining exception: {exc}",
                samples_used=0,
            )

        # Step 3c: Candidate Model Validation on clean validation data
        self.current_state = AdaptationState.VALIDATING
        try:
            active_cloud = self.deployer.cloud_runtime.model
            val_res = self.validator.validate(
                candidate_model=candidate_cloud,
                active_model=active_cloud,
            )
        except Exception as exc:
            self.current_state = AdaptationState.FAILED
            self._failed_adaptations += 1
            self.last_adaptation_index = obs_idx
            return AdaptationResult(
                state=AdaptationState.FAILED,
                triggered=True,
                candidate_version=cand_v,
                active_version=self.deployer.active_system_version,
                cloud_version=self.deployer.cloud_version,
                edge_version=self.deployer.edge_version,
                validation_result=None,
                deployment_success=False,
                rolled_back=False,
                error=f"Validation exception: {exc}",
                samples_used=train_meta.get("total_samples_trained", 0),
            )

        if not val_res.candidate_valid:
            # Candidate rejected: preserve active model intact
            self.current_state = AdaptationState.REJECTED
            self._rejected_adaptations += 1
            self.last_adaptation_index = obs_idx
            return AdaptationResult(
                state=AdaptationState.REJECTED,
                triggered=True,
                candidate_version=cand_v,
                active_version=self.deployer.active_system_version,
                cloud_version=self.deployer.cloud_version,
                edge_version=self.deployer.edge_version,
                validation_result=val_res,
                deployment_success=False,
                rolled_back=False,
                error=f"Candidate rejected: {val_res.reason}",
                samples_used=train_meta.get("total_samples_trained", 0),
            )

        # Step 3d: Atomic Cloud + Edge Deployment with Rollback
        updated_edge = self._prepare_updated_edge_model(eligible_feedback)
        deploy_succ, rolled_back, deploy_err = self.deployer.deploy(
            candidate_cloud_model=candidate_cloud,
            updated_edge_model=updated_edge,
            candidate_version=cand_v,
            validation_result=val_res,
            samples_used=train_meta.get("total_samples_trained", 0),
            baseline_samples_used=train_meta.get("baseline_samples_used", 0),
            feedback_samples_used=train_meta.get("feedback_samples_used", 0),
            force_cloud_failure=force_cloud_deployment_failure,
            force_edge_failure=force_edge_deployment_failure,
        )

        self.last_adaptation_index = obs_idx

        if deploy_succ:
            self.current_state = AdaptationState.ACCEPTED
            self._successful_adaptations += 1
            self.adaptation_count += 1
            return AdaptationResult(
                state=AdaptationState.ACCEPTED,
                triggered=True,
                candidate_version=cand_v,
                active_version=self.deployer.active_system_version,
                cloud_version=self.deployer.cloud_version,
                edge_version=self.deployer.edge_version,
                validation_result=val_res,
                deployment_success=True,
                rolled_back=False,
                error=None,
                samples_used=train_meta.get("total_samples_trained", 0),
            )
        else:
            final_state = AdaptationState.ROLLBACK if rolled_back else AdaptationState.FAILED
            self.current_state = final_state
            self._failed_adaptations += 1
            return AdaptationResult(
                state=final_state,
                triggered=True,
                candidate_version=cand_v,
                active_version=self.deployer.active_system_version,
                cloud_version=self.deployer.cloud_version,
                edge_version=self.deployer.edge_version,
                validation_result=val_res,
                deployment_success=False,
                rolled_back=rolled_back,
                error=deploy_err,
                samples_used=train_meta.get("total_samples_trained", 0),
            )

    def get_stats(self) -> dict[str, Any]:
        return {
            "current_state": self.current_state.value,
            "adaptation_count": self.adaptation_count,
            "total_triggers": self._total_triggers,
            "successful_adaptations": self._successful_adaptations,
            "rejected_adaptations": self._rejected_adaptations,
            "failed_adaptations": self._failed_adaptations,
            "last_adaptation_index": self.last_adaptation_index,
            "active_system_version": self.deployer.active_system_version,
            "cloud_version": self.deployer.cloud_version,
            "edge_version": self.deployer.edge_version,
            "feedback": self.feedback_queue.get_stats(),
            "retrainer": self.retrainer.get_stats(),
            "validator": self.validator.get_stats(),
            "deployer": self.deployer.get_stats(),
        }

    def reset(self) -> None:
        self.current_state = AdaptationState.IDLE
        self.last_adaptation_index = None
        self.adaptation_count = 0
        self._total_triggers = 0
        self._successful_adaptations = 0
        self._rejected_adaptations = 0
        self._failed_adaptations = 0
        self.feedback_queue.reset()
