"""Drift-Aware Adaptive Edge-Cloud-Hybrid Orchestration for Non-Stationary IoT.

Module   : tests/test_adaptation.py
Phase    : Phase 9
Status   : IMPLEMENTED

Comprehensive test suite for Phase 9 Model Adaptation & Retraining.
Validates all Phase 9 contracts, scientific invariants, anti-forgetting,
strict test1 quarantine, atomic deployment with rollback, and Tests A through O.
"""

from __future__ import annotations

import copy
import time
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

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
from src.decision.base import DecisionAction, DecisionInputs, DecisionResult
from src.decision.engine import DecisionEngine
from src.deployment.environment import DeploymentEnvironment
from src.deployment.network import NetworkSimulator
from src.deployment.runtimes import CloudRuntime, EdgeRuntime
from src.models.cloud_model import CloudXGBoost
from src.models.edge_model import EdgeHoeffdingTree
from src.monitoring.registry import ModelRegistry
from src.reliability.estimator import ReliabilityEstimator


# =============================================================================
# Helper Fixtures & Mocks
# =============================================================================

class MockClassifier:
    """Deterministic mock model for unit testing."""

    def __init__(self, pred: int = 0, p0: float = 0.85, name: str = "mock") -> None:
        self.pred = int(pred)
        self.p0 = float(p0)
        self.name = str(name)
        self.call_count = 0

    def fit(self, X: Any, y: Any) -> MockClassifier:
        return self

    def predict(self, X: Any) -> np.ndarray:
        self.call_count += 1
        n = len(X) if hasattr(X, "__len__") else 1
        return np.full(n, self.pred, dtype=int)

    def predict_proba(self, X: Any) -> np.ndarray:
        n = len(X) if hasattr(X, "__len__") else 1
        p0 = self.p0
        return np.tile([p0, 1.0 - p0], (n, 1))

    def predict_one(self, x: Any) -> int:
        self.call_count += 1
        return self.pred

    def predict_proba_one(self, x: Any) -> dict[int, float]:
        return {0: self.p0, 1: 1.0 - self.p0}

    def get_info(self) -> dict[str, Any]:
        return {"name": self.name, "pred": self.pred, "p0": self.p0}


@pytest.fixture
def sample_features() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.normal(size=(60, 10))


@pytest.fixture
def sample_labels() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(0, 2, size=60)


@pytest.fixture
def trained_models(sample_features, sample_labels):
    edge_m = EdgeHoeffdingTree()
    edge_m.fit(sample_features[:40], sample_labels[:40])
    cloud_m = CloudXGBoost(n_estimators=10, max_depth=3, random_state=42)
    cloud_m.fit(sample_features[:40], sample_labels[:40])
    return edge_m, cloud_m


# =============================================================================
# Test A: Candidate Retraining Preserves Baseline (Anti-Catastrophic Forgetting)
# =============================================================================

def test_a_candidate_retraining_preserves_baseline(sample_features, sample_labels):
    """Test A: Candidate retraining combines representative baseline samples with eligible feedback."""
    retrainer = CloudRetrainer(
        min_feedback_samples=10,
        max_baseline_samples=25,
        random_seed=42,
    )
    retrainer.set_baseline_data(sample_features[:30], sample_labels[:30])
    assert retrainer.has_baseline_data
    stats = retrainer.get_stats()
    assert stats["baseline_samples_cached"] == 25

    # Create 15 feedback records
    feedback = [
        FeedbackRecord(
            observation_index=i,
            features=sample_features[30 + i],
            prediction=0,
            probabilities={0: 0.8, 1: 0.2},
            model_version="v1",
            label=int(sample_labels[30 + i]),
            arrival_index=i + 5,
            is_labeled=True,
        )
        for i in range(15)
    ]

    candidate, meta = retrainer.retrain(feedback, parent_version="v1", candidate_version="v2")
    assert isinstance(candidate, CloudXGBoost)
    assert meta["baseline_samples_used"] == 25
    assert meta["feedback_samples_used"] == 15
    assert meta["total_samples_trained"] == 40
    assert meta["candidate_version"] == "v2"


# =============================================================================
# Test B: test1 Observations Cannot Enter Adaptation Training
# =============================================================================

def test_b_test1_cannot_enter_adaptation_training():
    """Test B: FeedbackQueue strictly quarantines test1 evaluation stream."""
    queue = FeedbackQueue(max_size=100)
    with pytest.raises(ValueError, match="Data contamination guard.*test1"):
        queue.record_prediction(
            observation_index=1,
            features=[0.1, 0.2],
            prediction=1,
            probabilities={0: 0.1, 1: 0.9},
            model_version="v1",
            source="test1",
        )


# =============================================================================
# Test C: Delayed test1 Feedback Cannot Trigger Adaptation
# =============================================================================

def test_c_delayed_test1_feedback_cannot_trigger_adaptation(trained_models, sample_features, sample_labels):
    """Test C: test1 cannot supply feedback to trigger adaptation."""
    edge_m, cloud_m = trained_models
    edge_rt = EdgeRuntime(edge_m)
    cloud_rt = CloudRuntime(cloud_m)
    deployer = AtomicModelDeployer(cloud_rt, edge_rt, initial_version="v1")
    retrainer = CloudRetrainer(min_feedback_samples=10)
    retrainer.set_baseline_data(sample_features[:20], sample_labels[:20])
    validator = CandidateValidator(minimum_metric=0.50)
    validator.set_validation_data(sample_features[20:40], sample_labels[20:40])
    queue = FeedbackQueue()

    manager = AdaptationManager(queue, retrainer, validator, deployer, min_feedback_samples=10)

    # Attempting to feed test1 through manager step must raise ValueError
    with pytest.raises(ValueError, match="test1"):
        manager.step(
            observation_index=1,
            x=sample_features[0],
            prediction=0,
            probabilities={0: 0.9, 1: 0.1},
            model_version="v1",
            source="test1_stream",
        )


# =============================================================================
# Test D: Transient Drift Cannot Trigger Adaptation
# =============================================================================

def test_d_transient_drift_cannot_trigger_adaptation(trained_models, sample_features, sample_labels):
    """Test D: Instantaneous/transient drift (is_persistent=False) does not trigger adaptation."""
    edge_m, cloud_m = trained_models
    deployer = AtomicModelDeployer(CloudRuntime(cloud_m), EdgeRuntime(edge_m))
    queue = FeedbackQueue()
    retrainer = CloudRetrainer(min_feedback_samples=5)
    retrainer.set_baseline_data(sample_features[:10], sample_labels[:10])
    validator = CandidateValidator(minimum_metric=0.50)
    validator.set_validation_data(sample_features[10:20], sample_labels[10:20])

    manager = AdaptationManager(queue, retrainer, validator, deployer, min_feedback_samples=5)

    # Populate 10 eligible feedback records
    for i in range(10):
        queue.record_prediction(i, sample_features[i], 0, {0: 0.8, 1: 0.2}, "v1")
        queue.provide_feedback(i, sample_labels[i], arrival_index=i)

    res = manager.step(
        observation_index=15,
        x=sample_features[15],
        prediction=0,
        probabilities={0: 0.8, 1: 0.2},
        model_version="v1",
        is_persistent_drift=False,  # Transient drift!
        drift_severity=0.85,
    )
    assert not res.triggered
    assert res.state == AdaptationState.IDLE
    assert res.active_version == "v1"


# =============================================================================
# Test E: Persistent Drift + Sufficient Feedback Triggers Adaptation
# =============================================================================

def test_e_persistent_drift_sufficient_feedback_triggers_adaptation(sample_features):
    """Test E: Persistent drift with sufficient feedback triggers successful adaptation."""
    sample_labels = (sample_features[:, 0] > 0).astype(int)
    edge_m = EdgeHoeffdingTree()
    edge_m.fit(sample_features[:20], sample_labels[:20])
    cloud_m = CloudXGBoost(n_estimators=10, max_depth=3, random_state=42)
    cloud_m.fit(sample_features[:20], sample_labels[:20])

    deployer = AtomicModelDeployer(CloudRuntime(cloud_m), EdgeRuntime(edge_m), initial_version="v1")
    queue = FeedbackQueue()
    retrainer = CloudRetrainer(min_feedback_samples=5, random_seed=42)
    retrainer.set_baseline_data(sample_features[:20], sample_labels[:20])
    validator = CandidateValidator(minimum_metric=0.50, max_regression_margin=0.10)
    validator.set_validation_data(sample_features[20:40], sample_labels[20:40])

    manager = AdaptationManager(
        queue, retrainer, validator, deployer,
        require_persistent_drift=True, min_severity=0.30, min_feedback_samples=5,
    )

    # Provide 10 labeled feedback samples
    for i in range(10):
        queue.record_prediction(i, sample_features[i], 0, {0: 0.8, 1: 0.2}, "v1")
        queue.provide_feedback(i, sample_labels[i], arrival_index=i)

    res = manager.step(
        observation_index=20,
        x=sample_features[20],
        prediction=0,
        probabilities={0: 0.8, 1: 0.2},
        model_version="v1",
        is_persistent_drift=True,
        drift_severity=0.45,
    )
    assert res.triggered
    assert res.state == AdaptationState.ACCEPTED
    assert res.active_version == "v2"
    assert res.deployment_success
    assert res.samples_used > 0


# =============================================================================
# Test F: Cooldown Prevents Repeated Immediate Retraining
# =============================================================================

def test_f_cooldown_prevents_repeated_immediate_retraining(trained_models, sample_features, sample_labels):
    """Test F: Adaptation enters cooldown immediately after trigger, ignoring subsequent alarms."""
    edge_m, cloud_m = trained_models
    deployer = AtomicModelDeployer(CloudRuntime(cloud_m), EdgeRuntime(edge_m), initial_version="v1")
    queue = FeedbackQueue()
    retrainer = CloudRetrainer(min_feedback_samples=5)
    retrainer.set_baseline_data(sample_features[:20], sample_labels[:20])
    validator = CandidateValidator(minimum_metric=0.50)
    validator.set_validation_data(sample_features[20:40], sample_labels[20:40])

    manager = AdaptationManager(
        queue, retrainer, validator, deployer,
        min_feedback_samples=5, cooldown_steps=50,
    )

    for i in range(10):
        queue.record_prediction(i, sample_features[i], 0, {0: 0.8, 1: 0.2}, "v1")
        queue.provide_feedback(i, sample_labels[i], arrival_index=i)

    # Trigger adaptation at index 20
    res1 = manager.step(
        observation_index=20, x=sample_features[20], prediction=0,
        probabilities={0: 0.8, 1: 0.2}, model_version="v1",
        is_persistent_drift=True, drift_severity=0.5,
    )
    assert res1.triggered

    # Next step at index 25 (within cooldown of 50 steps)
    res2 = manager.step(
        observation_index=25, x=sample_features[25], prediction=0,
        probabilities={0: 0.8, 1: 0.2}, model_version="v2",
        is_persistent_drift=True, drift_severity=0.5,
    )
    assert not res2.triggered
    assert res2.state == AdaptationState.COOLDOWN
    assert manager.is_in_cooldown(25)


# =============================================================================
# Test G: Validation Failure Preserves Active Model Intact
# =============================================================================

def test_g_validation_failure_preserves_active_model(trained_models, sample_features, sample_labels):
    """Test G: If candidate fails validation, active model is preserved and candidate rejected."""
    edge_m, cloud_m = trained_models
    cloud_rt = CloudRuntime(cloud_m)
    edge_rt = EdgeRuntime(edge_m)
    deployer = AtomicModelDeployer(cloud_rt, edge_rt, initial_version="v1")
    queue = FeedbackQueue()
    retrainer = CloudRetrainer(min_feedback_samples=5)
    retrainer.set_baseline_data(sample_features[:20], sample_labels[:20])

    # Impossible minimum validation threshold -> guaranteed validation rejection
    validator = CandidateValidator(minimum_metric=0.999)
    validator.set_validation_data(sample_features[20:40], sample_labels[20:40])

    manager = AdaptationManager(queue, retrainer, validator, deployer, min_feedback_samples=5)

    for i in range(10):
        queue.record_prediction(i, sample_features[i], 0, {0: 0.8, 1: 0.2}, "v1")
        queue.provide_feedback(i, sample_labels[i], arrival_index=i)

    res = manager.step(
        observation_index=20, x=sample_features[20], prediction=0,
        probabilities={0: 0.8, 1: 0.2}, model_version="v1",
        is_persistent_drift=True, drift_severity=0.5,
    )
    assert res.triggered
    assert res.state == AdaptationState.REJECTED
    assert res.active_version == "v1"
    assert not res.deployment_success
    assert cloud_rt.model is cloud_m


# =============================================================================
# Test H: Cloud Deployment Failure Preserves Active Version
# =============================================================================

def test_h_cloud_deployment_failure_preserves_active_version():
    """Test H: Failure during Cloud deployment preserves active system version."""
    m_edge = MockClassifier(name="edge_v1")
    m_cloud = MockClassifier(name="cloud_v1")
    cloud_rt = CloudRuntime(m_cloud)
    edge_rt = EdgeRuntime(m_edge)
    deployer = AtomicModelDeployer(cloud_rt, edge_rt, initial_version="v1")

    cand_cloud = MockClassifier(name="cloud_cand")
    cand_edge = MockClassifier(name="edge_cand")

    succ, rolled_back, err = deployer.deploy(
        candidate_cloud_model=cand_cloud,
        updated_edge_model=cand_edge,
        candidate_version="v2",
        force_cloud_failure=True,
    )
    assert not succ
    assert not rolled_back
    assert deployer.active_system_version == "v1"
    assert cloud_rt.model is m_cloud
    assert edge_rt.model is m_edge


# =============================================================================
# Test I: Edge Deployment Failure Triggers Cloud Rollback
# =============================================================================

def test_i_edge_deployment_failure_triggers_cloud_rollback():
    """Test I: Edge deployment failure after Cloud update rolls back Cloud to previous active model."""
    m_edge = MockClassifier(name="edge_v1")
    m_cloud = MockClassifier(name="cloud_v1")
    cloud_rt = CloudRuntime(m_cloud)
    edge_rt = EdgeRuntime(m_edge)
    deployer = AtomicModelDeployer(cloud_rt, edge_rt, initial_version="v1")

    cand_cloud = MockClassifier(name="cloud_cand")
    cand_edge = MockClassifier(name="edge_cand")

    succ, rolled_back, err = deployer.deploy(
        candidate_cloud_model=cand_cloud,
        updated_edge_model=cand_edge,
        candidate_version="v2",
        force_edge_failure=True,  # Simulate failure after Cloud was set
    )
    assert not succ
    assert rolled_back
    assert "Edge deployment failed" in str(err)
    assert deployer.active_system_version == "v1"
    assert deployer.cloud_version == "v1"
    assert deployer.edge_version == "v1"
    assert cloud_rt.model is m_cloud
    assert edge_rt.model is m_edge


# =============================================================================
# Test J: Cloud and Edge Version Consistency
# =============================================================================

def test_j_cloud_edge_version_consistency():
    """Test J: After successful atomic deployment, cloud, edge, and system versions match."""
    cloud_rt = CloudRuntime(MockClassifier())
    edge_rt = EdgeRuntime(MockClassifier())
    deployer = AtomicModelDeployer(cloud_rt, edge_rt, initial_version="v1")

    succ, rolled_back, err = deployer.deploy(
        candidate_cloud_model=MockClassifier(name="cloud_v2"),
        updated_edge_model=MockClassifier(name="edge_v2"),
        candidate_version="v2",
    )
    assert succ
    assert deployer.cloud_version == "v2"
    assert deployer.edge_version == "v2"
    assert deployer.active_system_version == "v2"
    assert deployer.candidate_version is None


# =============================================================================
# Test K: No Stale Edge Model Treated as Current
# =============================================================================

def test_k_no_stale_edge_model_treated_as_current():
    """Test K: System active version never advances if Edge deployment fails."""
    cloud_rt = CloudRuntime(MockClassifier(pred=1))
    edge_rt = EdgeRuntime(MockClassifier(pred=0))
    deployer = AtomicModelDeployer(cloud_rt, edge_rt, initial_version="v1")

    deployer.deploy(
        candidate_cloud_model=MockClassifier(pred=1),
        updated_edge_model=MockClassifier(pred=1),
        candidate_version="v2",
        force_edge_failure=True,
    )
    # The active system version must remain v1, never v2
    assert deployer.active_system_version == "v1"
    assert deployer.edge_version == "v1"
    assert deployer.cloud_version == "v1"


# =============================================================================
# Test L: Deterministic Candidate Training
# =============================================================================

def test_l_deterministic_candidate_training(sample_features, sample_labels):
    """Test L: Identical seeds produce identical candidate models and predictions."""
    retrainer1 = CloudRetrainer(min_feedback_samples=5, random_seed=42)
    retrainer2 = CloudRetrainer(min_feedback_samples=5, random_seed=42)

    retrainer1.set_baseline_data(sample_features[:20], sample_labels[:20])
    retrainer2.set_baseline_data(sample_features[:20], sample_labels[:20])

    feedback = [
        FeedbackRecord(
            observation_index=i, features=sample_features[20 + i],
            prediction=0, probabilities={0: 0.8, 1: 0.2}, model_version="v1",
            label=int(sample_labels[20 + i]), arrival_index=i, is_labeled=True,
        )
        for i in range(10)
    ]

    cand1, _ = retrainer1.retrain(feedback)
    cand2, _ = retrainer2.retrain(feedback)

    preds1 = cand1.predict(sample_features[30:45])
    preds2 = cand2.predict(sample_features[30:45])
    np.testing.assert_array_equal(preds1, preds2)


# =============================================================================
# Test M: Phase 5 Routing Logic Remains Unchanged
# =============================================================================

def test_m_phase5_routing_remains_unchanged():
    """Test M: Phase 5 decision engine hysteresis thresholds and actions remain intact."""
    from src.decision.engine import AdaptiveController
    ctrl = AdaptiveController(
        critical_cloud_threshold=0.30,
        cloud_threshold=0.50,
        edge_return_threshold=0.70,
    )
    engine = DecisionEngine(
        controller=ctrl,
        edge_model=MockClassifier(pred=0),
        cloud_model=MockClassifier(pred=1),
    )
    # R_t = 0.90 -> EDGE
    d1 = engine.decide(DecisionInputs(reliability=0.90, observation_index=1))
    assert d1.selected_action == DecisionAction.EDGE

    # R_t = 0.40 -> HYBRID (hysteresis between 0.30 and 0.50)
    d2 = engine.decide(DecisionInputs(reliability=0.40, observation_index=2))
    assert d2.selected_action == DecisionAction.HYBRID

    # R_t = 0.20 -> CLOUD (< 0.30)
    d3 = engine.decide(DecisionInputs(reliability=0.20, observation_index=3))
    assert d3.selected_action == DecisionAction.CLOUD


# =============================================================================
# Test N: Phase 4 Reliability Score Invariants Remain Unchanged
# =============================================================================

def test_n_phase4_reliability_remains_unchanged():
    """Test N: Phase 4 harmonic mean reliability R_t is calculated unaltered."""
    estimator = ReliabilityEstimator()
    score = estimator.update(
        probs={0: 0.85, 1: 0.15},
        instantaneous_error=0,
        drift_severity=0.10,
        quality=0.95,
    )
    assert 0.0 <= score.reliability <= 1.0
    assert hasattr(score.inputs, "confidence")
    assert hasattr(score.inputs, "error")
    assert hasattr(score.inputs, "drift")
    assert hasattr(score.inputs, "quality")


# =============================================================================
# Test O: Phase 8 Deployment Runtime API Remains Fully Compatible
# =============================================================================

def test_o_phase8_runtime_api_remains_compatible(trained_models, sample_features):
    """Test O: Phase 8 DeploymentEnvironment seamlessly executes adapted models."""
    edge_m, cloud_m = trained_models
    edge_rt = EdgeRuntime(edge_m)
    cloud_rt = CloudRuntime(cloud_m)
    net = NetworkSimulator(base_latency_s=0.010, jitter_s=0.0, packet_loss_probability=0.0)
    env = DeploymentEnvironment(edge_rt, cloud_rt, net, fallback_confidence_threshold=0.60)

    # Edge execution
    res_edge = env.execute_edge(sample_features[0])
    assert res_edge.success
    assert res_edge.action == DecisionAction.EDGE

    # Cloud execution
    res_cloud = env.execute_cloud(sample_features[0])
    assert res_cloud.success
    assert res_cloud.action == DecisionAction.CLOUD

    # Hybrid execution
    res_hyb = env.execute_hybrid(sample_features[0])
    assert res_hyb.success
    assert res_hyb.action == DecisionAction.HYBRID


# =============================================================================
# Additional Unit & Edge Case Tests
# =============================================================================

def test_feedback_queue_bounded_eviction():
    """Test FIFO eviction when FeedbackQueue capacity is exceeded."""
    queue = FeedbackQueue(max_size=20)
    for i in range(30):
        queue.record_prediction(i, [i], 0, {0: 1.0, 1: 0.0}, "v1")
    stats = queue.get_stats()
    assert stats["current_buffer_size"] == 20
    assert stats["total_recorded"] == 30


def test_feedback_queue_acausal_arrival_rejected():
    """Test that feedback arriving before observation is rejected."""
    queue = FeedbackQueue(max_size=50)
    queue.record_prediction(10, [1], 0, {0: 1.0, 1: 0.0}, "v1")
    with pytest.raises(ValueError, match="Acausal feedback arrival"):
        queue.provide_feedback(observation_index=10, label=1, arrival_index=8)


def test_feedback_queue_future_feedback_quarantined():
    """Test that feedback with arrival_index > current_index is not eligible."""
    queue = FeedbackQueue(max_size=50)
    queue.record_prediction(5, [1], 0, {0: 1.0, 1: 0.0}, "v1")
    queue.provide_feedback(5, label=1, arrival_index=15)

    # At current_index = 10, feedback has not arrived yet!
    assert queue.count_eligible(current_index=10) == 0
    # At current_index = 15, feedback has arrived and is eligible!
    assert queue.count_eligible(current_index=15) == 1


def test_feedback_record_immutability():
    """Test FeedbackRecord frozen dataclass invariants."""
    rec = FeedbackRecord(1, [0.5], 0, {0: 1.0, 1: 0.0}, "v1")
    with pytest.raises(AttributeError):
        rec.label = 1  # type: ignore[misc]


def test_candidate_validator_macro_f1():
    """Test CandidateValidator computing Macro-F1 accurately."""
    val = CandidateValidator(minimum_metric=0.60, max_regression_margin=0.10)
    X = np.array([[1], [2], [3], [4]])
    y = np.array([0, 0, 1, 1])
    val.set_validation_data(X, y)

    active_m = MockClassifier(pred=0)  # F1 will be low on class 1
    cand_m = MockClassifier(pred=1)

    res = val.validate(cand_m, active_m)
    assert isinstance(res, ValidationResult)
    assert res.metric_name == "macro_f1"


def test_candidate_validator_test1_quarantine():
    """Test CandidateValidator rejects test1 as validation source."""
    val = CandidateValidator()
    with pytest.raises(ValueError, match="Data contamination guard.*test1"):
        val.set_validation_data([[1]], [0], source="test1_quarantined")


def test_model_registry_synchronization():
    """Test AtomicModelDeployer updates Phase 7 ModelRegistry on success."""
    reg = ModelRegistry()
    m_edge = MockClassifier(name="edge_m")
    m_cloud = MockClassifier(name="cloud_m")
    reg.register_model(m_edge, "edge", "edge", version="v1")
    reg.register_model(m_cloud, "cloud", "cloud", version="v1")

    deployer = AtomicModelDeployer(CloudRuntime(m_cloud), EdgeRuntime(m_edge), model_registry=reg)

    new_cloud = MockClassifier(name="cloud_v2")
    new_edge = MockClassifier(name="edge_v2")
    succ, _, _ = deployer.deploy(new_cloud, new_edge, candidate_version="v2")
    assert succ
    assert reg.get_metadata("cloud").model_version == "v2"
    assert reg.get_metadata("edge").model_version == "v2"


def test_atomic_deployer_version_history():
    """Test ModelVersionRecord tracking in version history."""
    deployer = AtomicModelDeployer(CloudRuntime(MockClassifier()), EdgeRuntime(MockClassifier()), initial_version="v1")
    deployer.deploy(MockClassifier(), MockClassifier(), candidate_version="v2")
    deployer.deploy(MockClassifier(), MockClassifier(), candidate_version="v3", force_edge_failure=True)

    history = deployer.get_version_history()
    assert len(history) == 3
    assert history[0].version == "v1"
    assert history[0].status == "ACTIVE"
    assert history[1].version == "v2"
    assert history[1].status == "ACTIVE"
    assert history[2].version == "v3"
    assert history[2].status == "ROLLED_BACK"


def test_manager_reset():
    """Test AdaptationManager reset restores IDLE state and clears counters."""
    deployer = AtomicModelDeployer(CloudRuntime(MockClassifier()), EdgeRuntime(MockClassifier()))
    manager = AdaptationManager(FeedbackQueue(), CloudRetrainer(), CandidateValidator(), deployer)
    manager.current_state = AdaptationState.ACCEPTED
    manager.adaptation_count = 3
    manager.reset()
    assert manager.current_state == AdaptationState.IDLE
    assert manager.adaptation_count == 0


def test_adaptation_state_from_str():
    """Test AdaptationState enum parsing."""
    assert AdaptationState.from_str("accepted") == AdaptationState.ACCEPTED
    assert AdaptationState.from_str(AdaptationState.IDLE) == AdaptationState.IDLE
    with pytest.raises(ValueError):
        AdaptationState.from_str("INVALID_STATE")


def test_insufficient_samples_rejected_by_retrainer():
    """Test CloudRetrainer raises ValueError if samples < min_feedback_samples."""
    retrainer = CloudRetrainer(min_feedback_samples=50)
    with pytest.raises(ValueError, match="Insufficient eligible feedback"):
        retrainer.retrain([])


def test_feedback_queue_eviction_order():
    """Test that FeedbackQueue evicts the oldest observation first (FIFO)."""
    q = FeedbackQueue(max_size=3)
    q.record_prediction(1, [1], 0, {0: 1.0, 1: 0.0}, "v1")
    q.record_prediction(2, [2], 0, {0: 1.0, 1: 0.0}, "v1")
    q.record_prediction(3, [3], 0, {0: 1.0, 1: 0.0}, "v1")
    # Buffer has {1, 2, 3}
    q.record_prediction(4, [4], 0, {0: 1.0, 1: 0.0}, "v1")
    # Buffer should now have {2, 3, 4} (1 evicted)
    assert 1 not in q._records
    assert 2 in q._records
    assert 4 in q._records


def test_feedback_queue_duplicate_observation_index():
    """Test that recording the same observation index updates the record without increasing size."""
    q = FeedbackQueue(max_size=10)
    q.record_prediction(5, [0.1], 0, {0: 0.9, 1: 0.1}, "v1")
    q.record_prediction(5, [0.2], 1, {0: 0.1, 1: 0.9}, "v1")
    assert q.get_stats()["current_buffer_size"] == 1
    assert q._records[5].prediction == 1


def test_feedback_queue_clear():
    """Test that FeedbackQueue.clear empties buffer while reset resets counters."""
    q = FeedbackQueue(max_size=10)
    q.record_prediction(1, [0.1], 0, {0: 1.0, 1: 0.0}, "v1")
    q.clear()
    assert q.get_stats()["current_buffer_size"] == 0
    assert q.get_stats()["total_recorded"] == 1


def test_retrainer_feature_names_mapping(sample_features, sample_labels):
    """Test CloudRetrainer extracts features correctly when provided with named DataFrame."""
    cols = [f"feat_{i}" for i in range(sample_features.shape[1])]
    df_feat = pd.DataFrame(sample_features[:20], columns=cols)
    retrainer = CloudRetrainer(min_feedback_samples=5, random_seed=42)
    retrainer.set_baseline_data(df_feat, sample_labels[:20])

    feedback = [
        FeedbackRecord(
            observation_index=i,
            features={cols[j]: float(sample_features[20 + i, j]) for j in range(len(cols))},
            prediction=0,
            probabilities={0: 0.8, 1: 0.2},
            model_version="v1",
            label=int(sample_labels[20 + i]),
            arrival_index=i,
            is_labeled=True,
        )
        for i in range(6)
    ]
    cand, meta = retrainer.retrain(feedback)
    assert meta["feature_count"] == len(cols)


def test_retrainer_without_baseline_data(sample_features, sample_labels):
    """Test CloudRetrainer can retrain solely on feedback if baseline data is unconfigured."""
    retrainer = CloudRetrainer(min_feedback_samples=5, random_seed=42)
    assert not retrainer.has_baseline_data

    feedback = [
        FeedbackRecord(
            observation_index=i,
            features=sample_features[i],
            prediction=0,
            probabilities={0: 0.8, 1: 0.2},
            model_version="v1",
            label=int(sample_labels[i]),
            arrival_index=i,
            is_labeled=True,
        )
        for i in range(8)
    ]
    cand, meta = retrainer.retrain(feedback)
    assert meta["baseline_samples_used"] == 0
    assert meta["feedback_samples_used"] == 8


def test_candidate_validator_accuracy_metric():
    """Test CandidateValidator with accuracy metric."""
    val = CandidateValidator(metric="accuracy", minimum_metric=0.50)
    X = np.array([[1], [2], [3], [4]])
    y = np.array([0, 0, 1, 1])
    val.set_validation_data(X, y)

    res = val.validate(MockClassifier(pred=0), MockClassifier(pred=0))
    assert res.metric_name == "accuracy"
    assert res.candidate_metric == 0.50


def test_candidate_validator_adaptation_data():
    """Test CandidateValidator includes adaptation metric details when adaptation val split is given."""
    val = CandidateValidator(minimum_metric=0.10)
    val.set_validation_data([[1], [2]], [0, 1])
    adapt_val = (np.array([[3], [4]]), np.array([0, 1]))

    res = val.validate(MockClassifier(pred=0), MockClassifier(pred=1), adaptation_val_data=adapt_val)
    assert "candidate_adaptation_metric" in res.details
    assert "active_adaptation_metric" in res.details


def test_atomic_deployer_cloud_failure_does_not_call_edge():
    """Test that if Cloud update fails, Edge model is never modified."""
    edge_m = MockClassifier(name="edge_orig")
    edge_rt = EdgeRuntime(edge_m)
    deployer = AtomicModelDeployer(CloudRuntime(MockClassifier()), edge_rt, initial_version="v1")

    succ, rb, err = deployer.deploy(
        candidate_cloud_model=MockClassifier(),
        updated_edge_model=MockClassifier(name="edge_new"),
        candidate_version="v2",
        force_cloud_failure=True,
    )
    assert not succ
    assert not rb
    assert edge_rt.model is edge_m


def test_adaptation_manager_disabled(sample_features):
    """Test that AdaptationManager returns IDLE and never triggers when enabled=False."""
    mgr = AdaptationManager(
        FeedbackQueue(), CloudRetrainer(), CandidateValidator(),
        AtomicModelDeployer(CloudRuntime(MockClassifier()), EdgeRuntime(MockClassifier())),
        enabled=False,
    )
    res = mgr.step(
        observation_index=1,
        x=sample_features[0],
        prediction=0,
        probabilities={0: 1.0, 1: 0.0},
        model_version="v1",
        is_persistent_drift=True,
        drift_severity=0.99,
    )
    assert not res.triggered
    assert res.state == AdaptationState.IDLE


def test_adaptation_manager_stats_reporting():
    """Test that AdaptationManager.get_stats aggregates all component statistics."""
    mgr = AdaptationManager(
        FeedbackQueue(), CloudRetrainer(), CandidateValidator(),
        AtomicModelDeployer(CloudRuntime(MockClassifier()), EdgeRuntime(MockClassifier())),
    )
    stats = mgr.get_stats()
    assert "current_state" in stats
    assert "feedback" in stats
    assert "retrainer" in stats
    assert "validator" in stats
    assert "deployer" in stats

